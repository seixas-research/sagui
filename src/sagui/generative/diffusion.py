r"""Joint diffusion model for crystal structures.

Combines the three forward processes of
:class:`~sagui.generative.corruption.MaterialsCorruption` with the equivariant
denoiser, and provides the two operations a generative model must support:

``loss``
    one term per modality -- the D3PM hybrid bound for the types, denoising
    score matching for the coordinates, and epsilon-prediction for the lattice;

``sample``
    ancestral sampling of all three jointly.  Every reverse step rebuilds the
    neighbour list, because the graph itself is part of what is being
    generated -- unlike a potential, where the geometry is given.

The reverse updates are the standard ones for each process: the exact
categorical posterior for the types, the variance-exploding ancestral step of
Song et al. for the coordinates,

.. math:: x_{t-1} = x_t + (\sigma_t^2 - \sigma_{t-1}^2)\, s_\theta
          + \sigma_{t-1}\sqrt{1 - \sigma_{t-1}^2/\sigma_t^2}\; z ,

and the usual DDPM posterior mean for the lattice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from ..config import DiffusionConfig, ModelConfig
from ..data.atomic_data import AtomicGraph, collate_graphs
from .corruption import MaterialsCorruption
from .denoiser import EquivariantDenoiser
from .structures import graph_from_arrays, sanitize_lattice, wrap_fractional

__all__ = ["MaterialsDiffusion", "GeneratedStructure"]

logger = logging.getLogger(__name__)


@dataclass
class GeneratedStructure:
    """A sampled crystal, in the units a caller expects."""

    species: torch.Tensor  # [N] indices into the ZTable
    frac: torch.Tensor  # [N, 3]
    cell: torch.Tensor  # [3, 3] in Angstrom


class MaterialsDiffusion(nn.Module):
    """D3PM types + wrapped-normal coordinates + DDPM lattice."""

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        num_species: int,
        lattice_scale: float = 2.5,
        avg_num_neighbors: float = 12.0,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.num_species = int(num_species)
        self.corruption = MaterialsCorruption(
            num_species=num_species,
            num_steps=diffusion_config.num_steps,
            transition=diffusion_config.type_transition,
            sigma_min=diffusion_config.sigma_min,
            sigma_max=diffusion_config.sigma_max,
        )
        self.denoiser = EquivariantDenoiser(
            model_config,
            num_tokens=self.corruption.num_tokens,
            num_species=num_species,
            num_steps=diffusion_config.num_steps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.register_buffer("lattice_scale", torch.tensor(float(lattice_scale)))
        self.register_buffer("r_max", torch.tensor(float(model_config.r_max)))

    # ------------------------------------------------------------- helpers
    @property
    def num_steps(self) -> int:
        return self.corruption.num_steps

    def normalise_lattice(self, cell: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
        """``L -> Y = L / (scale * N^(1/3))``, making cells of different sizes comparable."""
        factor = self.lattice_scale * num_atoms.to(cell.dtype) ** (1.0 / 3.0)
        return cell / factor.view(-1, 1, 1)

    def denormalise_lattice(self, lattice: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
        factor = self.lattice_scale * num_atoms.to(lattice.dtype) ** (1.0 / 3.0)
        return lattice * factor.view(-1, 1, 1)

    # ---------------------------------------------------------------- loss
    def loss(self, batch, weights: DiffusionConfig | None = None) -> tuple[torch.Tensor, dict]:
        """Weighted sum of the three denoising objectives for a noised batch.

        ``batch`` is a :class:`~sagui.generative.dataset.DiffusionBatch`
        carrying the corrupted structure and the corresponding targets.
        """
        config = weights or self.diffusion_config
        prediction = self.denoiser(batch.graph, batch.t, batch.lattice_t)

        types_loss, terms = self.corruption.types.loss(
            prediction["type_logits"],
            batch.types_0,
            batch.graph.species,
            batch.t_atom,
            ce_weight=config.type_ce_weight,
        )
        coord_loss = (prediction["coord_score"] - batch.coord_target).pow(2).mean()
        lattice_loss = (prediction["lattice_noise"] - batch.lattice_noise).pow(2).mean()

        total = (
            config.type_weight * types_loss
            + config.coord_weight * coord_loss
            + config.lattice_weight * lattice_loss
        )
        terms.update(
            {
                "loss": float(total.detach()),
                "types": float(types_loss.detach()),
                "coords": float(coord_loss.detach()),
                "lattice": float(lattice_loss.detach()),
            }
        )
        return total, terms

    # -------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(
        self,
        num_atoms: list[int] | torch.Tensor,
        device: torch.device | str = "cpu",
        num_steps: int | None = None,
        progress: bool = False,
    ) -> list[GeneratedStructure]:
        """Generate crystals with the requested numbers of atoms.

        ``num_steps`` may be smaller than the training horizon to trade quality
        for speed; the schedule is then traversed on a strided sub-grid.
        """
        self.eval()
        counts = torch.as_tensor(num_atoms, dtype=torch.long, device=device).view(-1)
        n_structures = int(counts.shape[0])
        total_atoms = int(counts.sum())
        dtype = self.lattice_scale.dtype
        batch_index = torch.repeat_interleave(
            torch.arange(n_structures, device=device), counts
        )

        # --- draw from the prior of each modality -------------------------
        types = self.corruption.types.prior_sample(total_atoms, device=device)
        frac = torch.rand(total_atoms, 3, device=device, dtype=dtype)
        lattice = torch.randn(n_structures, 3, 3, device=device, dtype=dtype)

        schedule = self._timestep_grid(num_steps, device)
        sigmas = self.corruption.sigmas
        alpha_bar = self.corruption.alpha_bar

        for step, (t_now, t_next) in enumerate(zip(schedule[:-1], schedule[1:], strict=True)):
            t = torch.full((n_structures,), int(t_now), device=device, dtype=torch.long)
            t_atom = t[batch_index]
            graph, lattice_seen = self._build_batch(frac, lattice, types, counts, device)
            prediction = self.denoiser(graph, t, lattice_seen)

            types = self._reverse_types(prediction["type_logits"], types, t_atom, int(t_next))
            frac = self._reverse_coords(
                frac, prediction["coord_score"], sigmas[t_now], sigmas[t_next]
            )
            lattice = self._reverse_lattice(
                lattice, prediction["lattice_noise"], alpha_bar[t_now], alpha_bar[t_next]
            )
            if progress and step % max(1, len(schedule) // 10) == 0:
                logger.info("sampling step %d/%d (t=%d)", step, len(schedule) - 1, int(t_now))

        cells = self.denormalise_lattice(lattice, counts)
        cells = sanitize_lattice(cells, min_length=0.5 * float(self.r_max))
        frac = wrap_fractional(frac)

        results = []
        offset = 0
        for index in range(n_structures):
            size = int(counts[index])
            results.append(
                GeneratedStructure(
                    species=types[offset : offset + size].clamp_max(self.num_species - 1).cpu(),
                    frac=frac[offset : offset + size].cpu(),
                    cell=cells[index].cpu(),
                )
            )
            offset += size
        return results

    # ------------------------------------------------------- reverse steps
    def _timestep_grid(self, num_steps: int | None, device) -> torch.Tensor:
        """Descending grid ``T -> 0``, optionally strided for fast sampling."""
        if num_steps is None or num_steps >= self.num_steps:
            return torch.arange(self.num_steps, -1, -1, device=device)
        grid = torch.linspace(self.num_steps, 0, num_steps + 1, device=device).round().long()
        # Already descending; drop the repeats a coarse grid can produce.
        return torch.unique_consecutive(grid)

    def _reverse_types(
        self, logits: torch.Tensor, types_t: torch.Tensor, t_atom: torch.Tensor, t_next: int
    ) -> torch.Tensor:
        if t_next == 0:
            return self.corruption.types.pad_logits(logits).argmax(-1)
        return self.corruption.types.sample_step(logits, types_t, t_atom)

    def _reverse_coords(
        self,
        frac: torch.Tensor,
        prediction: torch.Tensor,
        sigma_now: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Variance-exploding ancestral step on the torus."""
        score = prediction / sigma_now.clamp_min(1e-8)
        step = (sigma_now**2 - sigma_next**2).clamp_min(0.0)
        mean = frac + step * score
        if float(sigma_next) > 0.0:
            std = (step * sigma_next**2 / sigma_now**2).clamp_min(0.0).sqrt()
            mean = mean + std * torch.randn_like(frac)
        return wrap_fractional(mean)

    def _reverse_lattice(
        self,
        lattice: torch.Tensor,
        noise: torch.Tensor,
        alpha_bar_now: torch.Tensor,
        alpha_bar_next: torch.Tensor,
    ) -> torch.Tensor:
        r"""DDPM posterior step, expressed through a *clipped* estimate of ``Y_0``.

        The naive form ``(Y_t - beta \hat\varepsilon / \sqrt{1-\bar\alpha_t})
        / \sqrt{\alpha_t}`` amplifies any error in the predicted noise by
        :math:`1/\sqrt{\alpha_t}` at every step, and an under-trained model
        diverges within a few dozen of them.  Reconstructing ``Y_0``, clipping
        it to the range the data actually occupies, and then taking the exact
        posterior mean

        .. math::
            \mu = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\hat Y_0
                 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} Y_t

        is algebraically identical when the prediction is perfect and bounded
        when it is not.  This is the standard stabilisation used by DDPM
        implementations, and it matters more here than for images because a
        diverging lattice also destroys the neighbour list.
        """
        one_minus = (1.0 - alpha_bar_now).clamp_min(1e-8)
        y0 = (lattice - one_minus.sqrt() * noise) / alpha_bar_now.clamp_min(1e-8).sqrt()
        clip = self.diffusion_config.lattice_clip
        if clip is not None:
            y0 = y0.clamp(-float(clip), float(clip))

        alpha = (alpha_bar_now / alpha_bar_next).clamp(1e-8, 1.0)
        beta = 1.0 - alpha
        mean = (
            alpha_bar_next.sqrt() * beta / one_minus * y0
            + alpha.sqrt() * (1.0 - alpha_bar_next) / one_minus * lattice
        )
        if float(alpha_bar_next) < 1.0:
            variance = beta * (1.0 - alpha_bar_next) / one_minus
            mean = mean + variance.clamp_min(0.0).sqrt() * torch.randn_like(lattice)
        return mean

    # -------------------------------------------------------------- graphs
    def _build_batch(
        self,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        types: torch.Tensor,
        counts: torch.Tensor,
        device,
    ) -> tuple[AtomicGraph, torch.Tensor]:
        """Graph of the current state, plus the lattice it was actually built from.

        ``graph_from_arrays`` widens near-singular cells; the denoiser has to be
        given that same widened lattice, or its inputs would be inconsistent
        with the graph -- and with what it saw during training.
        """
        cells = self.denormalise_lattice(lattice, counts)
        graphs = []
        offset = 0
        for index in range(int(counts.shape[0])):
            size = int(counts[index])
            graphs.append(
                graph_from_arrays(
                    frac[offset : offset + size].cpu(),
                    cells[index].cpu(),
                    types[offset : offset + size].cpu(),
                    r_max=float(self.r_max),
                    max_neighbors=self.diffusion_config.max_neighbors,
                )
            )
            offset += size
        batch = collate_graphs(graphs)
        factor = self.lattice_scale * counts.to(lattice.dtype) ** (1.0 / 3.0)
        lattice_seen = batch.cell.to(device) / factor.view(-1, 1, 1)
        return batch.to(device), lattice_seen
