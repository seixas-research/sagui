r"""D3PM -- discrete denoising diffusion for atom types.

Following Austin et al., *Structured Denoising Diffusion Models in Discrete
State-Spaces* (NeurIPS 2021).  The forward process is a Markov chain on the
finite set of chemical species,

.. math:: q(a_t \mid a_{t-1}) = \mathrm{Cat}\big(a_t;\ \mathbf{e}_{a_{t-1}} Q_t\big),

so that ``t`` steps compose into a single matrix
:math:`\bar Q_t = Q_1 Q_2 \cdots Q_t` and

.. math:: q(a_t \mid a_0) = \mathrm{Cat}\big(a_t;\ \mathbf{e}_{a_0} \bar Q_t\big).

Two transition kernels are provided:

``uniform``
    :math:`Q_t = (1-\beta_t) I + \beta_t \mathbf{1}\mathbf{1}^T / K` --
    each atom keeps its identity or is replaced by a uniformly random species.
    The chain converges to the uniform distribution over species.

``absorbing``
    :math:`Q_t = (1-\beta_t) I + \beta_t \mathbf{1}\mathbf{e}_{\text{mask}}^T` --
    atoms are progressively replaced by a ``[MASK]`` token and never come back.
    The chain converges to an all-masked crystal, and generation becomes a
    BERT-like unmasking, which is usually easier to learn.

The network predicts the *clean* types :math:`\tilde p_\theta(a_0 \mid a_t)`;
the reverse kernel then follows from the exact posterior

.. math::
    p_\theta(a_{t-1} \mid a_t) \propto
        Q_t[\cdot, a_t] \odot \big(\tilde p_\theta(a_0\mid a_t)\, \bar Q_{t-1}\big),

which is the parameterisation the paper recommends: it is exact at ``t = 1``
and lets a single network serve every noise level.

Because the number of species in a dataset is small, the transition matrices
are materialised in full (``[T + 1, K, K]``) instead of using closed forms --
exact, and trivially extensible to a chemistry-aware kernel.
"""

from __future__ import annotations

import torch
from torch import nn

from .schedules import betas_from_alpha_bar, cosine_alpha_bar

__all__ = ["D3PM"]

_EPS = 1e-30


class D3PM(nn.Module):
    """Discrete diffusion over atom types."""

    def __init__(
        self,
        num_species: int,
        num_steps: int = 1000,
        transition: str = "absorbing",
        schedule_offset: float = 0.008,
    ) -> None:
        super().__init__()
        if transition not in {"uniform", "absorbing"}:
            raise ValueError(f"transition must be 'uniform' or 'absorbing', got '{transition}'")
        self.num_species = int(num_species)
        self.num_steps = int(num_steps)
        self.transition = transition
        #: ``[MASK]`` occupies one extra token in the absorbing formulation.
        self.num_tokens = self.num_species + (1 if transition == "absorbing" else 0)
        self.mask_token = self.num_species if transition == "absorbing" else -1

        alpha_bar = cosine_alpha_bar(self.num_steps, schedule_offset)
        betas = betas_from_alpha_bar(alpha_bar)

        k = self.num_tokens
        identity = torch.eye(k, dtype=torch.float64)
        q_mats = torch.zeros(self.num_steps + 1, k, k, dtype=torch.float64)
        q_mats[0] = identity
        for t in range(1, self.num_steps + 1):
            beta = betas[t]
            if transition == "uniform":
                target = torch.full((k, k), 1.0 / k, dtype=torch.float64)
            else:
                target = torch.zeros(k, k, dtype=torch.float64)
                target[:, self.mask_token] = 1.0
            q_mats[t] = (1.0 - beta) * identity + beta * target

        q_bar = torch.zeros_like(q_mats)
        q_bar[0] = identity
        for t in range(1, self.num_steps + 1):
            q_bar[t] = q_bar[t - 1] @ q_mats[t]

        dtype = torch.get_default_dtype()
        self.register_buffer("q_mats", q_mats.to(dtype), persistent=False)
        self.register_buffer("q_bar", q_bar.to(dtype), persistent=False)

    # ------------------------------------------------------------ forward
    def q_sample(self, types_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Draw ``a_t ~ q(a_t | a_0)`` for per-atom timesteps ``t`` ``[N]``."""
        probabilities = self.q_bar[t, types_0]  # [N, K]
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def prior_sample(self, num_atoms: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw from the ``t = T`` limit distribution."""
        if self.transition == "absorbing":
            return torch.full((num_atoms,), self.mask_token, dtype=torch.long, device=device)
        return torch.randint(0, self.num_tokens, (num_atoms,), device=device)

    # ---------------------------------------------------------- posterior
    def posterior_logits(
        self, types_0_probs: torch.Tensor, types_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""Unnormalised :math:`\log q(a_{t-1} \mid a_t, a_0)` for a *distribution* over ``a_0``.

        Parameters
        ----------
        types_0_probs:
            ``[N, K]`` probabilities over the clean type (a one-hot for the
            true posterior, the model's prediction for the reverse kernel).
        types_t, t:
            ``[N]`` current types and timesteps.
        """
        # Q_t[k, a_t]: how likely each candidate k is to have produced a_t.
        q_t = self.q_mats[t]  # [N, K, K]
        index = types_t.view(-1, 1, 1).expand(-1, self.num_tokens, 1)
        fact_step = q_t.gather(2, index).squeeze(-1)  # [N, K]
        # (p(a_0) Q_bar_{t-1})[k]: how likely each candidate k is a priori.
        fact_history = torch.einsum("ni,nij->nj", types_0_probs, self.q_bar[t - 1])
        return torch.log(fact_step.clamp_min(_EPS)) + torch.log(fact_history.clamp_min(_EPS))

    def true_posterior_logits(
        self, types_0: torch.Tensor, types_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Posterior conditioned on the *known* clean types (the training target)."""
        one_hot = torch.zeros(
            types_0.shape[0], self.num_tokens, dtype=self.q_bar.dtype, device=types_0.device
        )
        one_hot.scatter_(1, types_0.view(-1, 1), 1.0)
        return self.posterior_logits(one_hot, types_t, t)

    def model_posterior_logits(
        self, logits_0: torch.Tensor, types_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Reverse kernel :math:`p_\\theta(a_{t-1} \\mid a_t)` from predicted ``a_0`` logits."""
        return self.posterior_logits(self.pad_logits(logits_0).softmax(-1), types_t, t)

    def pad_logits(self, logits_0: torch.Tensor) -> torch.Tensor:
        """Extend network logits over real species to the full token alphabet.

        The clean crystal never contains a ``[MASK]``, so that column is driven
        to probability zero rather than being predicted.
        """
        if logits_0.shape[-1] == self.num_tokens:
            return logits_0
        padding = torch.full(
            (*logits_0.shape[:-1], self.num_tokens - logits_0.shape[-1]),
            -1e9,
            dtype=logits_0.dtype,
            device=logits_0.device,
        )
        return torch.cat([logits_0, padding], dim=-1)

    # --------------------------------------------------------------- loss
    def loss(
        self,
        logits_0: torch.Tensor,
        types_0: torch.Tensor,
        types_t: torch.Tensor,
        t: torch.Tensor,
        ce_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Hybrid D3PM objective :math:`L_{vb} + \lambda\, L_{CE}`.

        ``L_vb`` is the per-step KL between the true posterior and the reverse
        kernel (replaced by the negative log-likelihood of the data at
        ``t = 1``, where the "posterior" is the decoder); ``L_CE`` is the plain
        cross-entropy of the ``a_0`` prediction, which gives a much stronger
        gradient early in training.
        """
        target = self.true_posterior_logits(types_0, types_t, t)
        predicted = self.model_posterior_logits(logits_0, types_t, t)

        log_q = torch.log_softmax(target, dim=-1)
        log_p = torch.log_softmax(predicted, dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(-1)

        # At t = 1 the reverse step *is* the reconstruction of the data.
        nll = -torch.gather(log_p, 1, types_0.view(-1, 1)).squeeze(-1)
        vb = torch.where(t == 1, nll, kl).mean()

        cross_entropy = torch.nn.functional.cross_entropy(
            self.pad_logits(logits_0), types_0, reduction="mean"
        )
        total = vb + ce_weight * cross_entropy
        return total, {
            "types_vb": float(vb.detach()),
            "types_ce": float(cross_entropy.detach()),
        }

    # ------------------------------------------------------------ reverse
    @torch.no_grad()
    def sample_step(
        self, logits_0: torch.Tensor, types_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """One ancestral step ``a_t -> a_{t-1}``."""
        logits = self.model_posterior_logits(logits_0, types_t, t)
        if int(t[0]) == 1:
            # Final step: take the mode rather than a sample, as usual for the
            # decoder -- a stray sample here is a visible defect in the output.
            return logits.argmax(-1)
        return torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(-1)
