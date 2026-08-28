"""The generative stack: D3PM, the periodic forward process, and sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.build import bulk

from sagui.config import DiffusionConfig, ModelConfig
from sagui.data import ZTable, collate_graphs
from sagui.generative import (
    D3PM,
    DiffusionDataset,
    MaterialsCorruption,
    MaterialsDiffusion,
    collate_diffusion,
    cosine_alpha_bar,
    geometric_sigmas,
    graph_from_arrays,
    sanitize_lattice,
    wrapped_normal_score,
)
from sagui.generative.dataset import random_rotation
from sagui.generative.schedules import betas_from_alpha_bar
from sagui.nn.o3 import rotation_matrix

NUM_STEPS = 40


@pytest.fixture
def crystals() -> list:
    """Rattled rocksalt cells: periodic, binary, and all the same size."""
    rng = np.random.default_rng(0)
    template = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    frames = []
    for _ in range(6):
        atoms = template.copy()
        atoms.set_cell(atoms.get_cell() @ (np.eye(3) + 0.03 * rng.normal(size=(3, 3))),
                       scale_atoms=True)
        atoms.rattle(stdev=0.08, seed=int(rng.integers(1 << 30)))
        frames.append(atoms)
    return frames


# --------------------------------------------------------------- schedules
def test_alpha_bar_is_consistent_with_its_betas():
    """alpha_bar must equal the running product of (1 - beta); a clamp that
    breaks this makes the reverse process blow up."""
    alpha_bar = cosine_alpha_bar(NUM_STEPS)
    betas = betas_from_alpha_bar(alpha_bar)
    assert alpha_bar[0] == pytest.approx(1.0)
    assert torch.all(betas <= 0.999) and torch.all(betas >= 0.0)
    assert torch.allclose(alpha_bar, torch.cumprod(1.0 - betas, dim=0))
    assert torch.all(alpha_bar[1:] <= alpha_bar[:-1]), "alpha_bar must decrease"


def test_sigma_ladder_starts_at_zero_and_increases():
    sigmas = geometric_sigmas(NUM_STEPS, 0.01, 0.5)
    assert sigmas.shape == (NUM_STEPS + 1,)
    assert sigmas[0] == 0.0
    assert torch.all(sigmas[2:] > sigmas[1:-1])


# -------------------------------------------------------------------- D3PM
@pytest.mark.parametrize("transition", ["uniform", "absorbing"])
def test_transition_matrices_are_stochastic(transition):
    d3pm = D3PM(num_species=3, num_steps=NUM_STEPS, transition=transition)
    ones = torch.ones(NUM_STEPS + 1, d3pm.num_tokens, dtype=d3pm.q_bar.dtype)
    assert torch.allclose(d3pm.q_mats.sum(-1), ones)
    assert torch.allclose(d3pm.q_bar.sum(-1), ones)
    assert torch.allclose(d3pm.q_bar[0], torch.eye(d3pm.num_tokens, dtype=d3pm.q_bar.dtype))
    assert torch.all(d3pm.q_bar >= 0.0)


def test_q_bar_composes_the_single_step_matrices():
    d3pm = D3PM(num_species=4, num_steps=NUM_STEPS)
    assert torch.allclose(d3pm.q_bar[3], d3pm.q_bar[2] @ d3pm.q_mats[3])


def test_uniform_chain_converges_to_the_uniform_distribution():
    d3pm = D3PM(num_species=4, num_steps=200, transition="uniform")
    limit = d3pm.q_bar[-1]
    assert torch.allclose(limit, torch.full_like(limit, 0.25), atol=1e-3)


def test_absorbing_chain_converges_to_all_masked():
    d3pm = D3PM(num_species=4, num_steps=200, transition="absorbing")
    limit = d3pm.q_bar[-1]
    assert torch.allclose(limit[:, d3pm.mask_token], torch.ones(d3pm.num_tokens), atol=1e-3)
    assert d3pm.prior_sample(5).unique().tolist() == [d3pm.mask_token]


@pytest.mark.parametrize("transition", ["uniform", "absorbing"])
def test_posterior_recovers_the_clean_types(transition):
    """With the true a_0 the posterior at t = 1 must put its mass back on it."""
    d3pm = D3PM(num_species=4, num_steps=NUM_STEPS, transition=transition)
    types_0 = torch.randint(0, 4, (200,))
    t = torch.ones(200, dtype=torch.long)
    types_t = d3pm.q_sample(types_0, t)
    posterior = d3pm.true_posterior_logits(types_0, types_t, t)
    assert torch.equal(posterior.argmax(-1), types_0)


@pytest.mark.parametrize("transition", ["uniform", "absorbing"])
def test_loss_vanishes_for_a_perfect_prediction(transition):
    d3pm = D3PM(num_species=3, num_steps=NUM_STEPS, transition=transition)
    types_0 = torch.randint(0, 3, (128,))
    t = torch.randint(1, NUM_STEPS + 1, (128,))
    types_t = d3pm.q_sample(types_0, t)
    logits = torch.zeros(128, 3).scatter_(1, types_0.view(-1, 1), 40.0)
    loss, terms = d3pm.loss(logits, types_0, types_t, t)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    assert terms["types_vb"] == pytest.approx(0.0, abs=1e-5)


def test_loss_is_positive_for_a_wrong_prediction():
    d3pm = D3PM(num_species=3, num_steps=NUM_STEPS)
    types_0 = torch.zeros(64, dtype=torch.long)
    t = torch.randint(1, NUM_STEPS + 1, (64,))
    types_t = d3pm.q_sample(types_0, t)
    logits = torch.zeros(64, 3).scatter_(1, torch.ones(64, 1, dtype=torch.long), 40.0)
    loss, _ = d3pm.loss(logits, types_0, types_t, t)
    assert float(loss) > 1.0


def test_masked_atoms_are_never_predicted_as_clean():
    d3pm = D3PM(num_species=3, num_steps=NUM_STEPS, transition="absorbing")
    padded = d3pm.pad_logits(torch.zeros(4, 3))
    assert padded.shape == (4, 4)
    assert torch.all(padded.softmax(-1)[:, d3pm.mask_token] < 1e-6)


# ------------------------------------------------------------- coordinates
@pytest.mark.parametrize("sigma_value", [0.02, 0.1, 0.45])
def test_wrapped_normal_score_matches_autograd(sigma_value):
    """The analytic score must equal d/dx log sum_k N(x - k)."""
    delta = torch.linspace(-0.49, 0.49, 21, dtype=torch.float64, requires_grad=True)
    sigma = torch.full_like(delta, sigma_value)
    images = torch.arange(-8, 9, dtype=torch.float64)
    log_density = torch.logsumexp(
        -0.5 * ((delta[:, None] - images) / sigma[:, None]) ** 2, dim=-1
    )
    (reference,) = torch.autograd.grad(log_density.sum(), delta)
    assert torch.allclose(wrapped_normal_score(delta.detach(), sigma), reference, atol=1e-8)


def test_corrupted_coordinates_stay_in_the_unit_cell():
    corruption = MaterialsCorruption(num_species=2, num_steps=NUM_STEPS, sigma_max=0.5)
    frac_0 = torch.rand(200, 3)
    t = torch.full((200,), NUM_STEPS, dtype=torch.long)
    frac_t, target, sigma = corruption.corrupt_coords(frac_0, t)
    assert torch.all(frac_t >= 0.0) and torch.all(frac_t < 1.0)
    assert target.shape == frac_0.shape
    assert torch.isfinite(target).all()


def test_lattice_corruption_interpolates_between_data_and_noise():
    corruption = MaterialsCorruption(num_species=2, num_steps=NUM_STEPS)
    lattice_0 = torch.eye(3).expand(8, 3, 3).contiguous()
    early, _ = corruption.corrupt_lattice(lattice_0, torch.ones(8, dtype=torch.long))
    late, _ = corruption.corrupt_lattice(lattice_0, torch.full((8,), NUM_STEPS, dtype=torch.long))
    assert (early - lattice_0).abs().mean() < (late - lattice_0).abs().mean()


# ------------------------------------------------------------------ graphs
def test_sanitize_lattice_bounds_a_degenerate_cell():
    singular = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    fixed = sanitize_lattice(singular, min_length=2.5)
    assert torch.linalg.det(fixed) > 0.0
    assert torch.linalg.svdvals(fixed).min() >= 2.5 - 1e-6


def test_sanitize_lattice_leaves_a_healthy_cell_alone():
    cell = torch.diag(torch.tensor([6.0, 7.0, 8.0]))
    assert torch.allclose(sanitize_lattice(cell, min_length=2.5), cell, atol=1e-9)


def test_neighbour_list_is_capped():
    """A tiny cell has hundreds of images inside the cutoff; the cap must bite."""
    graph = graph_from_arrays(
        torch.rand(4, 3), torch.eye(3) * 3.0, torch.zeros(4, dtype=torch.long),
        r_max=6.0, max_neighbors=10,
    )
    assert graph.num_edges <= 4 * 10
    counts = torch.bincount(graph.receivers, minlength=4)
    assert int(counts.max()) <= 10


def test_random_rotation_is_a_proper_rotation():
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = random_rotation(rng)
        assert np.allclose(q @ q.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(q), 1.0)


# ------------------------------------------------------------------- model
def build_diffusion(num_species: int = 2) -> MaterialsDiffusion:
    model_config = ModelConfig(
        type="mace", r_max=5.0, lmax=2, channels=8, num_layers=2,
        num_radial_basis=6, radial_mlp_hidden=[16], scalar_mlp_hidden=[16], correlation=2,
    )
    diffusion_config = DiffusionConfig(num_steps=NUM_STEPS, type_transition="absorbing")
    return MaterialsDiffusion(
        model_config, diffusion_config, num_species=num_species,
        lattice_scale=2.1, avg_num_neighbors=12.0,
    )


def _denoise(model, frac, lattice, types, factor):
    graph = collate_graphs(
        [graph_from_arrays(frac, lattice * factor, types, r_max=5.0, max_neighbors=24)]
    )
    return model.denoiser(graph, torch.tensor([NUM_STEPS // 2]), graph.cell / factor)


def test_denoiser_symmetries(crystals):
    """Types and fractional scores are invariant; the lattice noise co-rotates.

    Rotating the cell while holding the fractional coordinates fixed is the
    same crystal in a different frame, so nothing fractional may change.
    """
    model = build_diffusion()
    atoms = crystals[0]
    frac = torch.as_tensor(atoms.get_scaled_positions())
    types = torch.zeros(len(atoms), dtype=torch.long)
    factor = 2.1 * len(atoms) ** (1 / 3)
    lattice = torch.as_tensor(atoms.get_cell().array) / factor

    rotation = rotation_matrix(0.4, 1.1, 2.3).to(lattice.dtype)
    plain = _denoise(model, frac, lattice, types, factor)
    rotated = _denoise(model, frac, lattice @ rotation.T, types, factor)

    assert torch.allclose(plain["type_logits"], rotated["type_logits"], atol=1e-9)
    assert torch.allclose(plain["coord_score"], rotated["coord_score"], atol=1e-9)
    assert torch.allclose(
        rotated["lattice_noise"], plain["lattice_noise"] @ rotation.T, atol=1e-9
    )


def test_denoiser_output_shapes(crystals):
    model = build_diffusion()
    atoms = crystals[0]
    factor = 2.1 * len(atoms) ** (1 / 3)
    out = _denoise(
        model,
        torch.as_tensor(atoms.get_scaled_positions()),
        torch.as_tensor(atoms.get_cell().array) / factor,
        torch.zeros(len(atoms), dtype=torch.long),
        factor,
    )
    assert out["type_logits"].shape == (len(atoms), 2)
    assert out["coord_score"].shape == (len(atoms), 3)
    assert out["lattice_noise"].shape == (1, 3, 3)


def test_training_step_runs_and_produces_gradients(crystals):
    model = build_diffusion()
    z_table = ZTable.from_atoms(crystals)
    dataset = DiffusionDataset(
        crystals, z_table, model.corruption, r_max=5.0, lattice_scale=2.1, max_neighbors=24
    )
    batch = collate_diffusion([dataset[i] for i in range(3)])
    assert batch.num_graphs == 3
    assert batch.t.shape == (3,)
    assert batch.t_atom.shape == (batch.graph.num_nodes,)

    loss, terms = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"types", "coords", "lattice"} <= set(terms)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients), "no parameter received a gradient"


def test_dataset_rejects_non_periodic_structures(crystals):
    model = build_diffusion()
    molecule = crystals[0].copy()
    molecule.set_pbc(False)
    with pytest.raises(ValueError, match="periodic"):
        DiffusionDataset(
            [molecule], ZTable([8, 12]), model.corruption, r_max=5.0, lattice_scale=2.1
        )


def test_sampling_produces_valid_crystals():
    """A random-weight model still has to yield well-formed structures."""
    model = build_diffusion()
    structures = model.sample([4, 6], num_steps=NUM_STEPS)
    assert len(structures) == 2
    for structure, size in zip(structures, [4, 6], strict=True):
        assert structure.species.shape == (size,)
        assert structure.frac.shape == (size, 3)
        assert structure.cell.shape == (3, 3)
        assert torch.all(structure.frac >= 0.0) and torch.all(structure.frac < 1.0)
        assert torch.isfinite(structure.cell).all()
        assert float(torch.linalg.det(structure.cell)) > 0.0, "cell must be right-handed"
        assert int(structure.species.max()) < model.num_species, "no [MASK] may survive"


def test_fast_sampling_uses_a_strided_schedule():
    model = build_diffusion()
    grid = model._timestep_grid(8, torch.device("cpu"))
    assert int(grid[0]) == NUM_STEPS and int(grid[-1]) == 0
    assert torch.all(grid[1:] < grid[:-1]), "the grid must be strictly descending"
    assert len(grid) <= 9
