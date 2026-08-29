"""The training objective: Huber residuals, the stress term, the phase switch."""

from __future__ import annotations

import logging

import pytest
import torch
from ase.build import bulk
from ase.calculators.emt import EMT

from sagui.data.atomic_data import collate_graphs, graph_from_atoms
from sagui.data.ztable import ZTable
from sagui.train.loss import EnergyForcesStressLoss
from sagui.train.trainer import switch_loss_phase


@pytest.fixture
def labelled_crystal():
    """A periodic cell carrying energy, forces *and* stress labels."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(stdev=0.05, seed=7)
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return collate_graphs([graph_from_atoms(atoms, ZTable([29]), 4.0)])


def _prediction(reference, scale=1.0):
    return {
        "energy": reference.energy * scale,
        "forces": reference.forces * scale,
        "stress": reference.stress * scale,
    }


def test_stress_labels_survive_the_round_trip(labelled_crystal):
    assert labelled_crystal.stress is not None
    assert labelled_crystal.stress.shape == (1, 3, 3)


def test_huber_matches_mean_square_for_small_residuals():
    """Below delta the Huber loss is exactly half the square."""
    mse = EnergyForcesStressLoss(1.0, 0.0, huber_delta=None)
    huber = EnergyForcesStressLoss(1.0, 0.0, huber_delta=1e3)
    error = torch.randn(8) * 1e-4
    assert torch.allclose(mse._residual(error), 2.0 * huber._residual(error), atol=1e-12)


def test_huber_is_linear_for_large_residuals():
    """Above delta an outlier contributes linearly, not quadratically."""
    huber = EnergyForcesStressLoss(huber_delta=0.01)
    small, large = huber._residual(torch.tensor([1.0])), huber._residual(torch.tensor([100.0]))
    assert large / small < 110.0  # a mean square would give 10 000x


def test_stress_term_is_only_added_when_asked(labelled_crystal):
    prediction = _prediction(labelled_crystal, scale=1.5)
    without = EnergyForcesStressLoss(1.0, 1.0, stress_weight=0.0)
    with_stress = EnergyForcesStressLoss(1.0, 1.0, stress_weight=10.0)

    _, terms_off = without(prediction, labelled_crystal)
    total_on, terms_on = with_stress(prediction, labelled_crystal)
    assert not any(key.startswith("stress") for key in terms_off)
    assert any(key.startswith("stress") for key in terms_on)
    assert terms_on["loss"] > terms_off["loss"]
    assert torch.isfinite(total_on)


def test_wants_stress_reports_whether_the_derivative_is_needed():
    assert not EnergyForcesStressLoss(stress_weight=0.0).wants_stress
    assert EnergyForcesStressLoss(stress_weight=1.0).wants_stress


def test_a_perfect_prediction_has_zero_loss(labelled_crystal):
    loss_fn = EnergyForcesStressLoss(1.0, 1.0, stress_weight=1.0, huber_delta=0.01)
    total, _ = loss_fn(_prediction(labelled_crystal), labelled_crystal)
    assert float(total) == pytest.approx(0.0, abs=1e-20)


def test_phase_switch_swaps_the_weights_and_drops_the_learning_rate():
    loss_fn = EnergyForcesStressLoss(energy_weight=1.0, forces_weight=100.0)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-2)
    switch_loss_phase(loss_fn, optimizer, logging.getLogger("test"))
    assert (loss_fn.energy_weight, loss_fn.forces_weight) == (100.0, 1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_loss_still_raises_without_any_labels(labelled_crystal):
    bare = graph_from_atoms(
        bulk("Cu", "fcc", a=3.6, cubic=True), ZTable([29]), 4.0, with_labels=False
    )
    batch = collate_graphs([bare])
    with pytest.raises(ValueError, match="nothing to train on"):
        EnergyForcesStressLoss()({"energy": torch.zeros(1)}, batch)


def test_stress_gradient_reaches_the_model(labelled_crystal):
    """A stress-only loss must still produce parameter gradients."""
    from sagui.config import ModelConfig
    from sagui.models.registry import build_model

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(type="strictly_local", r_max=4.0, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16]),
        [29],
    )
    loss_fn = EnergyForcesStressLoss(energy_weight=0.0, forces_weight=0.0, stress_weight=1.0)
    prediction = model(labelled_crystal, compute_forces=False, compute_stress=True, training=True)
    total, _ = loss_fn(prediction, labelled_crystal)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
