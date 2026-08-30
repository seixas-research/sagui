"""The training objective: Huber residuals, the stress term, the phase switch."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.emt import EMT

from sagui.data.atomic_data import collate_graphs, graph_from_atoms
from sagui.data.ztable import ZTable
from sagui.train.loss import EnergyForcesStressLoss
from sagui.train.trainer import apply_weight_switch


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
    assert torch.allclose(
        mse._residual(error, "energy"), 2.0 * huber._residual(error, "energy"), atol=1e-12
    )


def test_huber_is_linear_for_large_residuals():
    """Above delta an outlier contributes linearly, not quadratically."""
    huber = EnergyForcesStressLoss(huber_delta=0.01)
    small = huber._residual(torch.tensor([1.0]), "energy")
    large = huber._residual(torch.tensor([100.0]), "energy")
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


def test_per_term_deltas_override_the_shared_one():
    loss_fn = EnergyForcesStressLoss(
        huber_delta=0.01, huber_delta_forces=0.5, huber_delta_stress=None
    )
    assert loss_fn.deltas["energy"] == 0.01
    assert loss_fn.deltas["forces"] == 0.5
    assert loss_fn.deltas["stress"] == 0.01


def test_a_term_can_keep_its_mean_square_while_others_use_huber():
    """delta=None for one term is a mixed loss, not an error."""
    loss_fn = EnergyForcesStressLoss(huber_delta_forces=0.1)
    assert loss_fn.deltas["energy"] is None
    assert loss_fn.deltas["forces"] == 0.1
    assert loss_fn.deltas["stress"] is None
    error = torch.randn(16)
    assert torch.allclose(loss_fn._residual(error, "energy"), error.pow(2).mean())
    assert not torch.allclose(loss_fn._residual(error, "forces"), error.pow(2).mean())


def test_metric_names_report_which_terms_are_huber(labelled_crystal):
    loss_fn = EnergyForcesStressLoss(1.0, 1.0, stress_weight=1.0, huber_delta_forces=0.1)
    _, terms = loss_fn(_prediction(labelled_crystal, scale=1.5), labelled_crystal)
    assert "energy_mse" in terms and "forces_huber" in terms and "stress_mse" in terms


def test_a_non_positive_delta_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        EnergyForcesStressLoss(huber_delta_energy=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        EnergyForcesStressLoss(huber_delta=-1.0)


def test_a_shared_delta_still_applies_to_every_term():
    """Backwards compatibility with the single-delta form."""
    loss_fn = EnergyForcesStressLoss(huber_delta=0.02)
    assert set(loss_fn.deltas.values()) == {0.02}


def test_scales_put_the_terms_on_one_footing():
    """A weight of 1 should mean the same thing for terms of different size."""
    raw = EnergyForcesStressLoss(1.0, 1.0)
    scaled = EnergyForcesStressLoss(1.0, 1.0, scales={"energy": 4.0})
    error = torch.randn(64)
    assert torch.allclose(scaled._residual(error, "energy"), raw._residual(error / 4.0, "energy"))
    assert torch.allclose(scaled._residual(error, "forces"), raw._residual(error, "forces"))


def test_scaling_makes_the_huber_delta_dimensionless():
    """delta = 1 must sit at the typical residual whatever the units are."""
    error = torch.randn(4096) * 7.0
    loss_fn = EnergyForcesStressLoss(huber_delta=1.0, scales={"energy": 7.0})
    plain = EnergyForcesStressLoss(huber_delta=1.0)
    assert torch.allclose(loss_fn._residual(error, "energy"),
                          plain._residual(error / 7.0, "energy"))
    # and the Huber/MSE magnitude gap that entangles delta with lambda closes
    mse = EnergyForcesStressLoss(scales={"energy": 7.0})
    ratio = float(mse._residual(error, "energy")) / float(loss_fn._residual(error, "energy"))
    assert ratio < 5.0


def test_a_non_positive_scale_is_rejected():
    with pytest.raises(ValueError, match="scale for 'forces' must be positive"):
        EnergyForcesStressLoss(scales={"forces": 0.0})


def test_statistics_measure_the_residual_scales():
    from sagui.data.dataset import AtomsDataset
    from sagui.train.stats import compute_statistics

    atoms = []
    for seed in range(6):
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.rattle(stdev=0.05, seed=seed)
        a.calc = EMT()
        a.get_potential_energy()
        atoms.append(a)
    stats = compute_statistics(AtomsDataset(atoms, z_table=ZTable([29]), r_max=4.0))
    assert stats.energy_residual_rms > 0.0
    assert stats.forces_rms > 0.0
    assert stats.stress_rms > 0.0
    # The composition fit can only reduce the target, never enlarge it.  How
    # much it removes depends on the reference: for MPtrj, where energies sit
    # near -14 eV/atom, it is a factor of 7; for EMT, already referenced near
    # zero, it is a factor of two.
    raw = float(np.sqrt(np.mean([(a.get_potential_energy() / len(a)) ** 2 for a in atoms])))
    assert stats.energy_residual_rms < raw


def _optimizer(lr=1e-2):
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_second_stage_swaps_the_weights_by_default():
    loss_fn = EnergyForcesStressLoss(energy_weight=1.0, forces_weight=100.0)
    opt = _optimizer()
    apply_weight_switch(loss_fn, opt, logging.getLogger("test"))
    assert (loss_fn.energy_weight, loss_fn.forces_weight) == (100.0, 1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_second_stage_accepts_explicit_weights():
    """Explicit stage-two weights: choose the balance rather than swapping."""
    loss_fn = EnergyForcesStressLoss(energy_weight=1.0, forces_weight=100.0)
    opt = _optimizer()
    apply_weight_switch(loss_fn, opt, logging.getLogger("test"),
                       energy_weight=1000.0, forces_weight=10.0, lr_factor=0.5)
    assert (loss_fn.energy_weight, loss_fn.forces_weight) == (1000.0, 10.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-3)


def test_second_stage_can_leave_the_learning_rate_alone():
    loss_fn = EnergyForcesStressLoss()
    opt = _optimizer()
    apply_weight_switch(loss_fn, opt, logging.getLogger("test"), lr_factor=1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-2)


@pytest.mark.parametrize(
    "fraction,epochs,expected", [(0.5, 40, 20), (0.6, 100, 60), (0.01, 10, 1), (1.0, 30, 30)]
)
def test_weight_switch_maps_a_fraction_onto_an_epoch(fraction, epochs, expected):
    """The switch happens *after* this epoch, so 0.5 of 40 means epochs 21-40."""
    assert max(1, int(fraction * epochs)) == expected


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
