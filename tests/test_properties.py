"""Per-atom property heads: partial charges and magnetic moments."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.emt import EMT

from sagui.config import ModelConfig
from sagui.data.atomic_data import collate_graphs, extract_labels, graph_from_atoms
from sagui.data.ztable import ZTable
from sagui.models.registry import build_model
from sagui.nn.scatter import scatter_sum
from sagui.train.loss import EnergyForcesStressLoss

Z_TABLE = ZTable([29])
R_MAX = 4.0


def _atoms(seed: int = 7, charges=None, magmoms=None, labelled: bool = True):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(stdev=0.05, seed=seed)
    if labelled:
        # The loss refuses a batch with no energy or force labels at all, so
        # give the fixture the ordinary ones too.
        atoms.calc = EMT()
        atoms.get_potential_energy()
        atoms.get_forces()
    if charges is not None:
        atoms.arrays["charges"] = np.asarray(charges, dtype=float)
    if magmoms is not None:
        atoms.arrays["magmoms"] = np.asarray(magmoms, dtype=float)
    return atoms


def _model(**flags):
    torch.manual_seed(0)
    return build_model(
        ModelConfig(type="strictly_local", r_max=R_MAX, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16], **flags),
        Z_TABLE.zs,
    ).eval()


def _batch(*atoms):
    return collate_graphs(
        [graph_from_atoms(a, Z_TABLE, R_MAX, with_labels=True) for a in atoms]
    )


# ------------------------------------------------------------------- labels
def test_charges_and_magmoms_are_read_from_the_arrays():
    atoms = _atoms(charges=np.linspace(-0.4, 0.4, 4), magmoms=np.full(4, 1.5))
    labels = extract_labels(atoms)
    assert labels.charges is not None and labels.charges.shape == (4,)
    assert labels.magmoms is not None and np.allclose(labels.magmoms, 1.5)
    assert labels.total_charge == pytest.approx(0.0)


def test_an_explicit_total_charge_wins_over_the_sum():
    atoms = _atoms(charges=np.full(4, 0.25))
    atoms.info["charge"] = -1.0
    assert extract_labels(atoms).total_charge == pytest.approx(-1.0)


def test_labels_batch_correctly():
    batch = _batch(_atoms(1, charges=np.zeros(4), magmoms=np.ones(4)),
                   _atoms(2, charges=np.zeros(4), magmoms=np.ones(4)))
    assert batch.charges.shape == (8,) and batch.magmoms.shape == (8,)
    assert batch.total_charge.shape == (2,)


# -------------------------------------------------------------------- heads
def test_heads_are_absent_unless_requested():
    out = _model()(_batch(_atoms()), training=False)
    assert "charges" not in out and "magmoms" not in out


def test_charges_are_conserved_exactly():
    """A projection, not a penalty: the constraint holds for any network output."""
    model = _model(predict_charges=True)
    batch = _batch(_atoms(1), _atoms(2))
    charges = model(batch, training=False)["charges"]
    totals = scatter_sum(charges, batch.batch, batch.num_graphs)
    assert torch.allclose(totals, torch.zeros_like(totals), atol=1e-12)


def test_charges_are_conserved_against_a_non_zero_total():
    model = _model(predict_charges=True)
    atoms = _atoms(1)
    atoms.info["charge"] = -2.0
    batch = _batch(atoms)
    totals = scatter_sum(model(batch, training=False)["charges"], batch.batch, batch.num_graphs)
    assert float(totals[0]) == pytest.approx(-2.0, abs=1e-12)


def test_total_magmom_is_the_sum_of_the_local_ones():
    model = _model(predict_magmoms=True)
    batch = _batch(_atoms(1), _atoms(2))
    out = model(batch, training=False)
    assert torch.allclose(
        scatter_sum(out["magmoms"], batch.batch, batch.num_graphs), out["total_magmom"]
    )


def test_property_heads_are_rotation_invariant():
    """They read the invariant track, so a rotation must leave them alone."""
    from sagui.nn.o3 import rotation_matrix

    model = _model(predict_charges=True, predict_magmoms=True)
    atoms = _atoms(3)
    rotation = rotation_matrix(0.6, 1.2, 0.4).numpy()
    turned = atoms.copy()
    turned.set_positions(atoms.get_positions() @ rotation.T)
    turned.set_cell(atoms.get_cell().array @ rotation.T)

    a = model(_batch(atoms), training=False)
    b = model(_batch(turned), training=False)
    for key in ("charges", "magmoms"):
        assert torch.allclose(a[key], b[key], atol=1e-10)


def test_the_extra_heads_do_not_disturb_the_energy():
    """They share the trunk but must not enter the energy or the forces."""
    plain = _model()
    withheads = _model(predict_charges=True, predict_magmoms=True)
    batch = _batch(_atoms(5))
    assert torch.allclose(plain(batch, training=False)["energy"],
                          withheads(batch, training=False)["energy"], atol=1e-12)


# --------------------------------------------------------------------- loss
def test_property_terms_enter_the_loss_only_when_weighted_and_labelled():
    model = _model(predict_charges=True, predict_magmoms=True)
    batch = _batch(_atoms(1, charges=np.zeros(4), magmoms=np.ones(4)))
    prediction = model(batch, training=False)

    off = EnergyForcesStressLoss(1.0, 1.0)
    on = EnergyForcesStressLoss(1.0, 1.0, charges_weight=1.0, magmoms_weight=1.0)
    _, terms_off = off(prediction, batch)
    _, terms_on = on(prediction, batch)
    assert not any(k.startswith(("charges", "magmoms")) for k in terms_off)
    assert any(k.startswith("charges") for k in terms_on)
    assert any(k.startswith("magmoms") for k in terms_on)


def test_a_missing_label_silently_skips_its_term():
    model = _model(predict_magmoms=True)
    batch = _batch(_atoms(1))  # no magmom labels
    loss_fn = EnergyForcesStressLoss(1.0, 1.0, magmoms_weight=1.0)
    _, terms = loss_fn(model(batch, training=False), batch)
    assert not any(k.startswith("magmoms") for k in terms)


def test_the_property_heads_receive_gradients():
    model = _model(predict_magmoms=True)
    batch = _batch(_atoms(1, magmoms=np.linspace(-1, 1, 4)))
    loss_fn = EnergyForcesStressLoss(energy_weight=0.0, forces_weight=0.0, magmoms_weight=1.0)
    loss, _ = loss_fn(model(batch, training=True), batch)
    loss.backward()
    head = model.property_heads["magmoms"]
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)
