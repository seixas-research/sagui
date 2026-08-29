"""Physical invariants that any interatomic potential must satisfy.

These are the tests that matter: a potential that is not invariant under
rotations, or whose forces are not the gradient of its energy, is wrong no
matter how low its training loss goes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from sagui.config import ModelConfig
from sagui.data import ZTable, collate_graphs, graph_from_atoms
from sagui.models import available_models, build_model, get_model_class, register_model
from sagui.models.base import InteratomicPotential
from sagui.nn.o3 import rotation_matrix

ARCHITECTURES = list(available_models())
R_MAX = 4.5


def make_model(architecture: str, z_table: ZTable, **overrides) -> InteratomicPotential:
    settings = dict(
        type=architecture,
        r_max=R_MAX,
        lmax=2,
        channels=8,
        num_layers=2,
        num_radial_basis=6,
        radial_mlp_hidden=[16],
        scalar_mlp_hidden=[16],
        latent_dim=16,
        correlation=3,
    )
    settings.update(overrides)
    return build_model(ModelConfig(**settings), z_table.zs, avg_num_neighbors=6.0)


def predict(model, atoms: Atoms, z_table: ZTable, **kwargs) -> dict[str, torch.Tensor]:
    graph = graph_from_atoms(atoms, z_table, R_MAX, with_labels=False)
    kwargs.setdefault("compute_forces", True)
    return model(collate_graphs([graph]), training=False, **kwargs)


def test_both_required_architectures_are_registered():
    assert set(ARCHITECTURES) >= {"mace", "strictly_local"}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_output_shapes(architecture, cluster):
    z_table = ZTable.from_atoms([cluster])
    out = predict(make_model(architecture, z_table), cluster, z_table)
    assert out["energy"].shape == (1,)
    assert out["node_energy"].shape == (len(cluster),)
    assert out["forces"].shape == (len(cluster), 3)
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_translation_invariance(architecture, cluster):
    z_table = ZTable.from_atoms([cluster])
    model = make_model(architecture, z_table)
    reference = predict(model, cluster, z_table)

    moved = cluster.copy()
    moved.positions += np.array([3.7, -2.1, 0.6])
    shifted = predict(model, moved, z_table)

    assert torch.allclose(shifted["energy"], reference["energy"], atol=1e-10)
    assert torch.allclose(shifted["forces"], reference["forces"], atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_rotation_equivariance(architecture, cluster):
    """Energy is invariant, forces rotate with the structure."""
    z_table = ZTable.from_atoms([cluster])
    model = make_model(architecture, z_table)
    reference = predict(model, cluster, z_table)

    R = rotation_matrix(0.42, 1.13, 2.31).numpy()
    rotated_atoms = cluster.copy()
    rotated_atoms.positions = rotated_atoms.positions @ R.T
    rotated = predict(model, rotated_atoms, z_table)

    assert torch.allclose(rotated["energy"], reference["energy"], atol=1e-10)
    assert torch.allclose(rotated["forces"], reference["forces"] @ torch.as_tensor(R.T), atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_reflection_equivariance(architecture, cluster):
    """Full O(3), not just SO(3): the parity selection rule must hold."""
    z_table = ZTable.from_atoms([cluster])
    model = make_model(architecture, z_table)
    reference = predict(model, cluster, z_table)

    mirror = np.diag([1.0, 1.0, -1.0])
    reflected_atoms = cluster.copy()
    reflected_atoms.positions = reflected_atoms.positions @ mirror.T
    reflected = predict(model, reflected_atoms, z_table)

    assert torch.allclose(reflected["energy"], reference["energy"], atol=1e-10)
    assert torch.allclose(
        reflected["forces"], reference["forces"] @ torch.as_tensor(mirror.T), atol=1e-10
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_permutation_invariance(architecture, cluster):
    """Relabelling identical atoms must permute the forces, nothing else."""
    z_table = ZTable.from_atoms([cluster])
    model = make_model(architecture, z_table)
    reference = predict(model, cluster, z_table)

    order = np.argsort(cluster.get_atomic_numbers(), kind="stable")[::-1].copy()
    permuted = cluster[order]
    result = predict(model, permuted, z_table)

    assert torch.allclose(result["energy"], reference["energy"], atol=1e-10)
    assert torch.allclose(result["forces"], reference["forces"][order.tolist()], atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forces_are_minus_the_energy_gradient(architecture, cluster):
    """Central finite differences against the autograd forces."""
    z_table = ZTable.from_atoms([cluster])
    model = make_model(architecture, z_table)
    analytic = predict(model, cluster, z_table)["forces"]

    eps = 1e-6
    for atom, axis in [(0, 0), (2, 1), (5, 2)]:
        plus, minus = cluster.copy(), cluster.copy()
        plus.positions[atom, axis] += eps
        minus.positions[atom, axis] -= eps
        derivative = (
            predict(model, plus, z_table)["energy"] - predict(model, minus, z_table)["energy"]
        ) / (2 * eps)
        assert torch.allclose(analytic[atom, axis], -derivative[0], atol=1e-7)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_batching_matches_individual_evaluation(architecture, labelled_frames):
    """A batch is only a bookkeeping device: results must be identical."""
    z_table = ZTable.from_atoms(labelled_frames)
    model = make_model(architecture, z_table)
    frames = labelled_frames[:4]

    graphs = [graph_from_atoms(a, z_table, R_MAX, with_labels=False) for a in frames]
    batched = model(collate_graphs(graphs), compute_forces=True, training=False)
    individually = [
        model(collate_graphs([g]), compute_forces=True, training=False) for g in graphs
    ]

    assert torch.allclose(
        batched["energy"], torch.cat([o["energy"] for o in individually]), atol=1e-10
    )
    assert torch.allclose(
        batched["forces"], torch.cat([o["forces"] for o in individually]), atol=1e-10
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_energy_is_size_extensive(architecture):
    """A perfect 2x1x1 supercell must have exactly twice the energy."""
    unit = bulk("Cu", "fcc", a=3.6, cubic=True)
    super_cell = unit * (2, 1, 1)
    z_table = ZTable.from_atoms([unit])
    model = make_model(architecture, z_table)

    single = predict(model, unit, z_table)["energy"]
    doubled = predict(model, super_cell, z_table)["energy"]
    assert torch.allclose(doubled, 2.0 * single, rtol=1e-9)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_isolated_atom_has_no_forces_and_no_geometry_dependence(architecture):
    """With no neighbours there is nothing to depend on: the energy is a
    per-element constant and the forces vanish."""
    z_table = ZTable([1, 8])
    model = make_model(architecture, z_table, num_layers=1)
    here = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    there = Atoms("O", positions=[[11.0, -4.0, 3.0]])

    out = predict(model, here, z_table)
    assert torch.allclose(out["forces"], torch.zeros(1, 3), atol=1e-12)
    assert torch.allclose(out["energy"], predict(model, there, z_table)["energy"], atol=1e-12)

    # Two atoms further apart than the receptive field cannot interact.
    pair = Atoms("O2", positions=[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    assert torch.allclose(predict(model, pair, z_table)["energy"], 2.0 * out["energy"], atol=1e-10)


def test_strictly_local_isolated_atom_is_exactly_the_reference_energy():
    """Its energy is a sum over *pairs*, so an atom with no neighbours
    contributes exactly its reference energy -- a message-passing model
    instead keeps a learned per-element constant from the self-interaction."""
    z_table = ZTable([1, 8])
    config = ModelConfig(type="strictly_local", r_max=R_MAX, lmax=1, channels=4, num_layers=1)
    model = build_model(
        config, z_table.zs, atomic_energies=[-13.6, -2000.0], avg_num_neighbors=1.0
    )
    out = predict(model, Atoms("O", positions=[[0.0, 0.0, 0.0]]), z_table)
    assert float(out["energy"]) == pytest.approx(-2000.0)


def _energy_response_of_atom_zero(model, z_table, displacement: float) -> float:
    """How much the energy of atom 0 changes when a distant atom is moved.

    The three atoms sit in a line, each pair one 0.8 cutoff apart, so atom 2 is
    1.6 cutoffs from atom 0: outside the range of a single hop, inside the
    range of two.
    """
    spacing = R_MAX * 0.8
    atoms = Atoms(
        "H3", positions=[[0.0, 0.0, 0.0], [spacing, 0.0, 0.0], [2 * spacing, 0.0, 0.0]]
    )
    moved = atoms.copy()
    moved.positions[2, 1] += displacement

    before = predict(model, atoms, z_table)["node_energy"][0]
    after = predict(model, moved, z_table)["node_energy"][0]
    return float((after - before).abs())


def test_strictly_local_receptive_field_is_one_cutoff():
    """The defining property of the Allegro-style model: moving an atom more
    than one cutoff away leaves the central atom's energy *bit-for-bit* the
    same, however many layers are stacked."""
    z_table = ZTable([1])
    model = make_model("strictly_local", z_table, num_layers=3)
    assert _energy_response_of_atom_zero(model, z_table, 1.0) < 1e-14


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_atomic_energy_resolves_bond_angles(architecture):
    """``E_i`` must depend on the angles at *i*, not only on the distances to
    its neighbours.

    This is the test that separates a genuine many-body potential from a radial
    descriptor network, and the only one in this file that does: an
    angle-blind model satisfies every symmetry and conservation check here
    perfectly, because invariance under rotation is exactly what it has in
    abundance.  ``strictly_local`` failed this before it gained the equivariant
    environment tensor, with a response of *zero* to machine precision.
    """
    z_table = ZTable([8])
    model = make_model(architecture, z_table, num_layers=2)

    energies = []
    for theta in (0.6, 1.2, 2.0):
        # Three neighbours, all at 1.9 A from atom 0: every r_0k is the same in
        # all three geometries and only the angles subtended at atom 0 differ.
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.9, 0.0, 0.0],
                [0.0, 1.9, 0.0],
                [1.9 * np.cos(theta), 0.0, 1.9 * np.sin(theta)],
            ]
        )
        atoms = Atoms(numbers=[8] * 4, positions=positions, cell=np.eye(3) * 30.0, pbc=False)
        energies.append(float(predict(model, atoms, z_table)["node_energy"][0]))

    assert max(energies) - min(energies) > 1e-8


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_energy_is_smooth_across_the_cutoff(architecture):
    """No kink where a neighbour enters the cutoff sphere.

    The cutoff envelope vanishes with its first two derivatives, so the energy
    is C^2 and the second derivative near ``r_max`` must be no larger than in
    the interior.  A C^0 or C^1 potential injects energy at every crossing and
    ruins microcanonical dynamics; this catches a normalisation or read-out
    change that quietly breaks the property.
    """
    z_table = ZTable([8])
    model = make_model(architecture, z_table)

    def energy(r: float) -> float:
        positions = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.7, 0.0], [r, 0.0, 0.0]])
        atoms = Atoms(numbers=[8] * 4, positions=positions, cell=np.eye(3) * 30.0, pbc=False)
        return float(predict(model, atoms, z_table)["energy"])

    h = 2e-3
    curvature = lambda centre: np.abs(  # noqa: E731
        np.diff([energy(centre + k * h) for k in range(-6, 7)], 2) / h**2
    ).max()
    assert curvature(R_MAX) < 10.0 * max(curvature(2.0), 1e-3)


def test_strictly_local_is_angle_blind_without_the_environment_tensor():
    """Pin the failure mode the flag exists to reproduce.

    Coupling ``V_ij`` against the edge's own harmonic keeps it proportional to
    ``Y(rhat_ij)`` at every depth, so every invariant read out of it is a
    constant and the atomic energy collapses to a function of the distances.
    """
    z_table = ZTable([8])
    model = make_model(
        "strictly_local", z_table, environment_tensor=False, refresh_environment=False
    )

    energies = []
    for theta in (0.6, 1.2, 2.0):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.9, 0.0, 0.0],
                [0.0, 1.9, 0.0],
                [1.9 * np.cos(theta), 0.0, 1.9 * np.sin(theta)],
            ]
        )
        atoms = Atoms(numbers=[8] * 4, positions=positions, cell=np.eye(3) * 30.0, pbc=False)
        energies.append(float(predict(model, atoms, z_table)["node_energy"][0]))

    assert max(energies) - min(energies) < 1e-14


def test_environment_tensor_leaves_the_receptive_field_alone():
    """The fix aggregates over N(i), so it buys angular resolution for free.

    Guards the one way this change could have gone wrong: aggregating over the
    neighbours of *j* instead of *i* would also fix the angles, and would
    silently turn the model into a message-passing one.
    """
    z_table = ZTable([1])
    model = make_model("strictly_local", z_table, num_layers=3, environment_tensor=True)
    assert _energy_response_of_atom_zero(model, z_table, 1.0) < 1e-14


def test_message_passing_reaches_beyond_one_cutoff():
    """The counterpart: two MACE layers propagate information two cutoffs, so
    the same displacement does move the central atom's energy."""
    z_table = ZTable([1])
    two_layers = make_model("mace", z_table, num_layers=2)
    assert _energy_response_of_atom_zero(two_layers, z_table, 1.0) > 1e-10
    # ... and one layer alone does not.
    one_layer = make_model("mace", z_table, num_layers=1)
    assert _energy_response_of_atom_zero(one_layer, z_table, 1.0) < 1e-14


# ------------------------------------------------------------------------ stress
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stress_matches_finite_differences(architecture, crystal):
    """The analytic stress must be the numerical derivative of the energy.

    The strain is applied to the cell *and* the scaled positions, which is the
    deformation the analytic route differentiates; getting the shifts rebuilt
    from the strained cell is the only reason the energy depends on the cell at
    all.
    """
    z_table = ZTable([29])
    model = make_model(architecture, z_table)
    atoms = crystal * (2, 1, 1)

    analytic = predict(model, atoms, z_table, compute_stress=True)["stress"][0]
    volume = abs(np.linalg.det(atoms.get_cell().array))
    cell0 = atoms.get_cell().array.copy()
    scaled = atoms.get_scaled_positions().copy()

    def energy(a: int, b: int, eps: float) -> float:
        deform = np.eye(3)
        deform[a, b] += eps / 2
        deform[b, a] += eps / 2
        strained = atoms.copy()
        strained.set_cell(cell0 @ deform, scale_atoms=False)
        strained.set_scaled_positions(scaled)
        return float(predict(model, strained, z_table, compute_forces=False)["energy"])

    h = 1e-6
    numerical = np.array(
        [[(energy(a, b, h) - energy(a, b, -h)) / (2 * h) / volume for b in range(3)]
         for a in range(3)]
    )
    assert np.abs(analytic.detach().numpy() - numerical).max() < 1e-8


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stress_is_symmetric(architecture, crystal):
    """Only the symmetric part of the strain is physical, and it is all we apply."""
    z_table = ZTable([29])
    stress = predict(make_model(architecture, z_table), crystal, z_table,
                     compute_stress=True)["stress"][0]
    assert torch.allclose(stress, stress.T, atol=1e-14)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stress_is_rotation_covariant(architecture, crystal):
    """A rank-two tensor: ``sigma -> Q sigma Q^T``."""
    z_table = ZTable([29])
    model = make_model(architecture, z_table)
    rotation = rotation_matrix(0.6, 1.2, 0.4).to(torch.get_default_dtype())

    rotated = crystal.copy()
    rotated.set_positions(crystal.get_positions() @ rotation.numpy().T)
    rotated.set_cell(crystal.get_cell().array @ rotation.numpy().T)

    plain = predict(model, crystal, z_table, compute_stress=True)["stress"][0]
    turned = predict(model, rotated, z_table, compute_stress=True)["stress"][0]
    assert torch.allclose(turned, rotation @ plain @ rotation.T, atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stress_is_zero_without_a_cell(architecture, cluster):
    """A molecule has no volume, so no stress -- and certainly no NaN."""
    z_table = ZTable([1, 6, 8])
    stress = predict(make_model(architecture, z_table), cluster, z_table,
                     compute_stress=True)["stress"]
    assert torch.isfinite(stress).all()
    assert torch.count_nonzero(stress) == 0


def test_stress_requires_the_lattice_offsets(crystal):
    """Hand-assembled graphs lack them; the error must say so."""
    z_table = ZTable([29])
    model = make_model("strictly_local", z_table)
    graph = graph_from_atoms(crystal, z_table, R_MAX, with_labels=False)
    graph.unit_shifts = None
    with pytest.raises(ValueError, match="unit_shifts"):
        model(collate_graphs([graph]), compute_stress=True, training=False)


def test_registry_rejects_unknown_architectures():
    with pytest.raises(KeyError, match="unknown architecture"):
        get_model_class("does_not_exist")


def test_registry_accepts_new_architectures():
    @register_model("test_dummy")
    class Dummy(InteratomicPotential):
        def __init__(self, config, atomic_numbers, **kwargs):
            super().__init__(config.r_max, atomic_numbers, **kwargs)

        def node_energies(self, data, vectors, lengths):
            return vectors.new_zeros(data.num_nodes)

    assert "test_dummy" in available_models()
    assert isinstance(build_model(ModelConfig(type="test_dummy"), [1]), Dummy)
