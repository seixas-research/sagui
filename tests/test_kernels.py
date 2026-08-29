"""Equivalence tests for the interchangeable inference fast paths.

Nothing here checks physics -- ``test_models.py`` does that.  These tests pin
the one property the optimisations must never break: swapping a kernel changes
how a quantity is computed, never what it is.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from sagui.config import ModelConfig
from sagui.data.atomic_data import graph_from_atoms
from sagui.data.neighborlist import build_neighbor_list, has_fast_neighbor_list
from sagui.data.ztable import ZTable
from sagui.models.registry import available_models, build_model
from sagui.nn.blocks import (
    TENSOR_PRODUCT_KINDS,
    EquivariantRMSNorm,
    GemmWeightedTensorProduct,
    build_weighted_tensor_product,
    invariant_features,
)
from sagui.nn.o3 import SphericalLayout, rotation_matrix, wigner_D

ARCHITECTURES = sorted(available_models())


# --------------------------------------------------------------- tensor product
@pytest.mark.parametrize("lmax", [0, 1, 2, 3])
def test_gemm_tensor_product_matches_the_path_loop(lmax):
    """The fused kernel must reproduce the reference, not merely approximate it."""
    reference = build_weighted_tensor_product("loop", lmax, lmax, lmax)
    fused = build_weighted_tensor_product("gemm", lmax, lmax, lmax, channels=8)
    dim = (lmax + 1) ** 2
    x = torch.randn(64, 8, dim)
    sh = torch.randn(64, dim)
    weights = torch.randn(64, reference.num_paths, 8)
    assert torch.allclose(reference(x, sh, weights), fused(x, sh, weights), atol=1e-12)


def test_gemm_tensor_product_accepts_a_channelled_second_operand():
    """Needed for products against an aggregated (per-atom) equivariant tensor."""
    fused = build_weighted_tensor_product("gemm", 2, 2, 2, channels=4)
    x = torch.randn(16, 4, 9)
    y = torch.randn(16, 4, 9)
    weights = torch.randn(16, fused.num_paths, 4)
    assert fused(x, y, weights).shape == (16, 4, 9)


def test_gemm_tensor_product_chunking_is_transparent():
    """Chunking bounds peak memory; it must not change the result."""
    whole = GemmWeightedTensorProduct(2, 2, 2, chunk_edges=10_000)
    chunked = GemmWeightedTensorProduct(2, 2, 2, chunk_edges=7)
    x = torch.randn(50, 4, 9)
    sh = torch.randn(50, 9)
    weights = torch.randn(50, whole.num_paths, 4)
    assert torch.allclose(whole(x, sh, weights), chunked(x, sh, weights), atol=1e-14)


def test_gemm_tensor_product_rejects_wrong_weight_count():
    fused = build_weighted_tensor_product("gemm", 2, 2, 2, channels=4)
    with pytest.raises(ValueError, match="path weights"):
        fused(torch.randn(4, 4, 9), torch.randn(4, 9), torch.randn(4, 3, 4))


def test_unknown_tensor_product_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown tensor product"):
        build_weighted_tensor_product("fourier", 1, 1, 1)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_both_tensor_products_give_the_same_model(architecture, crystal):
    """Energies and forces must not depend on the kernel schedule."""
    settings = dict(type=architecture, r_max=4.0, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16], radial_mlp_hidden=[16])
    graph = graph_from_atoms(crystal, z_table=ZTable([29]), r_max=4.0, with_labels=False)

    outputs = {}
    for kind in TENSOR_PRODUCT_KINDS:
        torch.manual_seed(0)
        model = build_model(ModelConfig(**settings, tensor_product=kind), atomic_numbers=[29])
        outputs[kind] = model(graph, training=False)

    assert torch.allclose(outputs["loop"]["energy"], outputs["gemm"]["energy"], atol=1e-10)
    assert torch.allclose(outputs["loop"]["forces"], outputs["gemm"]["forces"], atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_a_checkpoint_is_portable_between_tensor_products(architecture):
    """The kernels carry no parameters, so weights must load across both."""
    settings = dict(type=architecture, lmax=1, channels=4, num_layers=1, latent_dim=8,
                    scalar_mlp_hidden=[8], radial_mlp_hidden=[8])
    loop = build_model(ModelConfig(**settings, tensor_product="loop"), atomic_numbers=[1])
    gemm = build_model(ModelConfig(**settings, tensor_product="gemm"), atomic_numbers=[1])
    assert set(loop.state_dict()) == set(gemm.state_dict())
    gemm.load_state_dict(loop.state_dict())  # must not raise


# -------------------------------------------------------------- normalisation
def _rotate(v, layout, R):
    return torch.cat(
        [v[..., SphericalLayout.block(l)] @ wigner_D(l, R).T for l in layout.ls], dim=-1
    )


def test_equivariant_rms_norm_is_equivariant():
    """The divisor is a sum of squared norms, so it must be rotation invariant."""
    layout = SphericalLayout(lmax=2, channels=4)
    norm = EquivariantRMSNorm(layout)
    with torch.no_grad():
        norm.gain.normal_(1.0, 0.2)
    v = torch.randn(5, 4, 9) * 7.3
    R = rotation_matrix(0.7, 1.1, 0.3)
    assert torch.allclose(norm(_rotate(v, layout, R)), _rotate(norm(v), layout, R), atol=1e-12)


def test_equivariant_rms_norm_is_scale_invariant():
    """Which is why the 1/sqrt(2) residual factor before it is a no-op."""
    layout = SphericalLayout(lmax=2, channels=4)
    norm = EquivariantRMSNorm(layout, eps=1e-30)
    v = torch.randn(5, 4, 9)
    assert torch.allclose(norm(3.7 * v), norm(v), atol=1e-10)


def test_equivariant_rms_norm_preserves_the_higher_degree_invariants():
    """Normalising *per channel* instead of across them would be a silent disaster.

    :func:`invariant_features` reads the per-channel mean square straight back
    out, so a per-channel normalisation would pin every ``l > 0`` invariant to
    the gain and destroy that half of the scalar track -- the same class of
    blindness the equivariant environment tensor exists to avoid.
    """
    layout = SphericalLayout(lmax=2, channels=4)
    norm = EquivariantRMSNorm(layout)
    invariants = invariant_features(norm(torch.randn(16, 4, 9)), layout)
    higher = invariants[:, layout.channels :]  # the l > 0 half
    assert higher.std() > 0.1


def test_equivariant_rms_norm_bounds_a_wild_input():
    """The point of the layer: bound the invariants the scalar track reads."""
    layout = SphericalLayout(lmax=2, channels=4)
    norm = EquivariantRMSNorm(layout)
    wild = torch.randn(8, 4, 9) * 1e4
    assert float(norm(wild).abs().max().detach()) < 50.0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_layer_norm_can_be_switched_off(architecture, crystal):
    """Both settings must build, run and produce finite energies and forces."""
    graph = graph_from_atoms(crystal, z_table=ZTable([29]), r_max=4.0, with_labels=False)
    for enabled in (False, True):
        torch.manual_seed(0)
        model = build_model(
            ModelConfig(type=architecture, r_max=4.0, lmax=2, channels=8, num_layers=2,
                        latent_dim=16, scalar_mlp_hidden=[16], radial_mlp_hidden=[16],
                        layer_norm=enabled),
            atomic_numbers=[29],
        )
        out = model(graph, training=False)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()


# ------------------------------------------------------- cross-degree invariants
def test_cross_degree_invariant_is_rotation_invariant():
    layout = SphericalLayout(lmax=2, channels=3)
    v = torch.randn(4, 3, 9)
    R = rotation_matrix(0.7, 1.1, 0.3)
    turned = invariant_features(_rotate(v, layout, R), layout, cross_degree=True)
    plain = invariant_features(v, layout, cross_degree=True)
    assert torch.allclose(turned, plain, atol=1e-12)


def test_cross_degree_invariant_sees_what_the_norms_cannot():
    """Rotate only the l=2 block: the per-degree norms are blind to it.

    That is exactly the information the extra term exists to expose -- how the
    degree-one and degree-two blocks sit relative to one another.
    """
    layout = SphericalLayout(lmax=2, channels=3)
    v = torch.randn(4, 3, 9)
    R = rotation_matrix(0.7, 1.1, 0.3)
    w = v.clone()
    w[..., SphericalLayout.block(2)] = v[..., SphericalLayout.block(2)] @ wigner_D(2, R).T

    norms_v = invariant_features(v, layout, cross_degree=False)
    norms_w = invariant_features(w, layout, cross_degree=False)
    assert torch.allclose(norms_v, norms_w, atol=1e-12)

    cross_v = invariant_features(v, layout, cross_degree=True)[:, -layout.channels:]
    cross_w = invariant_features(w, layout, cross_degree=True)[:, -layout.channels:]
    assert (cross_v - cross_w).abs().max() > 1e-3


def test_cross_degree_invariant_widens_the_read_out_consistently():
    from sagui.nn.blocks import num_invariants

    layout = SphericalLayout(lmax=2, channels=3)
    v = torch.randn(2, 3, 9)
    for flag in (False, True):
        assert invariant_features(v, layout, flag).shape[-1] == num_invariants(layout, flag)


def test_cross_degree_invariant_is_skipped_below_lmax_two():
    """There is no (1, 1) -> 2 path to form, so the width must not change."""
    from sagui.nn.blocks import num_invariants

    layout = SphericalLayout(lmax=1, channels=3)
    v = torch.randn(2, 3, 4)
    assert invariant_features(v, layout, True).shape[-1] == num_invariants(layout, True)
    assert torch.allclose(invariant_features(v, layout, True), invariant_features(v, layout))


def test_cross_degree_model_stays_equivariant(crystal):
    """The extra scalar must not leak into the transformation law."""
    graph = graph_from_atoms(crystal, z_table=ZTable([29]), r_max=4.0, with_labels=False)
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(type="strictly_local", r_max=4.0, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16], cross_degree_invariants=True),
        atomic_numbers=[29],
    )
    rotated = crystal.copy()
    R = rotation_matrix(0.6, 1.2, 0.4).numpy()
    rotated.set_positions(crystal.get_positions() @ R.T)
    rotated.set_cell(crystal.get_cell().array @ R.T)
    turned = graph_from_atoms(rotated, z_table=ZTable([29]), r_max=4.0, with_labels=False)

    a, b = model(graph, training=False), model(turned, training=False)
    assert torch.allclose(a["energy"], b["energy"], atol=1e-10)
    assert torch.allclose(
        a["forces"] @ torch.as_tensor(R, dtype=a["forces"].dtype).T, b["forces"], atol=1e-10
    )


# ------------------------------------------------------------------ compilation
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_compile_layers_preserves_the_state_dict(architecture):
    """Compiling the module (rather than its forward) would rename every key."""
    model = build_model(
        ModelConfig(type=architecture, lmax=1, channels=4, num_layers=2), atomic_numbers=[1]
    )
    before = list(model.state_dict())
    assert model.compile_layers() == 2
    assert list(model.state_dict()) == before


def test_compile_layers_warns_for_the_unfused_kernel():
    model = build_model(
        ModelConfig(type="strictly_local", lmax=1, channels=4, tensor_product="loop"),
        atomic_numbers=[1],
    )
    with pytest.warns(RuntimeWarning, match="gemm"):
        model.compile_layers()


@pytest.mark.skipif(
    os.environ.get("SAGUI_TEST_COMPILE") != "1",
    reason="torch.compile takes ~1 min per graph; set SAGUI_TEST_COMPILE=1 to run",
)
def test_compiled_layers_reproduce_eager(crystal):
    graph = graph_from_atoms(crystal, z_table=ZTable([29]), r_max=4.0, with_labels=False)
    settings = dict(type="strictly_local", r_max=4.0, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16])
    torch.manual_seed(0)
    eager = build_model(ModelConfig(**settings), atomic_numbers=[29])
    torch.manual_seed(0)
    compiled = build_model(ModelConfig(**settings), atomic_numbers=[29])
    compiled.compile_layers()
    a, b = eager(graph, training=False), compiled(graph, training=False)
    assert torch.allclose(a["energy"], b["energy"], atol=1e-8)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-8)


# --------------------------------------------------------------- neighbour list
@pytest.mark.skipif(not has_fast_neighbor_list(), reason="vesin is not installed")
@pytest.mark.parametrize("periodic", [True, False])
def test_vesin_and_ase_agree_edge_for_edge(periodic, monkeypatch):
    """Both backends must return the same edges *and* the same lattice offsets."""
    if periodic:
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.rattle(stdev=0.05, seed=7)
    else:
        rng = np.random.default_rng(0)
        atoms = Atoms("H3O2C2", positions=rng.normal(scale=1.7, size=(7, 3)))

    fast = build_neighbor_list(atoms, 4.0)
    monkeypatch.setattr("sagui.data.neighborlist._VesinNeighborList", None)
    slow = build_neighbor_list(atoms, 4.0)

    def canonical(edge_index, unit_shifts):
        return sorted(
            zip(
                edge_index[0].tolist(),
                edge_index[1].tolist(),
                map(tuple, unit_shifts.tolist()),
                strict=True,
            )
        )

    assert canonical(fast[0], fast[2]) == canonical(slow[0], slow[2])
    assert np.allclose(np.sort(fast[1], axis=0), np.sort(slow[1], axis=0))


@pytest.mark.skipif(not has_fast_neighbor_list(), reason="vesin is not installed")
def test_mixed_periodicity_falls_back_to_ase():
    """vesin cannot express a synthetic box for the open directions."""
    slab = bulk("Cu", "fcc", a=3.6, cubic=True) * (1, 1, 2)
    slab.set_pbc((True, True, False))
    cell = slab.get_cell().array.copy()
    cell[2] *= 3
    slab.set_cell(cell)
    edge_index, shifts, unit_shifts = build_neighbor_list(slab, 4.0)
    assert edge_index.shape[0] == 2
    assert (unit_shifts[:, 2] == 0).all()  # no wrap along the open direction
