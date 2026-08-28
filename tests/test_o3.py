"""The O(3) algebra: harmonics, representation matrices, invariant tensors."""

from __future__ import annotations

import math

import pytest
import torch

from sagui.nn.o3 import (
    LMAX_SUPPORTED,
    SphericalLayout,
    rotation_matrix,
    spherical_harmonics,
    wigner_3j,
    wigner_D,
)

ALL_L = list(range(LMAX_SUPPORTED + 1))


@pytest.fixture
def points() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(64, 3, dtype=torch.float64)


@pytest.mark.parametrize("l", ALL_L)
def test_component_normalisation(l, points):
    """sum_m Y_lm^2 = 2l + 1 on the unit sphere (component convention)."""
    block = spherical_harmonics(l, points)[:, l * l :]
    assert torch.allclose(block.pow(2).sum(-1), torch.full((len(points),), 2.0 * l + 1))


@pytest.mark.parametrize("l", ALL_L)
def test_homogeneous_polynomials(l, points):
    """Without normalisation each block is homogeneous of degree l."""
    scale = 2.7
    scaled = spherical_harmonics(l, scale * points, normalize=False)[:, l * l :]
    plain = spherical_harmonics(l, points, normalize=False)[:, l * l :]
    assert torch.allclose(scaled, scale**l * plain)


@pytest.mark.parametrize("l", ALL_L)
def test_parity(l, points):
    """Y_l(-r) = (-1)^l Y_l(r): the natural parity assumed by the layouts."""
    block = spherical_harmonics(l, points)[:, l * l :]
    inverted = spherical_harmonics(l, -points)[:, l * l :]
    assert torch.allclose(inverted, (-1.0) ** l * block)


@pytest.mark.parametrize("l", ALL_L)
def test_wigner_d_is_the_representation(l, points):
    """Y_l(Rx) = D^l(R) Y_l(x), and D is orthogonal."""
    R = rotation_matrix(0.31, 1.27, 2.02)
    D = wigner_D(l, R)
    identity = torch.eye(2 * l + 1, dtype=torch.float64)
    assert torch.allclose(D @ D.T, identity, atol=1e-12)

    rotated = spherical_harmonics(l, points @ R.T)[:, l * l :]
    transformed = spherical_harmonics(l, points)[:, l * l :] @ D.T
    assert torch.allclose(rotated, transformed, atol=1e-12)


@pytest.mark.parametrize("l", ALL_L)
def test_wigner_d_composition(l):
    """D(R1 R2) = D(R1) D(R2): a genuine group homomorphism."""
    R1, R2 = rotation_matrix(0.4, 0.9, 1.6), rotation_matrix(2.2, 0.3, 0.8)
    assert torch.allclose(wigner_D(l, R1 @ R2), wigner_D(l, R1) @ wigner_D(l, R2), atol=1e-11)


@pytest.mark.parametrize(
    "l1,l2,l3",
    [(0, 0, 0), (1, 1, 0), (1, 1, 2), (2, 1, 1), (2, 2, 2), (3, 2, 1), (3, 3, 0), (3, 2, 3)],
)
def test_wigner_3j_is_invariant(l1, l2, l3):
    """The defining property: contracting with three D matrices is a no-op."""
    C = wigner_3j(l1, l2, l3)
    assert C.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    assert math.isclose(float(C.norm()), 1.0, rel_tol=1e-12)

    for angles in [(0.4, 1.1, 2.3), (1.9, 0.6, 0.2)]:
        R = rotation_matrix(*angles)
        rotated = torch.einsum(
            "ia,jb,kc,abc->ijk", wigner_D(l1, R), wigner_D(l2, R), wigner_D(l3, R), C
        )
        assert torch.allclose(rotated, C, atol=1e-11)


def test_wigner_3j_rejects_impossible_couplings():
    with pytest.raises(ValueError, match="triangle"):
        wigner_3j(1, 1, 3)


def test_wigner_3j_is_deterministic():
    """Checkpoints depend on the coefficients being reproducible."""
    first = wigner_3j(2, 1, 1).clone()
    wigner_3j.cache_clear()
    assert torch.equal(first, wigner_3j(2, 1, 1))


def test_wigner_3j_couples_scalars_trivially():
    """0 x l -> l must be (a multiple of) the identity."""
    C = wigner_3j(0, 2, 2)
    identity = torch.eye(5, dtype=torch.float64) / math.sqrt(5.0)
    assert torch.allclose(C[0], identity, atol=1e-12)


def test_layout_blocks():
    layout = SphericalLayout(lmax=2, channels=8)
    assert layout.dim == 9
    assert layout.ls == (0, 1, 2)
    assert SphericalLayout.block(0) == slice(0, 1)
    assert SphericalLayout.block(2) == slice(4, 9)
    with pytest.raises(ValueError):
        SphericalLayout(lmax=LMAX_SUPPORTED + 1, channels=4)


def test_spherical_harmonics_rejects_high_l(points):
    with pytest.raises(ValueError, match="lmax"):
        spherical_harmonics(LMAX_SUPPORTED + 1, points)


# ------------------------------------------------- bridges back to Cartesian
def test_l1_block_becomes_a_cartesian_vector(points):
    """A degree-one block must transform as an ordinary vector."""
    from sagui.nn.o3 import spherical_to_cartesian_vector

    R = rotation_matrix(0.5, 1.2, 2.1)
    block = torch.randn(8, 3, dtype=torch.float64)
    rotated = spherical_to_cartesian_vector(block @ wigner_D(1, R).T)
    assert torch.allclose(rotated, spherical_to_cartesian_vector(block) @ R.T, atol=1e-12)


def test_l1_conversion_agrees_with_the_harmonics(points):
    """Converting Y_1(r) back must return r itself, up to the sqrt(3) scale."""
    from sagui.nn.o3 import spherical_to_cartesian_vector

    unit = points / points.norm(dim=-1, keepdim=True)
    block = spherical_harmonics(1, unit)[:, 1:]
    assert torch.allclose(spherical_to_cartesian_vector(block), math.sqrt(3.0) * unit, atol=1e-12)


def test_l0_and_l2_blocks_become_a_symmetric_tensor():
    """S must be symmetric and transform as S -> R S R^T."""
    from sagui.nn.o3 import spherical_to_symmetric_matrix

    R = rotation_matrix(0.5, 1.2, 2.1)
    scalar = torch.randn(6, dtype=torch.float64)
    block = torch.randn(6, 5, dtype=torch.float64)

    S = spherical_to_symmetric_matrix(scalar, block)
    assert torch.allclose(S, S.transpose(-1, -2))
    rotated = spherical_to_symmetric_matrix(scalar, block @ wigner_D(2, R).T)
    assert torch.allclose(rotated, R @ S @ R.T, atol=1e-12)


def test_l2_basis_reproduces_the_harmonics(points):
    """Y_2m(r) = r^T A_m r is what defines the Cartesian basis."""
    from sagui.nn.o3 import _l2_cartesian_basis

    unit = points / points.norm(dim=-1, keepdim=True)
    quadratic = torch.einsum("ni,mij,nj->nm", unit, _l2_cartesian_basis(), unit)
    assert torch.allclose(quadratic, spherical_harmonics(2, unit)[:, 4:], atol=1e-12)


def test_traceless_part_carries_no_scalar():
    """The l=2 block must contribute nothing to the trace: the split is clean."""
    from sagui.nn.o3 import spherical_to_symmetric_matrix

    block = torch.randn(4, 5, dtype=torch.float64)
    S = spherical_to_symmetric_matrix(torch.zeros(4, dtype=torch.float64), block)
    trace = torch.einsum("nii->n", S)
    assert torch.allclose(trace, torch.zeros(4, dtype=torch.float64), atol=1e-12)
