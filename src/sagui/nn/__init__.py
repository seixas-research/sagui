"""Equivariant neural-network primitives (no ``e3nn`` dependency)."""

from .blocks import (
    EquivariantLinear,
    SelfTensorProduct,
    SpeciesLinear,
    WeightedTensorProduct,
    embed_scalars,
    invariant_features,
    num_invariants,
    one_hot_species,
    scalars,
)
from .o3 import LMAX_SUPPORTED, SphericalLayout, spherical_harmonics, wigner_3j, wigner_D
from .radial import MLP, BesselBasis, PolynomialCutoff
from .scatter import scatter_sum

__all__ = [
    "LMAX_SUPPORTED",
    "MLP",
    "BesselBasis",
    "EquivariantLinear",
    "PolynomialCutoff",
    "SelfTensorProduct",
    "SpeciesLinear",
    "SphericalLayout",
    "WeightedTensorProduct",
    "embed_scalars",
    "invariant_features",
    "num_invariants",
    "one_hot_species",
    "scalars",
    "scatter_sum",
    "spherical_harmonics",
    "wigner_3j",
    "wigner_D",
]
