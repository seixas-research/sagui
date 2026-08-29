"""Declarative configuration: dataclasses with YAML (de)serialisation.

A run is fully described by one YAML file plus optional command-line
overrides.  The ``model.type`` key selects the architecture from the registry,
so adding a new architecture never requires touching this module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

__all__ = ["ModelConfig", "DataConfig", "DiffusionConfig", "TrainingConfig", "Config"]


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Most entries are shared; the few architecture-specific ones are marked.
    Unused keys are simply ignored by the architecture that does not need them.
    """

    #: Registered architecture name -- ``"mace"`` or ``"strictly_local"``.
    type: str = "mace"
    #: Interaction cutoff in Angstrom.
    r_max: float = 5.0
    #: Highest degree kept in the node/edge features.
    lmax: int = 2
    #: Highest degree of the spherical harmonics of the edge directions.
    #: ``None`` -- the default -- means "track ``lmax``"; it is resolved lazily
    #: by :attr:`spherical_lmax` so that a later override of ``lmax`` still
    #: propagates.
    sh_lmax: int | None = None
    #: Feature channels per degree.
    channels: int = 32
    #: Interaction (``mace``) or tensor-product (``strictly_local``) layers.
    num_layers: int = 2
    #: Number of Bessel radial basis functions.
    num_radial_basis: int = 8
    #: Hidden widths of the MLP producing the radial filter weights.
    radial_mlp_hidden: list[int] = field(default_factory=lambda: [64, 64])
    #: Polynomial order of the cutoff envelope.
    cutoff_p: int = 6
    #: Radius below which a parameter-free ZBL nuclear repulsion is added, in
    #: angstrom; ``None`` disables it.  Training sets contain almost no close
    #: contacts, so the learned repulsive wall is an extrapolation and is
    #: usually far too soft; ~1.5-2.0 is the useful range.
    zbl_cutoff: float | None = None
    #: Hidden width of the final energy readout.
    readout_hidden: int = 16
    #: ``mace`` only: maximum correlation order of the many-body expansion.
    correlation: int = 3
    #: ``strictly_local`` only: width of the invariant (scalar) track.
    latent_dim: int = 64
    #: ``strictly_local`` only: hidden widths of the scalar-track MLPs.
    scalar_mlp_hidden: list[int] = field(default_factory=lambda: [64, 64])
    #: Add the ``(1, 1) -> 2`` cross-degree invariant to the scalar read-out.
    #: Needs ``lmax >= 2``; the per-degree norms alone cannot express how the
    #: degree-one and degree-two blocks are oriented relative to each other.
    cross_degree_invariants: bool = False
    #: ``strictly_local`` only: couple the edge tensor against an *aggregated*
    #: equivariant environment tensor rather than the edge's own spherical
    #: harmonic.  Without this the model is exactly blind to bond angles -- see
    #: the module docstring of ``models/strictly_local.py``.  Turn it off only
    #: to reproduce the pre-fix architecture.
    environment_tensor: bool = True
    #: ``strictly_local`` only: rebuild the invariant environment descriptor at
    #: every layer from the current latents, instead of once from the two-body
    #: embedding.
    refresh_environment: bool = True
    #: Normalise the equivariant features per degree, across channels, after
    #: every layer (:class:`sagui.nn.blocks.EquivariantRMSNorm`).
    #:
    #: **The sign of this one depends on the data.**  On a small clean angular
    #: benchmark it hurt badly -- force MAE 84 -> 137 meV/A even after re-tuning
    #: the learning rate for each setting.  On an 800-frame, 81-element MPtrj
    #: subset it *helped*: energy MAE 1019 -> 831 meV/atom and force MAE
    #: 56.0 -> 45.2 meV/A.  That is the regime its published gains come from.
    #: Default off because a small dataset is the usual starting point; turn it
    #: on for large, chemically diverse ones, and measure.
    layer_norm: bool = False
    #: Average neighbour count used to normalise sums; ``None`` -> from data.
    avg_num_neighbors: float | None = None
    #: Scheduling of the Clebsch-Gordan product: ``"gemm"`` (one fused matrix
    #: product, ~2x faster once compiled) or ``"loop"`` (one einsum per path,
    #: lower peak memory).  The two compute the same function and share the
    #: same parameters, so a checkpoint is portable between them.
    tensor_product: str = "gemm"

    @property
    def spherical_lmax(self) -> int:
        return int(self.sh_lmax if self.sh_lmax is not None else self.lmax)


@dataclass
class DataConfig:
    """Where the structures come from and how their labels are named."""

    train_file: str | None = None
    valid_file: str | None = None
    test_file: str | None = None
    #: Fraction held out for validation when ``valid_file`` is not given.
    valid_fraction: float = 0.1
    #: Explicit label keys; ``None`` tries a list of common conventions.
    energy_key: str | None = None
    forces_key: str | None = None
    stress_key: str | None = None
    #: Keep every neighbour list in memory (fast, only for small datasets).
    cache_graphs: bool = False
    num_workers: int = 0


@dataclass
class DiffusionConfig:
    """Forward process and loss weights of the generative model.

    Only consulted when ``task: generative``.
    """

    #: Length of the Markov chain.  Sampling cost is linear in this.
    num_steps: int = 1000
    #: D3PM kernel for the atom types: ``"absorbing"`` (masking) or ``"uniform"``.
    type_transition: str = "absorbing"
    #: Fractional-coordinate noise ladder; 1.0 already means "uniform in the cell".
    sigma_min: float = 0.005
    sigma_max: float = 0.5
    #: Relative weight of the three denoising objectives.
    type_weight: float = 1.0
    coord_weight: float = 1.0
    lattice_weight: float = 1.0
    #: Weight of the auxiliary cross-entropy in the D3PM hybrid loss.
    type_ce_weight: float = 1.0
    #: Cap on the neighbours per atom; noisy lattices can be pathologically dense.
    max_neighbors: int = 24
    #: Range the reconstructed lattice is clipped to during sampling, in units
    #: of the normalised lattice (``None`` disables the safeguard).
    lattice_clip: float | None = 3.0
    #: Randomly rotate every training cell.  The lattice process is only
    #: rotation-covariant if the data distribution is, so leave this on.
    rotation_augmentation: bool = True
    #: Numbers of atoms to generate when the user does not say; ``None`` -> use
    #: the empirical distribution of the training set.
    sample_num_atoms: list[int] | None = None


@dataclass
class TrainingConfig:
    """Optimisation schedule and bookkeeping."""

    epochs: int = 100
    batch_size: int = 8
    valid_batch_size: int | None = None
    learning_rate: float = 1e-2
    weight_decay: float = 5e-7
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    #: Weight of the stress term.  Zero (the default) skips the strain
    #: derivative entirely; a positive value needs stress labels in the data.
    stress_weight: float = 0.0
    #: Huber transition point.  ``None`` keeps the mean-square loss.  An
    #: absolute residual threshold, so it must match the data: ``0.01`` is the
    #: MACE-MP value and suits MPtrj-scale energies.  Measured on MPtrj it
    #: trades energy for forces -- force MAE 56.0 -> 36.3 meV/A but energy MAE
    #: 1019 -> 2106 meV/atom -- because a single ``delta`` is shared by terms
    #: with very different residual scales.  Per-term deltas would be the fix.
    huber_delta: float | None = None
    #: Fraction of training after which the energy and force weights swap and
    #: the learning rate drops tenfold -- the MACE-MP two-phase schedule.
    #: ``None`` disables it.
    force_weight_switch: float | None = None
    max_grad_norm: float | None = 10.0
    #: Exponential moving average of the weights; ``None`` disables it.
    ema_decay: float | None = 0.99
    lr_factor: float = 0.8
    lr_patience: int = 10
    min_lr: float = 1e-6
    seed: int = 0
    device: str = "auto"
    default_dtype: str = "float32"
    output_dir: str = "runs"
    name: str = "sagui"
    log_every: int = 1
    #: Fit per-element reference energies by least squares before training.
    fit_atomic_energies: bool = True


#: What ``sagui-train`` should fit.
TASKS = ("potential", "generative")


@dataclass
class Config:
    """Top-level configuration object.

    ``task`` selects between the two things SAGUI can learn from the same
    structures: an interatomic potential (energies and forces) or a generative
    model of the structures themselves.
    """

    task: str = "potential"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got '{self.task}'")

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw or {})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown configuration section(s): {sorted(unknown)}; "
                f"expected {sorted(f.name for f in fields(cls))}"
            )
        return cls(
            task=str(raw.get("task", "potential")),
            model=_build(ModelConfig, raw.get("model", {})),
            data=_build(DataConfig, raw.get("data", {})),
            training=_build(TrainingConfig, raw.get("training", {})),
            diffusion=_build(DiffusionConfig, raw.get("diffusion", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    # ------------------------------------------------------------ overrides
    def apply_overrides(self, overrides: dict[str, Any]) -> Config:
        """Apply ``{"model.channels": 64}``-style overrides in place.

        Values coming from the command line are strings; they are coerced to
        the type declared on the dataclass field.
        """
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            if not key:
                # A bare key addresses a scalar field of Config itself, e.g. "task".
                declared_top = {f.name: f for f in fields(self)}
                if section not in declared_top or is_dataclass(declared_top[section].type):
                    raise ValueError(f"override '{dotted}' must be of the form section.key")
                setattr(self, section, _coerce(value, declared_top[section].type))
                continue
            if section not in {f.name for f in fields(self)}:
                raise ValueError(f"unknown configuration section '{section}' in '{dotted}'")
            target = getattr(self, section)
            declared = {f.name: f for f in fields(target)}
            if key not in declared:
                raise ValueError(
                    f"unknown key '{key}' for section '{section}'; "
                    f"available: {sorted(declared)}"
                )
            setattr(target, key, _coerce(value, declared[key].type))
        self.__post_init__()
        return self


def _build(cls: type, raw: dict[str, Any]):
    """Instantiate a config dataclass, rejecting unknown keys loudly."""
    raw = dict(raw or {})
    declared = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(declared)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} for {cls.__name__}; available: {sorted(declared)}"
        )
    return cls(**{k: _coerce(v, declared[k].type) for k, v in raw.items()})


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort conversion of ``value`` to the annotated type.

    Handles the ``X | None`` unions used throughout the config and leaves
    anything it does not recognise untouched -- validation of exotic values is
    the job of whoever consumes them.
    """
    if value is None:
        return None
    if is_dataclass(annotation):
        return value

    text = annotation if isinstance(annotation, str) else None
    if text is not None:
        # ``from __future__ import annotations`` turns annotations into strings.
        options = [part.strip() for part in text.split("|")]
        if isinstance(value, str) and value.lower() in {"none", "null"} and "None" in options:
            return None
        for option in options:
            if option == "None":
                continue
            if option.startswith("list"):
                return _coerce_list(value, option)
            if option == "bool":
                return _to_bool(value)
            if option == "int":
                return int(value)
            if option == "float":
                return float(value)
            if option == "str":
                return str(value)
        return value

    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _coerce(value, args[0]) if args else value
    if annotation is bool:
        return _to_bool(value)
    if annotation in (int, float, str):
        return annotation(value)
    return value


def _coerce_list(value: Any, annotation: str) -> list:
    """Parse a list-valued option, including the ``--set key=[64, 64]`` spelling.

    Command-line overrides arrive as strings, so ``"[64, 64]"`` has to be
    parsed rather than iterated over character by character.
    """
    if isinstance(value, str):
        value = yaml.safe_load(value)
    if not isinstance(value, (list, tuple)):
        value = [value]
    inner = annotation[annotation.find("[") + 1 : annotation.rfind("]")].strip()
    if inner == "int":
        return [int(item) for item in value]
    if inner == "float":
        return [float(item) for item in value]
    return list(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1", "on"}:
            return True
        if value.lower() in {"false", "no", "0", "off"}:
            return False
        raise ValueError(f"cannot interpret '{value}' as a boolean")
    return bool(value)
