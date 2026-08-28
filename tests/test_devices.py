"""Every backend SAGUI claims to support, exercised on the hardware present.

Devices are discovered at collection time, so this file is a no-op beyond CPU
on a machine without an accelerator and full coverage on one with. The
preference order SAGUI advertises is CUDA, then Apple MPS, then CPU.

Precision note: Metal has no ``float64`` whatsoever, so the MPS cases run in
``float32`` and use correspondingly loose tolerances. Anything asserting
machine-precision equivariance belongs in ``test_models.py``, which is CPU and
double precision by design.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from sagui import SaguiCalculator, load_model
from sagui.checkpoint import save_checkpoint
from sagui.config import Config, DiffusionConfig, ModelConfig
from sagui.data import ZTable, collate_graphs, graph_from_atoms
from sagui.data.statistics import DatasetStatistics
from sagui.generative import MaterialsDiffusion, collate_diffusion
from sagui.generative.dataset import DiffusionDataset
from sagui.models import build_model
from sagui.nn.o3 import rotation_matrix
from sagui.utils import resolve_device, resolve_device_and_dtype

R_MAX = 4.5


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = _available_devices()

#: MPS is float32-only, and float32 message passing accumulates ~1e-6 error.
TOLERANCES = {"cpu": 1e-10, "cuda": 1e-4, "mps": 1e-4}


@pytest.fixture(params=DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture
def dtype_for(device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


@pytest.fixture(autouse=True)
def _default_float32():
    """Override the double-precision default: MPS cannot do float64."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def _cluster() -> Atoms:
    rng = np.random.default_rng(0)
    positions = rng.normal(scale=1.7, size=(7, 3))
    positions[0] += np.array([2.0, 0.0, 0.0])
    return Atoms("H3O2C2", positions=positions)


def _potential(architecture: str, z_table: ZTable, device, dtype):
    config = ModelConfig(
        type=architecture, r_max=R_MAX, lmax=2, channels=8, num_layers=2,
        num_radial_basis=6, radial_mlp_hidden=[16], scalar_mlp_hidden=[16],
        latent_dim=16, correlation=2,
    )
    model = build_model(config, z_table.zs, avg_num_neighbors=6.0)
    return model.to(device=device, dtype=dtype)


def _batch(atoms: Atoms, z_table: ZTable, device, dtype):
    graph = graph_from_atoms(atoms, z_table, R_MAX, with_labels=False, dtype=dtype)
    return collate_graphs([graph]).to(device)


# ------------------------------------------------------- device resolution
def test_auto_prefers_cuda_then_mps_then_cpu():
    """The advertised preference order, checked against what is installed."""
    resolved = resolve_device("auto").type
    if torch.cuda.is_available():
        assert resolved == "cuda"
    elif torch.backends.mps.is_available():
        assert resolved == "mps"
    else:
        assert resolved == "cpu"


@pytest.mark.parametrize("name", DEVICES)
def test_explicit_device_requests_are_honoured(name):
    assert resolve_device(name).type == name


def test_float64_on_mps_is_downgraded_not_crashed(caplog):
    """Metal has no float64; asking for it must warn, not explode mid-forward."""
    device, dtype = resolve_device_and_dtype("mps", "float64")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS on this machine")
    assert device.type == "mps"
    assert dtype == torch.float32
    assert any("float64" in record.message for record in caplog.records)


def test_float64_survives_on_cpu():
    device, dtype = resolve_device_and_dtype("cpu", "float64")
    assert device.type == "cpu" and dtype == torch.float64


# ------------------------------------------------------------- potentials
@pytest.mark.parametrize("architecture", ["mace", "strictly_local"])
def test_forward_and_forces_run_on_device(architecture, device, dtype_for):
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    model = _potential(architecture, z_table, device, dtype_for)
    out = model(_batch(atoms, z_table, device, dtype_for), compute_forces=True, training=False)

    assert out["energy"].device.type == device.type
    assert out["forces"].device.type == device.type
    assert out["forces"].shape == (len(atoms), 3)
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()


@pytest.mark.parametrize("architecture", ["mace", "strictly_local"])
def test_device_agrees_with_cpu(architecture, device, dtype_for):
    """The same weights must give the same answer wherever they run."""
    if device.type == "cpu":
        pytest.skip("this is the reference")
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])

    torch.manual_seed(7)
    reference = _potential(architecture, z_table, torch.device("cpu"), dtype_for)
    on_device = _potential(architecture, z_table, device, dtype_for)
    on_device.load_state_dict({k: v.to(device) for k, v in reference.state_dict().items()})

    expected = reference(
        _batch(atoms, z_table, torch.device("cpu"), dtype_for), training=False
    )
    actual = on_device(_batch(atoms, z_table, device, dtype_for), training=False)

    tol = TOLERANCES[device.type]
    assert torch.allclose(actual["energy"].cpu(), expected["energy"], atol=tol, rtol=tol)
    assert torch.allclose(actual["forces"].cpu(), expected["forces"], atol=tol, rtol=tol)


@pytest.mark.parametrize("architecture", ["mace", "strictly_local"])
def test_rotation_equivariance_holds_on_device(architecture, device, dtype_for):
    """Equivariance is a property of the maths, not of the backend."""
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    model = _potential(architecture, z_table, device, dtype_for)

    rotation = rotation_matrix(0.42, 1.13, 2.31).numpy()
    rotated = atoms.copy()
    rotated.positions = rotated.positions @ rotation.T

    plain = model(_batch(atoms, z_table, device, dtype_for), training=False)
    turned = model(_batch(rotated, z_table, device, dtype_for), training=False)

    tol = TOLERANCES[device.type]
    assert torch.allclose(turned["energy"], plain["energy"], atol=tol, rtol=tol)
    expected = plain["forces"] @ torch.as_tensor(
        rotation.T, dtype=dtype_for, device=device
    )
    assert torch.allclose(turned["forces"], expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("architecture", ["mace", "strictly_local"])
def test_training_step_runs_on_device(architecture, device, dtype_for):
    """Force training needs a second derivative through the whole network --
    the operation most likely to be missing on an accelerator backend."""
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    model = _potential(architecture, z_table, device, dtype_for)
    batch = _batch(atoms, z_table, device, dtype_for)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(batch, compute_forces=True, training=True)
    loss = out["energy"].pow(2).mean() + out["forces"].pow(2).mean()
    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    optimizer.step()


def test_checkpoint_round_trips_across_devices(tmp_path, device, dtype_for):
    """Weights trained on an accelerator must reload (and be reloadable on CPU)."""
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    model = _potential("mace", z_table, device, dtype_for)

    config = Config()
    config.model = model.config
    stats = DatasetStatistics(
        atomic_energies=np.zeros(len(z_table)), energy_scale=1.0, avg_num_neighbors=6.0
    )
    path = save_checkpoint(tmp_path / "m.model", model, config, z_table, stats)

    on_cpu, _, _ = load_model(path, device="cpu", dtype=torch.float64)
    assert next(on_cpu.parameters()).device.type == "cpu"

    back, _, _ = load_model(path, device=device, dtype=dtype_for)
    assert next(back.parameters()).device.type == device.type
    out = back(_batch(atoms, z_table, device, dtype_for), training=False)
    assert torch.isfinite(out["energy"]).all()


def test_calculator_runs_on_device(tmp_path, device):
    """The ASE entry point, which is where most users meet the device logic."""
    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    dtype = torch.float32 if device.type == "mps" else torch.float64
    model = _potential("mace", z_table, device, dtype)

    config = Config()
    config.model = model.config
    stats = DatasetStatistics(
        atomic_energies=np.zeros(len(z_table)), energy_scale=1.0, avg_num_neighbors=6.0
    )
    path = save_checkpoint(tmp_path / "m.model", model, config, z_table, stats)

    atoms.calc = SaguiCalculator(
        path, device=device.type, default_dtype="float32" if device.type == "mps" else "float64"
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3) and np.isfinite(forces).all()
    assert isinstance(energy, float), "ASE expects plain Python floats back"


# ------------------------------------------------------------- generative
def _diffusion(device, dtype, num_species=2):
    model_config = ModelConfig(
        type="mace", r_max=5.0, lmax=2, channels=8, num_layers=1,
        num_radial_basis=6, radial_mlp_hidden=[16], scalar_mlp_hidden=[16], correlation=2,
    )
    model = MaterialsDiffusion(
        model_config, DiffusionConfig(num_steps=20), num_species=num_species,
        lattice_scale=2.1, avg_num_neighbors=12.0,
    )
    return model.to(device=device, dtype=dtype)


def test_diffusion_training_step_runs_on_device(device, dtype_for):
    """Also covers the trap that the dataset corrupts on CPU while the model's
    schedule buffers live on the accelerator."""
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(3):
        atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True)
        atoms.rattle(stdev=0.08, seed=int(rng.integers(1 << 30)))
        frames.append(atoms)

    model = _diffusion(device, dtype_for)
    z_table = ZTable.from_atoms(frames)
    dataset = DiffusionDataset(
        frames, z_table, model.corruption, r_max=5.0, lattice_scale=2.1, dtype=dtype_for
    )
    assert all(b.device.type == "cpu" for b in dataset.corruption.buffers()), (
        "the dataset must keep a CPU copy of the corruption schedules"
    )

    batch = collate_diffusion([dataset[i] for i in range(3)]).to(device)
    loss, terms = model.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.device.type == device.type
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_sampling_runs_on_device(device, dtype_for):
    """Sampling rebuilds neighbour lists on CPU each step and moves back, and
    clamps the lattice through an SVD -- which Metal cannot do in float64."""
    model = _diffusion(device, dtype_for)
    structures = model.sample([4, 6], device=device, num_steps=20)

    assert len(structures) == 2
    for structure, size in zip(structures, [4, 6], strict=True):
        assert structure.frac.shape == (size, 3)
        assert torch.isfinite(structure.cell).all()
        assert float(torch.linalg.det(structure.cell)) > 0.0
        assert int(structure.species.max()) < model.num_species


# ------------------------------------------------- non-finite gradient guard
def test_clip_gradients_reports_the_norm_and_clips():
    from sagui.utils import clip_gradients

    layer = torch.nn.Linear(4, 4)
    layer(torch.ones(2, 4)).sum().backward()
    norm = clip_gradients(layer, max_norm=1e-3)
    assert torch.isfinite(norm)
    after = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in layer.parameters()])
    )
    assert float(after) <= 1e-3 + 1e-6


def test_clip_gradients_detects_a_poisoned_backward():
    """An inf gradient must be *reported*, because clip_grad_norm_ divides by
    the total norm and would turn every parameter into NaN."""
    from sagui.utils import clip_gradients

    layer = torch.nn.Linear(4, 4)
    layer(torch.ones(2, 4)).sum().backward()
    layer.weight.grad[0, 0] = float("inf")
    assert not torch.isfinite(clip_gradients(layer, max_norm=10.0))


def test_a_single_bad_batch_does_not_destroy_the_run(tmp_path, monkeypatch):
    """The regression this guard exists for: one overflowing batch used to turn
    every weight -- and the EMA shadow -- into NaN for the rest of training."""
    import sagui.train.trainer as trainer_module
    from sagui.train.trainer import _train_epoch

    atoms = _cluster()
    z_table = ZTable.from_atoms([atoms])
    model = _potential("mace", z_table, torch.device("cpu"), torch.float64)
    graph = graph_from_atoms(atoms, z_table, R_MAX, with_labels=False, dtype=torch.float64)
    graph.energy = torch.zeros(1, dtype=torch.float64)
    graph.forces = torch.zeros(len(atoms), 3, dtype=torch.float64)
    batches = [collate_graphs([graph]) for _ in range(4)]

    from sagui.train.loss import EnergyForcesLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Poison exactly one backward pass, as a float32 overflow would.
    calls = {"n": 0}
    real_clip = trainer_module.clip_gradients

    def flaky(module, max_norm):
        calls["n"] += 1
        if calls["n"] == 2:
            return torch.tensor(float("inf"))
        return real_clip(module, max_norm)

    monkeypatch.setattr(trainer_module, "clip_gradients", flaky)
    _train_epoch(
        model, batches, optimizer, EnergyForcesLoss(1.0, 1.0),
        torch.device("cpu"), None, 10.0,
    )
    assert all(torch.isfinite(p).all() for p in model.parameters()), (
        "a single non-finite gradient must not corrupt the weights"
    )
