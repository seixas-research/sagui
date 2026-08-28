# SAGUI

**S**calable **A**tomistic **G**raph networks for **U**niversal **I**nteractions —
an equivariant graph neural network framework for machine-learned interatomic
potentials (MLIPs) and for generative models of crystal structures.

SAGUI trains two kinds of model from the same atomic structures:

| task | what it learns | entry point |
|---|---|---|
| `potential` | energies and forces, `E(R)` and `F = -∇E` | `sagui-train`, `sagui-inference` |
| `generative` | the distribution of structures itself | `sagui-train --task generative`, `sagui-generate` |

Two interchangeable architectures are provided, selected by a single
configuration key:

- **`mace`** — message passing with a many-body (Atomic Cluster Expansion)
  basis, in the spirit of [MACE](https://arxiv.org/abs/2206.07697) and
  [NequIP](https://www.nature.com/articles/s41467-022-29939-5).
- **`strictly_local`** — no message passing at all: every quantity stays inside
  one cutoff sphere however deep the network is, following
  [Allegro](https://www.nature.com/articles/s41467-023-36329-y).

Everything is built on PyTorch, ASE and NumPy. The O(3) tensor algebra —
spherical harmonics, Wigner-D matrices, Clebsch–Gordan coefficients — is
implemented from scratch in `sagui.nn.o3`, so there is **no `e3nn`
dependency**.

## Installation

```bash
conda activate sagui
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Installing provides three commands: `sagui-train`,
`sagui-inference` and `sagui-generate`.

## Quick start — an interatomic potential

```bash
# 1. make a toy dataset (Lennard-Jones-labelled argon clusters)
python examples/make_toy_dataset.py --output examples/toy.xyz --frames 200

# 2. train
sagui-train examples/config_mace.yaml
#    ... or without a config file:
sagui-train --train-file examples/toy.xyz --model-type strictly_local --epochs 200

# 3. predict, and score against the reference labels if the file has them
sagui-inference --model runs/toy-mace/best.model --input examples/toy.xyz --evaluate \
                --output predictions.xyz
```

Any format ASE can read works (`.xyz`, `.extxyz`, `.traj`, …); energies and
forces are picked up from `atoms.info` / `atoms.arrays` or from an attached
calculator, and the key names can be set with `data.energy_key` /
`data.forces_key`.

From Python, a trained model is an ordinary ASE calculator:

```python
from ase.io import read
from ase.optimize import BFGS
from sagui import SaguiCalculator

atoms = read("structure.xyz")
atoms.calc = SaguiCalculator("runs/toy-mace/best.model")
BFGS(atoms).run(fmax=0.01)
```

## Quick start — a generative model

```bash
# periodic, binary training data (the generative model diffuses a lattice)
python examples/make_toy_crystals.py --output examples/crystals.xyz --frames 400

sagui-train examples/config_generative.yaml
sagui-generate --model runs/toy-d3pm/best.model -n 16 -o generated.xyz
```

The generative model diffuses the three parts of a crystal jointly:

- **atom types** with **D3PM** ([Austin et al. 2021](https://arxiv.org/abs/2107.03006)),
  using either an absorbing (`[MASK]`) or a uniform transition kernel;
- **fractional coordinates** with a *wrapped* normal on the unit torus;
- **the lattice** with a standard DDPM, in units of the mean interatomic distance.

## Devices

`--device auto` (the default) picks **CUDA, then Apple MPS, then CPU**. All
three run the full pipeline: forward, force training (which needs a second
derivative through the whole network), checkpointing, the ASE calculator, and
diffusion sampling.

```bash
sagui-train  config.yaml --device auto     # cuda > mps > cpu
sagui-train  config.yaml --device mps      # force a specific backend
sagui-inference --model best.model --input x.xyz --device cpu
```

One caveat is worth knowing: **Metal has no `float64` at all.** Asking for
double precision on MPS logs a warning and falls back to `float32` rather than
failing inside a forward pass. Use `--device cpu` when the precision is what
matters — finite-difference checks, tight relaxations, or comparing energies at
the micro-eV level.

MPS is not automatically *faster*: for the model sizes here the per-kernel
launch overhead roughly cancels the parallelism, and CPU and MPS come out
within ~10 % of each other. Measure before assuming.

Verified on MPtrj (500 real DFT frames, 81 elements) by training each
architecture on each backend and then running one CPU-trained checkpoint on
both — the two devices agree to float32 round-off:

| architecture | max abs energy difference | max abs force difference |
|---|---|---|
| `mace` | 3.8e-06 eV (1.1e-07 relative) | 1.9e-08 eV/Å |
| `strictly_local` | 0 eV | 8.6e-08 eV/Å |

## Training on MPtrj

`examples/mptrj_to_xyz.py` streams the Materials Project trajectory dataset
(~1.6 M DFT frames, shipped as one ~11 GB JSON object that cannot be loaded
into memory) and extracts a portion as extended XYZ:

```bash
python examples/mptrj_to_xyz.py \
    --input /path/to/MPtrj_2022.9_full.json \
    --output mptrj_2k.xyz --limit 2000 --max-atoms 40

sagui-train --train-file mptrj_2k.xyz --model-type mace --r-max 5.0
```

It keeps a bounded buffer and stops as soon as it has enough frames, so
reading a few thousand takes seconds. `--elements` restricts to a chemical
subsystem, `--frames-per-material` trades correlated relaxation frames for
diversity.

## Configuration

One YAML file describes a run; every key can be overridden from the command
line, either through a named flag or with `--set section.key=value`:

```bash
sagui-train config.yaml --set model.channels=64 --set training.epochs=500
```

```yaml
task: potential          # or: generative

model:
  type: mace             # or: strictly_local
  r_max: 5.0             # cutoff radius in Angstrom
  lmax: 2                # highest spherical-harmonic degree in the features
  channels: 32
  num_layers: 2
  correlation: 3         # mace only: many-body order within each layer

data:
  train_file: data/train.xyz
  valid_fraction: 0.1

training:
  epochs: 300
  batch_size: 8
  learning_rate: 0.01
  forces_weight: 100.0
  ema_decay: 0.99
```

See `examples/config_mace.yaml`, `examples/config_strictly_local.yaml` and
`examples/config_generative.yaml` for annotated, runnable configurations.

## Architecture at a glance

```
             ┌──────────────────── sagui.nn ───────────────────┐
             │  o3.py      real spherical harmonics, Wigner-D, │
             │             Clebsch-Gordan (derived, not hard-  │
             │             coded), Cartesian conversions       │
             │  blocks.py  equivariant linear, tensor products │
             │  radial.py  Bessel basis, smooth cutoff, MLPs   │
             └────────────────────────┬────────────────────────┘
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        │                             │                              │
  models/mace.py             models/strictly_local.py       generative/
  message passing +          pair features, receptive       D3PM types +
  many-body products         field = one cutoff             coords + lattice
        │                             │                              │
        └────────── models/base.py ───┘                    generative/denoiser.py
           E = Σ (s·ε_i + E0_Z),  F = -∂E/∂R               shares the same backbone
```

| module | contents |
|---|---|
| `sagui/nn/` | the O(3) algebra and equivariant layers |
| `sagui/data/` | ASE → graphs, neighbour lists, batching, dataset statistics |
| `sagui/models/` | `InteratomicPotential` base class, the two architectures, the registry |
| `sagui/generative/` | noise schedules, D3PM, the joint corruption, the denoiser, sampling |
| `sagui/train/` | losses, EMA, and the two training loops |
| `sagui/cli/` | the three console scripts |

Adding an architecture takes a decorated class — nothing else changes:

```python
from sagui.models import InteratomicPotential, register_model

@register_model("my_arch")
class MyPotential(InteratomicPotential):
    def node_energies(self, data, vectors, lengths):
        ...   # return a per-atom energy; forces come from autograd
```

It is then selectable as `model.type: my_arch`.

## Testing

```bash
python -m pytest            # ~120 tests, a few seconds
```

The suite checks the properties that actually matter for a potential:
rotation, reflection, translation and permutation equivariance to machine
precision; forces against central finite differences; size extensivity of a
supercell; the receptive field of each architecture; and the exactness of the
D3PM transition matrices and the wrapped-normal score.

## Status

Early but functional: both architectures train and converge, and the
generative model trains and samples. See `sagui_context.md` (not tracked in
git) for the design rationale and the roadmap.

## License

MIT — see `LICENSE`.
