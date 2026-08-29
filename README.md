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
pip install -e ".[dev]"          # development
pip install -e ".[fast]"         # optional: vesin, for fast neighbour lists
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

## Angular resolution

`strictly_local` couples its edge tensor against an **aggregated equivariant
environment tensor** (Allegro's environment embedding), not against the edge's own
spherical harmonic. This is not a tuning choice. Coupling against `Y(r̂_ij)` alone
keeps `V_ij` proportional to `Y(r̂_ij)` at every depth, so every rotation invariant
read out of it is a constant and the atomic energy collapses to a function of the
*distances* `{(Z_k, r_ik)}` — the model becomes exactly blind to bond angles.

Two config keys control it, both on by default:

| key | effect |
| --- | --- |
| `model.environment_tensor` | couple against `Ŷ_i = Σ_k u(r_ik) g(x_ik) Y(r̂_ik)` instead of `Y(r̂_ij)` |
| `model.refresh_environment` | rebuild the invariant descriptor `e_i` at every layer |

Both sums run over `N(i)` only, so the receptive field stays at exactly one cutoff
— `test_environment_tensor_leaves_the_receptive_field_alone` pins that. Set either
to `false` only to reproduce the pre-fix architecture for comparison.

> A symmetry test suite cannot catch this failure: rotational invariance is exactly
> what an angle-blind model has in abundance. It must be tested for directly, which
> `test_atomic_energy_resolves_bond_angles` does.

On a reference potential whose force variance is 41 % three-body, the fix is worth
**9.3× on energy MAE and 2.6× on force MAE** (mean over 3 seeds), for 1.13× the
runtime and 1.17× the parameters.

## Equivariant normalisation (opt-in, and it did not help here)

`model.layer_norm` enables `EquivariantRMSNorm`: each degree is rescaled by a
rotation-invariant factor, taken **across channels**, after every layer. It is
implemented, tested and exactly equivariant — and **off by default, because it was
measured to hurt.**

| learning rate | F-MAE, `layer_norm: false` | F-MAE, `layer_norm: true` |
| --- | --- | --- |
| 3e-3 | 154.5 | 214.4 |
| 8e-3 | **84.2** | 252.9 |
| 2e-2 | 198.6 | **136.8** |

Best-against-best, after re-tuning the learning rate for each setting, it is
84 → 137 meV/Å *worse* on that small clean target.

**On real data it flips sign.** 800 MPtrj frames, 81 elements, 5 epochs, lr 2e-3:

| configuration | E-MAE (meV/atom) | F-MAE (meV/Å) |
| --- | --- | --- |
| baseline | 1019.1 | 56.02 |
| **`layer_norm: true`** | **831.4** | **45.18** |
| `huber_delta: 0.01` | 2105.6 | **36.28** |
| `huber_delta: 0.5` | 1665.5 | 42.12 |

So `layer_norm` is worth **−18 % energy and −19 % force MAE** on a large, chemically
diverse set and a clear loss on a small clean one — turn it on for the former. (The
absolute errors are large because five epochs on 640 frames of 81-element data is
badly undertrained; only the relative direction means anything.)

`huber_delta` trades energy for forces here: forces improve 35 %, energies double.
One `delta` is shared by terms whose residuals differ by orders of magnitude;
per-term deltas would be the fix, and are not implemented.

> Normalising **across channels** rather than per channel is a correctness
> requirement: `invariant_features` reads the per-channel mean square straight back
> out, so a per-channel norm would pin every `l>0` invariant to the gain and destroy
> that half of the scalar track. `test_equivariant_rms_norm_preserves_the_higher_degree_invariants`
> guards it.

## Physical constraints and richer invariants

Two `model` keys, both off by default, both cheap:

| key | what it does |
| --- | --- |
| `zbl_cutoff` | add a parameter-free ZBL nuclear repulsion below this radius (Å) |
| `cross_degree_invariants` | add the `(1,1)→2` cross-degree scalar to the read-out |

**`zbl_cutoff`** fixes a specific failure. Training sets are built from
configurations a sampler actually visits, so they contain almost no close contacts,
and the repulsive wall the network learns there is pure extrapolation — often
*attractive*. With the core on, the short-range force is repulsive by construction:

```
   r (Å)     F_x off      F_x on
     0.4      0.1116    2062.2439
     1.0     -0.0825      85.5013     <- the untrained network pulls atoms together
     2.2     -0.0279      -0.0279     <- beyond the cutoff, bit-identical
```

It carries no parameters — it is a constraint, not another thing to fit — and it
switches off through the same `C²` envelope used everywhere else, so the energy is
still twice differentiable (verified: `E`, `E′` and `E″` all reach zero together).
Around 1.5–2.0 Å is the useful range.

**`cross_degree_invariants`** widens what the scalar track can see.
`invariant_features` otherwise returns only per-degree squared norms, which are
blind to how the `l=1` and `l=2` blocks sit relative to each other:

```
rotate ONLY the l=2 block:
  per-degree norms  differ by 1.3e-15   <- blind
  cross-degree term differs by 1.6e+01  <- sees it
```

Needs `lmax >= 2`; costs one extra channel-width per node.

## Stress, virials and the training objective

The stress comes from the same autograd pass as the forces: a symmetric strain is
applied to the positions *and* the cell, the edge shifts are rebuilt from the
strained cell, and `σ = V⁻¹ ∂E/∂ε` falls out of the backward. It agrees with finite
differences to 1e-12, is exactly symmetric, and transforms as `σ → QσQᵀ`.

```python
out = model(batch, compute_forces=True, compute_stress=True)   # out["stress"]: [G, 3, 3]

calc = SaguiCalculator(model="best.model")
atoms.calc = calc
atoms.get_stress()          # ASE Voigt order; only then is the extra backward paid for
```

Three training-objective options, all off by default:

| key | effect |
| --- | --- |
| `training.stress_weight` | add a stress term (needs stress labels; `stress_key` configurable) |
| `training.huber_delta` | Huber residual instead of mean square — **see the warning below** |
| `training.force_weight_switch` | fraction of epochs after which the energy/force weights swap and the LR drops 10× (the MACE-MP schedule) |

Ablated on 32 EMT-labelled Cu cells (one seed, 60 epochs — directions, not digits):

| configuration | E-MAE | F-MAE | S-MAE (meV/Å³) |
| --- | --- | --- | --- |
| MSE, no stress | 44.5 | 66.4 | — |
| MSE, `stress_weight=10` | **41.6** | 81.2 | 10.7 |
| `huber_delta=0.01`, no stress | 69.0 | 114.6 | — |
| `huber_delta=0.01`, `stress_weight=10` | 68.5 | 158.4 | **9.0** |

> ⚠️ **`huber_delta` is an absolute residual threshold and must match your data.**
> The MACE-MP value of `0.01` is tuned for MPtrj-scale per-atom energies; on a small
> clean dataset nearly every residual already exceeds it, the loss degenerates to L1
> with a small gradient, and accuracy drops sharply — as above. Default is `None`
> (mean square). Tune it against your own residual scale.

## Inference speed

Two settings matter for molecular dynamics, both measured on a 216-atom Si cell
(6 048 edges, `lmax=2`, `C=32`, 2 layers, CPU float32, energy **and** forces):

| configuration | neighbour list | model | step | speed-up |
| --- | --- | --- | --- | --- |
| ASE list, `tensor_product: loop`, eager | 5.15 ms | 233.7 ms | 238.9 ms | 1.00× |
| + `vesin` neighbour list | 0.22 ms | 233.7 ms | 233.9 ms | 1.02× |
| + `tensor_product: gemm` *(the default)* | 0.22 ms | 214.5 ms | 214.7 ms | 1.11× |
| + `model.compile_layers()` | 0.22 ms | 105.2 ms | 105.5 ms | **2.27×** |

- **`model.tensor_product`** selects how the Clebsch–Gordan product is scheduled.
  `gemm` (the default) evaluates every coupling path in one fused matrix product;
  `loop` is the reference implementation, one `einsum` per path, kept because it
  uses less peak memory and because the fused kernel is tested against it. The two
  compute the same function and share the same parameters, so checkpoints are
  portable between them.
- **`compile_layers()`** hands the layers to `torch.compile`. It is opt-in: warm-up
  costs tens of seconds, and the compiled *backward* has been observed to fail on
  large systems, so benchmark it at your own system size before relying on it.

```python
calc = SaguiCalculator(model="best.model", compile_layers=True)
```

```bash
sagui-inference --model best.model --input material.xyz --compile
```

Installing `vesin` (`pip install -e ".[fast]"`) is transparent: the neighbour list
is 36–67× faster and returns the same edges, and SAGUI falls back to ASE when it
is absent or when the cell is only partly periodic.

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

## License

MIT — see `LICENSE`.
