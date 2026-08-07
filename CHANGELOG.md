# Changelog

Notable changes to dc1d. Newest first.

The release workflow reads the section matching a `v*` tag out of this file and
uses it as the GitHub Release body, so the heading format is load-bearing:
`## [x.y.z] - YYYY-MM-DD`, one blank line, then the section. A version with no
section here falls back to the annotated tag's message.

0.0.7 and 0.1.0 were never tagged; the previous tag is `v0.0.6`. PyPI carries
0.0.2, 0.0.4, 0.0.7 (yanked) and 0.2.0.

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-08-07

The first tagged release since v0.0.6: 55 commits, +15,120 lines. `0.1.0` was
bumped in-repo but never tagged or published, so everything from that line is
included here.

At v0.0.6 this package had no tests, no declared dependencies (`pip install
dc1d` produced an `ImportError` on first use), and imported private
`torchvision` symbols at module scope. It now has 403 tests, a locked `uv`
environment, CI on Python 3.10-3.13, a measured comparison against three other
backends, and no torchvision dependency at all.

### Upgrading from 0.0.x: read this first

- **`PackedDeformConv1d` checkpoints do not transfer when `dilation != 1` or
  `stride != 1`, and the failure is silent.** `offset_dconv.weight` has the same
  shape in both versions, so a 0.0.x `state_dict` loads with `strict=True` and no
  warning, then predicts entirely different offsets. **The DTCN paper's
  configuration is affected** (dilations 1 to 128): same weights, same input, the
  separated waveforms from 0.0.7 and 0.2.0 agree to only **4.84 dB SI-SDR**.
  Reproducing the published model requires pinning `dc1d==0.0.4` or lower. It is
  simultaneously a bug fix: the old wiring dropped stride and dilation from the
  offset convolution, and with `stride > 1` returned an output of the wrong
  length. Full measurement in `EQUIVALENCE.md` R1.
- **Not bit-exact with 0.0.7** except where sampling positions land on exact
  integers, where `torch.equal` is True. The rewrite changed the position
  arithmetic. Against an implementation-independent analytic reference, 0.2.0 is
  closer at every length tested: 33x at `L=256`, 131,000x at `L=2**20`.
- A caller-supplied non-integer `dilated_positions` is now rounded rather than
  interpolated. Only reachable by calling `efficient_linterpolate` directly.

### Removed

- `deform_conv1d`: dead, broken, and the only importer of torchvision.
- `_jit_efficient_linterpolate`: never called, and its `@torch.jit.script`
  decorator compiled eagerly at import.
- The multiprocessing path, which could not participate in autograd.
- **`self.device`** on both layers, and the `forward` mutation of it. Device is
  inferred from the input. Use `next(layer.parameters()).device`.

### Changed

- **Python `>=3.10`** (was `>=3.8`) and **torch `>=2.7`** (was undeclared, then
  `>=2.4`). 2.7 is the oldest release the ONNX export path is verified against.
- Benchmarks use `torch.utils.benchmark.Timer`. Earlier timings had no
  `torch.cuda.synchronize` around `time.time`, so on GPU they measured Python
  dispatch, and gave the deformable path three warmup iterations against a
  vanilla conv's one cold call. Do not compare against numbers printed by 0.0.x.
- `dilated_positions` is a non-persistent buffer, so it appears in `state_dict`
  and follows `.cuda()` instead of being patched by hand in `forward`.
- `__init__.py` exports something: `DeformConv1d`, `PackedDeformConv1d`, `cLN`,
  `gLN`, the three interpolation kernels, and `__version__`.
- User-facing `assert`s are raised exceptions; `== None` comparisons and
  numpy-compat `axis=` kwargs fixed; type annotations and docstrings corrected.

### Fixed

Seven bugs, all of which failed silently rather than raising:

| bug | symptom before |
|---|---|
| `linspace` inherited the offsets' dtype | fp16 cannot represent integers past 2048; 13,952 of 15,998 rows wrong at `L=16000`, no NaN |
| stride and dilation not forwarded to the offset conv | wrong number of offset positions, absorbed by the index clamp |
| `repeat` instead of `repeat_interleave` for offset groups | `1 < offset_groups < in_channels` raised, though documented as supported |
| unconstrained positions clamped to `L`, not `L-1` | at `T == L` both interpolation weights collapsed and output was exactly 0.0 |
| `from turtle import forward` | `import dc1d.nn` failed on any Python without tkinter (slim containers, CI images) |
| `extra_repr` copy-pasted from `_ConvNd` | `print(model)` raised `TypeError` |
| `2^7` is XOR, not a power | the "large dilation" example had been testing `dilation=5` |

Two more found while adding export support:

- **Exported ONNX models were wrong at any length other than the export
  length.** `take_along_dim` decomposed to a negative-index wrap, `index %
  self.size(dim)`, and from torch 2.10 the exporter constant-folded that modulus
  against the export-time length. A model exported at `L=200` got a literal
  `Mod(index, 200)`, so longer inputs wrapped around and read the wrong samples,
  at full signal magnitude, with the correct output shape and no error raised.
  Bisected: clean on torch 2.7-2.9, broken on 2.10-2.13.

  | T | before | after |
  |---|---|---|
  | 120 | 1.8e-07 | 1.8e-07 |
  | 200 | 1.2e-07 | 1.2e-07 |
  | 300 | **3.4e+00** | 1.8e-07 |
  | 1600 | **3.8e+00** | 1.8e-07 |

  Shorter lengths pass too, because their indices never reach the modulus, which
  is why a single-length test protects nothing. `tests/test_export.py` sweeps
  both sides of the export length and separately asserts the graph contains no
  constant-divisor `Mod`.
- **`gLN` and `cLN` returned NaN under onnxruntime for a zero-variance input.**
  The exporter folds away `Add(var, 1e-9)`, leaving `Div(x, Pow(var, 0.5))`. Now
  written with `clamp_min`, which survives export. Reproduces on all seven torch
  versions tested, so not a regression.

### Added

- **Modulation (DCNv2, Zhu et al. 2019).** `DeformConv1d.forward` no longer
  raises `NotImplementedError` on `mask`. It is applied verbatim to the sampled
  positions, matching the `torchvision.ops.deform_conv2d` contract, so apply the
  sigmoid yourself. `PackedDeformConv1d(modulated=True)` predicts the mask from a
  second pointwise branch off the shared depthwise trunk. Default off, so DCNv1
  behaviour and parameter counts are unchanged. The mask head architecture is a
  judgement call, not from the paper.
- **ONNX export** through the dynamo exporter, with no custom operators and no
  baked length. Export at one length, run at any other. New `export` dependency
  group and a dedicated CI job for it.
- **`gather_lerp='recompute'`**, an opt-in hand-written backward that saves only
  the index and the fraction and re-gathers, passed through
  `interpolation_function=functools.partial(...)`.
- **`expected_offset_positions(length)`**, the closed form for the offset
  tensor's size, and a `forward` check that raises when the offsets disagree with
  it. Previously the index clamp absorbed the mistake and returned plausible
  wrong-length output.
- **403 tests**, from zero at v0.0.6, passing on torch 2.7.1 and 2.13.0. The 72
  skips are all `padding='same'` with `stride > 1`, which is not a defined
  combination. `test_equivalence.py` pins the `nn.Conv1d` bit-exactness invariant
  across stride, dilation and groups; `test_gradients.py` runs `gradcheck` in
  float64 against both `input` and `offsets`.
- CI on Python 3.10-3.13 with `ruff` and `pytest`, plus a separate export job and
  a build job that installs the wheel and imports it.
- `docs/demo.ipynb`, seven plotted demonstrations committed with outputs, every
  exact property asserted in the cell that shows it. It also records the sampling
  direction, stated nowhere else: the offset is added to the read position, so
  `out[i] = x[i+n]` and a positive offset moves the trace left.
- `benchmarks/BACKENDS.md`, `EQUIVALENCE.md`, `CLAUDE.md`, `TODO.md`.

### Performance

- **Interpolation kernel rewritten.** The old one produced a 43 MB result using
  roughly 800 MB to 1 GB of memory traffic, dominated by a 174 MB int64 index
  tensor that existed only because `torch.gather` does not broadcast. Since `U1 =
  U0 + 1` by construction, the two weights are exactly `1-f` and `f`, so the
  whole abs/max/multiply/sum chain is a `lerp`. CPU, `B=1 C=512 L=16000 K=3`:
  forward **1696 to 502 ms**, forward+backward **3309 to 760 ms**, peak RSS **283
  to 91 MiB**.
- **`torch.compile` works with `fullgraph=True`**: 0 graph breaks under
  `torch._dynamo.explain`, worth **2.3x to 9.1x** forward on an RTX 3090. Each
  distinct sequence length is a separate compilation (~0.3 s), which matters for
  variable-length audio.
- **The gather rewrite cut backward memory as a side effect.** `gather` on an
  expanded index saves the stride-0 view on the tape where `take_along_dim` saved
  a materialised copy: 0.094 MiB against 5.988 MiB at the same configuration. The
  default backward now holds 3.0x-3.4x the output rather than 7.02x, **only when
  `offset_groups < channels`**. Consequently `recompute` is now worth 1.30x
  rather than the 1.6x-2.1x documented in `BACKENDS.md` 5.9.4, except at
  `offset_groups == channels`, the depthwise case the DTCN paper uses, where it
  keeps full value. New section 5.9.4a.

### Measured against alternatives

`benchmarks/BACKENDS.md` compares dc1d with `torchvision.ops.deform_conv2d`, an
`F.grid_sample` backend and tinymera's implementation, on an RTX 3090:

- **`grid_sample` beats dc1d in 13 of 13 configurations on both axes**: 1.6x-5.3x
  faster forward, 1.4x-2.6x faster fwd+bwd, 2.3x-5.2x less fwd+bwd memory. It is
  an ATen builtin, so adopting it would preserve the no-compilation property, and
  its offset gradient matches dc1d's to 7.1e-15.
- torchvision's compiled kernel is erratic rather than decisively better: fastest
  in only 2 of 13 forward configurations, and 4x-10x **slower** than pure-PyTorch
  dc1d in six depthwise ones, which is the regime this package exists for.

### Known limits

- **0.0.6 was never published to PyPI**, so the paper pin is `dc1d==0.0.4`, the
  newest release predating the offset-convolution rewiring. 0.0.7 is yanked and
  installs only when pinned exactly.
- No version-guarded shim or `load_state_dict` hook warns about the checkpoint
  break above. It is documented only.
- `BACKENDS.md` section 5.9.4a came off a contended A100 rather than the idle
  3090 the rest of the file uses. Its memory figures reproduce section 5.9.4 to
  the decimal, which is what licenses the comparison, but treat its latency
  columns as indicative.
- CUDA numbers for `modulated=True` were not measured at all, and
  `benchmarks/benchmark.py` (old versus new) has only ever run on CPU.
- torch 2.4 to 2.6 are untested rather than known-broken. Nothing in the layer
  needs a 2.7 API, so they will probably import and run.
- `torch.compile` with `mode="max-autotune"` measured worse than
  `mode="default"` in 8 of 10 configurations and is not recommended.
- Dynamic shapes under `torch.compile` are unavailable: the graph specialises on
  `L`, so `mark_dynamic` raises and `dynamic=True` silently gives one graph per
  length. ONNX export does not have this problem.
- `padding_mode` defaults to `'reflect'`, unlike `nn.Conv1d`'s `'zeros'`, and
  reflect padding requires `pad < L`, which a deep TCN can violate.

## [0.1.0] - unreleased

Bumped in-repo, never tagged and never published. Folded into 0.2.0 above.

## [0.0.7] - 2022

README fixes and bug fixes on top of 0.0.6. Yanked on PyPI, so it installs only
when pinned exactly, which is why the paper pin above is 0.0.4 rather than 0.0.7.

## [0.0.6] - 2022

**The version the ICASSP 2023 DTCN paper was built against.** It was tagged but
never published, so pin `dc1d==0.0.4` or lower to reproduce it: 0.0.4 is the
newest release on PyPI from before the offset-convolution rewiring. See the
checkpoint note under 0.2.0 for why the pin matters.

[Unreleased]: https://github.com/jwr1995/dc1d/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jwr1995/dc1d/compare/v0.0.6...v0.2.0
