# dc1d 0.0.7 versus 0.1.0: equivalence evidence

What this is: a measured comparison of the interpolation kernel published on PyPI as
dc1d 0.0.7 (repository commit `6ca2e01`) against dc1d 0.1.0 (`origin/main`, after
PRs #9 and #10), using the ICASSP 2023 DTCN speech separation model as the downstream
test vehicle.

Everything below was produced by `verification/equivalence_check.py`,
`verification/dtcn_forward.py` and `verification/dtcn_compare.py` in this branch.
Raw output: `verification/results/`.

## Headline

Three things, in order of how much they should change what you do.

1. **dc1d 0.1.0 is not checkpoint-compatible with 0.0.x for `PackedDeformConv1d` when
   `dilation != 1` or `stride != 1`, and it fails silently.** The `offset_dconv`
   weight has the same shape in both versions, so a 0.0.x checkpoint loads with
   `strict=True` and no warning, and then predicts entirely different offsets.
   **The DTCN paper's configuration is affected**: it uses dilations 1, 2, 4, ..., 128.
   Loading the same weights and feeding the same input, the separated waveforms from
   0.0.7 and 0.1.0 agree to only **4.84 dB SI-SDR**, which is to say they are different
   signals. This is a real regression in the sense that reproducing the published model
   now requires pinning `dc1d<0.1`; it is simultaneously a real bug fix, because with
   `stride > 1` the old wiring returns an output of the wrong length. It should be a
   release note, and arguably a version-guarded shim.

2. **Bit-exactness between the two kernels does not hold in the regime the brief
   expected it to.** In float32 and float64, constrained mode, `offset_groups=1`,
   `stride=1`, in-range offsets and away from the boundary, `torch.equal` is **False**
   for the forward and for both gradients. The rewrite changed the position arithmetic
   itself, so this was never achievable except where the sampling positions land on
   exact integers, where `torch.equal` **is** True. What replaces the bit-exactness
   claim is a measurement: 0.1.0 is closer to an implementation-independent analytic
   reference than 0.0.7 at every sequence length tested, by 33x at `L=256` and by
   131000x at `L=2**20`.

3. **The one place 0.1.0 removed a capability 0.0.7 had**, other than the checkpoint
   break: a caller-supplied non-integer `dilated_positions` is now silently rounded to
   the nearest integer instead of being interpolated. Neither layer class can trigger
   this, only a direct call to `efficient_linterpolate`.

Everything else the brief predicted was confirmed, and 0.1.0 is the correct side of
each divergence.

## Environment

Every claim below is under this environment unless stated otherwise.

| | |
|---|---|
| torch | `2.13.0+cu130`, CUDA 13.0 |
| GPU | NVIDIA GeForce RTX 3090 (`cuda:0`) |
| CPU rows | same host, x86-64 |
| TF32 | **disabled** for both cuDNN and matmul, so every float32 claim is a true float32 claim |
| Python | 3.11 (main harness), 3.14.4 (the tkinter test only) |
| torchvision | `0.28.0+cu130`, installed only because 0.0.7 cannot be imported without it |
| speechbrain | 1.0.3, torchaudio 2.11.0 (DTCN only) |
| seed | 20260728 for the kernel harness, 1234 (the DTCN yaml's own seed) for model construction |

`OLD` is `/home/will/scratch-dc1d/old`, a worktree at `6ca2e01`. Its `dc1d/ops.py` is
byte-identical to the `dc1d-0.0.7-py3-none-any.whl` published on PyPI on 2023-08-17,
and its `dc1d/nn.py` differs from that wheel by two blank lines, so `OLD` is the
published kernel. `NEW` is `origin/main` at `670a631`, version 0.1.0.

`dc1d-0.0.2-py3-none-any.whl` (2022-10-26), which is what both
[DTCN](https://github.com/jwr1995/DTCN) and its successor
[PubSep](https://github.com/jwr1995/PubSep) pin in `requirements.txt`, was also
downloaded and inspected: it has the same `offset_dconv` construction as 0.0.7, so
finding 1 applies to the version the paper actually ran.

## The vehicle and its configuration

DTCN's `separation/src/deformable.py` line 16 is `from dc1d.nn import
PackedDeformConv1d`. The only instantiation is in
`DeformableDepthwiseSeparableConv.__init__`:

```python
self.depthwise_conv = PackedDeformConv1d(
    in_channels=in_channels,
    out_channels=in_channels,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    padding=padding,
    groups=in_channels,
    bias=bias,
)
```

Resolved against `hparams/deformable/dtcn-whamr.yaml`:

| argument | value | source |
|---|---|---|
| `in_channels`, `out_channels` | 512 | `H` |
| `kernel_size` | 3 | `P` |
| `stride` | 1 | hardcoded in `DeformableTemporalBlocksSequential` |
| `dilation` | 1, 2, 4, ..., 128 | `2 ** x` for `x` in `range(X)`, `X = 8` |
| `padding` | `"same"` | hardcoded |
| `groups` | 512 | depthwise |
| `bias` | True | |
| `offset_groups` | 1 | not passed, dc1d default |
| `padding_mode` | `"reflect"` | not passed, dc1d default |
| `unconstrained` | `None`, treated as False (constrained) | not passed, dc1d default |
| `interpolation_function` | `efficient_linterpolate` | not passed, dc1d default |

`hparams/deformable/dtcn-whamr.yaml` gives `X=8`, `R=3`, so there are 24 such layers,
three repeats of dilations 1 through 128. The encoder is `kernel_size=16`,
`stride=8`, `out_channels=512`; the masknet is `N=512, B=128, H=512, P=3, C=2`,
`norm_type='gLN'`, `causal=False`, `mask_nonlinear='relu'`; `sample_rate=8000` and
`training_signal_len=32000`, which is a 4 s utterance and a masknet sequence length of
3999.

**This is exactly the configuration that finding 1 breaks.**

## A. Where behaviour was meant to be unchanged

### A1. The zero-offset `nn.Conv1d` invariant: holds for both, bit for bit

With zero offsets in constrained mode, both versions reproduce `nn.Conv1d` given the
same weights. `torch.equal`, 20 configurations (2 devices x 2 dtypes x 5 shapes):

| device | dtype | k, d, stride, groups | OLD == `nn.Conv1d` | NEW == `nn.Conv1d` | OLD == NEW |
|---|---|---|---|---|---|
| cpu, cuda:0 | float64, float32 | (3,1,1,1) (3,4,1,1) (5,2,3,1) (3,2,1,4) (3,8,1,1) | True (20/20) | True (20/20) | True (20/20) |

All 20 rows are `True` in all three columns. Note this needs TF32 off, which the harness
sets explicitly: measured separately with the PyTorch default
(`torch.backends.cudnn.allow_tf32 = True`), the *reference* `nn.Conv1d` drifts from both
dc1d versions identically, by up to 9.06e-04 in float32 on CUDA at `k=3, d=4`. That is
cuDNN's algorithm choice, not a dc1d difference, but it would break a naive rerun of
this table.

### A2. Forward with non-trivial offsets: not bit-identical, and 0.1.0 is the accurate one

Real weights, random offsets uniform on `[0, d*(k-1)]`, constrained, `offset_groups=1`,
`stride=1`, `k=3`, `d=4`, `cuda:0`. `torch.equal` is False in every row.

| dtype | L | `torch.equal` | max abs OLD - NEW | err(OLD) vs float64 | err(NEW) vs float64 |
|---|---|---|---|---|---|
| float64 | 64 | False | 1.07e-14 | | |
| float64 | 256 | False | 4.64e-14 | | |
| float64 | 4096 | False | 9.96e-13 | | |
| float64 | 65536 | False | 1.86e-11 | | |
| float64 | 1048576 | False | 3.39e-10 | | |
| float32 | 64 | False | 4.77e-06 | 4.81e-06 | 7.31e-07 |
| float32 | 256 | False | 2.56e-05 | 2.60e-05 | 9.25e-07 |
| float32 | 4096 | False | 5.86e-04 | 5.86e-04 | 1.28e-06 |
| float32 | 65536 | False | 1.08e-02 | 1.08e-02 | 1.43e-06 |
| float32 | 1048576 | False | 1.82e-01 | 1.82e-01 | 1.61e-06 |

The float32 reference is the same computation in float64; the two float64 evaluations
of it differ by at most 8.88e-16, so the reference is well defined at this precision.
OLD's error grows linearly in `L`; NEW's is flat and at the float32 rounding floor.

**A2b, the same question against a reference that depends on neither implementation.**
Feed `x[i] = i mod 2`. The two gathered samples are then exactly 0 and 1, so linear
interpolation at position `index + f` returns exactly `f` (even `index`) or `1 - f`
(odd). The output is O(1) while the positions are large, which isolates the error in
the sub-sample fraction; positions are computed in exact integer arithmetic outside
both kernels. Offsets are drawn so that no tap sits on the clamp.

| dtype | L | max err OLD | max err NEW | eps |
|---|---|---|---|---|
| float64 | 256 | 1.42e-14 | **0.0** | 2.22e-16 |
| float64 | 4096 | 2.27e-13 | **0.0** | 2.22e-16 |
| float64 | 65536 | 3.64e-12 | **0.0** | 2.22e-16 |
| float64 | 1048576 | 5.82e-11 | **0.0** | 2.22e-16 |
| float32 | 256 | 7.87e-06 | 2.38e-07 | 1.19e-07 |
| float32 | 4096 | 1.22e-04 | 2.38e-07 | 1.19e-07 |
| float32 | 65536 | 1.95e-03 | 2.38e-07 | 1.19e-07 |
| float32 | 1048576 | 3.13e-02 | 2.38e-07 | 1.19e-07 |

0.1.0 is exact in float64 and within 2 eps in float32, independent of `L`. 0.0.7's error
is proportional to `L`, which is the signature of computing the fraction at the
magnitude of the absolute position rather than relative to the window start.

### A3. Where the two kernels *are* bit-identical

`k=3`, `d=2`, `L=64`, float32, `cuda:0`:

| offsets | `torch.equal` | max abs difference |
|---|---|---|
| zero | **True** | 0.0 |
| integer-valued | **True** | 0.0 |
| all 0.5 | False | 1.19e-07 |
| all 0.25 | False | 1.19e-07 |
| random in range | False | 4.77e-06 |

The pattern: the two agree bit for bit when every sampling position lands on an integer,
so that one interpolation weight is exactly 1 and the other exactly 0, and in no other
case tested. Where they differ it is for two independent reasons, both deliberate:

1. 0.0.7 forms `T = t0 + dilated + offset` in the offset dtype and derives the fraction
   from that; 0.1.0 keeps `t0 + dilated` in `long` and carries only the fraction in the
   offset dtype. Different fractions in the low bits, and increasingly so with `L`.
2. 0.0.7 contracts as `x0*(1-f) + x1*f`; 0.1.0 uses `torch.lerp(x0, x1, f)`. Same value
   in exact arithmetic, a different rounding in floating point. This one is visible even
   at `L`-independent scale, which is why the 0.5 and 0.25 rows above differ.

### A4. Gradients: not bit-identical either, and OLD's offset gradient is badly wrong at long `L`

Same configuration, float32, `cuda:0`, upstream gradient shared between the two runs.
Reference is the same backward in float64.

| L | gradient | `torch.equal` | max abs OLD - NEW | err(OLD) vs float64 | err(NEW) vs float64 | gradient scale |
|---|---|---|---|---|---|---|
| 256 | d/d input | False | 2.74e-05 | 2.74e-05 | 5.08e-07 | 5.08 |
| 256 | d/d offsets | False | 9.54e-07 | 4.11e-07 | 5.64e-07 | 9.00 |
| 4096 | d/d input | False | 7.48e-04 | 7.48e-04 | 7.15e-07 | 7.38 |
| 4096 | d/d offsets | False | **6.47** | **6.47** | 9.83e-07 | 13.18 |
| 65536 | d/d input | False | 1.21e-02 | 1.21e-02 | 8.34e-07 | 8.32 |
| 65536 | d/d offsets | False | **8.80** | **8.80** | 1.35e-06 | 14.58 |

The offset gradient failure in 0.0.7 is rare but total, not a small bias. Counting
elements whose error exceeds 1e-3:

| L | elements | OLD wrong | OLD max err | OLD rms err | NEW wrong |
|---|---|---|---|---|---|
| 256 | 744 | 0 | 4.57e-07 | 7.09e-08 | 0 |
| 4096 | 12264 | 1 | 6.99 | 6.31e-02 | 0 |
| 65536 | 196584 | 295 (0.15%) | 8.07 | 6.79e-02 | 0 |

### A5. Why: 0.0.7's offset gradient is structurally wrong wherever a position is an integer

Interpolation weights sum to 1, so `d(output)/d(offset)` must be unchanged if a constant
is added to the input. Test, float64, `cuda:0`, `k=3`, `d=1`, zero offsets, then
`x -> x + 1000`:

| | d/d offset at taps 0,1,2 | max abs change when `x -> x + 1000` |
|---|---|---|
| OLD | `[0.105268, -0.505334, 0.035734]` | **5.00e+02** |
| NEW | `[1.574364, -1.431742, 1.153603]` | 1.05e-13 |

The analytic right-derivative is `x[j+1] - x[j]` = `[1.574364, -1.431742, 1.153603]`,
which is what 0.1.0 returns exactly. 0.0.7 returns `0.5*x[j+1]`, halved again to
`0.25*x[j+1]` at the two taps that also sit on a constrained-clamp boundary:
`0.25*x[1], 0.5*x[2], 0.25*x[3]` = `[0.105268, -0.505334, 0.035734]`, matching to all
printed digits. The cause is three non-differentiable points that 0.0.7 lands on
exactly, in `G = torch.max(zeros, 1 - torch.abs(U - T))`:

- for the lower sample, `|U - T| = 0` and `torch.abs` has derivative 0 there, so the
  `x[j]` term contributes nothing rather than `-x[j]`;
- for the upper sample, `1 - |U + 1 - T| = 0` and `torch.max` ties with the zero
  tensor, splitting the gradient, so the `x[j+1]` term contributes `0.5*x[j+1]`;
- at a tap that also sits on the constrained clamp, `torch.max(T, t0s)` or
  `torch.min(T, t0s + max_tap)` ties as well, halving it once more.

A gradient that depends on the *value* of `x[j+1]` rather than on a difference of
neighbouring samples cannot be a subgradient of an interpolation, and the shift test
proves that it is not one.

This is also the mechanism behind A4: as `L` grows, float32 rounding makes `T` land
exactly on an integer more and more often, so this wrong branch is taken more often.

The brief listed a further divergence at the constrained clamp boundary. It is the same
phenomenon and not a separate one: pinning a tap to the upper bound gives
`OLD 0.1913` against `NEW -0.1395`, which is again `0.5*x[i+1]` against `x[i+1] - x[i]`.

## B. Deliberate divergences, and the evidence that 0.1.0 is right

### B1. fp16 positions: confirmed, and worse than "L > 2048"

float16 input and offsets, `k=3`, `d=1`, `cuda:0`. A row counts as wrong if any element
in it deviates from a float64 evaluation by more than 1e-3.

| L | out_len | OLD output dtype | NEW output dtype | OLD rows wrong | NEW rows wrong | OLD max err | NEW max err |
|---|---|---|---|---|---|---|---|
| 256 | 254 | float32 | float16 | 233 | 7 | 0.213 | 0.0020 |
| 1024 | 1022 | float32 | float16 | 991 | 38 | 0.867 | 0.0022 |
| 2048 | 2046 | float32 | float16 | 2012 | 61 | 1.859 | 0.0021 |
| 4096 | 4094 | float32 | float16 | 4046 | 114 | 5.098 | 0.0025 |
| 16000 | 15998 | float32 | float16 | **15958** | 487 | 7.083 | 0.0025 |

15958 of 15998 wrong at `L=16000` confirms the reported ~13952. Two corrections to the
framing, though: the damage starts immediately, not past 2048 (233 of 254 rows are
already wrong at `L=256`), because float16 cannot represent `t0 + fraction` to
sub-sample accuracy well before it loses integers; and the residual NEW rows are not
errors in the positions but the float16 rounding of the interpolated *value*, which is
why NEW's max error stays at 0.0025 regardless of `L` while OLD's grows to 7.08.

Second, undocumented divergence visible here: **0.0.7 silently returns float32 for a
float16 input**, because its `torch.max(torch.zeros(...), ...)` allocates a default
float32 tensor and promotes. 0.1.0 returns float16. Under autocast this changes what
the following `F.conv1d` sees.

### B2. Unconstrained mode at the right-hand edge: confirmed

`x = ones`, so the output is exactly the sum of the two interpolation weights. `L=16`,
`k=1`, float64. A correct kernel returns 1.0 for every in-range position.

| sampling position T | OLD weight sum | NEW weight sum |
|---|---|---|
| 0.0 | 1.000000 | 1.000000 |
| 8.0 | 1.000000 | 1.000000 |
| 14.0 | 1.000000 | 1.000000 |
| 14.5 | 1.000000 | 1.000000 |
| 15.0 | 1.000000 | 1.000000 |
| **15.5** | **0.500000** | 1.000000 |
| **16.0 (== L)** | **0.000000** | 1.000000 |

0.0.7 clamps `T` to `L` rather than `L-1`, so across the whole half-open band
`T` in `(L-1, L]` its two bilinear weights sum to `L - T` instead of 1, decaying
linearly to exactly 0.0 at `T == L`. The brief called out `T == L`; the defect covers
the last sample interval, not just its endpoint.

### B3. `PackedDeformConv1d` with `stride > 1` or `dilation > 1`: confirmed

0.0.7 builds `offset_dconv` as `nn.Conv1d(..., stride=1, <dilation omitted>)`, so it
emits the wrong number of offset positions, which the index clamp then absorbs.
Ground truth is `nn.Conv1d`'s own output length. `padding='valid'`, `L=64`, `k=3`,
`C=4`, depthwise.

| stride | dilation | `nn.Conv1d` length | OLD | NEW |
|---|---|---|---|---|
| 1 | 1 | 62 | 62 | 62 |
| 2 | 1 | 31 | **62 wrong** | 31 |
| 1 | 4 | 56 | **62 wrong** | 56 |
| 3 | 2 | 20 | **62 wrong** | 20 |
| 4 | 8 | 12 | **62 wrong** | 12 |

With `padding='same'` and `stride=1` the *length* happens to come out right for any
dilation, which is why DTCN never noticed; the offsets are still computed from a
dilation-1 receptive field. See finding 1 and section D.

### B4. `offset_groups` strictly between 1 and `in_channels`: confirmed

`in_channels=8`, `offset_groups=2`, `k=3`, `L=32`.

| | OLD | NEW |
|---|---|---|
| `PackedDeformConv1d(offset_groups=2)` | `AssertionError: offset_groups only implemented for offset_groups in {1,in_channels}` | ok, output `(1, 8, 30)` |
| `DeformConv1d` given 2-group offsets | `RuntimeError: Size does not match at dimension 1 expected index [1, 16, 30, 2] to be no larger than self [1, 8, 32, 2] apart from dimension 2` | ok, output `(1, 8, 30)` |

The brief said `RuntimeError`; the packed path raises `AssertionError` and the unpacked
path `RuntimeError`. Both are hard failures in 0.0.7 and both work in 0.1.0.

### B5. `extra_repr`: confirmed

| | `repr(layer)` |
|---|---|
| OLD | `TypeError: object of type 'int' has no len()` |
| NEW | `DeformConv1d(8, 8, kernel_size=3, stride=1, padding='valid', padding_mode=reflect)` |

0.0.7 copied `nn.ConvNd.extra_repr` verbatim, which indexes `self.output_padding` and
takes `len(self.padding)`; neither exists on a `DeformConv1d` with an int `padding`.
Any `print(model)` on a network containing the layer raised.

### B6. `import dc1d.nn` without tkinter: confirmed

Reproduced on Python 3.14.4 (`/usr/bin/python3`) with torch 2.13.0+cpu and
torchvision 0.28.0+cpu installed and no `tkinter`:

| | result |
|---|---|
| OLD | `ImportError: No module named 'tkinter', please install the python3-tk package` |
| NEW | imports cleanly |

Cause is `from turtle import forward` at `dc1d/nn.py` line 21 of 0.0.7, an IDE
autocomplete accident; `turtle` pulls in `tkinter`. Note this does not reproduce under
uv's bundled CPython, which ships tkinter.

### B7. Integer `padding` with `padding_mode='zeros'` (not on the brief)

0.0.7's forward pads only when `padding_mode != 'zeros'`, or when `padding == 'same'`.
An int `padding` combined with `padding_mode='zeros'` therefore applies no padding at
all, and the index clamp inside the interpolation absorbs the resulting shape error and
returns a plausible, wrong answer of the right length. `C=2`, `L=16`, `k=3`,
`padding=1`, zero offsets, float64:

| | output length | `torch.equal` to `nn.Conv1d` | max abs deviation |
|---|---|---|---|
| OLD | 16 (matches) | False | 1.39 |
| NEW | 16 | **True** | 0.0 |

## C. The opt-in backward, `gather_lerp='recompute'`

Claim under test: gradients bit-identical to the default path. **Partly true.** The
forward and `d/d offsets` are bit-identical; `d/d input` is not, by one rounding.

`k=3`, `d=2`, `stride=1`, `L=257`, `channels=6`, `offset_groups=3`:

| device | dtype | impl | forward `torch.equal` | d/d input `torch.equal` | max abs diff | d/d offsets `torch.equal` | max abs diff |
|---|---|---|---|---|---|---|---|
| cpu | float64 | recompute | **True** | False | 1.78e-15 | **True** | 0.0 |
| cpu | float64 | save-diff | **True** | False | 1.78e-15 | **True** | 0.0 |
| cpu | float32 | recompute | **True** | False | 4.77e-07 | **True** | 0.0 |
| cpu | float32 | save-diff | **True** | False | 4.77e-07 | **True** | 0.0 |
| cuda:0 | float64 | recompute | **True** | False | 8.88e-16 | **True** | 0.0 |
| cuda:0 | float64 | save-diff | **True** | False | 8.88e-16 | **True** | 0.0 |
| cuda:0 | float32 | recompute | **True** | False | 4.77e-07 | **True** | 0.0 |
| cuda:0 | float32 | save-diff | **True** | False | 4.77e-07 | **True** | 0.0 |

The difference is deterministic and reproduces on CPU, so it is not CUDA scatter-add
nondeterminism. It is `_scatter_input_grad`'s `lo = grad_out - hi` where the autograd
path computes `grad_out * (1 - w)`: algebraically equal, one different rounding. It is
not a loss of accuracy. Against a float64 reference, float32, `cuda:0`:

| impl | max abs d/d input error vs float64 |
|---|---|
| autograd (default) | 7.95e-07 |
| recompute | **6.21e-07** |
| save-diff | **6.21e-07** |

The custom Functions are very slightly *closer* to the float64 answer than the default.

`d/d input` is not run-to-run reproducible on CUDA for any of the three variants (a
scatter-add with atomics), and is reproducible on CPU for all three. This matches what
`dc1d/ops.py` documents; it is a property of the scatter, not of the opt-in backward.

| device | autograd | recompute | save-diff |
|---|---|---|---|
| cpu | `torch.equal` True | True | True |
| cuda:0 | `torch.equal` False | False | False |

**Recommendation:** the docstring for `gather_lerp` says "the same gradients to within
rounding", which is accurate. Nothing needs to change in the code. But no test should
be written that asserts `torch.equal` on `d/d input` across variants, and the release
notes should not claim bit-identical gradients.

## D. DTCN, end to end

No trained DTCN checkpoint is published (no releases on `jwr1995/DTCN` or
`jwr1995/PubSep`, no models under that account on Hugging Face), and WHAMR! derives from
LDC-licensed WSJ0, so the *paper's* weights could not be used. Instead: the model is
built once from the yaml's own seed (1234), its `state_dict` is saved, and every run
loads that same checkpoint with `strict=True` and is fed the same fixed input
(`torch.randn`, generator seed 20260728, 1 x 32000 samples at 8 kHz, matching the
yaml's `training_signal_len`). The forward pass reproduces the inference branch of
`Separation.compute_forward` from `separation/train.py`. The checkpoint is untrained;
that affects how the numbers should be *interpreted*, not whether the two libraries
compute the same function.

speechbrain 1.0.3 needed one shim (`torchaudio.list_audio_backends`, removed in
torchaudio 2.11), applied in `verification/dtcn_forward.py`. Nothing else about DTCN
needed changing; it ran unmodified.

### D1. As shipped: 0.0.7 against 0.1.0

Same checkpoint, same input, float32, `cuda:0`, TF32 off.

| | |
|---|---|
| encoder output `mix_w` | `torch.equal` **True** (the deformable layers are the only difference) |
| separated waveform | `torch.equal` **False** |
| max abs deviation | **6.245** (waveform peak is 12.854) |
| rms deviation | 1.217 |
| SI-SDR, 0.1.0 against 0.0.7, speaker 0 | **4.840 dB** |
| SI-SDR, speaker 1 | **4.985 dB** |
| deformable layers with differing predicted offsets | **23 of 24** |

Per layer, the pattern is unambiguous:

| layer | dilation | offsets `torch.equal` | max abs offset difference | offset scale |
|---|---|---|---|---|
| temporalblock_0_0 | 1 | **True** | 0.0 | 5.04 |
| temporalblock_0_1 | 2 | False | 4.55 | 4.90 |
| temporalblock_0_2 | 4 | False | 5.67 | 5.89 |
| temporalblock_0_3 | 8 | False | 4.54 | 5.20 |
| temporalblock_0_4 | 16 | False | 5.62 | 5.18 |
| temporalblock_0_5 | 32 | False | 7.81 | 7.74 |
| temporalblock_0_6 | 64 | False | 5.07 | 5.02 |
| temporalblock_0_7 | 128 | False | 4.57 | 4.82 |
| temporalblock_1_0 | 1 | False | 4.09 | 4.92 |
| ... | ... | False | 4.02 to 7.52 | |
| temporalblock_2_7 | 128 | False | 4.35 | 4.00 |

The first layer is the only one that still sees an identical input when it runs, and it
is also a `dilation=1` layer, so the two versions wire its `offset_dconv` identically:
its offsets agree **bit for bit**. Divergence begins at the first `dilation=2` layer and
never recovers. Every later layer differs by an amount comparable to the full scale of
the offsets themselves, the later `dilation=1` layers included, those only because their
input has already diverged.

### D2. Isolating the cause

Re-running 0.1.0 with every `offset_dconv` rewired back to the 0.0.7 construction
(`stride=1`, `dilation=1`, same weights) leaves the new interpolation kernel as the only
difference:

| comparison | dtype | `torch.equal` | max abs deviation | SI-SDR |
|---|---|---|---|---|
| 0.0.7 vs 0.1.0, as shipped | float32 | False | 6.245 | **4.84 dB** |
| 0.0.7 vs 0.1.0 with 0.0.7's offset-conv wiring | float32 | False | 1.88e-02 | **57.57 dB** |
| same, in float64 | float64 | False | 5.96e-12 | **249.11 dB** |

So the 4.84 dB is entirely the `offset_dconv` rewiring, and the interpolation kernel
rewrite by itself moves the output of a 24-layer network by 1.9e-2 out of a peak of
12.85 in float32, and by 6e-12 in float64. The float64 row is the proof that the kernel
change is a rounding difference and not a semantic one.

### D3. Which side is more accurate at DTCN's operating point

Scoring both float32 runs against the float64 evaluation of the same network:

| run | max abs deviation from float64 | rms | SI-SDR |
|---|---|---|---|
| 0.0.7, float32 | 3.35e-02 | 5.671e-03 | 52.353 dB |
| 0.1.0 (compat wiring), float32 | 4.00e-02 | 5.684e-03 | 52.349 dB |

Honest answer: **at DTCN's sequence length there is no measurable accuracy difference.**
The masknet runs at length 3999, where 0.0.7's position error is ~1e-4 (table A2b) and
is swamped by the float32 noise of the 24 surrounding convolutions, norms and PReLUs.
0.1.0's precision fix is real (A2, A2b, A4) but it buys DTCN nothing at 4 s and 8 kHz.
It would begin to matter for sequences an order of magnitude longer, for float16 or
autocast (B1, where it matters immediately), and for the offset gradient during training
at `L` in the tens of thousands (A4).

## Regressions: where 0.1.0 changed behaviour that 0.0.7 had

Actively looked for. Two found.

### R1. Silent checkpoint incompatibility for `PackedDeformConv1d`, `dilation != 1` or `stride != 1`

This is finding 1, isolated at layer level. `C=8`, `k=3`, `L=512`, `padding='same'`,
depthwise, float32, `cuda:0`. A 0.0.7 `state_dict` is loaded into the 0.1.0 layer:

| dilation | stride | `load_state_dict(strict=True)` | offsets `torch.equal` | max abs offset difference | max abs output difference | output scale |
|---|---|---|---|---|---|---|
| 1 | 1 | ok | **True** | 0.0 | 3.60e-05 | 2.89 |
| 2 | 1 | **ok** | False | 4.76 | 2.60 | 2.68 |
| 4 | 1 | **ok** | False | 4.71 | 2.53 | 2.34 |
| 128 | 1 | **ok** | False | 5.45 | 2.83 | 2.98 |

The load succeeds with no missing or unexpected keys and no warning, because
`offset_dconv.weight` is `(C, 1, k)` regardless of dilation. The layer then computes a
different function. There is no diagnostic a user could reasonably be expected to see.

Is 0.1.0 wrong here? Not on the merits. Feeding the offset predictor a different
receptive field from the deformable convolution it feeds is at best an odd choice, and
at `stride > 1` the old wiring is outright broken (B3). But it is a behaviour change to
a trained architecture, it is silent, and it lands in a version whose other changes are
presented as bug fixes. Suggested mitigations, in order of preference:

1. Note it prominently in the 0.1.0 release notes and the README, naming DTCN and
   PubSep, and telling anyone reproducing those papers to pin `dc1d<0.1`.
2. Add an `offset_dilation` / `offset_stride` argument, defaulting to matching the
   deformable path, so 0.0.x behaviour is expressible rather than lost.
3. Consider a `load_state_dict` hook that warns when a checkpoint is loaded into a
   `PackedDeformConv1d` with `dilation != 1`.

### R2. Non-integer `dilated_positions` are silently rounded

0.0.7's `efficient_linterpolate` treated `dilated_positions` as arbitrary float tap
positions and interpolated them. 0.1.0's `_dilated_positions_long` does
`.round().long()`. `k=3`, `d=2`, `L=32`, zero offsets, float64, tap positions
`[0.0, 1.5, 3.7]`:

| | tap values returned |
|---|---|
| x[0:5] | `[-0.884192, 0.593211, 0.188965, -2.657912, 0.797067]` |
| OLD | `[-0.884192, 0.391088, -0.239427]` (interpolated at 1.5 and 3.7) |
| NEW | `[-0.884192, 0.188965, 0.797067]` (= `x[0]`, `x[2]`, `x[4]`) |

Neither `DeformConv1d` nor `PackedDeformConv1d` can reach this: both pass an integer
ramp buffer. It only affects a direct call to `efficient_linterpolate` with a custom
`dilated_positions`, which is a documented parameter of a public function. The new
docstring does say "integer kernel tap positions", so the contract is stated, but the
rounding is silent. **Suggested fix: raise rather than round**, i.e. check
`torch.equal(dilated_positions, dilated_positions.round())` and raise `ValueError`
otherwise. That converts a silent wrong answer into an error and costs one comparison
per call.

### Checked and found not to be regressions

| check | result |
|---|---|
| out-of-range offsets in constrained mode (-50, -1, +50) | `torch.equal` True, identical clamping |
| unconstrained mode with offsets far outside `[0, L]` (-1e4) | identical |
| the `device=` argument now being ignored | 0.1.0 takes the device from `x`, which is strictly better; 0.0.7 could allocate its zero tensor on the wrong device |
| grouped, strided, dilated `nn.Conv1d` equivalence | bit-exact in both (A1) |

## What was not tested, and why

- **The paper's trained weights.** Not published anywhere reachable, and the training
  data is LDC-licensed. D1 therefore uses an untrained checkpoint. The conclusion that
  0.0.7 and 0.1.0 compute different functions from identical weights does not depend on
  the weights being trained; the specific 4.84 dB number does.
- **Real speech.** The input is Gaussian noise of the yaml's `training_signal_len`.
  SI-SDR between two versions of the same network is a comparison of two outputs, not a
  separation quality metric, so the input distribution does not affect the argument.
  Separation quality of either version was not and could not be measured.
- **Training, as opposed to a forward pass and a single backward.** No claim is made
  here about whether a model trained under 0.1.0 reaches the paper's SI-SDR. A5 and A4
  give reason to expect training to be *better* behaved (0.0.7's offset gradient is
  wrong at every integer-valued position), but that is an inference, not a measurement.
- **`torch.compile`, `vmap`, double backward.** Out of scope for an old-versus-new
  comparison; 0.0.7 supports none of them meaningfully.
- **float16 end to end.** DTCN's own README says autocast training is broken and not
  recommended, so there is no fp16 baseline to compare against. B1 covers the kernel.
- **Multi-GPU.** All runs are on `cuda:0`; `cuda:1` was left idle.

## Reproducing

The kernel harness needs torch, plus torchvision solely so that 0.0.7 can be imported.
The DTCN harness additionally needs speechbrain and a DTCN checkout. Neither is a
dependency of dc1d, and neither can run in this repository's CPU-only dev environment,
so they are run from a separate virtualenv rather than through `uv run`.

```
git worktree add --detach /tmp/dc1d-old 6ca2e01
git clone https://github.com/jwr1995/DTCN /tmp/DTCN

uv venv --python 3.11 /tmp/eqv
VIRTUAL_ENV=/tmp/eqv uv pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
VIRTUAL_ENV=/tmp/eqv uv pip install speechbrain==1.0.3

/tmp/eqv/bin/python verification/equivalence_check.py --old /tmp/dc1d-old --new .

/tmp/eqv/bin/python verification/dtcn_forward.py --variant old --dc1d-root /tmp/dc1d-old \
    --dtcn /tmp/DTCN --save-state /tmp/ckpt.pt --out /tmp/old_fp32.pt
/tmp/eqv/bin/python verification/dtcn_forward.py --variant new --dc1d-root . \
    --dtcn /tmp/DTCN --load-state /tmp/ckpt.pt --out /tmp/new_fp32.pt
/tmp/eqv/bin/python verification/dtcn_compare.py /tmp/old_fp32.pt /tmp/new_fp32.pt --per-layer
```

Add `--offset-conv-compat` to `dtcn_forward.py` to rewire 0.1.0's offset convolutions
back to the 0.0.7 construction, and `--dtype float64` for the float64 rows.
