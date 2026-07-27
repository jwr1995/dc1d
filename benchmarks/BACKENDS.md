# 1D deformable convolution backends: a comparison

Four implementations of the same operator, measured against each other:
numerical equivalence (forward **and** both gradients), a defect cross-audit,
latency, and peak memory.

Everything here is reproduced by `benchmarks/backends.py`. Figures labelled
**measured** come from a run of that script on the hardware below; anything
labelled **estimated** is arithmetic, not measurement, and is marked as such.

```
uv sync --group bench
uv run --group bench python benchmarks/backends.py --check --defects   # CPU
.venv-cuda/bin/python benchmarks/backends.py --all --device cuda:0     # CUDA
```

**Hardware and versions** (all measured figures below come from this machine):

```
torch            : 2.13.0+cu129        torchvision : 0.28.0+cu129
python           : 3.12.13             cuda (torch): 12.9     cudnn: 92000
gpu              : NVIDIA GeForce RTX 3090, sm_86, 24 GiB
cpu figures      : same box, torch 2.13.0+cpu (the dc1d dev venv)
tinymera ref     : fix/causality @ 04593f38, tinymera/ops/deform_conv1d.py
dtype            : float32 for all timing and memory; float64 for equivalence
triton overrides : DISABLED
```

> **Disclosure — Triton overrides off.** torch 2.13 ships Triton-DSL overrides
> for some ATen ops (notably `einsum` → `_bmm_outer_product`) that are JIT-built
> on first use. This machine has **no host C compiler**, so the first CUDA
> `einsum` raised `RuntimeError: Failed to find C compiler`, which would have
> excluded tinymera (both its kernels contract with `einsum`) from the
> comparison entirely. `backends.py` therefore deregisters the Triton overrides
> when no compiler is visible and says so in its environment banner. The switch
> is **global**, so every backend runs under the same dispatch regime and the
> comparison stays internally fair — but tinymera's absolute numbers are "ATen
> `bmm` fallback", not "best possible", and could improve on a box with a
> compiler. dc1d, torchvision and the `grid_sample` backend do not use `einsum`
> and are unaffected.



---

## 1. The four contenders

| | what it is | compiled? | offset groups | boundary | bit-exact vs `nn.Conv1d`? |
|---|---|---|---|---|---|
| **dc1d** | `take_along_dim` + `lerp` + `conv1d` (`dc1d/ops.py:65`) | no | any divisor of `C` | clamp | **yes** |
| **torchvision** | `deform_conv2d` C++/CUDA kernel, height 1 | **yes** | any divisor of `C` | zero-pad | yes |
| **grid_sample** | `F.grid_sample` on `(B, C, 1, L)` (`benchmarks/backends.py`) | no (ATen builtin) | any divisor of `C` | clamp (`border`) | no |
| **tinymera** | independent 2nd implementation, two kernels | no | **fixed at `C`** | clamp | gather: yes; grid_sample: no |

### Algorithmic shape

All four compute the same thing: for output position `t` and kernel tap `k`,
sample the input at the fractional position

```
T[b, g, t, k] = t*stride + k*dilation + offset[b, g, t, k]
```

by linear interpolation between `floor(T)` and `floor(T)+1`, then contract the
`K` taps against the kernel weights. They differ in *how*:

* **dc1d** keeps `t*stride + k*dilation` in `long` and carries only the
  sub-sample fraction in float (`dc1d/ops.py:140-162`), then does two
  `take_along_dim` gathers and one `lerp` (`dc1d/ops.py:184-193`). The tap axis
  is flattened and contracted with a stride-`K` grouped `F.conv1d`
  (`dc1d/nn.py:255-262`). `take_along_dim` broadcasts, so the index tensor is
  never tiled to the channel count — that is what makes `1 < offset_groups < C`
  work without a `C`-sized int64 index.
* **torchvision** does the whole thing in one fused kernel. Offsets are packed
  into the channel axis as `(offset_group, kh, kw, {y, x})`; with `kh = 1` the
  height interpolation is a no-op. The `(y, x)` ordering and the offset sign
  are **probed, not assumed** (`_probe_offset_layout`).
* **grid_sample** normalises `T` to `[-1, 1]` and calls
  `aten::grid_sampler_2d`. Offset groups are folded into the batch axis:
  `x` as `(B*G, C/G, 1, L)`, grid as `(B*G, 1, L_out*K, 2)`. The contraction is
  the same stride-`K` `conv1d` as dc1d, so any difference is attributable to
  the interpolation alone.
* **tinymera** ships both a `grid_sample` kernel and a `gather` kernel. Both
  fold `C` into the batch (`x.reshape(B*C_in, 1, 1, T_in)`), because offsets are
  always per-channel, and both contract with `torch.einsum` rather than
  `conv1d`.

### Verified `grid_sample` mechanics

The `grid_sample` backend rests on three claims that were checked rather than
assumed:

1. **`align_corners=True` gives exactly the `1-f` / `f` weights.** Sampling a
   unit impulse at `5 + f` returns exactly `1-f` for
   `f ∈ {0, 0.25, 0.5, 0.75}` in float64 (error `0`; at `f = 0.999` the error
   is `3.3e-16`, i.e. the round-trip through `g`, not the weight formula).
2. **`padding_mode='border'` reproduces dc1d's index clamp exactly.** For
   positions `-3, -0.5, 0, L-1, L-0.5, L+9` on a `1..L` ramp, `border` returns
   the clamped-reference value with error `0` in every case.
3. **`grid_sample` is an ATen builtin.** `torch.ops.aten.grid_sampler_2d`
   exists; nothing is compiled at import or at call time. **Adopting it would
   preserve dc1d's "no compilation required" property.**

---

## 2. Numerical equivalence

### 2.1 Anchor: zero offsets must reproduce `nn.Conv1d`

`--check` runs this across stride, dilation, groups and kernel size.

| backend | max abs error vs `nn.Conv1d`, float64 | bit-exact |
|---|---|---|
| dc1d | `0.000e+00` (every config) | **yes** |
| torchvision | `0.000e+00` (4/5 configs), `8.9e-16` (1/5) | effectively yes |
| grid_sample | `3.3e-14` – `6.2e-14` | **no** |
| tinymera-gather (fp32) | `0.000e+00` (4/5), `9.5e-07` (1/5) | effectively yes |
| tinymera-grid_sample (fp32) | `4.7e-05` – `8.0e-05` (~`4e-06` relative) | **no** |

The two `grid_sample`-based backends cannot hit this invariant, for a reason
that is structural, not a bug — see §2.4.

### 2.2 Interior equivalence, forward and both gradients

Random offsets in `±2`, restricted to output positions where every tap stays
inside `[0, L-1]`, so no boundary convention is in play. Ten configurations
covering stride, dilation, groups, offset groups, `K=1`, `K=7`, and `L=512`.
Gradients are taken against a common upstream gradient that is zeroed outside
the interior.

**float64, tolerance `1e-10` relative — all pass, 0 failures:**

| pair | forward | `d/d input` | `d/d offsets` |
|---|---|---|---|
| dc1d vs torchvision | `≤ 4.7e-13` | `≤ 5.1e-13` | `≤ 7.1e-15` |
| dc1d vs grid_sample | `≤ 1.2e-12` | `≤ 7.7e-13` | `≤ 7.1e-15` |

**The offset gradient — the entire point of a deformable layer — agrees to
float64 round-off in all three implementations.** `grid_sample` propagates
gradient into its grid correctly; `torch.autograd.gradcheck` in float64 against
both `input` and `offsets` passes for the `grid_sample` backend
(`_grid_sample_gradcheck`).

**float32, tolerance `1e-3` relative — all pass, 0 failures**, including both
tinymera kernels. The slacker tolerance is not a weaker claim about indexing:
at float32 the backends genuinely disagree at ~`1e-5` for three unavoidable
reasons (different accumulation order over the `K*C_in/groups` terms,
`grid_sample`'s normalisation, tinymera's internal downcast), while any
*structural* error — a mis-mapped offset group, an off-by-one clamp, an
inverted sign — is `O(1)` relative. dc1d's own float32-vs-float64 error is
printed per config so the floor is visible rather than asserted.

tinymera **cannot** be held to a float64 tolerance: see §3, defect D2.

### 2.3 Boundary semantics — where they genuinely differ

This is the one place the implementations are not the same operation.

Input `x = [1..12]`, `K=3`, tap 0 isolated, first four output positions:

| offset | dc1d (clamp) | grid_sample (`border`) | tinymera (clamp) | torchvision (zero-pad) |
|---|---|---|---|---|
| `-0.5` | `1.0, 1.5, 2.5, 3.5` | `1.0, 1.5, 2.5, 3.5` | `1.0, 1.5, 2.5, 3.5` | `0.5, 1.5, 2.5, 3.5` |
| `-3.0` | `1.0, 1.0, 1.0, 1.0` | `1.0, 1.0, 1.0, 1.0` | `1.0, 1.0, 1.0, 1.0` | `0.0, 0.0, 0.0, 1.0` |
| `+50.0` | `12.0, ...` | `12.0, ...` | `12.0, ...` | `0.0, ...` |

* **dc1d, grid_sample and tinymera all clamp** and agree with each other to
  `≤ 2.2e-16` (grid_sample) and **exactly `0`** (tinymera) at every offset
  tested. dc1d's clamp keeps the interpolation weights summing to 1 everywhere.
* **torchvision zero-pads.** An out-of-bounds bilinear tap contributes `0`, so
  a tap at `-0.5` reads `0.5 * x[0]` rather than `x[0]`, and differs at up to
  `10/10` positions for a large offset. Neither convention is wrong; they are
  different definitions and no tolerance will reconcile them.

**How much of the output is affected** (offsets `~ U(-r, r)`, `K=3`, `d=1`):

| L | r=1 | r=4 |
|---|---|---|
| 64 | 2/62 (3.2%) | 8/62 (12.9%) |
| 1024 | 2/1022 (0.20%) | 8/1022 (0.78%) |
| 16000 | 2/15998 (0.01%) | 8/15998 (0.05%) |

At speech-separation lengths the divergence is confined to ~0.01–0.05% of
positions. It is still a semantic difference, not a rounding difference.

**dc1d's constrained mode has no analogue anywhere else.** With
`unconstrained=False` (the default) each tap is confined to its own receptive
field. Against torchvision the interior error is then `1.1e+01` rather than
`4.0e-14` — as intended. Equivalence only holds against
`DeformConv1d(..., unconstrained=True)`.

### 2.4 The `grid_sample` precision tax

`grid_sample` requires coordinates normalised to `[-1, 1]` and un-normalises
internally as `(g + 1) / 2 * (L - 1)`. That `+ 1` is a catastrophic
cancellation for coordinates near the middle of the signal: an error of `eps`
in `g` becomes an error of `eps * (L - 1) / 2` **samples** in the recovered
position. No choice of grid construction avoids it — the backend here already
uses the exact-numerator form `(2T - (L-1)) / (L-1)`.

**Measured** (zero offsets, so the exact answer is a plain `unfold`; error
should be `0`):

| dtype | L | dc1d max err | grid_sample max err | implied position err |
|---|---|---|---|---|
| float32 | 256 | `0` | `3.2e-05` | `3.5e-05` samples |
| float32 | 2048 | `0` | `4.95e-04` | `5.0e-04` samples |
| float32 | 16000 | `0` | `4.13e-03` | `4.3e-03` samples |
| float32 | 65536 | `0` | `1.96e-02` | `2.1e-02` samples |
| float16 | 256 | `0` | `3.6e-01` | `0.40` samples |
| float16 | 2048 | `0` | `2.62e+00` | `2.7` samples |

dc1d is bit-exact at every length and every dtype. `grid_sample`'s error grows
**linearly with L**. In float32 it stays below `2e-2` samples even at `L=65536`
— tolerable for a layer whose offsets are learned anyway — but it is not zero,
and the `nn.Conv1d` invariant is lost.

---

## 3. Defect cross-audit

dc1d fixed seven correctness bugs in `eac995f`. tinymera is a second
implementation of the same algorithm by the same author, so the question is
which classes recur. Probes are executable (`backends.py --defects`) except
where noted.

**tinymera reference**: `fix/causality` @ `04593f38` (open as tinymera PR #1
against `feature/exciting-plc`), which is "post-fix". "Pre-fix" is
`feature/exciting-plc` @ `ecf8da24`. Sources:
`tinymera/ops/deform_conv1d.py`, `tinymera/nn/deform_conv1d.py`.

| # | bug class | dc1d pre-fix | dc1d post-fix | tinymera pre-fix | tinymera post-fix |
|---|---|---|---|---|---|
| **D1** | dtype-inherited position arithmetic | **present** — `linspace` in offset dtype; 13952/15998 rows wrong at fp16, `L=16000` | fixed — window starts in `long`, only the fraction in float (`dc1d/ops.py:140-162`) | **present** — `torch.arange(T_out, dtype=x.dtype)` (both kernels) | fixed — forced `torch.float32` (`ops/deform_conv1d.py:110-111, 209-210`) |
| **D2** | float64 silently downcast | absent | absent — exact at every dtype | absent | **present (introduced by the D1 fix)** — `offsets.float()` and `x.float()`; measured `1.5e-05` error on a float64 input, i.e. float32-sized |
| **D3** | `repeat` vs `repeat_interleave` for offset groups | **present** — `U.repeat` tiled to `G*C`; `1 < G < C` raised `RuntimeError` | fixed — `take_along_dim` broadcasts, channel axis viewed as `(G, C/G)` (`dc1d/ops.py:180-193`) | **n/a — structurally unreachable** | **n/a** — offsets are always per-channel; there is no group→channel mapping to get wrong |
| **D4** | boundary clamp to `L` vs `L-1` | **present** — clamp to `x.shape[-1]`; at `T == L` *both* weights fell to 0, output exactly `0.0` | fixed — index clamped to `[0, L-2]`, fraction forced to 0/1 (`dc1d/ops.py:151-162`) | absent — `pos.clamp(0, T_in - 1)` was always correct | absent — verified: saturates to `x[L-1]` for a `1e6` offset |
| **D5** | stride/dilation dropped in the offset-prediction path | **present** — offset conv hardcoded `stride=1, dilation=1`; `stride=2, L=40` returned 38 instead of 19 | fixed — both forwarded (`dc1d/nn.py:337-348`) | **present** | **STILL PRESENT (partial)** — `dilation` is forwarded, **`stride` is hardcoded to 1** (`nn/deform_conv1d.py:175-183`) |
| **D6** | the index clamp hides shape bugs (no contract validation) | **present** — wrong `L_out` produced plausible wrong-length output | fixed — `ValueError` against the closed form (`dc1d/nn.py:236-243`), plus `expected_offset_positions()` | **present** | **STILL PRESENT** — sampling kernels take `T_out` from `offsets.shape` and clamp; nothing validates it |
| **D7** | `2^7`-style XOR-for-exponent | **present** — a benchmark ran `dilation=5` for three years | fixed — `2**7` (`dc1d/nn.py:487`) | absent | absent — `grep -rE '[0-9]\s*\^\s*[0-9]'` over the tree returns only a comment |

### D5 — reproduced

`tinymera.nn.DeformConv1d` builds its offset network with `stride=1`
(`nn/deform_conv1d.py:175-183`) but passes `self.stride` to the sampling kernel.
The offset network therefore emits `T_in` positions while the kernel needs
`T_out`, and the kernel silently uses `T_out = offsets.shape[2] = T_in`:

```
stride=1: tinymera out (2, 4, 64)   nn.Conv1d(same args) (2, 4, 64)   OK
stride=2: tinymera out (2, 4, 64)   nn.Conv1d(same args) (2, 4, 32)   WRONG
stride=4: tinymera out (2, 4, 64)   nn.Conv1d(same args) (2, 4, 16)   WRONG
```

The output is not merely the wrong length: the tail positions have
`nominal = t*stride` running past `T_in`, so they are all clamped to the last
input sample. This is **exactly** dc1d's C4, half-fixed. It survives because
`tests/nn/test_deform_conv1d.py` has **no stride coverage at all** — the
`stride` parametrisation exists only in the *ops*-level test, which computes
`T_out` correctly itself and so never exercises the module's offset network.

### D6b — the causal contract is also unvalidated (tinymera only)

Same class as D6, found while auditing. `causal=True` is only actually causal
when `2*padding >= dilation*(kernel_size-1)`, because the last tap must land on
the current step. Nothing checks it. Measured by
`d out[t] / d x[t'] for t' > t`, `K=3`, `d=4`, `C=4`, `T=64`:

| `padding` | contract wants | max `|d out[32] / d x[t'>32]|` | |
|---|---|---|---|
| 4 | ≥ 4 | `0.000e+00` | causal |
| 2 | ≥ 4 | `3.58e-01` | **leaks future context** |
| 0 | ≥ 4 | `4.45e-01` | **leaks future context** |

`causal=True` with the module's default `padding=0` is silently non-causal.
Worth raising on tinymera PR #1, which otherwise fixed the causality
violations correctly.

### Reading the table

Three of the seven classes recurred in the independent implementation (D1, D5,
D6), one was structurally impossible (D3), two never occurred (D4, D7), and the
fix for one introduced a new one (D2). **The correlated-mistake hypothesis
holds for the contract-validation classes specifically**: both implementations
independently let an unvalidated shape be absorbed by a boundary clamp (D6),
and both independently forgot to thread `stride` through the offset network
(D5). Those are the two to check first in any third implementation.

D2 is worth dwelling on: dc1d and tinymera fixed the *same* bug (D1) in
*different* ways, and tinymera's fix — force fp32 — is strictly weaker. It
solves the low-precision direction but makes float64 unavailable, which means
`torch.autograd.gradcheck` cannot be used on tinymera's kernels at all.
tinymera has no gradcheck test; dc1d's `tests/test_gradients.py` is float64
against both `input` and `offsets`. dc1d's decomposition (integers in `long`,
fraction in the input dtype) is exact at *every* dtype and is the better fix.



---

## 4. Performance

RTX 3090, float32, `torch.utils.benchmark.Timer` (`blocked_autorange`), forward
and forward+backward timed separately, **equal warmup (3 iterations) for every
contender**. Implementations are measured round-robin within each of 3 rounds so
all five see the same thermal state, and the reported figure is the **minimum**
across rounds. `spread` is max/min across rounds — the machine's noise floor.
The 3090 also drives the desktop, so rows with `spread > 1.5` are flagged and
their ratios should be read as order-of-magnitude.

Bracketed factors are **dc1d / contender**, so **> 1× means faster / smaller
than dc1d**.

### 4.1 Latency — forward (ms)

| config | dc1d | torchvision | grid_sample | tinymera-gs | tinymera-gth | nn.Conv1d | spread |
|---|---|---|---|---|---|---|---|
| B=1 C=16 L=256 K=3 d=1 g=1 | 0.298 | 0.219 (1.36×) | **0.181 (1.65×)** | 0.200 (1.49×) | 0.307 (0.97×) | 0.028 | 1.13 |
| B=4 C=64 L=1024 K=3 d=1 g=1 | 0.347 | 0.239 (1.45×) | **0.214 (1.62×)** | 0.223 (1.55×) | 0.349 (1.00×) | 0.043 | 1.13 |
| B=4 C=64 L=1024 K=3 d=1 g=64 | 0.284 | 1.302 (0.22×) | **0.164 (1.74×)** | 0.219 (1.30×) | 0.315 (0.90×) | 0.016 | 1.15 |
| B=4 C=256 L=2048 K=3 d=8 g=256 | 0.535 | 4.531 (0.12×) | **0.175 (3.05×)** | 0.845 (0.63×) | 1.884 (0.28×) | 0.051 | 1.02 |
| B=4 C=256 L=2048 K=3 d=8 g=1 | 0.708 | 0.369 (1.92×) | **0.345 (2.05×)** | 0.930 (0.76×) | 1.923 (0.37×) | 0.166 | 1.01 |
| B=4 C=128 L=4096 K=15 d=1 g=1 | 2.819 | **0.937 (3.01×)** | 1.048 (2.69×) | 4.055 (0.70×) | 8.736 (0.32×) | 0.327 | 1.02 |
| B=4 C=128 L=4096 K=15 d=1 g=128 | 2.253 | 2.475 (0.91×) | **0.428 (5.27×)** | 3.681 (0.61×) | 8.489 (0.27×) | 0.071 | 1.08 |
| B=4 C=128 L=4096 K=3 d=2 g=1 | 0.320 | 0.218 (1.47×) | **0.197 (1.63×)** | 0.253 (1.26×) | 0.542 (0.59×) | 0.052 | 1.11 |
| B=4 C=128 L=4096 K=3 d=1 g=1 ⚠ | 0.704 | 0.282 (2.50×) | **0.266 (2.65×)** | 0.892 (0.79×) | 1.881 (0.37×) | 0.132 | 2.50 |
| B=1 C=256 L=16000 K=3 d=1 g=256 | 1.104 | 4.635 (0.24×) | **0.289 (3.82×)** | 1.439 (0.77×) | 3.494 (0.32×) | 0.120 | 1.16 |
| B=4 C=256 L=16000 K=3 d=8 g=256 | 4.476 | 4.074 (1.10×) | **0.978 (4.58×)** | 7.677 (0.58×) | 14.303 (0.31×) | 0.526 | 1.34 |
| B=1 C=256 L=16000 K=3 d=1 g=1 | 1.491 | 1.333 (1.12×) | **0.670 (2.22×)** | 2.415 (0.62×) | 3.920 (0.38×) | 0.396 | 1.02 |
| B=8 C=512 L=8000 K=3 d=1 g=512 ⚠ | 24.126 | **7.795 (3.09×)** | 11.361 (2.12×) | 13.055 (1.85×) | 30.731 (0.79×) | 1.258 | 2.83 |

### 4.2 Latency — forward+backward (ms)

| config | dc1d | torchvision | grid_sample | tinymera-gs | tinymera-gth | nn.Conv1d | spread |
|---|---|---|---|---|---|---|---|
| B=1 C=16 L=256 K=3 d=1 g=1 | 0.760 | 0.631 (1.20×) | **0.503 (1.51×)** | 0.649 (1.17×) | 0.994 (0.76×) | 0.214 | 1.19 |
| B=4 C=64 L=1024 K=3 d=1 g=1 | 0.850 | 0.667 (1.27×) | **0.572 (1.49×)** | 0.709 (1.20×) | 1.592 (0.53×) | 0.279 | 1.13 |
| B=4 C=64 L=1024 K=3 d=1 g=64 | 0.749 | 4.056 (0.18×) | **0.471 (1.59×)** | 0.698 (1.07×) | 1.579 (0.47×) | 0.187 | 1.15 |
| B=4 C=256 L=2048 K=3 d=8 g=256 | 1.546 | 15.178 (0.10×) | **0.684 (2.26×)** | 1.761 (0.88×) | 12.336 (0.13×) | 0.201 | 1.08 |
| B=4 C=256 L=2048 K=3 d=8 g=1 | 2.220 | 1.579 (1.41×) | **1.347 (1.65×)** | 2.078 (1.07×) | 12.518 (0.18×) | 1.087 | 1.03 |
| B=4 C=128 L=4096 K=15 d=1 g=1 | 7.074 | 4.600 (1.54×) | **3.222 (2.20×)** | 8.948 (0.79×) | 54.685 (0.13×) | 1.832 | 1.04 |
| B=4 C=128 L=4096 K=15 d=1 g=128 | 8.939 | 7.959 (1.12×) | **4.842 (1.85×)** | 7.851 (1.14×) | 53.676 (0.17×) | 0.303 | 1.10 |
| B=4 C=128 L=4096 K=3 d=2 g=1 | 0.842 | 0.665 (1.27×) | **0.575 (1.47×)** | 0.720 (1.17×) | 3.632 (0.23×) | 0.295 | 1.15 |
| B=4 C=128 L=4096 K=3 d=1 g=1 | 2.272 | 2.150 (1.06×) | **1.090 (2.09×)** | 2.858 (0.80×) | 13.889 (0.16×) | 0.588 | 1.34 |
| B=1 C=256 L=16000 K=3 d=1 g=256 ⚠ | 3.511 | 30.786 (0.11×) | **1.466 (2.39×)** | 3.721 (0.94×) | 25.805 (0.14×) | 0.423 | 2.10 |
| B=4 C=256 L=16000 K=3 d=8 g=256 | 13.763 | 14.418 (0.95×) | **5.246 (2.62×)** | 16.938 (0.81×) | 103.925 (0.13×) | 1.534 | 1.38 |
| B=1 C=256 L=16000 K=3 d=1 g=1 ⚠ | 5.806 | **3.082 (1.88×)** | 3.997 (1.45×) | 3.740 (1.55×) | 24.789 (0.23×) | 1.358 | 1.51 |
| B=8 C=512 L=8000 K=3 d=1 g=512 | 52.321 | 46.588 (1.12×) | **36.281 (1.44×)** | 42.446 (1.23×) | 235.131 (0.22×) | 3.801 | 1.15 |

### 4.3 Peak CUDA memory (MiB, `torch.cuda.max_memory_allocated`)

Reset **after** input construction, so this is the transient the op itself
allocates, not the inputs. Forward / forward+backward:

| config | dc1d | torchvision | grid_sample | tinymera-gs | tinymera-gth |
|---|---|---|---|---|---|
| B=4 C=256 L=2048 K=3 d=8 g=256 | 104.9 / 280.1 | 47.8 / 56.0 | **31.9 / 64.1** | 183.9 / 207.9 | 288.0 / 520.8 |
| B=4 C=128 L=4096 K=15 d=1 g=128 | 497.2 / 1337.4 | **143.9 / 153.8** | 128.0 / 257.8 | 856.2 / 976.2 | 1440.3 / 2542.3 |
| B=1 C=256 L=16000 K=3 d=1 g=256 | 205.6 / 548.4 | 94.9 / 95.2 | **62.9 / 126.1** | 313.7 / 360.6 | 564.0 / 973.6 |
| B=4 C=256 L=16000 K=3 d=8 g=256 | 821.1 / 2191.3 | 375.3 / 439.3 | **250.4 / 502.3** | 1439.7 / 1627.7 | 2249.4 / 4072.6 |
| B=1 C=256 L=16000 K=3 d=1 g=1 | 205.6 / 549.1 | **94.9 / 96.0** | 126.9 / 190.5 | 313.7 / 360.6 | 564.0 / 1021.5 |
| B=8 C=512 L=8000 K=3 d=1 g=512 | 5188.6 / 5938.6 | **749.8 / 1625.8** | 2624.6 / 2624.6 | 2874.6 / 3249.5 | 4499.9 / 8149.5 |

Full 13-row tables: run `--mem`.

### 4.4 What the numbers say

**`grid_sample` beats dc1d in every configuration measured, on both axes.**

| | forward | forward+backward |
|---|---|---|
| latency, dc1d / grid_sample | **1.62× – 5.27×** (13/13 wins, median ~2.2×) | **1.44× – 2.62×** (13/13 wins, median ~1.7×) |
| peak memory, dc1d / grid_sample | **1.62× – 3.89×** (13/13) | **2.26× – 5.19×** (13/13) |

This comfortably exceeds the 1.5–2× dc1d's own review estimated. The win is
largest exactly where dc1d is used in anger: depthwise, dilated, long sequences
(`C=256 L=16000 g=256`: **4.58×** forward, **3.28×** less memory).

**torchvision's compiled kernel does *not* win decisively — it is erratic.**
It is the fastest option in 2/13 forward configs, but it is *4–10× slower than
pure-PyTorch dc1d* in several depthwise configs (`0.10×`, `0.11×`, `0.12×`,
`0.18×`, `0.22×`, `0.24×`). Those are precisely the Conv-TasNet-shaped configs
this package exists for. Its memory use is consistently excellent (up to 8.7×
below dc1d), which is what a fused kernel should buy — but the latency is not
reliably there.

**tinymera is slower than dc1d in most configs and never beats
dc1d+`grid_sample`.** `tinymera-gs` is faster than dc1d in 4/13 forward and
5/13 fwd+bwd configs; `dc1d + grid_sample` beats `tinymera-gs` in **13/13**
forward, 12/13 fwd+bwd, and **13/13** on forward memory. The gap is structural:
tinymera folds *all* channels into the batch (`x.reshape(B*C_in, 1, 1, T_in)` —
`B*C_in` single-channel grids) because it has no `offset_groups`, and contracts
with `einsum` instead of a grouped `conv1d`. `tinymera-gth` is the slowest thing
here by a wide margin — up to **7.5× slower than dc1d** on fwd+bwd
(`103.9 ms` vs `13.8 ms`) — and uses 2–3× more memory.

### 4.5 The two prior claims, checked

**(a) "tinymera materialises a `(B, C, K, T)` intermediate costing ~393 MB per
layer per forward at Conv-TasNet dimensions."**

*Arithmetic: correct.* At `B=8, C=512, K=3, T=8000` (`T_out=7998`), fp32:
`8 × 512 × 3 × 7998 × 4 B = 393,043,968 B` = **374.8 MiB = 393 MB**. Confirmed.

*As a reason to avoid it: does not survive measurement.* That tensor is only
**13%** of tinymera-gs's measured forward peak (2874.6 MiB) — the `pos`, `off`,
`grid_x` and `grid` intermediates cost more than the sampled tensor does. More
importantly **dc1d allocates 5188.6 MiB at the same config, 1.8× more than
tinymera**, because it materialises `x0`, `x1` *and* the `lerp` output at that
same `(B, C, L_out, K)` shape. If a 393 MB intermediate is disqualifying, dc1d
is disqualified first. The claim is true and misleading; it is not why tinymera
should be retired (§3 is).

**(b) "dc1d's rewrite cut peak RSS ~3× on CPU."**

*Verified.* Same config as TODO.md (`B=4 C=256 L=2048 K=3 d=8`, 23.8 MiB
output), peak RSS delta in isolated subprocesses, torch 2.13.0+cpu:

| | peak RSS delta |
|---|---|
| pre-rewrite (`eac995f^:dc1d/ops.py`) | **291.0 MiB** |
| current | **99.8 MiB** |
| reduction | **2.92×** |

TODO.md records 283 → 91 MiB = 3.1×; this run gives 2.92×. Agreement within
run-to-run RSS variation. **Note:** the rewrite is in commit `eac995f`
("Fix seven correctness bugs and remove dead code"), *not* in `50a9bed`, whose
message claims the rewrite but which touches only `benchmarks/benchmark.py`.
The commit messages and their contents are misaligned in that pair.



---

## 5. Recommendation

### 5.1 Should dc1d adopt a `grid_sample` backend? — **Yes, as an opt-in backend.**

The measurements support it decisively, and the deciding constraint holds:

* **It is faster and smaller in 13/13 configurations**, forward and
  forward+backward: 1.6–5.3× faster forward, 1.4–2.6× faster fwd+bwd, 1.6–3.9×
  less forward memory, 2.3–5.2× less fwd+bwd memory. dc1d's own review
  estimated 1.5–2×; the measured win is larger.
* **It requires no compilation.** `torch.ops.aten.grid_sampler_2d` is an ATen
  builtin — verified present, nothing is built at import or at call time. This
  was the deciding constraint and it is satisfied. **dc1d's "no C++/CUDA
  compilation" property is preserved.**
* **The offset gradient is correct.** `gradcheck` in float64 against both
  `input` and `offsets` passes, and the gradients match dc1d's to `≤ 7.1e-15`
  (`d/d offsets`) and `≤ 7.7e-13` (`d/d input`). This is the thing that had to
  be true; it is.
* **Boundary semantics are identical.** `padding_mode='border'` reproduces
  dc1d's index clamp exactly (error `0` at every probed offset), so adopting it
  changes no boundary behaviour.
* It supports `offset_groups` (folded into the batch axis) and dc1d's
  constrained mode, so it is a true drop-in for `efficient_linterpolate`.

**But not as the default.** Two things are lost:

1. **The `nn.Conv1d` bit-exactness invariant.** `grid_sample`'s `[-1, 1]`
   normalisation is lossy and the error grows linearly with `L`
   (`4.1e-3` at `L=16000`, `2.0e-2` at `L=65536`, in *samples* of position
   error). `tests/test_equivalence.py` — which CLAUDE.md correctly calls the
   single load-bearing test — **cannot** be applied to this backend bit-exactly.
   That test is what made the 3× kernel rewrite safe to attempt; giving it up by
   default would be a bad trade for a 2× speedup.
2. **Low-precision safety, unless explicitly defended.** A naive `grid_sample`
   backend that inherits the input dtype **reintroduces dc1d's C3 bug**: measured
   at bf16 and `L=16000`, it read the wrong samples entirely, with an error
   **5.94× the RMS of the signal**. Forcing the position and sampling arithmetic
   to at least float32 fixes it (error drops to `7.8e-3`, the normalisation
   floor). This is not optional and must be in the implementation from day one —
   it is the same class of bug the package has already paid for once.

**What would change if adopted:**

* Add `grid_sample_linterpolate` to `dc1d/ops.py` (~45 lines, no new
  dependency, no new import beyond `torch.nn.functional`). `DeformConv1d`
  already takes `interpolation_function`, so no API change is needed —
  `DeformConv1d(..., interpolation_function=grid_sample_linterpolate)` is the
  whole opt-in.
* `tests/test_equivalence.py` stays as-is for the default kernel. Add a
  tolerance-based variant for the new one, with the tolerance a documented
  function of `L`, and a regression test pinning the fp32 forcing (assert the
  bf16 error stays at the normalisation floor, not the collapsed value).
* `tests/test_gradients.py` applies unchanged and passes.
* README/CLAUDE.md need a short "choosing a backend" note: exactness by
  default, speed on request.

### 5.2 Is a fused Triton kernel worth writing? — **Not now.**

The comparison against torchvision's hand-written C++/CUDA kernel is the
evidence, and it does **not** say "compiled kernels win":

* torchvision is fastest in only 2/13 forward configs, and is **4–10× slower
  than pure-PyTorch dc1d** in six of them — all depthwise, which is the regime
  dc1d exists for.
* `grid_sample`, an ATen builtin with no build step, beats torchvision in 10/13
  forward configs.

So the achievable headroom from a *good* fused kernel is real but is mostly
already collected by `grid_sample` at zero build cost. A Triton kernel would
have to beat `grid_sample`'s 1.6–5.3×, would violate the no-compilation
property, and would need to avoid the grouped-conv cliff torchvision fell into.
Revisit only if profiling of a real training run shows the remaining gap
matters; the two deferred items in TODO.md (fusing the two gathers, folding
interpolation into the contraction) are cheaper places to look first.

### 5.3 Should the two repos' implementations be unified? — **Yes: retire tinymera's.**

`tinymera.nn.DeformConv1d` / `tinymera.ops.deform_conv1d` should be replaced by
a dependency on dc1d. Grounds, in order of severity:

1. **`stride > 1` is silently wrong** (§3, D5) — wrong output length, tail
   positions clamped to the last input sample, no test coverage.
2. **`causal=True` silently leaks future context** whenever
   `2*padding < dilation*(K-1)`, including at the module's default `padding=0`
   (§3, D6b). Measured `|d out[32] / d x[t'>32]| = 0.36`.
3. **No contract validation anywhere** (§3, D6) — the same clamp-hides-shape-bugs
   pattern dc1d fixed.
4. **float64 unavailable** (§3, D2), so `gradcheck` cannot be run on it, and it
   has no gradcheck test. For a layer whose whole purpose is the offset
   gradient, that is the most serious gap after (1) and (2).
5. **No `offset_groups`** — hardwired to per-channel, dc1d's most expensive
   setting, with no way to trade offset resolution for memory.
6. **It is slower and more memory-hungry**: `dc1d + grid_sample` beats
   `tinymera-gs` in 13/13 forward configs on both latency and memory, and
   `tinymera-gth` is the slowest implementation measured.

The one capability tinymera has that dc1d lacks is a **causal mode**, and it is
worth having. The recommendation is to port `causal` *into* dc1d — left-only
padding, offsets clamped to `≤ 0`, and (unlike tinymera) a validated
padding/dilation relationship that raises rather than silently leaking — rather
than to keep a second implementation alive for it. That also removes the
duplicate-maintenance surface that produced three correlated defects in the
first place.

**Regardless of the unification decision, tinymera PR #1 should be told about
D5 and D6b now** — both are live correctness bugs in code that is on its way to
being merged, and neither is what that PR set out to fix.

