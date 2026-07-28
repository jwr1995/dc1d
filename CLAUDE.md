# dc1d

Pure-PyTorch 1D deformable convolution (Dai et al. 2017, DCNv1 — no modulation mask). Backs the ICASSP 2023 paper *Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation*.

The selling point is that it needs **no C++/CUDA compilation** — everything rides on `torch.autograd`. Preserve that property. Any proposal that requires a build step (custom CUDA, Triton kernels shipped as source) must be justified against it explicitly.

## Conventions

**Commits: do not add a `Co-Authored-By` trailer.**

**Delegate chores and benchmarking to subagents.** Routine mechanical work — dependency bumps, lint sweeps, formatting, docstring fixes, test scaffolding, file moves — and *all* benchmarking should be dispatched to a subagent rather than done inline. Benchmarking in particular is long-running and produces a lot of output that is not worth carrying in the main context; a subagent should run the sweep and report the table and the conclusion.

Use `uv` for everything — `uv run pytest`, `uv run ruff check`, `uv sync`. Do not call `pip` or a bare `python`. The dev environment pins Python 3.12 and CPU-only torch; the published wheel requires `torch>=2.4` on any supported Python.

Before pushing: `uv run ruff check && uv run ruff format --check && uv run pytest`. CI runs the same across 3.10–3.13.

## Layout

- `dc1d/ops.py` — interpolation kernels. `efficient_linterpolate` is the default and the only one that matters; `kernel_width_linterpolate` and `full_seq_linterpolate` are kept for reference and are memory-explosive by design.
- `dc1d/nn.py` — `DeformConv1d` (caller supplies offsets) and `PackedDeformConv1d` (predicts offsets internally via depthwise→gLN→PReLU→pointwise→gLN→PReLU).
- `tests/` — see below; these are load-bearing.
- `benchmarks/benchmark.py` — `torch.utils.benchmark.Timer`, forward and forward+backward timed separately.
- `TODO.md` — tracks outstanding work. Keep it current; tick items as they land and leave unticked items annotated with what remains.

## The invariant that matters

**With zero offsets in constrained mode, `DeformConv1d` must reproduce `nn.Conv1d` bit-for-bit** given the same weights. Sample positions land on exact integers, so the interpolation weights are exactly `[1,0]`/`[0,1]`.

`tests/test_equivalence.py` asserts this across stride, dilation and groups, and it is the single test that pins down indexing, dilation, stride, groups and the output contraction simultaneously. **Never weaken it.** Any change to `efficient_linterpolate` must keep it passing bit-exactly — that is what made the 3× kernel rewrite safe to attempt.

`tests/test_gradients.py` runs `gradcheck` in float64 against both `input` and `offsets`. The gradient w.r.t. offsets is the entire point of a deformable layer; if you touch the interpolation maths, this is the test that catches you.

## Things that have bitten before

These are all fixed. They are listed because the same classes of bug are easy to reintroduce.

- **Precision and position arithmetic.** Window starts are kept in `long`; only the sub-sample fraction carries the low-precision dtype. Never derive positions with a `linspace`/`arange` that inherits `offsets.dtype` — fp16 cannot represent integers past 2048, and the failure is silent (wrong numbers, no NaN).
- **The index clamp hides shape bugs.** `efficient_linterpolate` clamps out-of-range positions, so a wrong `L_out` produces plausible wrong-length output rather than an exception. `DeformConv1d.forward` therefore validates `offsets.shape[-2]` against input length, stride, dilation and padding. Keep that check.
- **`torch.gather` does not broadcast; `take_along_dim` does.** The current kernel relies on this — it is why there is no longer a channel-sized int64 index tensor. Reintroducing `gather` reintroduces a 4×-output-size allocation.
- **No state mutation in `forward`.** Device is inferred from the input. `dilated_positions` is a non-persistent buffer, not a plain attribute. Together these give 0 graph breaks under `torch._dynamo.explain`. Check `explain` after touching `forward`.

  Compiling is worth **2.3–9.1× forward** on CUDA (`benchmarks/BACKENDS.md` §5), and the *interpolation* forward stays **bit-exact against eager** at fp32 and fp64. The whole layer does not: Inductor reassociates the grouped `F.conv1d`, so with `groups > 1` the `nn.Conv1d` invariant comes out **2 ulp of float64** off and `tests/test_equivalence.py` would fail as written under `torch.compile`. That is a reassociated sum, not a mis-indexed gather — but do not weaken the test to accommodate it. Do not use `mode="max-autotune"`: measured worse than `mode="default"` in 8/10 configurations, once slower than eager, and not reproducible run to run (§5.4).
- **`^` is XOR.** `2^7` is 5. A benchmark silently tested `dilation=5` for three years.

## Benchmarking

GPU numbers are **not** currently measured — the dev environment is CPU-only torch. The committed figures (2.8–3.4× forward, 3.5–4.4× fwd+bwd, 283→91 MiB peak RSS) are CPU wall-clock and CPU RSS. Do not quote GPU speedups until the benchmarks are run on CUDA; TODO.md tracks this.

Always `torch.cuda.synchronize()` around GPU timing, or use `torch.utils.benchmark`. Give both sides of any comparison the same warmup — an earlier benchmark gave the deformable path three warmup iterations against a vanilla conv's one cold call, including cuDNN algorithm selection.

## API notes

`self.device` does not exist on either layer. Use `next(layer.parameters()).device`.

`padding` accepts `int | str`; `padding_mode` defaults to `'reflect'`, unlike `nn.Conv1d`'s `'zeros'`. Reflect padding requires `pad < L`, which a deep TCN can violate — this is a known sharp edge documented in the source.
