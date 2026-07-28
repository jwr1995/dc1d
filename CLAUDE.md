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

  **The same allocation was still there on the backward side, unnoticed until 2026-07.** `take_along_dim` broadcasts for the forward, but its *backward* saves the broadcast index **materialised** — 8 bytes per output element, twice. Measured (`BACKENDS.md` §5.9.4): the default backward holds **7.02× the output tensor** between forward and backward (output + `x0` + `x1` + two full-size int64 indices); `efficient_linterpolate(..., gather_lerp='recompute')` holds **1.03×**. When looking for memory, look at what autograd saves, not only at what the forward allocates.
- **No state mutation in `forward`.** Device is inferred from the input. `dilated_positions` is a non-persistent buffer, not a plain attribute. Together these give 0 graph breaks under `torch._dynamo.explain`. Check `explain` after touching `forward`.

  Compiling is worth **2.3–9.1× forward** on CUDA (`benchmarks/BACKENDS.md` §5), and the *interpolation* forward stays **bit-exact against eager** at fp32 and fp64. The whole layer does not: Inductor reassociates the grouped `F.conv1d`, so with `groups > 1` the `nn.Conv1d` invariant comes out **2 ulp of float64** off and `tests/test_equivalence.py` would fail as written under `torch.compile`. That is a reassociated sum, not a mis-indexed gather — but do not weaken the test to accommodate it. Do not use `mode="max-autotune"`: measured worse than `mode="default"` in 8/10 configurations, once slower than eager, and not reproducible run to run (§5.4).
- **`^` is XOR.** `2^7` is 5. A benchmark silently tested `dilation=5` for three years.

## Benchmarking

Two separate sets of numbers exist; do not confuse them.

- **`benchmarks/benchmark.py`** compares old-dc1d against new-dc1d, and has only ever run on **CPU** — the default dev environment is CPU-only torch. Its figures (2.8–3.4× forward, 3.5–4.4× fwd+bwd, 283→91 MiB peak RSS) are CPU wall-clock and CPU RSS. The GPU equivalent of *that* comparison is still unmeasured.
- **`benchmarks/BACKENDS.md`** compares dc1d against three other backends on **CUDA** (3090), eager and compiled, with peak `max_memory_allocated`. Those are real GPU numbers.

Profiled launch counts, since the estimate was wrong and is still quoted in older commit messages: the rewrite moved the interpolation from **26 to 23** kernels — a 12% cut, not the ~5× that was projected. Its win was memory and correctness. `torch.compile` is what actually delivers the reduction (forward 27→4; backward only 66→31).

**The custom autograd `Function` has been written and measured** (`BACKENDS.md` §5.9), so do not re-propose it as the fix for the backward. It does **not** close the latency gap to `grid_sample`: the compiled backward launches **30** kernels whether the backward is hand-written or derived by autograd, because AOTAutograd's partitioner already reaches that schedule from the generic graph. Eager it is a wash or 10–23% slower. What it buys is **memory** — tape 7.0× the output → 1.03×, peak fwd+bwd 1.6–2.1× lower eager and 1.3–1.9× lower compiled. It ships as the opt-in `efficient_linterpolate(..., gather_lerp='recompute')`; the default is unchanged.

Two things a custom `autograd.Function` costs, both found by testing and both easy to reintroduce: **`vmap`/`torch.func` break** unless the Function declares `setup_context` *and* `generate_vmap_rule`, and **double backward is silently wrong** if the Function saves an intermediate rather than an input (`save-diff` returned a zero second-order term until it was made to raise).

**Timing protocol.** `bench_compile_config` reverses the round-robin on alternate rounds. Plain round-robin controls for drift between rounds but not within one, so whatever is measured last is penalised in *every* round — and min-across-rounds cannot remove that. Do not "simplify" it back.

Dynamic shapes are unavailable: `mark_dynamic` on the length axis raises `ConstraintViolationError` because the graph specialises on `L` (`dc1d/ops.py:184`). `dynamic=True` does not raise only because it specialises silently — one graph per distinct length, plus a slower steady state for asking. For variable-length speech that is roughly 0.3 s of compile per new length.

Always `torch.cuda.synchronize()` around GPU timing, or use `torch.utils.benchmark`. Give both sides of any comparison the same warmup — an earlier benchmark gave the deformable path three warmup iterations against a vanilla conv's one cold call, including cuDNN algorithm selection.

## API notes

`self.device` does not exist on either layer. Use `next(layer.parameters()).device`.

`padding` accepts `int | str`; `padding_mode` defaults to `'reflect'`, unlike `nn.Conv1d`'s `'zeros'`. Reflect padding requires `pad < L`, which a deep TCN can violate — this is a known sharp edge documented in the source.
