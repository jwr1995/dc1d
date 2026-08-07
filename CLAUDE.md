# dc1d

Pure-PyTorch 1D deformable convolution (Dai et al. 2017): DCNv1 by default, with opt-in DCNv2 modulation (Zhu et al. 2019) since 0.2.0. Backs the ICASSP 2023 paper *Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation*.

The selling point is that it needs **no C++/CUDA compilation**: everything rides on `torch.autograd`. Preserve that property. Any proposal that requires a build step (custom CUDA, Triton kernels shipped as source) must be justified against it explicitly.

## Conventions

**Commits: do not add a `Co-Authored-By` trailer.**

**Writing style.** Do not use em dashes. Use commas, colons, semicolons, parentheses, or a new sentence instead. Use emoji sparingly, and only where they carry information (a status column, for example), never as decoration. Prefer plain ASCII where it costs nothing: hyphens for numeric ranges, `x` or the word "times" rather than a multiplication sign in prose. Formatting should prioritise readability and portability, so favour short paragraphs, plain tables, and fenced code blocks over anything that depends on a particular renderer.

**Pull request descriptions: short, and mostly bullets.** A reviewer should get the whole picture in under a minute.

- Lead with one or two sentences on what changed and why it matters. No preamble, no throat-clearing headers.
- Then bullets. Use prose only where a bullet genuinely cannot carry the idea.
- One bullet per claim, each with its evidence attached: the measurement, the file:line, or the test that proves it. A claim with no evidence should either get some or be cut.
- A table beats a paragraph for anything with more than two dimensions (before/after, config/result, bug/symptom).
- State known limitations and anything deliberately not done. That section is not optional, and it is often the most useful part.
- Do not restate the diff. The reviewer can read it. Explain what is not obvious from reading it.
- Do not editorialise about urgency or importance. Give the facts and let them speak.

The same applies to commit messages: a short imperative subject, then bullets or short paragraphs explaining why rather than what.

**Delegate chores and benchmarking to subagents.** Routine mechanical work (dependency bumps, lint sweeps, formatting, docstring fixes, test scaffolding, file moves) and *all* benchmarking should be dispatched to a subagent rather than done inline. Benchmarking in particular is long-running and produces a lot of output that is not worth carrying in the main context; a subagent should run the sweep and report the table and the conclusion.

Use `uv` for everything: `uv run pytest`, `uv run ruff check`, `uv sync`. Do not call `pip` or a bare `python`. The dev environment pins Python 3.12 and CPU-only torch; the published wheel requires `torch>=2.7` on any supported Python (raised from 2.4 in 0.2.0: 2.7 is the oldest release the ONNX export path is verified against).

Before pushing: `uv run ruff check && uv run ruff format --check && uv run pytest`. CI lints once and runs `pytest` across 3.10 to 3.13, plus a separate ONNX-export job.

## Layout

- `dc1d/ops.py`: interpolation kernels. `efficient_linterpolate` is the default and the only one that matters; `kernel_width_linterpolate` and `full_seq_linterpolate` are kept for reference and are memory-explosive by design.
- `dc1d/nn.py`: `DeformConv1d` (caller supplies offsets) and `PackedDeformConv1d` (predicts offsets internally via depthwise, PReLU, gLN, pointwise, PReLU, gLN, plus a parallel mask branch when `modulated=True`).
- `tests/`: see below; these are load-bearing.
- `benchmarks/benchmark.py`: `torch.utils.benchmark.Timer`, forward and forward+backward timed separately.
- `docs/demo.ipynb`: the visual walkthrough, committed **with outputs** so it renders on GitHub, plus `docs/demo/*.png` which the notebook writes itself. Every property it shows is `assert`ed, so it is documentation and a slow test at once. It needs the `demo` dependency group, is CPU-only, and must stay under ten seconds. After any change that alters its numbers or figures, re-run `uv run --group demo jupyter nbconvert --to notebook --execute --inplace docs/demo.ipynb` and commit the regenerated outputs and PNGs. Ruff lints and formats notebook cells, so run `uv run ruff format docs/demo.ipynb` before re-executing, not after.
- `TODO.md`: tracks outstanding work. Keep it current; tick items as they land and leave unticked items annotated with what remains.

## The invariant that matters

**With zero offsets in constrained mode, `DeformConv1d` must reproduce `nn.Conv1d` bit-for-bit** given the same weights. Sample positions land on exact integers, so the interpolation weights are exactly `[1,0]` / `[0,1]`.

`tests/test_equivalence.py` asserts this across stride, dilation and groups, and it is the single test that pins down indexing, dilation, stride, groups and the output contraction simultaneously. **Never weaken it.** Any change to `efficient_linterpolate` must keep it passing bit-exactly. That is what made the 3x kernel rewrite safe to attempt.

`tests/test_gradients.py` runs `gradcheck` in float64 against both `input` and `offsets`. The gradient w.r.t. offsets is the entire point of a deformable layer; if you touch the interpolation maths, this is the test that catches you.

## Things that have bitten before

These are all fixed. They are listed because the same classes of bug are easy to reintroduce.

- **Precision and position arithmetic.** Window starts are kept in `long`; only the sub-sample fraction carries the low-precision dtype. Never derive positions with a `linspace` or `arange` that inherits `offsets.dtype`: fp16 cannot represent integers past 2048, and the failure is silent (wrong numbers, no NaN).
- **The index clamp hides shape bugs.** `efficient_linterpolate` clamps out-of-range positions, so a wrong `L_out` produces plausible wrong-length output rather than an exception. `DeformConv1d.forward` therefore validates `offsets.shape[-2]` against input length, stride, dilation and padding. Keep that check.
- **`torch.gather` does not broadcast; `take_along_dim` does.** What matters is that the index is never *tiled* to the channel count. `take_along_dim` gets that by broadcasting; `gather` gets it from an explicit `expand`, which is a stride-0 view that the strided iterator reads for free. Both are fine. Materialising a channel-sized int64 index is what is not. Note the ordering trap in `_gather_pair`: `(idx + 1).expand(...)`, never `idx.expand(...) + 1`, or the arithmetic materialises the thing the expand was avoiding.

  **`take_along_dim` is nevertheless banned here, because it does not survive ONNX export from torch 2.10 onwards.** It decomposes to a negative-index wrap, `index % self.size(dim)`, and the exporter constant-folds that modulus against the *export-time* length: a model exported at `L = 200` gets a literal `Mod(index, 200)` in its graph. Bisected across CPU wheels: **clean on 2.7, 2.8, 2.9; broken on 2.10, 2.11, 2.12, 2.13**. It is a torch regression, not a property of the operator, so do not restore `take_along_dim` on the grounds that some older torch exports it correctly. Longer inputs then wrap around and read the wrong samples, at full signal magnitude, with the correct output shape and no error raised (at the export length the broken build is still correct to the ordinary fp32 export tolerance, ~2e-07; at `T = 300` it is 3.4e+00 on torch 2.13. Per-version table: `TODO.md` E8). A shape assertion cannot see it, and neither can a numerical test at a single length, because the convenient length to test at is the export length, which is the one length where it is invisible. `tests/test_export.py` therefore sweeps lengths on both sides of the export length and separately asserts that no constant-divisor `Mod` appears in the graph at all. `gather` accepts no negative indices, emits no `Mod`, and is bit-identical in eager.

  **A separate allocation was still there on the backward side, unnoticed until 2026-07.** `take_along_dim` broadcasts for the forward, but its *backward* saved the broadcast index **materialised**, 8 bytes per output element, twice: the default backward held **7.02x the output tensor** between forward and backward, against **1.03x** for `gather_lerp='recompute'`. When looking for memory, look at what autograd saves, not only at what the forward allocates.

  **The 2026-08 gather rewrite fixed that too, as a side effect** (`BACKENDS.md` section 5.9.4a, A100). Handing `gather` an explicit `expand` keeps the index a stride-0 view all the way onto the tape: saved index stride `(3066, 3066, 0, 1)` at 0.094 MiB, against `take_along_dim`'s `(196224, 196224, 3066, 1)` at 5.988 MiB. The default now holds **3.0x to 3.4x** the output rather than 7.02x, and peak fwd+bwd drops **1.5x to 1.6x** eager and **2.2x to 2.5x** compiled. **This only applies when `offset_groups < channels`.** With depthwise offsets (`offset_groups == channels`) the expand is an identity, the tape is unchanged, and only a forward transient improves (1.24x eager, nothing compiled). Consequence for `recompute`: its memory advantage over the default is now **1.30x**, not the 1.6x-2.1x that section 5.9.4 and the `gather_lerp` docstring were written against, *except* at `offset_groups == channels` where it keeps full value. Do not quote the old ratios.
- **No state mutation in `forward`.** Device is inferred from the input. `dilated_positions` is a non-persistent buffer, not a plain attribute. Together these give 0 graph breaks under `torch._dynamo.explain`. Check `explain` after touching `forward`.

  Compiling is worth **2.3x to 9.1x forward** on CUDA (`benchmarks/BACKENDS.md` section 5), and the *interpolation* forward stays **bit-exact against eager** at fp32 and fp64. The whole layer does not: Inductor reassociates the grouped `F.conv1d`, so with `groups > 1` the `nn.Conv1d` invariant comes out **2 ulp of float64** off and `tests/test_equivalence.py` would fail as written under `torch.compile`. That is a reassociated sum, not a mis-indexed gather, but do not weaken the test to accommodate it. Do not use `mode="max-autotune"`: measured worse than `mode="default"` in 8 of 10 configurations, once slower than eager, and not reproducible run to run (section 5.4).
- **`^` is XOR.** `2^7` is 5. A benchmark silently tested `dilation=5` for three years.

## Benchmarking

Two separate sets of numbers exist; do not confuse them.

- **`benchmarks/benchmark.py`** compares dc1d against `nn.Conv1d`, forward and forward+backward, and prints peak memory only on CUDA. It has only ever been *run* on CPU, because the default dev environment is CPU-only torch. It does **not** measure old-dc1d against new-dc1d: those figures (2.8x to 3.4x forward, 3.5x to 4.4x fwd+bwd, 283 to 91 MiB peak RSS, CPU wall-clock and CPU RSS) are in `TODO.md` under Benchmarks, and came from an ad-hoc harness that is not in the tree, so they cannot be reproduced from a clean clone. The GPU equivalent of that comparison is `BACKENDS.md` section 7 (A100, both versions from PyPI against the same torch build), with raw data in `benchmarks/results/` and figures regenerated by `benchmarks/plot_version_comparison.py`.
- **`benchmarks/BACKENDS.md`** compares dc1d against three other backends on **CUDA** (3090), eager and compiled, with peak `max_memory_allocated`. Those are real GPU numbers.

Profiled launch counts, since the estimate was wrong and is still quoted in older commit messages: the rewrite moved the interpolation from **26 to 23** kernels, a 12% cut, not the 5x that was projected. Its win was memory and correctness. `torch.compile` is what actually delivers the reduction (forward 27 to 4; backward only 66 to 31).

**The custom autograd `Function` has been written and measured** (`BACKENDS.md` section 5.9), so do not re-propose it as the fix for the backward. It does **not** close the latency gap to `grid_sample`: the compiled backward launches **30** kernels whether the backward is hand-written or derived by autograd, because AOTAutograd's partitioner already reaches that schedule from the generic graph. Eager it is a wash or 10% to 23% slower. What it bought was **memory**: tape 7.0x the output down to 1.03x, peak fwd+bwd 1.6x to 2.1x lower eager and 1.3x to 1.9x lower compiled. It ships as the opt-in `efficient_linterpolate(..., gather_lerp='recompute')`; the default is unchanged. **Those ratios were against the `take_along_dim` default and are now smaller** (section 5.9.4a): the gather rewrite took most of that saving for free wherever `offset_groups < channels`. `recompute` is still the right answer at `offset_groups == channels`, which is the depthwise-offset case a DTCN actually uses.

Two things a custom `autograd.Function` costs, both found by testing and both easy to reintroduce: **`vmap` and `torch.func` break** unless the Function declares `setup_context` *and* `generate_vmap_rule`, and **double backward is silently wrong** if the Function saves an intermediate rather than an input (`save-diff` returned a zero second-order term until it was made to raise).

**Timing protocol.** `bench_compile_config` reverses the round-robin on alternate rounds. Plain round-robin controls for drift between rounds but not within one, so whatever is measured last is penalised in *every* round, and min-across-rounds cannot remove that. Do not "simplify" it back.

Dynamic shapes are unavailable: `mark_dynamic` on the length axis raises `ConstraintViolationError` because the graph specialises on `L` at the `out_length * kernel_size` reshape in `efficient_linterpolate`. `dynamic=True` does not raise only because it specialises silently, giving one graph per distinct length plus a slower steady state for asking. For variable-length speech that is roughly 0.3 s of compile per new length.

Always `torch.cuda.synchronize()` around GPU timing, or use `torch.utils.benchmark`. Give both sides of any comparison the same warmup. An earlier benchmark gave the deformable path three warmup iterations against a vanilla conv's one cold call, including cuDNN algorithm selection.

## API notes

`self.device` does not exist on either layer. Use `next(layer.parameters()).device`.

`padding` accepts `int | str`; `padding_mode` defaults to `'reflect'`, unlike `nn.Conv1d`'s `'zeros'`. Reflect padding requires `pad < L`, which a deep TCN can violate. This is a known sharp edge documented in the source.
