# dc1d modernisation TODO

Tracking document for the 2026-07 modernisation pass. Everything below was left
in the working tree — nothing has been committed, branched or pushed. Review
with `git diff` / `git status`.

Environment used for verification: `uv` 0.11.32, CPython 3.12.13,
`torch 2.13.0+cpu`. Full suite also run green on 3.10.20, 3.11 and 3.13.14.

---

## Packaging

- [x] Migrate to a `uv`-managed project: `pyproject.toml` + `uv.lock` + `.python-version` (3.12).
- [x] **Add the missing `dependencies` field.** `pyproject.toml` had none at all, so
      `pip install dc1d` produced a package that `ImportError`ed on first use.
      Now `torch>=2.4` (`pyproject.toml:15-17`).
- [x] Do **not** add `torchvision` — the only consumer was the dead `deform_conv1d`
      (see Correctness / C8 below), which is now deleted. dc1d is torchvision-free.
- [x] `requires-python` `>=3.8` → `>=3.10`; classifiers updated for 3.10–3.13.
- [x] Dev dependency group with `pytest` + `ruff` (`pyproject.toml:[dependency-groups]`).
- [x] CPU-only torch for the dev venv via `[tool.uv.sources]` + `[[tool.uv.index]] pytorch-cpu`
      (explicit index, so only `torch` is pulled from it). The published wheel keeps
      a plain `torch>=2.4`, so end users still get whatever build pip resolves.
- [x] `__version__` added to `dc1d/__init__.py:15`; hatchling reads it via
      `[tool.hatch.version] path = "dc1d/__init__.py"`. Bumped `0.0.7` → **`0.1.0`**
      for the breaking changes in this pass (see Release below).
- [x] `dc1d/__init__.py` now re-exports `DeformConv1d`, `PackedDeformConv1d`,
      `gLN`, `cLN` and the three interpolation kernels. It previously exported nothing.
- [x] `[tool.ruff]` config added (line-length 100, `E/F/I/UP/B/W`). `ruff check .`
      and `ruff format --check .` both clean.
- [x] **`.gitignore` no longer ends with `**/*.txt`.** That rule silently swallowed
      any `requirements.txt`. Rewritten with a comment warning against re-adding it.
- [ ] **`py.typed` marker — NOT added.** The annotations on the public API are now
      correct (see C-anno below), but they are incomplete: `gLN.__init__`,
      `cLN.__init__`, `gLN.forward`, `cLN.forward` and
      `PackedDeformConv1d.forward` (`dc1d/nn.py:360`) are unannotated, and
      `interpolation_function: Callable` is unparameterised. No type checker has
      been run against the package. **Left:** add `mypy` (or `ty`/`pyright`) to the
      dev group, get a clean run, then add `dc1d/py.typed` and
      `[tool.hatch.build.targets.wheel] force-include`.

## Correctness

Every item below was **reproduced against the pre-change code** (extracted with
`git show HEAD:...` into a scratch tree, with a `torchvision` stub) before being
fixed. Findings that differ from the original review notes are called out.

- [x] **C1 — `from turtle import forward` deleted** (was `dc1d/nn.py:21`).
      Reproduced: `PYTHONPATH=<orig> /usr/bin/python3 -c "import dc1d.nn"` →
      `ImportError: No module named 'tkinter', please install the python3-tk package`.
      *Nuance:* this does **not** reproduce inside the uv-managed CPython, because
      `python-build-standalone` bundles tkinter. It breaks on the system python
      (3.14) and on any distro python without `python3-tk`. The import was unused
      either way.
- [x] **C2 — `extra_repr` rewritten** (`dc1d/nn.py:171-186`). Reproduced:
      `repr(model)` → `TypeError: object of type 'int' has no len()`. The old body
      was copy-pasted from `_ConvNd`, which stores padding/dilation as tuples, and
      referenced `self.output_padding`, which this class never assigns. The new
      version reads the actual scalar attributes and also surfaces `unconstrained`.
- [x] **C3 — low-precision sampling positions fixed** (`dc1d/ops.py:130-162`).
      Reproduced: with fp16 offsets and `L=16000`, **13952 of 15998** output rows
      were wrong, first divergence at row **2046**, max abs error 7.22, no NaN.
      `torch.linspace(0, 15997, 15998, dtype=float16)` yields only 5073 distinct
      values and saturates at 16000. Fixed by decomposition: the integer window
      start is `torch.arange(Lo) * stride` in `long` (`dc1d/ops.py:140`), the
      dilated tap offsets are `long` (`_dilated_positions_long`, `dc1d/ops.py:47`),
      and only the sub-sample fraction is carried in the offset dtype
      (`dc1d/ops.py:149`). Verified: bf16 offsets (8 mantissa bits, integers exact
      only to 256) still gather exactly at `L=8192`.
- [x] **C4 — `PackedDeformConv1d` offset conv now forwards `stride` and `dilation`**
      (`dc1d/nn.py:337-348`). Reproduced: with `stride=2, L=40` the original
      returned length **38** instead of 19; with `dilation=3` it returned **38**
      instead of 34, silently. The index clamp absorbed the mismatch.
- [x] **C4b — explicit offset-shape assertion in `DeformConv1d.forward`**
      (`dc1d/nn.py:236-243`), raising `ValueError` against the closed form
      `output_length(padded_len, K, dilation, stride)` (`dc1d/ops.py:21`).
      Also exposed as `DeformConv1d.expected_offset_positions()` (`dc1d/nn.py:202`)
      so callers can size their offset tensors without duplicating the formula.
- [x] **C5 — `offset_groups` strictly between 1 and C** (was `dc1d/ops.py:227-228`,
      `U.repeat` tiling to `G*C` channels). Reproduced: `offset_groups=4, C=8` →
      `RuntimeError: Size does not match at dimension 1 expected index
      [2, 32, 38, 2] to be no larger than self [2, 8, 40, 2]`. The tiled index
      tensor no longer exists at all — `take_along_dim` broadcasts, and the channel
      axis is viewed as `(groups, channels_per_group)` (`dc1d/ops.py:180-193`),
      which serves every group ratio including `1 < G < C`.
      `PackedDeformConv1d` was also relaxed from `offset_groups in {1, in_channels}`
      to any divisor of `in_channels` (`dc1d/nn.py:312-316`), matching what the
      docstring at the old `nn.py:179` had always claimed.
- [x] **C6 — unconstrained clamp `x.shape[-1]` → `x.shape[-1]-1`.** Reproduced, and
      it is **worse than the review described**: at exactly `T == L` *both* bilinear
      weights fall to zero, so the output is exactly **0.0**, not merely attenuated.
      Handled on the split representation at `dc1d/ops.py:151-162` (clamp the
      integer index to `[0, L-2]` and force the fraction to 0/1 outside the range),
      which reproduces the correct clamp gradient without ever forming a
      large-magnitude float. Guarded by `test_interpolation_weights_sum_to_one_at_right_edge`.

### Cleanup

- [x] **C8 — `deform_conv1d` deleted** (was `dc1d/ops.py:312-407`). Confirmed dead
      (imported at the old `nn.py:34`, never called) *and* broken: reproduced
      `ValueError: not enough values to unpack (expected 4, got 3)` at the old
      `ops.py:380`. Its removal is what makes dc1d torchvision-free — the old
      `ops.py:15-16` imported the **private** `torchvision.extension._assert_has_ops`
      and `torchvision.utils._log_api_usage_once` at module scope.
- [x] **`_jit_efficient_linterpolate` deleted** (was `dc1d/ops.py:245-309`). Never
      exported or called, a divergent copy of the main kernel, and the
      `@torch.jit.script` decorator compiled it eagerly at import — so any
      TorchScript regression became a package-wide import failure.
- [x] **`torch.multiprocessing` branch deleted** (was `dc1d/ops.py:167-178`), along
      with its `_interpolate` helper and the `multiprocessing.dummy` / `functools`
      imports. It wrote into a preallocated tensor in place and could not
      participate in autograd. The remaining `_max_memory=False` loop path was
      rewritten to `torch.stack` a list instead of in-place writes, so it is now
      differentiable too (`dc1d/ops.py:350-360`).
- [x] **`full_seq_linterpolate` `_test` NameError fixed** (was `dc1d/ops.py:81,83`).
      Reproduced: `NameError: name 'batch_size' is not defined`. Now uses `x.shape[0]`.
- [x] **XOR bug fixed.** `dilation = 2^7` is XOR and evaluates to **5** — confirmed.
      The "large dilation" benchmark had been running dilation=5 since 2022.
      Fixed at `dc1d/nn.py:487` (`2**7`) and `playground/param_example.py:22`
      (`2**5`, with the channel count reduced so it stays runnable on CPU).
      Both sites carry an explanatory comment.
- [x] `if dilated_positions == None` → `is None` (was `ops.py:45,111,196`).
- [x] **`forward` no longer mutates `self.device`** (was `dc1d/nn.py:201-204` and
      `nn.py:314`). The device now comes from the input tensor, and
      `dilated_positions` is a non-persistent buffer (`dc1d/nn.py:148-152`), so
      `.to()`/`.cuda()` move it automatically. The `device=` constructor kwarg is
      kept for back-compat but only triggers a `self.to(device)` at the end of
      `__init__`. **The `self.device` attribute no longer exists** — this is the one
      deliberate API removal; downstream code reading `layer.device` will need
      `next(layer.parameters()).device`.
      Verified: `torch._dynamo.explain` reports **0 graph breaks / 1 graph**, and
      `torch.compile(model, fullgraph=True, backend="aot_eager")` matches eager
      bit-for-bit and still produces offset gradients.
- [x] **`unconstrained` is now a plain bool set unconditionally** (`dc1d/nn.py:128`).
      The `if "unconstrained" in self.__dict__.keys()` dispatch (old `nn.py:206`,
      driven by `nn.py:132`) is gone. `__setstate__` backfills it for old pickles.
- [x] **User-input asserts converted to exceptions** (old `ops.py:108,193`;
      `nn.py:235,271,316,317`). All validation is now `ValueError` /
      `RuntimeError` / `NotImplementedError` and survives `python -O`. The only
      remaining `assert`s are in tests and example scripts.
- [x] **C-anno — type annotations fixed.** `padding: int = "valid"` → `int | str`
      (`dc1d/nn.py:51`, `nn.py:278`); `unconstrained: str = None` → `bool | None`
      (`dc1d/nn.py:58`, `nn.py:286`); `mask: Optional[Tensor]` → `Tensor | None`;
      `device: str` → `torch.device | str | None`. `from __future__ import annotations`
      added to both modules.
- [x] `torch.sum(..., axis=-1)` → `dim=` (was `ops.py:74,94,151,164,242`).
- [x] **Docstrings fixed.** `in_channels`/`out_channels` were documented as
      "convolution kernel size" / "kernel dilation factor" (old `nn.py:57-58`,
      `nn.py:258-259`); the `forward` return was documented as
      `[batch, in_channels, length]` (old `nn.py:183`) when it is
      `[batch, out_channels, output_length]`.
- [x] **`README.md` — `PackedConv1d` → `PackedDeformConv1d`** (old `README.md:92`),
      plus a real usage example for it. README also updated for Python/torch
      requirements, the removed torchvision dependency, uv dev workflow, and the
      benchmark caveat.
- [x] Bonus: int `padding` with `padding_mode="zeros"` previously applied **no
      padding at all** (the old `forward` only handled the `'same'` string in that
      branch). Fixed in `DeformConv1d._pad` (`dc1d/nn.py:195-201`).
- [x] Bonus: `gLN`/`cLN` used the deprecated `torch.Tensor(1, 1, n)` constructor;
      now `torch.empty`.

## Tests

There were **zero** tests. `tests/` now holds **305 passing / 72 skipped**
(the skips are all `padding='same'` × `stride>1`, which is undefined).

- [x] **1. Zero-offset equivalence** — `tests/test_equivalence.py:27`
      (`test_zero_offset_matches_conv1d`), parametrised over
      stride ∈ {1,2,3} × dilation ∈ {1,2,4} × groups ∈ {1,2,8} × kernel_size ∈ {1,3,5},
      asserting `torch.equal` against `F.conv1d`.
      *Caveat, deliberate:* for `kernel_size=1 and stride>1` the assertion is
      relaxed to `assert_close(atol=1e-6, rtol=0)`. A strided 1×1 grouped conv and
      a dense stride-1 grouped conv dispatch to different oneDNN kernels, so the
      accumulation order differs by ~1e-7. The **interpolation** is still asserted
      bit-exact for that case by `test_zero_offset_gather_is_exact`
      (`tests/test_equivalence.py:67`), which compares against `Tensor.unfold`.
      *Honest correction to the review's rationale:* the original `DeformConv1d`
      **did** pass zero-offset equivalence bit-for-bit at every stride/dilation/groups
      combination tried. This test would not by itself have caught the stride/dilation
      bug — that bug lives in `PackedDeformConv1d`'s offset conv, and the guard for
      it is the shape-contract test (4) below.
- [x] **2. `torch.autograd.gradcheck` in float64** — `tests/test_gradients.py:26`
      (interpolation kernel, w.r.t. both `input` and `offsets`, constrained and
      unconstrained, offset_groups ∈ {1,2,4}) and `tests/test_gradients.py:48`
      (whole module, B=2 C=4 L=16 K=3, over stride/dilation). Offsets are drawn
      from [0.25, 0.75] so no sampling position lands on an integer, where the
      piecewise-linear interpolation has a kink.
- [x] **3. Integer-shift equivalence** — `tests/test_equivalence.py:127`
      (`test_integer_shift_equivalence`), shift ∈ {-3,-1,1,2} × dilation ∈ {1,2},
      compared on the interior. Plus `test_offset_groups_are_applied_per_group`
      (`tests/test_equivalence.py:101`), which gives each offset group a different
      integer shift and checks the right channels moved — the direct guard for C5.
- [x] **4. Shape contract** — `tests/test_shapes.py:30` and `:57`, parametrised over
      stride × dilation × padding ∈ {'same','valid'} × groups × offset_groups, for
      both `DeformConv1d` and `PackedDeformConv1d`. **This is the test that catches
      C4.** Plus `test_wrong_offset_count_raises` for the new loud failure.
- [x] **5. Import / repr regression** — `tests/test_smoke.py` (`test_package_imports`
      asserts `turtle` and `tkinter` are not in `sys.modules`;
      `test_no_torchvision_dependency`; `test_repr_does_not_raise`).
- [x] Extra guards beyond the brief: `test_half_offsets_on_long_sequence_are_exact`
      (C3), `test_interpolation_weights_sum_to_one_at_right_edge` + `test_left_edge_clamp`
      (C6), `test_dilated_positions_is_a_buffer`,
      `test_forward_does_not_mutate_module_state`, `test_state_dict_roundtrip`,
      `test_index_tensor_carries_no_gradient`, and
      `test_legacy_kernels_agree_and_are_differentiable`
      (`tests/test_equivalence.py:217`), which covers the two legacy interpolation
      kernels that are still exported.
- [ ] **CUDA tests — not written.** Only a CPU-only torch is installed here, so a
      GPU test would be dead code that has never executed. **Left:** add a
      `@pytest.mark.skipif(not torch.cuda.is_available())` module that runs the
      equivalence + gradcheck suites on `cuda`, and asserts CPU/CUDA parity to
      within fp32 tolerance. Owner has a 3090 + 2080 Ti and can validate.
- [ ] **Autocast test — not written.** `efficient_linterpolate` was checked by hand
      under `torch.autocast('cpu', bfloat16)` and with bf16 offsets at L=8192
      (exact), but there is no test asserting it. **Left:** a parametrised
      fp16/bf16 test, ideally on CUDA where fp16 autocast is the realistic case.

## CI

- [x] `.github/workflows/ci.yml`, on push (all branches) + PR + manual.
- [x] `lint` job: `uv sync --locked`, `ruff check .`, `ruff format --check .`.
- [x] `test` job: matrix over Python 3.10 / 3.11 / 3.12 / 3.13 on `ubuntu-latest`,
      `uv sync --locked`, `uv run pytest -q`. CPU-only torch comes from the
      `pytorch-cpu` index declared in `pyproject.toml` (~180 MB instead of ~2.5 GB).
- [x] `astral-sh/setup-uv@v5` with `enable-cache: true` and
      `cache-dependency-glob: "uv.lock"`.
- [x] `build` job: `uv build`, `uvx twine check dist/*`, then install the built
      wheel into a clean venv and import it — this is the job that would have
      caught the missing `dependencies` field.
- [ ] **CI has not been executed.** All four jobs' commands were run locally and
      pass on all four matrix versions (3.10 / 3.11 / 3.12 / 3.13), but the YAML
      itself has never run on GitHub. **Left:** push and confirm the first run,
      particularly that `setup-uv`'s `python-version` input plays well with the
      checked-in `.python-version` pin of 3.12.
- [ ] **No coverage reporting, no Windows/macOS runners, no CUDA runner.** Not asked
      for; noted as an obvious next step if the owner wants them.

## Performance

Landed **after** tests 1–4 were green, and re-verified green afterwards.

- [x] `efficient_linterpolate` rewritten (`dc1d/ops.py:65-194`) using
      `torch.take_along_dim` + `torch.lerp`. Gone: the `U.repeat` int64 tile, the
      `torch.stack([...gather...])` list comprehension, the `torch.zeros` memset,
      and the abs/max/multiply/sum chain. Since `U1 ≡ U0+1` by construction the
      two weights are exactly `1-f` and `f`.
- [x] **Verified bit-exact against the old implementation** for zero offsets at
      `(B=4,C=256,L=2048,K=3,d=8)`, `(16,64,512,5,1)` and `(1,512,16000,3,1)`.
- [x] **Measured speedup (CPU, single thread, `torch.utils.benchmark`):**

      | config                    | fwd old | fwd new | fwd+bwd old | fwd+bwd new |
      |---------------------------|--------:|--------:|------------:|------------:|
      | B=4 C=256 L=2048 K=3 d=8  |  422 ms |  139 ms |      789 ms |      194 ms |
      | B=16 C=64 L=512 K=5 d=1   |  167 ms |   59 ms |      273 ms |       78 ms |
      | B=1 C=512 L=16000 K=3 d=1 | 1696 ms |  502 ms |     3309 ms |      760 ms |

      → **2.8–3.4× forward, 3.5–4.4× forward+backward, on CPU.**
- [x] **Measured memory (CPU peak RSS delta, separate processes,
      B=4 C=256 L=2048 K=3 d=8, 23.8 MiB output):** old **283 MiB** → new **91 MiB**,
      i.e. **~3.1× lower**. Note this is coarser than a CUDA allocator measurement
      and includes allocator retention.
- [x] `dilated_positions` registered as a non-persistent buffer (`dc1d/nn.py:148`)
      instead of the plain attribute at the old `nn.py:122`. It was missing from
      `state_dict()` and ignored by `.cuda()`.
- [x] `torch.max(dilated_positions)` (old `ops.py:206`), a device reduction, replaced
      with the Python constant `dilation * (kernel_size - 1)` (`dc1d/ops.py:128`).
- [x] `L >= 2` guarded (`dc1d/ops.py:110`).
- [ ] **GPU speedup and kernel-launch counts are NOT measured.** Only CPU-only torch
      is installed. The review's "~25 kernels → ~5" and "4–8× / ~10× memory" figures
      are **estimates that this pass did not confirm**; the numbers reported above
      are CPU wall-clock and CPU RSS, and the op-count reduction is a static reading
      of the code, not a profile. **Left:** run `benchmarks/benchmark.py --device cuda`
      on the 3090 and, for launch counts, `torch.profiler` with
      `activities=[CUDA]` or `nsys`.
- [ ] **Deferred perf work, not attempted:**
      - [ ] Fuse the two `take_along_dim` gathers. `x1` is always `x0` shifted by one
            sample, so a single gather of a `(Lo, K, 2)` window — or a `Tensor.unfold`
            over the receptive field followed by one gather — should halve the gather
            traffic.
      - [ ] Fold the interpolation and the `F.conv1d` contraction together. The
            `(B, C, Lo, K)` intermediate is the dominant allocation and never needs to
            be materialised; a `torch.compile`-generated or hand-written fused kernel
            would remove it.
      - [ ] A custom autograd `Function` for the interpolation. The backward is
            currently derived by autograd through gather+lerp, which stores `x0`, `x1`
            and `frac`; an explicit backward only needs the integer index and `frac`.
      - [ ] `channels_last`/contiguity study — `x.reshape(B, G, C//G, L)` assumes a
            contiguous channel axis and will silently copy otherwise.
      - [ ] Benchmark `torch.compile(mode="max-autotune")` end-to-end. Dynamo already
            traces the layer with zero graph breaks, so this is now reachable; it was
            not before (`self.device` mutation in `forward`).

## Docs

- [x] `README.md` rewritten: correct class name, Python/torch requirements, no
      torchvision, uv dev workflow, `PackedDeformConv1d` example, offset-shape
      explanation, benchmark caveat, removed the "has not been tested for
      numerical accuracy" disclaimer (it is now tested).
- [x] All docstrings in `dc1d/nn.py` and `dc1d/ops.py` corrected (see C-anno above).
- [x] `playground/readme_example.py` and `playground/param_example.py` updated to
      match, and both run clean.
- [x] This file.
- [ ] **No CHANGELOG.** **Left:** write `CHANGELOG.md` covering this pass before
      the next release — the `self.device` removal and the new offset-shape
      `ValueError` are behaviour changes users need to see.
- [ ] **No API reference / docs site.** Not asked for.

## Benchmarks

- [x] `benchmarks/benchmark.py` added, using `torch.utils.benchmark.Timer`
      (`blocked_autorange`), which warms up, adapts repeat counts, and synchronises
      CUDA around the timed region.
- [x] Forward and forward+backward timed separately, and `DeformConv1d` vs
      `nn.Conv1d` now get **identical** treatment. The old comparison
      (old `nn.py:494-511`) gave the deformable path 3 warmup iterations and the
      vanilla conv a single cold call including cuDNN algorithm selection.
- [x] The `time.time()`-based timing loops were removed from `dc1d/nn.py`'s
      `__main__` (now a plain smoke demo) and from `playground/param_example.py`.
- [x] Measured on CPU, B=4 C=256 L=2048 K=3 d=8, torch 2.13.0+cpu:
      interpolation only **141 ms**, `DeformConv1d` fwd **162 ms**, `nn.Conv1d` fwd
      **2.1 ms**; fwd+bwd **218 ms** vs **20 ms**. dc1d is ~77× slower than a
      native depthwise conv on CPU in this config — that is the honest cost of a
      pure-Python gather-based implementation, and it is *not* representative of
      GPU, where the arithmetic intensity is very different.
- [ ] **No GPU numbers.** Cannot be produced here (CPU-only torch by design).
      **Left:** `uv run python benchmarks/benchmark.py --device cuda` on the 3090.
      Do not quote GPU speedups until then.

## Release

- [x] `.github/workflows/release.yml.disabled` written as a **scaffold only**. It is
      named `.disabled` so GitHub Actions will not pick it up, its only trigger is
      `workflow_dispatch` (the `push: tags:` trigger is commented out), and it
      references **no secrets** — publishing would go through PyPI trusted
      publishing (OIDC), hence `permissions: id-token: write`.
- [x] It includes a tag↔`__version__` consistency check and a `twine check` step.
- [ ] **NOT activated. Nothing has been published, and no PyPI configuration was
      touched.** Remaining, in order:
      1. **Trusted publisher on PyPI.** Add a publisher for project `dc1d`, owner
         `jwr1995`, repo `dc1d`, workflow filename `release.yml`, environment `pypi`.
         Do the same on TestPyPI with environment `testpypi`.
         https://docs.pypi.org/trusted-publishers/adding-a-publisher/
      2. **GitHub environments.** Create `pypi` and `testpypi`. Recommend a required
         reviewer on `pypi` so every upload is human-approved.
      3. **Tag convention — needs an owner decision.** The scaffold assumes
         `v<version>` (e.g. `v0.1.0`) matching `dc1d/__init__.py:__version__`, and
         verifies the two agree.
      4. **Publish from tags? — needs an owner decision.** Currently
         `workflow_dispatch` only. To publish from tags, uncomment the
         `push: tags: ["v*"]` trigger *and* rename the file to `release.yml`.
         Recommendation: keep `workflow_dispatch` for a first TestPyPI dry run,
         then switch to tags.
      5. ~~**Version bump.**~~ Done: `0.0.7` → **`0.1.0`**, for the breaking changes
         in this pass (`self.device` removed, offset-shape mismatches now raise,
         `requires-python >= 3.10`). PyPI still has `0.0.7`, so the first publish
         from this line will be `v0.1.0`.
      6. **Rename** `release.yml.disabled` → `release.yml` as the last step.

---

## Explicitly not done

- Nothing was committed, branched, amended or pushed. All work is uncommitted in
  the working tree.
- No GPU was used and no GPU numbers are reported.
- `py.typed` not shipped (see Packaging).
- No CHANGELOG (see Docs).
- CI never executed on GitHub (see CI); the YAML itself is unvalidated.
