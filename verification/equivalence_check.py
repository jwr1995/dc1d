#!/usr/bin/env python3
"""
Old-versus-new equivalence evidence for the dc1d interpolation kernel.

Compares dc1d 0.0.7 (commit 6ca2e01, the state published on PyPI since 2023)
against dc1d 0.1.0 (origin/main after PRs #9 and #10), by loading both
checkouts into a single process. Every number quoted in EQUIVALENCE.md is
printed by this script.

The two checkouts both name their package ``dc1d`` and use absolute
intra-package imports, so they cannot both own that name in ``sys.modules``.
Import one, keep references to the module objects, evict it, import the other:
each module's functions keep resolving through their own ``__dict__``, so both
stay usable afterwards.

0.0.7 imports private torchvision symbols at module scope
(``torchvision.extension._assert_has_ops``,
``torchvision.utils._log_api_usage_once``), for a ``deform_conv1d`` helper that
neither layer class ever calls, so torchvision must be installed to import it
at all. 0.1.0 is deliberately torchvision-free.

Usage::

    python equivalence_check.py --old /path/to/dc1d@6ca2e01 --new /path/to/dc1d@main
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# ---------------------------------------------------------------------------
# dual import
# ---------------------------------------------------------------------------


def _purge() -> None:
    for name in [m for m in sys.modules if m == "dc1d" or m.startswith("dc1d.")]:
        del sys.modules[name]


def _load(root: str):
    root = os.path.abspath(root)
    _purge()
    sys.path.insert(0, root)
    try:
        import dc1d.nn as nn_mod
        import dc1d.ops as ops_mod

        assert nn_mod.__file__.startswith(root), nn_mod.__file__
        return nn_mod, ops_mod
    finally:
        sys.path.remove(root)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

OLD_NN = OLD_OPS = NEW_NN = NEW_OPS = None
DEV = "cuda:0"


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def old_interp(x, offsets, k, d, s, **kw):
    return OLD_OPS.efficient_linterpolate(x, offsets, k, d, s, device=x.device, **kw)


def new_interp(x, offsets, k, d, s, **kw):
    return NEW_OPS.efficient_linterpolate(x, offsets, k, d, s, **kw)


def old_layer(cls_name, dtype, **kw):
    """0.0.7 keeps `dilated_positions` as a plain attribute and caches a device
    string, so `.to()` does not move it; patch both, as any 0.0.7 user would
    have had to."""
    m = getattr(OLD_NN, cls_name)(**kw).to(DEV, dtype)
    m.device = DEV
    m.dilated_positions = m.dilated_positions.to(DEV, dtype)
    return m


def stats(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a.double() - b.double()).abs()
    return f"equal={str(torch.equal(a, b)):5s} max|d|={d.max():.4e} rms={((a.double() - b.double()) ** 2).mean().sqrt():.4e}"


def grads(fn, x, o, g):
    x = x.clone().requires_grad_(True)
    o = o.clone().requires_grad_(True)
    y = fn(x, o)
    y.backward(g.to(y.dtype))
    return y.detach(), x.grad, o.grad


# ---------------------------------------------------------------------------
# A. Where behaviour was meant to be unchanged
# ---------------------------------------------------------------------------


def section_a() -> None:
    head("A1. Zero-offset nn.Conv1d equivalence (the load-bearing invariant)")
    print("torch.equal against nn.Conv1d with identical weights, constrained mode.\n")
    print(
        f"{'device':7s} {'dtype':8s} {'k':>2s} {'d':>2s} {'s':>2s} {'g':>2s}  {'OLD==conv':10s} {'NEW==conv':10s} OLD==NEW"
    )
    for dev in ("cpu", DEV):
        for dtype in (torch.float64, torch.float32):
            for k, d, s, g in (
                (3, 1, 1, 1),
                (3, 4, 1, 1),
                (5, 2, 3, 1),
                (3, 2, 1, 4),
                (3, 8, 1, 1),
            ):
                C, L = 4, 128
                x = torch.randn(2, C, L, device=dev, dtype=dtype)
                ref = torch.nn.Conv1d(
                    C, C, k, stride=s, dilation=d, padding="valid", groups=g, bias=True
                ).to(dev, dtype)
                lo = (L - d * (k - 1) - 1) // s + 1
                off = torch.zeros(2, 1, lo, k, device=dev, dtype=dtype)
                om = OLD_NN.DeformConv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    padding="valid",
                    groups=g,
                    bias=True,
                ).to(dev, dtype)
                om.device = dev
                om.dilated_positions = om.dilated_positions.to(dev, dtype)
                nm = NEW_NN.DeformConv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    padding="valid",
                    groups=g,
                    bias=True,
                ).to(dev, dtype)
                for m in (om, nm):
                    m.weight.data.copy_(ref.weight.data)
                    m.bias.data.copy_(ref.bias.data)
                yo, yn, r = om(x, off), nm(x, off), ref(x)
                print(
                    f"{dev:7s} {str(dtype).split('.')[-1]:8s} {k:2d} {d:2d} {s:2d} {g:2d}  {str(torch.equal(yo, r)):10s} {str(torch.equal(yn, r)):10s} {torch.equal(yo, yn)}"
                )

    head("A2. Forward, non-trivial random offsets, constrained, groups=1, stride=1")
    print("Is OLD bit-identical to NEW? float32 rows are additionally scored against")
    print("a float64 evaluation; the last column is how far the two float64")
    print("evaluations are from each other, i.e. how well defined that reference is.\n")
    print(
        f"{'dtype':8s} {'L':>8s} {'torch.equal':11s} {'max|OLD-NEW|':>13s} {'err(OLD) vs f64':>16s} {'err(NEW) vs f64':>16s} {'f64 OLD vs NEW':>15s}"
    )
    k, d = 3, 4
    for dtype in (torch.float64, torch.float32):
        for L in (64, 256, 4096, 65536, 1048576):
            lo = L - d * (k - 1)
            x = torch.randn(1, 2, L, device=DEV, dtype=dtype)
            o = torch.rand(1, 1, lo, k, device=DEV, dtype=dtype) * (d * (k - 1))
            a = old_interp(x, o, k, d, 1)
            b = new_interp(x, o, k, d, 1)
            r1 = new_interp(x.double(), o.double(), k, d, 1)
            r2 = old_interp(x.double(), o.double(), k, d, 1)
            if dtype is torch.float64:
                ea = eb = float("nan")
            else:
                ea = (a.double() - r1).abs().max().item()
                eb = (b.double() - r1).abs().max().item()
            print(
                f"{str(dtype).split('.')[-1]:8s} {L:8d} {str(torch.equal(a, b)):11s} {(a.double() - b.double()).abs().max().item():13.4e} {ea:16.4e} {eb:16.4e} {(r1 - r2).abs().max().item():15.2e}"
            )

    head("A2b. Analytic ground truth: x[i] = i mod 2")
    print("With a 0/1 sawtooth input the two gathered samples are exactly 0 and 1, so")
    print("linear interpolation at position `index + f` returns exactly `f` (even index)")
    print("or `1 - f` (odd index). The output is O(1) while the positions are large, so")
    print("this isolates the error in the sub-sample fraction. Positions are computed")
    print("here in exact integer arithmetic and the reference depends on neither")
    print("implementation. Offsets are drawn so that no tap sits on the clamp.\n")
    print(f"{'dtype':8s} {'L':>8s} {'max err OLD':>13s} {'max err NEW':>13s} {'eps':>11s}")
    for dtype in (torch.float64, torch.float32):
        for L in (256, 4096, 65536, 1048576):
            lo = L - d * (k - 1)
            x = (torch.arange(L, device=DEV) % 2).to(dtype).reshape(1, 1, L)
            # The last tap already sits at d*(k-1), so it needs a negative offset
            # to stay strictly inside the constraint band; all others a positive
            # one. Nothing then hits the clamp.
            u = torch.rand(1, 1, lo, k, dtype=torch.float64) * 0.9 + 0.05
            u[..., -1] -= 1.0
            o = u.to(DEV, dtype)
            tap = (torch.arange(k, dtype=torch.float64) * d).reshape(1, 1, 1, k)
            t0 = torch.arange(lo, dtype=torch.float64).reshape(1, 1, lo, 1)
            # positions as actually requested, from the offsets as actually stored
            rel = tap + o.double().cpu()
            assert rel.min() > 0 and rel.max() < d * (k - 1), "a tap hit the clamp"
            fl = torch.floor(rel)
            f = rel - fl
            index = (t0 + fl).long()
            exact = torch.where(index % 2 == 0, f, 1.0 - f).to(DEV)
            a = old_interp(x, o, k, d, 1)
            b = new_interp(x, o, k, d, 1)
            eps = torch.finfo(dtype).eps
            print(
                f"{str(dtype).split('.')[-1]:8s} {L:8d} {(a.double() - exact).abs().max().item():13.4e} {(b.double() - exact).abs().max().item():13.4e} {eps:11.2e}"
            )

    head("A3. When IS the forward bit-identical?")
    k, d, L = 3, 2, 64
    lo = L - d * (k - 1)
    x = torch.randn(1, 4, L, device=DEV)
    cases = {
        "zero offsets": torch.zeros(1, 1, lo, k, device=DEV),
        "integer offsets": torch.randint(0, 3, (1, 1, lo, k), device=DEV).float(),
        "half-integer offsets": torch.full((1, 1, lo, k), 0.5, device=DEV),
        "quarter-integer offsets": torch.full((1, 1, lo, k), 0.25, device=DEV),
        "random offsets": torch.rand(1, 1, lo, k, device=DEV) * (d * (k - 1)),
    }
    print(f"{'case':26s} {'torch.equal':11s} max|OLD-NEW|")
    for lbl, o in cases.items():
        a, b = old_interp(x, o, k, d, 1), new_interp(x, o, k, d, 1)
        print(f"{lbl:26s} {str(torch.equal(a, b)):11s} {(a - b).abs().max().item():.4e}")

    head("A4. Gradients, random offsets, constrained, groups=1, stride=1, float32")
    print(
        f"{'L':>8s}  {'grad':6s} {'torch.equal':11s} {'max|OLD-NEW|':>13s} {'err(OLD) vs f64':>16s} {'err(NEW) vs f64':>16s} {'|grad| scale':>13s}"
    )
    for L in (256, 4096, 65536):
        lo = L - d * (k - 1)
        x = torch.randn(1, 2, L, device=DEV)
        o = torch.rand(1, 1, lo, k, device=DEV) * (d * (k - 1))
        g = torch.randn(1, 2, lo, k, device=DEV)
        _, gxa, goa = grads(lambda a, b: old_interp(a, b, k, d, 1), x, o, g)
        _, gxb, gob = grads(lambda a, b: new_interp(a, b, k, d, 1), x, o, g)
        _, gxr, gor = grads(
            lambda a, b: new_interp(a, b, k, d, 1), x.double(), o.double(), g.double()
        )
        for lbl, ta, tb, tr in (("d/dx", gxa, gxb, gxr), ("d/doff", goa, gob, gor)):
            print(
                f"{L:8d}  {lbl:6s} {str(torch.equal(ta, tb)):11s} {(ta.double() - tb.double()).abs().max().item():13.4e} {(ta.double() - tr).abs().max().item():16.4e} {(tb.double() - tr).abs().max().item():16.4e} {tr.abs().max().item():13.4e}"
            )

    head("A5. Gradient w.r.t. offsets where sampling positions land on integers")
    print("Interpolation weights sum to 1, so d(out)/d(offset) must be invariant to")
    print("adding a constant to the input. Test: zero offsets, then x -> x + 1000.\n")
    k, d, L = 3, 1, 32
    lo = L - d * (k - 1)
    x = torch.randn(1, 1, L, device=DEV, dtype=torch.float64)
    g = torch.ones(1, 1, lo, k, device=DEV, dtype=torch.float64)
    for lbl, fn in (
        ("OLD", lambda a, b: old_interp(a, b, k, d, 1)),
        ("NEW", lambda a, b: new_interp(a, b, k, d, 1)),
    ):
        o = torch.zeros(1, 1, lo, k, device=DEV, dtype=torch.float64)
        _, _, go0 = grads(fn, x, o, g)
        _, _, go1 = grads(fn, x + 1000.0, o, g)
        shift = (go1 - go0).abs().max().item()
        print(f"  {lbl}: d/doffset[0,0,0,:] = {[round(v, 6) for v in go0[0, 0, 0].tolist()]}")
        print(f"       shift-invariance violation max|grad(x+1000) - grad(x)| = {shift:.4e}")
    xs = [x[0, 0, i].item() for i in range(4)]
    print(f"  x[0:4] = {[round(v, 6) for v in xs]}")
    print(
        f"  analytic right-derivative at tap j = x[j+1]-x[j]:  {[round(xs[j + 1] - xs[j], 6) for j in range(3)]}"
    )
    print("  OLD instead returns 0.5*x[j+1] from the `torch.max(zeros, 1-|U+1-T|)` tie,")
    print("  halved again at the two taps that also sit on the constrained clamp")
    print("  boundary (`torch.max(T, t0s)` and `torch.min(T, t0s+max_tap)`), which for")
    print("  k=3, d=1, zero offsets is taps 0 and 2:")
    print(
        f"    0.25*x[1], 0.5*x[2], 0.25*x[3] = {[round(v, 6) for v in (0.25 * xs[1], 0.5 * xs[2], 0.25 * xs[3])]}"
    )


# ---------------------------------------------------------------------------
# B. Deliberate divergences
# ---------------------------------------------------------------------------


def section_b() -> None:
    head("B1. fp16 offsets: position arithmetic inherits the offset dtype in 0.0.7")
    print("Rows counted as wrong if max|value - fp64 reference| > 1e-3 anywhere in the row.\n")
    print(
        f"{'L':>7s} {'out_len':>8s} {'OLD dtype':>10s} {'NEW dtype':>10s} {'OLD rows wrong':>15s} {'NEW rows wrong':>15s} {'OLD max err':>12s} {'NEW max err':>12s}"
    )
    k, d = 3, 1
    for L in (256, 1024, 2048, 4096, 16000):
        lo = L - d * (k - 1)
        x = torch.randn(1, 1, L, device=DEV, dtype=torch.float16)
        o = (torch.rand(1, 1, lo, k, device=DEV) * 2).half()
        ref = new_interp(x.double(), o.double(), k, d, 1)
        a, b = old_interp(x, o, k, d, 1), new_interp(x, o, k, d, 1)
        da, db = (a.double() - ref).abs(), (b.double() - ref).abs()
        print(
            f"{L:7d} {lo:8d} {str(a.dtype).split('.')[-1]:>10s} {str(b.dtype).split('.')[-1]:>10s} "
            f"{(da.amax(dim=(1, 3)) > 1e-3).sum().item():15d} {(db.amax(dim=(1, 3)) > 1e-3).sum().item():15d} {da.max().item():12.4f} {db.max().item():12.4f}"
        )

    head("B2. Unconstrained mode near the right-hand edge")
    print("x = ones, so the output equals the sum of the two interpolation weights.")
    print("A correct kernel returns 1.0 for every in-range sampling position.\n")
    k, d, L = 1, 1, 16
    x = torch.ones(1, 1, L, device=DEV, dtype=torch.float64)
    print(f"{'sampling position T':>20s} {'OLD weight sum':>15s} {'NEW weight sum':>15s}")
    for t in (0.0, 8.0, 14.0, 14.5, 15.0, 15.5, 16.0):
        o = torch.full((1, 1, L, k), t, device=DEV, dtype=torch.float64)
        a = old_interp(x, o, k, d, 1, unconstrained=True)
        b = new_interp(x, o, k, d, 1, unconstrained=True)
        print(f"{t:20.1f} {a[0, 0, 0, 0].item():15.6f} {b[0, 0, 0, 0].item():15.6f}")

    head("B3. PackedDeformConv1d output length vs nn.Conv1d, padding='valid'")
    print("0.0.7 builds offset_dconv as nn.Conv1d(..., stride=1, <no dilation>),")
    print("so it emits the wrong number of offset positions.\n")
    print(f"{'stride':>6s} {'dilation':>8s} {'nn.Conv1d':>10s} {'OLD':>18s} {'NEW':>6s}")
    for st, dil in ((1, 1), (2, 1), (1, 4), (3, 2), (4, 8)):
        C, k, L = 4, 3, 64
        xx = torch.randn(1, C, L, device=DEV)
        ref = (
            torch.nn.Conv1d(C, C, k, stride=st, dilation=dil, padding="valid", groups=C, bias=False)
            .to(DEV)(xx)
            .shape[-1]
        )
        row = [f"{st:6d}", f"{dil:8d}", f"{ref:10d}"]
        for mod in (OLD_NN, NEW_NN):
            try:
                m = mod.PackedDeformConv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=k,
                    stride=st,
                    dilation=dil,
                    padding="valid",
                    groups=C,
                    bias=True,
                ).to(DEV)
                if mod is OLD_NN:
                    m.device = DEV
                    m.dilated_positions = m.dilated_positions.to(DEV)
                got = m(xx).shape[-1]
                row.append(f"{got}{'' if got == ref else ' WRONG'}")
            except Exception as e:  # noqa: BLE001
                row.append(type(e).__name__)
        print(f"{row[0]} {row[1]} {row[2]} {row[3]:>18s} {row[4]:>6s}")

    head("B4. offset_groups strictly between 1 and in_channels")
    C, k, L = 8, 3, 32
    xx = torch.randn(1, C, L, device=DEV)
    for lbl, mod in (("OLD", OLD_NN), ("NEW", NEW_NN)):
        try:
            m = mod.PackedDeformConv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=k,
                padding="valid",
                groups=C,
                offset_groups=2,
            ).to(DEV)
            if mod is OLD_NN:
                m.device = DEV
                m.dilated_positions = m.dilated_positions.to(DEV)
            print(f"  {lbl} PackedDeformConv1d(offset_groups=2): ok, output {tuple(m(xx).shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {lbl} PackedDeformConv1d(offset_groups=2): {type(e).__name__}: {str(e)[:90]}")
    off = torch.rand(1, 2, L - (k - 1), k, device=DEV) * 2
    for lbl, mod in (("OLD", OLD_NN), ("NEW", NEW_NN)):
        try:
            m = mod.DeformConv1d(in_channels=C, out_channels=C, kernel_size=k, padding="valid").to(
                DEV
            )
            if mod is OLD_NN:
                m.device = DEV
                m.dilated_positions = m.dilated_positions.to(DEV)
            print(
                f"  {lbl} DeformConv1d, offsets with 2 groups:  ok, output {tuple(m(xx, off).shape)}"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  {lbl} DeformConv1d, offsets with 2 groups:  {type(e).__name__}: {str(e)[:90]}"
            )

    head("B5. repr(layer)")
    for lbl, mod in (("OLD", OLD_NN), ("NEW", NEW_NN)):
        m = mod.DeformConv1d(in_channels=8, out_channels=8, kernel_size=3, padding="valid")
        try:
            print(f"  {lbl}: {repr(m)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {lbl}: {type(e).__name__}: {e}")

    head("B7. Integer padding with padding_mode='zeros' (extra fix, not on the brief)")
    print("0.0.7's forward pads only when padding_mode != 'zeros' or padding == 'same',")
    print("so an int padding with padding_mode='zeros' is silently not applied; the")
    print("index clamp then absorbs the shape error and returns a plausible wrong answer.\n")
    C, L, k = 2, 16, 3
    xx = torch.randn(1, C, L, device=DEV, dtype=torch.float64)
    off = torch.zeros(1, 1, L, k, device=DEV, dtype=torch.float64)
    ref = torch.nn.Conv1d(C, C, k, padding=1, padding_mode="zeros", bias=False).to(
        DEV, torch.float64
    )
    for lbl, mod in (("OLD", OLD_NN), ("NEW", NEW_NN)):
        m = mod.DeformConv1d(
            in_channels=C,
            out_channels=C,
            kernel_size=k,
            padding=1,
            padding_mode="zeros",
            bias=False,
        ).to(DEV, torch.float64)
        if mod is OLD_NN:
            m.device = DEV
            m.dilated_positions = m.dilated_positions.to(DEV, torch.float64)
        m.weight.data.copy_(ref.weight.data)
        y = m(xx, off)
        r = ref(xx)
        print(
            f"  {lbl}: out_len={y.shape[-1]} (nn.Conv1d {r.shape[-1]}), torch.equal to nn.Conv1d = {torch.equal(y, r)}, max|d|={(y - r).abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# C. The opt-in backward
# ---------------------------------------------------------------------------


def section_c() -> None:
    head("C. gather_lerp variants of 0.1.0 against its own default ('autograd')")
    print(
        f"{'device':7s} {'dtype':8s} {'impl':10s} {'fwd equal':10s} {'d/dx equal':11s} {'max|d/dx diff|':>15s} {'d/doff equal':13s} {'max|d/doff diff|':>17s}"
    )
    k, d, st, L = 3, 2, 1, 257
    lo = (L - d * (k - 1) - 1) // st + 1
    for dev in ("cpu", DEV):
        for dtype in (torch.float64, torch.float32):
            x = torch.randn(2, 6, L, device=dev, dtype=dtype)
            o = torch.rand(2, 3, lo, k, device=dev, dtype=dtype) * (d * (k - 1))
            g = torch.randn(2, 6, lo, k, device=dev, dtype=dtype)
            res = {}
            for impl in ("autograd", "recompute", "save-diff"):
                res[impl] = grads(
                    lambda a, b, _i=impl: NEW_OPS.efficient_linterpolate(
                        a, b, k, d, st, gather_lerp=_i
                    ),
                    x,
                    o,
                    g,
                )
            base = res["autograd"]
            for impl in ("recompute", "save-diff"):
                y, gx, go = res[impl]
                print(
                    f"{dev:7s} {str(dtype).split('.')[-1]:8s} {impl:10s} {str(torch.equal(base[0], y)):10s} "
                    f"{str(torch.equal(base[1], gx)):11s} {(base[1] - gx).abs().max().item():15.4e} "
                    f"{str(torch.equal(base[2], go)):13s} {(base[2] - go).abs().max().item():17.4e}"
                )

    print("\nd/dx accuracy of each variant against a float64 reference (float32, cuda):")
    x = torch.randn(2, 6, L, device=DEV)
    o = torch.rand(2, 3, lo, k, device=DEV) * (d * (k - 1))
    g = torch.randn(2, 6, lo, k, device=DEV)
    _, gxr, _ = grads(
        lambda a, b: NEW_OPS.efficient_linterpolate(a, b, k, d, st),
        x.double(),
        o.double(),
        g.double(),
    )
    for impl in ("autograd", "recompute", "save-diff"):
        _, gx, _ = grads(
            lambda a, b, _i=impl: NEW_OPS.efficient_linterpolate(a, b, k, d, st, gather_lerp=_i),
            x,
            o,
            g,
        )
        print(f"  {impl:10s} max|d/dx - fp64| = {(gx.double() - gxr).abs().max().item():.4e}")

    print("\nRun-to-run determinism of d/dx (same inputs, two runs), float32:")
    for dev in ("cpu", DEV):
        xx, oo, gg = x.to(dev), o.to(dev), g.to(dev)
        for impl in ("autograd", "recompute", "save-diff"):
            r = [
                grads(
                    lambda a, b, _i=impl: NEW_OPS.efficient_linterpolate(
                        a, b, k, d, st, gather_lerp=_i
                    ),
                    xx,
                    oo,
                    gg,
                )[1]
                for _ in range(2)
            ]
            print(f"  {dev:7s} {impl:10s} torch.equal = {torch.equal(r[0], r[1])}")


# ---------------------------------------------------------------------------
# R. Hunting for regressions: places 0.1.0 might be wrong where 0.0.7 was right
# ---------------------------------------------------------------------------


def section_r() -> None:
    head("R1. User-supplied non-integer `dilated_positions`")
    print("0.0.7 treated `dilated_positions` as arbitrary float tap positions and")
    print("interpolated them. 0.1.0's `_dilated_positions_long` does .round().long().\n")
    k, d, L = 3, 2, 32
    lo = L - d * (k - 1)
    x = torch.randn(1, 1, L, device=DEV, dtype=torch.float64)
    o = torch.zeros(1, 1, lo, k, device=DEV, dtype=torch.float64)
    dp = torch.tensor([0.0, 1.5, 3.7], device=DEV, dtype=torch.float64)
    a = old_interp(x, o, k, d, 1, dilated_positions=dp)
    b = new_interp(x, o, k, d, 1, dilated_positions=dp)
    print(f"  tap positions requested: {dp.tolist()}")
    print(f"  OLD output taps: {[round(v, 6) for v in a[0, 0, 0].tolist()]}")
    print(f"  NEW output taps: {[round(v, 6) for v in b[0, 0, 0].tolist()]}")
    print(f"  x[0:5]         : {[round(v, 6) for v in x[0, 0, :5].tolist()]}")
    print(f"  torch.equal = {torch.equal(a, b)}  (NEW == x at integer indices 0, 2, 4)")

    head("R2. Out-of-range offsets, constrained mode")
    k, d, L = 3, 1, 16
    lo = L - d * (k - 1)
    x = torch.randn(1, 1, L, device=DEV, dtype=torch.float64)
    for val in (-50.0, -1.0, 50.0):
        o = torch.full((1, 1, lo, k), val, device=DEV, dtype=torch.float64)
        print(
            f"  offset={val:+8.1f}: {stats(old_interp(x, o, k, d, 1), new_interp(x, o, k, d, 1))}"
        )

    head("R3. Subgradient exactly at the constrained clamp boundary")
    k, d, L = 3, 2, 16
    lo = L - d * (k - 1)
    x = torch.randn(1, 1, L, device=DEV, dtype=torch.float64)
    g = torch.ones(1, 1, lo, k, device=DEV, dtype=torch.float64)
    for lbl, val in (
        ("tap pinned to upper bound", float(d * (k - 1))),
        ("tap pinned to lower bound", -3.0),
    ):
        out = {}
        for name, fn in (
            ("OLD", lambda a, b: old_interp(a, b, k, d, 1)),
            ("NEW", lambda a, b: new_interp(a, b, k, d, 1)),
        ):
            o = torch.full((1, 1, lo, k), val, device=DEV, dtype=torch.float64)
            out[name] = grads(fn, x, o, g)[2]
        print(
            f"  {lbl}: OLD d/doff[0,0,0,:]={[round(v, 4) for v in out['OLD'][0, 0, 0].tolist()]} "
            f"NEW={[round(v, 4) for v in out['NEW'][0, 0, 0].tolist()]} equal={torch.equal(out['OLD'], out['NEW'])}"
        )

    head("R4. PackedDeformConv1d state_dict portability, 0.0.7 -> 0.1.0")
    print("The offset_dconv weight has the same shape either way, so a 0.0.7")
    print("checkpoint loads with strict=True and then computes different offsets.\n")
    print(
        f"{'dilation':>8s} {'stride':>6s} {'strict load':>12s} {'offsets equal':>14s} {'max|offset diff|':>17s} {'max|output diff|':>17s} {'output scale':>13s}"
    )
    C, k, L = 8, 3, 512
    xx = torch.randn(1, C, L, device=DEV)
    for dil, st in ((1, 1), (2, 1), (4, 1), (128, 1)):
        mo = old_layer(
            "PackedDeformConv1d",
            torch.float32,
            in_channels=C,
            out_channels=C,
            kernel_size=k,
            stride=st,
            dilation=dil,
            padding="same",
            groups=C,
            bias=True,
        )
        mn = NEW_NN.PackedDeformConv1d(
            in_channels=C,
            out_channels=C,
            kernel_size=k,
            stride=st,
            dilation=dil,
            padding="same",
            groups=C,
            bias=True,
        ).to(DEV)
        ok = "ok"
        try:
            mn.load_state_dict(mo.state_dict(), strict=True)
        except Exception as e:  # noqa: BLE001
            ok = type(e).__name__
        yo, oo = mo(xx, True)
        yn, on = mn(xx, True)
        print(
            f"{dil:8d} {st:6d} {ok:>12s} {str(torch.equal(oo, on)):>14s} {(oo - on).abs().max().item():17.4e} {(yo - yn).abs().max().item():17.4e} {yo.abs().max().item():13.4e}"
        )


def main() -> None:
    global OLD_NN, OLD_OPS, NEW_NN, NEW_OPS, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    DEV = args.device

    # Every float32 claim below is a true float32 claim.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)

    OLD_NN, OLD_OPS = _load(args.old)
    NEW_NN, NEW_OPS = _load(args.new)
    _purge()

    print("dc1d equivalence evidence")
    print(f"  torch          {torch.__version__} (CUDA {torch.version.cuda})")
    print(
        f"  device         {DEV} = {torch.cuda.get_device_name(DEV) if DEV.startswith('cuda') else 'cpu'}"
    )
    print(
        f"  TF32           cudnn={torch.backends.cudnn.allow_tf32} matmul={torch.backends.cuda.matmul.allow_tf32}"
    )
    print(f"  OLD            {OLD_NN.__file__}")
    print(f"  NEW            {NEW_NN.__file__}")
    print(f"  seed           {args.seed}")

    section_a()
    section_b()
    section_c()
    section_r()


if __name__ == "__main__":
    main()
