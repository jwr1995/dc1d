#!/usr/bin/env python3
"""
Compare two runs produced by ``dtcn_forward.py``.

Reports agreement of the separated waveforms (``torch.equal``, max absolute
deviation, RMS, and SI-SDR of one against the other treated as reference),
and per deformable layer whether the predicted offsets agree.
"""

from __future__ import annotations

import argparse

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    est = est.double() - est.double().mean()
    ref = ref.double() - ref.double().mean()
    target = ((est * ref).sum() / (ref * ref).sum()) * ref
    noise = est - target
    return (10 * torch.log10((target**2).sum() / (noise**2).sum())).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--per-layer", action="store_true")
    args = ap.parse_args()

    a = torch.load(args.a, weights_only=False)
    b = torch.load(args.b, weights_only=False)
    x, y = a["est_source"], b["est_source"]

    def tag(r):
        return f"{r['variant']}/{r['dtype']}" + ("/compat" if r["offset_conv_compat"] else "")

    print(f"A = {tag(a)}  ({args.a})")
    print(f"B = {tag(b)}  ({args.b})")
    print(f"  encoder output mix_w         torch.equal = {torch.equal(a['mix_w'], b['mix_w'])}")
    print(f"  separated waveform           torch.equal = {torch.equal(x, y)}")
    print(f"  max |A - B|                  = {(x - y).abs().max().item():.6e}")
    print(
        f"  rms  (A - B)                 = {((x.double() - y.double()) ** 2).mean().sqrt().item():.6e}"
    )
    print(f"  max |A|                      = {x.abs().max().item():.6e}")
    for s in range(x.shape[-1]):
        print(f"  SI-SDR(B vs A), speaker {s}    = {si_sdr(y[0, :, s], x[0, :, s]):.3f} dB")

    n_diff = sum(1 for k in a["offsets"] if not torch.equal(a["offsets"][k], b["offsets"][k]))
    worst = max((a["offsets"][k] - b["offsets"][k]).abs().max().item() for k in a["offsets"])
    print(f"  deformable layers with differing offsets: {n_diff}/{len(a['offsets'])}")
    print(f"  max |offset difference| over all layers : {worst:.6e}")

    if args.per_layer:
        print(
            f"\n  {'layer':22s} {'dilation':>8s} {'offsets equal':>14s} {'max |diff|':>12s} {'offset scale':>13s}"
        )
        for k in a["offsets"]:
            oa, ob = a["offsets"][k], b["offsets"][k]
            short = k.split(".")[1] if "." in k else k
            print(
                f"  {short:22s} {a['dilations'][k]:8d} {str(torch.equal(oa, ob)):>14s} {(oa - ob).abs().max().item():12.4e} {oa.abs().max().item():13.4f}"
            )


if __name__ == "__main__":
    main()
