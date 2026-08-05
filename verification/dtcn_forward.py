#!/usr/bin/env python3
"""
Run the DTCN separation model (ICASSP 2023) once, under a chosen dc1d checkout.

DTCN -- "Deformable Temporal Convolutional Networks for Monaural Noisy
Reverberant Speech Separation", https://github.com/jwr1995/DTCN -- is the
downstream model dc1d was written for. Its ``separation/src/deformable.py``
does ``from dc1d.nn import PackedDeformConv1d``, so which dc1d is on
``sys.path`` decides which interpolation kernel the whole network runs.

This script is deliberately one-variant-per-process: the two dc1d checkouts
both call their package ``dc1d`` and use absolute intra-package imports, so
only one can own that name in a given interpreter.

Usage::

    python dtcn_forward.py --variant old --dc1d-root /path/to/old \\
        --dtcn /path/to/DTCN --save-state ckpt.pt --out old.pt
    python dtcn_forward.py --variant new --dc1d-root /path/to/new \\
        --dtcn /path/to/DTCN --load-state ckpt.pt --out new.pt

The model is built with the hyperparameters of
``separation/hparams/deformable/dtcn-whamr.yaml`` -- the configuration the
paper reports -- and the forward pass reproduces the inference branch of
``Separation.compute_forward`` in ``separation/train.py``.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["old", "new"])
    ap.add_argument("--dc1d-root", required=True, help="checkout containing the dc1d package")
    ap.add_argument("--dtcn", required=True, help="DTCN checkout root")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)  # the yaml's seed
    ap.add_argument("--length", type=int, default=32000)  # the yaml's training_signal_len
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--save-state", default=None)
    ap.add_argument("--load-state", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument(
        "--offset-conv-compat",
        action="store_true",
        help="rewire every PackedDeformConv1d offset_dconv back to the 0.0.7 "
        "wiring (stride=1, dilation=1), isolating the interpolation kernel "
        "from the offset-generator change",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    args.dc1d_root = os.path.abspath(args.dc1d_root)
    sys.path.insert(0, args.dc1d_root)
    sys.path.insert(0, os.path.abspath(f"{args.dtcn}/separation"))

    import torch
    import torchaudio

    # speechbrain 1.0.3 calls torchaudio.list_audio_backends() at import time;
    # torchaudio 2.11 removed it. Nothing here reads audio from disk.
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    # Every fp32 claim in EQUIVALENCE.md is a true-fp32 claim.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    import dc1d.nn

    assert dc1d.nn.__file__.startswith(args.dc1d_root), dc1d.nn.__file__

    from speechbrain.lobes.models.dual_path import Decoder, Encoder
    from src.deformable import MaskNet

    # --- the dtcn-whamr.yaml configuration -------------------------------
    sample_rate = 8000
    N_encoder_out = 512
    kernel_size = 16
    kernel_stride = 8
    num_spks = 2
    X, R = 8, 3
    hp = dict(N=N_encoder_out, B=128, H=512, P=3, X=X, R=R, C=num_spks)

    torch.manual_seed(args.seed)
    encoder = Encoder(kernel_size=kernel_size, out_channels=N_encoder_out)
    masknet = MaskNet(**hp, norm_type="gLN", causal=False, mask_nonlinear="relu")
    decoder = Decoder(
        in_channels=N_encoder_out,
        out_channels=1,
        kernel_size=kernel_size,
        stride=kernel_stride,
        bias=False,
    )

    modules = torch.nn.ModuleDict({"encoder": encoder, "masknet": masknet, "decoder": decoder})
    if args.load_state:
        state = torch.load(args.load_state, map_location="cpu", weights_only=True)
        missing, unexpected = modules.load_state_dict(state, strict=True)
        assert not missing and not unexpected
    if args.save_state:
        torch.save(modules.state_dict(), args.save_state)

    if args.offset_conv_compat:
        # dc1d 0.0.7 built this conv as
        #   nn.Conv1d(C, C, k, stride=1, groups=C, padding=padding,
        #             padding_mode=padding_mode, bias=False)
        # i.e. it dropped `stride` and `dilation`. Reinstating that wiring while
        # keeping the new interpolation kernel separates the two changes.
        n_patched = 0
        for mod in masknet.modules():
            if isinstance(mod, dc1d.nn.PackedDeformConv1d):
                old_conv = mod.offset_dconv
                compat = torch.nn.Conv1d(
                    old_conv.in_channels,
                    old_conv.out_channels,
                    old_conv.kernel_size[0],
                    stride=1,
                    groups=old_conv.groups,
                    padding=mod.padding,
                    padding_mode=mod.padding_mode,
                    bias=False,
                )
                compat.weight.data.copy_(old_conv.weight.data)
                mod.offset_dconv = compat
                n_patched += 1
        print(f"offset-conv compat: rewired {n_patched} layers to stride=1, dilation=1")

    modules.to(args.device).to(getattr(torch, args.dtype)).eval()

    # Deterministic input, generated on the CPU so it is identical in both
    # processes regardless of device RNG differences.
    gen = torch.Generator().manual_seed(20260728)
    mix = torch.randn(args.batch, args.length, generator=gen)
    mix = mix.to(args.device).to(getattr(torch, args.dtype))

    # Capture the input of every PackedDeformConv1d so its offsets can be
    # recomputed afterwards and a divergence attributed to a specific layer.
    deform_layers = {
        name: mod
        for name, mod in masknet.named_modules()
        if isinstance(mod, dc1d.nn.PackedDeformConv1d)
    }
    captured: dict[str, torch.Tensor] = {}
    handles = [
        mod.register_forward_pre_hook(
            lambda m, inp, _n=name: captured.__setitem__(_n, inp[0].detach())
        )
        for name, mod in deform_layers.items()
    ]

    # --- inference branch of Separation.compute_forward -------------------
    with torch.no_grad():
        mix_w = encoder(mix)
        est_mask = masknet(mix_w)
        stacked = torch.stack([mix_w] * num_spks)
        sep_h = stacked * est_mask
        est_source = torch.cat([decoder(sep_h[i]).unsqueeze(-1) for i in range(num_spks)], dim=-1)
        T_origin, T_est = mix.size(1), est_source.size(1)
        if T_origin > T_est:
            est_source = torch.nn.functional.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

    for h in handles:
        h.remove()

    # Per-layer offsets, so a divergence can be attributed to a layer.
    offsets = {}
    dilations = {}
    with torch.no_grad():
        for name, mod in deform_layers.items():
            offsets[name] = mod(captured[name], True)[1].detach().cpu()
            dilations[name] = mod.dilation

    torch.save(
        {
            "variant": args.variant,
            "dtype": args.dtype,
            "offset_conv_compat": args.offset_conv_compat,
            "dc1d_file": dc1d.nn.__file__,
            "est_source": est_source.detach().cpu(),
            "mix_w": mix_w.detach().cpu(),
            "est_mask": est_mask.detach().cpu(),
            "offsets": offsets,
            "dilations": dilations,
            "torch": torch.__version__,
            "device": args.device,
            "sample_rate": sample_rate,
        },
        args.out,
    )
    print(f"{args.variant}: est_source {tuple(est_source.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
