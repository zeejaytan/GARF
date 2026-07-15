#!/usr/bin/env python3
"""Swap GARF's frozen encoder for the worn-break fine-tuned one (Exp 11).

The denoiser checkpoint (GARF.ckpt) bundles the feature extractor weights under
``feature_extractor.*``. To test the Exp 11 worn-adapted encoder in the FULL
assembly pipeline while changing NOTHING else, we replace exactly those keys with
the fine-tuned encoder (from finetune_frac_seg.py's exported
``feature_extractor_worn.ckpt``, whose keys are encoder/batch_norm/coarse_segmenter
without the ``feature_extractor.`` prefix) and save a new full checkpoint. The
denoiser weights are untouched, so any change in Juglet assembly is attributable
solely to the encoder.

Usage
-----
  python scripts/merge_worn_encoder.py \
      --base output/GARF.ckpt \
      --worn output/frac_seg_worn_<stamp>/feature_extractor_worn.ckpt \
      --out output/GARF_worn_encoder.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("output/GARF.ckpt"))
    ap.add_argument("--worn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    worn = torch.load(args.worn, map_location="cpu", weights_only=True)["state_dict"]

    sd = base["state_dict"]
    fe_keys = [k for k in sd if k.startswith("feature_extractor.")]
    replaced, missing = 0, []
    for k in fe_keys:
        sub = k[len("feature_extractor."):]
        if sub in worn:
            if sd[k].shape != worn[sub].shape:
                raise ValueError(f"shape mismatch for {k}: {sd[k].shape} vs {worn[sub].shape}")
            sd[k] = worn[sub].to(sd[k].dtype)
            replaced += 1
        else:
            missing.append(sub)
    extra = [k for k in worn if k not in {kk[len("feature_extractor."):] for kk in fe_keys}]
    if missing:
        raise ValueError(f"{len(missing)} base feature_extractor keys not in worn ckpt: {missing[:5]}")
    if extra:
        print(f"WARNING: {len(extra)} worn keys unused (not in base): {extra[:5]}")

    base["state_dict"] = sd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.out)
    print(f"replaced {replaced}/{len(fe_keys)} feature_extractor tensors; wrote {args.out}")


if __name__ == "__main__":
    main()
