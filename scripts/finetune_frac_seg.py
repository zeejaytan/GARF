#!/usr/bin/env python3
"""Exp 11 — fine-tune GARF's FracSeg encoder to recognize WORN breaks.

Exp 10 found GARF's frozen fracture-aware encoder is blind to Juglet's worn
archaeological rims (fires on 0.57% of points vs 3.4% on fresh ceramics it
assembles well), because it was pretrained on synthetic FRESH breaks. Exp 10b
confirmed the cause: eroding fresh breaks toward worn drives the response down.

This fine-tunes the FracSeg feature extractor (PTv3 backbone + coarse_segmenter)
on labeled synthetic fractures WITH the worn-break augmentation
(`data.frac_erode_prob`), so it learns worn/smoothed breaks are still fracture
surfaces. The output is saved in the feature-extractor checkpoint format
(``{"state_dict": encoder+batch_norm+coarse_segmenter}``) so it drops straight
into the denoiser's ``feature_extractor_ckpt``.

Validation is external: re-run scripts/fracseg_introspection.py with the new
checkpoint — success = Juglet fired% rises toward the control's ~3.4%.

Usage
-----
  python scripts/finetune_frac_seg.py \
      --init-ckpt output/feature_extractor.ckpt \
      --data-root input/Fractura/bone_synthetic.hdf5 \
      --categories pig hip leg rib vertebra \
      --out-dir output/frac_seg_worn_<stamp> \
      --epochs 60 --batch-size 16 --lr 2e-5 \
      --erode-prob 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


def build_fracseg(init_ckpt: str, lr: float):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import open_dict

    cfg_dir = str((Path(__file__).resolve().parent.parent / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="model/frac_seg")
    with open_dict(cfg):
        cfg.model.optimizer.lr = lr
    model = instantiate(cfg.model)
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location="cpu", weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.startswith(("encoder", "batch_norm", "coarse_segmenter"))]
        assert not bad, f"missing critical weights on init: {bad[:5]}"
        print(f"loaded init weights from {init_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def build_datamodule(args):
    from assembly.data.breaking_bad import BreakingBadDataModule

    return BreakingBadDataModule(
        data_root=args.data_root,
        categories=list(args.categories),
        min_parts=2,
        max_parts=args.max_parts,
        num_points_to_sample=args.num_points,
        min_points_per_part=20,
        sample_method="weighted",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multi_ref=False,
        frac_erode_prob=args.erode_prob,
        frac_erode_min=args.erode_min,
        frac_erode_max=args.erode_max,
    )


def export_feature_extractor(model, path: Path):
    """Save FracSeg weights in the denoiser's feature_extractor_ckpt format."""
    sd = {k: v.cpu() for k, v in model.state_dict().items()
          if k.startswith(("encoder", "batch_norm", "coarse_segmenter"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd}, path)
    print(f"exported feature extractor ({len(sd)} tensors) -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-ckpt", default="output/feature_extractor.ckpt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--erode-prob", type=float, default=0.6)
    ap.add_argument("--erode-min", type=float, default=0.3)
    ap.add_argument("--erode-max", type=float, default=1.0)
    ap.add_argument("--num-points", type=int, default=5000)
    ap.add_argument("--max-parts", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_fracseg(args.init_ckpt, args.lr)
    datamodule = build_datamodule(args)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.out_dir / "checkpoints"),
        filename="fracseg-{epoch:03d}-{train_loss:.4f}",
        every_n_epochs=args.save_every,
        save_top_k=-1,
        save_last=True,
        monitor=None,
    )
    logger = CSVLogger(save_dir=str(args.out_dir), name="logs")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=5,
        # No fracture-seg validation loop needed — remedy is validated externally
        # on Juglet via fracseg_introspection.py. Skip val (and its mesh collate).
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    print(f"=== fine-tuning FracSeg on {args.categories} with worn-break aug "
          f"(prob={args.erode_prob}, strength [{args.erode_min},{args.erode_max}]) ===")
    trainer.fit(model, datamodule=datamodule)

    export_feature_extractor(model, args.out_dir / "feature_extractor_worn.ckpt")
    print(f"\ndone. checkpoints in {args.out_dir}/checkpoints, "
          f"deployable extractor at {args.out_dir}/feature_extractor_worn.ckpt")


if __name__ == "__main__":
    main()
