"""Plumbing gate for the GARF -> TORA scorer bridge (.scratch/rough-worn-dose ticket 01).

GARF's own part accuracy and own_place's own-place count are the same measurement
(per-sherd chamfer of the sherd against ITS OWN true place, both directions summed,
threshold 0.01) on different rulers: GARF's units vs a fraction of pot size. Undo the
pot-size scaling and the two must agree attempt by attempt, sherd count for sherd
count. If they do, the cloud files carry GARF's attempts faithfully, and own_place's
pot-size score of them is a like-for-like number beside TORA's.

A sherd within 5% of the threshold may flip between float16 poses (GARF) and float32
(here); those are reported, not failed.

    python scripts/garf_clouds_reconcile.py <eval log dir with clouds/> [--tora ../tora]
Exit 0 = reconciled.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

TAU = 0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--tora", type=Path, default=Path(__file__).resolve().parents[2] / "tora")
    a = ap.parse_args()
    sys.path.insert(0, str(a.tora / "scripts"))
    from own_place import cost_matrix  # noqa: E402
    from readout import part_slices  # noqa: E402

    files = sorted((a.run / "clouds").glob("*.npz"))
    if not files:
        print(f"no clouds in {a.run / 'clouds'}")
        return 1
    bad = borderline = attempts = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        slices = part_slices(d["points_per_part"])
        n = int(d["garf_n_parts"])
        gt = d["pts_gt"].astype(float)
        garf_counts = []
        for g, (pred, acc) in enumerate(zip(d["generations_pred"], d["garf_part_acc"])):
            attempts += 1
            diag = np.diag(cost_matrix(gt, pred.astype(float), slices))
            ours = int((diag < TAU).sum())
            garf = int(round(float(acc) * n))
            garf_counts.append(garf)
            if ours != garf:
                near = np.abs(diag - TAU) < 0.05 * TAU
                if near.any() and abs(ours - garf) <= int(near.sum()):
                    borderline += 1
                    print(f"{d['name']} attempt {g}: GARF {garf} vs {ours} of {n}, "
                          f"borderline sherds {np.flatnonzero(near).tolist()}")
                else:
                    bad += 1
                    print(f"MISMATCH {d['name']} attempt {g}: GARF {garf} vs {ours} of {n}; "
                          f"chamfer per sherd {np.round(diag, 5).tolist()}")
        print(f"{d['name']}: {len(garf_counts)} attempts, GARF best {max(garf_counts)} of {n}")
    print(f"{attempts} attempts over {len(files)} objects: {bad} mismatches, "
          f"{borderline} borderline")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
