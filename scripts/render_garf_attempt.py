"""Draw one GARF attempt against the truth, and the input GARF was given (wear-beyond-juglet 05).

For a check that a pose was not copied from the truth: per pot, three columns in two views
  1. the input as the network receives it: each sherd recentred and turned at random, so an
     assembled pot here means the pose leaked into the input;
  2. the truth (true assembled pose);
  3. the attempt (coloured by sherd) over the truth (grey), the anchor sherd in black.
Per sherd it writes the attempt's centre offset (mm) and turn (degrees) against the truth, and how
far the input points sit from their true place (mm), so the picture and the numbers say the same.

The input is recovered from the run's json_results gt_trans_rots: GARF writes
pts_gt = q_gt * input + t_gt (denoiser_base._save_clouds), so input = q_gt^-1 (pts_gt - t_gt).
The anchor is the sherd with the most points (weighted.py, random_anchor=false).
mm per unit is copies.csv mm_per_unit; GARF does not rescale these sets (pot within +-0.5 units, so
meshes_max_scale = 1); the script prints pot_mm against the truth's longest side as a cross-check.

    python scripts/render_garf_attempt.py RUN_DIR --copies copies.csv --cards p149_d00 p148_d00 \
        --attempt 0 --out OUT_DIR          # --cards all: numbers for every pot, pictures for none
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def qrot(q, v):
    """Rotate points v (N,3) by scalar-first quaternion q."""
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T


def turn_deg(a, b):
    """Angle of the best rotation taking a onto b (same point order), degrees."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load(run: Path):
    out = {}
    for f in sorted((run / "clouds").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        card = str(d["name"]).split("/")[-1]
        js = json.load(open(run / "json_results" / f"{f.stem}.json"))
        out[card] = (d, js)
    return out


def per_sherd(d, js, attempt, mmu):
    ppp = d["points_per_part"]
    cuts = np.concatenate([[0], np.cumsum(ppp)])
    gt = d["pts_gt"].astype(float)
    pred = d["generations_pred"][attempt].astype(float)
    poses = np.asarray(js["gt_trans_rots"], dtype=float)
    rows, inp = [], np.zeros_like(gt)
    for i in range(len(ppp)):
        a, b = cuts[i], cuts[i + 1]
        t, q = poses[i, :3], poses[i, 3:]
        q_inv = q * np.array([1, -1, -1, -1])
        inp[a:b] = qrot(q_inv, gt[a:b] - t)
        rows.append(dict(
            sherd=i, points=int(ppp[i]),
            centre_mm=float(np.linalg.norm(pred[a:b].mean(0) - gt[a:b].mean(0)) * mmu),
            turn_deg=turn_deg(gt[a:b], pred[a:b]),
            input_from_true_mm=float(np.median(np.linalg.norm(inp[a:b] - gt[a:b], axis=1)) * mmu),
            input_centre_mm=float(np.linalg.norm(inp[a:b].mean(0)) * mmu),
        ))
    return gt, pred, inp, cuts, rows


def draw(card, gt, pred, inp, cuts, anchor, rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    n = len(cuts) - 1
    rng = np.random.default_rng(0)
    views = [(15, -60, "side"), (90, -90, "top")]
    fig = plt.figure(figsize=(15, 10))
    lim = np.abs(gt).max() * 1.05
    for r, (el, az, vname) in enumerate(views):
        for c, label in enumerate(["input as GARF receives it", "truth", "attempt (colour) over truth (grey)"]):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            for i in range(n):
                a, b = cuts[i], cuts[i + 1]
                k = rng.permutation(b - a)[:1500] + a
                col = "black" if i == anchor else cmap(i % 10)
                if c == 0:
                    ax.scatter(*inp[k].T, s=0.6, color=col)
                elif c == 1:
                    ax.scatter(*gt[k].T, s=0.6, color=col)
                else:
                    ax.scatter(*gt[k].T, s=0.4, color="0.8")
                    ax.scatter(*pred[k].T, s=0.6, color=col)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.view_init(elev=el, azim=az)
            ax.set_title(f"{label} ({vname})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    txt = "sherd  points  centre off mm  turn deg  input-from-true mm\n" + "\n".join(
        f"{x['sherd']:>5}{'*' if x['sherd'] == anchor else ' '} {x['points']:>6}  {x['centre_mm']:>12.2f}"
        f"  {x['turn_deg']:>8.2f}  {x['input_from_true_mm']:>12.1f}" for x in rows)
    fig.text(0.01, 0.01, txt + "\n* anchor (pinned at its true pose; black)", family="monospace", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--copies", type=Path, required=True)
    ap.add_argument("--cards", nargs="+", required=True)
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = load(a.run)
    if not data:
        print(f"no clouds in {a.run}")
        return 1
    # set name -> copies.csv condition: csc_whole_worn -> worn, csc_whole20_worn4 -> worn4
    cond = re.sub(r"^csc_whole\d*_", "", next(iter(data.values()))[0]["name"].item().split("/")[0])
    meta = {(r["card"], r["condition"]): r for r in csv.DictReader(open(a.copies))}
    cards = sorted(data) if a.cards == ["all"] else a.cards
    pictures = a.cards != ["all"]
    summary = []
    for card in cards:
        if card not in data:
            print(f"{card}: not in run"); return 1
        d, js = data[card]
        m = meta[(card, cond)]
        mmu, pot_mm = float(m["mm_per_unit"]), float(m["pot_mm"])
        gt, pred, inp, cuts, rows = per_sherd(d, js, a.attempt, mmu)
        anchor = int(np.argmax(d["points_per_part"]))
        side = float((gt.max(0) - gt.min(0)).max()) * mmu
        acc = float(d["garf_part_acc"][a.attempt])
        print(f"== {card} ({cond}) attempt {a.attempt}: {len(rows)} sherds, GARF part_acc {acc:.3f}; "
              f"pot_mm {pot_mm} vs truth's longest side {side:.1f} mm (sampled)")
        for x in rows:
            print(f"   sherd {x['sherd']}{'*' if x['sherd'] == anchor else ' '} centre {x['centre_mm']:.2f} mm, "
                  f"turn {x['turn_deg']:.2f} deg; input {x['input_from_true_mm']:.1f} mm from true place "
                  f"(input centre at {x['input_centre_mm']:.2f} mm from origin)")
        free = [x for x in rows if x["sherd"] != anchor]
        summary.append((card, len(rows), max(x["centre_mm"] for x in free), max(x["turn_deg"] for x in free),
                        min(x["input_from_true_mm"] for x in free)))
        if pictures:
            p = a.out / f"garf_attempt_{cond}_{card}_a{a.attempt}.png"
            draw(card, gt, pred, inp, cuts, anchor, rows, p,
                 f"{card} {cond}, GARF attempt {a.attempt + 1} (run {a.run.parent.name}), pot {pot_mm} mm")
            print(f"   picture {p}")
    print("\npot        sherds  worst centre mm  worst turn deg  closest input mm (non-anchor)")
    for s in summary:
        print(f"{s[0]:<10} {s[1]:>6}  {s[2]:>15.2f}  {s[3]:>14.2f}  {s[4]:>14.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
