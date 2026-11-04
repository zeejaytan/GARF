"""Are GARF's inputs the same across models for one draw seed? (wear-beyond-juglet 05, read only)

The pre-registration pairs clean-break and adapter on the same seed, which assumes the seed fixes
the presentation: the sampled points and each sherd's input turn. For every run dir given, this
groups runs by (set, draw seed) from the dir name (rwlora_eval_<arm>_ds<seed>_<set>_<job>) and,
per pot, compares across arms:
  - pts_gt, points_per_part (the sampled points, in the true pose) from clouds/<i>.npz;
  - gt_trans_rots from json_results/<i>.json (the pose that undoes each sherd's input turn and
    shift, so equal poses + equal pts_gt = equal input).
Byte-identical is reported as such; otherwise the largest difference.

    python scripts/compare_garf_inputs.py logs/rwlora/rwlora_eval_*_ds1_csc_whole_*_3430474?/version_0 ...
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"rwlora_eval_(?P<arm>.+?)_ds(?P<seed>\d+)_(?P<set>csc_whole_\w+?)_(?P<job>\d+)$")


def load(run: Path):
    out = {}
    for f in sorted((run / "clouds").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        js = json.load(open(run / "json_results" / f"{f.stem}.json"))
        out[str(d["name"])] = (d["pts_gt"], d["points_per_part"], np.asarray(js["gt_trans_rots"], dtype=np.float64))
    return out


def main(argv) -> int:
    groups = defaultdict(list)
    for p in map(Path, argv):
        m = PAT.match(p.parent.name if p.name.startswith("version_") else p.name)
        if not m:
            print(f"skip (name not recognised): {p}")
            continue
        groups[(m["set"], int(m["seed"]))].append((f"{m['arm']}@{m['job']}", p))
    bad = 0
    for (set_, seed), runs in sorted(groups.items()):
        print(f"== {set_} seed {seed}: {', '.join(a for a, _ in runs)}")
        if len(runs) < 2:
            print("   only one run, nothing to compare")
            continue
        data = [(a, load(p)) for a, p in runs]
        ref_arm, ref = data[0]
        for pot in sorted(ref):
            line = []
            for arm, d in data[1:]:
                if pot not in d:
                    line.append(f"{arm}: MISSING"); bad += 1; continue
                g0, n0, q0 = ref[pot]; g1, n1, q1 = d[pot]
                same = (g0.tobytes() == g1.tobytes() and n0.tobytes() == n1.tobytes()
                        and q0.tobytes() == q1.tobytes())
                if same:
                    line.append(f"{arm}: identical")
                else:
                    bad += 1
                    dg = float(np.abs(g0 - g1).max()) if g0.shape == g1.shape else float("nan")
                    dq = float(np.abs(q0 - q1).max()) if q0.shape == q1.shape else float("nan")
                    line.append(f"{arm}: DIFFERENT (points_per_part equal {np.array_equal(n0, n1)}, "
                                f"max |pts_gt| diff {dg:.3g}, max |pose| diff {dq:.3g})")
            print(f"   {pot.split('/')[-1]} vs {ref_arm}: " + "; ".join(line))
    print(f"\n{bad} pot comparisons differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
