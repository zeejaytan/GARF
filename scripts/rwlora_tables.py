"""Tables for wear-beyond-juglet 05 from rwlora_arm.slurm logs (own_place's printed per-draw rows).

Per condition (set), pot x model, as the pre-registration asks:
  best right-way-round of all attempts (attempts reaching it), best own place (attempts),
  full reassemblies (n of n right way round), and the mean own-place count per presentation
  (one mean per draw seed, 20 attempts each).
Counts are own_place's SOLID-sherd rows (raw rows are printed too and checked equal here).
Arm and seed come from each "--- <set> run dir: .../rwlora_eval_<arm>_ds<seed>_<set>_<job>/..." line.

    python scripts/rwlora_tables.py LOG [LOG ...] [--tsv out.tsv]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

RUN = re.compile(r"^--- (csc_whole_\w+) run dir: \S*rwlora_eval_(.+?)_ds(\d+)_csc_whole_\w+?_(\d+)/")
POT = re.compile(r"^(csc_whole_\w+)/(p\d+_d\d+): (\d+) sherds, (\d+) draws")
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s")
ORDER = ["untouched", "fresh", "noise_d100", "worn_d400"]
NAMES = {"untouched": "released", "fresh": "clean-break", "noise_d100": "rough 1x", "worn_d400": "worn 4x"}


def parse(path):
    """yield (set, arm, seed, pot, n, kind, [(own, rwr), ...])"""
    cur = pot = kind = None
    rows = []
    def flush():
        if pot and kind and rows:
            return (cur[0], cur[1], cur[2], pot[0], pot[1], kind, list(rows))
    for line in open(path, errors="replace"):
        m = RUN.match(line)
        if m:
            r = flush(); rows.clear(); pot = kind = None
            if r: yield r
            cur = (m[1], m[2], int(m[3]))
            continue
        m = POT.match(line)
        if m and cur:
            r = flush(); rows.clear(); kind = None
            if r: yield r
            pot = (m[2], int(m[3]))
            continue
        if "raw output" in line or "solid sherds" in line:
            r = flush(); rows.clear()
            if r: yield r
            kind = "raw" if "raw output" in line else "solid"
            continue
        m = ROW.match(line)
        if m and pot and kind:
            rows.append((int(m[2]), int(m[4])))
    r = flush()
    if r: yield r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--tsv", type=Path)
    a = ap.parse_args()
    data = defaultdict(dict)  # (set, arm, pot) -> {(kind, seed): rows}
    sizes = {}
    for log in a.logs:
        for set_, arm, seed, pot, n, kind, rows in parse(log):
            key = (set_, arm, pot)
            if (kind, seed) in data[key]:
                print(f"DUPLICATE {key} {kind} seed {seed} in {log}", file=sys.stderr); return 1
            data[key][(kind, seed)] = rows
            sizes[pot] = n
    bad = 0
    for key, d in data.items():
        for (kind, seed), rows in d.items():
            if kind == "solid" and d.get(("raw", seed)) != rows:
                bad += 1
                print(f"raw != solid: {key} seed {seed}", file=sys.stderr)
            if len(rows) != 20:
                print(f"{key} {kind} seed {seed}: {len(rows)} draws, not 20", file=sys.stderr)
    out = []
    sets = sorted({k[0] for k in data})
    arms = [x for x in ORDER if any(k[1] == x for k in data)] + sorted({k[1] for k in data} - set(ORDER))
    for set_ in sets:
        pots = sorted({k[2] for k in data if k[0] == set_})
        print(f"\n=== {set_}: best right way round (attempts reaching it) / best own place (attempts) / "
              f"full n of n (of attempts) / mean own per presentation by seed")
        for pot in pots:
            n = sizes[pot]
            print(f"{pot} ({n} sherds)")
            for arm in arms:
                d = data.get((set_, arm, pot))
                if not d:
                    continue
                seeds = sorted(s for k, s in d if k == "solid")
                rows = [r for s in seeds for r in d[("solid", s)]]
                own = [o for o, _ in rows]; rwr = [w for _, w in rows]
                b_r, b_o = max(rwr), max(own)
                means = [sum(o for o, _ in d[("solid", s)]) / len(d[("solid", s)]) for s in seeds]
                full = sum(w == n for w in rwr)
                print(f"   {NAMES.get(arm, arm):<12} rwr {b_r}/{n} ({rwr.count(b_r)} of {len(rows)})  "
                      f"own {b_o}/{n} ({own.count(b_o)})  full {full}/{len(rows)}  seeds {seeds[0]}-{seeds[-1]} "
                      f"means " + " ".join(f"{m:.2f}" for m in means))
                out.append([set_, pot, str(n), NAMES.get(arm, arm), str(b_r), str(rwr.count(b_r)), str(b_o),
                            str(own.count(b_o)), str(full), str(len(rows)), ",".join(map(str, seeds)),
                            ",".join(f"{m:.2f}" for m in means)])
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("set\tpot\tsherds\tmodel\tbest_rwr\tbest_rwr_attempts\tbest_own\tbest_own_attempts\t"
                    "full\tattempts\tseeds\tmean_own_per_seed\n")
            f.writelines("\t".join(r) + "\n" for r in out)
    print(f"\n{bad} raw/solid disagreements")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
