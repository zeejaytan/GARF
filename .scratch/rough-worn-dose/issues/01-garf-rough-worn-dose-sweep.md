# 01: GARF rough/worn dose sweep, up to 8×

**Answers:** G1 (if training on rough or worn break faces changes how many Juglet sherds
GARF seats, the break-face surface is a named mechanism; if nothing moves, it is ruled out
for GARF too)
**Blocked by:** None (can start immediately)
**Status:** ready-for-agent

**What to build:** TORA's ticket 09 repeated on GARF (conservator, 2026-09-28): the
same training files (`TORA/dataset/u10_{noise,worn}_dPPP.hdf5`, the same pooled fresh
file), the same strengths — ¼×, ½×, 1×, 2×, 4×, 8× for rough and for worn — and the same
read-out: Juglet, the eight Fractura pots, the erosion ladder, 20 attempts each, scored by
`tora/scripts/own_place.py` as the **best attempt** (with how many reached it), best
attempts staged in visual-qa.

## Recipe (matched to TORA's, so the comparison is about the method, not the recipe)

- Start from `output/GARF.ckpt`. LoRA r128 / α256 / dropout 0.1 on the denoiser's
  attention layers in all 6 blocks (TORA: last 6 blocks — GARF has 6). Heads and shape
  embedding frozen. Fracture-reading encoder frozen, as GARF always has it.
- lr 2e-5, 20 passes, seed 42, last pass. GARF has no alignment term, so TORA's
  "alignment off" is already true here.
- Arms: untouched GARF; fresh (pooled, unaltered); rough ×6; worn ×6 = 13 trained.
- Deliberately NOT GARF's released `finetune.yaml` (lr 2e-4, 500 passes, heads trained):
  that is ~10× the dose of training, and the TORA run showed a heavy fine-tune damaging
  placement in general.

## Plumbing GARF lacks (build first, gate before any arm)

- **20 attempts per object** in one eval run (GARF's test_step makes one).
- **Scorer bridge:** write TORA's cloud file (`pts_gt`, `points_per_part`,
  `generations_pred`/`generations_proposed`, `name`) from GARF's predicted poses. GARF
  moves each sherd rigidly, so raw = solid.
- **Anchor:** sherd 0 (largest) pinned at its true pose, as in every TORA run.
- **Gate:** untouched GARF on Fractura — own_place's swap-allowed count reconciles with
  GARF's own part accuracy, attempt by attempt; one Juglet attempt rendered beside the
  conservator's reassembly before any arm trains.

## Damage gate: the fine-tune must not cost sherds by itself (conservator, 2026-09-28)

TORA's first adapters seated fewer Juglet sherds than the untouched model and lost sherds
on galli_pot (8 → 2 of 10). The main cause was its alignment term, about 95% of the loss;
ticket 07 turned it off. **GARF has no such term:** its training optimises only
`vec_mse_loss`, the placement flow (`assembly/models/denoiser/denoiser_flow_matching.py`
`_loss`). The "alignment" in `denoiser_base.py` test_step is a scoring-frame fit and does
not train. Two contributors remain possible and are gated rather than assumed away:

- **Recipe.** A dedicated experiment config, not `experiment=finetune`: its
  `modules_to_save` would train the heads and shape embedding at 2e-4. Check the trainable
  parameter list before step 1 — LoRA weights only.
- **Forgetting.** The fresh arm trains first, alone. It must seat **no fewer** sherds than
  untouched GARF by best of 20 on the Juglet and on each of the 8 Fractura pots. Any pot
  that loses a sherd means stop and report: the 12 rough/worn arms do not start.

### Result, seed 42 (2026-09-29): gate FAILED on narrow_bottle1

Untouched 31449317 COMPLETED 0:0 (13 min); fresh 31449318 COMPLETED 0:0 (32 min; 48 LoRA
tensors, 4.7 M params, `lora_only` check passed). In-job scoring failed: the cloud writer
stored one `part_ids` per sherd, not per point (fixed 9e3af7b; the 98 files relabelled, no
coordinates touched). Rescored on CPU, 31481372 COMPLETED 0:0. Reconcile: 0 mismatches,
0 borderline over 980 attempts per arm.

Own place, pot-size ruler, best of 20 (attempts reaching it; median):

| pot | untouched | fresh |
|---|---|---|
| Juglet | 4/9 (2; 2) | 6/9 (1; 2) |
| narrow_bottle1 | 6/12 (1; 4) | **2/12 (3; 1)** |
| galli_pot | 10/10 (1; 8) | 10/10 (5; 9) |
| narrow_bottle3 | 2/4 (1; 1) | 2/4 (2; 1) |
| blue_pot, narrow_bottle2, narrow_bottle4, pink_bowl, plate | all home | all home |

narrow_bottle1's whole spread moved down, not only the best. Looked at
(`artifacts/rwlora/nb1_best2.png`, own debugging view): untouched's best seats neck and upper
body with the lower sherds thrown below; fresh's best crowds sherds into the body, misplaced,
neck gone. A real change in placement, not a ruler fault. The Juglet gain rests on one
attempt of 20. Same pot TORA's generic adapter lost a sherd on.

Seed 7 repeat of the fresh training submitted: 31482030 (GARF cfc7b5a, polled). If
narrow_bottle1 drops again, the loss is the fine-tune; if not, seed 42 was an unlucky run.

**Conservator's decision (2026-09-29): run the 12 arms anyway.** Each rough/worn arm is
judged against the **retrained (fresh) GARF**, not released GARF, so the comparison isolates
the break surface from the fine-tune itself. narrow_bottle1 is a known casualty of the
fine-tune and is reported but not read as a rough/worn effect. Submitted at b3b8558:
noise_d025 31482363, d050 31482364, d100 31482365, d200 31482366, d400 31482367,
d800 31482368; worn_d025 31482369, d050 31482370, d100 31482371, d200 31482372,
d400 31482373, d800 31482374 (all polled).

## Results (2026-09-29)

All 13 jobs COMPLETED 0:0, ~34 min each: fresh seed 7 31482030; rough 31482363-68; worn
31482369-74. Every job: `lora_only` passed, reconcile 0 mismatches / 0 borderline over 980
attempts, in-job own_place ran (writer fix 9e3af7b). Table from
`logs/rwlora/rw_table.py`, own place on the pot-size ruler, best of 20 (attempts reaching it):

```
JUGLET + FRACTURA, best of 20 (attempts reaching it)
arm         Juglet     blue_pot  galli_pot narrow_bo narrow_bo narrow_bo narrow_bo pink_bowl plate     fract sum
untouched   4/9 (2)    5/5(20)   10/10(1)  6/12(1)   3/3(20)   2/4(1)    4/4(20)   3/3(20)   6/6(18)   39
fresh s42   6/9 (1)    5/5(20)   10/10(5)  2/12(3)   3/3(20)   2/4(2)    4/4(19)   3/3(20)   6/6(17)   35
fresh s7    6/9 (2)    5/5(20)   10/10(10) 2/12(3)   3/3(20)   2/4(6)    4/4(20)   3/3(20)   6/6(17)   35
rough 1/4   5/9 (1)    5/5(20)   10/10(9)  2/12(5)   3/3(20)   3/4(2)    4/4(20)   3/3(20)   6/6(20)   36
rough 1/2   7/9 (1)    5/5(20)   10/10(14) 5/12(1)   3/3(20)   3/4(2)    4/4(20)   3/3(20)   6/6(20)   39
rough 1     7/9 (4)    5/5(20)   10/10(11) 2/12(13)  3/3(20)   4/4(2)    4/4(20)   3/3(20)   6/6(20)   37
rough 2     5/9 (8)    5/5(20)   10/10(10) 2/12(8)   3/3(20)   4/4(1)    4/4(20)   3/3(20)   6/6(19)   37
rough 4     6/9 (1)    5/5(20)   10/10(10) 3/12(4)   3/3(20)   4/4(3)    4/4(20)   3/3(20)   6/6(19)   38
rough 8     4/9 (1)    5/5(20)   10/10(9)  2/12(3)   3/3(20)   3/4(1)    4/4(19)   3/3(20)   6/6(19)   36
worn 1/4    6/9 (1)    5/5(20)   10/10(8)  2/12(5)   3/3(20)   2/4(6)    4/4(19)   3/3(20)   6/6(14)   35
worn 1/2    5/9 (4)    5/5(20)   10/10(5)  2/12(4)   3/3(20)   2/4(5)    4/4(20)   3/3(20)   6/6(16)   35
worn 1      4/9 (3)    5/5(20)   10/10(9)  2/12(6)   3/3(20)   2/4(2)    4/4(20)   3/3(20)   6/6(14)   35
worn 2      7/9 (1)    5/5(20)   10/10(13) 2/12(4)   3/3(20)   3/4(8)    4/4(20)   3/3(20)   6/6(18)   36
worn 4      9/9 (1)    5/5(20)   10/10(15) 2/12(7)   3/3(20)   4/4(2)    4/4(20)   3/3(20)   6/6(19)   37
worn 8      7/9 (4)    5/5(20)   10/10(16) 2/12(3)   3/3(20)   4/4(1)    4/4(20)   2/3(20)   6/6(18)   36

LADDER: sum over pots of best-of-20 sherds home, per erosion level
arm         e000     e025     e050     e075     e100    
untouched   39/47    38/47    36/47    23/47    21/47   
fresh s42   38/47    36/47    34/47    25/47    19/47   
fresh s7    40/47    36/47    34/47    25/47    19/47   
rough 1/4   41/47    38/47    36/47    28/47    21/47   
rough 1/2   39/47    37/47    40/47    30/47    23/47   
rough 1     41/47    39/47    40/47    32/47    26/47   
rough 2     40/47    38/47    38/47    29/47    27/47   
rough 4     41/47    38/47    38/47    28/47    28/47   
rough 8     39/47    38/47    37/47    26/47    26/47   
worn 1/4    38/47    34/47    35/47    24/47    19/47   
worn 1/2    39/47    35/47    35/47    24/47    18/47   
worn 1      40/47    34/47    35/47    26/47    21/47   
worn 2      41/47    38/47    37/47    28/47    22/47   
worn 4      40/47    39/47    38/47    30/47    21/47   
worn 8      42/47    39/47    38/47    28/47    24/47
```

**Damage gate, settled:** seed 7 repeats seed 42 exactly on narrow_bottle1 (6 → 2 of 12)
and on the Juglet (4 → 6). The loss is the fine-tune, not an unlucky run. Every arm is read
against fresh (6/9 Juglet, 35 Fractura, both seeds), as the conservator decided.

**Juglet:** worn 4× best attempt seats 9 of 9 (1 of 20 attempts; next best arm 7). Looked at
(`artifacts/rwlora/jug_arms.png`, own debugging view): the whole juglet assembled, the
furthest sherd 5.1% of pot size (~3.3 mm) from home. Staged for the conservator as
`garf_juglet_worn4x_rw01` (meshes posed by `tora/.scratch/anchor-choice/scripts/pose_meshes.py`,
fit ≤ 0.021 mm). Otherwise the Juglet column is noisy: most arms' best rests on 1-4 attempts,
worn 1× is 4, worn 2× 7, worn 8× 7 — no clean dose curve on one pot.

**Ladder (the cross-check):** rough training lifts GARF on eroded breaks and the heaviest
erosion prefers heavier roughness: at e075 fresh 25/47 → rough 1× 32; at e100 fresh 19 →
rough 4× 28, rough 2× 27. Worn lifts e075 (30 at 4×) but not e100 (≤ 24). At e000-e025 all
arms sit within 3 sherds of each other.

**Fractura:** no rough/worn arm restores narrow_bottle1 (best 5/12 at rough ½×); narrow_bottle3
gains 2 → 4 of 4 at rough 1×-4× and worn 4×-8×; worn 8× loses one sherd of pink_bowl.

Weight: one training run per arm, one real pot, 20 attempts. The Juglet 9/9 is a single
attempt — a lead for the eye, not a result, until the conservator has looked and it repeats
at a second seed.

## Follow-up: seed-7 repeats and 16× (conservator, 2026-09-29: "both")

Advised against 16× first: rough 8× already zig-zags ±1 mm on a 2 mm wall and has passed
its peak; worn 16× trains on ~3 mm gaps against the Juglet's 0.17 mm. The open question is
whether worn 4×'s 9/9 repeats. Conservator chose both.

| run | job |
|---|---|
| worn 4× seed 7 (GARF) | 31526349 |
| rough 1× seed 7 (GARF) | 31526350 |
| build rough 16× `u10_noise_d1600` (TORA) | 31526347 |
| build worn 16× `u10_worn_d1600` (TORA) | 31526348 |

16× builds are rendered at the same two joins in the 5 mm window before linking into
`dataset/` or training. Readings, fixed before results:
- Worn 4× seed 7 reaches ≥ 8/9 on the Juglet → the 9/9 is a repeatable lead, stage and
  offer refute-finding; ≤ 6/9 (fresh's level) → one lucky run, as TORA's worn 4× was.
- Rough 1× seed 7 ≥ 7/9 and ladder e075 ≥ 30 → rough 1× holds for GARF as for TORA.
- 16× beats its 8× on the Juglet **and** e100 → the ceiling is past 8×; otherwise 8× was
  already past the peak and the sweep is bounded.

## What we already know, so it is not re-run

GARF's worn-break remedies (Exp 11–15: worn-trained encoder, co-adapted denoiser,
self-supervised Juglet adaptation) did not make Juglet mates separable. They used GARF's
own erosion on bone data and a pair oracle, not own place against the conservator's
reassembly. Rough (jitter) training has never been tried on GARF.

**Prediction that could fail:** earlier work put GARF's Juglet blindness in its encoder;
this sweep trains only the denoiser, as TORA's did. If rough helps TORA but nothing moves
GARF, that is consistent with the encoder being the limit — not evidence that rough
training is a TORA quirk.

- [x] Rough 8× built and one join rendered (TORA job 31449062 COMPLETED 0:0, 2026-09-28;
      `tora/artifacts/u10/r09/sectionsN8_{6_7,0_1}.png`: break face ±1 mm on a 2 mm wall,
      spikes ~0.7 mm proud of the wall faces — the extreme endpoint; linked as
      `dataset/u10_noise_d800.hdf5`)
- [x] Plumbing gate passed (2026-09-29): reconcile 0 mismatches over 980 attempts per arm;
      untouched best Juglet attempt drawn over its reference (`artifacts/rwlora/jug_untouched.png`):
      the reference is the juglet (neck, handle ring), 4 home sherds sit on it, the rest sink
      into the body — count and picture agree
- [x] Damage gate — failed, and confirmed at seed 7 (narrow_bottle1 6 → 2 of 12 both seeds);
      overridden by the conservator: arms judged against retrained GARF
- [x] 13 arms trained and evaluated; best-of-20 table (2026-09-29, Results above); beside TORA's below
- [ ] Best attempts staged for the conservator — worn 4× Juglet (9/9) staged 2026-09-29 as
      `garf_juglet_worn4x_rw01` (surface coincidence median 0.28 mm; gates check_annotations,
      check_stage_pair OK; check_loader / check_mesh_loader fail on their own missing fixtures,
      as since 2026-09-26). Awaiting the conservator's look; other arms not staged
