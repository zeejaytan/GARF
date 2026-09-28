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

Conservator chose (2026-09-29): repeat the fresh training at seed 7 before deciding —
31482030 (GARF cfc7b5a, polled). If narrow_bottle1 drops again, the loss is the fine-tune;
if not, seed 42 was an unlucky run.

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
- [ ] Plumbing gate passed (reconcile + render) — reconcile passed 2026-09-29; Juglet render
      beside the conservator's reassembly still to do
- [ ] Damage gate passed — seed 42 failed on narrow_bottle1 (6 → 2 of 12); seed 7 running
- [ ] 13 arms trained and evaluated; best-of-20 table beside TORA's
- [ ] Best attempts staged for the conservator
