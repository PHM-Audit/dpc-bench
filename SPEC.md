# DPC-Bench v0.2 — Specification

## 1. Label taxonomy (per Q/QZKJ 001-2026, three-caliber judgment)

| Label | Meaning | Examples in DPC-Bench |
|---|---|---|
| clean | passes physical-consistency checks | — |
| L1 | data-integrity/quality defect; values or organization deviate from declared schema/semantics | mixed m/s vs mph records; duplicate/backward/large-gap timestamps; dropped frame indices |
| L2 | physical violation; breaks a physics constraint regardless of format validity | non-unit quaternions (breaks SO(3)); position jumps beyond the kinematic envelope |
| L3 | malformed/unparseable | None/NaN/truncated/non-numeric fields |

## 2. Task families (v0.1)

| family | records | defective | schema | defect mechanics (seed=20261106) |
|---|---|---|---|---|
| quat_norm | 1200 | 180 | {rid, ts, quat[4]} | 10% gross scale 1.1–1.5; 5% subtle 1.005–1.02; rest unit |
| unit_semantics | 1200 | 216 | {rid, ts, speed} | 18% records in km/h mislabeled as m/s (values ×2.237); ~30% of those below the 13 m/s plausibility cutoff (hard) |
| time_align | 1200 | 180 | {rid, ts} | 15% with duplicates/backward/large gaps (nominal 10 ms ± jitter) |
| frame_drop | 1200 | 132 | {rid, frame_idx} | 11% frames removed silently (sparse, mostly single-frame) |
| kinematic_teleport | 1200 | 96 | {rid, ts, pos[3]} | 8% teleports; ~half borderline (inside naive threshold, outside 99.5-pct envelope ×1.2) |
| mixed_l3 | 1200 | 84 | {rid, ts, speed} | 7% None/NaN/"N/A"/truncated |
| vehicle_real (v0.2, real base) | 2967 | 320 | {rid, ts, frame_idx, vel, acc, pos[2]} | base=NGSIM I-80 rows (USDOT public domain, sha256 93c547a9…); injections: ~4% teleports (half borderline vs 99.9-pct envelope ×1.1), ~5% km/h-as-m/s unit flips (vel×2.237), ~2% L3; 2 real frame holes kept as L1; seed=20261107 |

## 3. Files

- `data/<family>.jsonl` — input streams (one JSON object per line)
- `data/ground_truth.json` — labels + generation parameters (maintainers only; participants should NOT tune on it; holdout protocol in v0.2)
- `generator/generate.py` — deterministic regeneration
- `eval/evaluate.py` — scorer
- `baselines/` — two reference rows

## 4. Prediction file format

```json
{
  "method": "your-method-name",
  "runtime_s": 12.3,
  "predictions": {
    "quat_norm":  { "0": "clean", "1": "L2", "...": "..." },
    "unit_semantics": { "...": "..." },
    "time_align": { "...": "..." },
    "frame_drop": { "...": "..." },
    "kinematic_teleport": { "...": "..." },
    "mixed_l3": { "...": "..." }
  }
}
```

Every record of every family must receive exactly one label.

## 5. Metrics

- `headline_macro_f1` = mean of per-class F1 over {clean, L1, L2, L3}
- `false_positive_rate_on_clean` = flagged-clean / total-clean (hard constraint target ≤ 0.001)
- per-class P/R/F1 + full confusion matrix
- runtime reported but not ranked in v0.1

## 6. Leaderboard rules

- One row per method version; PR or issue with predictions JSON
- State method family (rule-based / statistics / learned) in one sentence
- Self-reported runtime; spot-checks may be re-run by maintainers
- Scores are frozen to DPC-Bench vX.Y (v0.2 = synthetic seed 20261106 + vehicle_real seed 20261107). v0.2 resets the board.
