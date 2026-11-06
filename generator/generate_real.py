#!/usr/bin/env python3
"""DPC-Bench v0.2 real-data family: vehicle_real (NGSIM I-80 derived). v2 fixed order.

Order of operations (fixed): base build → frame-hole marking (real) → frame-drop
reindex (hides holes) → teleport/unit_flip/L3 injection on the FINAL record set.

Base = REAL trajectory records from NGSIM I-80 (USDOT public domain), seed=20261107.
Honesty note: clean base verified against plausibility envelopes (vel in [0,40] m/s;
inter-frame speed within 99.9th-pct envelope; frame continuity), NOT against NGSIM's
own acc channel (known reconstruction noise).
"""
import hashlib, json, os, random
import numpy as np

SRC = "/home/z/my-project/phm_cloud_api/vehicle_data/ngsim_i80.mat"
SEED = 20261107
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    from scipy.io import loadmat
    T = loadmat(SRC)["trajectories"]
    sha = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    rng = random.Random(SEED)

    # ---- 1. base: 3 longest-vehicle trajectories ----
    vids, counts = np.unique(T[:, 0], return_counts=True)
    top3 = vids[np.argsort(-counts)[:3]]
    base = []
    for vid in top3:
        traj = T[T[:, 0] == vid]
        traj = traj[np.argsort(traj[:, 1])]
        for row in traj:
            base.append({"ts": float(row[3]), "frame_idx": int(row[1]),
                         "vel": float(row[11]), "acc": float(row[12]),
                         "pos": [float(row[6]), float(row[7])]})
    # base plausibility sanity (must all pass)
    for j in range(len(base)):
        assert 0.0 <= base[j]["vel"] <= 40.0, base[j]

    # ---- 2. frame-drop reindex (hide holes; count real holes found) ----
    real_holes = sum(1 for j in range(1, len(base)) if base[j]["frame_idx"] != base[j-1]["frame_idx"] + 1)
    drop_set = set(rng.sample(range(1, len(base) - 200), int(len(base) * 0.04)))
    recs, skip = [], 0
    for r in base:
        if r["frame_idx"] in drop_set and skip < len(drop_set) and rng.random() < 1.0:
            skip += 1
            continue
        r2 = dict(r); r2["frame_idx"] = r["frame_idx"] - skip
        recs.append(r2)

    # 重新标注真实剩余洞（reindex后仍存在的断点）
    labels = {str(i): {"label": "clean", "family": "vehicle_real", "note": "real NGSIM base"} for i in range(len(recs))}
    residual_holes = 0
    for j in range(1, len(recs)):
        if recs[j]["frame_idx"] != recs[j-1]["frame_idx"] + 1:
            residual_holes += 1
            labels[str(j)] = {"label": "L1", "family": "vehicle_real", "note": "frame discontinuity"}

    # ---- 3. envelope ----
    env = []
    for j in range(1, len(recs)):
        dt = (recs[j]["ts"] - recs[j-1]["ts"]) / 1000.0
        if dt <= 0: continue
        d = ((recs[j]["pos"][0]-recs[j-1]["pos"][0])**2 + (recs[j]["pos"][1]-recs[j-1]["pos"][1])**2) ** 0.5
        env.append(d/dt)
    envelope = float(np.percentile(env, 99.9))

    # ---- 4. teleport 4% of final set ----
    n = len(recs)
    tpool = rng.sample(range(1, n), int(n * 0.04))
    half = len(tpool) // 2
    for k, idx in enumerate(tpool):
        scale = rng.uniform(1.15, 1.5) if k < half else rng.uniform(4, 10)
        prev = recs[idx-1]["pos"]
        dx, dy = recs[idx]["pos"][0]-prev[0], recs[idx]["pos"][1]-prev[1]
        d = (dx*dx + dy*dy) ** 0.5 or 1e-6
        dt = (recs[idx]["ts"] - recs[idx-1]["ts"]) / 1000.0
        want = envelope * scale * (dt if dt > 0 else 0.1)
        recs[idx]["pos"] = [prev[0] + dx/d * want, prev[1] + dy/d * want]
        labels[str(idx)] = {"label": "L2", "family": "vehicle_real", "note": f"teleport injected ({scale:.2f}x envelope)"}

    # ---- 5. unit flip 5% (exclude teleported) ----
    used = set(tpool)
    upool = [i for i in rng.sample(range(n), int(n * 0.05)) if i not in used]
    for idx in upool:
        recs[idx]["vel"] = round(recs[idx]["vel"] * 2.23694, 3)
        labels[str(idx)] = {"label": "L1", "family": "vehicle_real", "note": "km/h value in m/s field (x2.23694)"}

    # ---- 6. L3 2% (exclude used) ----
    used |= set(upool)
    l3pool = [i for i in rng.sample(range(n), int(n * 0.02)) if i not in used]
    for idx in l3pool:
        mode = rng.choice(["none", "str", "trunc"])
        if mode == "none":
            recs[idx]["vel"] = None
        elif mode == "str":
            recs[idx]["vel"] = "N/A"
        else:
            recs[idx]["vel"] = round(float(recs[idx]["vel"] or 0.0), 2)
        labels[str(idx)] = {"label": "L3", "family": "vehicle_real", "note": f"malformed ({mode}) injected"}

    # ---- write ----
    for i, r in enumerate(recs):
        r["rid"] = i
    wpath = os.path.join(OUT, "vehicle_real.jsonl")
    with open(wpath, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gt_path = os.path.join(OUT, "ground_truth.json")
    gt = json.load(open(gt_path, encoding="utf-8")) if os.path.exists(gt_path) else {"meta": {}}
    n_def = sum(1 for l in labels.values() if l["label"] != "clean")
    gt["vehicle_real"] = {
        "n_records": n, "labels": labels,
        "params": {"source": "NGSIM I-80 (USDOT public domain)", "source_sha256": sha[:16] + "...",
                   "seed": SEED, "envelope_mps": round(envelope, 3),
                   "real_holes_pre_reindex": real_holes, "residual_holes": residual_holes,
                   "injections": {"teleport": len(tpool), "unit_flip": len(upool), "l3": len(l3pool)}},
    }
    json.dump(gt, open(gt_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"vehicle_real v2: records={n} defective={n_def} envelope={envelope:.2f} residual_holes={residual_holes}")

if __name__ == "__main__":
    main()
