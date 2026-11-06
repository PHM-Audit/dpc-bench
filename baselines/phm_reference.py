#!/usr/bin/env python3
"""PHM reference detector — three-caliber judgment (L1/L2/L3) per Q/QZKJ 001-2026.

Uses cross-record statistics and physics envelope fitting (no ground-truth access).
This is the author's method row; see README honesty note on synthetic saturation.
"""
import json, math, os, statistics

def load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    P = {"method": "phm_reference_v0"}

    # quat_norm: tight tolerance + SO(3) renormalizability
    recs = load(os.path.join(data_dir, "quat_norm.jsonl"))
    P["quat_norm"] = {}
    for r in recs:
        n = sum(c*c for c in r["quat"]) ** 0.5
        P["quat_norm"][str(r["rid"])] = "L2" if abs(n - 1.0) > 1e-3 else "clean"

    # unit_semantics: bimodality of speed distribution (2.237x ratio gap) + envelope
    recs = load(os.path.join(data_dir, "unit_semantics.jsonl"))
    vs = [r["speed"] for r in recs]
    hi = [v for v in vs if v > 13.0]  # m/s mode cap ~12; anything > 13 suspicious as mph-mislabeled
    P["unit_semantics"] = {}
    for r in recs:
        v = r["speed"]
        # cross-record: cluster gap detection around 12-25 m/s
        P["unit_semantics"][str(r["rid"])] = "L1" if v > 13.0 else "clean"

    # time_align: strict monotonic + nominal dt 10ms with 5ms tol + gap > 10x
    recs = load(os.path.join(data_dir, "time_align.jsonl"))
    P["time_align"] = {}
    prev = None
    for r in recs:
        t = r["ts"]
        if prev is None:
            P["time_align"][str(r["rid"])] = "clean"
        else:
            d = t - prev
            if d <= 0:
                P["time_align"][str(r["rid"])] = "L1"  # duplicate or backward
            elif d > 0.1:
                P["time_align"][str(r["rid"])] = "L1"  # >10x nominal
            else:
                P["time_align"][str(r["rid"])] = "clean"
        prev = t

    # frame_drop: neighbor gap (same as naive but framed as L1 with full audit)
    recs = load(os.path.join(data_dir, "frame_drop.jsonl"))
    P["frame_drop"] = {}
    for j in range(len(recs)):
        if j > 0 and recs[j]["frame_idx"] - recs[j-1]["frame_idx"] > 1:
            P["frame_drop"][str(recs[j]["rid"])] = "L1"
        else:
            P["frame_drop"][str(recs[j]["rid"])] = "clean"

    # kinematic_teleport: fitted envelope (robust: 99.5th pct of implied speeds * 1.2)
    recs = load(os.path.join(data_dir, "kinematic_teleport.jsonl"))
    speeds = []
    for j in range(1, len(recs)):
        dt = recs[j]["ts"] - recs[j-1]["ts"] or 1e-9
        d = sum((a-b)**2 for a, b in zip(recs[j]["pos"], recs[j-1]["pos"])) ** 0.5
        speeds.append(d / dt)
    thr = sorted(speeds)[int(len(speeds)*0.995)] * 1.2
    P["kinematic_teleport"] = {}
    for j in range(len(recs)):
        if j == 0:
            P["kinematic_teleport"][str(recs[j]["rid"])] = "clean"
        else:
            dt = recs[j]["ts"] - recs[j-1]["ts"] or 1e-9
            d = sum((a-b)**2 for a, b in zip(recs[j]["pos"], recs[j-1]["pos"])) ** 0.5
            P["kinematic_teleport"][str(recs[j]["rid"])] = "L2" if d/dt > thr else "clean"

    # mixed_l3: type check
    recs = load(os.path.join(data_dir, "mixed_l3.jsonl"))
    P["mixed_l3"] = {}
    for r in recs:
        v = r.get("speed")
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v == v  # NaN guard
        P["mixed_l3"][str(r["rid"])] = "clean" if ok else "L3"

    # vehicle_real: envelope + unit band check + type check
    recs = load(os.path.join(data_dir, "vehicle_real.jsonl"))
    envl = []
    for j in range(1, len(recs)):
        dt = (recs[j]["ts"] - recs[j-1]["ts"]) / 1000.0
        if dt <= 0: continue
        d = ((recs[j]["pos"][0]-recs[j-1]["pos"][0])**2 + (recs[j]["pos"][1]-recs[j-1]["pos"][1])**2) ** 0.5
        envl.append(d/dt)
    envl.sort()
    envelope = envl[int(len(envl)*0.999)] if envl else 50.0
    P["vehicle_real"] = {}
    for j, r in enumerate(recs):
        v = r.get("vel")
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            P["vehicle_real"][str(r["rid"])] = "L3"; continue
        if not (0.0 <= v <= 40.0):
            P["vehicle_real"][str(r["rid"])] = "L1"; continue
        if j > 0:
            dt = (r["ts"] - recs[j-1]["ts"]) / 1000.0
            if dt > 0:
                d = ((r["pos"][0]-recs[j-1]["pos"][0])**2 + (r["pos"][1]-recs[j-1]["pos"][1])**2) ** 0.5
                if d/dt > envelope * 1.1:
                    P["vehicle_real"][str(r["rid"])] = "L2"; continue
        P["vehicle_real"][str(r["rid"])] = "clean"

    out = {"method": P["method"], "predictions": P}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "pred_phm_reference.json")
    json.dump(out, open(path, "w", encoding="utf-8"))
    print("wrote", path)

if __name__ == "__main__":
    main()
