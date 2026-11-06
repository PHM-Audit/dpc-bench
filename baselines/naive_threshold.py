#!/usr/bin/env python3
"""Naive threshold baseline — simple per-record sanity checks only.

Deliberately weak: no cross-record statistics, no physics envelope fitting,
no unit-inference. Demonstrates that DPC-Bench is not trivially saturated.
"""
import json, os

def load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    preds = {"method": "naive_threshold_v0"}
    P = {}

    # quat_norm: flag |norm-1| > 0.05 as L2 (misses subtle 1.005-1.02 corruptions)
    recs = load(os.path.join(data_dir, "quat_norm.jsonl"))
    P["quat_norm"] = {}
    for r in recs:
        q = r["quat"]; n = sum(c*c for c in q) ** 0.5
        P["quat_norm"][str(r["rid"])] = "L2" if abs(n - 1.0) > 0.05 else "clean"

    # unit_semantics: naive has NO cross-record view → predicts all clean (misses 100%)
    P["unit_semantics"] = {str(r["rid"]): "clean" for r in load(os.path.join(data_dir, "unit_semantics.jsonl"))}

    # time_align: duplicate ts / backward ts / gap > 100ms
    recs = load(os.path.join(data_dir, "time_align.jsonl"))
    P["time_align"] = {}
    for j, r in enumerate(recs):
        if j == 0:
            P["time_align"][str(r["rid"])] = "clean"
        else:
            dt = r["ts"] - recs[j-1]["ts"]
            P["time_align"][str(r["rid"])] = "L1" if (dt <= 0 or dt > 100) else "clean"

    # frame_drop: frame_idx continuity
    recs = load(os.path.join(data_dir, "frame_drop.jsonl"))
    P["frame_drop"] = {}
    for j, r in enumerate(recs):
        if j == 0:
            P["frame_drop"][str(r["rid"])] = "clean"
        else:
            P["frame_drop"][str(r["rid"])] = "L1" if r["frame_idx"] != recs[j-1]["frame_idx"] + 1 else "clean"

    # kinematic_teleport: naive applied-speed > 100 m/s
    recs = load(os.path.join(data_dir, "kinematic_teleport.jsonl"))
    P["kinematic_teleport"] = {}
    for j in range(1, len(recs)):
        dt = recs[j]["ts"] - recs[j-1]["ts"] or 1e-9
        d = sum((a-b)**2 for a, b in zip(recs[j]["pos"], recs[j-1]["pos"])) ** 0.5
        v = d / dt
        P["kinematic_teleport"][str(recs[j]["rid"])] = "L2" if v > 100 else "clean"
    P["kinematic_teleport"][str(recs[0]["rid"])] = "clean"

    # mixed_l3: None / missing / non-numeric speed
    recs = load(os.path.join(data_dir, "mixed_l3.jsonl"))
    P["mixed_l3"] = {}
    for r in recs:
        v = r.get("speed")
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        P["mixed_l3"][str(r["rid"])] = "clean" if ok else "L3"

    # vehicle_real: naive — only malformed type check (misses teleport/unit flip)
    recs = load(os.path.join(data_dir, "vehicle_real.jsonl"))
    P["vehicle_real"] = {}
    for r in recs:
        v = r.get("vel")
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        P["vehicle_real"][str(r["rid"])] = "clean" if ok else "L3"

    preds["predictions"] = P
    out = os.path.join(os.path.dirname(__file__), "..", "results", "pred_naive.json")
    json.dump(preds, open(out, "w", encoding="utf-8"))
    print("wrote", out)

if __name__ == "__main__":
    main()
