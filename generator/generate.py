#!/usr/bin/env python3
"""DPC-Bench v0.1 data generator — seeded, reproducible, stdlib-only.

Generates 6 task families of paired clean/corrupted records modeled on
REAL, publicly documented physical-consistency defects (see README table).

Usage:  python3 generator/generate.py [--out data] [--seed 20261106]
Output: data/<family>.jsonl  +  data/ground_truth.json
"""
import argparse, json, math, os, random

SEED_DEFAULT = 20261106
N_PER_FAMILY = 1200

def w_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- family 1: quat_norm (L2, SO(3) violation) ----------
# modeled after OpenDriveLab/AgiBot-World#164 (ee_base norm max 1.395)
def gen_quat_norm(rng, n, out_dir, gt):
    recs, labels = [], {}
    # 85% clean unit quaternions; 10% gross non-unit (scale 1.1~1.5); 5% subtle (1.005~1.02)
    n_gross, n_subtle = int(n * 0.10), int(n * 0.05)
    corrupt_ids = set(rng.sample(range(n), n_gross + n_subtle))
    gross_ids = set(list(corrupt_ids)[:n_gross]); subtle_ids = corrupt_ids - gross_ids
    t = 0.0
    for i in range(n):
        t += 0.01
        # random rotation via axis-angle
        axis = [rng.gauss(0,1) for _ in range(3)]
        m = math.sqrt(sum(a*a for a in axis)) or 1.0
        axis = [a/m for a in axis]
        ang = rng.uniform(-math.pi, math.pi)
        q = [math.cos(ang/2)] + [a*math.sin(ang/2) for a in axis]
        if i in gross_ids:
            s = rng.uniform(1.1, 1.5)
            q = [c*s for c in q]
            labels[str(i)] = {"label":"L2","family":"quat_norm","note":"norm %.3f"%(math.sqrt(sum(c*c for c in q)))}
        elif i in subtle_ids:
            s = rng.uniform(1.005, 1.02)
            q = [c*s for c in q]
            labels[str(i)] = {"label":"L2","family":"quat_norm","note":"subtle norm %.4f"%(math.sqrt(sum(c*c for c in q)))}
        else:
            labels[str(i)] = {"label":"clean","family":"quat_norm"}
        recs.append({"rid":i,"ts":round(t,3),"quat":[round(c,6) for c in q]})
    w_jsonl(os.path.join(out_dir,"quat_norm.jsonl"), recs)
    gt["quat_norm"] = {"labels":labels,"params":{"n":n,"gross":n_gross,"subtle":n_subtle,"real_world_source":"OpenDriveLab/AgiBot-World#164 (ee_base norm max 1.395, 88% frames off-unit)"}}

# ---------- family 2: unit_semantics (L1, mph vs m/s mixed) ----------
# modeled after NGSIM I-80 lane-8 velocity column recorded in mph while pipeline assumed m/s
def gen_unit_semantics(rng, n, out_dir, gt):
    recs, labels = [], {}
    n_corrupt = int(n * 0.18)
    corrupt_ids = set(rng.sample(range(n), n_corrupt))
    t = 0.0
    for i in range(n):
        t += 0.1
        if i in corrupt_ids:
            # genuine m/s value multiplied by 2.23694 (mph) — plausible in overlap zone,
            # detectable only via cross-record distribution (bimodal) & physics envelope
            v_ms = rng.uniform(2.0, 11.0)
            v_mph = v_ms * 2.23694
            labels[str(i)] = {"label":"L1","family":"unit_semantics","note":"mph-stored value %.2f (true m/s %.2f)"%(v_mph, v_ms)}
            recs.append({"rid":i,"ts":round(t,3),"speed":round(v_mph,3)})
        else:
            v_ms = rng.uniform(1.0, 12.0)
            labels[str(i)] = {"label":"clean","family":"unit_semantics"}
            recs.append({"rid":i,"ts":round(t,3),"speed":round(v_ms,3)})
    w_jsonl(os.path.join(out_dir,"unit_semantics.jsonl"), recs)
    gt["unit_semantics"] = {"labels":labels,"params":{"n":n,"corrupt":n_corrupt,"real_world_source":"NGSIM I-80 (USDOT public) velocity column mph/m-s semantics mismatch"}}

# ---------- family 3: time_align (L1, timestamp integrity) ----------
# modeled after px-DataCollection/px_omnisharing_dataprocess_kit#3 (tactile-encoder timestamp ±7ms semantics undefined)
def gen_time_align(rng, n, out_dir, gt):
    recs, labels = [], {}
    n_dup, n_back, n_gap = int(n*0.05), int(n*0.05), int(n*0.05)
    corrupt = set(rng.sample(range(n), n_dup+n_back+n_gap))
    d_ids = set(list(corrupt)[:n_dup]); b_ids = set(list(corrupt)[n_dup:n_dup+n_back]); g_ids = corrupt-d_ids-b_ids
    t = 0.0
    prev_t = None
    for i in range(n):
        if i in d_ids and prev_t is not None:
            # duplicate previous timestamp (i=0 has no predecessor -> falls through)
            labels[str(i)] = {"label":"L1","family":"time_align","note":"duplicate timestamp"}
            recs.append({"rid":i,"ts":prev_t})
            continue
        t += 0.01 + rng.uniform(-0.0005, 0.0005)
        if i in b_ids:
            t -= rng.uniform(0.02, 0.05)  # goes backwards
            labels[str(i)] = {"label":"L1","family":"time_align","note":"non-monotonic step back"}
        elif i in g_ids:
            t += rng.uniform(0.12, 0.5)   # 12x..50x nominal gap
            labels[str(i)] = {"label":"L1","family":"time_align","note":"gap > 10x nominal dt"}
        labels[str(i)] = labels.get(str(i), {"label":"clean","family":"time_align"})
        recs.append({"rid":i,"ts":round(t,4)})
        prev_t = recs[-1]["ts"]
    w_jsonl(os.path.join(out_dir,"time_align.jsonl"), recs)
    gt["time_align"] = {"labels":labels,"params":{"n":n,"dup":n_dup,"back":n_back,"gap":n_gap,"real_world_source":"px-DataCollection/px_omnisharing_dataprocess_kit#3 (tactile-encoder timestamp semantics)"}}

# ---------- family 4: frame_drop (L1, lost frames) ----------
# modeled after 18-flight DJI log audit where 67,723 frames were silently lost
def gen_frame_drop(rng, n, out_dir, gt):
    recs, labels = [], {}
    n_drop = int(n * 0.12)
    drop_ids = set(rng.sample(range(1, n), n_drop))  # never drop frame 0
    idx = 0
    for i in range(n):
        while idx in drop_ids:
            labels[str(len(recs))] = None  # dropped frames simply absent
            idx += 1
        labels[str(i)] = {"label":"clean","family":"frame_drop"}
        recs.append({"rid":i,"frame_idx":idx})
        idx += 1
    # ground truth: frame_idx discontinuities reveal drops; detectable per-record via neighbor gap
    labels2 = {}
    for j, r in enumerate(recs):
        labels2[str(j)] = {"label":"clean","family":"frame_drop"}
    # mark records whose PREVIOUS frame gap > 1 (the record after a drop is the evidence point)
    for j in range(1, len(recs)):
        gap = recs[j]["frame_idx"] - recs[j-1]["frame_idx"]
        if gap > 1:
            labels2[str(j)] = {"label":"L1","family":"frame_drop","note":"frame_idx gap %d (frames silently lost)"%gap}
    w_jsonl(os.path.join(out_dir,"frame_drop.jsonl"), recs)
    gt["frame_drop"] = {"labels":labels2,"params":{"n":len(recs),"dropped_frames":n_drop,"real_world_source":"18-flight DJI log audit, 67,723 frames silently lost (public report, PHM-Audit)"}}

# ---------- family 5: kinematic_teleport (L2, velocity envelope violation) ----------
# modeled after FPV/crash-log position inconsistencies (UZH-FPV / PX4 public flight logs)
def gen_kinematic(rng, n, out_dir, gt):
    recs, labels = [], {}
    n_tele = int(n * 0.08)
    tele_ids = set(rng.sample(range(n), n_tele))
    # plausible envelope: max 25 m/s (fast quad). dt=0.02s → max jump 0.5m; borderline cases at 20-24 m/s equivalent
    x,y,z,t = 0.0,0.0,0.0,0.0
    for i in range(n):
        dt = 0.02
        vx,vy,vz = rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(-1.5,1.5)
        x+=vx*dt; y+=vy*dt; z+=vz*dt; t+=dt
        if i in tele_ids:
            hard = rng.random() < 0.5
            jump = rng.uniform(30,120) if hard else rng.uniform(20.5,24.5)
            ang = rng.uniform(0, 2*math.pi)
            x += jump*math.cos(ang)*dt; y += jump*math.sin(ang)*dt
            labels[str(i)] = {"label":"L2","family":"kinematic_teleport","note":"implied speed %.1f m/s (hard=%s)"%(jump,hard)}
        else:
            labels[str(i)] = {"label":"clean","family":"kinematic_teleport"}
        recs.append({"rid":i,"ts":round(t,3),"pos":[round(x,4),round(y,4),round(z,4)]})
    w_jsonl(os.path.join(out_dir,"kinematic_teleport.jsonl"), recs)
    gt["kinematic_teleport"] = {"labels":labels,"params":{"n":n,"teleport":n_tele,"envelope_ms":25.0,"real_world_source":"UZH-FPV dataset + PX4 flight-review public logs (position/attitude discontinuities)"}}

# ---------- family 6: mixed_l3 (L3, malformed records) ----------
# modeled after OpenGalaxea/GalaxeaLeRobotToolkit#5 (None values silently ingested)
def gen_mixed_l3(rng, n, out_dir, gt):
    recs, labels = [], {}
    n_mal = int(n * 0.07)
    mal_ids = set(rng.sample(range(n), n_mal))
    t = 0.0
    for i in range(n):
        t += 0.05
        if i in mal_ids:
            kind = rng.choice(["null_val","truncated","non_numeric"])
            if kind == "null_val":
                recs.append({"rid":i,"ts":round(t,3),"speed":None})
            elif kind == "truncated":
                recs.append({"rid":i,"ts":round(t,3)})  # missing required speed field
            else:
                recs.append({"rid":i,"ts":round(t,3),"speed":"N/A"})
            labels[str(i)] = {"label":"L3","family":"mixed_l3","note":kind}
        else:
            recs.append({"rid":i,"ts":round(t,3),"speed":round(rng.uniform(0.5,8.0),3)})
            labels[str(i)] = {"label":"clean","family":"mixed_l3"}
    w_jsonl(os.path.join(out_dir,"mixed_l3.jsonl"), recs)
    gt["mixed_l3"] = {"labels":labels,"params":{"n":n,"malformed":n_mal,"real_world_source":"OpenGalaxea/GalaxeaLeRobotToolkit#5 (None silently ingested)"}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    gt = {"meta": {"benchmark":"DPC-Bench","version":"0.1","seed":a.seed,
                   "label_schema":{"clean":"passes physical-consistency checks",
                                   "L1":"data quality defect (integrity/semantics)",
                                   "L2":"physical violation (breaks physics constraints)",
                                   "L3":"malformed/unparseable record"},
                   "spec_ref":"Q/QZKJ 001-2026 label taxonomy (three-caliber judgment)"}}
    gen_quat_norm(rng, N_PER_FAMILY, out_dir, gt)
    gen_unit_semantics(rng, N_PER_FAMILY, out_dir, gt)
    gen_time_align(rng, N_PER_FAMILY, out_dir, gt)
    gen_frame_drop(rng, N_PER_FAMILY, out_dir, gt)
    gen_kinematic(rng, N_PER_FAMILY, out_dir, gt)
    gen_mixed_l3(rng, N_PER_FAMILY, out_dir, gt)
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=1)
    print("generated 6 families ->", out_dir)
    for fam, v in gt.items():
        if fam == "meta": continue
        n_def = sum(1 for l in v["labels"].values() if l["label"] != "clean")
        print(f"  {fam:22s} records={len(v['labels'])} defective={n_def}")

if __name__ == "__main__":
    main()
