#!/usr/bin/env python3
"""DPC-Bench evaluator — scores a predictions file against ground truth.

Prediction file format (JSON):
{
  "method": "<name>",
  "predictions": { "<family>": { "<rid>": "clean|L1|L2|L3" } },
  "runtime_s": <float, optional>
}

Usage: python3 eval/evaluate.py --pred <predictions.json> [--gt data/ground_truth.json] [--out results/<name>.json]
"""
import argparse, json, os

LABELS = ["clean", "L1", "L2", "L3"]

def macro_f1(cm, cls):
    tp = cm[cls][cls]; fp = sum(cm[c][cls] for c in LABELS) - tp
    fn = sum(cm[cls][c] for c in LABELS) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    gt = json.load(open(a.gt))
    P = pred.get("predictions", {})
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    missing, total = 0, 0
    for fam, info in gt.items():
        if fam == "meta": continue
        pfam = P.get(fam, {})
        for rid, lab in info["labels"].items():
            total += 1
            gold = lab["label"]
            got = pfam.get(str(rid))
            if got is None:
                missing += 1; got = "clean"  # unrated defaults to clean (encourages full coverage)
            if got not in LABELS: got = "clean"
            cm[gold][got] += 1
    per = {}
    f1s = []
    for c in LABELS:
        p, r, f1 = macro_f1(cm, c)
        per[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    clean_gold = sum(cm["clean"].values())
    fpr = sum(cm["clean"][c] for c in LABELS if c != "clean") / clean_gold if clean_gold else 0.0
    out = {
        "benchmark": "DPC-Bench v0.1",
        "method": pred.get("method", os.path.basename(a.pred)),
        "records_scored": total,
        "missing_predictions": missing,
        "headline_macro_f1": round(sum(f1s) / 4, 4),
        "false_positive_rate_on_clean": round(fpr, 4),
        "per_class": per,
        "confusion_gold_row_pred_col": cm,
        "runtime_s": pred.get("runtime_s"),
    }
    print(json.dumps({k: out[k] for k in ["method", "headline_macro_f1", "false_positive_rate_on_clean", "per_class", "missing_predictions"]}, ensure_ascii=False, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("saved ->", a.out)

if __name__ == "__main__":
    main()
