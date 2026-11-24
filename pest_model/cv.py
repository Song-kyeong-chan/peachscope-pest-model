"""5-fold 교차검증으로 모델 비교 (평균 ± 표준편차).

- test(10%)는 건드리지 않는다. train+val(90%)만 5개 fold로 나눈다.
- fold마다: 4개 fold → 학습용, 그중 1/9을 epoch 선택용 val로 다시 떼어 냄. 남은 1개 fold → 평가.
- 묶음(group) 단위, 클래스 비율 유지 (StratifiedGroupKFold).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import train as T

CLASSES = T.CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="mobilenetv3_large_100,efficientnet_b0,resnet50,convnext_tiny")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    rows = [r for r in csv.DictReader(open(Path(args.data) / "splits.csv")) if r["split"] != "test"]
    items = [(r["file"], CLASSES.index(r["cls"])) for r in rows]
    y = np.array([i[1] for i in items])
    g = np.array([int(r["group"]) for r in rows])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "cv_results.json"
    allres = json.loads(rpath.read_text()) if rpath.exists() else {}
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y, g))

    for name in args.models.split(","):
        per = []
        for k, (tr_idx, te_idx) in enumerate(folds):
            tr_idx, va_idx = train_test_split(tr_idx, test_size=1 / 9, stratify=y[tr_idx], random_state=args.seed + k)
            sp = {"train": [items[i] for i in tr_idx], "val": [items[i] for i in va_idx],
                  "test": [items[i] for i in te_idx]}
            print(f"== {name} fold {k + 1}/{args.folds}: train {len(tr_idx)} val {len(va_idx)} eval {len(te_idx)}", flush=True)
            r = T.run(name, sp, args, dev)
            per.append({k2: r[k2] for k2 in ("test_acc", "test_macro_f1", "test_f1_per_class", "confusion")})
            print(f"== {name} fold {k + 1} macroF1 {r['test_macro_f1']:.3f}", flush=True)
        f1 = [p["test_macro_f1"] for p in per]
        acc = [p["test_acc"] for p in per]
        pc = {c: [p["test_f1_per_class"][c] for p in per] for c in CLASSES}
        allres[name] = dict(
            macro_f1_mean=round(float(np.mean(f1)), 4), macro_f1_std=round(float(np.std(f1, ddof=1)), 4),
            acc_mean=round(float(np.mean(acc)), 4), acc_std=round(float(np.std(acc, ddof=1)), 4),
            f1_per_class_mean={c: round(float(np.mean(v)), 4) for c, v in pc.items()},
            f1_per_class_std={c: round(float(np.std(v, ddof=1)), 4) for c, v in pc.items()},
            confusion_sum=np.sum([p["confusion"] for p in per], axis=0).tolist(),
            folds=per)
        rpath.write_text(json.dumps(allres, indent=2, ensure_ascii=False))
        print(f"== {name} CV macroF1 {np.mean(f1):.3f} ± {np.std(f1, ddof=1):.3f}", flush=True)


if __name__ == "__main__":
    main()
