"""실험 3 학습 데이터(train+val)의 교차 예측: 라벨 오류 의심 사진 찾기.

- train+val 크롭을 원본 사진 단위로 묶어 5조각으로 나눈다(같은 원본의 크롭·증강본이 다른 조각에 섞이지 않게).
  rt_NNNN_k → NNNN, zx_<이름>.rf.<hash>_k → <이름> (Roboflow 증강본은 같은 원본)
- 조각마다 나머지 4조각으로 ConvNeXt-Tiny를 학습하고, 빠진 조각을 판정한다(모든 사진이 자기를 안 본 모델의 판정을 받음).
- 라벨과 판정이 다르고 모델 확신이 높은 사진을 의심 목록으로 낸다.
출력: results/exp3_cv/{oof.csv, suspects.csv, summary.json}
"""
import argparse
import collections
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision import transforms as T

from train import DS, MEAN, STD, predict

CLASSES = ["healthy", "shot_hole", "scab", "brown_rot", "insect_damage", "others"]


def group_of(f):
    n = Path(f).name
    if n.startswith("rt_"):
        return "rt" + n.split("_")[1]
    return "zx" + re.split(r"\.rf\.|_jpg", n)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "peachscope-data/exp3_crops"))
    ap.add_argument("--out", default="results/exp3_cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.7, help="의심 기준: 다른 클래스 확신도")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = [r for r in csv.DictReader(open(Path(args.data) / "splits.csv")) if r["split"] != "test"]
    items = [(r["file"], CLASSES.index(r["cls"])) for r in rows]
    y = np.array([c for _, c in items]); g = np.array([group_of(f) for f, _ in items])
    print(f"device {dev}; {len(items)} crops, {len(set(g))} groups", flush=True)

    tf_eval = T.Compose([T.Resize(args.size), T.CenterCrop(args.size), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_train = T.Compose([T.Resize(args.size), T.CenterCrop(args.size), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    root = Path(args.data) / "images"
    probs = np.zeros((len(items), len(CLASSES)), dtype=np.float32); fold_of = np.zeros(len(items), int)
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for k, (tri, tei) in enumerate(skf.split(np.zeros(len(y)), y, g)):
        torch.manual_seed(42 + k)
        t0 = time.time()
        tr = DataLoader(DS([items[i] for i in tri], root, tf_train), batch_size=args.bs, shuffle=True)
        te = DataLoader(DS([items[i] for i in tei], root, tf_eval), batch_size=args.bs)
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=len(CLASSES)).to(dev)
        cnt = np.bincount(y[tri], minlength=len(CLASSES))
        crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(CLASSES) * cnt), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for ep in range(args.epochs):
            model.train(); tot = 0.0
            for x, t in tr:
                x, t = x.to(dev), t.to(dev)
                opt.zero_grad(); loss = crit(model(x), t); loss.backward(); opt.step()
                tot += loss.item() * len(t)
            sched.step()
            print(f"  fold{k + 1} ep{ep + 1} loss {tot / len(tri):.3f}", flush=True)
        _, p = predict(model, te, dev)
        probs[tei] = p; fold_of[tei] = k
        f1 = f1_score(y[tei], p.argmax(1), average="macro")
        print(f"fold{k + 1} done: held-out macroF1 {f1:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.save(out / "probs_partial.npy", probs)

    pred = probs.argmax(1)
    with open(out / "oof.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "split", "label", "pred", "p_label", "p_pred", "fold"])
        for i, (fn, c) in enumerate(items):
            w.writerow([fn, rows[i]["split"], CLASSES[c], CLASSES[pred[i]], f"{probs[i, c]:.3f}", f"{probs[i, pred[i]]:.3f}", fold_of[i]])
    sus = [i for i in range(len(items)) if pred[i] != y[i] and probs[i, pred[i]] >= args.thr]
    sus.sort(key=lambda i: -probs[i, pred[i]])
    with open(out / "suspects.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "label", "pred", "p_pred", "p_label"])
        for i in sus:
            w.writerow([items[i][0], CLASSES[y[i]], CLASSES[pred[i]], f"{probs[i, pred[i]]:.3f}", f"{probs[i, y[i]]:.3f}"])
    summary = dict(
        n=len(items), oof_acc=round(float((pred == y).mean()), 4),
        oof_macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
        oof_f1_per_class={c: round(float(v), 4) for c, v in zip(CLASSES, f1_score(y, pred, average=None, labels=range(len(CLASSES))))},
        n_suspects=len(sus), thr=args.thr,
        suspects_by_pair={f"{a}->{b}": n for (a, b), n in collections.Counter(
            (CLASSES[y[i]], CLASSES[pred[i]]) for i in sus).most_common()},
        suspects_by_source={s: n for s, n in collections.Counter(Path(items[i][0]).name[:2] for i in sus).items()},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
