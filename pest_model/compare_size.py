"""입력 크기 비교(224 vs 288): 확정 라벨(final_labels.csv status=ok)만으로 ConvNeXt-Tiny를 학습한다.

- 팀 검수 전 예비 비교다. 검수 대상(review)과 제외(exclude) 크롭은 train/val 모두에서 뺀다.
- 두 크기 모두 같은 데이터, 같은 seed, 같은 설정(lr 5e-5, AdamW, cosine, 좌우 뒤집기, 클래스 가중치)이다.
- 조기 종료: 최대 --epochs, val macro-F1이 --patience epoch 동안 오르지 않으면 멈춘다.
- 평가: val(확정 라벨), NCPMS 91장(exp3_crops test, 외부 평가), 부패병 그룹 정답, CPU 추론 속도.
출력: results/size_cmp/{results.json, convnext_tiny_<size>.pt}
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms as T

from train import DS, MEAN, STD, predict

CLASSES = ["healthy", "shot_hole", "scab", "brown_rot", "insect_damage", "others"]
ROT_ORIG = {"brown_rot", "gray_mold", "anthracnose"}


def ncpms_orig():
    m = json.load(open(Path.home() / "peachscope-data/ncpms/fruit_images.json"))
    return {Path(e["file"]).stem: e["cls"] for e in m}


def evaluate_ncpms(y, p, files, orig):
    pred = p.argmax(1)
    res = dict(correct=int((pred == y).sum()), n=len(y),
               anomaly_detected=int((pred != 0).sum()))
    per = {}
    for c in range(len(CLASSES)):
        idx = y == c
        if idx.any():
            per[CLASSES[c]] = f"{int((pred[idx] == c).sum())}/{int(idx.sum())}"
    res["per_class"] = per
    # 부패병 그룹: 원래 병명이 잿빛무늬병·잿빛곰팡이병·탄저병이면 brown_rot 또는 자기 라벨로 맞히면 정답
    rot = [i for i, f in enumerate(files) if orig.get(Path(f).stem[3:]) in ROT_ORIG]
    res["rot_group"] = f"{sum(pred[i] == 3 or pred[i] == y[i] for i in rot)}/{len(rot)}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "peachscope-data/exp3_crops"))
    ap.add_argument("--labels", default="results/relabel_trainval/final_labels.csv")
    ap.add_argument("--out", default="results/size_cmp")
    ap.add_argument("--sizes", default="224,288")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    sp = {"train": [], "val": [], "test": []}
    for r in csv.DictReader(open(args.labels)):
        if r["status"] == "ok":
            sp[r["split"]].append((r["file"], CLASSES.index(r["final"])))
    for r in csv.DictReader(open(Path(args.data) / "splits.csv")):
        if r["split"] == "test":
            sp["test"].append((r["file"], CLASSES.index(r["cls"])))
    print(f"device {dev}; train {len(sp['train'])} val {len(sp['val'])} ncpms {len(sp['test'])}", flush=True)
    print("train by class", dict(zip(CLASSES, np.bincount([c for _, c in sp["train"]], minlength=6).tolist())), flush=True)
    orig = ncpms_orig()
    root = Path(args.data) / "images"
    rpath = out / "results.json"
    results = json.loads(rpath.read_text()) if rpath.exists() else []

    for size in map(int, args.sizes.split(",")):
        torch.manual_seed(42); np.random.seed(42)
        tf_eval = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
        tf_train = T.Compose([T.Resize(size), T.CenterCrop(size), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
        tr = DataLoader(DS(sp["train"], root, tf_train), batch_size=args.bs, shuffle=True)
        va = DataLoader(DS(sp["val"], root, tf_eval), batch_size=args.bs)
        te = DataLoader(DS(sp["test"], root, tf_eval), batch_size=args.bs)
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=len(CLASSES)).to(dev)
        cnt = np.bincount([c for _, c in sp["train"]], minlength=len(CLASSES))
        crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(CLASSES) * cnt), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best, best_ep, best_state, hist, t0 = -1, 0, None, [], time.time()
        for ep in range(args.epochs):
            model.train(); tot = 0.0; te0 = time.time()
            for x, t in tr:
                x, t = x.to(dev), t.to(dev)
                opt.zero_grad(); loss = crit(model(x), t); loss.backward(); opt.step()
                tot += loss.item() * len(t)
            sched.step()
            yv, pv = predict(model, va, dev)
            f1 = f1_score(yv, pv.argmax(1), average="macro")
            hist.append(dict(epoch=ep + 1, loss=round(tot / len(sp["train"]), 4), val_macro_f1=round(f1, 4)))
            print(f"[{size}] ep{ep + 1:02d} loss {tot / len(sp['train']):.3f} val macroF1 {f1:.3f} ({time.time() - te0:.0f}s)", flush=True)
            if f1 > best:
                best, best_ep = f1, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif ep + 1 - best_ep >= args.patience:
                print(f"[{size}] early stop (best ep{best_ep})", flush=True)
                break
        model.load_state_dict(best_state)
        yv, pv = predict(model, va, dev)
        yt, pt = predict(model, te, dev)
        torch.save(best_state, out / f"convnext_tiny_{size}.pt")
        # CPU 추론 속도(서버 기준 근사): 1장씩 50회
        cpu = timm.create_model("convnext_tiny", num_classes=len(CLASSES)); cpu.load_state_dict(best_state); cpu.eval()
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            for _ in range(5): cpu(x)
            s = time.time()
            for _ in range(50): cpu(x)
        ms = (time.time() - s) / 50 * 1000
        res = dict(size=size, best_epoch=best_ep, epochs_run=len(hist), train_min=round((time.time() - t0) / 60, 1),
                   val_macro_f1=round(best, 4),
                   val_f1_per_class={c: round(float(v), 4) for c, v in zip(CLASSES, f1_score(yv, pv.argmax(1), average=None, labels=range(6)))},
                   ncpms=evaluate_ncpms(yt, pt, [f for f, _ in sp["test"]], orig),
                   cpu_ms_per_image=round(ms, 1), history=hist)
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False), flush=True)
        results = [r for r in results if r["size"] != size] + [res]
        rpath.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("saved", rpath)


if __name__ == "__main__":
    main()
