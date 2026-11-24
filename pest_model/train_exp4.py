"""실험 4 학습: 반점병 통합 + 탄저병 분리 6클래스 (results/exp4/labels.csv).

- ConvNeXt-Tiny, 288px, lr 5e-5, AdamW, cosine, 좌우 뒤집기, 클래스 가중치, seed 42 (compare_size.py와 같은 설정)
- 조기 종료: val macro-F1이 --patience epoch 동안 오르지 않으면 멈춘다.
- NCPMS 평가: 병명 정답, 클래스별, 이상 검출, 부패병 그룹(원래 병명 잿빛무늬병·잿빛곰팡이병·탄저병 → 모델 brown_rot/anthracnose/others)
출력: results/exp4/{results.json, ncpms_pred.csv, convnext_tiny.pt}
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
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms as T

from label_scheme import CLASSES4 as C, ROT4
from train import DS, MEAN, STD, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/exp4/labels.csv")
    ap.add_argument("--images", default=str(Path.home() / "peachscope-data/exp3_crops/images"))
    ap.add_argument("--out", default="results/exp4")
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = list(csv.DictReader(open(args.labels)))
    sp = {s: [(r["file"], C.index(r["cls"])) for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    test_orig = [r["orig"] for r in rows if r["split"] == "test"]
    print(f"device {dev}; train {len(sp['train'])} val {len(sp['val'])} ncpms {len(sp['test'])}", flush=True)

    torch.manual_seed(42); np.random.seed(42)
    s = args.size
    tf_eval = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_train = T.Compose([T.Resize(s), T.CenterCrop(s), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = DataLoader(DS(sp["train"], args.images, tf_train), batch_size=args.bs, shuffle=True)
    va = DataLoader(DS(sp["val"], args.images, tf_eval), batch_size=args.bs)
    te = DataLoader(DS(sp["test"], args.images, tf_eval), batch_size=args.bs)
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=len(C)).to(dev)
    cnt = np.bincount([c for _, c in sp["train"]], minlength=len(C))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(C) * cnt), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_ep, best_state, hist, t0 = -1, 0, None, [], time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0; e0 = time.time()
        for x, t in tr:
            x, t = x.to(dev), t.to(dev)
            opt.zero_grad(); loss = crit(model(x), t); loss.backward(); opt.step()
            tot += loss.item() * len(t)
        sched.step()
        yv, pv = predict(model, va, dev)
        f1 = f1_score(yv, pv.argmax(1), average="macro")
        hist.append(dict(epoch=ep + 1, loss=round(tot / len(sp["train"]), 4), val_macro_f1=round(f1, 4)))
        print(f"[exp4] ep{ep + 1:02d} loss {tot / len(sp['train']):.3f} val macroF1 {f1:.3f} ({time.time() - e0:.0f}s)", flush=True)
        if f1 > best:
            best, best_ep = f1, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif ep + 1 - best_ep >= args.patience:
            print(f"[exp4] early stop (best ep{best_ep})", flush=True)
            break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "convnext_tiny.pt")
    yv, pv = predict(model, va, dev)
    yt, pt = predict(model, te, dev)
    pred = pt.argmax(1)
    rot = [i for i, o in enumerate(test_orig) if o in ("brown_rot", "gray_mold", "anthracnose")]
    nonrot = [i for i in range(len(yt)) if i not in rot]
    per = {C[c]: f"{int((pred[yt == c] == c).sum())}/{int((yt == c).sum())}" for c in range(len(C)) if (yt == c).any()}
    res = dict(best_epoch=best_ep, epochs_run=len(hist), train_min=round((time.time() - t0) / 60, 1),
               val_macro_f1=round(best, 4),
               val_f1_per_class={c: round(float(v), 4) for c, v in zip(C, f1_score(yv, pv.argmax(1), average=None, labels=range(len(C))))},
               ncpms=dict(correct=int((pred == yt).sum()), n=len(yt), anomaly_detected=int((pred != 0).sum()), per_class=per,
                          rot_group=f"{sum(C[pred[i]] in ROT4 for i in rot)}/{len(rot)}",
                          nonrot_as_rot=f"{sum(C[pred[i]] in ROT4 for i in nonrot)}/{len(nonrot)}",
                          confusion=confusion_matrix(yt, pred, labels=range(len(C))).tolist()),
               history=hist)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    with open(out / "ncpms_pred.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "orig", "label", "pred", "conf"])
        for i, (fn, c) in enumerate(sp["test"]):
            w.writerow([fn, test_orig[i], C[c], C[pred[i]], f"{pt[i].max():.3f}"])
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False), flush=True)
    print("saved", out / "results.json", flush=True)


if __name__ == "__main__":
    main()
