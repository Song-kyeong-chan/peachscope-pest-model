"""과실 병해 분류 모델 비교.

- ImageNet 사전학습 CNN 여러 개를 같은 분할, 같은 설정으로 미세조정한다.
- 증강은 좌우 뒤집기만 쓴다 (교수님 피드백: 일반 증강은 의미가 적음). val/test는 증강하지 않는다.
- 클래스 불균형(잿빛무늬병이 적음)은 손실 가중치로 보정한다.
- val macro-F1이 가장 높은 epoch의 가중치로 test를 한 번만 평가한다.
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
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["healthy", "shot_hole", "brown_rot"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# 모델별 학습률. ConvNeXt는 3e-4에서 학습이 불안정했다 (실험 1, loss 0.59에서 멈춤)
LR = {"convnext_tiny": 5e-5}


class DS(Dataset):
    def __init__(self, items, root, tf):
        self.items, self.root, self.tf = items, Path(root), tf
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        if f not in self.cache:
            self.cache[f] = Image.open(self.root / f).convert("RGB")
        return self.tf(self.cache[f]), y


def load_split(data_dir):
    sp = {"train": [], "val": [], "test": []}
    with open(Path(data_dir) / "splits.csv") as f:
        for r in csv.DictReader(f):
            sp[r["split"]].append((r["file"], CLASSES.index(r["cls"])))
    return sp


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev)).softmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def run(name, sp, args, dev):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = args.size
    tf_eval = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_train = T.Compose([T.Resize(size), T.CenterCrop(size), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    img_root = Path(args.data) / "images"
    dl = lambda items, tf, sh: DataLoader(DS(items, img_root, tf), batch_size=args.bs, shuffle=sh, num_workers=0)
    tr, va, te = dl(sp["train"], tf_train, True), dl(sp["val"], tf_eval, False), dl(sp["test"], tf_eval, False)

    model = timm.create_model(name, pretrained=True, num_classes=len(CLASSES)).to(dev)
    counts = np.bincount([y for _, y in sp["train"]], minlength=len(CLASSES))
    w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    lr = args.lr or LR.get(name, 3e-4)
    print(f"  [{name}] lr {lr}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        yv, pv = predict(model, va, dev)
        f1 = f1_score(yv, pv.argmax(1), average="macro")
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(sp["train"]), val_macro_f1=f1))
        print(f"  [{name}] ep{ep + 1:02d} loss {tot / len(sp['train']):.3f} val macroF1 {f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_sec = time.time() - t0

    model.load_state_dict(best_state)
    yt, pt = predict(model, te, dev)
    pred = pt.argmax(1)
    res = dict(
        model=name,
        params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        best_val_macro_f1=round(best, 4),
        test_acc=round(accuracy_score(yt, pred), 4),
        test_macro_f1=round(f1_score(yt, pred, average="macro"), 4),
        test_f1_per_class={c: round(v, 4) for c, v in zip(CLASSES, f1_score(yt, pred, average=None, labels=range(len(CLASSES))))},
        confusion=confusion_matrix(yt, pred, labels=range(len(CLASSES))).tolist(),
        train_sec=round(train_sec, 1),
        history=hist,
    )
    print(classification_report(yt, pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=3, zero_division=0))
    out = Path(args.out)
    torch.save(best_state, out / f"{name}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="mobilenetv3_large_100,resnet50,efficientnet_b0,convnext_tiny")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="미지정 시 LR 표, 없으면 3e-4")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", default=",".join(CLASSES), help="쉼표로 구분한 클래스 목록 (실험 3: 6클래스)")
    args = ap.parse_args()
    CLASSES[:] = args.classes.split(",")

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sp = load_split(args.data)
    print(f"device {dev}; train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])}")

    # 기존 결과에 합친다 (같은 모델은 새 결과로 교체) → 모델 하나만 다시 돌려도 표가 유지됨
    rpath = Path(args.out, "results.json")
    results = json.loads(rpath.read_text()) if rpath.exists() else []
    for name in args.models.split(","):
        res = run(name, sp, args, dev)
        results = [r for r in results if r["model"] != name] + [res]
        rpath.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("saved", Path(args.out, "results.json"))


if __name__ == "__main__":
    main()
