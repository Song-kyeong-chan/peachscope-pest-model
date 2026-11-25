"""실험 7 학습·평가 (5클래스: healthy, spot, rot, anthracnose, damage).

모델(--model)
- timm 이름 (convnext_tiny, convnext_small …): ImageNet 사전학습, 288px
- clip:ViT-B/16 : OpenAI CLIP 이미지 인코더 + 분류 헤드. 헤드를 '증상 설명 문장'의 텍스트 임베딩으로 초기화한다(A안: 병해 지식 주입).
  --zeroshot 이면 학습 없이 증상 문장과의 유사도로만 판정한다.
손실(--loss)
- ce_w   : 클래스 가중치 CE (기준)
- ksmooth: 클래스 가중치 + '지식 스무딩'(D안). 정답에 1-eps, 대응이 비슷한 클래스(rot↔anthracnose, spot↔damage)에 eps의 대부분을 준다.
증강(--aug)
- flip   : 좌우 뒤집기만 (기준)
- symptom: + 약한 확대·자르기, 회전, 밝기·대비 변화 (D안). 병반 색이 바뀌지 않게 색조 변화는 작게.
- symptom_lite: symptom에서 자르기를 85~100%로 좁히고 색 변화를 뺀다. 구멍·상처의 형태·색 단서를 지키려는 설정 (실험 7 분해 뒤).
평가: val, test(NCPMS, 판별 불가 사진 제외 점수 포함), test2(팀 검수 test 정답지). 예측은 preds.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from label_scheme import CLASSES7 as C

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
RELATED = {"rot": ["anthracnose"], "anthracnose": ["rot"], "spot": ["damage"], "damage": ["spot"], "healthy": []}
PROMPTS = {
    "healthy": ["a photo of a healthy peach fruit with smooth unblemished skin",
                "a close-up photo of a clean ripe peach without any spots or wounds"],
    "spot": ["a photo of a peach fruit with many small dark spots with reddish halos",
             "a close-up photo of a peach with scattered tiny sunken black spots, bacterial spot or scab disease",
             "a peach fruit covered with small dark speckles and cracked crusty lesions"],
    "rot": ["a photo of a rotting peach fruit with a soft brown spreading lesion",
            "a close-up photo of a peach with brown rot covered in gray powdery mold",
            "a decaying peach with fuzzy white or gray mold growing on it"],
    "anthracnose": ["a photo of a peach fruit with sunken circular brown lesions",
                    "a close-up photo of a peach with anthracnose, round sunken spots with concentric rings",
                    "a peach fruit with depressed brown spots and orange spore masses"],
    "damage": ["a photo of a peach fruit with a hole or wound made by an insect",
               "a close-up photo of a peach with an insect hole, frass and oozing gum",
               "a peach fruit with a crack, scratch or physical wound on the skin"],
}


class DS(Dataset):
    def __init__(self, items, root, tf):
        self.items, self.root, self.tf, self.cache = items, Path(root), tf, {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        if f not in self.cache:
            self.cache[f] = Image.open(self.root / f).convert("RGB")
        return self.tf(self.cache[f]), y


class ClipClassifier(nn.Module):
    def __init__(self, name, dev):
        super().__init__()
        import clip
        self.clip_model, _ = clip.load(name, device="cpu", jit=False,
                                       download_root=str(Path(__file__).parent / "weights/clip"))
        self.clip_model = self.clip_model.float()
        with torch.no_grad():
            W = []
            for c in C:
                e = self.clip_model.encode_text(clip.tokenize(PROMPTS[c]))
                e = e / e.norm(dim=-1, keepdim=True)
                W.append(e.mean(0) / e.mean(0).norm())
        self.head = nn.Parameter(torch.stack(W))           # 증상 문장 임베딩으로 초기화한 분류 헤드
        self.scale = nn.Parameter(torch.tensor(np.log(100.0), dtype=torch.float32))
        del self.clip_model.transformer                       # 텍스트 인코더는 더 쓰지 않음

    def forward(self, x):
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        h = self.head / self.head.norm(dim=-1, keepdim=True)
        return self.scale.exp() * f @ h.t()


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev)).softmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def summarize(rows, flags):
    def one(rs):
        per = {}
        for c in C:
            sub = [r for r in rs if r["label"] == c]
            if sub:
                per[c] = f"{sum(r['pred'] == c for r in sub)}/{len(sub)}"
        rot = [r for r in rs if r["orig"] in ("brown_rot", "gray_mold", "anthracnose")]
        sick = [r for r in rs if r["label"] != "healthy"]
        hl = [r for r in rs if r["label"] == "healthy"]
        d = dict(n=len(rs), correct=sum(r["pred"] == r["label"] for r in rs), per_class=per,
                 anomaly=f"{sum(r['pred'] != 'healthy' for r in sick)}/{len(sick)}")
        if rot:
            d["rot_group"] = f"{sum(r['pred'] in ('rot', 'anthracnose') for r in rot)}/{len(rot)}"
        if hl:
            d["healthy_kept"] = f"{sum(r['pred'] == 'healthy' for r in hl)}/{len(hl)}"
        return d
    t = [r for r in rows if r["split"] == "test"]
    return dict(ncpms=one(t), ncpms_clean=one([r for r in t if r["file"] not in flags]),
                test2=one([r for r in rows if r["split"] == "test2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/exp7/labels.csv")
    ap.add_argument("--images", default=str(Path.home() / "peachscope-data/exp3_crops/images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--loss", default="ce_w", choices=["ce_w", "ksmooth"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--aug", default="flip", choices=["flip", "symptom", "symptom_lite"])
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    rows = list(csv.DictReader(open(a.labels)))
    sp = {s: [(r["file"], C.index(r["cls"])) for r in rows if r["split"] == s] for s in ("train", "val", "test", "test2")}
    meta = {s: [r for r in rows if r["split"] == s] for s in ("test", "test2")}
    is_clip = a.model.startswith("clip:")
    if is_clip:
        model = ClipClassifier(a.model[5:], dev).to(dev); size, mean, std = 224, CLIP_MEAN, CLIP_STD
        lr = a.lr or 1e-5
    else:
        import timm
        model = timm.create_model(a.model, pretrained=True, num_classes=len(C)).to(dev); size, mean, std = 288, MEAN, STD
        lr = a.lr or 5e-5
    tf_eval = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])
    aug = [T.RandomHorizontalFlip()]
    if a.aug == "symptom":
        aug = [T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.RandomRotation(15), T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.02)]
    elif a.aug == "symptom_lite":
        aug = [T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.RandomRotation(15)]
    tf_train = T.Compose([T.Resize(size), T.CenterCrop(size), *aug, T.ToTensor(), T.Normalize(mean, std)])
    L = lambda s, tf, sh: DataLoader(DS(sp[s], a.images, tf), batch_size=a.bs, shuffle=sh)
    print(f"device {dev}; model {a.model} loss {a.loss} aug {a.aug} zeroshot {a.zeroshot}; "
          f"train {len(sp['train'])} val {len(sp['val'])} ncpms {len(sp['test'])} test2 {len(sp['test2'])}", flush=True)

    hist, best, best_ep, t0 = [], -1, 0, time.time()
    if not a.zeroshot:
        cnt = np.bincount([c for _, c in sp["train"]], minlength=len(C))
        w = torch.tensor(cnt.sum() / (len(C) * cnt), dtype=torch.float32, device=dev)
        if a.loss == "ce_w":
            crit = nn.CrossEntropyLoss(weight=w)
        else:
            T_ = torch.zeros(len(C), len(C))
            for i, c in enumerate(C):
                rel = [C.index(r) for r in RELATED[c]]
                others = [j for j in range(len(C)) if j != i and j not in rel]
                T_[i, i] = 1 - a.eps
                for j in rel:
                    T_[i, j] = a.eps * 0.8 / len(rel)
                for j in others:
                    T_[i, j] = a.eps * (0.2 if rel else 1.0) / len(others)
            T_ = T_.to(dev)

            def crit(logits, t):
                return ((-T_[t] * logits.log_softmax(1)).sum(1) * w[t]).sum() / w[t].sum()
        tr, va = L("train", tf_train, True), L("val", tf_eval, False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4 if not is_clip else 0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        best_state = None
        for ep in range(a.epochs):
            model.train(); tot = 0.0; e0 = time.time()
            for x, t in tr:
                x, t = x.to(dev), t.to(dev)
                opt.zero_grad(); loss = crit(model(x), t); loss.backward(); opt.step(); tot += loss.item() * len(t)
            sched.step()
            yv, pv = predict(model, va, dev); f1 = f1_score(yv, pv.argmax(1), average="macro")
            hist.append(dict(epoch=ep + 1, loss=round(tot / len(sp["train"]), 4), val_macro_f1=round(f1, 4)))
            print(f"[{out.name}] ep{ep + 1:02d} loss {tot / len(sp['train']):.3f} val macroF1 {f1:.3f} ({time.time() - e0:.0f}s)", flush=True)
            if f1 > best:
                best, best_ep = f1, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif ep + 1 - best_ep >= a.patience and ep + 1 >= a.min_epochs:
                print(f"[{out.name}] early stop (best ep{best_ep})", flush=True); break
        model.load_state_dict(best_state)
        torch.save(best_state, out / "model.pt")
    yv, pv = predict(model, L("val", tf_eval, False), dev)
    pr = []
    for s in ("test", "test2"):
        yt, pt = predict(model, L(s, tf_eval, False), dev)
        for m, (y, p) in zip(meta[s], zip(yt, pt)):
            pr.append(dict(split=s, file=m["file"], orig=m["orig"], label=C[y], pred=C[p.argmax()], conf=f"{p.max():.3f}"))
    with open(out / "preds.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pr[0])); wr.writeheader(); wr.writerows(pr)
    flags = {r["file"] for r in csv.DictReader(open("results/ncpms_flags.csv"))}
    res = dict(model=a.model, loss=a.loss, eps=a.eps, aug=a.aug, zeroshot=a.zeroshot, lr=lr, best_epoch=best_ep,
               epochs_run=len(hist), train_min=round((time.time() - t0) / 60, 1),
               val_macro_f1=round(float(f1_score(yv, pv.argmax(1), average="macro")), 4),
               val_f1_per_class={c: round(float(v), 4) for c, v in zip(C, f1_score(yv, pv.argmax(1), average=None, labels=range(len(C))))},
               params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
               **summarize(pr, flags), history=hist)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
