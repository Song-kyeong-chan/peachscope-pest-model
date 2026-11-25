"""판별 불가 기준값: 최고 확률이 기준 미만이면 '판단 불가'로 돌린다. val로 정하고 NCPMS로 확인한다.

사용: python pick_threshold.py results/exp7_D/model.pt
출력: results/exp7_D/threshold.json (기준별 판정 비율·판정한 것의 정답률)
"""
import csv
import json
import sys
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms as T

from label_scheme import CLASSES7 as C

IMG = Path.home() / "peachscope-data/exp3_crops/images"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    w = Path(sys.argv[1]); dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(C))
    net.load_state_dict(torch.load(w, map_location="cpu")); net.eval().to(dev)
    tf = T.Compose([T.Resize(288), T.CenterCrop(288), T.ToTensor(), T.Normalize(MEAN, STD)])
    rows = [r for r in csv.DictReader(open("results/exp7/labels.csv")) if r["split"] in ("val", "test")]
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            x = torch.stack([tf(Image.open(IMG / r["file"]).convert("RGB")) for r in rows[i:i + 32]]).to(dev)
            P += net(x).softmax(1).cpu().tolist()
    out = {}
    for sp in ("val", "test"):
        d = [(max(p), C[p.index(max(p))] == r["cls"]) for r, p in zip(rows, P) if r["split"] == sp]
        out[sp] = {}
        for thr in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
            k = [ok for m, ok in d if m >= thr]
            out[sp][str(thr)] = dict(judged=f"{len(k)}/{len(d)}", judged_pct=round(100 * len(k) / len(d), 1),
                                     acc_judged=round(100 * sum(k) / max(1, len(k)), 1))
        print(sp, json.dumps(out[sp]))
    (w.parent / "threshold.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
