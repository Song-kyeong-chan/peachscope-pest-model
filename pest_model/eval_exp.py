"""실험 모델 평가: NCPMS(전체 / 판별 불가 사진 제외)와 팀 검수 test 정답지(과실 박스 기준).

- NCPMS: train_exp4.py가 남긴 results/<exp>/ncpms_pred.csv를 다시 집계한다.
  results/ncpms_flags.csv의 early(사진만으로 병명 판별이 안 되는 초기 병반)·unjudgeable(과실이 아니거나 증상 없음)을 뺀 점수도 낸다.
- test 정답지: ~/peachscope-data/relabel_test_v1 (Roboflow peach-test-relabel v1, YOLO 형식).
  정답 박스를 10% 넓혀 잘라 분류한다(탐지기 영향 없이 분류기만 평가). shot_hole·spot → spot.
출력: results/<exp>/eval.json
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import timm
import torch
import yaml
from PIL import Image, ImageOps
from torchvision import transforms as T

from label_scheme import CLASSES4 as C
from train import MEAN, STD

ROT_ORIG = {"brown_rot", "gray_mold", "anthracnose"}
ROT_PRED = {"brown_rot", "anthracnose", "others"}
TEST_MAP = {"shot_hole": "spot", "spot": "spot", "brown_rot": "brown_rot", "anthracnose": "anthracnose",
            "healthy": "healthy", "others": "others", "insect_damage": "insect_damage"}


def ncpms_summary(rows):
    n = len(rows)
    rot = [r for r in rows if r["orig"] in ROT_ORIG]
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["label"]][1] += 1; per[r["label"]][0] += r["pred"] == r["label"]
    return dict(n=n, correct=sum(r["pred"] == r["label"] for r in rows), anomaly=sum(r["pred"] != "healthy" for r in rows),
                rot_group=f"{sum(r['pred'] in ROT_PRED for r in rot)}/{len(rot)}",
                per_class={k: f"{a}/{b}" for k, (a, b) in sorted(per.items())})


@torch.no_grad()
def eval_test(weights, size, root, dev):
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    net = timm.create_model("convnext_tiny", num_classes=len(C)); net.load_state_dict(torch.load(weights, map_location="cpu")); net.eval().to(dev)
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    ys, ps = [], []
    for lp in sorted((root / "test/labels").glob("*.txt")):
        ip = next((root / "test/images").glob(lp.stem + ".*"))
        im = ImageOps.exif_transpose(Image.open(ip)).convert("RGB"); W, H = im.size
        crops, lab = [], []
        for l in lp.read_text().split("\n"):
            if not l.strip():
                continue
            c, x, y, w, h = l.split()[:5]; x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2; pw, ph = w * .1, h * .1
            crops.append(tf(im.crop((max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)))))
            lab.append(TEST_MAP[names[int(c)]])
        if crops:
            p = net(torch.stack(crops).to(dev)).softmax(1).argmax(1).cpu().tolist()
            ys += lab; ps += [C[i] for i in p]
    per = collections.defaultdict(lambda: [0, 0])
    for y, p in zip(ys, ps):
        per[y][1] += 1; per[y][0] += y == p
    sick = [(y, p) for y, p in zip(ys, ps) if y != "healthy"]
    healthy = [(y, p) for y, p in zip(ys, ps) if y == "healthy"]
    return dict(n=len(ys), correct=sum(y == p for y, p in zip(ys, ps)),
                per_class={k: f"{a}/{b}" for k, (a, b) in sorted(per.items())},
                anomaly_recall=f"{sum(p != 'healthy' for _, p in sick)}/{len(sick)}",
                healthy_kept=f"{sum(p == 'healthy' for _, p in healthy)}/{len(healthy)}",
                confusion={f"{y}->{p}": n for (y, p), n in collections.Counter(zip(ys, ps)).most_common() if y != p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--test", default=str(Path.home() / "peachscope-data/relabel_test_v1"))
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--scheme5", action="store_true", help="실험 6 클래스(brown_rot+others → rot)")
    a = ap.parse_args()
    if a.scheme5:
        from label_scheme import CLASSES5
        C[:] = CLASSES5
        ROT_PRED.clear(); ROT_PRED.update({"rot", "anthracnose"})
        TEST_MAP.update(brown_rot="rot", others="rot")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    flags = {r["file"]: r["flag"] for r in csv.DictReader(open("results/ncpms_flags.csv"))}
    for e in a.exps:
        rows = list(csv.DictReader(open(Path(e) / "ncpms_pred.csv")))
        res = dict(ncpms_all=ncpms_summary(rows),
                   ncpms_clean=ncpms_summary([r for r in rows if r["file"] not in flags]),
                   test_relabel=eval_test(Path(e) / "convnext_tiny.pt", a.size, Path(a.test), dev))
        (Path(e) / "eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(e, json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
