"""병해 진단 시스템 전체 평가: 사진 → (과실 박스, 병해 클래스) 목록을 정답과 비교한다.

비교하는 시스템
- 2stage : YOLO 과실 탐지기(1클래스) → 박스로 자르기(crop_fruits.crop_box) → CNN 분류기
- 1stage : YOLO 병해 탐지기(3클래스)가 박스와 클래스를 한 번에 낸다
- oracle : 정답 박스로 자르기 → CNN 분류기 (탐지가 완벽할 때의 상한, 분류기만의 성능)

채점 (모든 시스템이 같은 방식)
1) 과실 탐지: 예측 박스와 정답 박스를 IoU ≥ 0.5로 1:1 짝짓기(신뢰도 높은 순) → 재현율·정밀도
2) 짝지은 과실의 분류: 혼동행렬, macro-F1 (탐지가 맞은 과실만)
3) 끝단(end-to-end) 클래스별 정밀도·재현율·F1: 짝이 없는 정답 = 놓침, 짝이 없는 예측 = 오탐, 짝은 맞았는데 클래스가 틀림 = 둘 다
4) 사진 단위: "이 사진에 X병 과실이 있나"를 클래스별 F1로 (앱에서 사진 한 장으로 진단할 때의 성능)

정답 폴더: <gt>/images, <gt>/labels (YOLO txt), 클래스 이름은 --gt-names 순서. 'other' 등 3클래스 밖은 정답에서 '무시' 박스로 처리
(짝지어지면 오탐으로 세지 않고, 놓쳐도 놓침으로 세지 않는다).
결과: <out>/pipeline_<name>.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms as T
from ultralytics import YOLO

from crop_fruits import SIZE, crop_box

CLASSES = ["healthy", "shot_hole", "brown_rot"]
ALIAS = {"healty_Peach": "healthy", "healthy_peach": "healthy"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IOU = 0.5


def iou(a, b):  # (x, y, w, h) 정규화
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def load_gt(gt, names):
    out = {}
    for img in sorted((gt / "images").iterdir()):
        lab = gt / "labels" / (img.stem + ".txt")
        boxes = []
        for l in (lab.read_text().split("\n") if lab.exists() else []):
            p = l.split()
            if len(p) == 5:
                n = ALIAS.get(names[int(p[0])], names[int(p[0])])
                boxes.append((n if n in CLASSES else "ignore", tuple(map(float, p[1:]))))
        out[img] = boxes
    return out


class Classifier:
    def __init__(self, arch, weights, dev, size=224):
        self.m = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
        self.m.load_state_dict(torch.load(weights, map_location="cpu"))
        self.m.eval().to(dev)
        self.dev = dev
        self.tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])

    @torch.no_grad()
    def __call__(self, crops):
        if not crops:
            return []
        x = torch.stack([self.tf(c) for c in crops]).to(self.dev)
        p = self.m(x).softmax(1).cpu().numpy()
        return [(CLASSES[i], float(p[k, i])) for k, i in enumerate(p.argmax(1))]


def crops_for(im, boxes):
    out = []
    for b in boxes:
        c, _ = crop_box(im, *b, min_side=0)
        out.append(c if c is not None else im.resize((SIZE, SIZE)))
    return out


def detect(yolo, img, dev, conf):
    r = yolo.predict(str(img), conf=conf, device=dev, verbose=False)[0]
    names = r.names
    return [(ALIAS.get(names[int(c)], names[int(c)]), tuple(map(float, xywh)), float(s))
            for xywh, c, s in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def score(gt, pred):
    """gt: {img: [(cls, box)]}, pred: {img: [(cls, box, conf)]}"""
    tp_det = n_gt = n_matched = 0
    ys, ps = [], []
    e2e = {c: Counter() for c in CLASSES}
    img_true, img_pred = {c: [] for c in CLASSES}, {c: [] for c in CLASSES}
    for img, G in gt.items():
        P = sorted(pred.get(img, []), key=lambda t: -t[2])
        used = set()
        matched_pred = set()
        for pi, (pc, pb, _) in enumerate(P):
            best, bj = IOU, None
            for gj, (gc, gb) in enumerate(G):
                if gj not in used:
                    v = iou(pb, gb)
                    if v >= best:
                        best, bj = v, gj
            if bj is None:
                e2e[pc]["fp"] += 1
                continue
            used.add(bj)
            matched_pred.add(pi)
            gc = G[bj][0]
            if gc == "ignore":
                continue
            tp_det += 1
            ys.append(CLASSES.index(gc))
            ps.append(CLASSES.index(pc))
            if gc == pc:
                e2e[gc]["tp"] += 1
            else:
                e2e[gc]["fn"] += 1
                e2e[pc]["fp"] += 1
        for gj, (gc, _) in enumerate(G):
            if gj not in used and gc != "ignore":
                e2e[gc]["fn"] += 1
        n_gt += sum(1 for gc, _ in G if gc != "ignore")
        n_matched += len(matched_pred)
        gs, qs = {gc for gc, _ in G}, {pc for pc, _, _ in P}
        for c in CLASSES:
            img_true[c].append(c in gs)
            img_pred[c].append(c in qs)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return round(p, 4), round(r, 4), round(2 * p * r / (p + r) if p + r else 0.0, 4)

    n_pred_all = sum(len(v) for v in pred.values())
    res = {
        "n_images": len(gt), "n_gt_fruits": n_gt, "n_pred_boxes": n_pred_all,
        "det_recall": round(tp_det / n_gt, 4) if n_gt else None,
        "det_precision": round(n_matched / n_pred_all, 4) if n_pred_all else None,
        "matched_cls_macro_f1": round(f1_score(ys, ps, average="macro", labels=range(3)), 4) if ys else None,
        "matched_cls_acc": round(float(np.mean(np.array(ys) == np.array(ps))), 4) if ys else None,
        "matched_confusion": confusion_matrix(ys, ps, labels=range(3)).tolist() if ys else None,
        "e2e": {c: dict(zip(("precision", "recall", "f1"), prf(e2e[c]["tp"], e2e[c]["fp"], e2e[c]["fn"]))) for c in CLASSES},
        "image_level_f1": {c: round(f1_score(img_true[c], img_pred[c], zero_division=0), 4) for c in CLASSES},
    }
    res["e2e_macro_f1"] = round(float(np.mean([res["e2e"][c]["f1"] for c in CLASSES])), 4)
    res["image_level_macro_f1"] = round(float(np.mean(list(res["image_level_f1"].values()))), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="images/, labels/ 가 있는 정답 폴더")
    ap.add_argument("--gt-names", required=True, help="정답 라벨 클래스 이름, 쉼표 구분 (예: brown_rot,healthy,shot_hole,other)")
    ap.add_argument("--system", choices=["2stage", "1stage", "oracle"], required=True)
    ap.add_argument("--yolo", help="2stage: 과실 탐지기 best.pt / 1stage: 병해 탐지기 best.pt")
    ap.add_argument("--arch", help="분류기 timm 이름 (2stage, oracle)")
    ap.add_argument("--weights", help="분류기 가중치 .pt (2stage, oracle)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gt = load_gt(Path(args.gt), args.gt_names.split(","))
    yolo = YOLO(args.yolo) if args.system in ("2stage", "1stage") else None
    clf = Classifier(args.arch, args.weights, dev) if args.system in ("2stage", "oracle") else None

    pred = {}
    for img, G in gt.items():
        if args.system == "1stage":
            pred[img] = detect(yolo, img, dev, args.conf)
            continue
        im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
        if args.system == "2stage":
            dets = detect(yolo, img, dev, args.conf)
            boxes, confs = [b for _, b, _ in dets], [s for _, _, s in dets]
        else:  # oracle: 정답 박스, 'ignore'는 빼고
            boxes = [b for c, b in G if c != "ignore"]
            confs = [1.0] * len(boxes)
        cls = clf(crops_for(im, boxes))
        pred[img] = [(c, b, s) for (c, _), b, s in zip(cls, boxes, confs)]

    res = {"system": args.system, "name": args.name, "gt": args.gt, "yolo": args.yolo,
           "arch": args.arch, "weights": args.weights, "conf": args.conf, **score(gt, pred)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"pipeline_{args.name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("name", "det_recall", "det_precision", "matched_cls_macro_f1", "e2e_macro_f1",
                                          "image_level_macro_f1")}, ensure_ascii=False))
    print("  e2e:", res["e2e"])


if __name__ == "__main__":
    main()
