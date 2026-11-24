"""YOLO 과실 탐지기 학습·평가 (실험 2, 1단계).

- 사전학습 YOLO11 가중치에서 미세조정한다. 증강은 ultralytics 기본값(모자이크·색감·뒤집기)을 쓴다.
- val 기준으로 가장 좋은 가중치를 골라 test에서 한 번만 평가한다.
결과: <out>/<name>/ (ultralytics 기본 산출물), <out>/<name>/test_metrics.json
사용: python train_yolo.py --data <ds>/data.yaml --out pest_model/results/exp2_yolo --model yolo11n.pt --epochs 50
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    name = args.name or f"{Path(args.model).stem}_{Path(args.data).parent.name}"
    out = Path(args.out).resolve()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=dev,
                project=str(out), name=name, exist_ok=True, seed=args.seed, deterministic=False,
                patience=15, plots=True, workers=0)

    best = YOLO(out / name / "weights" / "best.pt")
    m = best.val(data=args.data, split="test", imgsz=args.imgsz, device=dev,
                 project=str(out), name=f"{name}_test", exist_ok=True, plots=True)
    res = {
        "model": args.model, "data": args.data, "epochs": args.epochs,
        "test_mAP50": round(float(m.box.map50), 4),
        "test_mAP50_95": round(float(m.box.map), 4),
        "test_precision": round(float(m.box.mp), 4),
        "test_recall": round(float(m.box.mr), 4),
        "per_class_mAP50": {best.names[i]: round(float(v), 4) for i, v in zip(m.box.ap_class_index, m.box.ap50)},
    }
    (out / name / "test_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
