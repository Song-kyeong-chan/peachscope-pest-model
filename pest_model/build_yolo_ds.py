"""재분할(splits.csv) 기준으로 YOLO 탐지 데이터셋을 만든다 (실험 2, 1단계 탐지기).

--mode fruit   : 모든 박스를 'peach' 1클래스로 합친다 (2단계 구조의 1단계: 과실 위치만 찾는다)
--mode disease : 원래 3클래스 그대로 (탐지기 하나로 병해까지 판정하는 1단계 구조, 비교용)
사진은 심볼릭 링크로 연결하고 라벨만 새로 쓴다. valid/test는 묶음당 1장(splits.csv와 같음).
결과: <out>/{train,val,test}/{images,labels}, <out>/data.yaml
"""
import argparse
import csv
import os
from pathlib import Path

import yaml

NAMES = {"fruit": ["peach"], "disease": ["brown_rot", "healthy", "shot_hole"]}  # disease는 원래 순서 유지


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islamia", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=NAMES, default="fruit")
    args = ap.parse_args()
    root, out = Path(args.islamia), Path(args.out)

    n = {"train": 0, "val": 0, "test": 0}
    boxes = 0
    with open(root / "splits.csv") as fh:
        for r in csv.DictReader(fh):
            src = Path(r["path"])
            split = {"valid": "val"}.get(r["split"], r["split"])
            lab = src.parent.parent / "labels" / (src.stem + ".txt")
            lines = [l.split() for l in lab.read_text().split("\n") if l.strip()]
            if args.mode == "fruit":
                lines = [["0"] + l[1:] for l in lines]
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            dst = out / split / "images" / src.name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            (out / split / "labels" / (src.stem + ".txt")).write_text("\n".join(" ".join(l) for l in lines) + "\n")
            n[split] += 1
            boxes += len(lines)

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
         "names": dict(enumerate(NAMES[args.mode]))}, allow_unicode=True, sort_keys=False))
    print(f"[{args.mode}] 사진 {n}, 박스 {boxes}개 → {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
