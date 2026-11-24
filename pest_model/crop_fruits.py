"""Islamia YOLO 라벨(정답 bbox)로 과실을 하나씩 잘라 분류용 데이터를 만든다 (실험 2, 2단계 분류기 입력).

- 박스를 정사각형으로 넓히고 여백 MARGIN을 준 뒤 SIZE×SIZE로 저장한다 → 모든 과실이 같은 크기라 촬영 거리·구도 단서가 사라진다.
- 너무 작은 박스(짧은 변 < MIN_SIDE px)는 병반이 안 보이므로 버린다.
- 같은 사진에서 나온 과실이 train/test에 갈라지지 않게, resplit_islamia.py의 splits.csv(원본 묶음 단위)를 따른다.
결과: <out>/images/<split>/<class>/*.jpg, <out>/crops.csv
"""
import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
SIZE = 288
MARGIN = 0.10
MIN_SIDE = 48
NAME_MAP = {"healty_Peach": "healthy", "brown_rot": "brown_rot", "shot_hole": "shot_hole"}


def crop_box(im, x, y, w, h, min_side=MIN_SIDE):
    """YOLO 정규화 박스 → 정사각형(여백 MARGIN) SIZE×SIZE 자르기. 너무 작으면 None. 밖으로 나가는 부분은 검은색."""
    W, H = im.size
    bw, bh = w * W, h * H
    if min(bw, bh) < min_side:
        return None, min(bw, bh)
    s = max(bw, bh) * (1 + 2 * MARGIN)
    cx, cy = x * W, y * H
    box = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    return im.crop(tuple(round(v) for v in box)).resize((SIZE, SIZE), Image.BICUBIC), min(bw, bh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islamia", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.islamia), Path(args.out)
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]

    # 분할은 resplit_islamia.py가 만든 splits.csv(원본 묶음 단위 재분할)를 따른다
    with open(root / "splits.csv") as fh:
        plan = list(csv.DictReader(fh))
    print(f"splits.csv: {len(plan)}장")

    rows, skipped = [], 0
    for r in plan:
        img, split = Path(r["path"]), r["split"]
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        if not lab.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
        for k, line in enumerate(lab.read_text().split("\n")):
            p = line.split()
            if len(p) != 5:  # 다각형 라벨 등은 건너뜀
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            crop, side = crop_box(im, x, y, w, h)
            if crop is None:
                skipped += 1
                continue
            cls = NAME_MAP[names[c]]
            dst = out / "images" / split / cls / f"{img.stem}_{k}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, quality=92)
            rows.append({"path": str(dst.relative_to(out)), "split": split, "label": cls,
                         "source": img.name, "group": r["group"], "in_pddicp": r["in_pddicp"],
                         "box_px": round(side)})

    with open(out / "crops.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    cnt = Counter((r["split"], r["label"]) for r in rows)
    print(f"과실 {len(rows)}개 저장, 작은 박스 {skipped}개 제외")
    for s in ("train", "valid", "test"):
        print(f"  {s}: " + ", ".join(f"{c} {cnt[(s, c)]}" for c in ("healthy", "shot_hole", "brown_rot")))


if __name__ == "__main__":
    main()
