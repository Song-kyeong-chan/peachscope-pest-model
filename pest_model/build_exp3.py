"""실험 3 (예비) 분류 데이터: 6클래스 과실 크롭.

- train/val: train/valid 재분류 크롭(relabel_trainval/crops, 1차 라벨, 검수 전)
             + zeerox 크롭(clean_b_crops, source=zeerox; healthy/shot_hole/brown_rot)
- test: NCPMS 공식 과실 사진(eval_ncpms.py와 같은 방식으로 가장 큰 과실 크롭)
  anthracnose·gray_mold·stink_bug·planthopper·viroid → others
출력: ~/peachscope-data/exp3_crops/{images/..., splits.csv}
"""
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO, YOLOWorld

H = Path.home() / "peachscope-data"
OUT = H / "exp3_crops"
NCPMS_MAP = {"brown_rot": "brown_rot", "shot_hole": "shot_hole", "scab": "scab", "insect_damage": "insect_damage"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    # 1) 재분류 크롭
    fb = json.load(open(H / "relabel_trainval/fruit_boxes.json"))
    for d in (H / "relabel_trainval/crops").iterdir():
        for f in d.glob("*.jpg"):
            n = str(int(f.name.split("_")[0]))
            sp = "train" if fb[n]["split"] == "train" else "val"
            dst = OUT / "images" / sp / d.name / ("rt_" + f.name)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst)
            rows.append((str(dst.relative_to(OUT / "images")), sp, d.name))
    # 2) zeerox
    for r in csv.DictReader(open(H / "clean_b_crops/crops.csv")):
        if r["source"] != "zeerox" or r["split"] == "test":
            continue
        sp = "train" if r["split"] == "train" else "val"
        src = H / "clean_b_crops" / r["path"]
        dst = OUT / "images" / sp / r["label"] / ("zx_" + src.name)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
        rows.append((str(dst.relative_to(OUT / "images")), sp, r["label"]))
    # 3) NCPMS test
    fd = YOLO("results/exp2_yolo/yolo11n_fruit/weights/best.pt")
    yw = YOLOWorld("yolov8l-worldv2.pt"); yw.set_classes(["peach", "fruit"])
    for e in json.load(open(H / "ncpms/fruit_images.json")):
        cls = NCPMS_MAP.get(e["cls"], "others")
        im = ImageOps.exif_transpose(Image.open(H / "ncpms" / e["file"])).convert("RGB"); W, Hh = im.size
        boxes = []
        for det, c in ((fd, 0.25), (yw, 0.1)):
            b = [x for x in det.predict(im, conf=c, verbose=False)[0].boxes.xyxy.tolist()
                 if min(x[2] - x[0], x[3] - x[1]) >= 0.12 * min(W, Hh)]
            if b:
                boxes = b; break
        if boxes:
            x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            pw, ph = (x1 - x0) * .1, (y1 - y0) * .1
            im = im.crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(Hh, y1 + ph)))
        dst = OUT / "images" / "test" / cls / ("nc_" + Path(e["file"]).name)
        dst.parent.mkdir(parents=True, exist_ok=True); im.save(dst, quality=92)
        rows.append((str(dst.relative_to(OUT / "images")), "test", cls))
    with open(OUT / "splits.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "split", "cls"]); w.writerows(rows)
    import collections
    c = collections.Counter((s, k) for _, s, k in rows)
    for s in ("train", "val", "test"):
        print(s, {k: v for (ss, k), v in sorted(c.items()) if ss == s})


if __name__ == "__main__":
    main()
