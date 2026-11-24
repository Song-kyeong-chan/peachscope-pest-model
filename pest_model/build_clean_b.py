"""실험 2-B: 믿을 만한 라벨만 모은 과실 분류 데이터 (라벨 점검 결과에 따른 임시 데이터).

포함
- zeerox(= PDDICP 사진 + 과실 박스): healthy_peach→정상, shot_hole→세균구멍병, brown_rot→잿빛무늬병
  (bacterial_spot은 PDDICP 라벨 오류 클래스, shot_hole_leaf는 잎이라 제외)
- Islamia `GOOD-` 정상, 숫자형 파일명의 잿빛무늬병 (눈으로 확인해 믿을 만했던 부분)
  PDDICP와 겹치는 Islamia 묶음은 제외한다 (zeerox가 PDDICP 라벨로 이미 포함).

분할
- Islamia 사진: resplit_islamia.py의 분할을 그대로 쓴다 → A(test 223장 재라벨)로 B 모델도 평가할 수 있다.
- zeerox 사진: dHash(≤10)·원본 이름으로 묶는다. Islamia와 겹치면 그 묶음의 분할(test > valid > train 우선)을 따르고,
  나머지는 묶음 단위 70/15/15(대표 클래스 층화, seed 42). valid/test는 묶음당 1장.
결과: <out>/images/<split>/<cls>/*.jpg, <out>/crops.csv, <out>/splits.csv(train.py 형식)
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from crop_fruits import crop_box
from dedup_islamia import HAMMING, hash_dir

SEED = 42
ZEEROX_MAP = {"healthy_peach": "healthy", "shot_hole": "shot_hole", "brown_rot": "brown_rot"}
ORDER = ("brown_rot", "shot_hole", "healthy")


def base(p):
    return Path(p).name.split(".rf.")[0]


def islamia_source_ok(group, label):
    g = group.lstrip("-")
    if label == "healthy":
        return g.startswith("GOOD")
    if label == "brown_rot":
        return g[:1].isdigit() and not g.startswith("202")
    return False


def find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def read_boxes(img):
    lab = Path(img).parent.parent / "labels" / (Path(img).stem + ".txt")
    out = []
    for l in lab.read_text().split("\n"):
        p = l.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
        elif len(p) >= 7 and len(p) % 2 == 1:  # 다각형(c x1 y1 x2 y2 ...) → 감싸는 박스
            xs, ys = list(map(float, p[1::2])), list(map(float, p[2::2]))
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            out.append((int(p[0]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islamia", required=True)
    ap.add_argument("--zeerox", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    isl, zx, out = Path(args.islamia), Path(args.zeerox), Path(args.out)
    isl_names = yaml.safe_load((isl / "data.yaml").read_text())["names"]
    zx_names = yaml.safe_load((zx / "data.yaml").read_text())["names"]

    # Islamia: 원본 이름 → 분할, 그리고 이번에 쓸 사진
    with open(isl / "splits.csv") as fh:
        plan = list(csv.DictReader(fh))
    base_split = {base(r["path"]): r["split"] for r in plan}
    items = []  # (img, split, group, source, [(cls, x, y, w, h)])
    for r in plan:
        if r["in_pddicp"] == "1":
            continue
        boxes = [(isl_names[c], x, y, w, h) for c, x, y, w, h in read_boxes(r["path"])]
        boxes = [("healthy" if c == "healty_Peach" else c, x, y, w, h) for c, x, y, w, h in boxes]
        boxes = [b for b in boxes if islamia_source_ok(r["group"], b[0])]
        if boxes:
            items.append((r["path"], r["split"], "isl:" + r["group"], "islamia", boxes))

    # zeerox: 묶음 만들기 + Islamia와 겹침 확인
    zfiles = sorted(str(p) for p in zx.glob("*/images/*"))
    zkeep, Hz = hash_dir(zfiles)
    afiles = sorted(str(p) for p in isl.glob("*/images/*"))
    akeep, Ha = hash_dir(afiles)
    par = {f: f for f in zkeep}
    by = defaultdict(list)
    for f in zkeep:
        by[base(f)].append(f)
    for v in by.values():
        for f in v[1:]:
            par[find(par, f)] = find(par, v[0])
    Hz8 = Hz.astype(np.uint8)
    for i in range(len(zkeep)):
        d = (Hz8[i + 1:] != Hz8[i]).sum(1)
        for j in np.nonzero(d <= HAMMING)[0]:
            a, b = find(par, zkeep[i]), find(par, zkeep[i + 1 + int(j)])
            if a != b:
                par[b] = a
    groups = defaultdict(list)
    for f in zkeep:
        groups[find(par, f)].append(f)

    forced = {}
    D = np.concatenate([(Hz[s:s + 256, None, :] != Ha[None, :, :]).sum(2) for s in range(0, len(Hz), 256)])
    rank = {"test": 3, "valid": 2, "train": 1}
    n_overlap = 0
    for i, j in zip(*np.nonzero(D <= HAMMING)):
        s = base_split.get(base(akeep[j]))
        if s is None:
            continue
        n_overlap += 1
        g = find(par, zkeep[i])
        if rank[s] > rank.get(forced.get(g), 0):
            forced[g] = s

    def zboxes(f):
        return [(ZEEROX_MAP[zx_names[c]], x, y, w, h) for c, x, y, w, h in read_boxes(f) if zx_names[c] in ZEEROX_MAP]

    def gclass(mem):
        cs = {b[0] for f in mem for b in zboxes(f)}
        return next((c for c in ORDER if c in cs), "none")

    rng = random.Random(SEED)
    strata = defaultdict(list)
    for g, mem in groups.items():
        if g not in forced:
            strata[gclass(mem)].append(g)
    assign = dict(forced)
    for c, gs in strata.items():
        gs.sort()
        rng.shuffle(gs)
        nv = nt = round(len(gs) * 0.15)
        for k, g in enumerate(gs):
            assign[g] = "test" if k < nt else "valid" if k < nt + nv else "train"
    for g, mem in groups.items():
        mem = sorted(mem)
        s = assign[g]
        for f in (mem if s == "train" else mem[:1]):
            boxes = zboxes(f)
            if boxes:
                items.append((f, s, "zx:" + base(g), "zeerox", boxes))

    # 자르기
    rows, small = [], 0
    for img, split, group, src, boxes in items:
        im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
        for k, (cls, x, y, w, h) in enumerate(boxes):
            crop, side = crop_box(im, x, y, w, h)
            if crop is None:
                small += 1
                continue
            dst = out / "images" / split / cls / f"{src}_{Path(img).stem}_{k}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, quality=92)
            rows.append({"path": str(dst.relative_to(out)), "split": split, "label": cls, "source": src,
                         "group": group, "box_px": round(side)})
    with open(out / "crops.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "splits.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "split", "cls"])
        for r in rows:
            wr.writerow([r["path"].removeprefix("images/"), {"valid": "val"}.get(r["split"], r["split"]), r["label"]])

    print(f"zeerox 사진 {len(zkeep)}장 → 묶음 {len(groups)}개, Islamia와 겹치는 쌍 {n_overlap}개(묶음 {len(forced)}개 분할 고정)")
    print(f"과실 {len(rows)}개, 작은 박스 {small}개 제외")
    cnt = Counter((r["split"], r["source"], r["label"]) for r in rows)
    for s in ("train", "valid", "test"):
        print(f"  {s}: " + ", ".join(f"{c} {cnt[(s, 'zeerox', c)]}+{cnt[(s, 'islamia', c)]}" for c in ("healthy", "shot_hole", "brown_rot"))
              + "   (zeerox+islamia)")


if __name__ == "__main__":
    main()
