"""PDDICP 과실 3클래스 준비: 압축 해제 → 중복 묶기 → 축소 저장 → 8:1:1 분할.

같은 사진이나 거의 같은 사진(dHash 거리 <= 6)은 한 묶음으로 처리해서
train과 test에 동시에 들어가지 않게 한다.
"""
import argparse
import csv
import hashlib
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

# 원본 폴더명 → 우리 클래스명. bacterial spot은 라벨 오류라서 제외한다.
CLASSES = {"healthy": "healthy", "shot hole": "shot_hole", "brown rot": "brown_rot"}
SHORT_SIDE = 288
HAMMING = 6


def dhash(im, size=16):
    g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with zipfile.ZipFile(args.zip) as zf:
        for name in sorted(zf.namelist()):
            p = Path(name)
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or len(p.parts) < 2:
                continue
            cls = CLASSES.get(p.parent.name)
            if cls is None:
                continue
            data = zf.read(name)
            md5 = hashlib.md5(data).hexdigest()
            dst = img_dir / f"{cls}__{md5[:12]}.jpg"
            with Image.open(__import__("io").BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                s = SHORT_SIDE / min(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
                if not dst.exists():
                    im.save(dst, quality=92)
                dh = dhash(im)
            rows.append(dict(src=name, cls=cls, md5=md5, file=dst.name, w=w, h=h, dh=dh))

    # 중복 묶기 (완전 중복 + 유사 중복)
    n = len(rows)
    parent = list(range(n))
    H = np.array([r["dh"] for r in rows])
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= HAMMING)[0] + i + 1:
            parent[find(parent, j)] = find(parent, i)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(parent, i)].append(i)

    # 클래스가 섞인 묶음은 라벨 충돌이라 버린다.
    dropped, kept_groups = [], []
    for g in groups.values():
        cls = {rows[i]["cls"] for i in g}
        uniq = {rows[i]["md5"]: i for i in g}  # 완전 중복 파일은 하나만 남김
        if len(cls) > 1:
            dropped += [rows[i]["src"] for i in g]
        else:
            kept_groups.append((cls.pop(), list(uniq.values())))

    # 클래스별로 묶음 단위 8:1:1 분할
    rng = random.Random(args.seed)
    split_of = {}
    by_cls = defaultdict(list)
    for cls, idx in kept_groups:
        by_cls[cls].append(idx)
    for cls, gs in by_cls.items():
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        acc = 0
        for g in gs:
            frac = acc / total
            sp = "train" if frac < 0.8 else ("val" if frac < 0.9 else "test")
            for i in g:
                split_of[i] = sp
            acc += len(g)

    with open(out / "splits.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "cls", "split", "group", "src", "orig_w", "orig_h"])
        for gi, (cls, idx) in enumerate(kept_groups):
            for i in idx:
                r = rows[i]
                wr.writerow([r["file"], cls, split_of[i], gi, r["src"], r["w"], r["h"]])

    # 요약 출력
    cnt = defaultdict(lambda: defaultdict(int))
    for i, sp in split_of.items():
        cnt[rows[i]["cls"]][sp] += 1
    print(f"원본 {n}장, 묶음 {len(groups)}개, 라벨 충돌로 제외 {len(dropped)}장")
    multi = sum(1 for _, g in kept_groups if len(g) > 1)
    print(f"2장 이상 묶인 묶음 {multi}개 (같은 split에 배치)")
    for cls in sorted(cnt):
        c = cnt[cls]
        print(f"{cls:10s} train {c['train']:4d} val {c['val']:3d} test {c['test']:3d}")


if __name__ == "__main__":
    main()
