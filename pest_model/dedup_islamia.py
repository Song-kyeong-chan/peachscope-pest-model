"""Islamia(Roboflow) 데이터 중복 검사.

1) Islamia ↔ PDDICP: 같은 사진이 두 데이터에 모두 있는지 (합쳐 쓰면 누수)
2) Islamia 내부: 거의 같은 사진이 train/valid/test에 나뉘어 들어갔는지 (Roboflow 분할 누수)

PDDICP(pddicp3)는 짧은 변 288로 축소돼 있어 md5는 맞지 않는다.
그래서 크기에 영향을 덜 받는 dHash(16x16=256비트)로 비교한다. 거리 <= HAMMING이면 같은 사진으로 본다.
결과: <islamia>/dedup.json, 콘솔 요약
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
HAMMING = 10  # 256비트 중 (prepare.py는 같은 데이터 내부라 6, 여기는 재압축·회전 차이를 고려해 조금 느슨하게)


def dhash(path, size=16):
    im = ImageOps.exif_transpose(Image.open(path))
    g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hash_dir(paths):
    keep, hs = [], []
    for p in paths:
        try:
            hs.append(dhash(p))
            keep.append(p)
        except Exception as e:  # 깨진 파일은 건너뛰고 기록
            print("skip", p, e)
    return keep, np.array(hs, dtype=bool)


def pairs_within(H, thr):
    """같은 배열 안에서 거리 <= thr인 (i, j) 쌍, i < j."""
    out = []
    Hf = H.astype(np.uint8)
    for i in range(len(H)):
        d = (Hf[i + 1:] != Hf[i]).sum(1)
        for j in np.nonzero(d <= thr)[0]:
            out.append((i, i + 1 + int(j), int(d[j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islamia", required=True, help="YOLOv8 내보내기 폴더 (train/valid/test/images)")
    ap.add_argument("--pddicp", required=True, help="pddicp3/images")
    args = ap.parse_args()
    root = Path(args.islamia)

    isl = sorted(p for p in root.glob("*/images/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pdd = sorted(Path(args.pddicp).glob("*.jpg"))
    print(f"Islamia {len(isl)}장, PDDICP {len(pdd)}장 해시 계산 중...")
    isl, Hi = hash_dir(isl)
    pdd, Hp = hash_dir(pdd)
    split = [p.parent.parent.name for p in isl]

    # 1) Islamia ↔ PDDICP
    D = np.concatenate([(Hi[s:s + 256, None, :] != Hp[None, :, :]).sum(2) for s in range(0, len(Hi), 256)])
    cross = [(int(i), int(j), int(D[i, j])) for i, j in zip(*np.nonzero(D <= HAMMING))]
    cross_isl = sorted({i for i, _, _ in cross})
    print(f"\n[1] PDDICP와 겹치는 Islamia 사진: {len(cross_isl)}장 (쌍 {len(cross)}개)")
    print("    최소 거리 분포(Islamia 사진별):", np.bincount(np.minimum(D.min(1), 40))[:15].tolist(), "... (0~14)")

    # 2) Islamia 내부 (분할 사이 누수)
    within = pairs_within(Hi, HAMMING)
    leak = [(i, j, d) for i, j, d in within if split[i] != split[j]]
    by = defaultdict(int)
    for i, j, _ in leak:
        by[tuple(sorted((split[i], split[j])))] += 1
    print(f"\n[2] Islamia 내부 거의 같은 사진 쌍: {len(within)}개, 그중 분할이 다른 쌍: {len(leak)}개")
    for k, v in sorted(by.items()):
        print(f"    {k[0]} ↔ {k[1]}: {v}")

    out = {
        "hamming": HAMMING,
        "n_islamia": len(isl), "n_pddicp": len(pdd),
        "cross": [{"islamia": str(isl[i]), "pddicp": str(pdd[j]), "d": d} for i, j, d in cross],
        "within": [{"a": str(isl[i]), "b": str(isl[j]), "d": d, "split_a": split[i], "split_b": split[j]} for i, j, d in within],
    }
    (root / "dedup.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n저장: {root / 'dedup.json'}")


if __name__ == "__main__":
    main()
