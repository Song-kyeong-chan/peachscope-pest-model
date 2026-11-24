"""Islamia 재분할: 원 제작자의 분할은 증강본이 train/valid/test에 흩어져 있어 누수가 있다.

1) 같은 원본에서 나온 사진을 한 묶음으로 만든다: 파일명의 '.rf.' 앞부분이 같거나, dedup.json의 dHash 쌍(거리 ≤ 10).
2) PDDICP와 겹치는 사진(dedup.json의 cross)은 표시만 한다 (나중에 PDDICP로 외부 평가할 때 제외용).
3) 묶음 단위로 train/valid/test = 70/15/15. 묶음의 대표 클래스(brown_rot > shot_hole > healthy 순으로 포함 여부)로 층화한다.
4) valid/test에는 묶음마다 1장(파일명순 첫 장)만 남긴다. 증강본끼리 평가에 중복 반영되지 않게 한다. train에는 묶음 전체를 넣는다.
결과: <islamia>/splits.csv (path, split, group, group_class, in_pddicp)
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SEED = 42
RATIO = {"train": 0.70, "valid": 0.15, "test": 0.15}


def find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islamia", required=True)
    root = Path(ap.parse_args().islamia)
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    files = sorted(str(p) for p in root.glob("*/images/*"))
    par = {f: f for f in files}

    def union(a, b):
        ra, rb = find(par, a), find(par, b)
        if ra != rb:
            par[rb] = ra

    by_base = defaultdict(list)
    for f in files:
        by_base[Path(f).name.split(".rf.")[0]].append(f)
    for v in by_base.values():
        for f in v[1:]:
            union(v[0], f)
    dd = json.loads((root / "dedup.json").read_text())
    for w in dd["within"]:
        union(w["a"], w["b"])
    in_pdd = {c["islamia"] for c in dd["cross"]}

    groups = defaultdict(list)
    for f in files:
        groups[find(par, f)].append(f)

    def classes_of(f):
        lab = Path(f).parent.parent / "labels" / (Path(f).stem + ".txt")
        return {names[int(l.split()[0])] for l in lab.read_text().split("\n") if l.strip()}

    def gclass(members):
        cs = set().union(*(classes_of(f) for f in members))
        for c in ("brown_rot", "shot_hole", "healty_Peach"):
            if c in cs:
                return c
        return "none"

    rng = random.Random(SEED)
    strata = defaultdict(list)
    for gid, mem in groups.items():
        strata[gclass(mem)].append(gid)
    assign = {}
    for c, gids in strata.items():
        gids.sort()
        rng.shuffle(gids)
        n = len(gids)
        nv, nt = round(n * RATIO["valid"]), round(n * RATIO["test"])
        for i, g in enumerate(gids):
            assign[g] = "test" if i < nt else "valid" if i < nt + nv else "train"

    rows = []
    for gid, mem in groups.items():
        s = assign[gid]
        mem = sorted(mem)
        keep = mem if s == "train" else mem[:1]
        gc = gclass(mem)
        for f in keep:
            rows.append({"path": f, "split": s, "group": Path(gid).name.split(".rf.")[0],
                         "group_class": gc, "in_pddicp": int(any(m in in_pdd for m in mem))})
    with open(root / "splits.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    print(f"사진 {len(files)}장 → 묶음 {len(groups)}개 (PDDICP와 겹치는 묶음 {sum(1 for g in groups.values() if any(m in in_pdd for m in g))}개)")
    gs = Counter(assign.values())
    ims = Counter(r["split"] for r in rows)
    for s in ("train", "valid", "test"):
        cc = Counter(r["group_class"] for r in rows if r["split"] == s)
        print(f"  {s}: 묶음 {gs[s]}, 사진 {ims[s]}  (대표 클래스 {dict(cc)})")


if __name__ == "__main__":
    main()
