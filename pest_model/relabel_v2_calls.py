"""재분류 v2 판정 기록: python relabel_v2_calls.py <그룹> <기본판정> "코드:번호,번호-번호;코드:..."
코드: healthy spot rot anthracnose insect_damage / ?(애매, 사용자 판단) / X(제외)
결과: results/relabel_v2/calls_<그룹>.csv (idx, file, before, claude)
"""
import csv
import sys


def rng(s):
    o = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-"); o += range(int(a), int(b) + 1)
        else:
            o.append(int(p))
    return o


def main():
    g, default, spec = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""
    rows = list(csv.DictReader(open(f"results/relabel_v2/index_{g}.csv")))
    code = {}
    for part in filter(None, spec.split(";")):
        c, nums = part.split(":")
        for i in rng(nums):
            assert i not in code, (i, c, code[i]); code[i] = c
    assert max(code, default=0) < len(rows)
    with open(f"results/relabel_v2/calls_{g}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "file", "before", "claude"])
        for r in rows:
            w.writerow([r["idx"], r["file"], r["cls"], code.get(int(r["idx"]), default)])
    import collections
    print(g, dict(collections.Counter(code.get(int(r["idx"]), default) for r in rows)))


if __name__ == "__main__":
    main()
