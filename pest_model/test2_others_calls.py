"""test2 others 149개에 대한 Claude 1차 판정을 저장하고 검토 폴더를 만든다.

판정 코드: h 정상, s 반점, r 부패, a 탄저, d 구멍·상처, q 애매(사용자 판단), x 제외(과실이 거의 안 보임·복숭아 아님·흐림).
시트(300px 칸)를 보고 판정했다. 근거가 약한 것은 q로 둔다.
출력: results/test2_others/calls.csv, ~/peachscope-data/test2_others/review/<폴더>/<idx>_<크롭 stem>__D-<D모델 판정>.jpg
"""
import collections
import csv
import shutil
from pathlib import Path

H = Path.home() / "peachscope-data"
SRC = H / "exp3_crops/images"
REVIEW = H / "test2_others/review"
FOLDER = {"h": "healthy", "s": "spot", "r": "rot", "a": "anthracnose", "d": "damage", "q": "_애매", "x": "_제외"}
CALLS = (
    "d d d d d q h q q q r d q s s d "        # 0-15
    "q q d r d q q d d d d q d q q q "        # 16-31
    "d q d q d d q q d d d q q r r r "        # 32-47
    "s d d d h h h d q d q q d d h d "        # 48-63
    "d h q q d h x q q d d d d x x h "        # 64-79
    "x x x d q r q a r q q h h h h h "        # 80-95
    "q h a r q q d d x r q x h q q q "        # 96-111
    "q q q q q d q q r q d r h r q h "        # 112-127
    "h h q q r q r q q r q r q q r d "        # 128-143
    "r r r r q"                              # 144-148
).split()
NOTE = {
    0: "꼭지 쪽 구멍과 배설물", 2: "큰 구멍, 배설물(심식충류)", 11: "갈색 상처와 수지", 13: "작은 흑색 반점 여럿", 14: "작은 흑색 반점 여럿",
    34: "코르크화한 상처와 구멍", 41: "크게 파먹힌 자리", 46: "움푹한 갈색 병반과 흰 포자", 48: "그을음 같은 흑색 반점(검은별무늬병 의심)",
    50: "구멍과 주황 테두리(노린재 등)", 55: "유충이 보임", 78: "망고로 보임", 83: "열과", 87: "주황 포자 동심원(탄저병 의심)",
    98: "주황 포자 동심원(탄저병 의심)", 99: "푸른곰팡이", 102: "심한 열과", 117: "과육이 크게 드러남", 122: "과육이 크게 드러남",
    132: "둥근 갈색 병반(잿빛무늬병 의심)", 139: "보라빛으로 무른 자리", 142: "썩은 과실 더미", 146: "흰 곰팡이", 147: "흰 곰팡이",
}


def main():
    idx = list(csv.DictReader(open("results/test2_others/index.csv")))
    assert len(CALLS) == len(idx) == 149, (len(CALLS), len(idx))
    if REVIEW.exists():
        shutil.rmtree(REVIEW)
    rows = []
    for r, c in zip(idx, CALLS):
        i = int(r["idx"]); dst = REVIEW / FOLDER[c]; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / r["file"], dst / f"{i:03d}_{Path(r['file']).stem}__D-{r['d_pred']}.jpg")
        rows.append(dict(idx=i, file=r["file"], call=FOLDER[c], d_pred=r["d_pred"], d_conf=r["d_conf"], note=NOTE.get(i, "")))
    with open("results/test2_others/calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("판정", dict(collections.Counter(r["call"] for r in rows)))
    agree = [r for r in rows if r["call"] in ("healthy", "spot", "rot", "anthracnose", "damage")]
    print("확정", len(agree), "중 D와 같음", sum(r["call"] == r["d_pred"] for r in agree))
    print(collections.Counter((r["call"], r["d_pred"]) for r in agree if r["call"] != r["d_pred"]).most_common())


if __name__ == "__main__":
    main()
