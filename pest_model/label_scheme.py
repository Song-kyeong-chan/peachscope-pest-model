"""실험 4 클래스 체계 (2026-11-23 사용자 결정).

- 세균구멍병(shot_hole)과 검은별무늬병(scab)은 사진만으로 확실히 가르기 어려워 'spot'(반점병) 하나로 묶는다.
  둘의 구분은 앱에서 현장 질문(잎 구멍 등)으로 한다 → 대응방안 JSON의 disambiguation.
- 탄저병(anthracnose)을 others에서 떼어 독립 클래스로 둔다. 1차 분류에서 사진 코드에 ANT가 있는 others 크롭이 대상이다.
- others: 그 밖의 부패(곰팡이로 덮인 과실 등), 흰가루병, 드문 해충.
"""
from pathlib import Path

CLASSES4 = ["healthy", "spot", "brown_rot", "anthracnose", "insect_damage", "others"]
# 실험 6 (2026-11-24 사용자 결정): 잿빛무늬병과 수확 후 곰팡이 부패(others, 대부분 잿빛곰팡이병 모양)를 rot 하나로 묶는다.
CLASSES5 = ["healthy", "spot", "rot", "anthracnose", "insect_damage"]
TO5 = {"brown_rot": "rot", "others": "rot"}
KO4 = {"healthy": "정상", "spot": "반점병", "brown_rot": "잿빛무늬병", "anthracnose": "탄저병",
       "insect_damage": "충해과", "others": "기타"}
NCPMS4 = {"brown_rot": "brown_rot", "shot_hole": "spot", "scab": "spot", "anthracnose": "anthracnose",
          "insect_damage": "insect_damage"}  # 그 밖(gray_mold, stink_bug, planthopper, viroid) → others
ROT4 = {"brown_rot", "anthracnose", "others"}  # 부패병 그룹 정답으로 보는 모델 클래스


def ant_photos(decisions="results/relabel_trainval/decisions.txt"):
    """1차 분류에서 코드에 ANT가 들어 있는 사진 번호(str)."""
    s = set()
    for l in open(decisions):
        l = l.split("#")[0].split()
        if len(l) >= 2 and l[0].isdigit() and "ANT" in l[1].replace("?", "").split(","):
            s.add(str(int(l[0])))
    return s


def to4(cls6, file, ant):
    """6클래스 라벨 → 실험 4 클래스. file은 크롭 파일 경로(rt_NNNN_k / zx_...)."""
    if cls6 in ("shot_hole", "scab"):
        return "spot"
    if cls6 == "others":
        n = Path(file).name
        if n.startswith("rt_") and str(int(n.split("_")[1])) in ant:
            return "anthracnose"
    return cls6


MODEL_THR = 0.7


def model4(m):
    return "spot" if m in ("shot_hole", "scab") else m


def rough_label(r, ant):
    """검수 대상(review) 크롭의 임시 라벨: 교차 예측 모델이 0.7 이상 확신하면 모델 판정, 아니면 제안 라벨.
    사용자가 검수 폴더를 보고 '파일 이름(모델 의견)이 맞는 경우가 많다'고 판단했다(2026-11-23).
    모델 판정이 others이고 사진 코드에 ANT가 있으면 anthracnose로 둔다."""
    prop = to4(r["final"], r["file"], ant)
    if float(r["p_model"]) >= MODEL_THR:
        m = model4(r["model"])
        return to4("others", r["file"], ant) if m == "others" else m
    return prop


# 실험 7 (2026-11-25 사용자 결정): 충해과를 '구멍·상처'(damage)로 넓힌다.
# 사진만으로는 충해 초기, 노린재 흡즙 자국, 열과(갈라짐), 우박·새·긁힘을 가를 수 없어서 하나로 묶고 앱 질문으로 나눈다.
CLASSES7 = ["healthy", "spot", "rot", "anthracnose", "damage"]
TO7 = {"brown_rot": "rot", "others": "rot", "insect_damage": "damage"}
