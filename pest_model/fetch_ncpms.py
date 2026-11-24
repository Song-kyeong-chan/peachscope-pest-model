"""NCPMS OpenAPI(병해충검색)로 복숭아 병해충 상세정보와 과실 사진을 받는다.

- 키: 저장소 루트 .env 의 NCPMS_API_KEY (gitignore됨)
- SVC01 병 검색 / SVC03 해충 검색 → SVC05 병 상세 / SVC07 해충 상세
- 사진 제목에 '(열매)'가 있는 것만 받는다.
- 출력(저장소 밖): ~/peachscope-data/ncpms/{details.json, fruit_images.json, images/<class>/*.jpg}
- 사진 저작권은 NCPMS에 있다. 저장소에 올리지 않고, 평가·참고용으로만 쓴다.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "peachscope-data/ncpms"
URL = "http://ncpms.rda.go.kr/npmsAPI/service"
CLASS = {"D00004183": "scab", "D00004186": "shot_hole", "D00004189": "brown_rot", "D00004192": "anthracnose",
         "D00004188": "gray_mold", "D00004336": "viroid", "H00000477": "insect_damage",
         "H00000464": "insect_damage", "H00000386": "insect_damage", "H00000818": "stink_bug",
         "H00000646": "planthopper"}


def key():
    for l in open(ROOT / ".env"):
        if l.startswith("NCPMS_API_KEY="):
            return l.split("=", 1)[1].strip()
    raise SystemExit("NCPMS_API_KEY 없음")


def call(k, **p):
    q = urllib.parse.urlencode(dict(apiKey=k, serviceType="AA003", **p))
    return json.loads(urllib.request.urlopen(f"{URL}?{q}", timeout=30).read().decode("utf-8"))["service"]


def items(il):
    return il.get("item", []) if isinstance(il, dict) else (il or [])


def main():
    k = key()
    OUT.mkdir(parents=True, exist_ok=True)
    det = {}
    for it in call(k, serviceCode="SVC01", cropName="복숭아", displayCount=50)["list"]:
        det[it["sickKey"]] = call(k, serviceCode="SVC05", sickKey=it["sickKey"]); time.sleep(0.3)
    for it in call(k, serviceCode="SVC03", cropName="복숭아", displayCount=50)["list"]:
        det[it["insectKey"]] = call(k, serviceCode="SVC07", insectKey=it["insectKey"]); time.sleep(0.3)
    json.dump(det, open(OUT / "details.json", "w"), ensure_ascii=False, indent=1)
    man = []
    for kk, v in det.items():
        name = v.get("sickNameKor") or v.get("insectSpeciesKor")
        for i, im in enumerate(items(v.get("imageList"))):
            t = im.get("imageTitle") or ""
            if "(열매)" not in t:
                continue
            cls = CLASS.get(kk, "other")
            fp = OUT / "images" / cls / f"{kk}_{i:02d}.jpg"
            fp.parent.mkdir(parents=True, exist_ok=True)
            if not fp.exists():
                fp.write_bytes(urllib.request.urlopen(im["image"], timeout=30).read()); time.sleep(0.2)
            man.append({"key": kk, "name": name, "cls": cls, "title": t, "file": str(fp.relative_to(OUT))})
    json.dump(man, open(OUT / "fruit_images.json", "w"), ensure_ascii=False, indent=1)
    print(len(det), "entries,", len(man), "fruit images")


if __name__ == "__main__":
    main()
