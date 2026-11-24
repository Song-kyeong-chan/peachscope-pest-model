"""세 판정(1차 라벨, Claude 2차 판정, 교차 예측 모델)을 합쳐 확정 라벨과 팀 검수 목록을 만든다.

규칙 (과실 크롭 단위)
- 2차 판정 X → 제외
- 1차 확신(unsure 아님): 모델이 다른 클래스를 0.7 이상 확신하면 검수, 아니면 확정
- 1차 불확실: 2차 판정이 '?'면 검수
              2차 판정 == 모델 판정이면 그 라벨로 확정(1차와 달라도)
              2차 판정 == 1차 라벨이고 모델이 다른 클래스를 0.7 이상 확신하지 않으면 확정
              그 밖(판정끼리 충돌)은 검수
- zeerox 크롭: 모델 의심 목록(suspects.csv)에 있으면 검수, 아니면 원래 라벨
출력
- results/relabel_trainval/final_labels.csv (크롭별 확정/검수/제외와 근거)
- ~/peachscope-data/relabel_trainval/review/<제안 클래스>/<id>_모델-<판정>.jpg  (팀 검수 폴더)
  폴더는 실험 4 클래스 체계(label_scheme.py: 반점병 통합, 탄저병 분리)를 따르고, 모델이 0.7 이상 확신하면 모델 판정 폴더에 둔다.
  그때 원래 제안과 다르면 파일 이름 끝에 _원래-<제안>을 붙인다.
"""
import collections
import csv
import json
import shutil
from pathlib import Path

from label_scheme import KO4, ant_photos, rough_label, to4

H = Path.home() / "peachscope-data"
RT = H / "relabel_trainval"
CODE = {"H": "healthy", "SH": "shot_hole", "SC": "scab", "BR": "brown_rot", "INS": "insect_damage", "OT": "others"}
KO = {"healthy": "정상", "shot_hole": "세균구멍병", "scab": "검은별무늬병", "brown_rot": "잿빛무늬병",
      "insect_damage": "충해과", "others": "기타"}
THR = 0.7


def main():
    fb = json.load(open(RT / "fruit_boxes.json"))
    sp = {}
    for l in open("results/relabel_trainval/second_pass.txt"):
        l = l.split("#")[0].split()
        if l:
            sp[l[0]] = l[1]
    oof = {Path(r["file"]).name: r for r in csv.DictReader(open("results/exp3_cv/oof.csv"))}
    rows = []
    for name, r in oof.items():
        lab, pred, pp = r["label"], r["pred"], float(r["p_pred"])
        strong = pred != lab and pp >= THR
        if name.startswith("rt_"):
            n, k = name[3:-4].split("_")
            cid = f"{int(n)}.{k}"
            unsure = fb[str(int(n))]["unsure"]
            d = sp.get(cid)
            if d == "X":
                st, final, why = "exclude", "", "2차 판정 제외"
            elif not unsure:
                st, final, why = ("review", lab, "1차 확신이나 모델이 강하게 반대") if strong else ("ok", lab, "1차 확신, 모델 동의")
            elif d == "?":
                st, final, why = "review", lab, "2차 판정도 불확실"
            elif CODE[d] == pred:
                st, final, why = "ok", pred, "2차 판정과 모델 일치" + ("(1차와 다름)" if pred != lab else "")
            elif CODE[d] == lab and not strong:
                st, final, why = "ok", lab, "1차·2차 일치, 모델 강한 반대 없음"
            else:
                st, final, why = "review", CODE[d], f"판정 충돌(1차 {lab}, 2차 {CODE[d]}, 모델 {pred})"
            rid = f"{int(n):04d}_{k}"
        else:
            cid, d = name, None
            st, final, why = ("review", lab, "zeerox 원래 라벨, 모델이 강하게 반대") if strong else ("ok", lab, "zeerox 원래 라벨")
            rid = None
        rows.append(dict(file=r["file"], id=cid, split=r["split"], label1=lab, second=d or "", model=pred,
                         p_model=pp, status=st, final=final, reason=why, rid=rid))
    # zeerox 검수 크롭에 짧은 id 부여
    zi = 0
    for x in rows:
        if x["status"] == "review" and x["rid"] is None:
            zi += 1; x["rid"] = f"zx{zi:03d}"
    with open("results/relabel_trainval/final_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = RT / "review"
    if out.exists():
        shutil.rmtree(out)
    for c in KO4:
        (out / c).mkdir(parents=True)
    (out / "_제외").mkdir()
    src = H / "exp3_crops/images"
    ant = ant_photos()
    model_ko = lambda m: KO4["spot"] if m in ("shot_hole", "scab") else KO[m]
    for x in rows:
        if x["status"] == "review":
            x.update(p_model=str(x["p_model"]))
            folder, prop = rough_label(x, ant), to4(x["final"], x["file"], ant)
            tail = "" if folder == prop else f"_원래-{KO4[prop]}"
            shutil.copy(src / x["file"], out / folder / f"{x['rid']}_모델-{model_ko(x['model'])}{tail}.jpg")
    c = collections.Counter(x["status"] for x in rows)
    print("status", dict(c))
    print("review by folder", {d.name: len(list(d.glob("*.jpg"))) for d in sorted(out.iterdir())})
    print("review by reason", dict(collections.Counter(x["reason"].split("(")[0] for x in rows if x["status"] == "review")))
    ch = [x for x in rows if x["status"] == "ok" and x["final"] != x["label1"]]
    print("auto-changed", len(ch), dict(collections.Counter((x["label1"], x["final"]) for x in ch)))
    print("final ok by class", dict(collections.Counter(x["final"] for x in rows if x["status"] == "ok")))


if __name__ == "__main__":
    main()
