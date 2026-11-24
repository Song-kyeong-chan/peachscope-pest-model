#!/usr/bin/env bash
# 실험 2 시스템 비교 (할 일 3·4번): oracle / 2단계 / 1단계를 같은 정답으로 채점하고 표를 만든다.
# 사용: bash pest_model/run_eval_exp2.sh [정답폴더] [정답클래스이름] [결과이름]
#   기본: Islamia 원래 라벨 test (노이즈 있음)
#   재라벨 후: bash pest_model/run_eval_exp2.sh ~/peachscope-data/relabel/test brown_rot,healthy,shot_hole,other relabel
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
GT="${1:-$HOME/peachscope-data/islamia_yolo_disease/test}"
NAMES="${2:-brown_rot,healthy,shot_hole}"
TAG="${3:-orig}"
R="$HERE/results"
OUT="$R/exp2_eval_$TAG"
FRUIT="$R/exp2_yolo/yolo11n_fruit/weights/best.pt"
DIS="$R/exp2_yolo/yolo11n_disease/weights/best.pt"
cd "$HERE"
run() { "$PY" eval_pipeline.py --gt "$GT" --gt-names "$NAMES" --out "$OUT" "$@" 2>&1 | grep -v Warning; }
for cls in exp2_crops exp2b_clean; do
  for arch in resnet50 convnext_tiny; do
    W="$R/$cls/$arch.pt"; [ -f "$W" ] || continue
    run --system oracle --arch $arch --weights "$W" --name "oracle_${cls}_${arch}"
    [ -f "$FRUIT" ] && run --system 2stage --yolo "$FRUIT" --arch $arch --weights "$W" --name "2stage_${cls}_${arch}"
  done
done
[ -f "$DIS" ] && run --system 1stage --yolo "$DIS" --name "1stage_yolo11n_disease"
"$PY" - "$OUT" <<'PYEOF'
import json, sys, glob
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for f in sorted(glob.glob(str(out / "pipeline_*.json"))):
    r = json.load(open(f)); e = r["e2e"]; im = r["image_level_f1"]
    rows.append(f"| {r['name']} | {r['det_recall']} | {r['det_precision']} | {r['matched_cls_macro_f1']} | "
                f"{r['e2e_macro_f1']} | {e['healthy']['f1']} / {e['shot_hole']['f1']} / {e['brown_rot']['f1']} | "
                f"{r['image_level_macro_f1']} | {im['brown_rot']} |")
md = ("| 시스템 | 탐지 재현율 | 탐지 정밀도 | 짝지은 과실 분류 macro-F1 | 끝단 macro-F1 | 끝단 F1 정상/세균/잿빛 | 사진 단위 macro-F1 | 사진 단위 잿빛 F1 |\n"
      "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
(out / "summary.md").write_text(md); print(md)
PYEOF
