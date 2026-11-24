#!/usr/bin/env bash
# 실험 2 전체 실행: Islamia(YOLOv8 내보내기) → 중복 검사 → 재분할 → 과실 자르기 → 분류기 → YOLO 탐지기
# 전제: ~/peachscope-data/islamia 에 Roboflow 내보내기(yolov8)가 풀려 있다. pest_model/.venv 가 있다.
# 사용: bash pest_model/run_exp2.sh [steps]   예) bash pest_model/run_exp2.sh yolo
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
D="${DATA_ROOT:-$HOME/peachscope-data}"
R="$HERE/results"
STEPS="${1:-prep cls yolo}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ "$STEPS" == *prep* ]]; then
  log "중복 검사";  "$PY" -I "$HERE/dedup_islamia.py" --islamia "$D/islamia" --pddicp "$D/pddicp3/images"
  log "재분할";     "$PY" -I "$HERE/resplit_islamia.py" --islamia "$D/islamia"
  log "과실 자르기"; "$PY" -I "$HERE/crop_fruits.py" --islamia "$D/islamia" --out "$D/islamia_crops"
  "$PY" -I -c "
import csv
rows=list(csv.DictReader(open('$D/islamia_crops/crops.csv')))
with open('$D/islamia_crops/splits.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['file','split','cls'])
    for r in rows: w.writerow([r['path'].removeprefix('images/'), {'valid':'val'}.get(r['split'],r['split']), r['label']])"
  for m in fruit disease; do "$PY" -I "$HERE/build_yolo_ds.py" --islamia "$D/islamia" --out "$D/islamia_yolo_$m" --mode $m; done
fi
if [[ "$STEPS" == *cls* ]]; then
  log "2단계 분류기"; "$PY" "$HERE/train.py" --data "$D/islamia_crops" --out "$R/exp2_crops" --models resnet50,convnext_tiny --epochs 10
fi
if [[ "$STEPS" == *yolo* ]]; then
  cd "$R"  # 사전학습 가중치(yolo11n.pt)가 여기로 내려받아진다
  log "YOLO 과실 탐지 (1클래스)"; "$PY" "$HERE/train_yolo.py" --data "$D/islamia_yolo_fruit/data.yaml" --out "$R/exp2_yolo" --model yolo11n.pt --epochs 50 --name yolo11n_fruit
  log "YOLO 병해 탐지 (3클래스, 비교용)"; "$PY" "$HERE/train_yolo.py" --data "$D/islamia_yolo_disease/data.yaml" --out "$R/exp2_yolo" --model yolo11n.pt --epochs 50 --name yolo11n_disease
fi
log "끝"
