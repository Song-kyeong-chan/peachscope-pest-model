#!/usr/bin/env bash
# 병해 분류 베이스라인 전체 자동 실행: 환경 → 다운로드 → 준비 → 편향 점검 → 학습·비교 → 결과표
# 사용: bash pest_model/run_all.sh
#   환경변수: DATA_ROOT (기본 ~/peachscope-data), VENV (기본 pest_model/.venv), EPOCHS, MODELS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/peachscope-data}"
VENV="${VENV:-$HERE/.venv}"
OUT="$HERE/results"
ZIP="$DATA_ROOT/raw/Peach_Dataset.zip"
ZIP_URL="https://data.mendeley.com/public-files/datasets/3pmj85snvw/files/e27b87d1-42f6-4fc4-844e-ad18a1b20476/file_downloaded"
ZIP_SIZE=2083758234

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -x "$VENV/bin/python" ]; then
  log "가상환경 생성: $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install torch torchvision timm scikit-learn pillow numpy
fi
PY="$VENV/bin/python -I"

mkdir -p "$DATA_ROOT/raw" "$OUT"
if [ ! -f "$ZIP" ] || [ "$(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP")" != "$ZIP_SIZE" ]; then
  log "PDDICP 다운로드 (2.08 GB)"
  curl -L -A curl/8.4.0 -o "$ZIP" "$ZIP_URL"
fi

if [ ! -f "$DATA_ROOT/pddicp3/splits.csv" ]; then
  log "준비: 3클래스 추출, 중복 묶기, 8:1:1 분할"
  $PY "$HERE/prepare.py" --zip "$ZIP" --out "$DATA_ROOT/pddicp3"
fi

log "편향 점검 (16x16 / 색 히스토그램 로지스틱 회귀)"
$PY "$HERE/bias_check.py" --data "$DATA_ROOT/pddicp3" --out "$OUT"

log "모델 학습·비교"
$PY "$HERE/train.py" --data "$DATA_ROOT/pddicp3" --out "$OUT" \
  --epochs "${EPOCHS:-10}" ${MODELS:+--models "$MODELS"}

log "결과표 생성"
$PY "$HERE/report.py" --out "$OUT"

log "Grad-CAM (ResNet-50)"
$PY "$HERE/gradcam.py" --data "$DATA_ROOT/pddicp3" --out "$OUT" --model resnet50

if [ "${SKIP_CV:-0}" != 1 ]; then
  log "5-fold 교차검증 (약 30분, 건너뛰려면 SKIP_CV=1)"
  (cd "$HERE" && "$VENV/bin/python" -I -c "import sys; sys.path.insert(0, '.'); sys.argv = ['cv.py', '--data', '$DATA_ROOT/pddicp3', '--out', '$OUT/cv'${CV_MODELS:+, '--models', '$CV_MODELS'}]; import cv; cv.main()")
fi
log "완료: $OUT/results.md, $OUT/cv/cv_results.json"
