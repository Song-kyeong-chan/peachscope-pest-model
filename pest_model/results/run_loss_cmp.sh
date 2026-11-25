set -e
cd "$(dirname "$0")/.."
.venv/bin/python train_exp4.py --loss focal_w --out results/exp4_focal_w
.venv/bin/python train_exp4.py --loss focal --out results/exp4_focal
