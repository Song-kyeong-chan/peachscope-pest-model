set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python train_exp7.py"
echo "== base";    $P --out results/exp7_base
echo "== D";       $P --out results/exp7_D --loss ksmooth --aug symptom
echo "== clip_ft"; $P --out results/exp7_clip_ft --model clip:ViT-B/16
echo "== small";   $P --out results/exp7_small --model convnext_small
echo "== done"
