set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python train_exp7.py"
echo "== ksmooth only"; $P --out results/exp7_ks --loss ksmooth
echo "== aug only";     $P --out results/exp7_aug --aug symptom
echo "== done"
