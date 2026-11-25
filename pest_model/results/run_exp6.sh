set -e
cd "$(dirname "$0")/.."
while pgrep -f "train_exp4.py --labels results/exp5/" >/dev/null; do sleep 20; done
echo "== eval exp5"; .venv/bin/python eval_exp.py results/exp5
echo "== train exp6"; .venv/bin/python train_exp4.py --labels results/exp6/labels.csv --out results/exp6 --patience 5 --min-epochs 8 --scheme5
echo "== eval exp6"; .venv/bin/python eval_exp.py results/exp6 --scheme5
echo "== done"
