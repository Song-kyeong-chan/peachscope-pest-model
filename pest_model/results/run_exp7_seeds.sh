# 약한 증강 D(seed 42) + D·지식 손실만·CLIP 미세조정을 seed 1, 2로 반복 (seed 42는 이미 있음)
cd "$(dirname "$0")/.."
P=".venv/bin/python train_exp7.py"
run() { echo "== $(date '+%H:%M') $1"; $P --out results/$1 "${@:2}" > results/$1.log 2>&1 || echo "!! $1 failed"; }
run exp7_Dlite --loss ksmooth --aug symptom_lite
for s in 1 2; do
  run exp7_D_s$s --loss ksmooth --aug symptom --seed $s
  run exp7_ks_s$s --loss ksmooth --seed $s
  run exp7_clip_ft_s$s --model clip:ViT-B/16 --seed $s
done
.venv/bin/python rescore_test2.py exp7_D exp7_Dlite exp7_D_s1 exp7_D_s2 exp7_ks exp7_ks_s1 exp7_ks_s2 exp7_clip_ft exp7_clip_ft_s1 exp7_clip_ft_s2 --out results/test2_others/rescore_seeds.json > results/test2_others/rescore_seeds.txt 2>&1
echo "== $(date '+%H:%M') done"
