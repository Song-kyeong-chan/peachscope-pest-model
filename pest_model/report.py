"""results.json + bias_check.json → results.md (실험 기록에 붙여 넣을 표)."""
import argparse
import json
from pathlib import Path

CLASSES = ["healthy", "shot_hole", "brown_rot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    res = json.loads((out / "results.json").read_text())
    L = ["| 모델 | 파라미터(M) | val macro-F1 | test 정확도 | test macro-F1 | F1 정상 | F1 세균구멍병 | F1 잿빛무늬병 | 학습 시간(s) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda r: -r["test_macro_f1"]):
        pc = r["test_f1_per_class"]
        L.append(f"| {r['model']} | {r['params_M']} | {r['best_val_macro_f1']:.3f} | {r['test_acc']:.3f} | "
                 f"**{r['test_macro_f1']:.3f}** | {pc['healthy']:.3f} | {pc['shot_hole']:.3f} | {pc['brown_rot']:.3f} | {r['train_sec']:.0f} |")
    best = max(res, key=lambda r: r["test_macro_f1"])
    L += ["", f"혼동행렬 ({best['model']}, 행=정답, 열=예측, 순서 {' / '.join(CLASSES)})", "", "```"]
    L += [" ".join(f"{v:4d}" for v in row) for row in best["confusion"]]
    L += ["```"]
    bias = out / "bias_check.json"
    if bias.exists():
        b = json.loads(bias.read_text())
        L += ["", "편향 점검 (병반이 안 보이는 입력)", "", "| 입력 | test 정확도 | test macro-F1 |", "|---|---|---|"]
        for k, v in b.items():
            L.append(f"| {k} | {v['test_acc']:.3f} | {v['test_macro_f1']:.3f} |")
    (out / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
