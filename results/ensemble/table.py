import json
import pandas as pd
import numpy as np

SUMMARY_PATH = "summary.json"
PROGRESS_PATH = "_progress.json"

def fmt_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(round(seconds % 60))
    return f"{mins} min {secs} s"

with open(SUMMARY_PATH) as f:
    summary = json.load(f)
with open(PROGRESS_PATH) as f:
    progress = json.load(f)

best_val_by_seed = {p["seed"]: p["best_val_auprc"] for p in progress}

rows = []
for m in summary["per_model"]:
    rows.append(
        {
            "name": f"seed_{m['seed']}",
            "best_val_auprc": best_val_by_seed.get(m["seed"], np.nan),
            "test_auroc": m["test_auroc"],
            "test_auprc": m["test_auprc"],
            "test_acc": m["test_acc"],
            "duration": fmt_duration(m["duration_s"]),
        }
    )

ens = summary["ensemble"]
rows.append(
    {
        "name": "ensemble",
        "best_val_auprc": np.nan,
        "test_auroc": ens["test_auroc"],
        "test_auprc": ens["test_auprc"],
        "test_acc": ens["test_acc"],
        "duration": fmt_duration(summary["total_duration_s"]),
    }
)

df = pd.DataFrame(rows)

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].round(3)

print(df.to_latex(index=False, na_rep="--"))
