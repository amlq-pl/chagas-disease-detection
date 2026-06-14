import pandas as pd
import numpy as np

SUMMARY_PATH = "summary.json"

df = pd.read_json(SUMMARY_PATH)
df.drop(columns=["n_params", "best_epoch"], inplace=True)
mins = df["duration_s"].apply(lambda x: int(x // 60))
secs = df["duration_s"].apply(lambda x: int(round(x % 60)))
df["duration"] = mins.astype(str) + ' min ' + secs.astype(str) + ' s'
df.drop(columns=["duration_s"], inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].round(3)

print(df.to_latex(index=False))