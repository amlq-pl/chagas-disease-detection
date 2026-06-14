import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

df = pd.read_json("_progress.json")

fig, ax = plt.subplots(figsize=(8, 5))
for row in df.itertuples(index=False):
    h = pd.DataFrame(row.history)
    ax.plot(h["epoch"], h["val_auprc"], label=f"seed={row.seed}")

ax.set_xlabel("epoch")
ax.set_ylabel("val AUPRC")
ax.set_title("Krzywe uczenia (val AUPRC) dla 5 seedów")
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.legend()
fig.tight_layout()
fig.savefig("val_auprc_per_seed.png", dpi=150)
plt.show()

