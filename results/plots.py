import json
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

VARIANTS = [
    "inception",
    "inception_meta",
    "inception_meta_aug",
    "resnet18",
    "resnet18_meta",
    "resnet18_meta_aug",
]

models = []
for v in VARIANTS:
    with open(f"{v}/metrics.json") as f:
        models.append(json.load(f))

fig, ax = plt.subplots(figsize=(8, 6))
for m in models:
    history = pd.DataFrame(m["history"])
    ax.plot(history["epoch"], history["val_auprc"], label=m["variant"])

ax.set_xlabel("epoch")
ax.set_ylabel("val AUPRC")
ax.set_title("Krzywe uczenia (val AUPRC) dla wszystkich modeli")
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.legend()
fig.tight_layout()
fig.savefig("val_auprc_all_models.png", dpi=150)