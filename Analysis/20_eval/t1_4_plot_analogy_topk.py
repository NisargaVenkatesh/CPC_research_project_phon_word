import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load CSVs
ctx = pd.read_csv("logs/ctx_add_full_topk.csv")
wb  = pd.read_csv("logs/wb_add_topk.csv")

def compute_topk(df):
    # group by category
    g = df.groupby("category")

    out = []
    for cat, sub in g:
        n = len(sub)
        top1 = sub["hit_top1"].mean()
        top5 = sub["hit_top5"].mean()
        top10 = sub["hit_top10"].mean()
        out.append((cat, top1, top5, top10))
    return pd.DataFrame(out, columns=["category", "Top1", "Top5", "Top10"]).sort_values("category")

ctx_k = compute_topk(ctx)
wb_k  = compute_topk(wb)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
models = [("Context", ctx_k), ("WordBuilder", wb_k)]

for ax, (name, dfk) in zip(axes, models):
    x = range(len(dfk))
    ax.bar([i - 0.2 for i in x], dfk["Top1"], width=0.2, label="Top-1")
    ax.bar([i       for i in x], dfk["Top5"], width=0.2, label="Top-5")
    ax.bar([i + 0.2 for i in x], dfk["Top10"], width=0.2, label="Top-10")

    ax.set_xticks(x)
    ax.set_xticklabels(dfk["category"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(name)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

axes[0].set_ylabel("Accuracy")
axes[0].legend()

Path("figs").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("figs/t1_analogy_topk_by_category.png", dpi=300)
plt.show()

print("[ok] saved to figs/analogy_topk_by_category.png")
