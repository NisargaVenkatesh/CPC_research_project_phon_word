"""
Show which analogies passed or failed in the logs.
Compares WordBuilder vs Context model side by side.
"""

import pandas as pd
from pathlib import Path

ctx_csv = "logs/ctx_add_full.csv"
wb_csv  = "logs/wb_add.csv"

def load_results(csv_path, label):
    df = pd.read_csv(csv_path)
    df["model"] = label
    df["mark"] = df["correct"].map({True:"✓", False:"✗"})
    return df

ctx = load_results(ctx_csv, "Context")
wb  = load_results(wb_csv,  "WordBuilder")

# Combine
all_df = pd.concat([ctx, wb])
all_df = all_df.sort_values(["category", "A", "B", "C", "D", "model"])

# Summarize by model
for model, sub in all_df.groupby("model"):
    print(f"\n=== {model} ===")
    print(f"Accuracy: {sub['correct'].mean():.3f} ({sub['correct'].sum()}/{len(sub)})")
    print(f"{'Cat':<15} {'A':<10} {'B':<10} {'C':<10} {'D (target)':<12} {'Pred':<12} {'✓/✗':<2}")
    print("-"*70)
    for _,r in sub.iterrows():
        print(f"{r.category:<15} {r.A:<10} {r.B:<10} {r.C:<10} {r.D:<12} {r.pred:<12} {r.mark:<2}")
    print("-"*70)

# just failures for inspection
failures = all_df[~all_df["correct"]]
Path("logs/analogy_failures.tsv").write_text(failures.to_csv(sep="\t", index=False))
print("\n[ok] wrote detailed failure list to logs/analogy_failures.tsv")
