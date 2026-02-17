
import re, numpy as np, pandas as pd
from pathlib import Path

EMB_IN  = "Analysis/40_reports/val_hui_context.npz"
META_IN = "Analysis/40_reports/val_hui_context_meta_words_v2.parquet"
EMB_OUT = "Analysis/40_reports/val_hui_context_clean.npz"
META_OUT= "Analysis/40_reports/val_hui_context_meta_words_clean.parquet"

V = np.load(EMB_IN)["emb"]
M = pd.read_parquet(META_IN)
assert len(V) == len(M), f"len mismatch {len(V)} vs {len(M)}"

is_clean = M["word"].astype(str).str.fullmatch(r"[a-zäöüß]{3,}", case=False)
V2 = V[is_clean.values]
M2 = M.loc[is_clean].reset_index(drop=True)

Path(META_OUT).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(EMB_OUT, emb=V2)
M2.to_parquet(META_OUT, index=False)

print(f"[ok] kept {len(M2)}/{len(M)} tokens; unique words={M2['word'].str.lower().nunique()}")
