
"""
Make a matched clean pair (embeddings NPZ + meta Parquet).
- Filters to whole words (letters incl. äöüß, len>=3)
- Optional min frequency threshold
- Keeps rows in sync so len(emb)==len(meta) after filtering
"""

import argparse, re, numpy as np, pandas as pd
from pathlib import Path
from numpy.linalg import norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_in",  required=True)
    ap.add_argument("--meta_in", required=True)
    ap.add_argument("--emb_out", required=True)
    ap.add_argument("--meta_out",required=True)
    ap.add_argument("--min_freq", type=int, default=1, help="min occurrences per word to keep")
    args = ap.parse_args()

    V = np.load(args.emb_in)["emb"]
    M = pd.read_parquet(args.meta_in)
    assert len(V) == len(M), f"length mismatch: emb={len(V)} meta={len(M)}"

    # keep only clean whole words (letters incl. äöüß, >=3 chars)
    words = M["word"].astype(str).str.lower()
    is_clean = words.str.fullmatch(r"[a-zäöüß]{3,}", case=False)

    # frequency filter on the clean subset
    M_tmp = M.loc[is_clean].copy()
    words_tmp = words[is_clean]
    freq = words_tmp.value_counts()
    keep_word = words_tmp.map(lambda w: freq.get(w, 0) >= args.min_freq)
    keep_mask = is_clean.copy()
    keep_mask.loc[is_clean] = keep_word.values

    V2 = V[keep_mask.values]
    M2 = M.loc[keep_mask].reset_index(drop=True)

    assert len(V2) == len(M2)
    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.emb_out, emb=V2)
    M2.to_parquet(args.meta_out, index=False)

    uniq = M2["word"].astype(str).str.lower().nunique()
    print(f"[ok] kept {len(M2)}/{len(M)} tokens; unique words={uniq}")
    print(f"[ok] wrote: {args.emb_out}")
    print(f"[ok] wrote: {args.meta_out}")

if __name__ == "__main__":
    main()
