"""
German Analogy Evaluator
========================

Evaluate German word analogies using precomputed word embeddings and the
3CosAdd solver.

Given type-level embeddings (one vector per word) and an analogy list
of the form:

    A  B  C  D

we test linear relations like:

    mann - frau ≈ könig - königin

by checking whether the predicted target word D̂ matches the gold D.

Pipeline
--------
1. Load token-level embeddings and metadata.
2. Filter to “clean” word forms (regex + minimum frequency).
3. Mean-pool token vectors to get type embeddings.
4. L2-normalize all type vectors.
5. For each analogy:
    - Compute query vector q = v_B - v_A + v_C (3CosAdd).
    - Search the candidate pool (optionally restricted per category).
    - Record the predicted word and accuracy.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm


def cos(a, b) -> float:
    """Cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))


def build_word_means(
    emb_npz: str,
    meta_parquet: str,
    case_sensitive: bool = False,
    min_freq: int = 1,
    clean_regex: str = r"[a-zäöüß]{3,}",
):
    """
    Build a word -> vector dictionary by:
      - loading token embeddings + metadata
      - filtering to clean word forms
      - applying a minimum token frequency
      - mean-pooling token vectors per word type
      - L2-normalizing type vectors

    Returns
    w2v : dict[str, np.ndarray]
        Mapping from word type to its normalized embedding.
    """
    V = np.load(emb_npz)["emb"]          
    M = pd.read_parquet(meta_parquet)   
    assert len(V) == len(M), f"len mismatch {len(V)} vs {len(M)}"

    words = M["word"].astype(str)
    if not case_sensitive:
        words = words.str.lower()

   
    mask = words.str.fullmatch(clean_regex, case=False)
    V = V[mask.values]
    words = words[mask].reset_index(drop=True)

   
    vc = words.value_counts()
    keep = words.map(lambda w: vc.get(w, 0) >= min_freq)
    V = V[keep.values]
    words = words[keep].reset_index(drop=True)

    # type means
    df = pd.DataFrame(V)
    df["word"] = words
    means = df.groupby("word").mean()
    X = means.to_numpy() 

    # L2 normalization
    X = X / (norm(X, axis=1, keepdims=True) + 1e-9)

    idx = list(means.index)
    w2v = {w: X[i] for i, w in enumerate(idx)}
    print(f"[info] type vocab after filters: {len(w2v)}")
    return w2v


def three_cos_add(a, b, c, word2vec, candidates=None):
    """
    3CosAdd solver:
        q = v_B - v_A + v_C
    """
    if any(w not in word2vec for w in [a, b, c]):
        return None, None

    tgt = word2vec[b] - word2vec[a] + word2vec[c]
    best, score = None, -1.0
    pool = candidates if candidates is not None else word2vec.keys()

    for w in pool:
        if w in (a, b, c):
            continue
        v = word2vec.get(w)
        if v is None:
            continue
        s = cos(v, tgt)
        if s > score:
            best, score = w, s
    return best, score


def build_category_pools(analogies_df: pd.DataFrame):
    pools = {}
    for cat, sub in analogies_df.groupby("category"):
        pool = set(sub["A"]) | set(sub["B"]) | set(sub["C"]) | set(sub["D"])
        pools[cat] = pool
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--meta_parquet", required=True)
    ap.add_argument("--analogies_tsv", required=True)
    ap.add_argument("--out_csv", default="analogy_results_3cosadd.csv")
    ap.add_argument("--case_sensitive", action="store_true")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--clean_regex", default=r"[a-zäöüß]{3,}")
    ap.add_argument("--restrict_by_category", action="store_true")
    args = ap.parse_args()

    # Build type-level word embeddings 
    word2vec = build_word_means(
        args.emb_npz,
        args.meta_parquet,
        case_sensitive=args.case_sensitive,
        min_freq=args.min_freq,
        clean_regex=args.clean_regex,
    )

    # Load analogy list
    analogies = pd.read_csv(args.analogies_tsv, sep="\t")
    if not args.case_sensitive:
        for col in ["A", "B", "C", "D", "category"]:
            analogies[col] = analogies[col].astype(str).str.lower()

    # Filter analogies to ones fully in vocab
    mask = analogies.apply(
        lambda r: all(w in word2vec for w in [r.A, r.B, r.C, r.D]), axis=1
    )
    analogies = analogies[mask].reset_index(drop=True)
    print(f"[info] usable analogies: {len(analogies)}")

    pools = build_category_pools(analogies) if args.restrict_by_category else {}

    rows = []
    for _, r in analogies.iterrows():
        cat, a, b, c, d = r["category"], r["A"], r["B"], r["C"], r["D"]
        cand = pools.get(cat) if args.restrict_by_category else None
        pred, score = three_cos_add(a, b, c, word2vec, candidates=cand)
        rows.append((cat, a, b, c, d, pred, score))

    df = pd.DataFrame(rows, columns=["category", "A", "B", "C", "D", "pred", "score"])
    df["correct"] = df["D"] == df["pred"]
    acc = df["correct"].mean() if len(df) > 0 else 0.0
    print(f"[RESULT] 3cosadd  acc={acc:.3f}  ({df['correct'].sum()}/{len(df)})")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[ok] saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
