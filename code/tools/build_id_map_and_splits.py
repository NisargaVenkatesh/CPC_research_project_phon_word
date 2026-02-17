import argparse, json, random
from pathlib import Path
import pandas as pd

SPECIALS = ["<pad>", "<unk>"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Dataset root containing HUI and BAS folders')
    ap.add_argument('--hui_dir', default='HUI_CPC_txt')
    ap.add_argument('--bas_dir', default='BAS_BITS_US_3_no_interp_50_400')
    ap.add_argument('--splits_dir', default='splits')
    ap.add_argument('--expect_vocab_size', type=int, default=None)  # optional
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def list_txts(d: Path):
    return sorted([p for p in d.rglob('*.txt') if p.is_file()])

def collect_phonemes(paths):
    toks = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep="\t", header=0)
            if "name" in df.columns:
                toks.extend(df.loc[df["name"] != "<p:>", "name"].astype(str).tolist())
        except Exception:
            pass
    return toks

def main():
    args = parse_args()
    random.seed(args.seed)

    root = Path(args.root)
    hui = root / args.hui_dir
    bas = root / args.bas_dir
    assert hui.exists(), f"HUI dir not found: {hui}"
    assert bas.exists(), f"BAS dir not found: {bas}"

    hui_txts = list_txts(hui)
    bas_txts = list_txts(bas)

    toks = collect_phonemes(hui_txts) + collect_phonemes(bas_txts)
    vocab = sorted(set(toks) - set(SPECIALS))

    id_map = {"<pad>": 0, "<unk>": 1}
    for i, t in enumerate(vocab, start=2):
        id_map[t] = i

    id_map_path = root / 'id_map.json'
    id_map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding='utf-8')

    # Splits (80/10/10) on HUI
    files = [str(p) for p in hui_txts]
    random.shuffle(files)
    n = len(files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    splits_dir = root / args.splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / 'train.txt').write_text('\n'.join(files[:n_train]) + '\n', encoding='utf-8')
    (splits_dir / 'val.txt').write_text('\n'.join(files[n_train:n_train+n_val]) + '\n', encoding='utf-8')
    (splits_dir / 'test.txt').write_text('\n'.join(files[n_train+n_val:]) + '\n', encoding='utf-8')

    report = {
        'vocab_size_including_specials': len(id_map),
        'id_map_path': str(id_map_path),
        'splits_dir': str(splits_dir),
        'counts': {
            'train': n_train, 'val': n_val, 'test': n_test,
            'hui_txts': len(hui_txts), 'bas_txts': len(bas_txts),
        }
    }
    if args.expect_vocab_size is not None:
        report['matches_expectation'] = (len(id_map) == args.expect_vocab_size)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
