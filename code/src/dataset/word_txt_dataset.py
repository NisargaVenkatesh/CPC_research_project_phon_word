from pathlib import Path
from typing import List, Dict, Optional, Sequence, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import random, numpy as np

# Reproducibility for any dataset-side randomness
random.seed(42); np.random.seed(42)

PathLike = Union[str, Path]

class WordTxtDataset(Dataset):
    """
    Dataset wrapper for HUI phone-table .txt files.

    Each .txt file is treated as one utterance. The file must contain at least:
      - 'name'          : phoneme symbol per row
      - 'start_of_word' : 1 if this row starts a new word, 0 otherwise

    For each utterance, the dataset:
      1) Reads the phone table.
      2) Segments the sequence into words using 'start_of_word'.
      3) Maps phoneme symbols -> integer IDs via id_map (with <unk> fallback).
      4) Returns: List[LongTensor], where each tensor is one word's phoneme ID sequence.
    
    Reads *.txt phone tables (HUI/BAS shape) and returns per-utterance words:
      __getitem__(i) -> List[LongTensor]  # each tensor is a single word's phoneme IDs

    """
    def __init__(
        self,
        data_dir: PathLike,
        id_map: Dict[str, int],
        pattern: str = "*.txt",
        max_files: Optional[int] = None,
        min_words: int = 1,
        split_file: Optional[PathLike] = None,
    ):
        self.data_dir = Path(data_dir)
        self.id_map = id_map
        self.unk_id = id_map.get("<unk>", 1)

        if split_file is not None:
            split_file = Path(split_file)
            lines = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
            files: List[Path] = []
            for p in lines:
                pth = Path(p)
                if pth.is_absolute() and pth.exists():
                    files.append(pth)
                elif p.startswith("Dataset/"):
                    files.append(Path(p))
                else:
                    files.append(self.data_dir / pth)
        else:
            files = sorted(self.data_dir.glob(pattern))

        if max_files is not None:
            files = files[:max_files]

        # Parse each file into list-of-words (list-of-tensors) 
        self.samples: List[List[torch.Tensor]] = []
        kept = 0
        for fp in files:
            try:
                df = pd.read_csv(fp, sep=r"\s+", engine="python")
                # need at least 'name' and 'start_of_word'
                if "name" not in df or "start_of_word" not in df:
                    continue
                phones = df["name"].astype(str).tolist()
                sow = df["start_of_word"].astype(int).tolist()

                # word start indices
                starts = [i for i, v in enumerate(sow) if v == 1]
                if not starts:
                    starts = [0]
                starts.append(len(phones))

                words: List[torch.Tensor] = []
                for i in range(len(starts) - 1):
                    seg = phones[starts[i]:starts[i+1]]
                    ids = [self.id_map.get(p, self.unk_id) for p in seg]
                    if ids:
                        words.append(torch.tensor(ids, dtype=torch.long))

                if len(words) >= min_words:
                    self.samples.append(words)
                    kept += 1
            except Exception:
                pass

        label = str(split_file) if split_file is not None else "FULL"
        print(f"[INFO] WordTxtDataset: {label} -> files={len(files)}  usable_utterances={kept}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        return self.samples[idx]

    @staticmethod
    def collate(batch: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        if not batch:
            raise RuntimeError("Empty batch in collate; check dataset/filtering.")
        return batch




     