from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# WordBuilder
class WordBuilder(nn.Module):
    """
    Encodes a word (sequence of phoneme IDs) -> fixed-size vector.

    Pipeline:
      IDs -> Embedding -> 1D CNN -> BiLSTM -> masked pooling -> Linear(E)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 96,
        cnn_channels: int = 128,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        use_cnn: bool = True,
        pooling: str = "mean",  
    ):
        super().__init__()
        assert pooling in {"mean", "last"}

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_cnn = use_cnn
        self.pooling = pooling

        if self.use_cnn:
            pad = (kernel_size - 1) // 2  
            self.cnn = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
                padding=pad,
            )
            self.post_cnn_dim = cnn_channels
        else:
            self.post_cnn_dim = embed_dim

        self.bilstm = nn.LSTM(
            input_size=self.post_cnn_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.out = nn.Linear(2 * lstm_hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        single = False
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)  
            single = True

        x = self.embed(word_ids)           
        x = self.dropout(x)

        if self.use_cnn:
            x = x.transpose(1, 2)         
            x = F.gelu(self.cnn(x))
            x = x.transpose(1, 2)           

        x, _ = self.bilstm(x)              
        x = self.dropout(x)

        if self.pooling == "mean":
            mask = (word_ids != 0).float()              
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom   
        else:  
            mask = (word_ids != 0)
            last_idx = mask.long().sum(dim=1).clamp(min=1) - 1     
            pooled = x[torch.arange(x.size(0), device=x.device), last_idx]  

        out = self.out(pooled)              
        return out.squeeze(0) if single else out


# WordCPCModel (masked InfoNCE)

class WordCPCModel(nn.Module):
    """
    CPC over sequences of word embeddings.
    - Proper masking of padded time-steps.
    - Cosine-style logits with learnable temperature (via log-param).
    - Across-batch negatives by default (stronger).
    """
    def __init__(
        self,
        word_builder: WordBuilder,
        context_hidden: int = 256,
        context_layers: int = 1,
        prediction_steps: int = 3,
        dropout: float = 0.1,
        use_across_batch_negatives: bool = True,
    ):
        super().__init__()
        self.word_builder = word_builder
        self.prediction_steps = int(prediction_steps)
        self.use_across_batch_negatives = use_across_batch_negatives

        E = self.word_builder.out.out_features

        self.context_rnn = nn.LSTM(
            input_size=E,
            hidden_size=context_hidden,
            num_layers=context_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if context_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        self.Wk = nn.ModuleList([nn.Linear(context_hidden, E) for _ in range(self.prediction_steps)])

        # Stable temperature: exp(logit_scale_log) > 0
        self.logit_scale_log = nn.Parameter(torch.tensor(0.0))  # init scale=1.0

    @staticmethod
    def _vectorize_words(word_seqs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[int]]:
        """
        Flatten words across batch and pad to [Nw, Tmax].
        Returns (padded_words, words_per_utt).
        """
        flat_words: List[torch.Tensor] = []
        words_per_utt: List[int] = []
        for seq in word_seqs:
            words_per_utt.append(len(seq))
            flat_words.extend(seq)

        if len(flat_words) == 0:
            flat_words = [torch.zeros(1, dtype=torch.long)]
            words_per_utt = [1]

        padded_words = pad_sequence(flat_words, batch_first=True, padding_value=0)  
        return padded_words, words_per_utt

    @staticmethod
    def _rebuild_utterances(word_vecs: torch.Tensor, words_per_utt: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack per-word vectors back to [B, T, E] with lengths.
        """
        splits = list(torch.split(word_vecs, words_per_utt))
        lengths = torch.tensor([x.size(0) for x in splits], device=word_vecs.device)
        padded_batch = pad_sequence(splits, batch_first=True)  
        return padded_batch, lengths

    def forward(self, word_seqs: List[List[torch.Tensor]]) -> torch.Tensor:
        device = next(self.parameters()).device
        scale = self.logit_scale_log.exp()

        # 1) Build all word embeddings
        padded_words, words_per_utt = self._vectorize_words(word_seqs)  
        padded_words = padded_words.to(device)
        all_word_vecs = self.word_builder(padded_words)                

        # 2) Rebuild utterance sequences
        padded_batch, lengths = self._rebuild_utterances(all_word_vecs, words_per_utt)
        padded_batch = padded_batch.to(device)                          
        lengths = lengths.to(device)                                    

        # 3) Context RNN
        packed = pack_padded_sequence(padded_batch, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.context_rnn(packed)
        context, _ = pad_packed_sequence(packed_out, batch_first=True)   
        context = self.dropout(context)

        # 4) Masked InfoNCE
        total_loss = 0.0
        B, T, _ = context.shape
        if B == 0 or T == 0:
            return context.sum() * 0.0

        for k in range(1, self.prediction_steps + 1):
            maxT = context.size(1) - k
            if maxT <= 0:
                continue

            pred = self.Wk[k - 1](context[:, :maxT, :])                  
            target = padded_batch[:, k : k + maxT, :]                    

            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

            # Row mask for valid steps at horizon k
            lengths_k = (lengths - k).clamp(min=0)                       
            t_idx = torch.arange(maxT, device=device).unsqueeze(0)       
            mask_bt = t_idx < lengths_k.unsqueeze(1)                    

            if self.use_across_batch_negatives:
                # Use all valid targets across batch as negatives.
                pred_valid = pred[mask_bt]                                
                target_valid = target[mask_bt]                            
                if pred_valid.numel() == 0:
                    continue

                logits = pred_valid @ target_valid.t() * scale            
                labels = torch.arange(pred_valid.size(0), device=device)  
                loss_k = F.cross_entropy(logits, labels)
            else:
                
                losses_b = []
                for b in range(B):
                    valid = int((lengths[b] - k).clamp(min=0).item())
                    if valid <= 0:
                        continue
                    logits_b = pred[b, :valid, :] @ target[b, :valid, :].T * scale  
                    labels_b = torch.arange(valid, device=device)
                    losses_b.append(F.cross_entropy(logits_b, labels_b))
                if not losses_b:
                    continue
                loss_k = torch.stack(losses_b).mean()

            total_loss = total_loss + loss_k

        return total_loss / max(1, self.prediction_steps)
