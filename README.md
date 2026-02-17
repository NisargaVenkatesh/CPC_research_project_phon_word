# Word-Level Contrastive Predictive Coding

This repository contains the code for Word-Level CPC (Word-CPC) project.
The goal is to extend phoneme-level CPC to word-level CPC representations and
evaluate whether word-level future prediction captures linguistically meaningful
structure, in particular for German.

The project focuses on:

- A **WordBuilder** module that aggregates phoneme sequences into word vectors.
- A **Word-CPC** model that predicts future word embeddings using an InfoNCE loss.
- **Intrinsic evaluations** on the validation set: 
  - German word analogy task (3CosAdd).
  - Verb morphology linear probe (person/number labels via spaCy).

---

## Repository Structure

From the root `wordcpc_final/`:

wordcpc_final/
├── code/
│ ├── train_word_cpc.py # Main training script for Word-CPC
│ ├── tools/
│ │ └── build_id_map_and_splits.py # Utility to build joint id_map and train/val/test splits
│ └── src/
│ ├── dataset/
│ │ ├── **init**.py
│ │ └── word_txt_dataset.py # Dataset wrapper for HUI \*.txt phone tables
│ └── torch_models/
│ ├── **init**.py
│ └── \_word_models.py # WordBuilder + WordCPCModel definitions
│
├── Analysis/
│ ├── 00_export/
│ │ └── export_word_embeddings.py # Export context / WordBuilder embeddings from a checkpoint
│ ├── 20_eval/
│ │ ├── hui_in_corpus_analogies.tsv # T1 analogy list used in the report
│ │ ├── t1_eval_german_analogies_boosted.py # 3CosAdd analogy evaluator
│ │ ├── t1_make_clean_context.py # Simple cleaner for context embeddings/meta
│ │ ├── t1_make_clean_pair.py # Generic cleaner for (emb, meta) pairs
│ │ ├── t1_show_analogy_pass_fail.py # Print pass/fail analogies
│ │ └── t1_4_plot_analogy_topk.py # Plot Top-1/5 analogy accuracy by category
│ ├── 40_reports/
│ │ ├── val_hui_context.npz # Context embeddings (tokens) [provided]
│ │ ├── val_hui_context_meta.parquet # Context meta (tokens) [provided]
│ │ ├── val_hui_wordbuilder.npz # WordBuilder embeddings (tokens) [provided]
│ │ ├── val_hui_wordbuilder_meta.parquet # WordBuilder meta (tokens) [provided]
│ │ └── probes/
│ │ ├── val_hui_context_clean.npz # Cleaned context embeddings
│ │ ├── val_hui_context_meta_words_clean.parquet
│ │ ├── val_hui_wordbuilder_clean.npz # Cleaned WB embeddings
│ │ ├── val_hui_wordbuilder_meta_words_clean.parquet
│ │ ├── context_probe_results.txt # Verb morphology probe results (context)
│ │ └── wordbuilder_probe_results.txt # Verb morphology probe results (WordBuilder)
│ └── 50_probing/
│ ├── build_verb_morph_labels.py # Build verb labels from POS+spaCy morphology
│ └── probe_verb_morph.py # Linear probe on verb morphology
│
├── Dataset/
│ ├── id_map.json # Joint phoneme→ID mapping for HUI
│ ├── HUI_sentence_df.json # Sentence-level metadata for HUI (used in analysis)
│ └── splits/ # (Expected) train/val/test file lists (not included here)
├── requirements.txt
└── README.md

Installation

1. Create and activate an environment

conda create -n wordcpc python=3.10
conda activate wordcpc

1. Install dependencies

pip install -r requirements.txt