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

```md
## Repository Structure

```text
CPC_research_project_phon_word/
|-- code/
|   |-- train_word_cpc.py
|   |-- __init__.py
|   |-- src/
|   |   |-- dataset/
|   |   |   |-- __init__.py
|   |   |   `-- word_txt_dataset.py
|   |   `-- torch_models/
|   |       |-- __init__.py
|   |       `-- _word_models.py
|   `-- tools/
|       `-- build_id_map_and_splits.py
|
|-- Analysis/
|   `-- 20_eval/
|       |-- hui_in_corpus_analogies.tsv
|       |-- t1_eval_german_analogies_boosted.py
|       |-- t1_make_clean_context.py
|       |-- t1_make_clean_pair.py
|       |-- t1_show_analogy_pass_fail.py
|       `-- t1_4_plot_analogy_topk.py
|
|-- Dataset/
|   `-- id_map.json
|
|-- Checkpoint/
|   |-- wordcpc_20251031_220029_fullfit_cnnfixed_best_es10_best.pt
|   `-- wordcpc_20251031_220029_fullfit_cnnfixed_best_es10_wordbuilder.pt
|
|-- requirements.txt
`-- README.md

Installation

1. Create and activate an environment

conda create -n wordcpc python=3.10
conda activate wordcpc

1. Install dependencies

pip install -r requirements.txt
