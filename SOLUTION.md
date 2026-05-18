# Track C Solution

## Table of Contents

1. [Reproducibility Instructions](#1-reproducibility-instructions)  
2. [Track C Overview](#2-track-c-overview)  
3. [Final Architecture](#3-final-architecture)  
4. [Why Track C Improved Over Track B](#4-why-track-c-improved-over-track-b)  
5. [Feature and Ensemble Search](#5-feature-and-ensemble-search)  
6. [Experiments and Failed Attempts](#6-experiments-and-failed-attempts)  
7. [Important Notes](#7-important-notes)  

---

# 1. Reproducibility Instructions

```bash
git clone https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection
cd SMILES-2026-Hallucination-Detection
git checkout second_iter_track_C
pip install -r requirements.txt
python solution.py
```

Track C additionally requires:

```python
output_attentions=True
```

because the solution extracts full transformer attention maps.

The implementation is experimental and extremely slow inside the official pipeline.

---

# 2. Track C Overview

Track C extends Track B with attention-aware infrastructure.

Main additions:

- attention-aware feature extraction;
- grounding decay dynamics;
- prompt-response attention persistence;
- retrieval-style grounding features;
- attention collapse features;
- attention sink statistics.

Unlike Track A and Track B, Track C uses both:

- hidden states;
- transformer attention maps.

The implementation is currently very raw and research-oriented.  
The code was originally developed inside notebook experiments and later partially transferred into the official pipeline.

As a result:

- the implementation contains significant technical debt;
- extraction is extremely slow;
- the official pipeline runtime becomes very large.

---

# 3. Final Architecture

The final Track C solution was:

```text
C_greedy_step4_4blocks_rank_fusion
```

It combines four independently trained feature views:

```text
B__extra_smart_prompt_len_all__top312_pca64
C__attention_all__top866_pca64
B__advanced_prompt_len_max_mean__top1250_pca128
C__attention_sink__selected4
```

Each block uses:

```text
SelectKBest
→ PCA
→ LogisticRegression(C=0.003)
```

Final probabilities are combined using rank fusion.

Final selected feature count:

```text
2432
```

Best notebook result:

```text
Test AUROC = 81.37%
```

---

# 4. Why Track C Improved Over Track B

Early Track C attention-only views performed poorly as standalone models.

The largest improvement appeared when attention features were combined with strong Track B prompt-aware hidden-state views.

The main breakthrough came from:

- combining prompt-aware geometry with attention grounding behavior;
- adding attention sink statistics;
- introducing grounding persistence and decay features;
- rank-fusion ensembling of complementary feature spaces.

Notebook experiments showed that Track C attention features alone were unstable, but they became highly useful as complementary signals inside multi-view ensembles.

This was the first stage where notebook AUROC consistently exceeded Track B results.

---

# 5. Feature and Ensemble Search

Track C was heavily tuned through iterative notebook experiments.

Main research stages included:

- standalone view benchmarking;
- feature-family ablations;
- compact-view pruning;
- pair/triple/greedy ensemble search;
- late-fusion experiments;
- meta-model experiments;
- threshold tuning;
- calibration experiments;
- bagged logistic regression;
- CatBoost meta-learning.

The final Track C architecture emerged from greedy fusion search over multiple Track A, Track B, and Track C feature spaces.

Several fusion strategies were tested:

```text
rank fusion
probability averaging
meta logistic regression
CatBoost meta models
```

The final notebook solution selected rank fusion because it provided the best balance between:

- AUROC;
- stability;
- train-test generalization gap.

---

# 6. Experiments and Failed Attempts

Several ideas improved validation metrics but were eventually discarded:

- large CatBoost meta-models;
- heavy bagging configurations;
- large attention-only solutions;
- overcompressed feature spaces;
- aggressive calibration pipelines.

Attention-only models were especially unstable and frequently overfit.

Track C worked best when attention signals were used as small complementary views rather than dominant standalone models.

---

# 7. Important Notes

Track C should be considered an experimental research branch rather than a production-ready official solution.

The notebook version relied heavily on:

- cached parquet features;
- offline extraction;
- iterative feature selection;
- precomputed attention statistics.

Reproducing the same pipeline inside the official framework is extremely slow because full attention tensors must be materialized online for every sample.

For this reason, the reported Track C quality should primarily be interpreted as a notebook research result rather than a lightweight reproducible benchmark pipeline.
