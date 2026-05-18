# Hallucination Detection — Track B Solution

## Table of Contents

1. [Reproducibility Instructions](#1-reproducibility-instructions)  
2. [Track B Overview](#2-track-b-overview)  
3. [Main Difference from the Honest Pipeline](#3-main-difference-from-the-honest-pipeline)  
4. [Single Model (`solution_single.py`)](#4-single-model-solution_singlepy)  
   - [4.1 Feature Extraction](#41-feature-extraction)  
   - [4.2 Final Architecture](#42-final-architecture)  
   - [4.3 Why This Worked](#43-why-this-worked)  
5. [Meta Model (`solution_meta.py`)](#5-meta-model-solution_metapy)  
6. [Important Implementation Details](#6-important-implementation-details)  
7. [Experiments and Discarded Ideas](#7-experiments-and-discarded-ideas)  
8. [Final Notes](#8-final-notes)  

---

# 1. Reproducibility Instructions

Clone the repository and install dependencies:

```bash
git clone https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection
cd SMILES-2026-Hallucination-Detection
git checkout second_iter_track_B
pip install -r requirements.txt
```

Run the single-model Track B pipeline:

```bash
python solution_single.py
```

Run the meta-model Track B pipeline:

```bash
python solution_meta.py
```

Both pipelines generate:

```text
predictions.csv
results.json
```

The repository uses the same official infrastructure as the baseline competition solution:

- Qwen/Qwen2.5-0.5B
- hidden-state extraction
- sklearn-based probe training
- official evaluation loop

The main hardware requirement is GPU inference support because the pipeline performs full hidden-state extraction over all transformer layers.

---

# 2. Track B Overview

Track B extends the honest hidden-state solution by introducing exact prompt/response segmentation through `prompt_len`.

Unlike Track A, which uses heuristic response regions such as `last30` or `last20`, Track B computes the exact boundary between prompt and generated response using tokenizer lengths.

This allows the pipeline to construct:

- exact prompt masks;
- exact response masks;
- prompt-aware geometry;
- response-only dynamics;
- precise prompt-response interaction statistics.

The rest of the infrastructure remains largely unchanged:

- hidden-state extraction;
- linear sklearn probes;
- feature selection;
- PCA compression.

Track B contains two separate solutions:

1. a single-model pipeline;
2. a meta-model combination pipeline.

---

# 3. Main Difference from the Honest Pipeline

The only conceptual difference between Track B and the fully honest solution is the addition of:

```text
prompt_len
```

This enables exact prompt/response separation during feature extraction.

No additional external models are used.

Track B still relies entirely on:

- hidden states;
- transformer geometry;
- handcrafted statistical features.

The pipeline does not use:

- logits;
- verifier models;
- retrieval systems;
- external APIs.

---

# 4. Single Model (`solution_single.py`)

The repository contains a standalone Track B solution implemented through:

```text
solution_single.py
aggregation_single.py
probe_single.py
```

This model uses exact prompt-aware hidden-state geometry.

---

## 4.1 Feature Extraction

The feature extractor implemented in `aggregation_single.py` builds a very large hidden-state feature space using exact prompt-aware masks.

The pipeline first computes:

- exact prompt tokens;
- exact response tokens;
- prompt subregions;
- response subregions.

The extractor then creates features for:

### Exact Prompt/Response Pooling

Features are extracted independently for:

- prompt;
- response;
- response early;
- response middle;
- response late;
- response last 5 tokens;
- response last 10 tokens.

Pooling statistics include:

- mean pooling;
- standard deviation;
- token norms;
- activation entropy;
- covariance structure.

---

### Cross-Layer Drift Features

For layers 10–16, the extractor computes:

- cosine drift;
- L2 drift;
- long-range layer differences;
- update consistency;
- response trajectory stability.

Several predefined layer-pair groups are used:

```text
DRIFT_PAIRS
RICH_DRIFT_PAIRS
LONG_DRIFT_PAIRS
```

These features capture how representations evolve through transformer depth.

---

### Temporal Dynamics

The pipeline computes token-level trajectory statistics over response regions:

- slope;
- roughness;
- acceleration;
- smoothness;
- late-vs-early behavior;
- sign-change statistics.

These dynamics are especially important for detecting unstable generations.

---

### Spectral Features

The extractor also computes compact spectral statistics:

- FFT energy;
- low/high frequency energy;
- spectral entropy;
- dominant frequency;
- participation ratio;
- effective rank.

These features characterize hidden-state structure complexity.

---

### Pairwise Geometry

Additional geometry statistics include:

- pairwise cosine similarity;
- centroid similarity;
- covariance PCA statistics;
- token disagreement;
- anisotropy-like behavior.

---

The final Track B single-model feature dimensionality is:

```text
41397 features
```

---

## 4.2 Final Architecture

The final classifier implemented in `probe_single.py` is:

```text
SimpleImputer(strategy="median")
→ SelectKBest(f_classif, k=312)
→ StandardScaler
→ PCA(n_components=32)
→ LogisticRegression(
      C=0.003,
      penalty="l2",
      solver="lbfgs"
  )
→ threshold=0.5
```

Important implementation details:

- `class_weight=None`
- `max_iter=3000`
- `random_state=42`

Threshold tuning is effectively disabled:

```text
threshold = 0.5
```

The model is optimized primarily for AUROC stability.

---

## 4.3 Why This Worked

The largest improvements came from:

- exact prompt-response separation;
- response-only hidden-state geometry;
- cross-layer drift statistics;
- compact spectral dynamics;
- late-layer response instability features.

Exact prompt masks significantly improved feature quality compared to heuristic response zones.

The final linear model generalized better than heavier nonlinear ensembles on the small dataset.

---

# 5. Meta Model (`solution_meta.py`)

Track B also includes a second solution variant based on model combination.

The repository contains separate prediction and evaluation artifacts for this approach:

```text
predictions_meta.csv
results_meta.json
```

The meta solution combines multiple hidden-state feature views into a higher-level ensemble-style pipeline.

Compared to the single model, the meta approach improves robustness through feature-view combination rather than relying on one compact representation alone.

The meta-model evaluation stored in `results_meta.json` reports:

```text
avg_test_auroc = 0.80366
```

The meta approach preserves the same core philosophy:

- exact prompt-aware hidden-state features;
- lightweight linear classifiers;
- compact sklearn pipelines.

---

# 6. Important Implementation Details

The Track B pipeline modifies the official extraction loop by introducing:

```python
output_hidden_states=True
```

and exact prompt-length computation through tokenizer-based prompt encoding.

The extractor uses exact prompt lengths to build:

```python
prompt_mask
response_mask
```

All features are generated directly from transformer hidden states.

The implementation keeps the official competition structure intact:

- same evaluation pipeline;
- same dataset format;
- same prediction format;
- same sklearn training interface.

---

# 7. Experiments and Discarded Ideas

The repository structure and feature extractor suggest that multiple larger feature configurations were explored, including:

- richer cross-layer geometry;
- long-range drift combinations;
- large spectral feature sets;
- response trajectory statistics.

However, the final solution intentionally keeps the classifier simple:

```text
SelectKBest
→ PCA
→ LogisticRegression
```

The experiments indicate that lightweight linear models generalized more reliably than heavier ensemble approaches on the small dataset.

Threshold tuning was also intentionally minimized to avoid unstable validation behavior.

---

# 8. Final Notes

Track B demonstrates that a very large improvement can be achieved with only one conceptual modification:

```text
adding exact prompt_len
```

This enables precise prompt-aware hidden-state analysis while preserving the original lightweight hidden-state-only philosophy of the honest solution.

The final approach remains:

- fully hidden-state-based;
- compact at inference time;
- reproducible within the official competition framework;
- free from external verifier systems or additional language models.
