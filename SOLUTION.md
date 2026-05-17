# Hallucination Detection Solution

## Repository setup

Current `main` branch contains the final honest Track A solution.

### Run instructions

```bash
git clone https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection
cd SMILES-2026-Hallucination-Detection

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python solution.py
```

The solution uses:
- official evaluation pipeline;
- honest stratified K-Fold evaluation;
- no label leakage;
- no modifications to the official scoring logic.

---

# Final selected models

| Track | Model | Train AUROC | Val AUROC | Test AUROC | Test Accuracy | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| A | `A__advanced_all__top1250__pca256` | 0.9317 | 0.7327 | **0.7682** | 0.7678 | 0.8432 |
| B | `B__prompt_len_features_all__top312__pca32` | 0.8464 | 0.7506 | **0.7993** | 0.7620 | 0.8394 |
| B | `B_prompt_len_prob_meta_logreg` | ~0.9367 | ~0.7629 | **~0.8050** | ~0.7010 | ~0.8242 |
| C | `baseline_C_rank + seed_shap_catboost blend` | not fixed | not fixed | **0.8137** | 0.7446 | 0.8075 |

## What do tracks mean

- Track A is the honest setting: it keeps the original pipeline unchanged and uses only the hidden states that are already available from the official solution flow. No prompt length, attentions, logits, or extra model outputs are used.
- Track B is a less strict setting: we add an extra dataset read inside `aggregation.py` to reconstruct `prompt_len` and build exact prompt/response masks. This gives stronger prompt-aware features, but it is not as clean as Track A because it relies on additional access to the original dataset inside aggregation.
- Track C is the least strict setting: it uses the Track B `prompt_len` reconstruction and additionally modifies model inference to return attentions. This enables attention-grounding and prompt-response attention features, but it moves furthest away from the original official pipeline.

---
# Repository branches

## First iteration

Initial hidden-state baseline experiments:

[`first_iter_12th_may`](https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection/tree/first_iter_12th_may)

Contains:
- first geometric hidden-state features;
- early linear probes;
- initial honest evaluation experiments.

More details you can find in `solution.md` inside this branch.

---

## Track B branch

Track B single + meta solutions:

[`second_iter_track_B`](https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection/tree/second_iter_track_B)

Contains:
- prompt_len-aware infrastructure;
- Track B single model;
- Track B meta model;
- prompt reconstruction logic.

More details you can find in `solution.md` inside this branch.

---

## Track C branch

Track C attention-based experiments:

[`second_iter_track_C`](https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection/tree/second_iter_track_C)

Contains:
- attention-aware features;
- grounding decay infrastructure;
- attention ensembles;
- final Track C solution.

More details you can find in `solution.md` inside this branch.

---

# Modified components

The following official pipeline files were modified (for track A and C):

- `aggregation.py`
- `probe.py`
- `splitting.py`

For Track B, alternative versions were created:
- `aggregation_single.py`
- `probe_single.py`
- `solution_single.py`
- `aggregation_meta.py`
- `probe_meta.py`
- `solution_meta.py`

---

# splitting.py modifications

The original splitting strategy was replaced with honest stratified K-Fold evaluation.

Key changes:
- stratified folds preserve label distribution;
- train / validation / test are fully separated;
- all experiments use the same deterministic folds;
- no feature extraction leakage across folds;
- hyperparameter tuning is performed only on validation splits.

This was critical because many early experiments showed strong overfitting when evaluation was not strictly isolated.

---

# Track A

## Final approach

Track A uses only hidden-state-based features without:
- prompt_len;
- attentions;
- logits;
- hooks.

### Feature extraction

The final feature space includes:
- exact response pooling;
- compact cross-layer geometry;
- SGI-style geometric statistics;
- temporal hidden-state dynamics;
- centroid similarity features;
- cross-layer update norms;
- compact spectral statistics.

Final feature count:

```text
21644
```

### Architecture

```text
SelectKBest(k=1250)
→ PCA(256)
→ LogisticRegression(C=0.003)
→ threshold=0.5
```

### Why this worked

The largest improvements came from:
- cross-layer geometric drift features;
- late-layer response dynamics;
- compact hidden-state geometry.

Simple linear models generalized better than heavy tree ensembles on the small dataset.

---

# Track B single model

## Final approach

Track B introduces prompt-aware segmentation using reconstructed `prompt_len`.

The official pipeline does not expose prompt boundaries directly, so prompt length was reconstructed inside aggregation by rereading:
- `data/dataset.csv`
- `data/test.csv`

This allowed building:
- exact prompt masks;
- exact response masks;
- prompt/response interaction features.

### Final feature count

```text
24912
```

### Architecture

```text
SelectKBest(k=312)
→ PCA(32)
→ LogisticRegression(C=0.003)
→ threshold=0.5
```

### Main improvements

Most metric gains came from:
- prompt-aware response geometry;
- response-vs-prompt centroid drift;
- response temporal dynamics;
- prompt-conditioned hidden-state statistics.

This significantly improved AUROC over Track A.

---

# Track B meta model

## Final approach

The meta-model combines several independently trained feature spaces.

### Base models

```text
A__drift_squared
→ SelectKBest(560)
→ PCA(64)
→ LogisticRegression

B__advanced_prompt_len_max_mean
→ SelectKBest(1250)
→ PCA(128)
→ LogisticRegression

B__extra_smart_prompt_len_all
→ SelectKBest(312)
→ PCA(64)
→ LogisticRegression
```

Their probabilities are concatenated and passed into:

```text
Meta LogisticRegression(C=0.1)
```

### Why this worked

Different prompt-aware feature spaces captured complementary geometric patterns.

The meta-logistic regression successfully learned:
- confidence calibration;
- disagreement patterns between base models;
- complementary prompt-response geometry.

This produced the best Track B AUROC.

---

# Track C

## Final approach

Track C extends Track B with:
- attention-aware infrastructure;
- grounding decay dynamics;
- prompt-response attention persistence;
- retrieval-style grounding features;
- attention collapse features.

The final Track C solution used a multi-stage ensemble architecture.

### Base ensemble

The first component was a rank-fusion meta ensemble:
- prompt-aware hidden-state views;
- attention-aware views;
- grounding infrastructure views.

### Secondary ensemble

A second ensemble component used:
- SHAP-selected meta-features;
- CatBoost meta-learning;
- multiple random seeds;
- probability aggregation.

### Final blending

Final probabilities were blended as:

```text
0.7 * baseline_C_rank
+ 0.3 * seed_shap_catboost
```
---

# Experiments and failed attempts

## Heavy tree-based models

Tried:
- CatBoost;
- RandomForest;
- ExtraTrees;
- HistGradientBoosting;
- MLP.

Result:
- severe overfitting;
- unstable fold performance;
- worse generalization than linear models.

---

## Threshold tuning

Tried:
- validation-based threshold optimization for F1.

Result:
- sometimes improved validation F1;
- frequently reduced test Accuracy/F1;
- did not affect AUROC.

Final solutions mostly used:

```text
threshold = 0.5
```

---

## Large ensemble blending

Tried:
- soft voting;
- rank blending;
- geometric probability blending;
- confidence-based ensembling.

Result:
- small AUROC gains;
- unstable threshold metrics;
- meta-logistic regression generalized more reliably.

---

## Attention-heavy architectures

Large attention feature sets were explored extensively.

Result:
- some gains on validation;
- significantly higher extraction cost;
- higher overfitting risk.

Only the most stable attention-based components were retained in Track C.

---

# Main conclusions

The strongest improvements came from:
1. honest evaluation setup;
2. prompt-aware segmentation;
3. compact geometric hidden-state features;
4. lightweight linear models with strong regularization;
5. probability-level meta ensembling.

The project showed that carefully engineered hidden-state geometry can provide strong hallucination detection quality even without logits or external supervision.
