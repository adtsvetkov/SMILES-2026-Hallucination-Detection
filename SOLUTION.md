# SMILES Hallucination Detection Solution Description

**Since application deadline was moved, I am going to improve this solution, so please check it after 17th of May.**
## Table of Contents

- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Run the official-compatible version](#run-the-official-compatible-version)
  - [Run the research version](#run-the-research-version)
  - [Difference between versions](#difference-between-versions)
- [Main Results](#main-results)
  - [Evaluation Summary](#evaluation-summary)
- [Final Modeling Pipeline](#final-modeling-pipeline)
  - [Aggregation (`aggregation.py`)](#aggregation-aggregationpy)
  - [Probe (`probe.py`)](#probe-probepy)
  - [Changes in splitting.py](#changes-in-splittingpy)
- [My Solution Logic](#my-solution-logic)
- [Why is the final official score much lower?](#why-is-the-final-official-score-much-lower)
- [Future plans](#future-plans)

## Quick Start

The solution was developed and tested with:

```text
Python 3.11
```

### Setup

Clone the repository:

```bash
git clone https://github.com/adtsvetkov/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
```

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the official-compatible version

```bash
python solution.py
```

This version modifies only:

- `aggregation.py`
- `probe.py`
- `splitting.py`

and keeps the original competition pipeline unchanged.

### Run the research version

This version additionally passes `prompt_len` into `aggregation.py`:

```python
aggregation_and_feature_extraction(
    hidden[i],
    mask[i],
    use_geometric=USE_GEOMETRIC,
    prompt_len=batch_prompt_lengths[i],
)
```

This allows more accurate separation between prompt tokens and response tokens during feature extraction.

---

### Difference between versions

| Version | Infrastructure changed | Prompt/response separation |
|---|---|---|
| `solution.py` | No | Heuristic |
| `solution_edited.py` | Yes | Exact (`prompt_len`) |

Both versions use the same general modeling pipeline:
- hidden-state drift features;
- CatBoost;
- recursive feature elimination;
- weighted ensemble.

## Main Results

### Evaluation Summary

| Version | Train AUROC | Val AUROC | Test AUROC | Test Accuracy | Test F1 |
|---|---:|---:|---:|---:|---:|
| `solution.py` | 95.84% | 71.74% | 75.03% | 74.17% | 83.97% |
| `solution_edited.py` | 97.06% | 72.62% | 77.88% | 74.02% | 83.93% |

Additional details:

| Version | Feature Dim | Folds | Extract Time |
|---|---:|---:|---:|
| `solution.py` | 89600 | 5 | 1290.3 s |
| `solution_edited.py` | 89600 | 5 | 1716.8 s |

The submitted `predictions.csv` file is available [here](https://drive.google.com/drive/folders/1XSbBXhhd0T8R8gLZM5uG9gDUgASJRe9Q?usp=drive_link).

---

## Final Modeling Pipeline

### Aggregation (`aggregation.py`)

The final feature space is based on hidden-state drift between transformer layers.

Main idea:
- extract mean hidden representations from response tokens;
- compare neighboring and long-range layers;
- measure how the representation changes across layers.

Feature groups:
- adjacent layer drift:
  - signed drift;
  - absolute drift;
  - squared drift;
  - sign drift;
  - normalized drift;
- long-range layer drift;
- token-zone drift:
  - first response third;
  - middle response third;
  - last response third.

Final feature dimensionality:

```text
89600 features
```

Main layers used:

```text
11 → 16
```

---

### Probe (`probe.py`)

Final classifier:
- CatBoost;
- weighted ensemble of 3 recursive feature elimination (RFE) selectors.

RFE selectors:
- `250 → 200 → 125 → 70`
- `250 → 200 → 125 → 60`
- `250 → 200 → 125 → 80`

Ensemble weights:

```python
(
    0.8326522912543401,
    0.13906799872596168,
    0.14228215132046573,
)
```

Final CatBoost parameters:

```python
{
    "iterations": 600,
    "depth": 4,
    "learning_rate": 0.0628238389168676,
    "l2_leaf_reg": 9.703703315819581,
    "random_strength": 6.728629794179622,
    "bagging_temperature": 1.3001972097295067,
    "border_count": 161,
    "auto_class_weights": "Balanced",
}
```

### Changes in splitting.py

The original repository used a single random train/validation/test split. :contentReference[oaicite:0]{index=0}

I replaced it with a fold-safe 5-fold stratified evaluation pipeline based on `StratifiedKFold`. :contentReference[oaicite:1]{index=1}

Main changes:
- added 5-fold cross-validation;
- preserved class balance in every fold;
- created validation splits only inside the corresponding train fold;
- ensured that every sample appears exactly once in the outer test split.

This made the evaluation more stable and reduced dependence on a single random split.

## My Solution Logic

The full sequence of experiments and code is available in `experiments.ipynb`.

I worked iteratively and optimized mainly for AUROC. Since the dataset is small (`689` samples), I did not expect a very high and stable score at the beginning. Most of the work was focused on finding where the useful signal is located in the hidden states and then reducing noise from the high-dimensional feature space.

| Step | Hypothesis / Method | Result / Conclusion |
|---:|---|---|
| 1 | Reproduce the original notebook pipeline and tune the basic probe regularization. | The baseline was relatively weak. It confirmed that a simple probe over initial hidden-state features was not enough. |
| 2 | Build richer hidden-state features using late-layer response representations. | Late layers contained much stronger signal than raw/basic aggregation. Response-side representations were more useful than prompt-side ones. |
| 3 | Compare old aggregation vs. selected rich aggregation. | Selected rich features improved some probes, but not consistently. More features alone were not enough; selection became necessary. |
| 4 | Try supervised feature selection by correlation, AUROC, and mutual information. | Feature selection helped. The best early setup used top AUROC-ranked features and reached around `0.79` test AUROC. |
| 5 | Try PCA after selecting top features. | PCA did not improve the result. It compressed useful sparse signal too aggressively. |
| 6 | Analyze text leakage and simple text statistics. | Response length, number of sentences, commas, and response ratio had signal, but it was much weaker than hidden-state drift. |
| 7 | Train text-only baselines with TF-IDF and statistical features. | TF-IDF response char n-grams reached only about `0.696` AUROC. Text-only features were not competitive. |
| 8 | Stack hidden-state, TF-IDF, and text-stat models. | Stacking did not beat the hidden-state model. The additional text signals mostly added noise. |
| 9 | Test better probes: calibrated boosting, margin models, MLPs, ExtraTrees. | Tree-based models worked best. ExtraTrees and gradient boosting were close, but CatBoost became the most promising model. |
| 10 | Tune ExtraTrees and CatBoost with Optuna. | Both improved, with CatBoost slightly stronger. This confirmed that tree-based models handle the engineered drift features well. |
| 11 | Try importance selection, stability selection, and SHAP pruning. | Stability selection helped a lot. SHAP pruning did not help and was discarded. |
| 12 | Tune stability selection. | Better stability settings improved AUROC to around `0.845`. Stable features across folds were more reliable than one-shot importance. |
| 13 | Test feature groups separately: mean, drift, response-minus-prompt, scalar groups. | `drift_only` performed best. This was the key insight: the main signal is in representation drift across layers. |
| 14 | Tune drift-only selection. | Drift-only features improved the score further, reaching around `0.862`. Removing unrelated feature groups reduced noise. |
| 15 | Build extended drift features: transformed drift, long-pair drift, token-zone drift. | `drift_extended_all` became the strongest feature space, reaching around `0.8686`. |
| 16 | Try explicit interaction features: pairwise ratios, normalized energy, curvature, monotonicity. | These did not improve performance. CatBoost was already able to capture useful interactions. |
| 17 | Try token-position dynamics: early/late response divergence, ending collapse, token-position slopes. | Did not improve over `drift_extended_all`. The useful token-position signal was mostly already covered. |
| 18 | Try pseudo-ensemble over feature families. | Did not beat the best single `drift_extended_all` pipeline. The families were too correlated. |
| 19 | Add advanced structural features: repetition, entropy, POS ratios, named entities, citation patterns. | Structural features alone had some signal, but adding them to drift features reduced AUROC. They were discarded. |
| 20 | Apply harder feature selection with recursive elimination. | RFE clearly improved performance. Reducing the feature set removed noise and stabilized the model. |
| 21 | Tune RFE trajectory. | Best RFE variants moved toward smaller final feature sets. A single RFE selector reached around `0.894`. |
| 22 | Build diversity ensemble from several RFE selectors. | Ensembling RFE `70 / 65 / 75` improved slightly. Weighted averaging was better than equal weights. |
| 23 | Tune CatBoost on the RFE ensemble. | Optuna improved the ensemble to about `0.898`. CatBoost parameters changed after feature selection, so retuning was useful. |
| 24 | Try OOF stacking over RFE models. | OOF stacking did not improve over manual weighted averaging. The dataset was too small for a stable meta-model. |
| 25 | Run adversarial validation and remove unstable features. | Removing adversarially important features hurt performance. These features were unstable but genuinely predictive. |
| 26 | Try calibration-aware checks: Platt scaling, isotonic calibration, Brier/ECE analysis. | Calibration did not improve AUROC. The uncalibrated ensemble remained best. |
| 27 | Jointly tune CatBoost, RFE choices, and ensemble weights. | Best research configuration used RFE `70 / 60 / 80`, weighted probability averaging, and tuned CatBoost. |
| 28 | Final local tuning around the best configuration. | Final research score reached `0.9037 ± 0.0088` AUROC in the notebook CV protocol. |

Main conclusions:

- The strongest signal is not in raw text, but in hidden-state drift across late transformer layers.
- Response-side drift is more useful than prompt-side representations.
- Adding more features is not enough; aggressive feature selection is necessary.
- CatBoost works better than linear models and shallow neural heads for this feature space.
- RFE and stability-based selection reduce noise strongly.
- Text-only, TF-IDF, structural features, PCA, SHAP pruning, OOF stacking, and calibration did not improve the final result.

## Why is the final official score much lower?

During the experiments I achieved a notebook result close to `0.90 AUROC`. However, later I realized that this score was overly optimistic.

The reason is that I ran a very large number of experiments on the same cross-validation folds:
- feature selection;
- RFE tuning;
- ensemble tuning;
- CatBoost tuning;
- feature engineering iterations.

Over time, the notebook pipeline became partially adapted to these exact folds. I recognize it as common problem in machine learning called cross-validation overfitting or feature-selection leakage.

Importantly, the model did not directly see the test labels. The issue is more subtle: feature selection and iterative tuning were repeatedly evaluated on the same folds, so the final setup became too specialized for them.

The official-compatible pipeline is much stricter:
- each fold is processed independently;
- feature selection happens only inside `fit()`;
- globally optimized feature subsets cannot be reused.

Because of this, the official score became lower, but also much more realistic and reproducible.

## Future plans

If I continue working on this project, the next directions would be:

- implement fully fold-safe feature selection with stronger stability constraints;
- reduce cross-validation overfitting during iterative tuning;
- test larger language models and compare hidden-state geometry across architectures;
- investigate token-level response dynamics more carefully;
- try contrastive and ranking-based objectives instead of only binary classification;
- evaluate the pipeline on larger and more diverse hallucination datasets.

I would also like to explore methods that can preserve the strong hidden-state drift signal while remaining fully reproducible inside the official evaluation pipeline.
