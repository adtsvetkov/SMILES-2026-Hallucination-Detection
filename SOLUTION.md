## Quick Start

### Official-compatible version

Run:

```bash
python solution.py
```

This version modifies only:

- `aggregation.py`
- `probe.py`
- `splitting.py`

and keeps the original competition pipeline unchanged.

---

### Research version

Run:

```bash
python solution_edited.py
```

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

Link to 

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

## Changes in splitting.py

The original repository used a single random train/validation/test split. :contentReference[oaicite:0]{index=0}

I replaced it with a fold-safe 5-fold stratified evaluation pipeline based on `StratifiedKFold`. :contentReference[oaicite:1]{index=1}

Main changes:
- added 5-fold cross-validation;
- preserved class balance in every fold;
- created validation splits only inside the corresponding train fold;
- ensured that every sample appears exactly once in the outer test split.

This made the evaluation more stable and reduced dependence on a single random split.

## 
