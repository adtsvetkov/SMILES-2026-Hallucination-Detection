============================================================
 Hallucination Detection — Evaluation Summary
 (averaged over 5 folds)
============================================================
  Checkpoint                           Accuracy      F1   AUROC
------------------------------------------------------------
  1. Majority-class baseline             70.10%  82.42%     N/A
  2. Probe (train split)                 83.85%  89.85%  97.06%
  3. Probe (val split)                   74.22%  83.90%  72.62%
  4. Probe (test split)                  74.02%  83.93%  77.88%
------------------------------------------------------------
  Feature dim  : 89600
  Total samples: 689
  Folds        : 5
  Extract time : 1716.8 s
============================================================

★  Primary metric — Test AUROC: 77.88%
