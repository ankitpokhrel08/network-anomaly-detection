# Isolation Forest — Network Anomaly Detection

**Dataset:** [KDD Cup 99 — Network Anomaly Detection](https://www.kaggle.com/datasets/anushonkar/network-anamoly-detection/data)

## What

Unsupervised anomaly detection on network traffic using `IsolationForest`, benchmarked against a supervised `RandomForestClassifier`.  
Attack types grouped into: `dos`, `probe`, `r2l`, `u2r`.

## Results

| Model                                                          | F1    |
| -------------------------------------------------------------- | ----- |
| IsolationForest (default, contamination=0.02)                  | ~0.14 |
| IsolationForest (tuned, contamination anchored to actual rate) | ~0.67 |
| Random Forest (supervised baseline)                            | ~0.99 |

## Key Findings

- IsolationForest works best when anomalies are **rare (<15%)**. Dataset anomaly rate is ~46% (DoS still included), which violates that assumption. **Fixing Required**
- Hardcoding `contamination=0.02` on a ~46% anomaly dataset was the main culprit for F1 ~0.14. Anchoring it to the actual anomaly rate brought F1 up to ~0.67.
- DoS class (~45k rows) **not yet removed** — expected to push IF performance even higher since remaining anomalies would be genuinely rare.
- `RandomizedSearchCV` doesn't work with `IsolationForest` out of the box — needs a custom scorer.
- Supervised RF dominates on labelled data, as expected.

## Status

- [x] It worked
- [ ] Needs a bit more tuning (contamination, feature selection, better anomaly score threshold)
- [ ] Need to implement IsolationForest from scratch
