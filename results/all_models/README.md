# All models — full output set

The main [README](../../README.md) highlights CNN-LSTM (the featured model). This folder has the same three diagnostic figures for **all six models**, so the comparison isn't just a table of numbers:

- `oof_scatter.png` — out-of-fold predicted vs. actual (5-fold × 5-repeat CV), both targets, with R²/RMSE per model
- `feature_importance.png` — which station/variable combination each model relied on most (permutation importance for the 3 neural nets, native importance for the 3 tree models — see each model's script for which exact metric, since these are not directly comparable across libraries)
- `hit_miss.png` — the 9 held-out test events, prediction vs. the natural-variability band at that wind speed level (see main README for what HIT/MISS means)

CNN-LSTM additionally has `learning_curve.png` (training/validation loss per epoch), since it's the only model in this set with an iterative training process worth visualizing that way.
