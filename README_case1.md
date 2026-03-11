# Case 1: Far‑Field (2D Plane Wave)

This document describes the Case 1 far‑field workflow for data‑driven propagation using a plane‑wave ray basis model. The goal is to reproduce the Case 1 training and evaluation flow using the updated library stack while keeping the training behavior deterministic.

## Files

- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/test/case1_env_fit.jl`  
  Entry point for the Case 1 workflow (environment setup, data loading, training, evaluation).

- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/test/case1_far_field_helpers.jl`  
  Helper functions, model definitions, and training loop.

## Data

All Case 1 data live in:

- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/data/case1_far_field/A_train.csv`  
- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/data/case1_far_field/capsule_rx_test.csv`  
- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/data/case1_far_field/capsule_TL_test.csv`  
- `/Users/rohit/Desktop/urop/DataDrivenAcoustics.jl/data/case1_far_field/ini_RBNN.bson`  

## Workflow

1. **Environment + propagation model**  
   A `UnderwaterEnvironment` is created and paired with `Bellhop` to generate grid test data.

1. **Measurement locations**  
   `zig_zag_samples` generates receiver locations along a zig‑zag path in range/depth.

1. **Training data**  
   `A_train.csv` provides training measurements.  
   The receiver locations are split into train/validation using `data_split`.

1. **Model initialization**  
   `ini_RBNN.bson` loads initial model parameters for deterministic training.

1. **Training**  
   `fit!` splits the data and calls `fit_case1_train!`, which trains using a legacy optimizer and early stopping logic.

1. **Evaluation**  
   RMSE is computed on the test grid using the fixed test data files.

## Helper functions

- `LegacyADAM`  
  Legacy optimizer used to preserve training behavior.

- `fit_case1_train!`  
  Training loop with early stopping and learning‑rate reduction.

- `fit!`  
  High‑level entry point that splits data and trains the model.

- `RayBasis`  
  Original 2D ray‑basis model used for initialization.

- `RayBasis2DCurv`  
  2D curvature formulation used for training.

- `zig_zag_samples`  
  Generates receiver locations along a zig‑zag trajectory.

- `data_split`  
  Splits receiver locations into training and validation sets.

- `generate_test_data`  
  Builds a dense receiver grid and corresponding transmission‑loss field.
