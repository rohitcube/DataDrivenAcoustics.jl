# Test Fixtures

This directory contains pre-computed ground truth data for expensive ray tracing simulations.

## Setup

Before running tests, generate the fixtures:

```bash
julia test/generate_fixtures.jl
```

This will create `pekeris_case2.dat` (~10-15 minutes to generate, but only needs to be done once).

## What's Saved

- **Training data**: 168 sensor locations and acoustic measurements from Pekeris waveguide simulation
- **Validation data**: Ground truth pressure at test location (102.5m range, 50m depth)
- **Environment parameters**: Frequency, sound speed, water depth, source location

## Benefits

- **Fast tests**: Reduces Case 2 test time from ~15 minutes to ~5 minutes
- **Reproducible**: Same ground truth data every test run
- **CI-friendly**: Generate once, commit to repo for consistent testing
