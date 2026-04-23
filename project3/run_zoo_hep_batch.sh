#!/bin/bash
# Batch testing script for zoo and hepatitis datasets

set -e

# Uses the zoo_hepatitis_config.json as the base configuration
# Tests all three optimizers: sga, nsga-ii, aco

echo "Running batch experiments on zoo and hepatitis datasets..."
julia --project=. src/run_batch_experiments.jl \
  zoo_hepatitis_config.json \
  "train_data/06-zoo_lr_F.h5,train_data/10-hepatitis_lr_F.h5" \
  "sga,nsga-ii,aco"

cat artifacts/batch/feature_experiment_summary.md
