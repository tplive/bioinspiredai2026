#!/bin/bash
# Visualize all batch artifact folders.
# Usage: ./run_batch_visualizations.sh [artifact_root] [sample_fraction]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ARTIFACT_ROOT="${1:-artifacts/batch}"
SAMPLE_FRACTION="${2:-0.20}"

if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "Artifact root does not exist: $ARTIFACT_ROOT"
  exit 1
fi

mapfile -t artifact_dirs < <(find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#artifact_dirs[@]} -eq 0 ]]; then
  echo "No artifact directories found in: $ARTIFACT_ROOT"
  exit 1
fi

echo "Visualizing ${#artifact_dirs[@]} artifact folders in $ARTIFACT_ROOT"

for dir in "${artifact_dirs[@]}"; do
  if [[ -f "$dir/penalized_fitness.csv" && -f "$dir/convergence.csv" ]]; then
    echo "- Visualizing $(basename "$dir")"
    julia --project=. scripts/visualize_artifact.jl "$dir" "$SAMPLE_FRACTION"
  else
    echo "- Skipping $(basename "$dir") (missing penalized_fitness.csv or convergence.csv)"
  fi
done

echo "Done."
