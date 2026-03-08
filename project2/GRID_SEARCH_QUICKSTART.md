# Grid Search Quick Start

## Running the Grid Search

### Full Grid Search (384 configurations × 4 problems = 1,536 evaluations)
```bash
cargo run --bin grid_search --release
```

### Test Version (8 configurations × 2 problems = 16 evaluations)
```bash
cargo run --bin grid_search_test --release
```

## Save Results to File
```bash
cargo run --bin grid_search --release | tee results_$(date +%Y%m%d_%H%M%S).txt
```

## Customization

### Modify Problem Files
Edit [src/bin/grid_search.rs](src/bin/grid_search.rs) line ~245:
```rust
let problem_files = vec![
    "train/train_0.json",
    "train/train_1.json",
    "train/train_2.json",
    "train/train_3.json",
    // Add more problems here
];
```

### Adjust Hyperparameter Ranges
Edit the `generate_param_grid()` function (line ~75):
```rust
let pop_sizes = vec![100, 150];  // Expand: [50, 100, 150, 200]
let selection_ratios = vec![0.75, 0.85];
let crossover_rates = vec![0.8, 0.9];
let mutation_rates = vec![0.08, 0.12];
// etc.
```

### Change Evaluation Duration
Edit line ~131:
```rust
cfg.generations = 500;  // Increase for better results
```

## Output Interpretation

### Progress Updates
```
[42/1536] Completed: train/train_0.json - pop=100 sel=0.75 ... | Cost: 234.56 (feasible: true) | 12.3s
```
- Shows: progress, problem, configuration, cost, feasibility, duration

### Final Results
```
Problem: train/train_0.json
─────────────────────────────────────────────────────────────────
  ✓ Best Feasible Solution:
    Cost:     234.56
    Config:   pop=150 sel=0.85 cx=0.90 mut=0.08/swap tour=3 rein=0.90 init=cw
    Duration: 12.3s
  Statistics:
    Feasible solutions: 287/384
    Success rate:       74.7%
```

### Global Analysis
Shows which initialization strategies and mutation types perform best across all problems.

## Performance Tips

1. **Start Small**: Use grid_search_test first
2. **Parallel Execution**: Automatically uses all CPU cores
3. **Long Runs**: Use screen/tmux for extended runs
4. **Resource Monitoring**: Watch CPU/memory with htop
5. **Save Output**: Always redirect to file for analysis

## Example Workflow

```bash
# 1. Quick test to verify everything works
cargo run --bin grid_search_test --release

# 2. Run full grid search and save results
cargo run --bin grid_search --release | tee grid_results.txt

# 3. Extract best configurations
grep "Best Feasible Solution" -A 2 grid_results.txt
```

## Architecture

- **Parallelization**: Rayon for CPU-level parallelism
- **Progress Tracking**: Thread-safe atomic counter
- **Evaluation**: Each config runs independently
- **Aggregation**: Results grouped by problem for analysis

## Test Results

The test run demonstrated:
- ✅ Successful parallel execution
- ✅ Progress tracking working
- ✅ Best configuration identification per problem
- ✅ 16 evaluations in ~6 seconds (0.4s per config)

Full grid search (1,536 evaluations) estimated at ~10-15 minutes.
