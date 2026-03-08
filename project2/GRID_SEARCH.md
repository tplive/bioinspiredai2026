# Grid Search for Hyperparameter Tuning

## Overview

The `grid_search` binary performs parallelized cross-validation across multiple problem instances to find optimal hyperparameter configurations for the genetic algorithm.

## Features

- **Parallelized Execution**: Uses Rayon for efficient parallel evaluation of parameter combinations
- **Progress Tracking**: Real-time updates as each configuration completes
- **Comprehensive Reporting**: Detailed analysis of best parameters per problem and global trends
- **Configurable Grid**: Easy to modify hyperparameter ranges

## Usage

### Basic Run

```bash
cargo run --bin grid_search --release
```

### Custom Configuration

Edit the `generate_param_grid()` function in `src/bin/grid_search.rs` to customize:
- Problem files to test
- Hyperparameter ranges
- Number of generations per run

## Hyperparameter Grid

Current configuration tests the following ranges:

- **pop_size**: [100, 150]
- **selection_ratio**: [0.75, 0.85]
- **crossover_rate**: [0.8, 0.9]
- **mutation_rate**: [0.08, 0.12]
- **mutation_type**: [swap, insert]
- **tournament_size**: [2, 3]
- **reinsertion_ratio**: [0.8, 0.9]
- **init**: [random, nn, cw]

Total: 2×2×2×2×2×2×2×3 = **384 configurations** per problem

## Output Format

### Progress Updates
```
[  42/1536] Completed: train/train_0.json - pop=100 sel=0.75 ... | Cost: 234.56 (feasible: true) | 12.3s
```

### Final Report

1. **Best Configurations Per Problem**
   - Best feasible solution cost
   - Hyperparameter configuration
   - Success statistics

2. **Global Analysis**
   - Initialization strategy performance
   - Mutation type performance
   - Overall success rates

## Performance

- **Parallel Execution**: Utilizes all available CPU cores
- **Run Time**: ~500 generations per configuration
- **Typical Duration**: Varies based on:
  - Number of problems
  - Parameter grid size
  - Problem complexity
  - Available CPU cores

## Example Output

```
╔═══════════════════════════════════════════════════════════════════╗
║     Genetic Algorithm - Grid Search Cross-Validation             ║
╚═══════════════════════════════════════════════════════════════════╝

Grid Search Configuration:
─────────────────────────────────────────────────────────────────
  Problem files:        4
  Parameter combinations: 384
  Total evaluations:     1536
  Generations per run:   500
  Parallel execution:    Enabled

Starting grid search...

[Progress updates...]

═════════════════════════════════════════════════════════════════
                     GRID SEARCH COMPLETE                         
═════════════════════════════════════════════════════════════════

Total time: 45.2 minutes
Average time per config: 1.8s

═════════════════════════════════════════════════════════════════
                  BEST CONFIGURATIONS PER PROBLEM                 
═════════════════════════════════════════════════════════════════

Problem: train/train_0.json
─────────────────────────────────────────────────────────────────
  ✓ Best Feasible Solution:
    Cost:     234.56
    Config:   pop=150 sel=0.85 cx=0.90 mut=0.08/swap tour=3 rein=0.90 init=cw
    Duration: 12.3s
  Statistics:
    Feasible solutions: 287/384
    Success rate:       74.7%

[More problems...]

═════════════════════════════════════════════════════════════════
                      GLOBAL ANALYSIS                             
═════════════════════════════════════════════════════════════════

Initialization Strategy Performance:
  cw       456/512 feasible (89.1%)
  nn       398/512 feasible (77.7%)
  random   234/512 feasible (45.7%)

Mutation Type Performance:
  swap     602/768 feasible (78.4%)
  insert   486/768 feasible (63.3%)
```

## Customization

### Adding More Problems

Edit the `problem_files` vector in `main()`:

```rust
let problem_files = vec![
    "train/train_0.json",
    "train/train_1.json",
    "train/train_2.json",
    "train/train_3.json",
    "train/train_4.json",  // Add more
];
```

### Modifying Hyperparameter Ranges

Edit the `generate_param_grid()` function:

```rust
let pop_sizes = vec![50, 100, 150, 200];  // Expand range
let selection_ratios = vec![0.7, 0.8, 0.9];
// ... etc
```

### Adjusting Generations

Modify in `evaluate_config()`:

```rust
cfg.generations = 1000;  // Increase for more thorough evaluation
```

## Tips

1. **Start Small**: Test with fewer problems and a smaller grid first
2. **Monitor Resources**: Watch CPU and memory usage during parallel execution
3. **Longer Runs**: Increase generations for final tuning
4. **Save Results**: Redirect output to a file for later analysis
   ```bash
   cargo run --bin grid_search --release | tee grid_search_results.txt
   ```

## Architecture

The grid search tool:
1. Generates all parameter combinations
2. Creates (problem, params) pairs
3. Evaluates all pairs in parallel using Rayon
4. Tracks progress with atomic counter
5. Aggregates results and finds best configurations
6. Produces comprehensive analysis report

## Integration

The tool uses the main project2 library modules:
- `config`: Configuration management
- `parse`: Problem loading
- `ga`: Genetic algorithm execution
- `population`: Population initialization
- `mutation`: Mutation operators
- `fitness`: Fitness evaluation

All modules are accessed through the library interface defined in `src/lib.rs`.
