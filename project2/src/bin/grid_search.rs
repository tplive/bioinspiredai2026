// Grid Search for Hyperparameter Tuning
//
// Runs parallelized cross-validation across multiple problem instances
// to find optimal hyperparameter configurations.

use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::collections::HashMap;
use rayon::prelude::*;
use serde::Serialize;

// Import from the main crate
extern crate project2;
use project2::config::Config;
use project2::parse;
use project2::ga;
use project2::population::{
    RandomGenomeBuilder, NearestNeighbourGenomeBuilder, 
    ClarkeWrightGenomeBuilder, KMeansGenomeBuilder
};
use project2::mutation::MutationType;
use genevo::prelude::*;

#[derive(Debug, Clone, Serialize)]
struct HyperParams {
    pop_size: usize,
    selection_ratio: f64,
    crossover_rate: f64,
    mutation_rate: f64,
    mutation_type: String,
    tournament_size: usize,
    reinsertion_ratio: f64,
    init: String,
}

impl HyperParams {
    fn apply_to_config(&self, mut cfg: Config) -> Config {
        cfg.pop_size = self.pop_size;
        cfg.selection_ratio = self.selection_ratio;
        cfg.crossover_rate = self.crossover_rate;
        cfg.mutation_rate = self.mutation_rate;
        cfg.mutation_type = self.mutation_type.clone();
        cfg.tournament_size = self.tournament_size;
        cfg.reinsertion_ratio = self.reinsertion_ratio;
        cfg.init = self.init.clone();
        cfg
    }

    fn to_string_short(&self) -> String {
        format!(
            "pop={} sel={:.2} cx={:.2} mut={:.2}/{} tour={} rein={:.2} init={}",
            self.pop_size,
            self.selection_ratio,
            self.crossover_rate,
            self.mutation_rate,
            self.mutation_type,
            self.tournament_size,
            self.reinsertion_ratio,
            self.init,
        )
    }
}

#[derive(Debug, Clone)]
struct EvaluationResult {
    params: HyperParams,
    problem_file: String,
    best_feasible_cost: f64,
    best_cost: f64,
    is_feasible: bool,
    duration_secs: f64,
}

fn generate_param_grid() -> Vec<HyperParams> {
    let mut grid = Vec::new();

    // Define ranges for each hyperparameter
    let pop_sizes = vec![100, 300];
    let selection_ratios = vec![0.25, 0.85];
    let crossover_rates = vec![0.2, 0.9];
    let mutation_rates = vec![0.08, 0.8];
    let mutation_types = vec!["swap", "insert"];
    let tournament_sizes = vec![2, 5];
    let reinsertion_ratios = vec![0.2, 0.9];
    let inits = vec!["nn", "kmeans", "cw"];

    // Generate all combinations
    for &pop_size in &pop_sizes {
        for &selection_ratio in &selection_ratios {
            for &crossover_rate in &crossover_rates {
                for &mutation_rate in &mutation_rates {
                    for &mutation_type in &mutation_types {
                        for &tournament_size in &tournament_sizes {
                            for &reinsertion_ratio in &reinsertion_ratios {
                                for &init in &inits {
                                    grid.push(HyperParams {
                                        pop_size,
                                        selection_ratio,
                                        crossover_rate,
                                        mutation_rate,
                                        mutation_type: mutation_type.to_string(),
                                        tournament_size,
                                        reinsertion_ratio,
                                        init: init.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    grid
}

fn print_hyperparameter_ranges(param_grid: &[HyperParams]) {
    use std::collections::BTreeSet;
    
    // Extract unique values for each hyperparameter
    let pop_sizes: BTreeSet<_> = param_grid.iter().map(|p| p.pop_size).collect();
    let selection_ratios: BTreeSet<_> = param_grid.iter().map(|p| {
        (p.selection_ratio * 100.0).round() as i32
    }).collect();
    let crossover_rates: BTreeSet<_> = param_grid.iter().map(|p| {
        (p.crossover_rate * 100.0).round() as i32
    }).collect();
    let mutation_rates: BTreeSet<_> = param_grid.iter().map(|p| {
        (p.mutation_rate * 100.0).round() as i32
    }).collect();
    let mutation_types: BTreeSet<_> = param_grid.iter().map(|p| p.mutation_type.as_str()).collect();
    let tournament_sizes: BTreeSet<_> = param_grid.iter().map(|p| p.tournament_size).collect();
    let reinsertion_ratios: BTreeSet<_> = param_grid.iter().map(|p| {
        (p.reinsertion_ratio * 100.0).round() as i32
    }).collect();
    let inits: BTreeSet<_> = param_grid.iter().map(|p| p.init.as_str()).collect();
    
    // Format as strings
    let format_int_set = |set: &BTreeSet<usize>| -> String {
        let vals: Vec<_> = set.iter().map(|v| v.to_string()).collect();
        format!("[{}]", vals.join(", "))
    };
    
    let format_float_set = |set: &BTreeSet<i32>| -> String {
        let vals: Vec<_> = set.iter().map(|v| format!("{:.2}", *v as f64 / 100.0)).collect();
        format!("[{}]", vals.join(", "))
    };
    
    let format_str_set = |set: &BTreeSet<&str>| -> String {
        let vals: Vec<_> = set.iter().copied().collect();
        format!("[{}]", vals.join(", "))
    };
    
    println!("Hyperparameter Grid:");
    println!("  pop_size:          {}", format_int_set(&pop_sizes));
    println!("  selection_ratio:   {}", format_float_set(&selection_ratios));
    println!("  crossover_rate:    {}", format_float_set(&crossover_rates));
    println!("  mutation_rate:     {}", format_float_set(&mutation_rates));
    println!("  mutation_type:     {}", format_str_set(&mutation_types));
    println!("  tournament_size:   {}", format_int_set(&tournament_sizes));
    println!("  reinsertion_ratio: {}", format_float_set(&reinsertion_ratios));
    println!("  init:              {}", format_str_set(&inits));
}

fn evaluate_config(
    params: &HyperParams,
    problem_file: &str,
    base_config: &Config,
    progress_counter: &Arc<Mutex<usize>>,
    total_configs: usize,
) -> EvaluationResult {
    let start = Instant::now();

    // Apply hyperparameters to base config
    let mut cfg = params.apply_to_config(base_config.clone());
    cfg.file = problem_file.to_string();
    cfg.plot = false; // Disable plotting for grid search
    cfg.quiet = true; // Suppress generation output
    cfg.generations = 5000; // Shorter runs for grid search

    // Load problem
    let ctx = Arc::new(parse::load_problem(&cfg.file, cfg.penalty_factor));

    // Generate seed
    let run_seed = genevo::random::random_seed();

    // Build initial population based on init strategy
    let initial_population = match cfg.init.as_str() {
        "nn" => {
            let builder = NearestNeighbourGenomeBuilder::new(ctx.clone());
            build_population()
                .with_genome_builder(builder)
                .of_size(cfg.pop_size)
                .using_seed(run_seed)
        }
        "cw" => {
            let builder = ClarkeWrightGenomeBuilder::new(ctx.clone());
            build_population()
                .with_genome_builder(builder)
                .of_size(cfg.pop_size)
                .using_seed(run_seed)
        }
        "kmeans" => {
            let builder = KMeansGenomeBuilder::new(ctx.clone());
            build_population()
                .with_genome_builder(builder)
                .of_size(cfg.pop_size)
                .using_seed(run_seed)
        }
        _ => {
            let builder = RandomGenomeBuilder::new(ctx.clone());
            build_population()
                .with_genome_builder(builder)
                .of_size(cfg.pop_size)
                .using_seed(run_seed)
        }
    };

    // Determine mutation type
    let mutation_op_type = match cfg.mutation_type.as_str() {
        "insert" => MutationType::Insert,
        _ => MutationType::Swap,
    };

    // Run GA (now with quiet mode enabled)
    let results = ga::run_ga(
        &cfg,
        &ctx,
        initial_population,
        run_seed,
        mutation_op_type,
    );

    // Extract best result
    let best_genome = results.best_genome.expect("No best genome found");
    let best_ind = project2::fitness::compute_individual(&best_genome, &ctx);

    let duration = start.elapsed();

    // Update progress
    {
        let mut counter = progress_counter.lock().unwrap();
        *counter += 1;
        let progress = *counter;
        println!(
            "[{:>4}/{:<4}] Completed: {} - {} | Cost: {:.2} (feasible: {}) | {:.1}s",
            progress,
            total_configs,
            problem_file,
            params.to_string_short(),
            best_ind.fitness,
            best_ind.feasible,
            duration.as_secs_f64(),
        );
    }

    EvaluationResult {
        params: params.clone(),
        problem_file: problem_file.to_string(),
        best_feasible_cost: if best_ind.feasible {
            best_ind.fitness
        } else {
            f64::INFINITY
        },
        best_cost: best_ind.fitness,
        is_feasible: best_ind.feasible,
        duration_secs: duration.as_secs_f64(),
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     Genetic Algorithm - Grid Search Cross-Validation             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Problem files to evaluate
    let problem_files = vec![
        "train/train_5.json",
    ];

    // Generate parameter grid
    let param_grid = generate_param_grid();

    // Base configuration
    let base_config = Config::default();

    println!("Grid Search Configuration:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("  Problem files:        {}", problem_files.len());
    println!("  Parameter combinations: {}", param_grid.len());
    println!("  Total evaluations:     {}", problem_files.len() * param_grid.len());
    println!("  Generations per run:   500");
    println!("  Parallel execution:    Enabled");
    println!();
    print_hyperparameter_ranges(&param_grid);
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let total_configs = problem_files.len() * param_grid.len();
    let progress_counter = Arc::new(Mutex::new(0));
    let overall_start = Instant::now();

    println!("Starting grid search...");
    println!();

    // Create all combinations of (problem, params)
    let mut evaluations = Vec::new();
    for problem_file in &problem_files {
        for params in &param_grid {
            evaluations.push(((*problem_file).to_string(), params.clone()));
        }
    }

    // Run evaluations in parallel
    let results: Vec<EvaluationResult> = evaluations
        .par_iter()
        .map(|(problem_file, params)| {
            evaluate_config(
                params,
                problem_file,
                &base_config,
                &progress_counter,
                total_configs,
            )
        })
        .collect();

    let overall_duration = overall_start.elapsed();

    println!();
    println!("═════════════════════════════════════════════════════════════════");
    println!("                     GRID SEARCH COMPLETE                         ");
    println!("═════════════════════════════════════════════════════════════════");
    println!();
    println!("Total time: {:.1} minutes", overall_duration.as_secs_f64() / 60.0);
    println!("Average time per config: {:.1}s", overall_duration.as_secs_f64() / total_configs as f64);
    println!();

    // Analyze results per problem
    println!("═════════════════════════════════════════════════════════════════");
    println!("                  BEST CONFIGURATIONS PER PROBLEM                 ");
    println!("═════════════════════════════════════════════════════════════════");
    println!();

    for problem_file in &problem_files {
        println!("Problem: {}", problem_file);
        println!("─────────────────────────────────────────────────────────────────");

        // Filter results for this problem
        let problem_results: Vec<_> = results
            .iter()
            .filter(|r| r.problem_file == *problem_file)
            .collect();

        // Find best feasible solution
        let best_feasible = problem_results
            .iter()
            .filter(|r| r.is_feasible)
            .min_by(|a, b| a.best_feasible_cost.partial_cmp(&b.best_feasible_cost).unwrap());

        // Find best overall (including infeasible)
        let best_overall = problem_results
            .iter()
            .min_by(|a, b| a.best_cost.partial_cmp(&b.best_cost).unwrap())
            .unwrap();

        if let Some(best) = best_feasible {
            println!("  ✓ Best Feasible Solution:");
            println!("    Cost:     {:.2}", best.best_feasible_cost);
            println!("    Config:   {}", best.params.to_string_short());
            println!("    Duration: {:.1}s", best.duration_secs);
        } else {
            println!("  ✗ No feasible solution found");
            println!("  ⚠ Best infeasible:");
            println!("    Cost:     {:.2}", best_overall.best_cost);
            println!("    Config:   {}", best_overall.params.to_string_short());
        }

        // Statistics
        let feasible_count = problem_results.iter().filter(|r| r.is_feasible).count();
        println!("  Statistics:");
        println!("    Feasible solutions: {}/{}", feasible_count, problem_results.len());
        println!("    Success rate:       {:.1}%", 
                 100.0 * feasible_count as f64 / problem_results.len() as f64);

        println!();
    }

    // Global analysis
    println!("═════════════════════════════════════════════════════════════════");
    println!("                      GLOBAL ANALYSIS                             ");
    println!("═════════════════════════════════════════════════════════════════");
    println!();

    // Count feasible solutions per hyperparameter value
    let mut init_scores: HashMap<String, (usize, usize)> = HashMap::new();
    let mut mutation_scores: HashMap<String, (usize, usize)> = HashMap::new();

    for result in &results {
        let init_entry = init_scores.entry(result.params.init.clone()).or_insert((0, 0));
        init_entry.1 += 1; // total
        if result.is_feasible {
            init_entry.0 += 1; // feasible
        }

        let mut_entry = mutation_scores.entry(result.params.mutation_type.clone()).or_insert((0, 0));
        mut_entry.1 += 1;
        if result.is_feasible {
            mut_entry.0 += 1;
        }
    }

    println!("Initialization Strategy Performance:");
    for (init, (feasible, total)) in &init_scores {
        let rate = 100.0 * *feasible as f64 / *total as f64;
        println!("  {:<8} {}/{} feasible ({:.1}%)", init, feasible, total, rate);
    }

    println!();
    println!("Mutation Type Performance:");
    for (mut_type, (feasible, total)) in &mutation_scores {
        let rate = 100.0 * *feasible as f64 / *total as f64;
        println!("  {:<8} {}/{} feasible ({:.1}%)", mut_type, feasible, total, rate);
    }

    println!();
    println!("═════════════════════════════════════════════════════════════════");
    println!("Grid search completed successfully!");
    println!("═════════════════════════════════════════════════════════════════");
}
