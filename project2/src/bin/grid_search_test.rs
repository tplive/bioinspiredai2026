// Grid Search Test - Small Grid
// Quick test with minimal configuration for verification

use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;
use serde::Serialize;

extern crate project2;
use project2::config::Config;
use project2::parse;
use project2::ga;
use project2::population::{
    RandomGenomeBuilder, NearestNeighbourGenomeBuilder, ClarkeWrightGenomeBuilder
};
use project2::mutation::MutationType;
use genevo::prelude::*;

#[derive(Debug, Clone, Serialize)]
struct HyperParams {
    pop_size: usize,
    mutation_rate: f64,
    init: String,
}

impl HyperParams {
    fn apply_to_config(&self, mut cfg: Config) -> Config {
        cfg.pop_size = self.pop_size;
        cfg.mutation_rate = self.mutation_rate;
        cfg.init = self.init.clone();
        cfg
    }

    fn to_string_short(&self) -> String {
        format!(
            "pop={} mut={:.2} init={}",
            self.pop_size, self.mutation_rate, self.init,
        )
    }
}

#[derive(Debug, Clone)]
struct EvaluationResult {
    params: HyperParams,
    problem_file: String,
    best_cost: f64,
    is_feasible: bool,
    duration_secs: f64,
}

fn evaluate_config(
    params: &HyperParams,
    problem_file: &str,
    base_config: &Config,
    progress_counter: &Arc<Mutex<usize>>,
    total_configs: usize,
) -> EvaluationResult {
    let start = Instant::now();

    let mut cfg = params.apply_to_config(base_config.clone());
    cfg.file = problem_file.to_string();
    cfg.plot = false;
    cfg.quiet = true; // Suppress generation output
    cfg.generations = 200; // Very short for testing

    let ctx = Arc::new(parse::load_problem(&cfg.file, cfg.penalty_factor));
    let run_seed = genevo::random::random_seed();

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
        _ => {
            let builder = RandomGenomeBuilder::new(ctx.clone());
            build_population()
                .with_genome_builder(builder)
                .of_size(cfg.pop_size)
                .using_seed(run_seed)
        }
    };

    let mutation_op_type = MutationType::Swap;
    let results = ga::run_ga(&cfg, &ctx, initial_population, run_seed, mutation_op_type);

    let best_genome = results.best_genome.expect("No best genome found");
    let best_ind = project2::fitness::compute_individual(&best_genome, &ctx);

    let duration = start.elapsed();

    {
        let mut counter = progress_counter.lock().unwrap();
        *counter += 1;
        let progress = *counter;
        println!(
            "[{:>2}/{:<2}] {} - {} | Cost: {:.2} | Feasible: {} | {:.1}s",
            progress, total_configs, problem_file, params.to_string_short(),
            best_ind.fitness, best_ind.feasible, duration.as_secs_f64(),
        );
    }

    EvaluationResult {
        params: params.clone(),
        problem_file: problem_file.to_string(),
        best_cost: best_ind.fitness,
        is_feasible: best_ind.feasible,
        duration_secs: duration.as_secs_f64(),
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║   Grid Search Test - Minimal Configuration        ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!();

    // Small test: just 2 problems
    let problem_files = vec!["train/train_0.json", "train/train_1.json"];

    // Minimal grid: 2×2×2 = 8 configurations
    let mut param_grid = Vec::new();
    for &pop_size in &[50, 100] {
        for &mutation_rate in &[0.08, 0.12] {
            for &init in &["random", "cw"] {
                param_grid.push(HyperParams {
                    pop_size,
                    mutation_rate,
                    init: init.to_string(),
                });
            }
        }
    }

    let base_config = Config::default();
    let total_configs = problem_files.len() * param_grid.len();
    let progress_counter = Arc::new(Mutex::new(0));
    let overall_start = Instant::now();

    println!("Test Configuration:");
    println!("  Problems: {}", problem_files.len());
    println!("  Parameter combinations: {}", param_grid.len());
    println!("  Total evaluations: {}", total_configs);
    println!("  Generations per run: 200");
    println!();

    let mut evaluations = Vec::new();
    for problem_file in &problem_files {
        for params in &param_grid {
            evaluations.push(((*problem_file).to_string(), params.clone()));
        }
    }

    println!("Running...");
    println!();

    let results: Vec<EvaluationResult> = evaluations
        .par_iter()
        .map(|(problem_file, params)| {
            evaluate_config(params, problem_file, &base_config, &progress_counter, total_configs)
        })
        .collect();

    let overall_duration = overall_start.elapsed();

    println!();
    println!("════════════════════════════════════════════════════");
    println!("               TEST COMPLETE                        ");
    println!("════════════════════════════════════════════════════");
    println!();
    println!("Total time: {:.1}s", overall_duration.as_secs_f64());
    println!("Average: {:.1}s per config", overall_duration.as_secs_f64() / total_configs as f64);
    println!();

    for problem_file in &problem_files {
        println!("Problem: {}", problem_file);
        let problem_results: Vec<_> = results
            .iter()
            .filter(|r| r.problem_file == *problem_file)
            .collect();

        let best = problem_results
            .iter()
            .filter(|r| r.is_feasible)
            .min_by(|a, b| a.best_cost.partial_cmp(&b.best_cost).unwrap());

        if let Some(best) = best {
            println!("  ✓ Best: {:.2} - {}", best.best_cost, best.params.to_string_short());
        } else {
            println!("  ✗ No feasible solution");
        }
    }

    println!();
    println!("Test successful! ✓");
}
