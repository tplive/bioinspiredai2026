mod config;
mod crossover;
mod fitness;
mod local_search;
mod mutation;
mod parse;
mod plot;
mod population;
mod types;

use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use config::PartialConfig;

use genevo::{
    operator::prelude::*,
    population::build_population,
    prelude::*,
    types::fmt::Display,
};

use crossover::RouteCrossover;
use fitness::{compute_individual, Genome, NurseFitness};
use mutation::{MutationType, NurseMutation};
use population::{NearestNeighbourGenomeBuilder, RandomGenomeBuilder};

// ── CLI definition ────────────────────────────────────────────────────────────

/// Configuration is resolved in this priority order (highest wins):
///
///   1. Built-in defaults
///   2. TOML configuration file (--config / -C)
///   3. Individual CLI flags
///
/// Example – run with a config file, overriding just the generation count:
///
///   project2 --config my.toml --generations 1000
#[derive(Parser, Debug)]
#[command(verbatim_doc_comment)]
struct Cli {
    /// Path to a TOML configuration file (optional; all keys are optional inside it too).
    #[arg(short = 'C', long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Problem instance JSON file.
    #[arg(short = 'f', long, value_name = "FILE")]
    file: Option<String>,

    /// Number of individuals in the population.
    #[arg(short = 'p', long)]
    pop_size: Option<usize>,

    /// Maximum number of generations.
    #[arg(short = 'g', long)]
    generations: Option<usize>,

    /// Fraction of the population forwarded to the parent pool [0.0–1.0].
    #[arg(long)]
    selection_ratio: Option<f64>,

    /// Probability of applying crossover to each parent pair [0.0–1.0].
    #[arg(long)]
    crossover_rate: Option<f64>,

    /// Probability that an individual is mutated [0.0–1.0].
    #[arg(short = 'm', long)]
    mutation_rate: Option<f64>,

    /// Intra-route mutation operator: "swap" or "insert".
    #[arg(long, value_name = "TYPE")]
    mutation_type: Option<String>,

    /// Fraction of offspring kept in the next generation [0.0–1.0].
    #[arg(long)]
    reinsertion_ratio: Option<f64>,

    /// Multiplier applied to each unit of constraint violation.
    #[arg(long)]
    penalty_factor: Option<f64>,

    /// Population initialisation method: "random" or "nn" (nearest-neighbour).
    #[arg(short = 'i', long)]
    init: Option<String>,

    /// Save a fitness-history PNG after the run (output: <instance>_fitness.png).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    plot: bool,
}

impl Cli {
    fn into_partial(self) -> PartialConfig {
        PartialConfig {
            file:              self.file,
            pop_size:          self.pop_size,
            generations:       self.generations,
            selection_ratio:   self.selection_ratio,
            crossover_rate:    self.crossover_rate,
            mutation_rate:     self.mutation_rate,
            mutation_type:     self.mutation_type,
            reinsertion_ratio: self.reinsertion_ratio,
            penalty_factor:    self.penalty_factor,
            init:              self.init,
            plot:              if self.plot { Some(true) } else { None },
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // ── Parse CLI and resolve full configuration ───────────────────────────────
    let cli = Cli::parse();

    // Layer 1 → 2: start from defaults, overlay config file if given.
    let base = match &cli.config {
        Some(path) => config::load_file(path).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }),
        None => config::Config::default(),
    };

    // Layer 3: overlay explicit CLI flags on top.
    let cfg = cli.into_partial().apply_onto(base);

    let mutation_op_type = match cfg.mutation_type.as_str() {
        "insert" => MutationType::Insert,
        _ => MutationType::Swap,
    };

    // ── Load problem instance ─────────────────────────────────────────────────
    println!("Loading instance: {}", cfg.file);
    let ctx = Arc::new(parse::load_problem(&cfg.file, cfg.penalty_factor));

    println!("Instance:         {}", ctx.instance.name);
    println!("Nurses:           {}", ctx.instance.num_nurses);
    println!("Capacity:         {}", ctx.instance.capacity);
    println!("Patients:         {}", ctx.patients.len() - 1);
    println!("Benchmark:        {:.2}", ctx.instance.benchmark);
    println!("-");
    println!("Population:       {}", cfg.pop_size);
    println!("Generations:      {}", cfg.generations);
    println!("Selection ratio:  {}", cfg.selection_ratio);
    println!("Crossover rate:   {}", cfg.crossover_rate);
    println!("Mutation rate:    {}", cfg.mutation_rate);
    println!("Mutation type:    {}", cfg.mutation_type);
    println!("Reinsertion ratio:{}", cfg.reinsertion_ratio);
    println!("Penalty factor:   {}", cfg.penalty_factor);
    println!("Init method:      {}", cfg.init);
    println!();

    // ── Build initial population ──────────────────────────────────────────────
    let initial_population: Population<Genome> = match cfg.init.as_str() {
        "nn" => build_population()
            .with_genome_builder(NearestNeighbourGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .uniform_at_random(),
        _ => build_population()
            .with_genome_builder(RandomGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .uniform_at_random(),
    };

    // ── Set up fitness function and operators ─────────────────────────────────
    let fitness_fn = NurseFitness::new(Arc::clone(&ctx));
    let crossover_op = RouteCrossover::new(Arc::clone(&ctx), cfg.crossover_rate);
    let mutation_op = NurseMutation::new(cfg.mutation_rate, mutation_op_type, Arc::clone(&ctx));

    // ── Assemble simulation ───────────────────────────────────────────────────
    let mut sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_fn.clone())
            .with_selection(MaximizeSelector::new(
                cfg.selection_ratio,
                2, // parents per group (fixed at 2 for pairwise crossover)
            ))
            .with_crossover(crossover_op)
            .with_mutation(mutation_op)
            .with_reinsertion(ElitistReinserter::new(
                fitness_fn,
                false,
                cfg.reinsertion_ratio,
            ))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(GenerationLimit::new(cfg.generations as u64))
    .build();

    // ── Run the simulation loop ───────────────────────────────────────────────
    let mut best_genome: Option<Genome> = None;
    let mut best_fitness = i64::MIN;
    let mut best_generation: u64 = 0;
    let mut history: Vec<plot::HistoryPoint> = Vec::new();

    println!("Running genetic algorithm...");
    println!("{:-<60}", "");

    'sim: loop {
        match sim.step() {
            Ok(SimResult::Intermediate(step)) => {
                let ep = &step.result.evaluated_population;
                let bs = &step.result.best_solution;

                if bs.solution.fitness > best_fitness {
                    best_fitness = bs.solution.fitness;
                    best_generation = step.iteration;
                    best_genome = Some(bs.solution.genome.clone());

                    // Decode the fitness value back to actual cost.
                    let ind = compute_individual(&bs.solution.genome, &ctx);
                    history.push(plot::HistoryPoint {
                        generation: step.iteration,
                        travel: ind.total_travel,
                        penalty: ind.total_penalty,
                        feasible: ind.feasible,
                    });
                    let pct_diff =
                        (ind.total_travel - ctx.instance.benchmark) / ctx.instance.benchmark
                            * 100.0;

                    println!(
                        "Gen {:>4} | travel: {:>8.2} | penalty: {:>7.2} | {}feasible{} | {:.2}% from benchmark",
                        step.iteration,
                        ind.total_travel,
                        ind.total_penalty,
                        if ind.feasible { "" } else { "NOT " },
                        if ind.feasible { "" } else { "  " },
                        pct_diff,
                    );
                } else {
                    print!(".");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }

                let _ = (ep.average_fitness(), step.duration.fmt(), step.processing_time.fmt());
            }

            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                println!();
                println!("{:-<60}", "");
                println!("Simulation finished: {stop_reason}");
                println!(
                    "Total time: {}  |  Processing time: {}",
                    duration.fmt(),
                    processing_time.fmt()
                );
                println!("Best fitness found in generation {best_generation}");

                if best_genome.is_none() {
                    best_genome = Some(step.result.best_solution.solution.genome.clone());
                }
                break 'sim;
            }

            Err(e) => {
                eprintln!("Simulation error: {e}");
                break 'sim;
            }
        }
    }

    // ── Print final solution ──────────────────────────────────────────────────
    if let Some(genome) = best_genome {
        println!();
        let ind = compute_individual(&genome, &ctx);
        let pct_diff =
            (ind.total_travel - ctx.instance.benchmark) / ctx.instance.benchmark * 100.0;

        println!("  Minimal travel:  {:.4}", ind.total_travel);
        println!("+ Penalty:         {:.4}", ind.total_penalty);
        println!("= Fitness (cost):  {:.4}", ind.fitness);
        println!("  Feasible?        {}", ind.feasible);
        println!(
            "  vs benchmark ({:.2}): {:.2}%",
            ctx.instance.benchmark, pct_diff
        );

        println!();
        println!("Routes:");
        for (i, route) in genome.iter().enumerate() {
            if route.is_empty() {
                println!("  Nurse {:>2}: []", i + 1);
            } else {
                println!("  Nurse {:>2}: {:?}", i + 1, route);
            }
        }

        // ── Optional plots ────────────────────────────────────────────────────────
        if cfg.plot {
            use rand::Rng;
            const KEY_CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
            let mut rng = rand::thread_rng();
            let key: String = (0..4).map(|_| KEY_CHARS[rng.gen_range(0..KEY_CHARS.len())] as char)
                .collect();

            // Route plot
            if let Err(e) = plot::save_route_plot(&genome, &ctx, &cfg, ind.fitness, &key) {
                eprintln!("Route plot error: {e}");
            }

            // Fitness history plot
            let output = format!("{}_fitness_{}.png", ctx.instance.name, key);
            print!("Saving plot → {output} ... ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            match plot::save_plot(
                &history,
                &cfg,
                &ctx.instance.name,
                ctx.instance.benchmark,
                &output,
            ) {
                Ok(()) => println!("done."),
                Err(e) => eprintln!("plot error: {e}"),
            }
        }
    }
}
