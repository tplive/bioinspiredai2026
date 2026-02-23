mod crossover;
mod fitness;
mod mutation;
mod parse;
mod population;
mod types;

use std::sync::Arc;

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

// ── Default hyper-parameters (all overridable via CLI) ───────────────────────

const DEFAULT_FILE: &str = "train/train_0.json";
const DEFAULT_POP_SIZE: usize = 100;
const DEFAULT_GENERATIONS: usize = 500;
const DEFAULT_SELECTION_RATIO: f64 = 0.8;
const DEFAULT_NUM_PARENTS: usize = 2; // individuals per parent group
const DEFAULT_CROSSOVER_RATE: f64 = 0.85;
const DEFAULT_MUTATION_RATE: f64 = 0.1;
const DEFAULT_REINSERTION_RATIO: f64 = 0.85;
const DEFAULT_PENALTY_FACTOR: f64 = 10.0;

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // ── Parse command-line arguments ──────────────────────────────────────────
    // Usage: project2 [file] [pop_size] [generations] [mutation_rate] [init]
    // init: "random" (default) | "nn" (nearest-neighbour)
    let args: Vec<String> = std::env::args().collect();
    let file = args.get(1).map(|s| s.as_str()).unwrap_or(DEFAULT_FILE);
    let pop_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_POP_SIZE);
    let generations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GENERATIONS);
    let mutation_rate: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MUTATION_RATE);
    let init_method = args.get(5).map(|s| s.as_str()).unwrap_or("random");

    // ── Load problem instance ─────────────────────────────────────────────────
    println!("Loading instance: {file}");
    let ctx = Arc::new(parse::load_problem(file, DEFAULT_PENALTY_FACTOR));

    println!("Instance:       {}", ctx.instance.name);
    println!("Nurses:         {}", ctx.instance.num_nurses);
    println!("Capacity:       {}", ctx.instance.capacity);
    println!("Patients:       {}", ctx.patients.len() - 1);
    println!("Benchmark:      {:.2}", ctx.instance.benchmark);
    println!("Population:     {pop_size}");
    println!("Generations:    {generations}");
    println!("Mutation rate:  {mutation_rate}");
    println!("Init method:    {init_method}");
    println!();

    // ── Build initial population ──────────────────────────────────────────────
    let initial_population: Population<Genome> = match init_method {
        "nn" => build_population()
            .with_genome_builder(NearestNeighbourGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(pop_size)
            .uniform_at_random(),
        _ => build_population()
            .with_genome_builder(RandomGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(pop_size)
            .uniform_at_random(),
    };

    // ── Set up fitness function and operators ─────────────────────────────────
    let fitness_fn = NurseFitness::new(Arc::clone(&ctx));
    let crossover_op = RouteCrossover::new(Arc::clone(&ctx), DEFAULT_CROSSOVER_RATE);
    let mutation_op = NurseMutation::new(mutation_rate, MutationType::Swap);

    // ── Assemble simulation ───────────────────────────────────────────────────
    let mut sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_fn.clone())
            .with_selection(MaximizeSelector::new(
                DEFAULT_SELECTION_RATIO,
                DEFAULT_NUM_PARENTS,
            ))
            .with_crossover(crossover_op)
            .with_mutation(mutation_op)
            .with_reinsertion(ElitistReinserter::new(
                fitness_fn,
                false,
                DEFAULT_REINSERTION_RATIO,
            ))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(GenerationLimit::new(generations as u64))
    .build();

    // ── Run the simulation loop ───────────────────────────────────────────────
    let mut best_genome: Option<Genome> = None;
    let mut best_fitness = i64::MIN;

    println!("Running genetic algorithm...");
    println!("{:-<60}", "");

    'sim: loop {
        match sim.step() {
            Ok(SimResult::Intermediate(step)) => {
                let ep = &step.result.evaluated_population;
                let bs = &step.result.best_solution;

                if bs.solution.fitness > best_fitness {
                    best_fitness = bs.solution.fitness;
                    best_genome = Some(bs.solution.genome.clone());

                    // Decode the fitness value back to actual cost.
                    let ind = compute_individual(&bs.solution.genome, &ctx);
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
                println!(
                    "Best fitness found in generation {}",
                    step.result.best_solution.generation
                );

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
    }
}
