use genevo::{operator::prelude::*, population::Population, prelude::*, types::fmt::Display};
use std::io::Write;
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::crossover::RouteCrossover;
use crate::fitness::{Genome, RouteFitness, compute_individual};
use crate::mutation::{MutationType, NurseMutation};
use crate::plot;
use crate::population::refresh_population;
use crate::types::ProblemContext;

const EARLY_STOP_GENERATIONS: u64 = 1000;

/// Results from running the genetic algorithm
pub struct GaResults {
    pub best_genome: Option<Genome>,
    pub history: Vec<plot::HistoryPoint>,
    pub generations_run: u64,
}

/// Run the genetic algorithm with the given configuration and initial population
pub fn run_ga(
    cfg: &Config,
    ctx: &Arc<ProblemContext>,
    initial_population: Population<Genome>,
    run_seed: genevo::random::Seed,
    mutation_op_type: MutationType,
) -> GaResults {
    let mut best_genome: Option<Genome> = None;
    let mut best_fitness = i64::MIN;
    let mut best_generation: u64 = 0;
    let mut best_feasible = false;
    let mut best_cost = f64::INFINITY;
    let mut best_feasible_cost = f64::INFINITY;
    let mut total_generations_without_feasible_improvement = 0;
    let mut generations_without_improvement = 0;
    let generations_without_improvement_shared = Arc::new(Mutex::new(0));
    let mut history: Vec<plot::HistoryPoint> = Vec::new();
    let mut generations_run: u64 = 0;
    let mut current_population = Some(initial_population);
    let mut generation_offset: u64 = 0;
    let mut restart_counter: u64 = 0;
    let mut should_stop = false;

    if !cfg.quiet {
        println!("Running genetic algorithm...");
        println!("{:-<60}", "");
    }

    'run: while generation_offset < cfg.generations as u64 {
        let remaining_generations = cfg.generations as u64 - generation_offset;
        let phase_seed = derive_seed(&run_seed, restart_counter);

        let fitness_fn = RouteFitness::new(ctx.clone());
        let crossover_op = RouteCrossover::new(ctx.clone(), cfg.crossover_rate);
        let mutation_op = NurseMutation::new(
            cfg.mutation_rate,
            mutation_op_type.clone(),
            ctx.clone(),
            Arc::clone(&generations_without_improvement_shared),
            cfg.hill_climb_steps,
        );
        let mutation_op_ref = mutation_op.clone();

        let phase_population = current_population
            .take()
            .expect("current population should be available at phase start");

        let mut sim = simulate(
            genetic_algorithm()
                .with_evaluation(fitness_fn.clone())
                .with_selection(TournamentSelector::new(
                    cfg.selection_ratio,
                    2,
                    cfg.tournament_size,
                    1.0,
                    false,
                ))
                .with_crossover(crossover_op)
                .with_mutation(mutation_op)
                .with_reinsertion(ElitistReinserter::new(
                    fitness_fn,
                    false,
                    cfg.reinsertion_ratio,
                ))
                .with_initial_population(phase_population)
                .build(),
        )
        .until(GenerationLimit::new(remaining_generations))
        .build_with_seed(phase_seed);

        'sim: loop {
            match sim.step() {
                Ok(SimResult::Intermediate(step)) => {
                    let ep = &step.result.evaluated_population;
                    let bs = &step.result.best_solution;
                    let global_generation = generation_offset + step.iteration;
                    generations_run = global_generation;
                    let ind = compute_individual(&bs.solution.genome, ctx);

                    if ind.feasible && ind.fitness < best_feasible_cost {
                        best_feasible_cost = ind.fitness;
                        best_feasible = true;
                        total_generations_without_feasible_improvement = 0;
                        generations_without_improvement = 0;
                        *generations_without_improvement_shared.lock().unwrap() = 0;
                    } else {
                        total_generations_without_feasible_improvement += 1;
                    }

                    // Calculate actual population diversity using Hamming distance
                    let population_genomes: Vec<Genome> =
                        ep.individuals().iter().cloned().collect();
                    let current_diversity =
                        crate::mutation::calculate_population_diversity(&population_genomes);
                    crate::mutation::update_mutation_rate(
                        &mutation_op_ref,
                        current_diversity,
                        cfg.mutation_rate,
                    );

                    if bs.solution.fitness > best_fitness {
                        let prev_best_cost = best_cost;
                        best_fitness = bs.solution.fitness;
                        best_generation = global_generation;
                        generations_without_improvement = 0;
                        *generations_without_improvement_shared.lock().unwrap() = 0;
                        best_genome = Some(bs.solution.genome.clone());
                        best_cost = ind.fitness;
                        best_feasible = ind.feasible;
                        history.push(plot::HistoryPoint {
                            generation: global_generation,
                            travel: ind.total_travel,
                            penalty: ind.total_penalty,
                            feasible: ind.feasible,
                        });
                        let pct_diff = (ind.total_travel - ctx.instance.benchmark)
                            / ctx.instance.benchmark
                            * 100.0;
                        let cost_delta = if prev_best_cost.is_finite() {
                            ind.fitness - prev_best_cost
                        } else {
                            0.0
                        };

                        let current_mutation_rate = *mutation_op_ref.mutation_rate.lock().unwrap();
                        if !cfg.quiet {
                            println!(
                                "\nGen {:>4} | cost: {:>10.2} ({:+9.2}) | travel: {:>8.2} | penalty: {:>9.2} | mut: {:>6.4} | div: {:>5.3} | {}feasible{} | travel {:.2}% vs benchmark",
                                global_generation,
                                ind.fitness,
                                cost_delta,
                                ind.total_travel,
                                ind.total_penalty,
                                current_mutation_rate,
                                current_diversity,
                                if ind.feasible { "" } else { "NOT " },
                                if ind.feasible { "" } else { "  " },
                                pct_diff,
                            );
                        }
                    } else {
                        generations_without_improvement += 1;
                        *generations_without_improvement_shared.lock().unwrap() =
                            generations_without_improvement;

                        // Print visual indicator for hill climbing activation
                        if !cfg.quiet {
                            if (100..180).contains(&generations_without_improvement) {
                                print!("h"); // Say "h" for "hill" when doing hill climbing
                            } else {
                                print!(".");
                            }
                            let _ = std::io::stdout().flush();
                        }

                    }

                    if cfg.stagnation_replace_after > 0
                        && generations_without_improvement >= cfg.stagnation_replace_after as u64
                    {
                        if !cfg.quiet {
                            print!(":");
                            let _ = std::io::stdout().flush();
                        }

                        let replacement_seed = derive_seed(&run_seed, restart_counter + 1);
                        current_population = Some(refresh_population(
                            &population_genomes,
                            ctx,
                            cfg.pop_size,
                            cfg.stagnation_replace_ratio,
                            replacement_seed,
                        ));

                        generations_without_improvement = 0;
                        *generations_without_improvement_shared.lock().unwrap() = 0;
                        generation_offset = global_generation;
                        restart_counter += 1;
                        if !cfg.quiet {
                            println!(
                                "\nPopulation refresh at generation {} (replace ratio {:.2})",
                                global_generation, cfg.stagnation_replace_ratio
                            );
                        }
                        continue 'run;
                    }

                    // Early stop after enough generations without improving the best feasible solution.
                    if best_feasible
                        && total_generations_without_feasible_improvement >= EARLY_STOP_GENERATIONS
                    {
                        if !cfg.quiet {
                            println!();
                            println!("{:-<60}", "");
                            println!(
                                "Early stopping: No feasible improvement for {} generations",
                                total_generations_without_feasible_improvement
                            );
                        }
                        should_stop = true;
                        break 'sim;
                    }

                    let _ = (
                        ep.average_fitness(),
                        step.duration.fmt(),
                        step.processing_time.fmt(),
                    );
                }

                Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                    generation_offset += step.iteration;
                    generations_run = generation_offset;

                    if best_genome.is_none() {
                        best_genome = Some(step.result.best_solution.solution.genome.clone());
                    }
                    
                    // Only print phase stop reason for debugging, not as final status
                    let _ = (stop_reason, processing_time, duration);
                    break 'sim;
                }

                Err(e) => {
                    eprintln!("Simulation error: {e}");
                    should_stop = true;
                    break 'sim;
                }
            }
        }

        if should_stop {
            break 'run;
        }
    }

    // Print completion status
    if !cfg.quiet {
        if !should_stop && generation_offset >= cfg.generations as u64 {
            println!();
            println!("{:-<60}", "");
            println!("Simulation completed: reached maximum generations limit ({})", cfg.generations);
            println!("Best fitness found in generation {best_generation}");
        }
    }

    GaResults {
        best_genome,
        history,
        generations_run,
    }
}

// Seed utilities

fn derive_seed(base_seed: &genevo::random::Seed, salt: u64) -> genevo::random::Seed {
    let mut derived = *base_seed;
    let salt_bytes = salt.to_le_bytes();
    for (i, b) in salt_bytes.iter().enumerate() {
        derived[i] ^= *b;
        derived[i + 8] = derived[i + 8].wrapping_add(*b);
        derived[i + 16] ^= b.rotate_left(1);
        derived[i + 24] = derived[i + 24].wrapping_sub(*b);
    }
    derived
}
