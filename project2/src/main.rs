mod config;
mod crossover;
mod fitness;
mod local_search;
mod mutation;
mod parse;
mod plot;
mod population;
mod types;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use config::PartialConfig;

use genevo::{
    operator::prelude::*,
    population::build_population,
    prelude::*,
    types::fmt::Display,
};

use crossover::RouteCrossover;
use fitness::{compute_individual, compute_detailed_route, Genome, NurseFitness};
use mutation::{MutationType, NurseMutation};
use population::{
    ClarkeWrightGenomeBuilder,
    KMeansGenomeBuilder,
    NearestNeighbourGenomeBuilder,
    RandomGenomeBuilder,
};

//  CLI definition 

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

    /// Population initialisation method: "random", "nn" (nearest-neighbour), "cw" (clarke-wright), or "kmeans".
    #[arg(short = 'i', long)]
    init: Option<String>,

    /// Selection method: "truncation" or "tournament".
    #[arg(long)]
    selection_type: Option<String>,

    /// Tournament size for tournament selection (typically 2-5).
    #[arg(long)]
    tournament_size: Option<usize>,

    /// Save route/fitness PNGs after the run (inside the per-run folder).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    plot: bool,

    /// Optional deterministic random seed as 64 hex chars (32 bytes).
    #[arg(long, value_name = "HEX32")]
    random_seed: Option<String>,

    /// Replace population after this many generations without improvement.
    #[arg(long)]
    stagnation_replace_after: Option<usize>,

    /// Fraction of population to replace on stagnation refresh [0.0–1.0].
    #[arg(long)]
    stagnation_replace_ratio: Option<f64>,

    /// Number of hill climbing steps when stagnating (100-179 gens without improvement).
    #[arg(long)]
    hill_climb_steps: Option<usize>,
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
            selection_type:    self.selection_type,
            tournament_size:   self.tournament_size,
            plot:              if self.plot { Some(true) } else { None },
            random_seed:       self.random_seed,
            stagnation_replace_after: self.stagnation_replace_after,
            stagnation_replace_ratio: self.stagnation_replace_ratio,
            hill_climb_steps: self.hill_climb_steps,
        }
    }
}

//  Entry point 

fn main() {
    //  Parse CLI and resolve full configuration 
    let cli = Cli::parse();
    let input_config_path = cli.config.clone();

    // Layer 1 → 2: start from defaults, overlay config file if given.
    let base = match &cli.config {
        Some(path) => config::load_file(path).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }),
        None => config::Config::default(),
    };

    // Layer 3: overlay explicit CLI flags on top.
    let mut cfg = cli.into_partial().apply_onto(base);

    let run_seed = match cfg.random_seed.as_deref() {
        Some(hex) => parse_seed_hex(hex).unwrap_or_else(|e| {
            eprintln!("Invalid --random-seed/config random_seed: {e}");
            std::process::exit(1);
        }),
        None => genevo::random::random_seed(),
    };
    let run_seed_hex = seed_to_hex(&run_seed);
    cfg.random_seed = Some(run_seed_hex.clone());

    let run_dir = create_run_dir(&cfg.file).unwrap_or_else(|e| {
        eprintln!("Could not create run directory: {e}");
        std::process::exit(1);
    });

    let used_config_path = run_dir.join("config_used.toml");
    if let Err(e) = write_used_config(&cfg, &used_config_path) {
        eprintln!("Could not write used config '{}': {e}", used_config_path.display());
        std::process::exit(1);
    }

    if let Some(path) = input_config_path {
        let original_cfg_copy = run_dir.join("config_input.toml");
        if let Err(e) = std::fs::copy(&path, &original_cfg_copy) {
            eprintln!(
                "Warning: could not copy input config '{}' to '{}': {e}",
                path.display(),
                original_cfg_copy.display()
            );
        }
    }

    let mutation_op_type = match cfg.mutation_type.as_str() {
        "insert" => MutationType::Insert,
        _ => MutationType::Swap,
    };

    // Load problem instance
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
    println!("Tournament size:  {}", cfg.tournament_size);
    println!("Crossover rate:   {}", cfg.crossover_rate);
    println!("Mutation rate:    {}", cfg.mutation_rate);
    println!("Mutation type:    {}", cfg.mutation_type);
    println!("Reinsertion ratio:{}", cfg.reinsertion_ratio);
    println!("Penalty factor:   {}", cfg.penalty_factor);
    println!("Init method:      {}", cfg.init);
    println!("Refresh after:    {}", cfg.stagnation_replace_after);
    println!("Refresh ratio:    {:.2}", cfg.stagnation_replace_ratio);
    println!("Random seed:      {}", run_seed_hex);
    println!("Run directory:    {}", run_dir.display());
    println!();

    //  Build initial population 
    let initial_population: Population<Genome> = match cfg.init.as_str() {
        "nn" => build_population()
            .with_genome_builder(NearestNeighbourGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        "cw" | "clarke-wright" => build_population()
            .with_genome_builder(ClarkeWrightGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        "kmeans" | "km" => build_population()
            .with_genome_builder(KMeansGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        _ => build_population()
            .with_genome_builder(RandomGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
    };

    //  Run the simulation loop 
    let mut best_genome: Option<Genome> = None;
    let mut best_fitness = i64::MIN;
    let mut best_generation: u64 = 0;
    let mut best_feasible = false;
    let mut best_cost = f64::INFINITY;
    let mut generations_without_improvement = 0u64;
    let generations_without_improvement_shared = Arc::new(Mutex::new(0u64));
    let mut history: Vec<plot::HistoryPoint> = Vec::new();
    let mut current_population = initial_population;
    let mut generation_offset: u64 = 0;
    let mut restart_counter: u64 = 0;
    const EARLY_STOP_GENERATIONS: u64 = 200;
    let mut should_stop = false;

    println!("Running genetic algorithm...");
    println!("{:-<60}", "");

    'run: while generation_offset < cfg.generations as u64 {
        let remaining_generations = cfg.generations as u64 - generation_offset;
        let phase_seed = derive_seed(&run_seed, restart_counter);

        let fitness_fn = NurseFitness::new(Arc::clone(&ctx));
        let crossover_op = RouteCrossover::new(Arc::clone(&ctx), cfg.crossover_rate);
        let mutation_op = NurseMutation::new(
            cfg.mutation_rate,
            mutation_op_type.clone(),
            Arc::clone(&ctx),
            Arc::clone(&generations_without_improvement_shared),
            cfg.hill_climb_steps,
        );
        let mutation_op_ref = mutation_op.clone();

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
                .with_initial_population(current_population.clone())
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

                // Calculate actual population diversity using Hamming distance
                let population_genomes: Vec<Genome> = ep.individuals()
                    .iter()
                    .cloned()
                    .collect();
                let current_diversity = mutation::calculate_population_diversity(&population_genomes);
                mutation::update_mutation_rate(&mutation_op_ref, current_diversity, cfg.mutation_rate);

                if bs.solution.fitness > best_fitness {
                    let prev_best_cost = best_cost;
                    best_fitness = bs.solution.fitness;
                    best_generation = global_generation;
                    generations_without_improvement = 0;
                    *generations_without_improvement_shared.lock().unwrap() = 0;
                    best_genome = Some(bs.solution.genome.clone());
                    generations_without_improvement = 0;
                    best_genome = Some(bs.solution.genome.clone());

                    // Decode the fitness value back to actual cost.
                    let ind = compute_individual(&bs.solution.genome, &ctx);
                    best_cost = ind.fitness;
                    best_feasible = ind.feasible;
                    history.push(plot::HistoryPoint {
                        generation: global_generation,
                        travel: ind.total_travel,
                        penalty: ind.total_penalty,
                        feasible: ind.feasible,
                    });
                    let pct_diff =
                        (ind.total_travel - ctx.instance.benchmark) / ctx.instance.benchmark
                            * 100.0;
                    let cost_delta = if prev_best_cost.is_finite() {
                        ind.fitness - prev_best_cost
                    } else {
                        0.0
                    };

                    let current_mutation_rate = *mutation_op_ref.mutation_rate.lock().unwrap();
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
                } else {
                    generations_without_improvement += 1;
                    *generations_without_improvement_shared.lock().unwrap() = generations_without_improvement;
                    
                    // Print visual indicator for hill climbing activation
                    if generations_without_improvement >= 100 && generations_without_improvement < 180 {
                        print!("0"); // Julia uses '0' to indicate hill climbing
                    } else {
                        print!(".");
                    }
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    
                    // Early stop after 200 generations without improvement, but only if best solution is feasible
                    if best_feasible && generations_without_improvement >= EARLY_STOP_GENERATIONS {
                        println!();
                        println!("{:-<60}", "");
                        println!("Early stopping: No improvement for {} generations (best solution is feasible)", EARLY_STOP_GENERATIONS);
                        should_stop = true;
                        break 'sim;
                    }
                }

                if cfg.stagnation_replace_after > 0
                    && generations_without_improvement >= cfg.stagnation_replace_after as u64
                {
                    print!(":");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();

                    let replacement_seed = derive_seed(&run_seed, restart_counter + 1);
                    current_population = refresh_population(
                        &population_genomes,
                        &ctx,
                        cfg.pop_size,
                        cfg.stagnation_replace_ratio,
                        replacement_seed,
                    );

                    generations_without_improvement = 0;
                    *generations_without_improvement_shared.lock().unwrap() = 0;
                    generation_offset = global_generation;
                    restart_counter += 1;
                    println!(
                        "\nPopulation refresh at generation {} (replace ratio {:.2})",
                        global_generation,
                        cfg.stagnation_replace_ratio
                    );
                    continue 'run;
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

                generation_offset += step.iteration;

                if best_genome.is_none() {
                    best_genome = Some(step.result.best_solution.solution.genome.clone());
                }
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

    //  Print final solution 
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
        
        // Format route output
        let mut route_output = String::new();
        route_output.push_str(&format!("Nurse capacity: {:.0}\n", ctx.instance.capacity));
        route_output.push_str(&format!("Depot return time: {:.2}\n", ctx.instance.depot_return_time));
        route_output.push_str(&format!("{:-<68}\n", ""));

        for (i, route) in genome.iter().enumerate() {
            if route.is_empty() {
                route_output.push_str(&format!("Nurse {:>2}    0.00     0.00  d(0) -> d(0.00)\n", i + 1));
            } else {
                let detailed = compute_detailed_route(route, &ctx);
                
                // Build route string: d(0) -> patient_i (arrival-departure) [window_start-window_end] -> ... -> d(duration)
                let mut route_str = String::from("d(0)");
                for visit in &detailed.visits {
                    let care_time = ctx.patients[visit.patient_id].care_time;
                    let departure_time = visit.arrival_time + care_time;
                    route_str.push_str(&format!(
                        " -> {} ({:.2}-{:.2}) [{:.0}-{:.0}]",
                        visit.patient_id,
                        visit.arrival_time,
                        departure_time,
                        visit.start_window as i64,
                        visit.end_window as i64,
                    ));
                }
                route_str.push_str(&format!(" -> d({:.2})", detailed.route_duration));

                route_output.push_str(&format!(
                    "Nurse {:>2}  {:.2}  {:.0}  {}\n",
                    i + 1,
                    detailed.route_duration,
                    detailed.route_demand,
                    route_str
                ));
            }
        }

        route_output.push_str(&format!("{:-<68}\n", ""));
        route_output.push_str(&format!("Objective value (total duration): {:.2}\n", ind.total_travel));

        // Print to console
        print!("{}", route_output);

        // Save to file
        let output_file = run_dir.join("routes.txt");
        print!("Saving routes → {} ... ", output_file.display());
        use std::io::Write;
        let _ = std::io::stdout().flush();
        match std::fs::write(&output_file, &route_output) {
            Ok(()) => println!("done."),
            Err(e) => eprintln!("error: {e}"),
        }

        //  Optional plots 
        if cfg.plot {
            // Route plot
            let route_plot = run_dir.join("routes.png");
            if let Err(e) = plot::save_route_plot(
                &genome,
                &ctx,
                &cfg,
                ind.fitness,
                &route_plot.display().to_string(),
            ) {
                eprintln!("Route plot error: {e}");
            }

            // Fitness history plot
            let output = run_dir.join("fitness.png");
            print!("Saving plot → {} ... ", output.display());
            use std::io::Write;
            let _ = std::io::stdout().flush();
            match plot::save_plot(
                &history,
                &cfg,
                &ctx.instance.name,
                ctx.instance.benchmark,
                &output.display().to_string(),
            ) {
                Ok(()) => println!("done."),
                Err(e) => eprintln!("plot error: {e}"),
            }
        }
    }
}

fn write_used_config(cfg: &config::Config, output_path: &Path) -> Result<(), String> {
    let text = toml::to_string_pretty(cfg)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(output_path, text)
        .map_err(|e| format!("failed to write file: {e}"))
}

fn create_run_dir(instance_file: &str) -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system time error: {e}"))?
        .as_secs();

    let instance_name = Path::new(instance_file)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(instance_file);
    let instance_tag = sanitize_for_filename(instance_name);
    let suffix = random_suffix(4);

    let run_name = format!("run_{}_{}_{}", timestamp, instance_tag, suffix);
    let run_dir = std::env::current_dir()
        .map_err(|e| format!("cannot read current dir: {e}"))?
        .join(run_name);

    std::fs::create_dir_all(&run_dir)
        .map_err(|e| format!("cannot create '{}': {e}", run_dir.display()))?;

    Ok(run_dir)
}

fn sanitize_for_filename(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "instance".to_string()
    } else {
        out
    }
}

fn random_suffix(len: usize) -> String {
    use rand::Rng;
    const KEY_CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| KEY_CHARS[rng.gen_range(0..KEY_CHARS.len())] as char)
        .collect()
}

fn seed_to_hex(seed: &genevo::random::Seed) -> String {
    let mut s = String::with_capacity(seed.len() * 2);
    for b in seed {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_seed_hex(hex: &str) -> Result<genevo::random::Seed, String> {
    let trimmed = hex.trim();
    if trimmed.len() != 64 {
        return Err(format!(
            "expected 64 hex chars (32 bytes), got {} chars",
            trimmed.len()
        ));
    }

    let mut out = [0u8; 32];
    for i in 0..32 {
        let part = &trimmed[i * 2..i * 2 + 2];
        out[i] = u8::from_str_radix(part, 16)
            .map_err(|_| format!("invalid hex at byte {}: '{part}'", i))?;
    }
    Ok(out)
}

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

fn refresh_population(
    current_population: &[Genome],
    ctx: &Arc<types::ProblemContext>,
    pop_size: usize,
    replace_ratio: f64,
    seed: genevo::random::Seed,
) -> Population<Genome> {
    let bounded_ratio = replace_ratio.clamp(0.0, 1.0);
    let replace_count = ((pop_size as f64) * bounded_ratio).round() as usize;
    let replace_count = replace_count.min(pop_size);
    let keep_count = pop_size.saturating_sub(replace_count);

    let mut ranked: Vec<(f64, Genome)> = current_population
        .iter()
        .cloned()
        .map(|genome| (compute_individual(&genome, ctx).fitness, genome))
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut mixed: Vec<Genome> = ranked
        .into_iter()
        .take(keep_count.min(current_population.len()))
        .map(|(_, genome)| genome)
        .collect();

    let missing = pop_size.saturating_sub(mixed.len());
    if missing > 0 {
        let random_population: Population<Genome> = build_population()
            .with_genome_builder(RandomGenomeBuilder::new(Arc::clone(ctx)))
            .of_size(missing)
            .using_seed(seed);
        mixed.extend(random_population.individuals().iter().cloned());
    }

    Population::with_individuals(mixed)
}
