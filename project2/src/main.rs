mod config;
mod crossover;
mod fitness;
mod ga;
mod local_search;
mod mutation;
mod parse;
mod plot;
mod population;
mod types;

use std::{
    env::current_dir,
    fs::copy,
    io::{Write, stdout},
    path::{Path, PathBuf},
    process::exit,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use config::PartialConfig;

use genevo::{population::build_population, prelude::*};

use fitness::{Genome, compute_detailed_route, compute_individual};
use mutation::MutationType;
use population::{
    ClarkeWrightGenomeBuilder, KMeansGenomeBuilder, NearestNeighbourGenomeBuilder,
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
    /// Path to TOML config file
    #[arg(short = 'C', long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Problem instance JSON file
    #[arg(short = 'f', long, value_name = "FILE")]
    file: Option<String>,

    /// Number of individuals in the population
    #[arg(short = 'p', long)]
    pop_size: Option<usize>,

    /// Maximum number of generations
    #[arg(short = 'g', long)]
    generations: Option<usize>,

    /// % of the population forwarded to the parent pool [0.0–1.0]
    #[arg(long)]
    selection_ratio: Option<f64>,

    /// Probability of applying crossover to each parent pair [0.0–1.0]
    #[arg(long)]
    crossover_rate: Option<f64>,

    /// Mutation probability [0.0–1.0].
    #[arg(short = 'm', long)]
    mutation_rate: Option<f64>,

    /// Mutation operator: "swap" or "insert".
    #[arg(long, value_name = "TYPE")]
    mutation_type: Option<String>,

    /// % of offspring kept in the next generation [0.0–1.0]
    #[arg(long)]
    reinsertion_ratio: Option<f64>,

    /// Multiplier applied to each unit of constraint violation.
    #[arg(long)]
    penalty_factor: Option<f64>,

    /// Population initialisation method: "random", "nn" (nearest-neighbour), "cw" (clarke-wright), or "kmeans" (kmeans clustering)
    #[arg(short = 'i', long)]
    init: Option<String>,

    /// Selection method: "truncation" or "tournament"
    #[arg(long)]
    selection_type: Option<String>,

    /// Tournament size for tournament selection (typically 2-5)
    #[arg(long)]
    tournament_size: Option<usize>,

    /// Plot and save to PNG after the run; results/run_<timestamp>_<instance file name>_<rand suff>/*
    #[arg(long, action = clap::ArgAction::SetTrue)]
    plot: bool,

    /// Optional deterministic random seed as 64 hex chars (32 bytes)
    #[arg(long, value_name = "HEX32")]
    random_seed: Option<String>,

    /// Replace population after this many generations without improvement
    #[arg(long)]
    stagnation_replace_after: Option<usize>,

    /// Fraction of population to replace on stagnation refresh [0.0–1.0]
    #[arg(long)]
    stagnation_replace_ratio: Option<f64>,

    /// Number of hill climbing steps when stagnating
    #[arg(long)]
    hill_climb_steps: Option<usize>,

}

impl Cli {
    fn into_partial(self) -> PartialConfig {
        PartialConfig {
            file: self.file,
            pop_size: self.pop_size,
            generations: self.generations,
            selection_ratio: self.selection_ratio,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            mutation_type: self.mutation_type,
            reinsertion_ratio: self.reinsertion_ratio,
            penalty_factor: self.penalty_factor,
            init: self.init,
            selection_type: self.selection_type,
            tournament_size: self.tournament_size,
            plot: if self.plot { Some(true) } else { None },
            random_seed: self.random_seed,
            stagnation_replace_after: self.stagnation_replace_after,
            stagnation_replace_ratio: self.stagnation_replace_ratio,
            hill_climb_steps: self.hill_climb_steps,
        }
    }
}

//  Entry point

fn main() {
    //  Parse CLI and resolve full configuration
    let mut cli = Cli::parse();
    let input_config_path = cli.config.take();

    // Start from default values, overlay config file if given.
    let base = match input_config_path.as_ref() {
        Some(path) => config::load_file(path).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }),
        None => config::Config::default(),
    };

    // CLI flags take precedence over config file
    let mut cfg = cli.into_partial().apply_onto(base);

    // If random seed given, use it to recreate previous results. Should be deterministic.
    let run_seed = match cfg.random_seed.as_deref() {
        Some(hex) => parse_seed_hex(hex).unwrap_or_else(|e| {
            eprintln!("Invalid --random-seed/config random_seed: {e}");
            exit(1);
        }),
        None => genevo::random::random_seed(), // Use random seed if not given
    };
    cfg.random_seed = Some(seed_to_hex(&run_seed));

    // Results foler and files
    let run_dir = create_run_dir(&cfg.file).unwrap_or_else(|e| {
        eprintln!("Could not create run directory: {e}");
        exit(1);
    });

    let used_config_path = run_dir.join("config_used.toml");
    if let Err(e) = write_used_config(&cfg, &used_config_path) {
        eprintln!(
            "Could not write used config '{}': {e}",
            used_config_path.display()
        );
        exit(1);
    }

    if let Some(path) = input_config_path {
        let original_cfg_copy = run_dir.join("config_input.toml");
        if let Err(e) = copy(&path, &original_cfg_copy) {
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
    let now = Instant::now();
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
    println!(
        "Random seed:      {}",
        cfg.random_seed.as_deref().unwrap_or("<none>")
    );
    println!("Run directory:    {}", run_dir.display());
    println!();
    println!("Loaded and parsed input in {:?}", Instant::now()-now);
    //  Build initial population
    let initial_population: Population<Genome> = match cfg.init.as_str() {
        "nn" => build_population()
            .with_genome_builder(NearestNeighbourGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        "cw" => build_population()
            .with_genome_builder(ClarkeWrightGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        "kmeans" => build_population()
            .with_genome_builder(KMeansGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
        _ => build_population()
            .with_genome_builder(RandomGenomeBuilder::new(Arc::clone(&ctx)))
            .of_size(cfg.pop_size)
            .using_seed(run_seed),
    };

    // ######################################
    // #### THE RUNNING OF THE ALGORITHM ####
    // ######################################
    let now = Instant::now();
    let ga_results = ga::run_ga(&cfg, &ctx, initial_population, run_seed, mutation_op_type);
    println!("The running took {:?}", Instant::now() - now);

    let best_genome = ga_results.best_genome;

    let history = ga_results.history;
    let generations_run = ga_results.generations_run;

    //  Print final solution
    if let Some(genome) = best_genome.as_ref() {
        println!();
        let ind = compute_individual(genome, &ctx);
        let pct_diff = (ind.total_travel - ctx.instance.benchmark) / ctx.instance.benchmark * 100.0;

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
        route_output.push_str(&format!(
            "Depot return time: {:.2}\n",
            ctx.instance.depot_return_time
        ));
        route_output.push_str(&format!("{:-<68}\n", ""));

        for (i, route) in genome.iter().enumerate() {
            if route.is_empty() {
                route_output.push_str(&format!(
                    "Nurse {:>2}    0.00     0.00  d(0) -> d(0.00)\n",
                    i + 1
                ));
            } else {
                let detailed = compute_detailed_route(route, &ctx);

                // Build route string: d(0) -> patient_id (arrival-departure) [window_start-window_end] -> ... -> d(duration)
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
        route_output.push_str(&format!(
            "Objective value (total duration): {:.2}\n",
            ind.total_travel
        ));

        // Save to file
        let output_file = run_dir.join("routes.txt");
        print!("Saving routes → {} ... ", output_file.display());
        let _ = stdout().flush();
        match std::fs::write(&output_file, &route_output) {
            Ok(()) => println!("done."),
            Err(e) => eprintln!("error: {e}"),
        }

        //  Optional plots
        if cfg.plot {
            // Route plot
            let route_plot = run_dir.join("routes.png");
            if let Err(e) = plot::save_route_plot(
                genome,
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
            let _ = std::io::stdout().flush();
            match plot::save_plot(
                &history,
                &cfg,
                generations_run,
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
    let text =
        toml::to_string_pretty(cfg).map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(output_path, text).map_err(|e| format!("failed to write file: {e}"))
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
    let run_dir = current_dir()
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

fn seed_to_hex(seed: &Seed) -> String {
    let mut s = String::with_capacity(seed.len() * 2);
    for b in seed {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_seed_hex(hex: &str) -> Result<Seed, String> {
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
