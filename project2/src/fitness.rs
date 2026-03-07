use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use genevo::genetic::FitnessFunction;
use crate::types::{IndividualResult, ProblemContext, RouteResult};

/// Detailed information for a single patient visit within a route.
#[derive(Debug)]
pub struct PatientVisit {
    pub patient_id: usize,
    pub arrival_time: f64,
    pub start_window: f64,
    pub end_window: f64,
    //pub demand: f64,
}

/// Detailed information for a complete nurse route.
#[derive(Debug)]
pub struct DetailedRoute {
    pub route_duration: f64,
    pub route_demand: f64,
    pub visits: Vec<PatientVisit>,
}

/// Genome type alias – outer Vec = one entry per nurse,
/// inner Vec = ordered list of patient IDs visited by that nurse.
pub type Genome = Vec<Vec<usize>>;

// ── Route and individual evaluation ─────────────────────────────────────────

/// Evaluate a single nurse route.
///
/// Mirrors the Julia `compute_route` function exactly:
/// - Start at the depot (time_of_day = 0).
/// - For each patient: drive, (wait if early), accumulate penalty if late,
///   provide care, accumulate penalty if care exceeds window.
/// - Return to the depot; penalise if the nurse arrives after `depot_return_time`.
/// - Penalise if cumulative demand exceeds the nurse's capacity.
pub fn compute_route(route: &[usize], ctx: &ProblemContext) -> RouteResult {
    if route.is_empty() {
        return RouteResult {
            total_travel: 0.0,
            total_penalty: 0.0,
            total_demand: 0.0,
            feasible: true,
        };
    }

    let penalty_factor = ctx.penalty_factor;
    let patients = &ctx.patients;
    let mat = &ctx.travel_matrix;
    let capacity = ctx.instance.capacity;
    let return_time = ctx.instance.depot_return_time;

    let mut time_of_day = 0.0_f64;
    let mut travel_time = 0.0_f64;
    let mut total_demand = 0.0_f64;
    let mut penalty = 0.0_f64;
    let mut prev_id = 0_usize; // depot is node 0

    for &id in route {
        // Cache patient data to avoid repeated indexing
        let patient = &patients[id];
        
        // Travel from previous location to current patient.
        let t = mat[prev_id][id];
        time_of_day += t;
        travel_time += t;

        // Wait if the nurse arrives before the patient's time window opens.
        let wait = patients[id].start_time - time_of_day;
        if wait > 0.0 {
            time_of_day += wait;
        }

        // Penalty if the nurse arrives after the patient's time window closes.
        let late_arrival = time_of_day - patients[id].end_time;
        if late_arrival > 0.0 {
            penalty += late_arrival * penalty_factor;
        }

        // Provide care.
        time_of_day += patients[id].care_time;

        // Penalty if care extends past the patient's time window.
        let late_care = time_of_day - patients[id].end_time;
        if late_care > 0.0 {
            penalty += late_care * penalty_factor;
        }

        total_demand += patients[id].demand;
        prev_id = id;
    }

    // Return to depot.
    let return_travel = mat[prev_id][0];
    time_of_day += return_travel;
    travel_time += return_travel;

    // Penalty for exceeding nurse capacity.
    if total_demand > capacity {
        let excess = total_demand - capacity;
        penalty += excess * penalty_factor;
    }

    // Penalty for returning to depot after deadline.
    let late_return = time_of_day - return_time;
    if late_return > 0.0 {
        penalty += late_return * penalty_factor;
    }

    RouteResult {
        total_travel: travel_time,
        total_penalty: penalty,
        total_demand,
        feasible: penalty == 0.0,
    }
}

/// Compute detailed route information including per-patient visit times.
pub fn compute_detailed_route(route: &[usize], ctx: &ProblemContext) -> DetailedRoute {
    let mut visits = Vec::new();
    let mut time_of_day = 0.0_f64;
    let mut travel_time = 0.0_f64;
    let mut total_demand = 0.0_f64;
    let mut prev_id = 0_usize; // depot is node 0

    let patients = &ctx.patients;
    let mat = &ctx.travel_matrix;

    for &id in route {
        // Travel from previous location to current patient.
        let t = mat[prev_id][id];
        time_of_day += t;
        travel_time += t;

        // Record arrival time and wait if needed
        let arrival_time = time_of_day;
        let wait = patients[id].start_time - time_of_day;
        if wait > 0.0 {
            time_of_day += wait;
        }

        // Provide care.
        time_of_day += patients[id].care_time;

        total_demand += patients[id].demand;

        visits.push(PatientVisit {
            patient_id: id,
            arrival_time,
            start_window: patients[id].start_time,
            end_window: patients[id].end_time,
            //demand: patients[id].demand,
        });

        prev_id = id;
    }

    // Return to depot.
    let return_travel = mat[prev_id][0];
    travel_time += return_travel;

    DetailedRoute {
        route_duration: travel_time,
        route_demand: total_demand,
        visits,
    }
}

/// Evaluate all routes of an individual and aggregate results.
pub fn compute_individual(genome: &Genome, ctx: &ProblemContext) -> IndividualResult {
    let results: Vec<RouteResult> = genome
        .iter()
        .map(|route| compute_route(route, ctx))
        .collect();

    let mut total_travel = 0.0_f64;
    let mut total_penalty = 0.0_f64;

    for r in results {
        total_travel += r.total_travel;
        total_penalty += r.total_penalty;
    }

    IndividualResult {
        total_travel,
        total_penalty,
        fitness: total_travel + total_penalty,
        feasible: total_penalty == 0.0,
    }
}

// ── genevo FitnessFunction wrapper ───────────────────────────────────────────

/// Wraps the shared `ProblemContext` and implements genevo's `FitnessFunction`.
///
/// genevo **maximises** fitness, so we return `-(travel + penalty) * SCALE`
/// (a large negative number for costly solutions, closer to 0 for cheap ones).
///
/// Includes a fitness cache to avoid recomputing fitness for previously evaluated genomes.
#[derive(Clone, Debug)]
pub struct RouteFitness {
    pub ctx: Arc<ProblemContext>,
    cache: Arc<Mutex<HashMap<Genome, i64>>>,
}

impl RouteFitness {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self {
            ctx,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Scale factor so fractional time differences are preserved in the `i64` fitness.
const FITNESS_SCALE: f64 = 1_000.0;

impl FitnessFunction<Genome, i64> for RouteFitness {
    fn fitness_of(&self, genome: &Genome) -> i64 {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&fitness) = cache.get(genome) {
                return fitness;
            }
        }

        // Not in cache - compute it
        let ind = compute_individual(genome, &self.ctx);
        let fitness = -(ind.fitness * FITNESS_SCALE) as i64;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(genome.clone(), fitness);
        }

        fitness
    }

    fn average(&self, values: &[i64]) -> i64 {
        if values.is_empty() {
            return 0;
        }
        values.iter().sum::<i64>() / values.len() as i64
    }

    /// The ideal individual has zero cost (0 travel + 0 penalty).
    fn highest_possible_fitness(&self) -> i64 {
        0
    }

    fn lowest_possible_fitness(&self) -> i64 {
        i64::MIN / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Patient, ProblemInstance};
    use std::time::Instant;

    fn create_test_context() -> Arc<ProblemContext> {
        // Create a simple test problem with 10 patients
        let depot = Patient {
            id: 0,
            demand: 0.0,
            start_time: 0.0,
            end_time: 500.0,
            care_time: 0.0,
            x: 0.0,
            y: 0.0,
        };

        let mut patients = vec![depot];
        for i in 1..=10 {
            patients.push(Patient {
                id: i,
                demand: 5.0,
                start_time: 0.0,
                end_time: 480.0,
                care_time: 10.0,
                x: (i as f64) * 10.0,
                y: (i as f64) * 5.0,
            });
        }

        // Simple Euclidean distance matrix
        let n = patients.len();
        let mut travel_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = patients[i].x - patients[j].x;
                let dy = patients[i].y - patients[j].y;
                travel_matrix[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }

        Arc::new(ProblemContext {
            instance: ProblemInstance {
                name: "test".to_string(),
                num_nurses: 3,
                capacity: 30.0,
                benchmark: 100.0,
                depot_return_time: 480.0,
                depot_x: 0.0,
                depot_y: 0.0,
            },
            patients,
            travel_matrix,
            penalty_factor: 10.0,
        })
    }

    #[test]
    fn test_fitness_computation_vs_cache_lookup() {
        let ctx = create_test_context();
        let test_genome: Genome = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9, 10],
        ];

        const ITERATIONS: usize = 100_000;

        // Test 1: Direct fitness computation (bypassing cache)
        println!("\n=== Fitness Computation vs Cache Lookup ===");
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let ind = compute_individual(&test_genome, &ctx);
            let _fitness = -(ind.fitness * FITNESS_SCALE) as i64;
        }
        let direct_duration = start.elapsed();
        println!(
            "Direct computation ({} iterations): {:?}",
            ITERATIONS, direct_duration
        );

        // Test 2: Cache lookup after initial population
        let fitness_fn = RouteFitness::new(ctx.clone());
        
        // Prime the cache with one evaluation
        let _first = fitness_fn.fitness_of(&test_genome);
        
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _fitness = fitness_fn.fitness_of(&test_genome);
        }
        let cache_duration = start.elapsed();
        println!(
            "Cache lookup ({} iterations):      {:?}",
            ITERATIONS, cache_duration
        );

        // Calculate speedup
        let speedup = direct_duration.as_secs_f64() / cache_duration.as_secs_f64();
        println!("\nSpeedup factor: {:.2}x", speedup);
        
        if speedup > 1.0 {
            println!("Cache is {:.2}% faster", (1.0 - 1.0 / speedup) * 100.0);
        } else {
            println!("Cache overhead: {:.2}%", (speedup - 1.0) * 100.0);
        }

        println!("\nNote: For small problems, cache overhead (mutex + hash) may exceed");
        println!("savings. Cache benefits increase with larger problems and repeated evaluations.");

        // Test passes - we're just measuring performance characteristics
    }
}
