use std::sync::Arc;
use genevo::genetic::FitnessFunction;
use rayon::prelude::*;
use crate::types::{IndividualResult, ProblemContext, RouteResult};

/// Detailed information for a single patient visit within a route.
#[derive(Debug, Clone)]
pub struct PatientVisit {
    pub patient_id: usize,
    pub arrival_time: f64,
    pub start_window: f64,
    pub end_window: f64,
    pub demand: f64,
}

/// Detailed information for a complete nurse route.
#[derive(Debug, Clone)]
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
            demand: patients[id].demand,
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

/// Evaluate all routes of an individual and aggregate results in parallel
pub fn compute_individual(genome: &Genome, ctx: &ProblemContext) -> IndividualResult {

    // Each route is evaluated independently, so this scales with the number of nurses/cores.
    let results: Vec<RouteResult> = genome
        .par_iter()
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
#[derive(Clone, Debug)]
pub struct NurseFitness {
    pub ctx: Arc<ProblemContext>,
}

impl NurseFitness {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self { ctx }
    }
}

/// Scale factor so fractional time differences are preserved in the `i64` fitness.
const FITNESS_SCALE: f64 = 1_000.0;

impl FitnessFunction<Genome, i64> for NurseFitness {
    fn fitness_of(&self, genome: &Genome) -> i64 {
        let ind = compute_individual(genome, &self.ctx);
        // Negate so that lower cost ↔ higher genevo fitness.
        -(ind.fitness * FITNESS_SCALE) as i64
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
