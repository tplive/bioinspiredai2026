/// A patient who needs home care from a nurse.
#[derive(Debug, Clone, PartialEq)]
pub struct Patient {
    pub id: usize,
    pub demand: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub care_time: f64,
    pub x: f64,
    pub y: f64,
}

/// Global problem instance metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemInstance {
    pub name: String,
    pub num_nurses: usize,
    /// Maximum demand each nurse can carry per route.
    pub capacity: f64,
    /// Known best travel time for comparison.
    pub benchmark: f64,
    /// Latest time a nurse must be back at the depot.
    pub depot_return_time: f64,
    pub depot_x: f64,
    pub depot_y: f64,
}

/// All shared, read-only problem data used by operators and the fitness function.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemContext {
    pub instance: ProblemInstance,
    /// 1-indexed: `patients[0]` is a dummy depot entry; real patients start at index 1.
    pub patients: Vec<Patient>,
    /// (num_patients+1) × (num_patients+1) matrix.
    /// Row/column 0 = depot; row/column i = patient with id i.
    pub travel_matrix: Vec<Vec<f64>>,
    pub penalty_factor: f64,
}

/// Evaluation result for a single nurse route.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub total_travel: f64,
    pub total_penalty: f64,
    #[allow(dead_code)]
    pub total_demand: f64,
    pub feasible: bool,
}

/// Aggregated evaluation result for a full individual (all routes).
#[derive(Debug, Clone)]
pub struct IndividualResult {
    pub total_travel: f64,
    pub total_penalty: f64,
    /// total_travel + total_penalty — lower is better.
    pub fitness: f64,
    pub feasible: bool,
}
