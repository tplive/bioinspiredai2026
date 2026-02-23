use std::collections::HashSet;
use std::sync::Arc;

use genevo::genetic::{Children, Parents};
use genevo::operator::{CrossoverOp, GeneticOperator};
use genevo::random::Rng;

use crate::fitness::{compute_route, Genome};
use crate::types::ProblemContext;

// ── Main operator struct ──────────────────────────────────────────────────────

/// Route-based crossover for the nurse scheduling VRP.
///
/// For each parent pair:
/// 1. Pick a "donor route" from parent-2 and remove those patients from parent-1.
/// 2. Greedily re-insert the orphaned patients into the fittest position in parent-1.
/// 3. Repeat in the other direction to produce the second child.
///
/// Mirrors the Julia `crossover_population` / `insert_orphans` logic.
#[derive(Clone, Debug)]
pub struct RouteCrossover {
    pub ctx: Arc<ProblemContext>,
    /// Probability of applying crossover to each parent pair. If not triggered,
    /// the children are copies of the parents.
    pub crossover_rate: f64,
}

impl RouteCrossover {
    pub fn new(ctx: Arc<ProblemContext>, crossover_rate: f64) -> Self {
        Self { ctx, crossover_rate }
    }
}

impl GeneticOperator for RouteCrossover {
    fn name() -> String {
        "Route-Based-Crossover".to_string()
    }
}

impl CrossoverOp<Genome> for RouteCrossover {
    fn crossover<R>(&self, parents: Parents<Genome>, rng: &mut R) -> Children<Genome>
    where
        R: Rng + Sized,
    {
        // `Parents<Genome> = Vec<Genome>`. With num_individuals_per_parents = 2,
        // we receive exactly two parent genomes and produce two children.
        if parents.len() < 2 {
            return parents;
        }

        let p1 = &parents[0];
        let p2 = &parents[1];

        let (c1, c2) = if rng.r#gen::<f64>() < self.crossover_rate {
            crossover_pair(p1, p2, &self.ctx, rng)
        } else {
            (p1.clone(), p2.clone())
        };

        vec![c1, c2]
    }
}

// ── Crossover logic ───────────────────────────────────────────────────────────

/// Produce two offspring from two parents.
fn crossover_pair<R: Rng + Sized>(
    p1: &Genome,
    p2: &Genome,
    ctx: &ProblemContext,
    rng: &mut R,
) -> (Genome, Genome) {
    // Pick a "donor" route from each parent.
    let donor1 = pick_route(p1, ctx, rng); // will be injected into p2's offspring
    let donor2 = pick_route(p2, ctx, rng); // will be injected into p1's offspring

    // Build child 1: start from p1, remove patients that are in donor2, re-insert.
    let (incomplete1, orphans1) = remove_patients(&donor2, p1.clone());
    let c1 = insert_orphans(incomplete1, orphans1, ctx, rng);

    // Build child 2: start from p2, remove patients that are in donor1, re-insert.
    let (incomplete2, orphans2) = remove_patients(&donor1, p2.clone());
    let c2 = insert_orphans(incomplete2, orphans2, ctx, rng);

    (c1, c2)
}

/// Select a route to act as the "crossover segment".
/// Prefer infeasible routes so that crossover pressure helps to fix them first.
/// Fall back to a random route when every route is feasible.
fn pick_route<R: Rng + Sized>(
    individual: &Genome,
    ctx: &ProblemContext,
    rng: &mut R,
) -> Vec<usize> {
    // Collect indices of infeasible routes.
    let infeasible: Vec<usize> = individual
        .iter()
        .enumerate()
        .filter(|(_, route)| !compute_route(route, ctx).feasible)
        .map(|(i, _)| i)
        .collect();

    if !infeasible.is_empty() {
        let idx = infeasible[rng.gen_range(0..infeasible.len())];
        return individual[idx].clone();
    }

    // All feasible – pick any route at random.
    let idx = rng.gen_range(0..individual.len());
    individual[idx].clone()
}

/// Remove every patient that appears in `donor` from `individual`.
/// Returns `(incomplete_individual, orphaned_patients)`.
fn remove_patients(donor: &[usize], mut individual: Genome) -> (Genome, Vec<usize>) {
    let donor_set: HashSet<usize> = donor.iter().copied().collect();
    let mut orphans = Vec::new();

    for route in &mut individual {
        let mut remaining = Vec::new();
        for &pid in route.iter() {
            if donor_set.contains(&pid) {
                orphans.push(pid);
            } else {
                remaining.push(pid);
            }
        }
        *route = remaining;
    }

    (individual, orphans)
}

/// Greedily re-insert orphaned patients into the closest available route.
///
/// For each orphan:
/// - Candidate routes are those where adding the orphan's demand does not
///   exceed the nurse's capacity.
/// - Among candidates, pick the one with the minimum travel distance from
///   its last patient (or the depot for an empty route).
/// - Routes that already contain patients from the same "region" (proximity
///   to the orphan) get a small distance bonus (`× 0.5`) – an approximation
///   of the Julia cluster-priority logic without requiring K-means clusters.
/// - If no capacity-respecting route is found, append to a random route.
fn insert_orphans<R: Rng + Sized>(
    mut routes: Genome,
    orphans: Vec<usize>,
    ctx: &ProblemContext,
    rng: &mut R,
) -> Genome {
    let patients = &ctx.patients;
    let mat = &ctx.travel_matrix;
    let capacity = ctx.instance.capacity;

    for orphan in orphans {
        let orphan_demand = patients[orphan].demand;

        // Current demand per route (recalculated from scratch for correctness).
        let demands: Vec<i32> = routes
            .iter()
            .map(|r| r.iter().map(|&pid| patients[pid].demand).sum())
            .collect();

        let mut best_route_idx: Option<usize> = None;
        let mut best_dist = f64::INFINITY;

        for (ri, route) in routes.iter().enumerate() {
            // Capacity check.
            if demands[ri] + orphan_demand > capacity {
                continue;
            }

            // Distance from the last patient in the route (or depot).
            let from = route.last().copied().unwrap_or(0);
            let mut dist = mat[from][orphan];

            // Proximity bonus: if any patient in the route is geographically
            // close to the orphan (within 20% of average inter-patient distance),
            // halve the effective distance (mirrors Julia's cluster bonus).
            let close_threshold = 10.0; // rough geographic unit threshold
            let has_nearby = route.iter().any(|&pid| {
                let dx = patients[pid].x - patients[orphan].x;
                let dy = patients[pid].y - patients[orphan].y;
                (dx * dx + dy * dy).sqrt() < close_threshold
            });
            if has_nearby {
                dist *= 0.5;
            }

            if dist < best_dist {
                best_dist = dist;
                best_route_idx = Some(ri);
            }
        }

        match best_route_idx {
            Some(ri) => routes[ri].push(orphan),
            None => {
                // No capacity-respecting route found – append to a random route.
                let ri = rng.gen_range(0..routes.len());
                routes[ri].push(orphan);
            }
        }
    }

    routes
}
