use std::sync::Arc;
use genevo::population::GenomeBuilder;
use genevo::random::Rng;
use crate::fitness::Genome;
use crate::types::ProblemContext;

// ── Random genome builder ─────────────────────────────────────────────────────

/// Builds random genomes by shuffling all patient IDs and distributing them
/// evenly across nurses.
///
/// Mirrors Julia's `init_rand_individual`.
#[derive(Clone, Debug)]
pub struct RandomGenomeBuilder {
    pub ctx: Arc<ProblemContext>,
}

impl RandomGenomeBuilder {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self { ctx }
    }
}

impl GenomeBuilder<Genome> for RandomGenomeBuilder {
    fn build_genome<R>(&self, _index: usize, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let num_nurses = self.ctx.instance.num_nurses;
        let num_patients = self.ctx.patients.len() - 1; // subtract dummy depot at [0]

        // Build a shuffled list of patient IDs (1-based).
        let mut ids: Vec<usize> = (1..=num_patients).collect();
        fisher_yates_shuffle(&mut ids, rng);

        split_into_routes(ids, num_nurses)
    }
}

// ── Nearest-neighbour genome builder ─────────────────────────────────────────

/// Builds genomes using a simple nearest-neighbour heuristic:
/// each nurse starts at a random patient and greedily visits the
/// nearest unvisited patient until the route reaches `route_length`.
///
/// Mirrors Julia's `init_individual_nearest_neighbour`.
#[derive(Clone, Debug)]
pub struct NearestNeighbourGenomeBuilder {
    pub ctx: Arc<ProblemContext>,
}

impl NearestNeighbourGenomeBuilder {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self { ctx }
    }
}

impl GenomeBuilder<Genome> for NearestNeighbourGenomeBuilder {
    fn build_genome<R>(&self, _index: usize, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let num_nurses = self.ctx.instance.num_nurses;
        let num_patients = self.ctx.patients.len() - 1; // subtract dummy depot
        let route_length = num_patients / num_nurses;

        let mat = &self.ctx.travel_matrix;

        let mut remaining: Vec<usize> = (1..=num_patients).collect();
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(num_nurses);

        for nurse in 0..num_nurses {
            // Extra patients go to the last nurse.
            let this_route_len = if nurse == num_nurses - 1 {
                remaining.len()
            } else {
                route_length
            };

            if remaining.is_empty() {
                routes.push(vec![]);
                continue;
            }

            // Pick a random starting patient for this nurse.
            let start_idx = rng.gen_range(0..remaining.len());
            let first_patient = remaining.remove(start_idx);
            let mut route = vec![first_patient];

            for _ in 1..this_route_len {
                if remaining.is_empty() {
                    break;
                }

                // Find the nearest remaining patient to the last visited one.
                let last = *route.last().unwrap();
                let (nearest_idx, _) = remaining.iter().enumerate().fold(
                    (0, f64::INFINITY),
                    |(best_i, best_d), (i, &pid)| {
                        let d = mat[last][pid];
                        if d < best_d { (i, d) } else { (best_i, best_d) }
                    },
                );

                let nearest = remaining.remove(nearest_idx);
                route.push(nearest);
            }

            routes.push(route);
        }

        // If any patients are leftover (shouldn't happen, but guard against it).
        if !remaining.is_empty() {
            routes.last_mut().unwrap().extend(remaining);
        }

        routes
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Fisher-Yates in-place shuffle using the provided Rng.
pub fn fisher_yates_shuffle<T, R: Rng + Sized>(slice: &mut [T], rng: &mut R) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    for i in (1..len).rev() {
        let j = rng.gen_range(0..=i);
        slice.swap(i, j);
    }
}

/// Distribute `ids` (already shuffled) as evenly as possible into `num_nurses` routes.
/// The last route absorbs any remainder.
fn split_into_routes(ids: Vec<usize>, num_nurses: usize) -> Genome {
    let total = ids.len();
    let base = total / num_nurses;
    let mut routes = Vec::with_capacity(num_nurses);
    let mut offset = 0;

    for nurse in 0..num_nurses {
        let len = if nurse == num_nurses - 1 {
            total - offset
        } else {
            base
        };
        routes.push(ids[offset..offset + len].to_vec());
        offset += len;
    }

    routes
}
