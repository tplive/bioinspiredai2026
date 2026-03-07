use crate::fitness::{Genome, compute_individual};
use crate::types::ProblemContext;
use genevo::population::{GenomeBuilder, Population, build_population};
use genevo::random::{Rng, Seed};
use std::cmp::Ordering;
use std::sync::Arc;

/// Builds random genomes by shuffling all patient IDs and distributing them
/// evenly across nurses.
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

        // Build a shuffled list of patient IDs
        let mut ids: Vec<usize> = (1..=num_patients).collect();
        shuffle(&mut ids, rng);

        // even number of patients for each nurse
        split_into_routes(ids, num_nurses)
    }
}

// Clarke-Wright Savings genome builder
// -because I thought it could be a good fit..

// https://www.kaggle.com/code/mayanksethia/clark-wright-savings-algorithm
/// Builds genomes using the Clarke-Wright Savings algorithm.
pub struct ClarkeWrightGenomeBuilder {
    pub ctx: Arc<ProblemContext>,
}

impl ClarkeWrightGenomeBuilder {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self { ctx }
    }
}

impl GenomeBuilder<Genome> for ClarkeWrightGenomeBuilder {
    fn build_genome<R>(&self, _index: usize, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let num_patients = self.ctx.patients.len() - 1; // subtract dummy depot
        let mat = &self.ctx.travel_matrix;
        let capacity = self.ctx.instance.capacity;
        let patients = &self.ctx.patients;

        if num_patients == 0 {
            return vec![vec![]];
        }

        // Step 1: Start with each patient in its own route
        let mut routes: Vec<Vec<usize>> = (1..=num_patients).map(|p| vec![p]).collect();
        let mut route_demands: Vec<f64> = patients[1..=num_patients]
            .iter()
            .map(|p| p.demand)
            .collect();

        // Step 2: Calculate all savings s(i,j) = dist(0,i) + dist(0,j) - dist(i,j)
        let mut savings: Vec<(f64, usize, usize)> = Vec::new();
        for i in 1..=num_patients {
            for j in (i + 1)..=num_patients {
                let saving = mat[0][i] + mat[0][j] - mat[i][j];
                if saving > 0.0 {
                    savings.push((saving, i, j));
                }
            }
        }

        // Step 3: Sort savings in descending order
        savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Step 4: Process savings and merge routes
        for (_saving, i, j) in savings {
            // Find which routes contain i and j
            let route_i = routes.iter().position(|r| r.contains(&i));
            let route_j = routes.iter().position(|r| r.contains(&j));

            if route_i.is_none() || route_j.is_none() {
                continue;
            }

            let idx_i = route_i.unwrap();
            let idx_j = route_j.unwrap();

            // Skip if same route
            if idx_i == idx_j {
                continue;
            }

            let route_i_ref = &routes[idx_i];
            let route_j_ref = &routes[idx_j];

            // Check if i is at the end of its route and j is at the start of its route
            // or vice versa (can only merge routes at endpoints)
            let i_at_end = route_i_ref.last() == Some(&i);
            let i_at_start = route_i_ref.first() == Some(&i);
            let j_at_end = route_j_ref.last() == Some(&j);
            let j_at_start = route_j_ref.first() == Some(&j);

            let can_merge = (i_at_end && j_at_start) || (j_at_end && i_at_start);

            if !can_merge {
                continue;
            }

            // Check capacity constraint
            let combined_demand = route_demands[idx_i] + route_demands[idx_j];
            if combined_demand > capacity {
                continue;
            }

            // Merge the routes
            let new_route = if i_at_end && j_at_start {
                // Append route_j to route_i
                let mut r = routes[idx_i].clone();
                r.extend(&routes[idx_j]);
                r
            } else {
                // j_at_end && i_at_start: Append route_i to route_j
                let mut r = routes[idx_j].clone();
                r.extend(&routes[idx_i]);
                r
            };

            // Remove old routes and add merged route
            // Remove higher index first to avoid index invalidation
            let (high_idx, low_idx) = if idx_i > idx_j {
                (idx_i, idx_j)
            } else {
                (idx_j, idx_i)
            };

            routes.remove(high_idx);
            route_demands.remove(high_idx);
            routes.remove(low_idx);
            route_demands.remove(low_idx);

            routes.push(new_route);
            route_demands.push(combined_demand);
        }

        // Shuffle routes to add some randomness between runs
        shuffle(&mut routes, rng);

        // Ensure we have exactly num_nurses routes by splitting or padding
        let num_nurses = self.ctx.instance.num_nurses;
        balance_routes(routes, num_nurses)
    }
}

//  Nearest-neighbour genome builder

/// Builds genomes using a simple nearest-neighbour heuristic:
/// each nurse starts at a random patient and greedily visits the
/// nearest unvisited patient until the route reaches `route_length`.
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

        // If any patients are leftover
        if !remaining.is_empty() {
            routes.last_mut().unwrap().extend(remaining);
        }

        routes
    }
}

// K-means genome builder
// Aurelien Géron, HOML page 263, and the Julia version from last year

/// Builds genomes by clustering patients with K-means on `(x, y)` coordinates,
/// then using each cluster as a nurse route.
/// Assumes that the travel times are somewhat coherent between the coordinates.
/// and for good results, coordinates must be clusterable..
pub struct KMeansGenomeBuilder {
    pub ctx: Arc<ProblemContext>,
}

impl KMeansGenomeBuilder {
    pub fn new(ctx: Arc<ProblemContext>) -> Self {
        Self { ctx }
    }
}

impl GenomeBuilder<Genome> for KMeansGenomeBuilder {
    fn build_genome<R>(&self, _index: usize, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let num_nurses = self.ctx.instance.num_nurses;
        let num_patients = self.ctx.patients.len() - 1; // subtract dummy depot at [0]

        if num_nurses == 0 {
            return vec![];
        }
        if num_patients == 0 {
            return vec![vec![]; num_nurses];
        }

        let k = num_nurses.min(num_patients).max(1);
        let mut patient_ids: Vec<usize> = (1..=num_patients).collect();
        shuffle(&mut patient_ids, rng);

        // Initialize centroids from random patients.
        let mut centroids: Vec<(f64, f64)> = patient_ids
            .iter()
            .take(k)
            .map(|&pid| {
                let p = &self.ctx.patients[pid];
                (p.x, p.y)
            })
            .collect();

        let mut assignments = vec![0usize; num_patients];

        // Small fixed number of iterations keeps this fast and stable.
        for _ in 0..15 {
            // Assignment step.
            for (idx, &pid) in patient_ids.iter().enumerate() {
                let p = &self.ctx.patients[pid];
                let (best_cluster, _) = centroids.iter().enumerate().fold(
                    (0usize, f64::INFINITY),
                    |(best_i, best_d), (i, &(cx, cy))| {
                        let dx = p.x - cx;
                        let dy = p.y - cy;
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d {
                            (i, d2)
                        } else {
                            (best_i, best_d)
                        }
                    },
                );
                assignments[idx] = best_cluster;
            }

            // Update step.
            let mut sums = vec![(0.0f64, 0.0f64, 0usize); k];
            for (idx, &pid) in patient_ids.iter().enumerate() {
                let c = assignments[idx];
                let p = &self.ctx.patients[pid];
                sums[c].0 += p.x;
                sums[c].1 += p.y;
                sums[c].2 += 1;
            }

            for c in 0..k {
                if sums[c].2 > 0 {
                    centroids[c] = (sums[c].0 / sums[c].2 as f64, sums[c].1 / sums[c].2 as f64);
                } else {
                    // Re-seed empty cluster from a random patient.
                    let pid = patient_ids[rng.gen_range(0..patient_ids.len())];
                    let p = &self.ctx.patients[pid];
                    centroids[c] = (p.x, p.y);
                }
            }
        }

        let mut routes = vec![Vec::<usize>::new(); k];
        for (idx, &pid) in patient_ids.iter().enumerate() {
            routes[assignments[idx]].push(pid);
        }

        // Add diversity in visit order inside each spatial cluster.
        for route in &mut routes {
            shuffle(route, rng);
        }

        balance_routes(routes, num_nurses)
    }
}

/// Shuffle using the provided Rng.
pub fn shuffle<T, R: Rng + Sized>(slice: &mut [T], rng: &mut R) {
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

/// Balance a variable number of routes to exactly `num_nurses` routes.
/// If there are too many routes, split the longest ones.
/// If there are too few routes, add empty routes.
fn balance_routes(mut routes: Vec<Vec<usize>>, num_nurses: usize) -> Genome {
    // Remove empty routes
    routes.retain(|r| !r.is_empty());

    while routes.len() < num_nurses {
        // If we need more routes, split the longest route
        if let Some(longest_idx) = routes
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.len())
            .map(|(i, _)| i)
        {
            if routes[longest_idx].len() > 1 {
                let mut route = routes.remove(longest_idx);
                let mid = route.len() / 2;
                let second_half = route.split_off(mid);
                routes.push(route);
                routes.push(second_half);
            } else {
                // All routes have length 1 or 0, just add empty routes
                routes.push(vec![]);
            }
        } else {
            routes.push(vec![]);
        }
    }

    while routes.len() > num_nurses {
        // If we have too many routes, merge the two smallest
        if routes.len() <= 1 {
            break;
        }

        // Find two smallest routes
        let mut indexed: Vec<(usize, usize)> = routes
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.len()))
            .collect();
        indexed.sort_by_key(|(_, len)| *len);

        let idx1 = indexed[0].0;
        let idx2 = indexed[1].0;

        let (low_idx, high_idx) = if idx1 < idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };

        let route2 = routes.remove(high_idx);
        routes[low_idx].extend(route2);
    }

    routes
}

//Population refresh
/// Refreshes the population by keeping the best individuals and replacing the rest
/// with newly generated random individuals.
///
/// # Arguments
/// * `current_population` - The current population to refresh
/// * `ctx` - Problem context
/// * `pop_size` - Target population size
/// * `replace_ratio` - Fraction of population to replace (0.0 to 1.0)
/// * `seed` - Random seed for generating new individuals
pub fn refresh_population(
    current_population: &[Genome],
    ctx: &Arc<ProblemContext>,
    pop_size: usize,
    replace_ratio: f64,
    seed: Seed,
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
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

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
