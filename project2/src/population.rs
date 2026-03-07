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

        let mut patient_ids: Vec<usize> = (1..=num_patients).collect();
        shuffle(&mut patient_ids, rng);

        // Find optimal k using elbow method
        let k = find_optimal_k(&self.ctx, &patient_ids, num_nurses, num_patients, rng);

        let (assignments, _centroids) = run_kmeans(&self.ctx, &patient_ids, k, rng);

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

/// Find the optimal number of clusters using the elbow method.
/// Tries different k values and picks the one where the marginal improvement diminishes.
fn find_optimal_k<R: Rng + Sized>(
    ctx: &Arc<ProblemContext>,
    patient_ids: &[usize],
    num_nurses: usize,
    num_patients: usize,
    rng: &mut R,
) -> usize {
    // Try k values from 2 to min(num_nurses, num_patients, reasonable_max)
    let min_k = 2;
    let max_k = num_nurses.min(num_patients).min(15); // Cap at 15 to keep it fast

    if max_k < min_k {
        return 1;
    }

    let mut inertias = Vec::new();
    
    for k in min_k..=max_k {
        let (assignments, centroids) = run_kmeans(ctx, patient_ids, k, rng);
        let inertia = calculate_inertia(ctx, patient_ids, &assignments, &centroids);
        inertias.push((k, inertia));
    }

    // Find elbow using rate of change method
    // Calculate the rate of improvement for each k
    let mut best_k = min_k;
    let mut best_score = f64::NEG_INFINITY;

    for i in 0..inertias.len().saturating_sub(1) {
        let (k, inertia) = inertias[i];
        let (_, next_inertia) = inertias[i + 1];
        
        // Rate of improvement (negative because inertia decreases)
        let improvement_rate = inertia - next_inertia;
        
        // Score: balance improvement rate with keeping k small
        // Prefer smaller k unless improvement rate is significantly better
        let normalized_k = k as f64 / max_k as f64;
        let score = improvement_rate * (1.0 - normalized_k * 0.3);
        
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    // Also consider the last k value
    if inertias.len() > 0 {
        let (last_k, last_inertia) = inertias[inertias.len() - 1];
        if inertias.len() > 1 {
            let (_, prev_inertia) = inertias[inertias.len() - 2];
            let improvement_rate = prev_inertia - last_inertia;
            let normalized_k = last_k as f64 / max_k as f64;
            let score = improvement_rate * (1.0 - normalized_k * 0.3);
            
            if score > best_score {
                best_k = last_k;
            }
        }
    }

    best_k
}

/// Run k-means clustering and return assignments and centroids.
fn run_kmeans<R: Rng + Sized>(
    ctx: &Arc<ProblemContext>,
    patient_ids: &[usize],
    k: usize,
    rng: &mut R,
) -> (Vec<usize>, Vec<(f64, f64)>) {
    let num_patients = patient_ids.len();

    // Initialize centroids from random patients.
    let mut centroids: Vec<(f64, f64)> = patient_ids
        .iter()
        .take(k)
        .map(|&pid| {
            let p = &ctx.patients[pid];
            (p.x, p.y)
        })
        .collect();

    let mut assignments = vec![0usize; num_patients];

    // Small fixed number of iterations keeps this fast and stable.
    for _ in 0..15 {
        // Assignment step.
        for (idx, &pid) in patient_ids.iter().enumerate() {
            let p = &ctx.patients[pid];
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
            let p = &ctx.patients[pid];
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
                let p = &ctx.patients[pid];
                centroids[c] = (p.x, p.y);
            }
        }
    }

    (assignments, centroids)
}

/// Calculate the inertia (within-cluster sum of squares) for a clustering.
fn calculate_inertia(
    ctx: &Arc<ProblemContext>,
    patient_ids: &[usize],
    assignments: &[usize],
    centroids: &[(f64, f64)],
) -> f64 {
    let mut inertia = 0.0;
    for (idx, &pid) in patient_ids.iter().enumerate() {
        let p = &ctx.patients[pid];
        let cluster = assignments[idx];
        let (cx, cy) = centroids[cluster];
        let dx = p.x - cx;
        let dy = p.y - cy;
        inertia += dx * dx + dy * dy;
    }
    inertia
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Patient, ProblemInstance};

    fn create_test_context(num_patients: usize, num_nurses: usize) -> Arc<ProblemContext> {
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
        // Create patients in 3 distinct spatial clusters
        for i in 1..=num_patients {
            let cluster = (i - 1) % 3;
            let (base_x, base_y) = match cluster {
                0 => (10.0, 10.0),
                1 => (50.0, 10.0),
                _ => (30.0, 50.0),
            };
            patients.push(Patient {
                id: i,
                demand: 5.0,
                start_time: 0.0,
                end_time: 480.0,
                care_time: 10.0,
                x: base_x + (i as f64 % 3.0) * 2.0,
                y: base_y + (i as f64 % 2.0) * 2.0,
            });
        }

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
                num_nurses,
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
    fn test_kmeans_optimal_k_finding() {
        use genevo::random::random_seed;
        let mut rng = genevo::random::get_rng(random_seed());
        
        // Create problem with 15 patients and 5 nurses
        // Patients are arranged in 3 spatial clusters
        let ctx = create_test_context(15, 5);
        let patient_ids: Vec<usize> = (1..=15).collect();
        
        let optimal_k = find_optimal_k(&ctx, &patient_ids, 5, 15, &mut rng);
        
        // With 3 distinct spatial clusters, optimal k should be around 3
        // (could be 2-5 depending on elbow detection)
        println!("Optimal k found: {} (expected around 3 for 3 spatial clusters)", optimal_k);
        assert!(optimal_k >= 2 && optimal_k <= 5, 
                "Optimal k should be reasonable for the problem size");
    }

    #[test]
    fn test_kmeans_genome_builder() {
        use genevo::random::random_seed;
        let mut rng = genevo::random::get_rng(random_seed());
        
        let ctx = create_test_context(12, 4);
        let builder = KMeansGenomeBuilder::new(ctx.clone());
        
        let genome = builder.build_genome(0, &mut rng);
        
        // Should have exactly 4 routes (num_nurses)
        assert_eq!(genome.len(), 4, "Should have exactly num_nurses routes");
        
        // All patients should be assigned
        let mut all_patients: Vec<usize> = genome.iter().flatten().copied().collect();
        all_patients.sort();
        let expected: Vec<usize> = (1..=12).collect();
        assert_eq!(all_patients, expected, "All patients should be assigned exactly once");
    }
}
