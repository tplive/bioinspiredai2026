use genevo::operator::{GeneticOperator, MutationOp};
use genevo::random::Rng;
use std::sync::{Arc, Mutex};

use crate::fitness::Genome;
use crate::local_search::two_opt;
use crate::types::ProblemContext;

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum MutationType {
    // Swap two random patients within the same route (intra-route).
    Swap,
    // Relocate one patient to follow another within the same route (intra-route).
    Insert,
}

// With probability `mutation_rate`, a genome is mutated.
// When mutated, with probability 0.5 an "inter-route move" is applied
// Otherwise an "intra-route" mutation (`Swap` or `Insert`) is applied 
// to a randomly chosen route.
//
// After mutation, a 2-opt local search pass improves the solution by
// eliminating crossing edges and reducing tour length.
//
// The mutation rate is adaptive: it increases when population diversity is low
// (exploitation → exploration boost) and decreases when diversity is high
// (preserve good solutions).
#[derive(Clone, Debug)]
pub struct NurseMutation {
    // Shared, mutable mutation rate that can be updated by the GA loop.
    pub mutation_rate: Arc<Mutex<f64>>,
    pub mutation_type: MutationType,
    pub ctx: Arc<ProblemContext>,
}

impl NurseMutation {
    pub fn new(initial_mutation_rate: f64, mutation_type: MutationType, ctx: Arc<ProblemContext>) -> Self {
        Self {
            mutation_rate: Arc::new(Mutex::new(initial_mutation_rate)),
            mutation_type,
            ctx,
        }
    }
}

impl GeneticOperator for NurseMutation {
    fn name() -> String {
        "Nurse-Scheduling-Mutation".to_string()
    }
}

impl MutationOp<Genome> for NurseMutation {
    fn mutate<R>(&self, mut genome: Genome, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let current_rate = *self.mutation_rate.lock().unwrap();

        if rng.r#gen::<f64>() >= current_rate {
            return genome; // no mutation this round
        }

        if rng.r#gen::<f64>() > 0.5 {
            // Inter-route move: move a random patient from one route to another.
            inter_move(&mut genome, rng);
        } else {
            // Intra-route mutation on a randomly chosen non-empty route.
            intra_mutate(&mut genome, &self.mutation_type, rng);
        }

        // Apply 2-opt local search to improve the mutated solution
        two_opt(&mut genome, &self.ctx);

        genome
    }
}

// Apply an intra-route mutation to a randomly-selected route.
fn intra_mutate<R: Rng + Sized>(genome: &mut Genome, mutation_type: &MutationType, rng: &mut R) {
    // Collect indices of non-trivial routes (>1 patient).
    let candidates: Vec<usize> = genome
        .iter()
        .enumerate()
        .filter(|(_, r)| r.len() > 1)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return;
    }

    let ri = candidates[rng.gen_range(0..candidates.len())];

    match mutation_type {
        MutationType::Swap => swap_mutation(&mut genome[ri], rng),
        MutationType::Insert => insert_mutation(&mut genome[ri], rng),
    }
}

// Swap two distinct random patients within a single route.
fn swap_mutation<R: Rng + Sized>(route: &mut [usize], rng: &mut R) {
    let len = route.len();
    if len < 2 {
        return;
    }
    let pos1 = rng.gen_range(0..len);
    let mut pos2 = rng.gen_range(0..len);
    while pos2 == pos1 {
        pos2 = rng.gen_range(0..len);
    }
    route.swap(pos1, pos2);
}

// Remove the patient at `pos2` and re-insert it right after `pos1`.
fn insert_mutation<R: Rng + Sized>(route: &mut Vec<usize>, rng: &mut R) {
    let len = route.len();
    if len < 2 {
        return;
    }
    let pos1 = rng.gen_range(0..len);
    let mut pos2 = rng.gen_range(0..len);
    while pos2 == pos1 {
        pos2 = rng.gen_range(0..len);
    }

    if pos2 == pos1 + 1 {
        // Already adjacent – nothing to do.
        return;
    }

    let patient = route.remove(pos2);

    // After removal, pos1 may shift by one if pos2 < pos1.
    let insert_after = if pos2 < pos1 { pos1 - 1 } else { pos1 };
    let insert_at = (insert_after + 1).min(route.len());
    route.insert(insert_at, patient);
}

// Move one random patient from one route to a different random route.
fn inter_move<R: Rng + Sized>(genome: &mut Genome, rng: &mut R) {
    // Find two distinct route indices; skip empty source routes.
    let non_empty: Vec<usize> = genome
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .map(|(i, _)| i)
        .collect();

    if non_empty.len() < 2 {
        return; // not enough routes to perform an inter-route move
    }

    let src_pos = rng.gen_range(0..non_empty.len());
    let mut dst_pos = rng.gen_range(0..non_empty.len());
    while dst_pos == src_pos {
        dst_pos = rng.gen_range(0..non_empty.len());
    }

    let src = non_empty[src_pos];
    let dst = non_empty[dst_pos];

    // Remove one random patient from the source route.
    let src_len = genome[src].len();
    let patient_idx = rng.gen_range(0..src_len);
    let patient = genome[src].remove(patient_idx);

    // Insert at random position
    let insert_at = rng.gen_range(0..=genome[dst].len());
    genome[dst].insert(insert_at, patient);
}

// Calculate population diversity as the average Hamming distance between all pairs of individuals.
// Diversity ranges from 0 (all identical) to ~1 (maximum dissimilarity).
pub fn calculate_population_diversity(population: &[Genome]) -> f64 {
    if population.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut pair_count = 0;

    for i in 0..population.len() {
        for j in i + 1..population.len() {
            // Hamming distance: count positions where the two genomes differ
            let distance = genome_hamming_distance(&population[i], &population[j]);
            total_distance += distance;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        return 0.0;
    }

    // Normalize by the maximum possible distance (all patients in different routes)
    // and the number of pairs
    total_distance / pair_count as f64
}

// Calculate Hamming distance between two genomes.
// Includes BOTH route assignments AND sequence positions, so intra-route mutations
// (swaps, inserts) register as diversity changes.
fn genome_hamming_distance(genome1: &Genome, genome2: &Genome) -> f64 {
    // Track both route and position: patient_id -> (route_index, position_in_route)
    let mut assignment1: Vec<(usize, usize)> = vec![(0, 0); 1000]; // patient IDs up to ~1000
    let mut assignment2: Vec<(usize, usize)> = vec![(0, 0); 1000];

    // Build assignment map for genome1
    for (route_idx, route) in genome1.iter().enumerate() {
        for (pos_in_route, &patient) in route.iter().enumerate() {
            if patient < assignment1.len() {
                assignment1[patient] = (route_idx, pos_in_route);
            }
        }
    }

    // Build assignment map for genome2
    for (route_idx, route) in genome2.iter().enumerate() {
        for (pos_in_route, &patient) in route.iter().enumerate() {
            if patient < assignment2.len() {
                assignment2[patient] = (route_idx, pos_in_route);
            }
        }
    }

    // Count differences: patient differs if EITHER route OR position is different
    let mut differences = 0;
    let mut total = 0;

    for i in 0..assignment1.len() {
        if assignment1[i] != assignment2[i] {
            differences += 1;
        }
        // Only count patients that exist in at least one genome
        if assignment1[i].0 > 0 || assignment2[i].0 > 0 {
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    differences as f64 / total as f64
}

// Update the mutation rate based on population diversity.
pub fn update_mutation_rate(
    mutation_op: &NurseMutation,
    population_diversity: f64,
    baseline_rate: f64,
) {
    let target_rate = if population_diversity < 0.2 {
        // Very low diversity: INCREASE mutation for more exploration
        baseline_rate + 0.3
    } else if population_diversity < 0.4 {
        // Low diversity: moderately increase mutation
        baseline_rate + 0.15
    } else if population_diversity > 0.65 {
        // High diversity: DECREASE mutation to exploit good solutions
        (baseline_rate - 0.2).max(0.05) // reduce by 0.2, but keep minimum at 0.05
    } else if population_diversity > 0.5 {
        // Medium-high diversity: slightly decrease mutation
        baseline_rate - 0.08
    } else {
        // Medium diversity [0.4, 0.5]: keep baseline
        baseline_rate
    };

    // Transition 15% toward target per update (faster adaptation)
    let current = *mutation_op.mutation_rate.lock().unwrap();
    let new_rate = current + 0.15 * (target_rate - current);

    // Clamp between 0.01 and 0.9 to allow high exploration when needed
    let clamped_rate = new_rate.clamp(0.01, 0.9);
    *mutation_op.mutation_rate.lock().unwrap() = clamped_rate;
}

