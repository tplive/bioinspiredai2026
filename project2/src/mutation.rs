use genevo::operator::{GeneticOperator, MutationOp};
use genevo::random::Rng;
use std::sync::Arc;

use crate::fitness::Genome;
use crate::local_search;
use crate::types::ProblemContext;

// ── Mutation type ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum MutationType {
    /// Swap two random patients within the same route (intra-route).
    Swap,
    /// Relocate one patient to follow another within the same route (intra-route).
    Insert,
}

// ── Main operator struct ──────────────────────────────────────────────────────

/// Combined intra-route and inter-route mutation for the nurse scheduling VRP.
///
/// With probability `mutation_rate`, a genome is mutated.  
/// When mutated, with probability 0.5 an **inter-route move** is applied
/// (a patient moves from one route to another); otherwise an **intra-route**
/// mutation (`Swap` or `Insert`) is applied to a randomly chosen route.
///
/// After mutation, a 2-opt local search pass improves the solution by
/// eliminating crossing edges and reducing tour length.
///
#[derive(Clone, Debug)]
pub struct NurseMutation {
    /// Probability that any given individual is mutated.
    pub mutation_rate: f64,
    pub mutation_type: MutationType,
    pub ctx: Arc<ProblemContext>,
}

impl NurseMutation {
    pub fn new(mutation_rate: f64, mutation_type: MutationType, ctx: Arc<ProblemContext>) -> Self {
        Self { mutation_rate, mutation_type, ctx }
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
        if rng.r#gen::<f64>() >= self.mutation_rate {
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
        local_search::two_opt(&mut genome, &self.ctx);

        genome
    }
}

// ── Intra-route mutations ─────────────────────────────────────────────────────

/// Apply an intra-route mutation to a randomly-selected route.
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

/// Swap two distinct random patients within a single route.
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

/// Remove the patient at `pos2` and re-insert it right after `pos1`.
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

// ── Inter-route mutation ──────────────────────────────────────────────────────

/// Move one random patient from one route to a different random route.
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
