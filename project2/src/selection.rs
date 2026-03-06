/// Tournament selection operator with configurable tournament size.
///
/// In each tournament, randomly select `tournament_size` individuals from the
/// population, and the one with the highest fitness wins the tournament.
/// This is repeated to build the parent pool.
///
/// Advantages over truncation selection:
/// - Tunable selection pressure via tournament size
/// - Less sensitive to fitness scaling artifacts
/// - Better maintained diversity in early generations
use genevo::operator::{GeneticOperator, SelectionOp};
use genevo::algorithm::EvaluatedPopulation;
use genevo::random::Rng;
use genevo::prelude::{Fitness, Genotype};

#[derive(Clone, Debug)]
pub struct TournamentSelector {
    /// Number of individuals to sample in each tournament
    pub tournament_size: usize,
    /// Fraction of population to forward to parent pool
    pub selection_ratio: f64,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize, selection_ratio: f64) -> Self {
        Self {
            tournament_size: tournament_size.max(2),
            selection_ratio,
        }
    }
}

impl GeneticOperator for TournamentSelector {
    fn name() -> String {
        "Tournament-Selection".to_string()
    }
}

impl<G: Genotype, F: Fitness + Ord> SelectionOp<G, F> for TournamentSelector {
    fn select_from<R>(&self, population: &EvaluatedPopulation<G, F>, rng: &mut R) -> Vec<Vec<G>>
    where
        R: Rng + Sized,
    {
        // Convert EvaluatedPopulation to a vector we can index
        let pop_vec = population.individuals();
        let pop_len = pop_vec.len();
        let num_to_select = ((pop_len as f64) * self.selection_ratio).ceil() as usize;
        let mut parents = Vec::with_capacity(num_to_select);

        // Run tournaments until we have enough parents
        for _ in 0..num_to_select {
            // Run one tournament: randomly sample tournament_size individuals
            let mut best_idx = rng.gen_range(0..pop_len);
            let mut best_fitness = pop_vec[best_idx].fitness();

            for _ in 1..self.tournament_size {
                let idx = rng.gen_range(0..pop_len);
                let fitness = pop_vec[idx].fitness();

                // Keep the individual with higher fitness (genevo maximizes)
                if fitness > best_fitness {
                    best_idx = idx;
                    best_fitness = fitness;
                }
            }

            parents.push(pop_vec[best_idx].genome().clone());
        }

        vec![parents]
    }
}

