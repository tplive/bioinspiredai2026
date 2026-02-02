use std::path::MAIN_SEPARATOR;

use rand::{Rng, seq::IndexedRandom};

use crate::{chromosome::{self, Chromosome}, fitness_evaluator::FitnessEvaluator};

pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub num_features: usize,
    pub max_generations: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub evaluator: FitnessEvaluator,
    pub radiation_levels: f64,
}

impl GeneticAlgorithm {
    pub fn run(&mut self) -> (Chromosome, Vec<f64>) {

        let mut population: Vec<Chromosome> = (0..self.population_size)
            .map(|_| Chromosome::new(self.num_features))
            .collect();
        
        self.evaluate_population(&mut population);
        
        // Run over generations
        let mut best_fitness_history: Vec<f64> = Vec::new();

        for g in 0..self.max_generations {

            let mut offspring = Vec::new();

            for _ in 0..(self.population_size / 2) {
                
                // Tournament selection
                let parent1 = tournament_selection(&population, self.tournament_size);
                let parent2 = tournament_selection(&population, self.tournament_size);

                // Crossover
                let (mut child1, mut child2) = if rand::rng().random_bool(self.crossover_rate) {
                    single_point_crossover(&parent1, &parent2)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                bit_flip_mutation(&mut child1, self.radiation_levels);
                bit_flip_mutation(&mut child2, self.radiation_levels);
                
                offspring.push(child1);
                offspring.push(child2);
            }

            self.evaluate_population(&mut offspring);

            population = offspring;

            let best = population.iter()
            .min_by(|a, b| {
                a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap()
            })
            .unwrap()
            .clone();
        
        best_fitness_history.push(best.fitness.unwrap());
        
        println!("Generation {}: Best RMSE = {:.6}", g+1, best.fitness.unwrap());
    }

    let best = population.iter()
        .min_by(|a, b| {
            a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap()
        })
        .unwrap()
        .clone();

        (best, best_fitness_history)
    }
    
    fn evaluate_population(&self, population: &mut [Chromosome]) {
        for c in population.iter_mut() {
            if c.fitness.is_none() {
                c.fitness = Some(self.evaluator.evaluate(c));
            }
        }
    }
}

fn bit_flip_mutation(chromosome: &mut Chromosome, radiation_levels: f64) {
    let mut rng = rand::rng();

    for gene in chromosome.genes.iter_mut() {
        if rng.random_bool(radiation_levels) {
            *gene = !*gene;
        }
    }
}

fn single_point_crossover(parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
    let mut rng = rand::rng();
    
    let point = rng.random_range(1..parent1.genes.len());

    let mut child1_genes = parent1.genes[..point].to_vec();
    child1_genes.extend_from_slice(&parent2.genes[point..]);

    let mut child2_genes = parent2.genes[..point].to_vec();
    child2_genes.extend_from_slice(&parent1.genes[point..]);

    (
        Chromosome::from_genes(child1_genes),
        Chromosome::from_genes(child2_genes),
    )
}

fn tournament_selection(population: &[Chromosome], tournament_size: usize) -> Chromosome {
    let mut rng = rand::rng();

    let tournament: Vec<&Chromosome> = population
        .choose_multiple(&mut rng, tournament_size)
        .collect();

    tournament.iter()
        .min_by(|a, b| {
            a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap()
        })
        .unwrap()
        .clone()
        .clone()
}