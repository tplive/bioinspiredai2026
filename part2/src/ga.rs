use rand::{Rng, seq::IndexedRandom};

use crate::{chromosome::{Chromosome}, fitness_evaluator::FitnessEvaluator};

pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub num_features: usize,
    pub max_generations: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elite_count: usize,
    pub evaluator: FitnessEvaluator,
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
                    uniform_crossover(&parent1, &parent2)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                bit_flip_mutation(&mut child1, self.mutation_rate);
                bit_flip_mutation(&mut child2, self.mutation_rate);
                
                offspring.push(child1);
                offspring.push(child2);
            }

            self.evaluate_population(&mut offspring);

            //population = elitism_selection(&mut population, &offspring, self.elite_count);
            //population = deterministic_crowding(&population, &offspring);
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

fn elitism_selection(
    population: &mut [Chromosome], 
    offspring: &[Chromosome], 
    elite_count: usize,) -> Vec<Chromosome> {
    
    population.sort_by(|a, b| {
        a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap()
    });

    let mut new_population: Vec<Chromosome> = population.iter()
        .take(elite_count)
        .cloned()
        .collect();

    new_population.extend_from_slice(&offspring[..(population.len() - elite_count)]);

    new_population
}

fn deterministic_crowding(parents: &[Chromosome], offspring: &[Chromosome]) -> Vec<Chromosome> {

    let mut survivors = Vec::new();

    for i in (0..parents.len()).step_by(2) {
        let p1 = &parents[i];
        let p2 = &parents[i + 1];
        let o1 = &offspring[i];
        let o2 = &offspring[i + 1];

        let d11 = p1.hamming_distance(o1);
        let d12 = p1.hamming_distance(o2);
        let d21 = p2.hamming_distance(o1);
        let d22 = p2.hamming_distance(o2);

        if d11 + d22 < d12 + d21 {
            survivors.push(if o1.fitness.unwrap() < p1.fitness.unwrap() {
                o1.clone()
            } else {
                p1.clone()
            });
            survivors.push(if o2.fitness.unwrap() < p2.fitness.unwrap() {
                o2.clone()
            } else {
                p2.clone()
            });
        } else {
            survivors.push(if o2.fitness.unwrap() < p1.fitness.unwrap() {
                o2.clone()
            } else {
                p1.clone()
            });
            survivors.push(if o1.fitness.unwrap() < p2.fitness.unwrap() {
                o1.clone()
            } else {
                p2.clone()
            });
        }
    }

    survivors
}


fn bit_flip_mutation(chromosome: &mut Chromosome, radiation_levels: f64) {
    let mut rng = rand::rng();

    for gene in chromosome.genes.iter_mut() {
        if rng.random_bool(radiation_levels) {
            *gene = !*gene;
        }
    }
}

fn uniform_crossover(parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
    let mut rng = rand::rng();

    let child1_genes: Vec<bool> = parent1.genes.iter()
        .zip(parent2.genes.iter())
        .map(|(&g1, &g2)| if rng.random_bool(0.5) { g1} else { g2 })
        .collect();

    let child2_genes: Vec<bool> = parent1.genes.iter()
        .zip(parent2.genes.iter())
        .map(|(&g1, &g2)| if rng.random_bool(0.5) { g2} else { g1 })
        .collect();

    (
        Chromosome::from_genes(child1_genes), 
        Chromosome::from_genes(child2_genes),
    )
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
        .map(|c| (**c).clone())
        .unwrap()
}