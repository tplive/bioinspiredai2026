use std::slice::IterMut;

use rand::{Rng, rng};

#[derive(Debug)]
pub struct Item {
    pub i: usize,
    pub p: usize,
    pub w: usize,
}

#[derive(Debug, Clone)]
pub struct Individual {
    genome: Vec<bool>, // For each index in the data, "is included in the knapsack" or not
    pub profit: usize, // Calculated profit (sum of items p-values)
    pub weight: usize, // Calculated weight (sum of items w-values)
    pub fitness_score: usize, // Calculated score included penalty if capacity is exceeded
}
impl Individual {
    fn new(n_items: usize) -> Self {
        let mut r = rng();
        let genes = (0..n_items).map(|_| r.random::<bool>()).collect();

        Self {
            genome: genes,
            profit: 0,
            weight: 0,
            fitness_score: 0,
        }
    }

    fn from_genome(genome: Vec<bool>) -> Self {
        Self {
            genome,
            profit: 0,
            weight: 0,
            fitness_score: 0,
        }
    }

    fn fitness(&mut self, items: &[Item], capacity: &usize) {
        self.profit = 0;
        self.weight = 0;
        self.fitness_score = 0;

        for (i, &in_sack) in self.genome.iter().enumerate() {
            if in_sack {
                self.profit += items[i].p;
                self.weight += items[i].w;

                let k: usize = 1;
                if self.weight > *capacity {
                    let d = self.weight - capacity;
                    let penalty = k * d;
                    self.fitness_score = self.profit.saturating_sub(penalty);
                } else {
                    self.fitness_score = self.profit;
                }
            }
        }
    }
}

pub struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    fn new(population_size: usize, items: &[Item]) -> Self {
        let individuals = (0..population_size)
            .map(|_| Individual::new(items.len()))
            .collect();

        Self { individuals }
    }

    fn iter_mut(&mut self) -> IterMut<'_, Individual> {
        self.individuals.iter_mut()
    }
}

pub struct GenStats {
    pub min: usize,
    pub mean: f64,
    pub max: usize,
}


fn tournament_selection(population: &Population, k: usize) -> &Individual {

    let mut r = rng();
    let n = population.individuals.len();

    let mut best_index = r.random_range(0..n);

    for _ in 1..k {
        let idx = r.random_range(0..n);
        if population.individuals[idx].fitness_score > population.individuals[best_index].fitness_score {
            best_index = idx;
        }
    }

    &population.individuals[best_index]
}

fn single_point_crossover(p1: &Individual, p2: &Individual, point: usize) -> (Individual, Individual) {

    let (h1, t1) = p1.genome.split_at(point);
    let (h2, t2) = p2.genome.split_at(point);

    let mut g1 = Vec::with_capacity(p1.genome.len());
    g1.extend_from_slice(h1);
    g1.extend_from_slice(t2);

    let mut g2 = Vec::with_capacity(p2.genome.len());
    g2.extend_from_slice(h2);
    g2.extend_from_slice(t1);

    (Individual::from_genome(g1), Individual::from_genome(g2))
}

fn mutate(individual: &Individual, probability: f64) -> Vec<bool> {
    let mut r = rng();
    let genome = individual.genome.clone();

    let mut mutant = Vec::with_capacity(genome.len());
    for b in genome {
        let flip = r.random_bool(probability);
        mutant.push(if flip { !b } else { b });
    }

    mutant
}

pub fn sga(
    items: &[Item],
    pop_size: usize,
    capacity: usize,
    optimal: usize,
    generations: usize,
) -> (Individual, Vec<GenStats>) {
    // Initialize population
    let mut population = Population::new(pop_size, items);

    // Calculate fitness for the whole population, return the best fit individual
    let mut best_fit_index = 0;
    let mut best_score = 0;

    let mut new_individuals = Vec::<Individual>::with_capacity(pop_size);
    let mut gen_stats = Vec::<GenStats>::with_capacity(generations);

    for _gen in 1..=generations {
        println!("Generation {:?}", _gen);
        
        if _gen > 1 {
            population.individuals = new_individuals.clone();
            new_individuals.clear();
        }

        for (index, i) in population.iter_mut().enumerate() {
            i.fitness(items, &capacity);
            
            if i.fitness_score > best_score && i.profit <= capacity {
                best_fit_index = index;
                best_score = i.fitness_score;
                println!(
                    "New best individual weighs {:?} with {:?} profit (fitness {:?})",
                    i.weight, i.profit, i.fitness_score
                );
            }
        }

        let mut min_fit = usize::MAX;
        let mut max_fit = 0usize;
        let mut sum_fit = 0usize;

        for ind in population.individuals.iter() {
            let f = ind.fitness_score;
            min_fit = min_fit.min(f);
            max_fit = max_fit.max(f);
            sum_fit += f;
        }

        let mean_fit = sum_fit as f64 / population.individuals.len() as f64;
        
        gen_stats.push(GenStats {min: min_fit, mean: mean_fit, max: max_fit});
        
        for _sel in 0..pop_size / 2 {

            // Tournament selection of parents
            let parent1 = tournament_selection(&population, 5);
            let parent2 = tournament_selection(&population, 5);
            
            // Cross over parents genomes
            let crossover_point = parent1.genome.len() / 2;
            
            let (offspring1, offspring2) = single_point_crossover(parent1, parent2, crossover_point);
            
            // Mutate the offspring
            let prob = 0.01;
            let (mutant1, mutant2) = (mutate(&offspring1, prob), mutate(&offspring2, prob));
                        
            // Populate new_individuals with the (possibly mutated) mutants
            new_individuals.extend_from_slice(&[Individual::from_genome(mutant1), Individual::from_genome(mutant2)]);
        }


    }
    
    (population.individuals[best_fit_index].clone(), gen_stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_individual_fitness() {
        let items = [
            Item { i: 0, p: 5, w: 2 },
            Item { i: 1, p: 6, w: 3 },
            Item { i: 2, p: 7, w: 4 },
        ];

        let mut individual1 = Individual {
            genome: vec![true, true, true],
            profit: 0,
            weight: 0,
            fitness_score: 0,
        };

        let capacity = 5;

        individual1.fitness(&items, &capacity);

        assert_eq!(individual1.profit, 18);
        assert_eq!(individual1.weight, 9);
        assert_eq!(individual1.fitness_score, 14);

        let mut individual2 = Individual {
            genome: vec![false, false, false],
            profit: 0,
            weight: 0,
            fitness_score: 0,
        };

        individual2.fitness(&items, &capacity);

        assert_eq!(individual2.profit, 0);
        assert_eq!(individual2.weight, 0);
        assert_eq!(individual2.fitness_score, 0);
    }

    #[test]
    fn test_single_point_crossover() {

        let genome1 = vec![true, true, true, true, true, true, true, true, true, true,];
        let genome2 = vec![false, false, false, false, false, false, false, false, false, false,];

        let i1 = Individual::from_genome(genome1);
        let i2 = Individual::from_genome(genome2);

        let (o1, o2) = single_point_crossover(&i1, &i2, 5);

        assert_eq!(vec![true, true, true, true, true, false, false, false, false, false, ], o1.genome);
        assert_eq!(vec![false, false, false, false, false, true, true, true, true, true, ], o2.genome);
    }

    #[test]
    fn test_mutate() {

        let genome = vec![true, true, true, true, true, true];
        let probability = 0.0;
        let mutant = mutate(&Individual::from_genome(genome), probability);
        assert_eq!(vec![true, true, true, true, true, true], mutant);

        let genome = vec![true, true, true, true, true, true];
        let probability = 1.0;
        let mutant = mutate(&Individual::from_genome(genome), probability);
        assert_eq!(vec![false, false, false, false, false, false], mutant);



    }
}
