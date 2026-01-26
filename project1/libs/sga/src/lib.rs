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

pub fn sga(
    items: &[Item],
    pop_size: usize,
    capacity: usize,
    optimal: usize,
    generations: usize,
) -> Individual {
    // Initialize population
    let mut population = Population::new(pop_size, items);

    // Calculate fitness for the whole population, return the best fit individual
    let mut best_fit_index = 0;
    let mut best_score = 0;

    for _gen in 1..=generations {
        println!("Generation {:?}", _gen);

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
    }

    population.individuals[best_fit_index].clone()
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
}
