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
}
impl Individual {
    fn new(n_items: usize) -> Self {
        let mut r = rng();
        let genes = (0..n_items).map(|_| r.random::<bool>()).collect();
        
        Self {
            genome: genes,
            profit: 0,
            weight: 0,
        }
    }

    fn fitness(&mut self, items: &[Item]) {
        let mut p:usize = 0;
        let mut w:usize = 0;

        for (i, &in_sack) in self.genome.iter().enumerate() {
            if in_sack {
                p += items[i].p;
                w += items[i].w;

            }
        }
        self.profit = p;
        self.weight = w;
    }
}


pub fn sga(items: &[Item], pop_size: usize, capacity: usize) -> Individual {

    // Initialize population
    let mut population: Vec<Individual> = (0..pop_size)
        .map(|_| {
            Individual::new(items.len())
        }).collect();
    
    // Calculate fitness for the whole population, return the best fit individual
    let mut best_fit_index = 0;
    let mut best_profit = 0;

    for (index, i) in population.iter_mut().enumerate() {
        i.fitness(items);
        if i.profit > best_profit && i.profit <= capacity {
            best_fit_index = index;
            best_profit = i.profit;
        }
    }
    
    population[best_fit_index].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_individual_fitness() {

        let items = [
            Item {i: 0, p: 5, w: 2},
            Item {i: 1, p: 6, w: 3},
            Item {i: 2, p: 7, w: 4},
        ];

        let mut individual1 = Individual {
            genome: vec![true, true, true,],
            profit: 0,
            weight: 0,
        };

        individual1.fitness(&items);

        assert_eq!(individual1.profit, 18);
        assert_eq!(individual1.weight, 9);

        let mut individual2 = Individual {
            genome: vec![false, false, false,],
            profit: 0,
            weight: 0,
        };

        individual2.fitness(&items);

        assert_eq!(individual2.profit, 0);
        assert_eq!(individual2.weight, 0);
    }    
}
