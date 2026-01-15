use rand::{Rng, rng};

#[derive(Debug)]
pub struct Item {
    pub i: usize,
    pub p: usize,
    pub w: usize,
}

#[derive(Debug)]
struct Individual {
    items: Vec<bool>, // For each index in the data, "is included in the knapsack" or not
    profit: usize, // Calculated profit (sum of items p-values)
    weight: usize, // Calculated weight (sum of items w-values)
}
impl Individual {
    fn new(n_items: usize) -> Self {
        let mut r = rng();
        let items = (0..n_items).map(|_| r.random::<bool>()).collect();
        
        Self {
            items,
            profit: 0,
            weight: 0,
        }
    }

    fn fitness(&mut self, items: &[Item]) {
        let mut p:usize = 0;
        let mut w:usize = 0;

        for (i, &in_sack) in self.items.iter().enumerate() {
            if in_sack {
                p += items[i].p;
                w += items[i].w;

            }
        }
        self.profit = p;
        self.weight = w;
    }
}


pub fn sga(items: Vec<Item>, pop_size: usize, capacity: usize) -> usize {

    // Initialize population
    let mut population: Vec<Individual> = (0..pop_size)
        .map(|_| {
            Individual::new(items.len())
        }).collect();

    population[0].fitness(&items);

    population[0].profit
}








pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
