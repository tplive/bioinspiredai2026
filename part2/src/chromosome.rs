use rand::{Rng, rng};

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub genes: Vec<bool>, // true if feature is selected/included
    pub fitness: Option<f64>, // RMSE
}

impl Chromosome {
    pub fn new(number_of_features: usize) -> Self {
        let mut r = rng();
        let genes = (0..number_of_features).map(|_| r.random_bool(0.5)).collect();

        Self {
            genes,
            fitness: None,
        }
    }

    pub fn from_genes(genes: Vec<bool>) -> Self {
        Self {
            genes,
            fitness: None,
        }
    }

    pub fn num_selected(&self) -> usize {
        self.genes.iter().filter(|&&g| g).count()
    }
}