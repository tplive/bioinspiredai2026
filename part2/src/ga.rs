use crate::chromosome::Chromosome;


pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub num_features: usize,
}

impl GeneticAlgorithm {
    pub fn run(&mut self) -> (Chromosome, Vec<f64>) {

        
        (Chromosome::new(3), vec![0.1, 0.2, 0.55])
    }
}