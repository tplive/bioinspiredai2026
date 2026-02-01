use crate::chromosome::Chromosome;


pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub num_features: usize,
}

impl GeneticAlgorithm {
    pub fn run(&mut self) -> (Chromosome, Vec<f64>) {

        let mut population: Vec<Chromosome> = (0..self.population_size)
            .map(|_| Chromosome::new(self.num_features))
            .collect();
        
        (Chromosome::new(3), vec![0.1, 0.2, 0.55])
    }
}