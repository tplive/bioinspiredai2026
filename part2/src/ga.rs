use crate::{chromosome::Chromosome, fitness_evaluator::FitnessEvaluator};


pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub num_features: usize,
    pub evaluator: FitnessEvaluator,
}

impl GeneticAlgorithm {
    pub fn run(&mut self) -> (Chromosome, Vec<f64>) {

        let mut population: Vec<Chromosome> = (0..self.population_size)
            .map(|_| Chromosome::new(self.num_features))
            .collect();
        
        self.evaluate_population(&mut population);

        let mut best_fitness_history: Vec<f64> = Vec::new();

        // TODO Run over generations

        let best = population.iter()
            .min_by(|a, b| {
                a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap()
            })
            .unwrap()
            .clone();

        best_fitness_history.push(best.fitness.unwrap());

        println!("Generation {}: Best RMSE = {:?}", 0, best.fitness.unwrap());
        
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