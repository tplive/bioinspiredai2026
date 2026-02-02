use std::{error::Error, time::Instant};
use csv::{ReaderBuilder, StringRecord};
use ndarray::{Array1, Array2, s};

mod chromosome;
mod ga;
use ga::GeneticAlgorithm;
use sysinfo::{System, get_current_pid};

mod fitness_evaluator;
use crate::{chromosome::Chromosome, fitness_evaluator::FitnessEvaluator};

fn main() -> Result<(), Box<dyn Error>> {
    let now = Instant::now();
    
    let file = "feature_selection/dataset.txt".to_string();

    let (x, y) = load_dataset(&file);

    // Train / test split at 80/20
    let split_at = (x.nrows() as f64 * 0.8) as usize;

    let x_train = x.slice(s![..split_at, ..]).to_owned();
    let y_train = y.slice(s![..split_at]).to_owned();
    let x_test = x.slice(s![split_at.., ..]).to_owned();
    let y_test = y.slice(s![split_at..]).to_owned();

    let evaluator = FitnessEvaluator::new(x_train, y_train, x_test, y_test);

    // Part 2, task f) Show results with all features selected
    let all_features = vec![true; x.ncols()];
    let baseline = Chromosome::from_genes(all_features);
    let baseline_rmse = evaluator.evaluate(&baseline);

    println!("Task f) Best solution without feature selection:");
    println!("Features in use: {}", baseline.num_selected());
    println!("RMSE: {:.6}", baseline_rmse);
    println!("-----------------------------------");

    let mut ga = GeneticAlgorithm {
        population_size: 300,
        num_features: 101,
        max_generations: 200,
        tournament_size: 3,
        crossover_rate: 0.9,
        radiation_levels: 0.01,
        elite_count: 5,
        evaluator,
    };

    let (best_genes, history) = ga.run();

    println!("Best solution found:");
    println!("Number of features selected: {}", best_genes.num_selected());
    println!("Pop={},Gen={},Tsize={},COrate={},Mrate={},Ecount={}", ga.population_size, ga.max_generations, ga.tournament_size, ga.crossover_rate, ga.radiation_levels, ga.elite_count);
    println!("RMSE: {:.6}", best_genes.fitness.unwrap());
    println!("History of RMSE: {:?}", history);

    // Report memory usage and running time
    let mut system = System::new_all();
    system.refresh_all();
    let process = system.process(get_current_pid().unwrap()).unwrap();
    println!("Memory usage: {:.2} MB", (process.memory() as f64 / 1024.0 / 1024.0) );
    println!("Total running time: {:.2?}", now.elapsed());

    Ok(())
}

fn load_dataset(path: &String) -> (Array2<f64>, Array1<f64>) {

    println!("Reading file {}", &path);
    let mut readerbuilder = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();

    let mut items = Vec::new();

    for record in readerbuilder.records() {
        let r: StringRecord = record.unwrap();

        let item: Vec<f64> = r.iter()
            .map(|f| f.parse::<f64>().unwrap())
            .collect();        
        items.push(item);
    }

    let rows = items.len();
    let cols = items[0].len();

    let x = items.iter()
        .flat_map(|row| row[..cols-1].iter().copied())
        .collect();
    let y = items.iter()
        .map(|row| row[cols-1])
        .collect();

    let x_array = Array2::from_shape_vec((rows, cols - 1), x).unwrap();
    let y_array = Array1::from_vec(y);

    (x_array, y_array)
}
