use std::{error::Error, fs::File};

use csv::StringRecord;
use rand::{Rng, rng};


mod chromosome;
mod ga;
use ga::GeneticAlgorithm;


fn main() -> Result<(), Box<dyn Error>> {
    
    let file = "feature_selection/dataset.txt".to_string();

    let items = read_from_file(&file)?;
    if items.is_empty() {
        return Err(format!("No items read from file {}", &file).into());
    }

    let expected = items[0].len();
    for (i, row) in items.iter().enumerate() {
        if row.len() != expected {
            return Err(format!("Row {} has {} features, expected {}", i, row.len(), expected).into());
        }
    }

    println!("Data shape {} rows x {} features", items.len(), expected);

    let mut ga = GeneticAlgorithm {
        population_size: 100,
        num_features: 102,
    };

    ga.run();

    Ok(())
}

fn read_from_file(path: &String) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    println!("Reading file {}", &path);
    let file = File::open(path)?;

    let mut readerbuilder = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut items = Vec::new();
    for record in readerbuilder.records() {
        let r: StringRecord = record?;

        let item: Vec<f64> = r
            .iter()
            .map(|f| f.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;        
        items.push(item);
    
    }

    Ok(items)
}