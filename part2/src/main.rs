use std::{error::Error, fs::File};

use csv::StringRecord;
use rand::{Rng, rng};

struct Chromosome {
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
}

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

    println!("All rows have {} features", expected);

    
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