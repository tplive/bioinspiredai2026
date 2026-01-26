use std::{error::Error, fs::File};
use lib_sga::{Item, sga};
use csv::StringRecord;


fn main() -> Result<(), Box<dyn Error>> {
    
    // Read data from project1/knapsack/knapPI_12_500_1000_82.csv
    // The data format is I(ndex), p(rofit), w(eight)
    // The knapsack capacity is given at 280785 units.
    // Optimal solution is 296735.
    // We need to find the combination of items where 
    // - the profit is highest, given that 
    // - the total weight of items don't exceed 280785
    const CAPACITY: usize = 280785;
    const OPTIMAL: usize = 296735;

    // Hyperparameters
    const POPULATION_SIZE: usize = 100;
    const GENERATIONS: usize = 1;

    let file = String::from("knapsack/knapPI_12_500_1000_82.csv");

    let items: Vec<Item> = read_from_file(&file)?;
    if items.is_empty() {
        return Err(format!("No items read from file {file}").into());
    }

    let best_individual = sga(&items, POPULATION_SIZE, CAPACITY, OPTIMAL, GENERATIONS);

    println!("Result of algorithm: {:?}", best_individual.fitness_score);

    Ok(())
}

fn read_from_file(path: &String) -> Result<Vec<Item>, Box<dyn Error>> {
    println!("Reading file...");
    let file = File::open(path)?;

    let mut readerbuilder = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut items = Vec::new();
    for record in readerbuilder.records() {
        let r: StringRecord = record?;
        let i: usize = r[0].trim().parse().unwrap();
        let p: usize = r[1].trim().parse().unwrap();
        let w: usize = r[2].trim().parse().unwrap();

        items.push(Item {i, p, w});
    
    }

    Ok(items)
}