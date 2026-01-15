use std::{error::Error, fs::File};

use csv::StringRecord;

#[derive(Debug)]
struct Item {
    i: usize,
    p: usize,
    w: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    
    // Read data from project1/knapsack/knapPI_12_500_1000_82.csv
    // The data format is I(ndex), p(rofit), w(eight)
    // The knapsack capacity is given at 280785 units
    // We need to find the combination of items where 
    // - the profit is highest, given that 
    // - the total weight of items don't exceed 280785


    let file = "knapsack/knapPI_12_500_1000_82.csv".to_string();

    let items: Vec<Item> = read_from_file(&file)?;
    if items.is_empty() {
        return Err(format!("No items read from file {file}").into());
    }

    println!("{:?}", items[0]);

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