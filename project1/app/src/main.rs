use std::{error::Error, fs::File};
use lib_sga::sga;
use csv::StringRecord;
use rand::{Rng, rng};

#[derive(Debug)]
struct Item {
    i: usize,
    p: usize,
    w: usize,
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

    const POPULATION_SIZE: usize = 100;

    let file = String::from("knapsack/knapPI_12_500_1000_82.csv");

    let items: Vec<Item> = read_from_file(&file)?;
    if items.is_empty() {
        return Err(format!("No items read from file {file}").into());
    }

    println!("{:?}", items[0]);

    let mut individual = Individual::new(10);
    individual.fitness(&items);
    print!("Profit: {:?}, Weight: {:?}", individual.profit, individual.weight);
    

    let population: Vec<Individual> = (0..POPULATION_SIZE)
        .map(|_| {
            Individual::new(items.len())
        }).collect();
    
    //print!("{:?}", population);

    let result = sga();

    print!("Result of algorithm: {:?}", result);

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