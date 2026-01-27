use std::{error::Error, fs::File, iter::once};
use lib_sga::{GenStats, Item, sga};
use csv::StringRecord;
use plotters::prelude::*;


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
    const GENERATIONS: usize = 50;

    let file = String::from("knapsack/knapPI_12_500_1000_82.csv");

    let items: Vec<Item> = read_from_file(&file)?;
    if items.is_empty() {
        return Err(format!("No items read from file {file}").into());
    }

    let (best_individual, gen_stats) = sga(&items, POPULATION_SIZE, CAPACITY, OPTIMAL, GENERATIONS);

    println!("Result of algorithm: {:?}", best_individual.fitness_score);

    plot_fitness_stats(gen_stats, "./plot.png");

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

fn plot_fitness_stats(stats: Vec<GenStats>, out_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_gen = stats.len().saturating_sub(1) as i32;
    let max_fit = stats.iter().map(|s| s.max).max().unwrap_or(1) as i32;

    let mut chart = ChartBuilder::on(&root)
        .caption("SGA Fitness (min / mean / max)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..max_gen, 0..max_fit)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness")
        .draw()?;

    // Min
    chart.draw_series(LineSeries::new(
        stats.iter().enumerate().map(|(g, s)| (g as i32, s.min as i32)), &BLUE,
    ))?
    .label("min")
    .legend(|(x,y)| PathElement::new(vec![(x,y), (x+20, y)], BLUE));

    // Mean
    chart.draw_series(LineSeries::new(
        stats.iter().enumerate().map(|(g, s)| (g as i32, s.mean.round() as i32)), &GREEN,
    ))?
    .label("mean")
    .legend(|(x,y)| PathElement::new(vec![(x,y), (x+20, y)], GREEN));

    // Max
    chart.draw_series(LineSeries::new(
        stats.iter().enumerate().map(|(g, s)| (g as i32, s.max as i32)), &RED,
    ))?
    .label("max")
    .legend(|(x,y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    // Plot the final values (last generation)
    if let Some(last) = stats.last() {
        let g = max_gen;

        // Points
        chart.draw_series(once(Circle::new((g, last.max as i32), 4, RED.filled())))?;

        // Labels
        chart.draw_series(once(Text::new(
            format!("max={}", last.max),
            (g-6, (last.max as i32)-10),
            ("sans-serif", 16).into_font().color(&RED),
        )))?;
    }

    chart.configure_series_labels().border_style(BLACK).draw()?;

    Ok(())
}