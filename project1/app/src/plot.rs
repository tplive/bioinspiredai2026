use std::{error::Error, iter::once};

use lib_sga::GenStats;
use plotters::prelude::*;

pub fn plot_fitness_stats(stats: Vec<GenStats>, out_path: &str) -> Result<(), Box<dyn Error>> {
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