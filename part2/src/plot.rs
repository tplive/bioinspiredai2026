use std::{error::Error, iter::once};

use plotters::prelude::*;

pub fn plot_fitness_histories(
    histories: Vec<Vec<f64>>,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_gen = histories
        .iter()
        .map(|h| h.len())
        .max()
        .unwrap_or(1)as i32;

    let mut min_fit = f64::INFINITY;
    let mut max_fit = f64::NEG_INFINITY;
    for h in &histories {
        for &v in h {
            if v < min_fit {
                min_fit = v;
            }
            if v > max_fit {
                max_fit = v;
            }
        }
    }

    if !min_fit.is_finite() || !max_fit.is_finite() || min_fit == max_fit {
        min_fit = 0.0;
        max_fit = 1.0;
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Grid Search Fitness Histories", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..max_gen, min_fit..max_fit)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness (RMSE)")
        .draw()?;

    let colors = vec![&BLUE, &GREEN, &RED, &MAGENTA, &CYAN, &BLACK, &YELLOW];

    for (i, history) in histories.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart.draw_series(LineSeries::new(
            history.iter().enumerate().map(|(g, &fit)| (g as i32, fit)),
            color,
        ))?;
    }

    Ok(())
}
