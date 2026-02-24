use plotters::prelude::*;
use crate::config::Config;
use crate::fitness::Genome;
use crate::types::ProblemContext;

// ── Public data type ──────────────────────────────────────────────────────────

/// One record in the fitness history – stored each time the best improves.
pub struct HistoryPoint {
    pub generation: u64,
    pub travel: f64,
    pub penalty: f64,
    pub feasible: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Render a fitness-history chart and write it to `output_path` as a PNG.
///
/// Layout:
///  - Left 820 px: fitness-over-generations chart with axes, step line,
///    benchmark reference line, and a labelled point for the final best.
///  - Right 380 px: configuration and results summary panel.
pub fn save_plot(
    history: &[HistoryPoint],
    cfg: &Config,
    instance_name: &str,
    benchmark: f64,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if history.is_empty() {
        return Ok(());
    }

    const W: u32 = 1200;
    const H: u32 = 650;
    const CHART_W: u32 = 820;

    let root = BitMapBackend::new(output_path, (W, H)).into_drawing_area();
    root.fill(&WHITE)?;

    let (chart_area, legend_area) = root.split_horizontally(CHART_W);

    // ── Y-axis range ──────────────────────────────────────────────────────────
    let costs: Vec<f64> = history.iter().map(|p| p.travel + p.penalty).collect();
    let y_data_min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_data_max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range_min = y_data_min.min(benchmark);
    let y_pad = ((y_data_max - y_range_min) * 0.08).max(1.0);
    let y_min = (y_range_min - y_pad).max(0.0);
    let y_max = y_data_max + y_pad;

    let max_gen = cfg.generations as u64;

    // ── Chart ─────────────────────────────────────────────────────────────────
    let mut chart = ChartBuilder::on(&chart_area)
        .caption(
            format!("GA Fitness History \u{2013} {instance_name}"),
            ("sans-serif", 18),
        )
        .margin(16)
        .x_label_area_size(44)
        .y_label_area_size(72)
        .build_cartesian_2d(0u64..max_gen, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Best cost  (travel + penalty)")
        .axis_desc_style(("sans-serif", 14))
        .draw()?;

    // Step line – best cost (travel + penalty) evolving over generations.
    chart
        .draw_series(LineSeries::new(to_step_series(history, max_gen), &BLUE))?
        .label("Best cost")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Benchmark reference line.
    chart
        .draw_series(LineSeries::new(
            vec![(0u64, benchmark), (max_gen, benchmark)],
            ShapeStyle {
                color: RED.mix(0.55),
                filled: false,
                stroke_width: 1,
            },
        ))?
        .label(format!("Benchmark ({benchmark:.1})"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle {
                    color: RED.mix(0.55),
                    filled: false,
                    stroke_width: 1,
                },
            )
        });

    // Highlighted point and value label for the final best result.
    let last = history.last().unwrap();
    let final_cost = last.travel + last.penalty;
    let final_gen = last.generation;

    chart.draw_series(std::iter::once(Circle::new(
        (final_gen, final_cost),
        6,
        ShapeStyle { color: RED.into(), filled: true, stroke_width: 1 },
    )))?;

    let offset = (max_gen / 60).max(1);
    chart.draw_series(std::iter::once(Text::new(
        format!("{final_cost:.2}"),
        (final_gen + offset, final_cost),
        ("sans-serif", 12).into_font(),
    )))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(&BLACK)
        .draw()?;

    // ── Right panel ───────────────────────────────────────────────────────────
    legend_area.fill(&RGBColor(245, 245, 250))?;

    // Vertical divider line.
    legend_area.draw(&PathElement::new(
        vec![(0i32, 0i32), (0i32, H as i32)],
        RGBColor(210, 210, 220),
    ))?;

    let lbl = ("sans-serif", 12).into_font().color(&RGBColor(110, 110, 120));
    let val = ("sans-serif", 13).into_font();
    let hdr = ("sans-serif", 14).into_font();
    let sep = RGBColor(205, 205, 215);

    // ── Config section ────────────────────────────────────────────────────────
    legend_area.draw(&Text::new("Configuration", (16, 18), hdr.clone()))?;
    draw_hsep(&legend_area, 36, W - CHART_W, sep.clone())?;

    let cfg_rows: &[(&str, String)] = &[
        ("Instance",    instance_name.to_string()),
        ("Population",  cfg.pop_size.to_string()),
        ("Generations", cfg.generations.to_string()),
        ("Selection",   format!("{:.2}", cfg.selection_ratio)),
        ("Crossover",   format!("{:.2}", cfg.crossover_rate)),
        ("Mutation",    format!("{:.2}  ({})", cfg.mutation_rate, cfg.mutation_type)),
        ("Reinsertion", format!("{:.2}", cfg.reinsertion_ratio)),
        ("Penalty ×",   format!("{:.1}", cfg.penalty_factor)),
        ("Init",        cfg.init.clone()),
    ];

    draw_rows(&legend_area, 46, cfg_rows, &lbl, &val)?;

    // ── Results section ───────────────────────────────────────────────────────
    let sep_y = 46 + cfg_rows.len() as i32 * 22 + 6;
    draw_hsep(&legend_area, sep_y, W - CHART_W, sep)?;
    legend_area.draw(&Text::new("Results", (16, sep_y + 8), hdr))?;

    let res_rows: &[(&str, String)] = &[
        ("Benchmark",  format!("{benchmark:.2}")),
        ("Best cost",  format!("{final_cost:.2}")),
        ("Best gen",   format!("{final_gen}")),
        ("Feasible?",  last.feasible.to_string()),
    ];

    draw_rows(&legend_area, sep_y + 26, res_rows, &lbl, &val)?;

    root.present()?;
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Expand sparse improvement history into a step-function series for plotting.
///
/// Each improvement produces a horizontal segment (hold previous value) and
/// then a vertical drop to the new value.  The series is extended to `max_gen`.
fn to_step_series(history: &[HistoryPoint], max_gen: u64) -> Vec<(u64, f64)> {
    let mut pts: Vec<(u64, f64)> = Vec::with_capacity(history.len() * 2 + 2);
    for (i, p) in history.iter().enumerate() {
        let cost = p.travel + p.penalty;
        if i > 0 {
            // Horizontal segment at the previous cost up to this generation.
            let prev = pts.last().unwrap().1;
            pts.push((p.generation, prev));
        }
        pts.push((p.generation, cost));
    }
    // Extend the last value to the end of the run.
    if let Some(&(_, last_cost)) = pts.last() {
        pts.push((max_gen, last_cost));
    }
    pts
}

type Panel<'a> = DrawingArea<BitMapBackend<'a>, plotters::coord::Shift>;

/// Draw a horizontal separator line across the panel.
fn draw_hsep(
    area: &Panel,
    y: i32,
    panel_w: u32,
    color: RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    area.draw(&PathElement::new(
        vec![(8i32, y), (panel_w as i32 - 8, y)],
        color,
    ))?;
    Ok(())
}

/// Draw a column of `(label, value)` text rows starting at `y_start`.
fn draw_rows(
    area: &Panel,
    y_start: i32,
    rows: &[(&str, String)],
    lbl_font: &TextStyle<'static>,
    val_font: &FontDesc<'static>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, (label, value)) in rows.iter().enumerate() {
        let y = y_start + i as i32 * 22;
        area.draw(&Text::new(format!("{label}:"), (16, y), lbl_font.clone()))?;
        area.draw(&Text::new(value.as_str(), (132, y), val_font.clone()))?;
    }
    Ok(())
}

/// Draw the best solution's routes onto a PNG file.
/// Each nurse route gets a distinct colour; the depot is a black square.
pub fn save_route_plot(
    genome: &Genome,
    context: &ProblemContext,
    cfg: &Config,
    best_cost: f64,
    key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = format!("{}_routes_{}.png", context.instance.name, key);

    // ── Canvas ────────────────────────────────────────────────────────────────
    let root = BitMapBackend::new(&output, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let (main_area, legend_area) = root.split_horizontally(860);

    // ── Coordinate bounds ─────────────────────────────────────────────────────
    let all_x = std::iter::once(context.instance.depot_x)
        .chain(context.patients.iter().skip(1).map(|p| p.x));
    let all_y = std::iter::once(context.instance.depot_y)
        .chain(context.patients.iter().skip(1).map(|p| p.y));

    let x_min = all_x.clone().fold(f64::INFINITY, f64::min);
    let x_max = all_x.fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_y.clone().fold(f64::INFINITY, f64::min);
    let y_max = all_y.fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min) * 0.07 + 1.0;
    let y_pad = (y_max - y_min) * 0.07 + 1.0;

    // ── Chart ─────────────────────────────────────────────────────────────────
    let mut chart = ChartBuilder::on(&main_area)
        .caption(
            format!("Route map – {} (cost {:.2})", context.instance.name, best_cost),
            ("sans-serif", 22).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (x_min - x_pad)..(x_max + x_pad),
            (y_min - y_pad)..(y_max + y_pad),
        )?;

    chart
        .configure_mesh()
        .x_desc("X coordinate")
        .y_desc("Y coordinate")
        .draw()?;

    // ── Colour palette (HSL rainbow, one hue per nurse) ───────────────────────
    let n_nurses = genome.len();
    let colour_for = |i: usize| -> RGBColor {
        let hue = (i as f64 / n_nurses as f64) * 360.0;
        hsl_to_rgb(hue, 0.75, 0.45)
    };

    let depot_x = context.instance.depot_x;
    let depot_y = context.instance.depot_y;

    // ── Draw routes ───────────────────────────────────────────────────────────
    for (nurse_idx, route) in genome.iter().enumerate() {
        if route.is_empty() {
            continue;
        }
        let colour = colour_for(nurse_idx);

        // Build full path: depot → patients → depot
        let mut path: Vec<(f64, f64)> = Vec::with_capacity(route.len() + 2);
        path.push((depot_x, depot_y));
        for &pid in route {
            if pid < context.patients.len() {
                let p = &context.patients[pid];
                path.push((p.x, p.y));
            }
        }
        path.push((depot_x, depot_y));

        // Route line
        chart.draw_series(LineSeries::new(path.clone(), colour.stroke_width(2)))?;

        // Patient dots
        chart.draw_series(
            route.iter().filter_map(|&pid| {
                if pid < context.patients.len() {
                    Some(Circle::new((context.patients[pid].x, context.patients[pid].y), 4, colour.filled()))
                } else {
                    None
                }
            }),
        )?;

        // Patient ID labels
        chart.draw_series(
            route.iter().filter_map(|&pid| {
                if pid < context.patients.len() {
                    let p = &context.patients[pid];
                    Some(Text::new(
                        format!("{}", pid),
                        (p.x + 0.4, p.y + 0.4),
                        ("sans-serif", 9).into_font().color(&colour.mix(0.8)),
                    ))
                } else {
                    None
                }
            }),
        )?;
    }

    // ── Depot marker ──────────────────────────────────────────────────────────
    chart.draw_series(std::iter::once(
        Rectangle::new(
            [
                (depot_x - 0.8, depot_y - 0.8),
                (depot_x + 0.8, depot_y + 0.8),
            ],
            BLACK.filled(),
        ),
    ))?;
    chart.draw_series(std::iter::once(Text::new(
        "Depot",
        (depot_x + 1.0, depot_y + 1.0),
        ("sans-serif", 11).into_font().color(&BLACK),
    )))?;

    // ── Legend panel ──────────────────────────────────────────────────────────
    legend_area.fill(&RGBColor(245, 245, 245))?;

    let small_font = ("sans-serif", 11).into_font();

    let mut y = 30i32;
    let lx = 20i32;

    legend_area.draw(&Text::new(
        "Hyperparameters",
        (lx, y),
        ("sans-serif", 15).into_font().color(&BLACK),
    ))?;
    y += 24;

    let params = vec![
        format!("Population:    {}", cfg.pop_size),
        format!("Generations:   {}", cfg.generations),
        format!("Crossover:     {:.2}", cfg.crossover_rate),
        format!("Mutation:      {:.2}", cfg.mutation_rate),
        format!("Selection:     {:.2}", cfg.selection_ratio),
        format!("Reinsertion:   {:.2}", cfg.reinsertion_ratio),
        format!("Penalty:       {:.1}", cfg.penalty_factor),
        format!("Nurses:        {}", context.instance.num_nurses),
        format!("Patients:      {}", context.patients.len() - 1),
    ];
    for line in &params {
        legend_area.draw(&Text::new(line.as_str(), (lx, y), small_font.clone()))?;
        y += 18;
    }

    y += 16;
    legend_area.draw(&Text::new(
        "Route colours",
        (lx, y),
        ("sans-serif", 14).into_font().color(&BLACK),
    ))?;
    y += 22;

    for (i, route) in genome.iter().enumerate() {
        if route.is_empty() {
            continue;
        }
        if y > 860 {
            break; // avoid overflow for large instances
        }
        let c = colour_for(i);
        // Colour swatch
        legend_area.draw(&Rectangle::new(
            [(lx, y - 10), (lx + 18, y + 2)],
            ShapeStyle { color: c.to_rgba(), filled: true, stroke_width: 0 },
        ))?;
        legend_area.draw(&Text::new(
            format!("Nurse {:>2}  ({} pts)", i + 1, route.len()),
            (lx + 24, y - 9),
            small_font.clone(),
        ))?;
        y += 17;
    }

    root.present()?;
    println!("Route plot saved → {output}");
    Ok(())
}

// ── HSL → RGB helper ──────────────────────────────────────────────────────────

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> RGBColor {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r1, g1, b1) = match h as u32 {
        0..=59   => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _        => (c, 0.0, x),
    };
    RGBColor(
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}
