use plotters::prelude::*;

use crate::config::Config;

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
