/// Local search operators: 2-opt and Or-opt for route optimization.
/// These post-mutation improvements can convert near-feasible solutions to feasible ones
/// and substantially reduce travel cost throughout a GA run.
use crate::fitness::Genome;
use crate::types::ProblemContext;

/// Perform a 2-opt local search pass on all routes of the genome.
///
/// For each route, try all pairs of edges (i, i+1) and (j, j+1) where i < j,
/// and reverse the segment between them if it reduces travel cost.
/// Repeat until no improving moves are found or max_iterations is reached (first-improvement strategy).
///
/// 2-opt is the canonical local search for TSP/VRP: reversing a segment can eliminate
/// crossing edges and reduce tour length, especially after crossover scrambles routes.
pub fn two_opt(genome: &mut Genome, ctx: &ProblemContext) {
    let mat = &ctx.travel_matrix;
    const MAX_ITERATIONS_PER_ROUTE: usize = 3;

    for route in genome.iter_mut() {
        if route.len() < 3 {
            // Routes with fewer than 3 patients cannot benefit from 2-opt
            continue;
        }

        let mut improved = true;
        let mut iterations = 0;

        while improved && iterations < MAX_ITERATIONS_PER_ROUTE {
            improved = false;
            iterations += 1;

            let n = route.len();

            // Try all pairs of edges (i, i+1) and (j, j+1) where i < j-1
            'outer: for i in 0..n - 2 {
                for j in i + 2..n {
                    let cost_before = two_opt_cost_before(route, i, j, mat);
                    let cost_after = two_opt_cost_after(route, i, j, mat);

                    if cost_after < cost_before {
                        // Reverse the segment between i+1 and j
                        route[i + 1..=j].reverse();
                        improved = true;
                        break 'outer;
                    }
                }
            }
        }
    }
}

/// Cost of edges (i, i+1) and (j, j+1) in the current route.
/// Includes connections to/from the depot (0) at start and end.
fn two_opt_cost_before(route: &[usize], i: usize, j: usize, mat: &[Vec<f64>]) -> f64 {
    let depot = 0;
    let n = route.len();

    let a = if i == 0 { depot } else { route[i - 1] };
    let b = route[i];
    let c = route[j];
    let d = if j + 1 == n { depot } else { route[j + 1] };

    mat[a][b] + mat[c][d]
}

/// Cost of edges (i, j) and (i+1, j+1) after reversing segment [i+1..=j].
/// This is the new configuration after the 2-opt reversal.
fn two_opt_cost_after(route: &[usize], i: usize, j: usize, mat: &[Vec<f64>]) -> f64 {
    let depot = 0;
    let n = route.len();

    let a = if i == 0 { depot } else { route[i - 1] };
    let b = route[i];
    let c = route[j];
    let d = if j + 1 == n { depot } else { route[j + 1] };

    // After reversing [i+1..=j], the connections change:
    // Old: a -> b ... c -> d
    // New: a -> c ... b -> d (with segment reversed in between)
    mat[a][c] + mat[b][d]
}

/// Perform an Or-opt local search pass: relocate segments of 1, 2, or 3 patients
/// to better positions within the same route or to other routes.
///
/// Or-opt can fix capacity violations by moving high-demand patients to routes
/// with available capacity, and reduces travel cost by finding better cluster positions.
#[allow(dead_code)]
pub fn or_opt(genome: &mut Genome, ctx: &ProblemContext, _rng: &mut impl rand::Rng) {
    let patients = &ctx.patients;
    let mat = &ctx.travel_matrix;
    let capacity = ctx.instance.capacity;

    // Try segment lengths 1, 2, 3
    for segment_len in 1..=3 {
        let mut improved = true;

        while improved {
            improved = false;

            // Try relocating each segment in each route
            for route_idx in 0..genome.len() {
                let route_len = genome[route_idx].len();

                if route_len < segment_len {
                    continue;
                }

                // Try each starting position for a segment
                for seg_start in 0..=(route_len - segment_len) {
                    let seg_end = seg_start + segment_len - 1;
                    let segment: Vec<usize> = genome[route_idx][seg_start..=seg_end].to_vec();
                    let _seg_demand: i32 = segment.iter().map(|&pid| patients[pid].demand).sum();

                    // Try inserting this segment at all positions in the current route
                    for insert_pos in 0..=route_len.saturating_sub(segment_len) {
                        // Skip if it's essentially the same position (moving to itself)
                        if insert_pos == seg_start || insert_pos == seg_start + 1 {
                            continue;
                        }

                        // Calculate cost delta
                        let mut test_route = genome[route_idx].clone();
                        let adjusted_insert = if insert_pos < seg_start {
                            insert_pos
                        } else {
                            insert_pos - segment_len
                        };

                        // Remove segment
                        let _removed: Vec<_> = test_route.drain(seg_start..=seg_end).collect();
                        // Insert at new position
                        for (i, &pid) in segment.iter().enumerate() {
                            test_route.insert(adjusted_insert + i, pid);
                        }

                        let cost_before = route_travel_cost(&genome[route_idx], mat);
                        let cost_after = route_travel_cost(&test_route, mat);

                        if cost_after < cost_before {
                            genome[route_idx] = test_route;
                            improved = true;
                            break;
                        }
                    }

                    if improved {
                        break;
                    }
                }

                if improved {
                    break;
                }
            }

            // Also try moving segments to other routes (if they fit capacity)
            if !improved {
                for route_idx in 0..genome.len() {
                    let route_len = genome[route_idx].len();
                    if route_len < segment_len {
                        continue;
                    }

                    for seg_start in 0..=(route_len - segment_len) {
                        let seg_end = seg_start + segment_len - 1;
                        let segment: Vec<usize> = genome[route_idx][seg_start..=seg_end].to_vec();
                        let seg_demand: i32 = segment.iter().map(|&pid| patients[pid].demand).sum();

                        // Try moving to each other route
                        for target_route_idx in 0..genome.len() {
                            if target_route_idx == route_idx {
                                continue;
                            }

                            let target_demand: i32 =
                                genome[target_route_idx].iter().map(|&pid| patients[pid].demand).sum();

                            // Check if capacity allows
                            if target_demand + seg_demand > capacity {
                                continue;
                            }

                            // Try all insertion positions in target route
                            for insert_pos in 0..=genome[target_route_idx].len() {
                                let cost_before_source = route_travel_cost(&genome[route_idx], mat);
                                let cost_before_target = route_travel_cost(&genome[target_route_idx], mat);

                                // Remove from source
                                let mut new_source = genome[route_idx].clone();
                                let _removed: Vec<_> = new_source.drain(seg_start..=seg_end).collect();

                                // Add to target
                                let mut new_target = genome[target_route_idx].clone();
                                for (i, &pid) in segment.iter().enumerate() {
                                    new_target.insert(insert_pos + i, pid);
                                }

                                let cost_after_source = route_travel_cost(&new_source, mat);
                                let cost_after_target = route_travel_cost(&new_target, mat);

                                if (cost_after_source + cost_after_target)
                                    < (cost_before_source + cost_before_target)
                                {
                                    genome[route_idx] = new_source;
                                    genome[target_route_idx] = new_target;
                                    improved = true;
                                    break;
                                }
                            }

                            if improved {
                                break;
                            }
                        }

                        if improved {
                            break;
                        }
                    }

                    if improved {
                        break;
                    }
                }
            }
        }
    }
}

/// Calculate the total travel cost of a single route (excluding connections to other routes).
#[allow(dead_code)]
fn route_travel_cost(route: &[usize], mat: &[Vec<f64>]) -> f64 {
    if route.is_empty() {
        return 0.0;
    }

    let depot = 0;
    let mut cost = mat[depot][route[0]];

    for i in 0..route.len() - 1 {
        cost += mat[route[i]][route[i + 1]];
    }

    cost += mat[route[route.len() - 1]][depot];

    cost
}
