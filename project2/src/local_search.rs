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
    const MAX_ITERATIONS_PER_ROUTE: usize = 10;

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
    let (a, b, c, d) = two_opt_endpoints(route, i, j);
    mat[a][b] + mat[c][d]
}

/// Cost of edges (i, j) and (i+1, j+1) after reversing segment [i+1..=j].
/// This is the new configuration after the 2-opt reversal.
fn two_opt_cost_after(route: &[usize], i: usize, j: usize, mat: &[Vec<f64>]) -> f64 {
    let (a, b, c, d) = two_opt_endpoints(route, i, j);

    // After reversing [i+1..=j], the connections change:
    // Old: a -> b ... c -> d
    // New: a -> c ... b -> d (with segment reversed in between)
    mat[a][c] + mat[b][d]
}

fn two_opt_endpoints(route: &[usize], i: usize, j: usize) -> (usize, usize, usize, usize) {
    let depot = 0;
    let n = route.len();

    let a = if i == 0 { depot } else { route[i - 1] };
    let b = route[i];
    let c = route[j];
    let d = if j + 1 == n { depot } else { route[j + 1] };

    (a, b, c, d)
}
