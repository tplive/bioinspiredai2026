use std::fs;
use serde_json::Value;
use crate::types::{Patient, ProblemContext, ProblemInstance};

/// Load and parse a problem instance JSON file.
///
/// The JSON format follows this format:
/// - `nbr_nurses`, `capacity_nurse`, `benchmark`, `depot`, `patients`, `travel_times`
///
/// `penalty_factor` scales the time-window and capacity violation penalties.
pub fn load_problem(file_path: &str, penalty_factor: f64) -> ProblemContext {
    
    let json_str = fs::read_to_string(file_path)
        .unwrap_or_else(|e| panic!("Cannot read '{file_path}': {e}"));
    
    let json: Value = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Cannot parse '{file_path}' as JSON: {e}"));

    // ── Instance metadata ────────────────────────────────────────────────────
    let name = json["instance_name"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    
    let num_nurses = json["nbr_nurses"].as_u64().expect("nbr_nurses") as usize;
    
    let capacity = json["capacity_nurse"].as_i64().expect("capacity_nurse") as i32;
    
    let benchmark = json["benchmark"].as_f64().unwrap_or(f64::INFINITY);
    
    let depot_return_time = json["depot"]["return_time"]
        .as_f64()
        .expect("depot.return_time");
    
    let depot_x = json["depot"]["x_coord"].as_f64().expect("depot.x_coord");
    
    let depot_y = json["depot"]["y_coord"].as_f64().expect("depot.y_coord");

    // ── Patients ─────────────────────────────────────────────────────────────
    let patients_json = json["patients"]
        .as_object()
        .expect("patients must be a JSON object");

    let mut patients: Vec<Patient> = patients_json
        .iter()
        .map(|(k, v)| {
            let id: usize = k.parse().unwrap_or_else(|_| panic!("patient id '{k}' not numeric"));
            Patient {
                id,
                demand: v["demand"].as_i64().expect("demand") as i32,
                start_time: v["start_time"].as_f64().expect("start_time"),
                end_time: v["end_time"].as_f64().expect("end_time"),
                care_time: v["care_time"].as_f64().expect("care_time"),
                x: v["x_coord"].as_f64().expect("x_coord"),
                y: v["y_coord"].as_f64().expect("y_coord"),
            }
        })
        .collect();

    // Sort by ID so that patients[i].id == i for 1-based indexing.
    patients.sort_by_key(|p| p.id);

    // Prepend "the depot" at index 0 so that `patients[id]` works
    // directly with the 1-based patient IDs used everywhere else in the code.
    // (inherited from Julia 1-based index)
    let mut patients_indexed = vec![Patient {
        id: 0,
        demand: 0,
        start_time: 0.0,
        end_time: f64::MAX,
        care_time: 0.0,
        x: depot_x,
        y: depot_y,
    }];
    
    patients_indexed.extend(patients);

    // ── Travel matrix ─────────────────────────────────────────────────────────
    let travel_times_raw = json["travel_times"]
        .as_array()
        .expect("travel_times must be a JSON array");
    
    let n = travel_times_raw.len();
    
    let travel_matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    travel_times_raw[i][j]
                        .as_f64()
                        .unwrap_or_else(|| panic!("travel_times[{i}][{j}] is not a number"))
                })
                .collect()
        })
        .collect();

    let instance = ProblemInstance {
        name,
        num_nurses,
        capacity,
        benchmark,
        depot_return_time,
        depot_x,
        depot_y,
    };

    ProblemContext {
        instance,
        patients: patients_indexed,
        travel_matrix,
        penalty_factor,
    }
}
