# Fitness Evaluation Test Suite

## Overview
Comprehensive test suite for verifying the correctness of fitness evaluation calculations in the home healthcare routing problem.

## Test Structure

### Test Context
The tests use a simplified problem with:
- **3 patients** with controlled parameters
- **Fixed travel times** (no Euclidean calculations) for easy verification
- **Configurable constraints**: capacity (20.0), depot return time (150.0), penalty factor (100.0)

**Travel Matrix**:
```
     Depot  P1   P2   P3
Depot  0    5    10   15
P1     5    0    5    10
P2     10   5    0    5
P3     15   10   5    0
```

**Patient Parameters**:
| Patient | Demand | Time Window | Care Time |
|---------|--------|-------------|-----------|
| 1       | 5.0    | [10, 50]    | 10        |
| 2       | 8.0    | [30, 80]    | 15        |
| 3       | 12.0   | [60, 120]   | 20        |

## Test Cases

### 1. `test_empty_route`
- **Purpose**: Verify baseline case
- **Expected**: Zero travel, zero penalties, feasible

### 2. `test_single_patient_no_violations`
- **Route**: depot → P1 → depot
- **Timeline**:
  - Depart at 0, arrive at P1 at 5
  - Wait until 10 (window opens)
  - Care from 10 to 20
  - Return to depot at 25
- **Expected**: Travel=10, Penalty=0, Feasible=true

### 3. `test_late_arrival_penalty`
- **Route**: depot → P2 → P1 → depot
- **Key violation**: Care for P1 ends at 60, but window closes at 50
- **Expected**: Penalty for care extending past window = (60-50) * 100 = 1000

### 4. `test_capacity_violation`
- **Route**: depot → P1 → P2 → P3 → depot
- **Total demand**: 5 + 8 + 12 = 25
- **Capacity**: 20
- **Expected**: Capacity penalty = (25-20) * 100 = 500

### 5. `test_late_depot_return`
- **Purpose**: Document behavior when returning after deadline
- **Route**: depot → P3 → P2 → P1 → depot

### 6. `test_waiting_for_time_window`
- **Purpose**: Verify nurse waits if arriving before window opens
- **Route**: depot → P1 → depot
- **Expected**: Arrives at 5, waits until 10 to start care

### 7. `test_multiple_routes_individual`
- **Genome**: [[1], [2]]
- **Expected**: Sum of two separate route costs

### 8. `test_fitness_function_maximization`
- **Purpose**: Verify genevo maximization behavior
- **Expected**: Better routes (less cost) → higher fitness value (less negative)

### 9. `test_fitness_cache`
- **Purpose**: Verify cache returns consistent results
- **Expected**: Same fitness for identical genomes

### 10. `test_penalty_accumulation`
- **Purpose**: Verify multiple penalties are accumulated correctly
- **Expected**: Total penalty includes all violation types

### 11. `test_detailed_route_visits`
- **Purpose**: Verify detailed route computation includes visit information
- **Expected**: Correct arrival times and demands for each patient

### 12. `test_fitness_computation_vs_cache_lookup`
- **Purpose**: Performance comparison between direct computation and cached lookup
- **Note**: Cache overhead may exceed savings for small problems

## Test Results

All 12 tests pass successfully:
```
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured
```

## Fitness Calculation Formula

```
Route Fitness = Travel Time + Penalties

Penalties:
1. Late arrival: (arrival_time - end_window) * penalty_factor  [if positive]
2. Care past window: (care_end_time - end_window) * penalty_factor  [if positive]
3. Capacity violation: (total_demand - capacity) * penalty_factor  [if positive]
4. Late depot return: (return_time - depot_deadline) * penalty_factor  [if positive]

Individual Fitness = Sum of all route fitnesses
```

## Testing Methodology

The tests use **small, hand-calculable examples** where:
- Travel times are fixed integers
- Time windows are simple ranges
- Penalties can be computed manually
- All calculations can be verified by inspection

This approach ensures:
- ✅ Correctness verification
- ✅ Easy debugging
- ✅ Clear documentation of expected behavior
- ✅ Confidence in the implementation

## Code Quality

The fitness functions are well-suited for testing:
- **Pure functions**: No side effects (except caching)
- **Single responsibility**: Each function has a clear purpose
- **Deterministic**: Same input always produces same output
- **Well-documented**: Clear comments and variable names
