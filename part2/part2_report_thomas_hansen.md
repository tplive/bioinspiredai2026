# IT3708 Bioinspired Artificial Intelligence
## Project 1 Feature Selection With Genetic Algorithms - Part 2 Feature Selection (group work)

Date: 01.02.2026

The group has consisted of:

* Thomas Hansen [\<thomasq@stud.ntnu.no\>](mailto:thomasq@stud.ntnu.no)

## Project Setup
This project builds on Part 1, individual work. I have opted to use Rust, as this is my primary language of choice. I chose to refactor most of the code from Part 1 due to readability and scalability, aligning more closely with the structure of comparable code in Python or Julia, as well as generally learning more Rust constructs and techniques.

## Overview
The `src/main.rs` file is the entrypoint. It uses the genetic algorithm from `src/ga.rs`. Starting out, I built the scaffolding code such that running the GA would produce a "mock" Chromosome and result.

I implemented the fitness and machine learning function with Linear Regression from the supplied Julia and Python code (see `src/fitness_evaluator`), soliciting some advice from Github Copilot for pointers (no pun intended), but ultimately writing all the Rust code myself.

I ducked [1] for "Machine Learning with Rust" which led me to discover the `linfa` crate [2], which "aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.", a scikit-learn equivalent for Rust. I decided to go with it for this project, despite its experimental status. I also decided to use the `ndarray` crate for supplying n-dimensional arrays, similar to well-known functionality in Python Numpy.

I have not focused intensely on performance and stability for this project. For instance, I use a lot of `.unwrap()` statements which essentially short-circuit error handling and can lead to panics if data doesn't match. In a production environment with unknown data, this must be handled, but works fine in this scenario.

Also, there is some use of object cloning that can possibly be solved with less resource utilization. Running the compiler with the release profile (`cargo run --profile release`) massively improves execution time, and profiling shows a mere ~50-120MB memory used.

## Parsing the dataset
The dataset is a csv file shaped 1994x102. We aim to find the combination of features that will minimize RMSE when doing linear regression over the data. I was able to refit the `read_from_file` function from Part 1, and utilize `ndarray` for creating n-dimensional arrays.

## Chromosome
A Chromosome in this function is the list of booleans representing genes, 0=off, 1=on. A Chromosome for this dataset will have 102 genes, one for each feature. By turning them on and off we aim to remove the datapoints that cause excessive error, while still training a good model. Training on a randomized population of chromosomes will yield some result for which we can calculate RMSE.

## Calculate RMSE

$$ \text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}} $$

In Rust code, the `rmse()` function is implemented like this:
1. Create an iterator over the predictions (`.iter()`)
2. Pair (`.zip()`) the predicted and actual values
3. Subtract actual value from the prediction using map and square it
4. Sum all squared errors
5. Take the square root of sum divided by number of values

## Putting it together
The scaffolding is done when:

a) The Chromosome maps active/inactive genes
b) RMSE (fitness) is calculated for a given set of genes
c) The GA can run over several generations

At this point, I'm seeing the same result every generation as there is no selection done, but a new random Chromosome is created for each run. $RMSE\approx0.135$.

## Crowding
Parent selection is done with the `tournament_selection()` function; pick $k$ individuals randomly from the population, and return the one with the highest fitness.

1. Create two parents.

2. Single-point-crossover chooses an gene index at random, splits the genes at that point, and swaps the tail-end of the parents to create two new children.

3. Bit-flip-mutation takes mutation rate into account, and flips bits of the genome according to probability `m_rate`.

## Population evaluation
This step calculates fitness for each individual (chromosome) of the population. A new chromosome has a `None` value for fitness, so only previously unseen chromosomes are evaluated.

## Tasks

### f) Best solution without feature selection:
Most trials have yielded $RMSE\approx0.135$. See `run_evaluator_with_all_features()` in `ga.rs` that borrows the dataset and calculates this value once with all features before moving into the parallelized grid search.

### g) Playing with hyperparameters
|#|Hyperparameters|RMSE|Comment|
|-|-------------------------|-------------|----------|
|1|Pop=1000,Gen=10,Tsize=50,COrate=0.9,Mrate=0.01|$0.123053$|
|2|Pop=10,Gen=1000,Tsize=50,COrate=0.9,Mrate=0.01|$0.123200$|Too high Tsize; caused catastrophic forgetting to higher RMSE|
|3|Pop=100,Gen=100,Tsize=5,COrate=0.9,Mrate=0.01|$0.123085$||
|4|Pop=100,Gen=100,Tsize=5,COrate=0.2,Mrate=0.01|$0.123751$|Not enough crossovers|
|5|Pop=100,Gen=100,Tsize=5,COrate=0.9,Mrate=0.1|$0.127005$|Too high mutation, overwrites better solutions
|6|Pop=100,Gen=100,Tsize=5,COrate=0.9,Mrate=0.0001|$0.124385$|Effectively no mutation|

Record result so far:
```bash
Number of features selected: 59
Pop=200,Gen=100,Tsize=5,COrate=0.9,Mrate=0.01
RMSE: 0.122760

# Same result with this one
Number of features selected: 58
Pop=300,Gen=200,Tsize=5,COrate=0.9,Mrate=0.01,Ecount=2
RMSE: 0.122760

# And again with
Number of features selected: 58
Pop=300,Gen=200,Tsize=3,COrate=0.9,Mrate=0.01,Ecount=5
RMSE: 0.122760
```
## Parallization and grid search
I parallellize the GA using the Rayon crate. It takes care of cloning the FitnessEvaluator across all cores, and sharing the dataset.

I set up a grid of hyperparameters to search several combinations (aka gridsearch), inspired by (Geron, 2022)[3].

This is an example of a grid:
```rust
let pop_sizes = [100, 200];
    let gen_sets = [100, 200];
    let t_sizes = [3, 5];
    let c_rates = [0.7, 0.9];
    let m_rates = [0.05, 0.01];
    let elites = [0, 2, 5];
```

The grid search yields the same record result as before:
```bash
Best RMSE: 0.122760
Best params: (200, 100, 3, 0.9, 0.01, 5)
Memory usage: 79.17 MB
Total running time: 912.36s
```

### h) Implement two new survivor selection functions
#### `deterministic_crowding()`: 
1. Compute Hamming distance between parent and offspring.
2. Pair each parent with most similar (closest Hamming distance) offspring.
3. For each pair, keep the fitter individual.

This should keep offspring that are close to their parents, reducing premature convergence.

#### `elitism_selection()`:
1. Sort population by fitness,
2. Keep the `elite_count` number of best individuals.
3. Backfill the rest of the population with offspring.

This should accelerate convergence, but will reduce diversity.

### i) Implement elitism
In my presentation, I will compare two grid search runs; one using elitism, the other with deterministic_crowding.

## References
[1] Searching with "Duck Duck Go"

[2] Linfa website: https://rust-ml.github.io/linfa/, crate page: https://crates.io/crates/linfa, docs: https://rust-ml.github.io/linfa/rustdocs/linfa/

[3] Géron, A. (2022). _Hands-On Machine Learning with Scikit-Learn, Keras and Tensorflow_ (3rd ed) O'Reilly Media, p.91-93