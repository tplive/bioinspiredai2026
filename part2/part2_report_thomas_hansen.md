# IT3708 Bioinspired Artificial Intelligence
## Project 1 Feature Selection With Genetic Algorithms - Part 2 Feature Selection (group work)

Date: 01.02.2026

The group has consisted of:

* Thomas Hansen [\<thomasq@stud.ntnu.no\>](mailto:thomasq@stud.ntnu.no)

## Project Setup
This project builds on Part 1, individual work. I have opted to use Rust, as this is my primary language of choice. I chose to refactor most of the code from Part 1 due to readability and scalability, aligning more closely with the structure of comparable code in Python or Julia, as well as generally learning more Rust, and striving to reach idiomatic ways of coding in Rust.

## Overview
The `src/main.rs` file is the entrypoint. It uses the genetic algorithm from `src/ga.rs`. Starting out, I built the scaffolding code such that running the GA would produce a "mock" Chromosome and result.

I implemented the fitness and machine learning function with Linear Regression from the supplied Julia and Python code (see `src/fitness_evaluator`), soliciting some advice from Github Copilot for pointers (no pun intended), but ultimately writing all the Rust code myself.

I ducked [1] for "Machine Learning with Rust" which led me to discover the `linfa` crate [2], which "aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.", a scikit-learn equivalent for Rust. I decided to go with it for this project, despite its experimental status. I also decided to use the `ndarray` crate for supplying n-dimensional arrays, similar to well-known functionality in Python Numpy.

## Parsing the dataset
The dataset is a csv file shaped 1994x102. We aim to find the combination of features that will minimize RMSE when doing linear regression over the data. I was able to refit the `read_from_file` function from Part 1, adding checks to ensure that the data is shaped as indicated.

## Chromosome
A Chromosome in this function is the list of booleans representing genes, 0 for off, 1 for on. A Chromosome for this dataset will have 102 genes, one for each feature. By turning them on and off we aim to find the best fit. Training on a randomized population of chromosomes will yield some result for which we can calculate RMSE.

## Calculate RMSE

$$ \text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}} $$

In Rust code, this function is implemented like this:
1. Create an iterator over the predictions (iter())
2. Pair (zip) the predicted and actual values
3. Subtract actual value from the prediction using map and square it
4. Sum all squared errors
5. Take the square root of sum divided by number of values

## Putting it together
The scaffolding is done when

a) The Chromosome maps active/inactive genes
b) RMSE (fitness) is calculated for a given set of genes
c) The GA can run over several generations

At this point, I'm seeing the same result every generation, but a new random Chromosome is created for each run. RMSE ~ 0.13.

It's time to implement selection, mutation, cross-over, etc...

## Parent selection




## References
[1] Searching with "Duck Duck Go"
[2] Linfa website: https://rust-ml.github.io/linfa/, crate page: https://crates.io/crates/linfa, docs: https://rust-ml.github.io/linfa/rustdocs/linfa/
