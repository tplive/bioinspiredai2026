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








## References
[1] Searching with Duck Duck Go
[2] Linfa website: https://rust-ml.github.io/linfa/, crate page: https://crates.io/crates/linfa, docs: https://rust-ml.github.io/linfa/rustdocs/linfa/
