
#set page(paper: "a4")
#set page(numbering: "1 / 1", number-align: center + bottom)

#include "title.typ"

#outline(title: [Table of Contents])
#outline(title: [List of Tables], target: figure.where(kind: table))

#set heading(numbering: "1.1.a")
#set par(justify: true)

= Introduction

This project studies how optimization landscape structure affects search performance. I compare three algorithm families, a single-objective evolutionary algorithm (SGA), a multi-objective evolutionary algorithm (NSGA-II), and a swarm-intelligence method (ACO), on HDF5-based feature-selection landscapes and a synthetic triangle benchmark. For each landscape, I analyze penalized fitness, identify local optima, and use repeated runs to compare convergence and robustness under the same settings. The goal is to show why different problem structures favor different search strategies and why no single optimizer is best for every landscape. 

A summary of how project goals map to code is given in @lab-goals-table. 

#set page(columns: 2)
= Algorithms
The three methods were implemented in a common pipeline so outputs are directly comparable.
== Single-objective algorithm (SGA)

SGA is a generational binary GA with tournament selection, one-point crossover, bit-flip mutation, and elitism; it records best-so-far convergence per run.

== Multi-objective algorithm (NSGA-II)

NSGA-II optimizes accuracy, feature count, and time using Pareto dominance, non-dominated sorting, crowding distance, and parent-offspring elitist selection.

=== Ant colony optimization algorithm (ACO)

ACO samples binary solutions from pheromone-heuristic probabilities, mixes guided and random exploration, applies local search to top ants, and updates pheromone with evaporation, bounds, and stagnation reset.

= Method

I compare SGA, NSGA-II, and ACO on two landscape types: precomputed feature-selection landscapes from HDF5 files and a synthetic triangle benchmark ($n = 16$, $m = 1$, $s = 4$). For feature selection, fitness is computed from averaged accuracy with penalties for selected-feature count and normalized training time.

To keep the comparison fair, all methods use the same experiment budget: 10 seeds (1000-1009), population/ant size 120, and 150 generations or iterations. Shared GA operators are tournament selection (size 3), one-point crossover (0.85), and bit-flip mutation (0.02).

== Synthetic landscape triangle function

The triangle function is implemented following @sanchez_diaz_mengshoel_2024 and @bergquist_sanchez_diaz_2025 p. 23. Fitness depends only on the number of active bits, producing a regular landscape that is useful for controlled optimizer comparison.

== Hamming-1 distance calculation

In an $n$-bit space, each solution has exactly $n$ Hamming-1 neighbors. This matches single-bit mutation and is the neighborhood used to detect local optima. A bitstring is locally optimal if its fitness is higher than all one-bit-flip neighbors. 

== Run configuration

Runs are configured through `configuration.json`, loaded by `main.jl`, and executed with optimizer-specific parameters (`sga`, `nsga-ii`, `aco`). Each run writes best fitness, convergence traces, and optional landscape/optima plots.

== Visualization

The visualization scripts generate landscape plots, 3D animations, and convergence charts so the structure of each problem and the behavior of each optimizer can be inspected visually.


= Results

This section tries to summarize some of the experiment outputs.

== Feature-selection landscapes

For the three feature-selection landscapes (breast_w_01, letter_r_05, and credit_a_08), ACO and NSGA-II consistently outperformed SGA in terms of best fitness and mean best fitness over 10 seeds, see @table-exp-results.

#figure(
	kind: table,
	placement: top,
	scope: "parent",
	caption: [Feature-selection landscape results across algorithms (10 seeds per configuration).],
)[
	#table(
		columns: (auto, auto, auto, auto, auto, auto),
		inset: 6pt,
		stroke: 0.4pt,
		align: center,
		table.header[
			Landscape
		][
			Local optima
		][
			Algorithm
		][
			Best fitness
		][
			Mean best fitness
		][
			Std
		],

		[01-breast-w], [9], [ACO], [0.97012195], [0.97012195], [0.00000000],
		[01-breast-w], [9], [NSGA-II], [0.97012195], [0.97007622], [0.00014462],
		[01-breast-w], [9], [SGA], [0.90658085], [0.90658085], [0.00000000],

		[05-credit-a], [1], [ACO], [0.89126275], [0.89123086], [0.00010084],
		[05-credit-a], [1], [NSGA-II], [0.89126275], [0.88826530], [0.00162388],
		[05-credit-a], [1], [SGA], [0.82601490], [0.82601490], [0.00000000],

		[08-letter-r], [7], [ACO], [0.95633334], [0.95633334], [0.00000000],
		[08-letter-r], [7], [NSGA-II], [0.95633334], [0.95633334], [0.00000000],
		[08-letter-r], [7], [SGA], [0.77579500], [0.77579500], [0.00000000],
	)
]<table-exp-results>

Overall, ACO and NSGA-II reached the same best values on all three feature landscapes, while SGA converged to clearly lower-quality solutions in this configuration.

== Synthetic triangle landscapes

For the main synthetic setting ($n = 16, m = 1, s = 4$), both ACO and NSGA-II reached the maximum observed fitness across all runs:
- Local optima (strict Hamming-1): 3640
- ACO: best/mean 4.0, std 0.0
- NSGA-II: best/mean 4.0, std 0.0

An additional smaller synthetic run ($n = 8, m = 1, s = 2$) with ACO also reached its top observed value consistently:
- Local optima (strict Hamming-1): 56
- ACO: best/mean 2.0, std 0.0

These results indicate that both ACO and NSGA-II handle the structured triangle landscape effectively, even when the number of local optima is large.

== Comparative interpretation

Two patterns are clear from the artifact summaries:
- Algorithm ranking: ACO and NSGA-II were the strongest methods on all reported feature-selection landscapes, with practically identical best performance.
- Robustness: Most runs had very low dispersion across seeds, and several configurations achieved zero standard deviation, suggesting highly stable convergence under the selected parameter settings.

At the same time, the 05-credit-a landscape showed slightly larger variance for NSGA-II than for ACO, which may indicate higher sensitivity to initialization or parent-offspring selection dynamics on that specific landscape.

== Comments on the included figures
- @img-01-aco-fitness-landscape-3d-png shows the breast-w-01 landscape after ACO has identified 9 local optima. With a landscape that is relatively small, peaks are easily identifiable. Even so, some of the optima are hard to read due to overlapping text.

- @img-01-nsga-ii-convergence-curve shows how the NSGA-II algoritm converges on the same dataset, with the best result discovered around generation 140.

- Dataset letter-r-05 has a lot more datapoints, as shown in @img-05-aco-fitness-landscape-3d. It's not really feasible to make out details, but the final optima is still plotted.

- I attempted a scatterplot of the same dataset in @img-05-nsga-ii-fitness-landscape-3d, with fitness as a function of the number of active features and normalized time. I also produced an animation of this plot, rotating around the fitness axis, revealing interesting artifacts of the data that cannot be seen at this angle.

- In @img-08-nsga-ii-fitness-landscape I visualize how the number of active features are distributed in the model. The local optima shown in the lower histogram are all on 4 active features.

- On the triangle landscape shown in @img-triangle-aco-fitness-landscape-3d, the local optima are more visible.


= Appendix: Lab goals

This appendix maps each required lab goal to the implementation in the codebase, so the coverage can be verified directly from source files, see @lab-goals-table.

#figure(
	kind: table,
	placement: top,
	scope: "parent",
	caption: [Lab goals and where they are covered in code.],
)[
	#table(
		columns: (2.2fr, 3.2fr, 2.4fr),
		inset: 6pt,
		stroke: 0.4pt,
		align: left,
		table.header[
			Lab goal
		][
			Where it is covered
		][
			Primary files
		],

		[
			Read and parse large evolutionary landscapes based on feature selection, and implement a synthetic landscape.
		], [
			HDF5 parsing reads accuracies and times from file, builds feature-selection lookup fitness. Synthetic triangle landscape and triangle fitness is implemented.
		], [
			`src/feature_landscape.jl`  \
			`src/triangle_landscape.jl`  \
		],

		[
			Create a visualization of the fitness landscape to highlight optima and show complexity.
		], [
			Local optima are detected with Hamming-1 neighborhood check. Landscape plots highlight local optima and summarize fitness structure.
		], [
			`scripts/visualize_landscape.jl`  \
			`scripts/animate_landscape_3d.jl`
		],

		[
			(Optional) Visualize algorithm behavior on search spaces.
		], [
			Convergence behavior is visualized from mean best-so-far traces across runs.
		], [
			`scripts/visualize_convergence.jl`
		],

		[
			Implement one single-objective evolutionary algorithm.
		], [
			SGA with tournament selection, one-point crossover, mutation, and elitism.
		], [
			`src/sga.jl`  \
		],

		[
			Implement one multi-objective evolutionary algorithm.
		], [
			NSGA-II with Pareto dominance, non-dominated sorting, crowding distance, elitism.
		], [
			`src/nsga2.jl`  \
		],

		[
			Implement one swarm intelligence optimization algorithm.
		], [
			Ant Colony Optimization with pheromone/heuristic sampling, elite updates, local search refinement, and stagnation reset.
		], [
			`src/ant_colony.jl`  \
		],

		[
			Compare algorithms and show experimentation results.
		], [
			Algorithm results are exported to CSV and summarized; result artifacts are generated per dataset/optimizer and discussed in this report.
		], [
			`src/main.jl`  \
			`artifacts/*/summary.md`  \
			`artifacts/*/runs.csv`  \
		],

		[
			Test algorithms on unseen landscape and try to find multiple optima.
		], [
			Support exists through json-file configuration. The same pipeline can be run on unseen HDF5 landscapes by changing `dataset_path`.
		], [
			`configuration.json`
		],
	)
]<lab-goals-table>

= Appendix: Images

Collection of images produced by the visualisation logic.

#figure(
	placement: top,
	scope: "parent",
	caption: [3D fitness landscape from ACO on dataset 01.],
)[
	#image("Images/01-aco-fitness_landscape_3d.png", width: 100%)
] <img-01-aco-fitness-landscape-3d-png>

#figure(
	placement: top,
	scope: "parent",
	caption: [NSGA-II convergence curve for dataset 01.],
)[
	#image("Images/01-nsga-ii-convergence_curve.png", width: 100%)
] <img-01-nsga-ii-convergence-curve>

#figure(
	placement: top,
	scope: "parent",
	caption: [3D fitness landscape from ACO on dataset 05.],
)[
	#image("Images/05-aco-fitness_landscape_3d.svg", width: 100%)
] <img-05-aco-fitness-landscape-3d>

#figure(
	placement: top,
	scope: "parent",
	caption: [3D scatterplot of the landscape using NSGA-II on dataset 05.],
)[
	#image("Images/05-nsga-ii-fitness_landscape_3d.png", width: 100%)
] <img-05-nsga-ii-fitness-landscape-3d>

#figure(
	placement: top,
	scope: "parent",
	caption: [NSGA-II fitness distribution plot for dataset 08.],
)[
	#image("Images/08-nsga-ii_fitness_landscape.png", width: 100%)
] <img-08-nsga-ii-fitness-landscape>

#figure(
	placement: top,
	scope: "parent",
	caption: [3D fitness landscape from ACO on the triangle landscape.],
)[
	#image("Images/triangle-aco-fitness_landscape_3d.svg", width: 100%)
] <img-triangle-aco-fitness-landscape-3d>

#pagebreak()
#bibliography("references.bib", title: [Bibliography])


