using Printf
using Statistics

include("landscape.jl")
include("sga.jl")

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
end

function write_runs_csv(path::AbstractString, runs)
    open(path, "w") do io
        println(io, "seed,best_fitness,best_bitstring")
        for r in runs
            @printf(io, "%d,%.8f,%s\n", r.seed, r.best_fitness, r.best_bitstring)
        end
    end
end

function write_optima_csv(path::AbstractString, optima)
    open(path, "w") do io
        println(io, "row,decimal,fitness,bitstring")
        for o in optima
            @printf(io, "%d,%d,%.8f,%s\n", o.row, o.decimal, o.fitness, o.bitstring)
        end
    end
end

function write_convergence_csv(path::AbstractString, mean_curve::Vector{Float64})
    open(path, "w") do io
        println(io, "generation,mean_best_so_far")
        for (g, v) in enumerate(mean_curve)
            @printf(io, "%d,%.8f\n", g - 1, v)
        end
    end
end

function main()
    dataset_path = "train_data/01-breast-w_lr_F.h5"
    epsilon = 0.02
    out_dir = "artifacts/mvp_01_breast_w"

    params = SGAParams(
        population_size=120,
        generations=150,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_rate=0.02,
    )

    landscape = load_feature_landscape(dataset_path; epsilon=epsilon)
    println("Loaded dataset: $(landscape.dataset_name)")
    println("States: $(length(landscape.values)), n=$(landscape.n), one_based=$(landscape.one_based_indexing)")

    optima = detect_local_optima(landscape; strict=true)
    println("Strict local optima count: $(length(optima))")

    seeds = collect(1000:1009)
    runs = [run_sga(landscape, params; seed=s) for s in seeds]

    best_values = [r.best_fitness for r in runs]
    mean_best = mean(best_values)
    std_best = std(best_values)

    best_idx = argmax(best_values)
    best_run = runs[best_idx]

    # Mean convergence curve across runs.
    gen_count = length(runs[1].best_so_far)
    mean_curve = [mean([r.best_so_far[g] for r in runs]) for g in 1:gen_count]

    ensure_dir(out_dir)
    write_runs_csv(joinpath(out_dir, "runs.csv"), runs)
    write_optima_csv(joinpath(out_dir, "local_optima.csv"), optima)
    write_convergence_csv(joinpath(out_dir, "convergence.csv"), mean_curve)

    open(joinpath(out_dir, "summary.md"), "w") do io
        println(io, "# MVP Summary")
        println(io)
        println(io, "- dataset: `$(dataset_path)`")
        println(io, "- selected HDF5 dataset: `$(landscape.dataset_name)`")
        println(io, "- states: $(length(landscape.values))")
        println(io, "- dimensions n: $(landscape.n)")
        println(io, "- epsilon: $(landscape.epsilon)")
        println(io, "- local optima (strict, Hamming-1): $(length(optima))")
        @printf(io, "- best run fitness: %.8f\n", best_run.best_fitness)
        println(io, "- best run bitstring: `$(best_run.best_bitstring)`")
        @printf(io, "- mean best fitness (10 runs): %.8f\n", mean_best)
        @printf(io, "- std best fitness (10 runs): %.8f\n", std_best)
    end

    println("----- SGA analysis summary -----")
    @printf("Best run fitness: %.8f\n", best_run.best_fitness)
    println("Best run bitstring: $(best_run.best_bitstring)")
    @printf("Mean best fitness (10 runs): %.8f\n", mean_best)
    @printf("Std best fitness (10 runs): %.8f\n", std_best)
    println("Artifacts written to $(out_dir)")
end

main()
