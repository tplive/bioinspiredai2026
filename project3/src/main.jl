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

function write_penalized_fitness_csv(path::AbstractString, landscape::FeatureLandscape)
    open(path, "w") do io
        println(io, "row,decimal,bitstring,accuracy,normalized_time,active_features,penalized_fitness")

        for row in eachindex(landscape.values)
            decimal = row_to_decimal(landscape, row)
            bits = decimal_to_bits(decimal, landscape.n)
            active_features = count(identity, bits)
            penalized_fitness = fitness_bits(landscape, bits)

            @printf(
                io,
                "%d,%d,%s,%.8f,%.8f,%d,%.8f\n",
                row,
                decimal,
                String(join(Int.(bits))),
                landscape.values[row],
                landscape.times[row],
                active_features,
                penalized_fitness,
            )
        end
    end
end

function run_visualization_script(project_root::AbstractString, out_dir::AbstractString)
    script_path = joinpath(project_root, "scripts", "visualize_landscape.jl")
    penalized_csv = joinpath(out_dir, "penalized_fitness.csv")
    local_optima_csv = joinpath(out_dir, "local_optima.csv")
    output_svg = joinpath(out_dir, "fitness_landscape_3d.svg")

    cmd = `$(Base.julia_cmd()) --project=$(project_root) $(script_path) $(penalized_csv) $(local_optima_csv) $(output_svg)`
    run(cmd)

    output_svg
end

function main()
    project_root = normpath(joinpath(@__DIR__, ".."))
    dataset_path = "train_data/08-letter-r_knn_F.h5"
    epsilon = 0.02
    time_penalty = 0.1
    out_dir = "artifacts/08_letter_r"

    params = SGAParams(
        population_size=120,
        generations=150,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_rate=0.02,
    )

    landscape = load_feature_landscape(
        dataset_path;
        epsilon=epsilon,
        time_penalty=time_penalty,
    )
    println("Loaded file: $(dataset_path)")
    println("Accuracy dataset: $(landscape.accuracy_dataset_name)")
    println("Times dataset: $(landscape.times_dataset_name)")
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
    write_penalized_fitness_csv(joinpath(out_dir, "penalized_fitness.csv"), landscape)

    open(joinpath(out_dir, "summary.md"), "w") do io
        println(io, "# Analysis Summary")
        println(io)
        println(io, "- dataset file: `$(dataset_path)`")
        println(io, "- accuracy dataset: `$(landscape.accuracy_dataset_name)`")
        println(io, "- times dataset: `$(landscape.times_dataset_name)`")
        println(io, "- states: $(length(landscape.values))")
        println(io, "- dimensions n: $(landscape.n)")
        println(io, "- feature penalty epsilon: $(landscape.epsilon)")
        println(io, "- time penalty weight: $(landscape.time_penalty)")
        println(io, "- local optima (strict, Hamming-1): $(length(optima))")
        println(io, "- full penalized fitness table: `$(joinpath(out_dir, "penalized_fitness.csv"))`")
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

    println("Generating landscape plot...")
    plot_path = run_visualization_script(project_root, out_dir)
    println("Landscape plot written to $(plot_path)")
end

main()
