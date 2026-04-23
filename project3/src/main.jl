using Printf
using Statistics
using JSON

include("feature_landscape.jl")
include("triangle_landscape.jl")
include("sga.jl")
include("nsga2.jl")
include("ant_colony.jl")

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function write_runs_csv(path::String, runs)
    open(path, "w") do io
        println(io, "seed,best_fitness,best_bitstring")
        for r in runs
            @printf(io, "%d,%.8f,%s\n", r.seed, r.best_fitness, r.best_bitstring)
        end
    end
end

function write_optima_csv(path::String, optima)
    open(path, "w") do io
        println(io, "row,decimal,fitness,bitstring")
        for o in optima
            @printf(io, "%d,%d,%.8f,%s\n", o.row, o.decimal, o.fitness, o.bitstring)
        end
    end
end

function write_convergence_csv(path::String, mean_curve::Vector{Float64})
    open(path, "w") do io
        println(io, "generation,mean_best_so_far")
        for (g, v) in enumerate(mean_curve)
            @printf(io, "%d,%.8f\n", g - 1, v)
        end
    end
end

function write_penalized_fitness_csv(path::String, landscape::FeatureLandscape)
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

function write_penalized_fitness_csv(path::String, landscape::TriangleLandscape)
    open(path, "w") do io
        println(io, "row,decimal,bitstring,accuracy,normalized_time,active_features,penalized_fitness")

        total_states = 1 << landscape.n
        for decimal in 0:(total_states - 1)
            bits = decimal_to_bits(decimal, landscape.n)
            active_features = count(identity, bits)
            fitness = fitness_bits(landscape, bits)

            @printf(
                io,
                "%d,%d,%s,%.8f,%.8f,%d,%.8f\n",
                decimal + 1,
                decimal,
                String(join(Int.(bits))),
                Float64(fitness),
                0.0,
                active_features,
                Float64(fitness),
            )
        end
    end
end

function resolve_path(project_root::String, path::String)
    isabspath(path) ? path : joinpath(project_root, path)
end

function display_path(project_root::String, path::String)
    isabspath(path) ? relpath(path, project_root) : path
end

function get_config_value(config, key::String, default)
    haskey(config, key) ? config[key] : default
end

function to_int_vector(values)
    [Int(v) for v in values]
end

function select_optima_for_plot(optima; plot_optima::Bool=true, plot_top_n_optima::Int=0)
    if !plot_optima
        return NamedTuple[]
    end

    if plot_top_n_optima <= 0
        return optima
    end

    sorted_optima = sort(optima; by=o -> (-o.fitness, o.decimal))
    keep_count = min(plot_top_n_optima, length(sorted_optima))
    sorted_optima[1:keep_count]
end

function resolve_optimizer(name::String)
    normalized = lowercase(strip(name))
    if normalized == "sga"
        return run_sga
    elseif normalized == "aco"
        return run_aco
    elseif normalized in ("nsga2", "nsga-ii", "nsga_ii")
        return run_nsga2
    else
        error("Unsupported optimizer=$(name). Use 'sga', 'aco', or 'nsga2'.")
    end
end

function main()
    project_root = normpath(joinpath(@__DIR__, ".."))

    config_arg = length(ARGS) >= 1 ? ARGS[1] : "configuration.json"
    config_path = resolve_path(project_root, config_arg)
    isfile(config_path) || error("Missing config file: $(display_path(project_root, config_path))")
    config = JSON.parsefile(config_path)

    landscape_mode = Symbol(lowercase(String(get_config_value(config, "landscape_mode", "feature"))))
    dataset_path = resolve_path(project_root, String(get_config_value(config, "dataset_path", "train_data/08-letter-r_knn_F.h5")))
    epsilon = Float64(get_config_value(config, "epsilon", 0.02))
    time_penalty = Float64(get_config_value(config, "time_penalty", 0.01))
    out_dir = resolve_path(project_root, String(get_config_value(config, "out_dir", "artifacts/08_letter_r")))
    strict_local_optima = Bool(get_config_value(config, "strict_local_optima", true))
    plot_optima = Bool(get_config_value(config, "plot_optima", true))
    plot_top_n_optima = Int(get_config_value(config, "plot_top_n_optima", 0))
    optimizer_name = String(get_config_value(config, "optimizer", "sga"))

    triangle_config = haskey(config, "triangle") ? config["triangle"] : Dict{String, Any}()
    triangle_n = Int(get_config_value(triangle_config, "n", 16))
    triangle_m = Int(get_config_value(triangle_config, "m", 1))
    triangle_s = Int(get_config_value(triangle_config, "s", 4))
    triangle_out_dir = if haskey(triangle_config, "out_dir")
        resolve_path(project_root, String(triangle_config["out_dir"]))
    else
        out_dir
    end

    sga_config = haskey(config, "sga") ? config["sga"] : Dict{String, Any}()
    aco_config = haskey(config, "aco") ? config["aco"] : Dict{String, Any}()

    seeds = if haskey(config, "seeds")
        to_int_vector(config["seeds"])
    else
        seed_start = Int(get_config_value(config, "seed_start", 1000))
        seed_end = Int(get_config_value(config, "seed_end", 1009))
        collect(seed_start:seed_end)
    end

    run_optimizer = resolve_optimizer(optimizer_name)

    params = if lowercase(strip(optimizer_name)) == "aco"
        ACOParams(
            ant_count=Int(get_config_value(aco_config, "ant_count", get_config_value(aco_config, "population_size", 120))),
            iterations=Int(get_config_value(aco_config, "iterations", get_config_value(aco_config, "generations", 150))),
            evaporation_rate=Float64(get_config_value(aco_config, "evaporation_rate", 0.25)),
            alpha=Float64(get_config_value(aco_config, "alpha", 1.0)),
            beta=Float64(get_config_value(aco_config, "beta", 2.0)),
            elite_count=Int(get_config_value(aco_config, "elite_count", 5)),
            deposit_weight=Float64(get_config_value(aco_config, "deposit_weight", 1.0)),
            initial_pheromone=Float64(get_config_value(aco_config, "initial_pheromone", 0.5)),
            min_pheromone=Float64(get_config_value(aco_config, "min_pheromone", 0.05)),
            max_pheromone=Float64(get_config_value(aco_config, "max_pheromone", 0.95)),
        )
    else
        SGAParams(
            population_size=Int(get_config_value(sga_config, "population_size", 120)),
            generations=Int(get_config_value(sga_config, "generations", 150)),
            tournament_size=Int(get_config_value(sga_config, "tournament_size", 3)),
            crossover_rate=Float64(get_config_value(sga_config, "crossover_rate", 0.85)),
            mutation_rate=Float64(get_config_value(sga_config, "mutation_rate", 0.02)),
        )
    end

    if landscape_mode == :feature
        landscape = load_feature_landscape(
            dataset_path;
            epsilon=epsilon,
            time_penalty=time_penalty,
        )
        println("Landscape mode: feature")
        println("Loaded file: $(display_path(project_root, dataset_path))")
        println("Accuracy dataset: $(landscape.accuracy_dataset_name)")
        println("Times dataset: $(landscape.times_dataset_name)")
        println("States: $(length(landscape.values)), n=$(landscape.n), one_based=$(landscape.one_based_indexing)")
    elseif landscape_mode == :triangle
        landscape = load_triangle_landscape(n=triangle_n, m=triangle_m, s=triangle_s)
        out_dir = triangle_out_dir
        println("Landscape mode: triangle")
        println("Triangle params: n=$(landscape.n), m=$(landscape.m), s=$(landscape.s)")
        println("States: $(1 << landscape.n)")
    else
        error("Unsupported landscape_mode=$(landscape_mode). Use feature or triangle in config JSON")
    end

    optima = detect_local_optima(landscape; strict=strict_local_optima)
    println("Strict local optima count: $(length(optima))")
    println("Optimizer: $(optimizer_name)")
    runs = [run_optimizer(landscape, params; seed=s) for s in seeds]

    best_values = [r.best_fitness for r in runs]
    mean_best = mean(best_values)
    std_best = std(best_values)

    best_idx = argmax(best_values)
    best_run = runs[best_idx]

    # Mean convergence curve across runs.
    gen_count = length(runs[1].best_so_far)
    mean_curve = [mean([r.best_so_far[g] for r in runs]) for g in 1:gen_count]

    ensure_dir(out_dir)
    local_optima_path = joinpath(out_dir, "local_optima.csv")
    local_optima_plot_path = joinpath(out_dir, "local_optima_plot.csv")

    write_runs_csv(joinpath(out_dir, "runs.csv"), runs)
    write_optima_csv(local_optima_path, optima)

    plot_optima_rows = select_optima_for_plot(
        optima;
        plot_optima=plot_optima,
        plot_top_n_optima=plot_top_n_optima,
    )
    write_optima_csv(local_optima_plot_path, plot_optima_rows)

    convergence_csv_path = joinpath(out_dir, "convergence.csv")

    write_convergence_csv(convergence_csv_path, mean_curve)
    write_penalized_fitness_csv(joinpath(out_dir, "penalized_fitness.csv"), landscape)

    open(joinpath(out_dir, "summary.md"), "w") do io
        println(io, "# Analysis Summary")
        println(io)
        println(io, "- landscape mode: `$(landscape_mode)`")
        println(io, "- optimizer: `$(optimizer_name)`")
        if landscape_mode == :feature
            println(io, "- states: $(length(landscape.values))")
            println(io, "- dimensions n: $(landscape.n)")
            println(io, "- feature penalty epsilon: $(landscape.epsilon)")
            println(io, "- time penalty weight: $(landscape.time_penalty)")
        else
            println(io, "- triangle n: $(landscape.n)")
            println(io, "- triangle m: $(landscape.m)")
            println(io, "- triangle s: $(landscape.s)")
            println(io, "- states: $(1 << landscape.n)")
        end
        println(io, "- local optima (strict, Hamming-1): $(length(optima))")
        println(io, "- optima plotted: $(length(plot_optima_rows))")
        println(io, "- plot_optima: $(plot_optima)")
        println(io, "- plot_top_n_optima: $(plot_top_n_optima)")
        @printf(io, "- best run fitness: %.8f\n", best_run.best_fitness)
        println(io, "- best run bitstring: `$(best_run.best_bitstring)`")
        @printf(io, "- mean best fitness (%d runs): %.8f\n", length(seeds), mean_best)
        @printf(io, "- std best fitness (%d runs): %.8f\n", length(seeds), std_best)
    end
    
    println("----- Analysis summary -----")
    @printf("Best run fitness: %.8f\n", best_run.best_fitness)
    println("Best run bitstring: $(best_run.best_bitstring)")
    @printf("Mean best fitness (%d runs): %.8f\n", length(seeds), mean_best)
    @printf("Std best fitness (%d runs): %.8f\n", length(seeds), std_best)

    println("Visualization is decoupled from GA runs.")
    println("Run: julia --project=$(project_root) scripts/visualize_artifact_folder.jl $(display_path(project_root, out_dir))")
end

main()