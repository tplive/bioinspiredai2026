using JSON
using Printf
using Statistics

local_display_path(path::AbstractString) = isabspath(path) ? relpath(path, pwd()) : String(path)

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
end

function read_runs_csv(path::AbstractString)
    isfile(path) || error("Missing runs CSV: $(local_display_path(path))")

    rows = NamedTuple[]
    open(path, "r") do io
        header = strip(readline(io))
        header == "seed,best_fitness,best_bitstring" || error("Unexpected runs CSV header in $(local_display_path(path)): $(header)")

        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            cols = split(s, ',')
            length(cols) == 3 || error("Malformed runs CSV row in $(local_display_path(path)): $(line)")
            push!(rows, (
                seed=parse(Int, cols[1]),
                best_fitness=parse(Float64, cols[2]),
                best_bitstring=cols[3],
            ))
        end
    end

    isempty(rows) && error("No run rows found in $(local_display_path(path))")
    rows
end

function read_optima_count(path::AbstractString)
    isfile(path) || return 0

    count = 0
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            isempty(strip(line)) && continue
            count += 1
        end
    end
    count
end

function sanitize_name(path::AbstractString)
    base = splitext(basename(path))[1]
    replace(base, r"[^A-Za-z0-9_-]" => "_")
end

function write_json(path::AbstractString, data)
    open(path, "w") do io
        write(io, JSON.json(data, 2))
    end
end

function run_main_with_config(project_root::AbstractString, config_path::AbstractString)
    main_path = joinpath(project_root, "src", "main.jl")
    cmd = `$(Base.julia_cmd()) --project=$(project_root) $(main_path) $(config_path)`
    run(cmd)
end

function display_path(project_root::AbstractString, path::AbstractString)
    isabspath(path) ? relpath(path, project_root) : path
end

function default_datasets(project_root::AbstractString)
    train_dir = joinpath(project_root, "train_data")
    isdir(train_dir) || error("Missing train_data directory: $(display_path(project_root, train_dir))")

    files = sort(filter(p -> endswith(lowercase(p), ".h5"), readdir(train_dir; join=true)))
    isempty(files) && error("No .h5 files found in $(display_path(project_root, train_dir))")

    [relpath(p, project_root) for p in files]
end

function write_summary_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "dataset,optimizer,runs,best_fitness,mean_best_fitness,std_best_fitness,best_bitstring,local_optima_count,out_dir")
        for r in rows
            @printf(
                io,
                "%s,%s,%d,%.8f,%.8f,%.8f,%s,%d,%s\n",
                r.dataset,
                r.optimizer,
                r.runs,
                r.best_fitness,
                r.mean_best_fitness,
                r.std_best_fitness,
                r.best_bitstring,
                r.local_optima_count,
                r.out_dir,
            )
        end
    end
end

function write_summary_md(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "# Batch Experiment Summary")
        println(io)
        println(io, "| Dataset | Optimizer | Runs | Best | Mean | Std | Best bitstring | Local optima | Out dir |")
        println(io, "|---|---:|---:|---:|---:|---:|---|---:|---|")
        for r in rows
            @printf(
                io,
                "| %s | %s | %d | %.8f | %.8f | %.8f | `%s` | %d | `%s` |\n",
                r.dataset,
                r.optimizer,
                r.runs,
                r.best_fitness,
                r.mean_best_fitness,
                r.std_best_fitness,
                r.best_bitstring,
                r.local_optima_count,
                r.out_dir,
            )
        end
    end
end

function main()
    project_root = normpath(joinpath(@__DIR__, ".."))

    base_config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(project_root, "configuration.json")
    isabspath(base_config_path) || (base_config_path = joinpath(project_root, base_config_path))
    isfile(base_config_path) || error("Missing base config: $(display_path(project_root, base_config_path))")
    base_config = JSON.parsefile(base_config_path)

    datasets = if length(ARGS) >= 2
        split(ARGS[2], ',')
    else
        default_datasets(project_root)
    end

    optimizers = if length(ARGS) >= 3
        split(ARGS[3], ',')
    else
        ["sga", "nsga-ii", "aco"]
    end

    output_root = joinpath(project_root, "artifacts", "batch")
    configs_dir = joinpath(output_root, "_generated_configs")
    ensure_dir(output_root)
    ensure_dir(configs_dir)

    summary_rows = NamedTuple[]

    for dataset in datasets
        dataset_rel = strip(dataset)
        isempty(dataset_rel) && continue

        dataset_abs = isabspath(dataset_rel) ? dataset_rel : joinpath(project_root, dataset_rel)
        isfile(dataset_abs) || error("Dataset does not exist: $(display_path(project_root, dataset_abs))")

        dataset_name = sanitize_name(dataset_abs)

        for optimizer in optimizers
            opt = lowercase(strip(optimizer))
            isempty(opt) && continue

            out_rel = joinpath("artifacts", "batch", "$(dataset_name)-$(opt)")
            out_abs = joinpath(project_root, out_rel)

            cfg = deepcopy(base_config)
            cfg["landscape_mode"] = "feature"
            cfg["dataset_path"] = relpath(dataset_abs, project_root)
            cfg["optimizer"] = opt
            cfg["out_dir"] = out_rel
            cfg["run_visualization"] = false
            cfg["plot_optima"] = true
            cfg["plot_top_n_optima"] = 0

            cfg_path = joinpath(configs_dir, "$(dataset_name)-$(opt).json")
            write_json(cfg_path, cfg)

            println("Running dataset=$(dataset_name), optimizer=$(opt)...")
            run_main_with_config(project_root, cfg_path)

            runs = read_runs_csv(joinpath(out_abs, "runs.csv"))
            best_values = [r.best_fitness for r in runs]
            best_idx = argmax(best_values)
            local_optima_count = read_optima_count(joinpath(out_abs, "local_optima.csv"))

            push!(summary_rows, (
                dataset=dataset_name,
                optimizer=opt,
                runs=length(runs),
                best_fitness=maximum(best_values),
                mean_best_fitness=mean(best_values),
                std_best_fitness=std(best_values),
                best_bitstring=runs[best_idx].best_bitstring,
                local_optima_count=local_optima_count,
                out_dir=out_rel,
            ))
        end
    end

    summary_csv = joinpath(output_root, "feature_experiment_summary.csv")
    summary_md = joinpath(output_root, "feature_experiment_summary.md")
    write_summary_csv(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows)

    println("Wrote summary CSV: $(display_path(project_root, summary_csv))")
    println("Wrote summary MD:  $(display_path(project_root, summary_md))")
end

main()
