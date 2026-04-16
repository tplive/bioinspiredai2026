using Plots
using Statistics

struct FitnessPoint
    row::Int
    decimal::Int
    bitstring::String
    accuracy::Float64
    normalized_time::Float64
    active_features::Int
    penalized_fitness::Float64
end

struct OptimumPoint
    row::Int
    decimal::Int
    fitness::Float64
    bitstring::String
end

function read_penalized_fitness_csv(path::AbstractString)
    isfile(path) || error("Missing penalized fitness CSV: $(path)")

    points = FitnessPoint[]
    open(path, "r") do io
        header = readline(io)
        expected = split(strip(header), ',')
        expected == ["row", "decimal", "bitstring", "accuracy", "normalized_time", "active_features", "penalized_fitness"] ||
            error("Unexpected CSV header in $(path): $(header)")

        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            cols = split(stripped, ',')
            length(cols) == 7 || error("Malformed CSV row in $(path): $(line)")
            push!(points, FitnessPoint(
                parse(Int, cols[1]),
                parse(Int, cols[2]),
                cols[3],
                parse(Float64, cols[4]),
                parse(Float64, cols[5]),
                parse(Int, cols[6]),
                parse(Float64, cols[7]),
            ))
        end
    end

    isempty(points) && error("No penalized fitness rows found in $(path)")
    points
end

function read_csv_local_optima(path::AbstractString)
    isfile(path) || error("Missing local optima CSV: $(path)")

    points = OptimumPoint[]
    open(path, "r") do io
        header = readline(io)
        expected = split(strip(header), ',')
        expected == ["row", "decimal", "fitness", "bitstring"] ||
            error("Unexpected CSV header in $(path): $(header)")

        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            cols = split(stripped, ',')
            length(cols) == 4 || error("Malformed CSV row in $(path): $(line)")
            push!(points, OptimumPoint(
                parse(Int, cols[1]),
                parse(Int, cols[2]),
                parse(Float64, cols[3]),
                cols[4],
            ))
        end
    end

    points
end

function infer_n_from_bitstrings(points::Vector{FitnessPoint}, optima::Vector{OptimumPoint})
    n1 = maximum(length(p.bitstring) for p in points)
    isempty(optima) && return n1
    n2 = maximum(length(p.bitstring) for p in optima)
    n1 == n2 || error("Fitness and optima CSV files disagree on bitstring width")
    n1
end

function mean_by_feature_count(points::Vector{FitnessPoint})
    counts = sort(unique(p.active_features for p in points))
    means = [mean(p.penalized_fitness for p in points if p.active_features == c) for c in counts]
    counts, means
end

function write_landscape_png(path::AbstractString, points::Vector{FitnessPoint}, local_optima::Vector{OptimumPoint}, n::Int)
    feature_counts = [p.active_features for p in points]
    fitness_values = [p.penalized_fitness for p in points]
    time_values = [p.normalized_time for p in points]

    isempty(fitness_values) && error("No valid fitness values to plot")

    opt_sorted = sort(local_optima; by=o -> o.fitness, rev=true)
    labeled_opt = opt_sorted[1:min(10, length(opt_sorted))]
    opt_feature_counts = [count(==('1'), opt.bitstring) for opt in local_optima]
    opt_fitness_values = [opt.fitness for opt in local_optima]

    count_levels, mean_values = mean_by_feature_count(points)

    p1 = scatter(
        feature_counts,
        fitness_values;
        xlabel="active features",
        ylabel="fitness",
        title="Landscape: fitness by feature count",
        marker_z=time_values,
        c=:viridis,
        alpha=0.45,
        markersize=4,
        markerstrokewidth=0,
        colorbar_title="normalized time",
        label=false,
        size=(1400, 900),
    )

    if !isempty(local_optima)
        scatter!(
            p1,
            opt_feature_counts,
            opt_fitness_values;
            marker=:diamond,
            markersize=7,
            markercolor=:red,
            markerstrokecolor=:darkred,
            label="local optima",
        )

        for opt in labeled_opt
            x = count(==('1'), opt.bitstring)
            annotate!(p1, x, opt.fitness, text("$(opt.bitstring)", 7, :black))
        end
    end

    plot!(
        p1,
        count_levels,
        mean_values;
        linewidth=3,
        color=:black,
        label="mean by feature count",
    )

    xlims!(p1, -0.5, n + 0.5)

    p2 = histogram(
        feature_counts;
        bins=0:(n + 1),
        xlabel="active features",
        ylabel="count",
        title="State distribution by feature count",
        color=:steelblue,
        alpha=0.8,
        label=false,
    )

    if !isempty(local_optima)
        opt_hist = [count(==('1'), opt.bitstring) for opt in local_optima]
        histogram!(p2, opt_hist; bins=0:(n + 1), color=:red, alpha=0.35, label="local optima")
    end

    final_plot = plot(
        p1,
        p2;
        layout=@layout([a{0.66h}; b{0.34h}]),
        size=(1400, 980),
    )

    savefig(final_plot, path)
end

function main()
    penalized_csv = length(ARGS) >= 1 ? ARGS[1] : "artifacts/01_breast_w/penalized_fitness.csv"
    local_optima_csv = length(ARGS) >= 2 ? ARGS[2] : "artifacts/01_breast_w/local_optima.csv"
    output_png = length(ARGS) >= 3 ? ARGS[3] : "artifacts/01_breast_w/fitness_landscape.png"

    fitness_points = read_penalized_fitness_csv(penalized_csv)
    local_optima = read_csv_local_optima(local_optima_csv)
    n = infer_n_from_bitstrings(fitness_points, local_optima)

    out_dir = dirname(output_png)
    isdir(out_dir) || mkpath(out_dir)

    write_landscape_png(output_png, fitness_points, local_optima, n)

    println("Wrote: $(output_png)")
    println("Source CSV: $(penalized_csv)")
    println("Optima CSV: $(local_optima_csv)")
    println("States represented: $(length(fitness_points)), n=$(n)")
    println("Strict local optima: $(length(local_optima))")
end

main()
