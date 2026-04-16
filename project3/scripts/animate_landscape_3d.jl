using Plots
using Random

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

function point_lookup(points::Vector{FitnessPoint})
    Dict(p.decimal => p for p in points)
end

function sample_uniform_fraction(points::Vector{FitnessPoint}, fraction::Float64; rng::AbstractRNG=MersenneTwister(42))
    fraction <= 0.0 && return FitnessPoint[]
    fraction >= 1.0 && return copy(points)

    sample_count = max(1, round(Int, fraction * length(points)))
    indices = randperm(rng, length(points))[1:sample_count]
    points[sort(indices)]
end

function build_landscape_plot(sampled_points::Vector{FitnessPoint}, local_optima::Vector{OptimumPoint}, all_points::Vector{FitnessPoint}; azimuth::Float64=45.0, elevation::Float64=30.0)
    lookup = point_lookup(all_points)

    sampled_x = [p.active_features for p in sampled_points]
    sampled_y = [p.normalized_time for p in sampled_points]
    sampled_z = [p.penalized_fitness for p in sampled_points]

    opt_x = Int[]
    opt_y = Float64[]
    opt_z = Float64[]

    for opt in local_optima
        haskey(lookup, opt.decimal) || continue
        p = lookup[opt.decimal]
        push!(opt_x, p.active_features)
        push!(opt_y, p.normalized_time)
        push!(opt_z, opt.fitness)
    end

    plt = scatter3d(
        sampled_x,
        sampled_y,
        sampled_z;
        xlabel="active features",
        ylabel="normalized time",
        zlabel="fitness",
        title="3D Landscape Around Local Optima",
        marker=:circle,
        markersize=3,
        markerstrokewidth=0,
        alpha=0.28,
        marker_z=sampled_z,
        c=:viridis,
        colorbar_title="fitness",
        legend=false,
        size=(1400, 980),
        camera=(azimuth, elevation),
    )

    if !isempty(opt_x)
        scatter3d!(
            plt,
            opt_x,
            opt_y,
            opt_z;
            marker=:diamond,
            markersize=7,
            markercolor=:red,
            markerstrokecolor=:darkred,
            label="local optima",
        )
    end

    plt
end

function main()
    length(ARGS) >= 3 || error("Usage: julia animate_landscape_3d.jl <penalized_fitness.csv> <local_optima.csv> <output.mp4> [sample_fraction] [frames] [fps]")

    penalized_csv = ARGS[1]
    local_optima_csv = ARGS[2]
    output_video = ARGS[3]
    sample_fraction = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.20
    frames = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 180
    fps = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 24

    sample_fraction <= 0.0 && error("sample_fraction must be > 0")
    sample_fraction > 1.0 && error("sample_fraction must be <= 1")
    frames >= 2 || error("frames must be >= 2")
    fps >= 1 || error("fps must be >= 1")

    points = read_penalized_fitness_csv(penalized_csv)
    local_optima = read_csv_local_optima(local_optima_csv)
    sampled_points = sample_uniform_fraction(points, sample_fraction)

    out_dir = dirname(output_video)
    isdir(out_dir) || mkpath(out_dir)

    animation = @animate for frame_idx in 1:frames
        azimuth = 360.0 * (frame_idx - 1) / (frames - 1)
        build_landscape_plot(sampled_points, local_optima, points; azimuth=azimuth, elevation=30.0)
    end

    mp4(animation, output_video; fps=fps)

    println("Wrote: $(output_video)")
    println("Source CSV: $(penalized_csv)")
    println("Optima CSV: $(local_optima_csv)")
    println("States represented (full): $(length(points))")
    println("States represented (sampled): $(length(sampled_points))")
    println("Local optima: $(length(local_optima))")
    println("Frames: $(frames), fps: $(fps), sample_fraction: $(sample_fraction)")
end

main()
