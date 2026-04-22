using Plots

struct ConvergencePoint
    generation::Int
    mean_best_so_far::Float64
end

function read_convergence_csv(path::AbstractString)
    isfile(path) || error("Missing convergence CSV: $(path)")

    points = ConvergencePoint[]
    open(path, "r") do io
        header = readline(io)
        expected = split(strip(header), ',')
        expected == ["generation", "mean_best_so_far"] ||
            error("Unexpected CSV header in $(path): $(header)")

        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            cols = split(stripped, ',')
            length(cols) == 2 || error("Malformed CSV row in $(path): $(line)")
            push!(points, ConvergencePoint(parse(Int, cols[1]), parse(Float64, cols[2])))
        end
    end

    isempty(points) && error("No convergence rows found in $(path)")
    points
end

function write_convergence_png(path::AbstractString, points::Vector{ConvergencePoint})
    generations = [p.generation for p in points]
    values = [p.mean_best_so_far for p in points]

    p = plot(
        generations,
        values;
        linewidth=3,
        color=:dodgerblue,
        marker=:circle,
        markersize=3,
        markerstrokewidth=0,
        xlabel="Generation",
        ylabel="Mean Best-So-Far Fitness",
        title="Algorithm Convergence",
        legend=false,
        gridalpha=0.25,
        size=(1200, 700),
    )

    best_idx = argmax(values)
    scatter!(p, [generations[best_idx]], [values[best_idx]]; color=:red, markersize=7)
    annotate!(p, generations[best_idx], values[best_idx], text(" best=$(round(values[best_idx], digits=6))", 10, :black, :left))

    savefig(p, path)
end

function main()
    length(ARGS) == 2 || error("Usage: julia visualize_convergence.jl <convergence.csv> <output.png>")

    csv_path = ARGS[1]
    out_path = ARGS[2]

    points = read_convergence_csv(csv_path)
    write_convergence_png(out_path, points)

    println("Generations represented: $(length(points))")
end

main()
