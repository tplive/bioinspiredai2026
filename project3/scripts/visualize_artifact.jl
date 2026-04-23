using Printf

local_display_path(path::String) = isabspath(path) ? relpath(path, pwd()) : String(path)

function ensure_file(path::String)
    isfile(path) || error("Missing required file: $(local_display_path(path))")
end

# Visualize landscape from generated data.
function run_landscape_visualization(project_root::String, scripts_dir::String, artifact_dir::String, sample_fraction::Float64)
    penalized_csv = joinpath(artifact_dir, "penalized_fitness.csv")
    local_optima_plot_csv = joinpath(artifact_dir, "local_optima_plot.csv")
    local_optima_csv = joinpath(artifact_dir, "local_optima.csv")

    ensure_file(penalized_csv)

    optima_csv = if isfile(local_optima_plot_csv)
        local_optima_plot_csv
    elseif isfile(local_optima_csv)
        local_optima_csv
    else
        error("Missing required file: $(local_display_path(local_optima_plot_csv)) or $(local_display_path(local_optima_csv))")
    end

    output_png = joinpath(artifact_dir, "fitness_landscape.png")
    output_3d_surface_png = joinpath(artifact_dir, "fitness_landscape_3d.png")

    script_path = joinpath(scripts_dir, "visualize_landscape.jl")
    cmd = `$(Base.julia_cmd()) --project=$(project_root) $(script_path) $(penalized_csv) $(optima_csv) $(output_png) $(output_3d_surface_png) $(sample_fraction)`
    run(cmd)

    output_3d_scatter_png = replace(output_3d_surface_png, ".png" => "_scatter.png")

    return output_png, output_3d_scatter_png, output_3d_surface_png
end

# Visualize convergence data
function run_convergence_visualization(project_root::String, scripts_dir::String, artifact_dir::String)
    convergence_csv = joinpath(artifact_dir, "convergence.csv")
    ensure_file(convergence_csv)

    output_png = joinpath(artifact_dir, "convergence_curve.png")
    script_path = joinpath(scripts_dir, "visualize_convergence.jl")
    cmd = `$(Base.julia_cmd()) --project=$(project_root) $(script_path) $(convergence_csv) $(output_png)`
    run(cmd)

    output_png
end

function main()

    # Ensure arguments are given to the script
    length(ARGS) >= 1 || error("Usage: julia visualize_artifact.jl <artifact_dir> [sample_fraction]")

    # Ensure path
    project_root = normpath(joinpath(@__DIR__, ".."))
    scripts_dir = @__DIR__

    # Parse cmdline arguments
    artifact_arg = ARGS[1]
    artifact_dir = isabspath(artifact_arg) ? artifact_arg : joinpath(project_root, artifact_arg)
    isdir(artifact_dir) || error("Artifact directory does not exist: $(local_display_path(artifact_dir))")

    # Get fraction or default to .2
    sample_fraction = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.20
    0.0 < sample_fraction <= 1.0 || error("sample_fraction must be in (0, 1]")

    println("Visualizing artifact folder: $(local_display_path(artifact_dir))")

    # Get visualization data and write to files.
    landscape_png, landscape_3d_scatter_png, landscape_3d_surface_png = run_landscape_visualization(project_root, scripts_dir, artifact_dir, sample_fraction)
    convergence_png = run_convergence_visualization(project_root, scripts_dir, artifact_dir)

    println("Wrote: $(local_display_path(landscape_png))")
    println("Wrote: $(local_display_path(landscape_3d_scatter_png))")
    println("Wrote: $(local_display_path(landscape_3d_surface_png))")
    println("Wrote: $(local_display_path(convergence_png))")
end

# Run the main function on invocation
main()
