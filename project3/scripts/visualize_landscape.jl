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
    return points
end

function read_csv_local_optima(path::AbstractString)
    isfile(path) || error("Missing local optima CSV: $(path)")

    points = OptimumPoint[]
    open(path, "r") do io
        header = readline(io)
        expected = split(strip(header), ',')
        expected == ["row", "decimal", "fitness", "bitstring"] || error("Unexpected CSV header in $(path): $(header)")

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

    isempty(points) && error("No local optima found in $(path)")
    return points
end

function infer_n_from_bitstrings(points::Vector{FitnessPoint}, optima::Vector{OptimumPoint})
    n1 = maximum(length(p.bitstring) for p in points)
    n2 = maximum(length(p.bitstring) for p in optima)
    n1 == n2 || error("Penalized fitness and local optima CSV files disagree on bitstring width")
    return n1
end

function decimal_to_xy(decimal::Int, nx::Int)
    xmask = (1 << nx) - 1
    x = decimal & xmask
    y = decimal >> nx
    return x, y
end

function split_bits(n::Int)
    nx = cld(n, 2)
    ny = n - nx
    return nx, ny, (1 << nx), (1 << ny)
end

function bit_range_label(start_bit::Int, stop_bit::Int)
    start_bit == stop_bit && return "bit $(start_bit)"
    return "bits $(start_bit)-$(stop_bit)"
end

function project_isometric(x::Float64, y::Float64, z::Float64, x_mid::Float64, y_mid::Float64, z_min::Float64, z_span::Float64,
    cx::Float64, cy::Float64, scale_xy::Float64, scale_z::Float64)
    xr = x - x_mid
    yr = y - y_mid
    zn = (z - z_min) / z_span
    px = cx + (xr - yr) * scale_xy
    py = cy + (xr + yr) * (scale_xy * 0.52) - zn * scale_z
    return px, py
end

function cell_color(z::Float64, z_min::Float64, z_span::Float64)
    t = clamp((z - z_min) / z_span, 0.0, 1.0)
    r = round(Int, 35 + 190 * t)
    g = round(Int, 105 + 95 * (1 - abs(0.5 - t) * 2))
    b = round(Int, 160 + 70 * (1 - t))
    return "rgb($(r),$(g),$(b))"
end

function tick_positions(count::Int, max_ticks::Int=5)
    if count <= 1
        return [0]
    end

    step = max(1, cld(count - 1, max_ticks - 1))
    ticks = collect(0:step:(count - 1))
    ticks[end] != count - 1 && push!(ticks, count - 1)
    return unique(ticks)
end

function axis_tick_label(value::Float64)
    abs(value - round(value)) < 1e-9 ? string(round(Int, value)) : string(round(value, digits=2))
end

function write_surface_svg(path::AbstractString, surface::Matrix{Float64}, local_optima::Vector{OptimumPoint};
    width::Int=1500,
    height::Int=980,
    title::String="3D Fitness Landscape from Analysis",
)
    ny, nx = size(surface)
    valid = filter(!isnan, vec(surface))
    isempty(valid) && error("No valid fitness values to plot")
    z_min = minimum(valid)
    z_max = maximum(valid)
    z_span = max(z_max - z_min, eps(Float64))

    m_left = 80
    m_top = 100
    plot_w = width - 170
    plot_h = height - 180

    x_mid = (nx - 1) / 2
    y_mid = (ny - 1) / 2
    scale_xy = min(plot_w / max(nx + ny, 2), plot_h / max(nx + ny, 2))
    scale_z = 0.58 * (plot_h / 2)
    cx = m_left + plot_w / 2
    cy = m_top + plot_h * 0.76

    function p(x::Int, y::Int)
        z = surface[y + 1, x + 1]
        return project_isometric(float(x), float(y), z, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
    end

    function p_plane(x::Int, y::Int)
        return project_isometric(float(x), float(y), z_min, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
    end

    function p_z(z::Float64)
        return project_isometric(0.0, 0.0, z, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
    end

    open(path, "w") do io
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$(width)\" height=\"$(height)\" viewBox=\"0 0 $(width) $(height)\">")
        println(io, "  <defs>")
        println(io, "    <linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">")
        println(io, "      <stop offset=\"0%\" stop-color=\"#fff8ed\"/>")
        println(io, "      <stop offset=\"100%\" stop-color=\"#e0ecff\"/>")
        println(io, "    </linearGradient>")
        println(io, "  </defs>")
        println(io, "  <rect x=\"0\" y=\"0\" width=\"$(width)\" height=\"$(height)\" fill=\"url(#bg)\"/>")

        println(io, "  <text x=\"80\" y=\"48\" font-size=\"30\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#111827\">$(title)</text>")
        println(io, "  <text x=\"80\" y=\"74\" font-size=\"15\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">Exact surface from penalized_fitness.csv; x maps to $(bit_range_label(1, nx)) and y maps to $(bit_range_label(nx + 1, nx + ny)); red diamonds mark strict local optima</text>")

        # Axes and scales.
        ox, oy = p_plane(0, 0)
        x_end = p_plane(nx - 1, 0)
        y_end = p_plane(0, ny - 1)
        z_end = p_z(z_max)

        println(io, "  <line x1=\"$(ox)\" y1=\"$(oy)\" x2=\"$(x_end[1])\" y2=\"$(x_end[2])\" stroke=\"#111827\" stroke-width=\"2.0\" stroke-linecap=\"round\"/>")
        println(io, "  <line x1=\"$(ox)\" y1=\"$(oy)\" x2=\"$(y_end[1])\" y2=\"$(y_end[2])\" stroke=\"#111827\" stroke-width=\"2.0\" stroke-linecap=\"round\"/>")
        println(io, "  <line x1=\"$(ox)\" y1=\"$(oy)\" x2=\"$(z_end[1])\" y2=\"$(z_end[2])\" stroke=\"#111827\" stroke-width=\"2.0\" stroke-linecap=\"round\"/>")

        x_ticks = tick_positions(nx, 6)
        y_ticks = tick_positions(ny, 6)
        z_ticks = collect(range(z_min, z_max; length=6))

        for tx in x_ticks
            px1, py1 = p_plane(tx, 0)
            px2, py2 = project_isometric(float(tx), 0.0, z_min + 0.02 * z_span, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
            println(io, "  <line x1=\"$(px1)\" y1=\"$(py1)\" x2=\"$(px2)\" y2=\"$(py2)\" stroke=\"#111827\" stroke-width=\"1.0\"/>")
            println(io, "  <text x=\"$(px1 - 2)\" y=\"$(py1 + 18)\" text-anchor=\"middle\" font-size=\"11\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">$(tx)</text>")
        end

        for ty in y_ticks
            px1, py1 = p_plane(0, ty)
            px2, py2 = project_isometric(0.0, float(ty), z_min + 0.02 * z_span, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
            println(io, "  <line x1=\"$(px1)\" y1=\"$(py1)\" x2=\"$(px2)\" y2=\"$(py2)\" stroke=\"#111827\" stroke-width=\"1.0\"/>")
            println(io, "  <text x=\"$(px1 - 18)\" y=\"$(py1 + 4)\" text-anchor=\"end\" font-size=\"11\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">$(ty)</text>")
        end

        for tz in z_ticks
            _, py2 = project_isometric(0.0, 0.0, tz, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
            println(io, "  <line x1=\"$(ox - 4)\" y1=\"$(py2)\" x2=\"$(ox + 4)\" y2=\"$(py2)\" stroke=\"#111827\" stroke-width=\"1.0\"/>")
            println(io, "  <text x=\"$(ox - 10)\" y=\"$(py2 + 4)\" text-anchor=\"end\" font-size=\"11\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">$(axis_tick_label(tz))</text>")
        end

        println(io, "  <text x=\"$(x_end[1] + 12)\" y=\"$(x_end[2] + 6)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#0f172a\">x ($(bit_range_label(1, nx)))</text>")
        println(io, "  <text x=\"$(y_end[1] - 10)\" y=\"$(y_end[2] + 6)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#0f172a\">y ($(bit_range_label(nx + 1, nx + ny)))</text>")
        println(io, "  <text x=\"$(z_end[1] - 8)\" y=\"$(z_end[2] - 10)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#0f172a\">z</text>")

        # Surface quads, back-to-front.
        for y in 0:(ny - 2)
            for x in 0:(nx - 2)
                z00 = surface[y + 1, x + 1]
                z10 = surface[y + 1, x + 2]
                z11 = surface[y + 2, x + 2]
                z01 = surface[y + 2, x + 1]
                if isnan(z00) || isnan(z10) || isnan(z11) || isnan(z01)
                    continue
                end

                p1x, p1y = p(x, y)
                p2x, p2y = p(x + 1, y)
                p3x, p3y = p(x + 1, y + 1)
                p4x, p4y = p(x, y + 1)
                fill = cell_color((z00 + z10 + z11 + z01) / 4, z_min, z_span)

                println(io, "  <polygon points=\"$(p1x),$(p1y) $(p2x),$(p2y) $(p3x),$(p3y) $(p4x),$(p4y)\" fill=\"$(fill)\" fill-opacity=\"0.62\" stroke=\"#334155\" stroke-opacity=\"0.18\" stroke-width=\"0.45\"/>")
            end
        end

        # Light mesh lines.
        for y in 0:(ny - 1)
            pts = String[]
            for x in 0:(nx - 1)
                z = surface[y + 1, x + 1]
                isnan(z) && continue
                px, py = p(x, y)
                push!(pts, "$(px),$(py)")
            end
            length(pts) > 1 && println(io, "  <polyline points=\"$(join(pts, " "))\" fill=\"none\" stroke=\"#0f172a\" stroke-opacity=\"0.12\" stroke-width=\"0.55\"/>")
        end
        for x in 0:(nx - 1)
            pts = String[]
            for y in 0:(ny - 1)
                z = surface[y + 1, x + 1]
                isnan(z) && continue
                px, py = p(x, y)
                push!(pts, "$(px),$(py)")
            end
            length(pts) > 1 && println(io, "  <polyline points=\"$(join(pts, " "))\" fill=\"none\" stroke=\"#0f172a\" stroke-opacity=\"0.12\" stroke-width=\"0.55\"/>")
        end

        # Mark all local optima.
        for opt in local_optima
            x, y = decimal_to_xy(opt.decimal, cld(length(opt.bitstring), 2))
            px, py = project_isometric(float(x), float(y), opt.fitness, x_mid, y_mid, z_min, z_span, cx, cy, scale_xy, scale_z)
            println(io, "  <polygon points=\"$(px),$(py-6) $(px+6),$(py) $(px),$(py+6) $(px-6),$(py)\" fill=\"#ef4444\" stroke=\"#7f1d1d\" stroke-width=\"1.0\"/>")
            println(io, "  <circle cx=\"$(px)\" cy=\"$(py)\" r=\"2.3\" fill=\"#fff7ed\" opacity=\"0.95\"/>")
            text_dx = x < nx / 2 ? 12 : -12
            text_anchor = x < nx / 2 ? "start" : "end"
            println(io, "  <text x=\"$(px + text_dx)\" y=\"$(py - 8)\" text-anchor=\"$(text_anchor)\" font-size=\"11\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#7f1d1d\">$(opt.decimal) ($(opt.bitstring))</text>")
        end

        lx = width - 390
        ly = 96
        println(io, "  <rect x=\"$(lx)\" y=\"$(ly)\" width=\"322\" height=\"150\" rx=\"12\" fill=\"#ffffff\" fill-opacity=\"0.92\" stroke=\"#cbd5e1\"/>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 26)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#1e293b\">Surface source: penalized_fitness.csv</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 48)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#1e293b\">Red diamonds: strict local optima</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 64)\" font-size=\"13\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#1e293b\">labels: decimal + bitstring</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 84)\" font-size=\"12\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">x axis = $(bit_range_label(1, nx))</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 104)\" font-size=\"12\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">y axis = $(bit_range_label(nx + 1, nx + ny))</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 124)\" font-size=\"12\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">z scale = $(round(z_min, digits=4)) to $(round(z_max, digits=4))</text>")
        println(io, "  <text x=\"$(lx + 14)\" y=\"$(ly + 144)\" font-size=\"12\" font-family=\"Avenir Next, Segoe UI, sans-serif\" fill=\"#334155\">exact table from analyzed CSV artifacts</text>")

        println(io, "</svg>")
    end
end

function main()
    penalized_csv = length(ARGS) >= 1 ? ARGS[1] : "artifacts/01_breast_w/penalized_fitness.csv"
    local_optima_csv = length(ARGS) >= 2 ? ARGS[2] : "artifacts/01_breast_w/local_optima.csv"
    output_svg = length(ARGS) >= 3 ? ARGS[3] : "artifacts/01_breast_w/fitness_landscape_3d.svg"

    fitness_points = read_penalized_fitness_csv(penalized_csv)
    local_optima = read_csv_local_optima(local_optima_csv)
    n = infer_n_from_bitstrings(fitness_points, local_optima)
    nx, ny, x_count, y_count = split_bits(n)

    surface = fill(NaN, y_count, x_count)
    for point in fitness_points
        x, y = decimal_to_xy(point.decimal, nx)
        surface[y + 1, x + 1] = point.penalized_fitness
    end

    out_dir = dirname(output_svg)
    isdir(out_dir) || mkpath(out_dir)

    title = "3D Landscape from penalized fitness table ($(basename(penalized_csv)))"
    write_surface_svg(output_svg, surface, local_optima; title=title)

    println("Wrote: $(output_svg)")
    println("Source CSV: $(penalized_csv)")
    println("Optima CSV: $(local_optima_csv)")
    println("States represented: $(count(!isnan, surface)), n=$(n)")
    println("Strict local optima: $(length(local_optima))")
end

main()
