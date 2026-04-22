using HDF5
using Statistics

local_display_path(path::AbstractString) = isabspath(path) ? relpath(path, pwd()) : String(path)

struct FeatureLandscape
    values::Vector{Float64}
    times::Vector{Float64}
    n::Int
    epsilon::Float64
    time_penalty::Float64
    one_based_indexing::Bool
    accuracy_dataset_name::String
    times_dataset_name::String
end

function read_h5_metric(path::AbstractString, dataset_name::String)

    h5open(path, "r") do h5
        raw = read(h5[dataset_name])
        ndims(raw) == 1 && return vec(Float64.(raw))
        ndims(raw) == 2 && return vec(mean(Float64.(raw), dims=2))
        error("Unsupported dataset shape $(size(raw)) in $(local_display_path(path))")
    end
end

"""Read the accuracies dataset as the base lookup table.

If accuracies are stored as a 2D matrix (states x repeats/folds),
this returns one value per state by taking the mean across columns.
"""
function read_accuracies_lookup(path::AbstractString, dataset_name::String)
    h5open(path, "r") do h5
        raw = read(h5[dataset_name])
        if ndims(raw) == 1
            return vec(Float64.(raw))
        elseif ndims(raw) == 2
            # Mean over columns -> one lookup fitness per state (row).
            return vec(mean(Float64.(raw), dims=2))
        end

        error("Unsupported accuracies dataset shape $(size(raw)) in $(local_display_path(path))")
    end
end

function normalize(v::Vector{Float64})
    vmin = minimum(v)
    vmax = maximum(v)
    d = vmax - vmin # Delta of max and min
    d <= eps(Float64) && return zeros(Float64, length(v)) # If delta < eps return vector of 0

    (v .- vmin) ./ d # Else do matrix calc on vector
end

function load_feature_landscape(path::AbstractString; epsilon::Float64=0.02, time_penalty::Float64=0.10)
    dataset_names = String[]

    h5open(path, "r") do h5
        for name in keys(h5)
            push!(dataset_names, String(name))
        end
    end

    isempty(dataset_names) && error("No datasets found in $(local_display_path(path))")

    accuracies_index = findfirst(n -> occursin("accuracies", lowercase(n)), dataset_names)
    times_index = findfirst(n -> occursin("times", lowercase(n)), dataset_names)

    isnothing(accuracies_index) && error("Could not find an accuracies dataset in $(local_display_path(path)). Found: $(dataset_names)")
    isnothing(times_index) && error("Could not find an times dataset in $(local_display_path(path)). Found: $(dataset_names)")

    accuracies_dataset_name = dataset_names[accuracies_index]
    times_dataset_name = dataset_names[times_index]

    values = read_accuracies_lookup(path, accuracies_dataset_name)
    times_raw = read_h5_metric(path, times_dataset_name)

    length(values) == length(times_raw) || error("Accuracy and times length mismatch: $(length(values)) vs $(length(times_raw))")

    n, one_based = infer_n_and_indexing(length(values))
    times = normalize(times_raw)

    FeatureLandscape(values, times, n, epsilon, time_penalty, one_based, accuracies_dataset_name, times_dataset_name,)
end

function infer_n_and_indexing(len::Int)
    if ispow2(len)
        return Int(round(log2(len))), false
    end

    if ispow2(len + 1)
        return Int(round(log2(len + 1))), true
    end

    error("Table length $(len) matches neither 2^n nor 2^n - 1")
end

bits_to_decimal(bits::AbstractVector{Bool}) = foldl((acc, b) -> (acc << 1) | Int(b), bits; init=0)

function decimal_to_bits(decimal::Int, n::Int)
    bits = falses(n)
    for i in 1:n
        shift = n - i
        bits[i] = ((decimal >> shift) & 0x1) == 0x1
    end
    bits
end

function decimal_to_row(landscape::FeatureLandscape, decimal::Int)
    if landscape.one_based_indexing
        decimal == 0 && return nothing
        row = decimal
    else
        row = decimal + 1
    end

    row < 1 || row > length(landscape.values) ? nothing : row
end

function row_to_decimal(landscape::FeatureLandscape, row::Int)
    landscape.one_based_indexing ? row : row - 1
end

function fitness_bits(landscape::FeatureLandscape, bits::AbstractVector{Bool})
    length(bits) == landscape.n || error("Bit length $(length(bits)) != n $(landscape.n)")

    decimal = bits_to_decimal(bits)
    row = decimal_to_row(landscape, decimal)
    isnothing(row) && return -Inf

    feature_penalty = landscape.epsilon * count(identity, bits)
    time_penalty = landscape.time_penalty * landscape.times[row]
    landscape.values[row] - feature_penalty - time_penalty
end

function detect_local_optima(landscape::FeatureLandscape; strict::Bool=true)
    optima = NamedTuple[]

    for row in eachindex(landscape.values)
        decimal = row_to_decimal(landscape, row)
        bits = decimal_to_bits(decimal, landscape.n)
        current = fitness_bits(landscape, bits)

        is_local = true
        for bit_idx in 1:landscape.n
            neighbor_bits = copy(bits)
            neighbor_bits[bit_idx] = !neighbor_bits[bit_idx]
            neighbor = fitness_bits(landscape, neighbor_bits)

            if strict
                if neighbor >= current
                    is_local = false
                    break
                end
            else
                if neighbor > current
                    is_local = false
                    break
                end
            end
        end

        if is_local
            push!(optima, (
                row=row,
                decimal=decimal,
                fitness=current,
                bitstring=String(join(Int.(bits))),
            ))
        end
    end

    optima
end
