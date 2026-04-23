using Test
using Base.Threads
using CSV
using DataFrames

struct TriangleLandscape
    n::Int
    m::Int
    s::Int
end

include("sga.jl")
include("nsga2.jl")

fitnesses = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]

function load_triangle_landscape(; n::Int=31, s::Int=4, m::Int=1)
    TriangleLandscape(n, m, s)
end

function fitness_bits(landscape::TriangleLandscape, bits::Vector{Bool})
    length(bits) == landscape.n || error("Bit length $(length(bits)) != n $(landscape.n)")

    return triangle_function(bits, landscape.n)
end

function decimal_to_bits(decimal::Int, n::Int)
    bits = falses(n)
    for i in 1:n
        shift = n - i
        bits[i] = ((decimal >> shift) & 0x1) == 0x1
    end
    bits
end

function triangle_function(bits::Vector{Bool}, n::Int)
    # From Assignment 3 Lecture, p. 23
    n_ones = sum(bits)

    # If all 1's, just return 6
    if n_ones == 31
        return 6
    end

    # Sample landscape uses period size 5
    s = 5

    m = 1

    i = Int(ceil(n_ones / s))
    if i % 2 == 1
        if n_ones % s == 0
            r = m * s
        else
            r = m * (n_ones % s)
        end
    else
        r = m * (i * s - n_ones)
    end
    return r
end

function detect_local_optima(landscape::TriangleLandscape; strict::Bool=true)
    optima = NamedTuple[]
    total_states = 1 << landscape.n

    for decimal in 0:(total_states-1)
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
                row=decimal + 1,
                decimal=decimal,
                fitness=Float64(current),
                bitstring=String(join(Int.(bits))),
            ))
        end
    end

    optima
end

function is_strict_local_optimum(landscape::TriangleLandscape, bits::Vector{Bool})
    current = fitness_bits(landscape, bits)
    for bit_idx in eachindex(bits)
        neighbor_bits = copy(bits)
        neighbor_bits[bit_idx] = !neighbor_bits[bit_idx]
        if fitness_bits(landscape, neighbor_bits) >= current
            return false
        end
    end
    return true
end

function generate_31bit_sample_landscape_parallel(; fitnesses::Vector{Int}=fitnesses)
    n = 31
    total_states = Int(1) << n

    # one_counts[i] stores number of active bits for the i-th 31-bit combination.
    one_counts = Vector{UInt8}(undef, total_states)

    @threads for idx in 0:(total_states-1)
        one_counts[idx+1] = UInt8(count_ones(UInt32(idx)))
    end

    # fitness_landscape[i] stores fitnesses[one_counts[i] + 1].
    fitness_landscape = Vector{UInt8}(undef, total_states)

    @threads for i in eachindex(one_counts)
        fitness_landscape[i] = fitnesses[UInt8(one_counts[i])+1]
    end

    return (
        one_counts=one_counts,
        fitness_landscape=fitness_landscape,
    )
end

function run_triangle_function_sample_test()

    n = 31

    bitstring_8 = Vector{Bool}(vcat(trues(8), falses(n - 8)))
    bitstring_26 = Vector{Bool}(vcat(trues(26), falses(n - 26)))
    bitstring_31 = Vector{Bool}(vcat(trues(31), falses(n - 31)))

    @testset "triangle_function sample fitness checks" begin
        for bits in (bitstring_8, bitstring_26, bitstring_31)
            n_ones = sum(bits)
            expected = fitnesses[n_ones+1]
            @test triangle_function(bits, n) == expected
        end
    end
end

function run_nsga2_and_write_optima(; output_file::String="triangle_local_optima.csv", landscape::TriangleLandscape=load_triangle_landscape())
    # Setup NSGA-II parameters
    params = SGAParams(
        population_size=1000,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.01,
        tournament_size=2,
    )

    # Run NSGA-II
    nsga_result = run_nsga2(landscape, params; seed=42)
    population = nsga_result.final_population
    objectives = nsga_result.final_objectives

    # Extract local optima (non-dominated solutions from final population)
    fronts, ranks, crowding = nsga2_rank_and_crowding(objectives)
    first_front = fronts[1]

    # Build results dataframe from unique strict local optima only.
    results = NamedTuple[]
    seen = Set{String}()
    for idx in first_front
        bits = population[idx]
        fit = nsga2_objectives(landscape, bits)[1] # fit should be UInt8
        n_ones = count(identity, bits)
        bitstring = String(join(Int.(bits)))

        if bitstring in seen
            continue
        end
        if !is_strict_local_optimum(landscape, bits)
            continue
        end

        push!(seen, bitstring)
        push!(results, (
            bitstring=bitstring,
            ones_count=n_ones,
            fitness=fit,
            rank=ranks[idx],
            crowding=crowding[idx],
        ))
    end

    df = DataFrame(results)
    CSV.write(output_file, df)
    println("Local optima written to $output_file ($(nrow(df)) solutions)")
    return df
end
