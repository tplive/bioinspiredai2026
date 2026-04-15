using Random

Base.@kwdef struct SGAParams
    population_size::Int = 120
    generations::Int = 150
    tournament_size::Int = 3
    crossover_rate::Float64 = 0.85
    mutation_rate::Float64 = 0.02
end

function initialize_population(n::Int, pop_size::Int, rng::AbstractRNG; enforce_non_empty::Bool)
    population = [rand(rng, Bool, n) for _ in 1:pop_size]

    if enforce_non_empty
        for bits in population
            if !any(bits)
                bits[rand(rng, 1:n)] = true
            end
        end
    end

    population
end

function tournament_pick(population, fitnesses, k::Int, rng::AbstractRNG)
    best_idx = rand(rng, eachindex(population))
    for _ in 2:k
        idx = rand(rng, eachindex(population))
        if fitnesses[idx] > fitnesses[best_idx]
            best_idx = idx
        end
    end
    copy(population[best_idx])
end

function one_point_crossover(p1::Vector{Bool}, p2::Vector{Bool}, rate::Float64, rng::AbstractRNG)
    n = length(p1)
    if n < 2 || rand(rng) > rate
        return copy(p1), copy(p2)
    end

    cut = rand(rng, 1:n-1)
    c1 = vcat(p1[1:cut], p2[cut+1:end])
    c2 = vcat(p2[1:cut], p1[cut+1:end])
    c1, c2
end

function mutate!(bits::Vector{Bool}, rate::Float64, rng::AbstractRNG)
    for i in eachindex(bits)
        if rand(rng) < rate
            bits[i] = !bits[i]
        end
    end
end

function enforce_non_empty!(bits::Vector{Bool}, rng::AbstractRNG)
    any(bits) && return
    bits[rand(rng, eachindex(bits))] = true
end

landscape_n(landscape) = getproperty(landscape, :n)

function enforce_non_empty_population(landscape)
    if hasproperty(landscape, :one_based_indexing)
        return getproperty(landscape, :one_based_indexing)
    end
    false
end

function run_sga(landscape, params::SGAParams; seed::Int)
    rng = MersenneTwister(seed)
    n = landscape_n(landscape)
    enforce_non_empty = enforce_non_empty_population(landscape)

    population = initialize_population(
        n,
        params.population_size,
        rng;
        enforce_non_empty=enforce_non_empty,
    )

    fitnesses = [fitness_bits(landscape, bits) for bits in population]
    best_idx = argmax(fitnesses)
    best_bits = copy(population[best_idx])
    best_fit = fitnesses[best_idx]
    best_so_far = Float64[]

    for _ in 1:params.generations
        elite = copy(population[best_idx])
        elite_fit = fitnesses[best_idx]

        next_population = Vector{Vector{Bool}}(undef, 0)
        push!(next_population, elite)

        while length(next_population) < params.population_size
            p1 = tournament_pick(population, fitnesses, params.tournament_size, rng)
            p2 = tournament_pick(population, fitnesses, params.tournament_size, rng)

            c1, c2 = one_point_crossover(p1, p2, params.crossover_rate, rng)
            mutate!(c1, params.mutation_rate, rng)
            mutate!(c2, params.mutation_rate, rng)

            if enforce_non_empty
                enforce_non_empty!(c1, rng)
                enforce_non_empty!(c2, rng)
            end

            push!(next_population, c1)
            if length(next_population) < params.population_size
                push!(next_population, c2)
            end
        end

        population = next_population
        fitnesses = [fitness_bits(landscape, bits) for bits in population]

        current_best_idx = argmax(fitnesses)
        if fitnesses[current_best_idx] < elite_fit
            worst_idx = argmin(fitnesses)
            population[worst_idx] = elite
            fitnesses[worst_idx] = elite_fit
            best_idx = argmax(fitnesses)
        else
            best_idx = current_best_idx
        end

        if fitnesses[best_idx] > best_fit
            best_fit = fitnesses[best_idx]
            best_bits = copy(population[best_idx])
        end

        push!(best_so_far, best_fit)
    end

    (
        seed=seed,
        best_fitness=best_fit,
        best_bits=best_bits,
        best_bitstring=String(join(Int.(best_bits))),
        best_so_far=best_so_far,
    )
end
