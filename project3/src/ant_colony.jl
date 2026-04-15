using Random

Base.@kwdef struct ACOParams
    ant_count::Int = 120
    iterations::Int = 150
    evaporation_rate::Float64 = 0.25
    alpha::Float64 = 1.0
    beta::Float64 = 2.0
    elite_count::Int = 5
    deposit_weight::Float64 = 1.0
    initial_pheromone::Float64 = 0.5
    min_pheromone::Float64 = 0.05
    max_pheromone::Float64 = 0.95
    local_search_count::Int = 3
    random_ant_fraction::Float64 = 0.10
    stagnation_limit::Int = 20
end

function ant_colony_fitness(landscape::FeatureLandscape, bits::AbstractVector{Bool})
    decimal = bits_to_decimal(bits)
    row = decimal_to_row(landscape, decimal)
    isnothing(row) && return -Inf
    Float64(landscape.values[row])
end

function ant_colony_fitness(landscape::TriangleLandscape, bits::AbstractVector{Bool})
    Float64(fitness_bits(landscape, bits))
end

function ant_colony_heuristic(
    ants::Vector{Vector{Bool}},
    fitnesses::Vector{Float64},
    params::ACOParams,
)
    order = sortperm(fitnesses; rev=true)
    elite_limit = min(params.elite_count, length(order))
    elite_indices = order[1:elite_limit]

    heuristic = fill(0.5, length(ants[1]))
    selected_fitnesses = fitnesses[elite_indices]
    min_fit = minimum(selected_fitnesses)
    max_fit = maximum(selected_fitnesses)
    fit_span = max_fit - min_fit

    for bit_idx in eachindex(heuristic)
        weighted_sum = 0.0
        total_weight = 0.0

        for idx in elite_indices
            weight = fit_span <= eps(Float64) ? 1.0 : ((fitnesses[idx] - min_fit) / fit_span)
            weight = max(weight, 0.0)
            weighted_sum += weight * (ants[idx][bit_idx] ? 1.0 : 0.0)
            total_weight += weight
        end

        if total_weight > 0.0
            heuristic[bit_idx] = clamp(weighted_sum / total_weight, 0.05, 0.95)
        end
    end

    heuristic
end

function ant_colony_construct_solution(
    pheromone::Vector{Float64},
    heuristic::Vector{Float64},
    params::ACOParams,
    rng::AbstractRNG,
    enforce_non_empty::Bool,
)
    n = length(pheromone)
    bits = falses(n)

    for i in 1:n
        tau = clamp(pheromone[i], params.min_pheromone, params.max_pheromone)
        eta = clamp(heuristic[i], params.min_pheromone, params.max_pheromone)
        desirability_one = (tau ^ params.alpha) * (eta ^ params.beta)
        desirability_zero = ((1.0 - tau) ^ params.alpha) * ((1.0 - eta) ^ params.beta)
        probability_one = desirability_one / (desirability_one + desirability_zero)
        bits[i] = rand(rng) < probability_one
    end

    if enforce_non_empty && !any(bits)
        bits[rand(rng, 1:n)] = true
    end

    bits
end

function ant_colony_local_search!(landscape, bits::Vector{Bool})
    current_fit = ant_colony_fitness(landscape, bits)
    improved = true

    while improved
        improved = false
        best_idx = 0
        best_fit = current_fit

        for bit_idx in eachindex(bits)
            bits[bit_idx] = !bits[bit_idx]
            candidate_fit = ant_colony_fitness(landscape, bits)
            bits[bit_idx] = !bits[bit_idx]

            if candidate_fit > best_fit
                best_fit = candidate_fit
                best_idx = bit_idx
            end
        end

        if best_idx != 0
            bits[best_idx] = !bits[best_idx]
            current_fit = best_fit
            improved = true
        end
    end

    current_fit
end

function ant_colony_random_solution(n::Int, rng::AbstractRNG, enforce_non_empty::Bool)
    bits = rand(rng, Bool, n)
    if enforce_non_empty && !any(bits)
        bits[rand(rng, 1:n)] = true
    end
    bits
end

function ant_colony_update_pheromone!(
    pheromone::Vector{Float64},
    ants::Vector{Vector{Bool}},
    fitnesses::Vector{Float64},
    params::ACOParams,
)
    order = sortperm(fitnesses; rev=true)
    elite_limit = min(params.elite_count, length(order))
    elite_indices = order[1:elite_limit]

    selected_fitnesses = fitnesses[elite_indices]
    min_fit = minimum(selected_fitnesses)
    max_fit = maximum(selected_fitnesses)
    fit_span = max_fit - min_fit

    target = zeros(Float64, length(pheromone))
    total_weight = 0.0

    for idx in elite_indices
        weight = fit_span <= eps(Float64) ? 1.0 : ((fitnesses[idx] - min_fit) / fit_span)
        weight = max(weight, 0.0)
        weight *= params.deposit_weight
        total_weight += weight

        bits = ants[idx]
        for bit_idx in eachindex(bits)
            target[bit_idx] += weight * (bits[bit_idx] ? 1.0 : 0.0)
        end
    end

    if total_weight > 0.0
        target ./= total_weight
    else
        target .= 0.5
    end

    for i in eachindex(pheromone)
        pheromone[i] = (1.0 - params.evaporation_rate) * pheromone[i] + params.evaporation_rate * target[i]
        pheromone[i] = clamp(pheromone[i], params.min_pheromone, params.max_pheromone)
    end

    pheromone
end

function run_aco(landscape, params::ACOParams; seed::Int)
    rng = MersenneTwister(seed)
    n = landscape_n(landscape)
    enforce_non_empty = enforce_non_empty_population(landscape)

    pheromone = fill(clamp(params.initial_pheromone, params.min_pheromone, params.max_pheromone), n)
    heuristic = fill(0.5, n)
    best_bits = falses(n)
    best_fit = -Inf
    best_so_far = Float64[]
    stagnation = 0

    for _ in 1:params.iterations
        ants = Vector{Vector{Bool}}(undef, params.ant_count)
        fitnesses = Vector{Float64}(undef, params.ant_count)
        random_ant_count = max(1, round(Int, params.ant_count * params.random_ant_fraction))

        for ant_idx in 1:params.ant_count
            if ant_idx <= random_ant_count
                candidate = ant_colony_random_solution(n, rng, enforce_non_empty)
            else
                candidate = ant_colony_construct_solution(pheromone, heuristic, params, rng, enforce_non_empty)
            end
            ants[ant_idx] = candidate
            fitnesses[ant_idx] = ant_colony_fitness(landscape, candidate)
        end

        order = sortperm(fitnesses; rev=true)
        local_search_limit = min(params.local_search_count, length(order))
        for idx in order[1:local_search_limit]
            improved_fit = ant_colony_local_search!(landscape, ants[idx])
            fitnesses[idx] = improved_fit
        end

        iteration_best_idx = argmax(fitnesses)
        if fitnesses[iteration_best_idx] > best_fit
            best_fit = fitnesses[iteration_best_idx]
            best_bits = copy(ants[iteration_best_idx])
            stagnation = 0
        else
            stagnation += 1
        end

        heuristic = ant_colony_heuristic(ants, fitnesses, params)
        ant_colony_update_pheromone!(pheromone, ants, fitnesses, params)

        if stagnation >= params.stagnation_limit
            pheromone .= clamp(params.initial_pheromone, params.min_pheromone, params.max_pheromone)
            heuristic .= 0.5
            stagnation = 0
        end

        push!(best_so_far, best_fit)
    end

    (
        seed=seed,
        best_fitness=best_fit,
        best_bits=best_bits,
        best_bitstring=String(join(Int.(best_bits))),
        best_so_far=best_so_far,
        pareto_size=1,
    )
end
