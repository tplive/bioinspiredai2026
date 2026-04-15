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
end

function ant_colony_fitness(landscape::FeatureLandscape, bits::AbstractVector{Bool})
    fitness_bits(landscape, bits)
end

function ant_colony_fitness(landscape::TriangleLandscape, bits::AbstractVector{Bool})
    Float64(fitness_bits(landscape, bits))
end

function ant_colony_construct_solution(
    pheromone::Vector{Float64},
    params::ACOParams,
    rng::AbstractRNG,
    enforce_non_empty::Bool,
)
    n = length(pheromone)
    bits = falses(n)

    for i in 1:n
        tau = clamp(pheromone[i], params.min_pheromone, params.max_pheromone)
        desirability_one = (tau ^ params.alpha) * (1.0 ^ params.beta)
        desirability_zero = ((1.0 - tau) ^ params.alpha) * (1.0 ^ params.beta)
        probability_one = desirability_one / (desirability_one + desirability_zero)
        bits[i] = rand(rng) < probability_one
    end

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
    best_bits = falses(n)
    best_fit = -Inf
    best_so_far = Float64[]

    for _ in 1:params.iterations
        ants = Vector{Vector{Bool}}(undef, params.ant_count)
        fitnesses = Vector{Float64}(undef, params.ant_count)

        for ant_idx in 1:params.ant_count
            candidate = ant_colony_construct_solution(pheromone, params, rng, enforce_non_empty)
            ants[ant_idx] = candidate
            fitnesses[ant_idx] = ant_colony_fitness(landscape, candidate)
        end

        iteration_best_idx = argmax(fitnesses)
        if fitnesses[iteration_best_idx] > best_fit
            best_fit = fitnesses[iteration_best_idx]
            best_bits = copy(ants[iteration_best_idx])
        end

        ant_colony_update_pheromone!(pheromone, ants, fitnesses, params)
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
