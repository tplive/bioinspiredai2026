using Random

function nsga2_objectives(landscape::FeatureLandscape, bits::Vector{Bool})
    decimal = bits_to_decimal(bits)
    row = decimal_to_row(landscape, decimal)
    if isnothing(row)
        return (-Inf, -Inf, -Inf)
    end

    accuracy = Float64(landscape.values[row])
    active_features = Float64(count(identity, bits))
    normalized_time = Float64(landscape.times[row])
    return (accuracy, -active_features, -normalized_time)
end

function nsga2_objectives(landscape::TriangleLandscape, bits::Vector{Bool})
    fitness = Float64(fitness_bits(landscape, bits))
    active_features = Float64(count(identity, bits))
    return (fitness, -active_features, 0.0)
end

function nsga2_dominates(a::NTuple{3, Float64}, b::NTuple{3, Float64})
    better_or_equal = (a[1] >= b[1]) && (a[2] >= b[2]) && (a[3] >= b[3])
    strictly_better = (a[1] > b[1]) || (a[2] > b[2]) || (a[3] > b[3])
    better_or_equal && strictly_better
end

function nsga2_non_dominated_sort(objectives::Vector{NTuple{3, Float64}})
    count = length(objectives)
    domination_counts = fill(0, count)
    dominates_sets = [Int[] for _ in 1:count]
    fronts = Vector{Vector{Int}}()

    first_front = Int[]
    for p in 1:count
        for q in 1:count
            p == q && continue

            if nsga2_dominates(objectives[p], objectives[q])
                push!(dominates_sets[p], q)
            elseif nsga2_dominates(objectives[q], objectives[p])
                domination_counts[p] += 1
            end
        end

        if domination_counts[p] == 0
            push!(first_front, p)
        end
    end

    push!(fronts, first_front)
    current = 1
    while current <= length(fronts)
        next_front = Int[]
        for p in fronts[current]
            for q in dominates_sets[p]
                domination_counts[q] -= 1
                if domination_counts[q] == 0
                    push!(next_front, q)
                end
            end
        end

        isempty(next_front) || push!(fronts, next_front)
        current += 1
    end

    ranks = fill(typemax(Int), count)
    for (rank, front) in enumerate(fronts)
        for idx in front
            ranks[idx] = rank
        end
    end

    fronts, ranks
end

function nsga2_crowding_distance(front::Vector{Int}, objectives::Vector{NTuple{3, Float64}})
    distances = Dict{Int, Float64}(idx => 0.0 for idx in front)
    length(front) <= 2 && begin
        for idx in front
            distances[idx] = Inf
        end
        return distances
    end

    for objective_idx in 1:3
        sorted_front = sort(front, by=idx -> objectives[idx][objective_idx])
        distances[sorted_front[1]] = Inf
        distances[sorted_front[end]] = Inf

        min_value = objectives[sorted_front[1]][objective_idx]
        max_value = objectives[sorted_front[end]][objective_idx]
        span = max_value - min_value
        span <= eps(Float64) && continue

        for position in 2:(length(sorted_front) - 1)
            idx = sorted_front[position]
            isinf(distances[idx]) && continue
            prev_value = objectives[sorted_front[position - 1]][objective_idx]
            next_value = objectives[sorted_front[position + 1]][objective_idx]
            distances[idx] += (next_value - prev_value) / span
        end
    end

    distances
end

function nsga2_rank_and_crowding(objectives::Vector{NTuple{3, Float64}})
    fronts, ranks = nsga2_non_dominated_sort(objectives)
    crowding = fill(0.0, length(objectives))

    for front in fronts
        distances = nsga2_crowding_distance(front, objectives)
        for idx in front
            crowding[idx] = distances[idx]
        end
    end

    fronts, ranks, crowding
end

function nsga2_better_index(i::Int, j::Int, ranks::Vector{Int}, crowding::Vector{Float64})
    if ranks[i] < ranks[j]
        return i
    elseif ranks[j] < ranks[i]
        return j
    elseif crowding[i] > crowding[j]
        return i
    elseif crowding[j] > crowding[i]
        return j
    end

    i
end

function nsga2_tournament_pick(population, ranks::Vector{Int}, crowding::Vector{Float64}, k::Int, rng::AbstractRNG)
    best_idx = rand(rng, eachindex(population))
    for _ in 2:k
        idx = rand(rng, eachindex(population))
        best_idx = nsga2_better_index(best_idx, idx, ranks, crowding)
    end
    copy(population[best_idx])
end

function nsga2_environmental_selection(population, objectives, population_size::Int)
    fronts, _, _ = nsga2_rank_and_crowding(objectives)

    selected = Int[]
    for front in fronts
        if length(selected) + length(front) <= population_size
            append!(selected, front)
        else
            distances = nsga2_crowding_distance(front, objectives)
            ordered_front = sort(front, by=idx -> distances[idx], rev=true)
            remaining = population_size - length(selected)
            append!(selected, ordered_front[1:remaining])
            break
        end
    end

    next_population = [copy(population[idx]) for idx in selected]
    next_objectives = [objectives[idx] for idx in selected]
    next_population, next_objectives
end

function run_nsga2(landscape, params::SGAParams; seed::Int)
    rng = MersenneTwister(seed)
    n = landscape_n(landscape)
    enforce_non_empty = enforce_non_empty_population(landscape)

    population = initialize_population(
        n,
        params.population_size,
        rng;
        enforce_non_empty=enforce_non_empty,
    )

    objectives = [nsga2_objectives(landscape, bits) for bits in population]
    primary_scores = [obj[1] for obj in objectives]
    best_idx = argmax(primary_scores)
    best_bits = copy(population[best_idx])
    best_fit = primary_scores[best_idx]
    best_so_far = Float64[]

    for _ in 1:params.generations
        _, ranks, crowding = nsga2_rank_and_crowding(objectives)

        offspring = Vector{Vector{Bool}}(undef, 0)
        while length(offspring) < params.population_size
            p1 = nsga2_tournament_pick(population, ranks, crowding, params.tournament_size, rng)
            p2 = nsga2_tournament_pick(population, ranks, crowding, params.tournament_size, rng)

            c1, c2 = one_point_crossover(p1, p2, params.crossover_rate, rng)
            mutate!(c1, params.mutation_rate, rng)
            mutate!(c2, params.mutation_rate, rng)

            if enforce_non_empty
                enforce_non_empty!(c1, rng)
                enforce_non_empty!(c2, rng)
            end

            push!(offspring, c1)
            if length(offspring) < params.population_size
                push!(offspring, c2)
            end
        end

        offspring_objectives = [nsga2_objectives(landscape, bits) for bits in offspring]
        combined_population = vcat(population, offspring)
        combined_objectives = vcat(objectives, offspring_objectives)

        population, objectives = nsga2_environmental_selection(combined_population, combined_objectives, params.population_size)
        primary_scores = [obj[1] for obj in objectives]

        best_idx = argmax(primary_scores)
        if primary_scores[best_idx] > best_fit
            best_fit = primary_scores[best_idx]
            best_bits = copy(population[best_idx])
        end

        push!(best_so_far, best_fit)
    end

    final_fronts, _, _ = nsga2_rank_and_crowding(objectives)
    pareto_size = isempty(final_fronts) ? 0 : length(final_fronts[1])

    (
        seed=seed,
        best_fitness=best_fit,
        best_bits=best_bits,
        best_bitstring=String(join(Int.(best_bits))),
        best_so_far=best_so_far,
        pareto_size=pareto_size,
    )
end
