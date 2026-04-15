struct TriangleLandscape
    n::Int
    m::Int
    s::Int
end

function load_triangle_landscape(; n::Int=16, m::Int=1, s::Int=4)
    TriangleLandscape(n, m, s)
end

function fitness_bits(landscape::TriangleLandscape, bits::AbstractVector{Bool})
    length(bits) == landscape.n || error("Bit length $(length(bits)) != n $(landscape.n)")

    return triangle_function(bits, landscape.n, landscape.m, landscape.s)
end

function decimal_to_bits(decimal::Int, n::Int)
    bits = falses(n)
    for i in 1:n
        shift = n - i
        bits[i] = ((decimal >> shift) & 0x1) == 0x1
    end
    bits
end

function triangle_function(bits::AbstractVector{Bool}, n::Int, m::Int, s::Int)
    # From Assignment 3 Lecture, p. 23
    n_ones = sum(bits)
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

    for decimal in 0:(total_states - 1)
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
