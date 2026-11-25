function _depth(N)
    D = 1
    while big(5)^D <= 2N * big(4)^D
        D += 2
    end
    return D
end

function _groups(N, R, D)
    slots = [collect(1:N); collect((-N):-1)]
    cells = _Cell[]
    for tuple in Iterators.product(ntuple(_ -> slots, 4)...)
        push!(cells, _Cell(tuple, 0, 0, _box(4, R), Q[1], Vector{Int}[]))
        for j in 1:D, pivot in slots
            push!(cells, _Cell(tuple, j, pivot, _box(5, R), Q[1], Vector{Int}[]))
        end
    end
    return cells
end

function _invest(cells, history, p)
    children = _Cell[]
    for cell in cells
        indices, offsets, slots = _options(cell, history, p)
        _winners(cell.bounds, indices, offsets, slots) do bounds, best
            share = last(cell.wealth) / 16384
            wealth = [cell.wealth[1:(end-1)]; last(cell.wealth) - share; share]
            gains = copy(cell.gains)
            push!(gains, slots[best])
            push!(
                children,
                _Cell(cell.slots, cell.level, cell.pivot, bounds, wealth, gains),
            )
        end
    end
    return children
end

function _settle!(cells, feedback)
    for cell in cells
        winners = pop!(cell.gains)
        g = minimum(s -> s == 0 ? 0 : _price(feedback, s), winners)
        child = pop!(cell.wealth)
        cell.wealth[end] = cell.wealth[end] * (1 + g // 16) + child
    end
end

function _median(xs)
    k = (length(xs) + 1) ÷ 2
    return [partialsort!([x[i] for x in xs], k) for i in eachindex(first(xs))]
end

function _node(label, cells, history, N, R, D, experts, rounds; uniform = false)
    length(history) == D && return _median(history), cells
    proposals = Vector{Q}[]
    nrounds = isempty(history) ? rounds.root : rounds.child
    for probe in 1:nrounds
        # Before the very first feedback all experts have equal wealth and
        # all history prices are zero. Sign reversal pairs equal-weight
        # experts, while the two repeated-slot, zero-bid experts contribute
        # a positive multiple of |yᵢ|. Zero uniquely clears this market.
        symmetric = uniform && probe == 1
        y = symmetric ? zeros(Q, N) : _proposal(cells, history, N, R, experts)
        children = _invest(cells, history, y)
        push!(history, y)
        x, cells =
            _node(label, children, history, N, R, D, experts, rounds; uniform = symmetric)
        pop!(history)
        isnothing(x) && return nothing, cells
        z = label(x)
        isnothing(z) && return nothing, cells
        v = [x[i] == y[i] ? z[i] : Int(sign(x[i] - y[i])) for i in 1:N]
        _settle!(cells, v)
        push!(proposals, x)
    end
    return _median(proposals), cells
end

function _prices(label, N, R)
    z = label(zeros(Q, N))
    isnothing(z) && return nothing
    corner = Q.(R .* z)
    feedback = label(corner)
    isnothing(feedback) && return nothing
    # At R*z any unmet demand has sign opposite z on its coordinate.
    # If feedback agrees everywhere, a positive maximum surplus would
    # violate consistency, so this corner already satisfies Theorem 2.
    feedback == z && return corner
    D = _depth(N)
    options = big(2N) * (2R + 1)
    experts = options^4 * (1 + D * options)
    r = 2^19 + 1
    rounds = (; root = r + 3072 * _bits(experts), child = r)
    cells = _groups(N, R, D)
    return first(_node(label, cells, Vector{Q}[], N, R, D, experts, rounds; uniform = true))
end
