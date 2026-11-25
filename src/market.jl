# Section 6.6. Each group has four demand variables and optionally a fifth
# for the pivot. Index 1 in the difference-bound matrix is the constant zero.
# bounds[i,j] is an integral upper bound on guess[i] - guess[j].
struct _Cell
    slots::NTuple{4,Int}
    level::Int
    pivot::Int
    bounds::Matrix{BigInt}
    wealth::Vector{Q} # local wealth at every active level
    gains::Vector{Vector{Int}} # maximizing slots at saved proposals (0 = outside)
end

function _box(d, R)
    bounds = fill(big(2) * R, d + 1, d + 1)
    bounds[1, :] .= R
    bounds[:, 1] .= R
    for j in axes(bounds, 1)
        bounds[j, j] = 0
    end
    return bounds
end

# A closed difference-bound matrix needs just one shortest-path update
# when one new edge is inserted. A negative cycle means an empty cell.
function _bound(B, i, j, t)
    B[i, j] <= t && return B
    B[j, i] + t < 0 && return nothing
    return min.(B, B[:, i] .+ t .+ transpose(B[j, :]))
end

# Count integer points and sum each coordinate. Eliminate the last variable
# between an active lower and upper bound. First-index tie breaking gives
# disjoint pieces. A finite-difference antiderivative sums the polynomial
# over the eliminated coordinate, including intervals below zero.
function _sum_cell(f, B, degree)
    n = size(B, 1)
    n == 1 && return f(BigInt[0])
    total = nothing
    for l in 1:(n-1), u in 1:(n-1)
        piece = B[1:(n-1), 1:(n-1)]
        for j in 1:(n-1)
            piece = _bound(piece, j, l, B[j, n] - B[l, n] - (j < l))
            isnothing(piece) && break
            piece = _bound(piece, u, j, B[n, j] - B[n, u] - (j < u))
            isnothing(piece) && break
        end
        isnothing(piece) && continue
        # Closure of B already includes x[l]-B[l,n] ≤ x[u]+B[n,u].
        term = _sum_cell(piece, degree + 1) do x
            lo = x[l] - B[l, n]
            hi = x[u] + B[n, u]
            values = [f([x; k]) for k in 0:degree]
            value = zero(first(values))
            for k in 0:degree
                value += (binomial(hi + 1, k + 1) - binomial(lo, k + 1)) * first(values)
                values = diff(values)
            end
            return value
        end
        total = isnothing(total) ? term : total + term
    end
    return total
end

function _moments(B)
    lower, upper = -B[1, 2:end], B[2:end, 1]
    if all(
        i == j || B[i+1, j+1] >= upper[i] - lower[j] for
        i in eachindex(lower), j in eachindex(lower)
    )
        count = prod(upper - lower .+ 1)
        return [count; div.(count .* (upper + lower), 2)]
    end
    return _sum_cell(x -> [one(BigInt); x[2:end]], B, 1)
end

# Options are utilities guess[index] + offset, after adding the common
# threshold to the outside option. Signed slot indices encode both sides.
_price(p, s) = sign(s) * p[abs(s)]

function _options(cell, history, p)
    if cell.level > length(history)
        return [1], Q[0], [0]
    end
    indices = [cell.level == 0 ? 1 : 6; 2:5]
    outside = cell.level == 0 ? zero(Q) : -_price(history[cell.level], cell.pivot)
    offsets = [outside; [-_price(p, s) for s in cell.slots]]
    return indices, offsets, [0; collect(cell.slots)]
end

# Refine only on comparisons needed to find the maximizing options. Every
# bound comes from two original guesses. All leaves are intersections of
# intervals from the global arrangement, so repeated splits stay polynomial.
function _winners(visit, B, indices, offsets, slots, best = [1], k = 2)
    if k > length(slots)
        visit(B, best)
        return
    end
    a, b = indices[k], indices[first(best)]
    t = offsets[first(best)] - offsets[k]
    left = _bound(B, a, b, ceil(BigInt, t) - 1)
    if !isnothing(left)
        _winners(visit, left, indices, offsets, slots, best, k + 1)
    end
    if isinteger(t)
        tie = _bound(B, a, b, BigInt(t))
        if !isnothing(tie)
            tie = _bound(tie, b, a, -BigInt(t))
        end
        if !isnothing(tie)
            _winners(visit, tie, indices, offsets, slots, [best; k], k + 1)
        end
    end
    right = _bound(B, b, a, -floor(BigInt, t) - 1)
    if !isnothing(right)
        _winners(visit, right, indices, offsets, slots, [k], k + 1)
    end
end

function _utility(cells, history, p)
    value = zero(Q)
    gradient = zero(p)
    for cell in cells
        indices, offsets, slots = _options(cell, history, p)
        length(slots) == 1 && continue
        _winners(cell.bounds, indices, offsets, slots) do B, best
            k = first(best)
            s = slots[k]
            s == 0 && return
            moments = _moments(B)
            count = moments[1]
            intercept = moments[indices[k]]
            if cell.level > 0
                intercept -= moments[end] + count * offsets[1]
            end
            w = last(cell.wealth)
            value += w * (intercept + count * offsets[k])
            gradient[abs(s)] -= w * count * sign(s)
        end
    end
    return value, gradient
end

# Section 6.1. At a minimum, wealth can be split among maximizing options
# so their weighted slopes cancel the tilt. This gives an exact early exit.
function _clears(cells, history, p, tilt)
    weights = Dict{Tuple{Vararg{Int}},Q}()
    for cell in cells
        indices, offsets, slots = _options(cell, history, p)
        length(slots) == 1 && continue
        _winners(cell.bounds, indices, offsets, slots) do B, best
            winners = Tuple(sort!(unique(slots[best])))
            winners == (0,) && return
            count = first(_moments(B))
            weights[winners] = get(weights, winners, zero(Q)) + count * last(cell.wealth)
        end
    end
    groups = sort!(collect(keys(weights)))
    m = length(groups)
    columns = sum(length, groups; init = 0)
    A = zeros(Q, m + length(p), columns)
    j = 0
    for (k, group) in pairs(groups), s in group
        j += 1
        A[k, j] = 1
        if s != 0
            A[m+abs(s), j] = sign(s)
        end
    end
    b = [Q[weights[g] for g in groups]; tilt]
    return _phase_one(A, b, (size(A, 1) + columns)^2) === true
end

# Minimize the implicit convex piecewise-affine function on a box containing
# every market vertex. Section 6.6 gives half-integrality, also for clipped
# cells (box walls are integral). Lexicographic perturbation separates all
# vertex objectives. An ellipsoid approximation then recovers the unique
# minimum by rounding to the half-grid.
function _proposal(cells, history, N, R, experts)
    bound = (2big(N) + 2) * (2R + 1) * (length(history) + 1)
    if N == 1
        # On the half-grid the first differences of a convex function are
        # monotone. Find the first minimizer, retaining lexicographic ties.
        k = _last_integer(-2bound, 2bound - 1) do j
            v, _ = _utility(cells, history, Q[j//2])
            w, _ = _utility(cells, history, Q[(j+1)//2])
            w < v
        end
        return Q[(k+1)//2]
    end
    den = foldl(lcm, (denominator(last(c.wealth)) for c in cells); init = big(1))
    base = 16bound + 1
    tilt = [inv(Q(4den * base^i)) for i in 1:N]
    gap = inv(Q(8den * base^N))
    tolerance = gap / (32bound)
    best = zeros(Q, N)
    bestval = nothing
    last_candidate = nothing
    function cut(p)
        for i in eachindex(p)
            if abs(p[i]) > bound
                g = zeros(Q, N)
                g[i] = sign(p[i])
                return g
            end
        end
        candidate = round.(BigInt, 2p) .// 2
        if candidate != last_candidate
            last_candidate = candidate
            if _clears(cells, history, candidate, tilt)
                best = candidate
                return nothing
            end
        end
        value, g = _utility(cells, history, p)
        value += LA.dot(tilt, p)
        if isnothing(bestval) || value < bestval
            bestval, best = value, copy(p)
        end
        return g + tilt
    end
    # Any box-clipped cell is the convex hull of its vertices. Objective
    # error < tolerance gives nonoptimal vertices total mass < 1/(32bound).
    # A ball near the optimum inside the box has this error by Lipschitz
    # continuity (total wealth never exceeds the initial expert count).
    r = tolerance / (8N * (experts + N))
    _ellipsoid(cut, N, N * (bound + 1), r)
    return round.(BigInt, 2best) .// 2
end
