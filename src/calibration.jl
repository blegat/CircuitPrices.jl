# Fast case for explicit numerator/denominator input. A common denominator
# and Hadamard's bound enclose *all* nonzero minors, including unseen ones.
# The doubling loop is capped in the dimensions before switching to Section 5.
_minor_radius(B) = nothing

function _minor_radius(B::Matrix{Q})
    d = size(B, 2)
    den = prod(denominator, B; init = big(1))
    height = maximum(abs, B; init = Q(1)) * den
    bound = (d * max(den, height))^size(B, 1)
    k = big(1)
    for _ in 0:(big(d+1)^4)
        if bound <= big(2)^k
            return 2k + 3
        end
        k *= 2
    end
    return nothing
end

function _cofactors(B, pivots, j)
    return [s in pivots ? zero(eltype(B)) : LA.det(B[1:j, [pivots; s]]) for s in axes(B, 2)]
end

# Prices from the market are half-integral. For a dimension-bounded log
# range, squared scaled magnitudes are exact rationals. Quantize the square
# root only after pivoting, when its value is in [0,1].
function _rounded_root(q, grid)
    iszero(q) && return zero(Q)
    isone(q) && return one(Q)
    if q isa Rational && denominator(q) <= grid
        a, b = isqrt(numerator(q)), isqrt(denominator(q))
        if a^2 == numerator(q) && b^2 == denominator(q)
            return a // b
        end
        return isqrt(floor(BigInt, q * grid^2)) // grid
    end
    bound = q * grid^2
    scaled = _last_integer(v -> v^2 <= bound, 0, grid)
    return scaled // grid
end

function _direct_basis(B, prices, precision)
    scale2 = [Q(2)^BigInt(-2p) for p in [prices; -prices]]
    pivots = Int[]
    prev = one(eltype(B))
    answer = zeros(Q, size(B))
    grid = big(2)^precision
    for j in axes(B, 1)
        c = _cofactors(B, pivots, j)
        mag = c .^ 2 .* scale2
        t = argmax(mag)
        for s in eachindex(c)
            root = _rounded_root(mag[s] / mag[t], grid)
            answer[j, s] = (prev * c[s] < 0 ? -root : root)
        end
        prev = c[t]
        push!(pivots, t)
    end
    return answer
end

# Section 4.4. Evaluate the log(2) and exponential series on a fixed grid.
# The guard bits cover the sum of all downward roundings and both tails.
function _exp_dyadic(w, E)
    w < -E && return zero(Q)
    isinteger(w) && return Q(2)^BigInt(w)
    k = floor(BigInt, w)
    grid = big(2)^(E + _bits(E + 10) + 10)
    terms = E + 10
    log2_scaled = sum(fld(grid >> j, j) for j in 1:terms)
    argument = floor(BigInt, (w - k) * log2_scaled)
    term = grid
    value = grid
    for j in 1:terms
        term = fld(term * argument, grid * j)
        value += term
    end
    return Q(2)^k * (value // grid)
end

function _compressed_basis!(B, prices, precision, state)
    signed = [prices; -prices]
    pivots = Int[]
    prev = one(eltype(B))
    answer = zeros(Q, size(B))
    for j in axes(B, 1)
        c = _cofactors(B, pivots, j)
        support = findall(!iszero, c)
        scores = Q[]
        for s in support
            f = _request!(state, abs(c[s]))
            isnothing(f) && return nothing
            push!(scores, f - signed[s])
        end
        k = argmax(scores)
        for (l, s) in pairs(support)
            root = _exp_dyadic(scores[l] - scores[k], precision + 6)
            answer[j, s] = prev * c[s] < 0 ? -root : root
        end
        prev = c[support[k]]
        push!(pivots, support[k])
    end
    return answer
end

function _labels(B, H)
    r, slots = size(B)
    N = slots ÷ 2
    I_r = Matrix{Q}(Identity, r, r)
    A = [I_r; -I_r; B'; -B']
    b = [fill(Q(H), 2r); fill(Q(2), slots); fill(inv(Q(H)^10), slots)]
    return map(1:N) do i
        rhs = copy(b)
        rhs[2r+slots+i] = -1//4
        _short_feasible(A, rhs, H) ? 1 : -1
    end
end

function _price_sides(V)
    N, k = size(V)
    U = _kernel(Matrix(V'))
    B = [V' zeros(eltype(V), k, N); zeros(eltype(V), N - k, N) U']
    P = 10000 * big(N)^2
    H = big(2)^(P ÷ 100)
    # A zero-price query often exposes the entire partition immediately.
    # Certify its labels by constructing a strictly positive vector on each
    # proposed support. The orthogonal projection is only a candidate.
    p = zeros(Q, N)
    z = _labels(_direct_basis(B, p, P), H)
    if _certifies(V, z .> 0) && _certifies(U, z .< 0)
        return z
    end
    R = _minor_radius(B)
    trial_radius = isnothing(R) ? 2N : R
    label(p) = _labels(_direct_basis(B, p, P), H)
    # Combine positive labels observed at different prices. Every guess
    # (including their union or intersection) is checked in the unscaled
    # spaces, since an invisible coordinate can receive either label.
    positive = z .> 0
    possible, common = copy(positive), copy(positive)
    for i in 1:N, h in (-1, 1)
        p[i] = h * trial_radius
        positive = label(p) .> 0
        possible .|= positive
        common .&= positive
        for guess in (positive, possible, common)
            if _certifies(V, guess) && _certifies(U, .!guess)
                return ifelse.(guess, 1, -1)
            end
        end
        p[i] = 0
    end
    if !isnothing(R)
        return _prices(label, N, R)
    end
    return _compressed_prices(B, N, P, H)
end

function _compressed_prices(B, N, P, H)
    E = P + 6
    K = _bits((N + 1) * ((E + 5) * big(2)^E)^(N + 1) - 1)
    state = _compression(2N, K)
    while true
        x = _prices(N, state.radius) do p
            basis = _compressed_basis!(B, p, P, state)
            isnothing(basis) ? nothing : _labels(basis, H)
        end
        !isnothing(x) && return x
    end
end

function _certifies(V, support)
    !any(support) && return true
    W = V * _kernel(V[.!support, :])
    size(W, 2) == 0 && return false
    gram, _ = _reduce(hcat(W' * W, W' * convert(Vector{eltype(V)}, support)), size(W, 2))
    v = W * gram[:, end]
    return all(>(0), v[support])
end
