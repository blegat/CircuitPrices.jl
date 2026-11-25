"""
    solve(A, b, c; pricing = false, field = Rational{BigInt})

Set `pricing = true` to start side decisions at the zero-price query. The
default first tries a dimension-limited phase-I pivot search.

Maximize `c' * x` over `A * x ≤ b` with free variables. Finite integer,
rational and binary floating-point coefficients are converted exactly to
`Rational{BigInt}`. Supply an exact ordered scalar type via `field` for other coefficients.
The named tuple contains one certificate:

* `:optimal` — `x`, `y`, `slack` and `objective` (strictly complementary).
* `:unbounded` — a feasible `x` and an improving `ray`.
* `:infeasible` — nonnegative Farkas multipliers `y`, with `b' * y == -1`.
"""
function solve(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector;
    pricing::Bool = false,
    field::Type = Q,
)
    if field <: Union{Integer,AbstractFloat}
        throw(ArgumentError("field must support exact division and comparison"))
    end
    m, n = size(A)
    if length(b) != m || length(c) != n
        throw(DimensionMismatch("A must have length(b) rows and length(c) columns"))
    end
    if !(all(isfinite, A) && all(isfinite, b) && all(isfinite, c))
        throw(ArgumentError("LP coefficients must be finite"))
    end
    side = pricing ? _priced_side : _positive_side
    return _solve(Matrix{field}(A), Vector{field}(b), Vector{field}(c), side)
end

_priced_side(V, i) = _price_sides(V)[i] > 0

function _solve(A, b, c, side)
    T = eltype(A)
    m, n = size(A)
    primal = _feasible(A, b, side)
    if !primal.feasible
        return (; status = :infeasible, primal.y)
    end
    x = primal.x
    ray = _feasible([A; -c'], [zeros(T, m); -1], side)
    if ray.feasible
        return (; status = :unbounded, x, ray = ray.x)
    end

    # Section 3.3. Homogenize x and y by α ≥ 1. A strictly positive
    # slack-plus-multiplier can then be required to be at least one.
    I_m = Matrix{T}(Identity, m, m)
    Z = zeros(T, n, n)
    eq = [Z A' (-c); c' (-b') 0]
    G = [
        A zeros(T, m, m) (-b)
        zeros(T, m, n) (-I_m) zeros(T, m)
        eq
        -eq
        A (-I_m) (-b)
        zeros(T, 1, n + m) -1
    ]
    pair = _feasible(G, [zeros(T, 2m + 2n + 2); -ones(T, m + 1)], side)
    @assert pair.feasible
    v = pair.x / pair.x[end]
    x, y = v[1:n], v[n .+ (1:m)]
    return (; status = :optimal, x, y, slack = b - A * x, objective = LA.dot(c, x))
end
