# Central cuts with dyadic rounding. The separator either accepts its argument
# (returns nothing) or gives an outward normal. Failure excludes an inscribed
# Euclidean ball of radius r in the initial ball of radius R.
_bits(n::Integer) = ndigits(n, base = 2)
function _ellipsoid(separate, n, R, r)
    c = zeros(Q, n)
    n == 0 && return isnothing(separate(c)) ? c : nothing
    ratio = max(big(1), ceil(BigInt, R / r))
    steps = 16n * (n + 1) * _bits(ratio)
    if n == 1
        width = Q(R)
        for _ in 1:steps
            g = separate(c)
            isnothing(g) && return c
            width /= 2
            c[1] -= sign(only(g)) * width
        end
        return nothing
    end
    H = Matrix{Q}(Identity, n, n) * R^2
    # Each exact cut retains at least 1/4 of the preceding minimum
    # eigenvalue. The grid thus resolves even the narrowest final axis.
    η = 1 // (big(100) * n^2)
    grid = one(BigInt) << (4steps + 2 * _bits(n) + 16)
    for _ in 1:steps
        g = separate(c)
        isnothing(g) && return c
        v = H * g
        s = LA.dot(g, v)
        @assert s > 0
        # Approximate sqrt(s) with relative error at most η. Scaling first
        # keeps the square-root computation valid for s much smaller than one.
        scale =
            one(BigInt) << max(
                0,
                cld(_bits(denominator(s)) - _bits(numerator(s)), 2) +
                _bits(denominator(η)) +
                3,
            )
        root = isqrt(fld(numerator(s) * scale^2, denominator(s))) // scale
        c .-= v ./ ((n + 1) * root)
        c .= round.(BigInt, grid .* c) .// grid
        H = (big(n)^2 // (n^2 - 1)) .* (H - (2 // (n + 1)) .* (v * v') ./ s)
        # In the new ellipsoid norm the square-root and center rounding
        # errors total less than 2η. Add a diagonal bound for matrix rounding.
        H .*= (1 + 2η)^2
        H .= round.(BigInt, grid .* H) .// grid
        for j in 1:n
            H[j, j] += n // grid
        end
    end
    return nothing
end

# Decide a bounded rational LP even when it has empty interior. Integerize
# each row. A normalized Farkas multiplier is a convex combination of
# vertices with denominator ≤ den_bound, by Cramer's rule.
function _short_feasible(A, b, bound)
    m, n = size(A)
    quick = _phase_one(hcat(A, -A, Matrix{Q}(Identity, m, m)), b, (m + n)^2)
    !isnothing(quick) && return quick
    return _bounded_feasible(A, b, bound)
end

function _bounded_feasible(A, b, bound)
    n = size(A, 2)
    rows = hcat(A, b)
    for i in axes(rows, 1)
        rows[i, :] .*= foldl(lcm, denominator.(rows[i, :]); init = big(1))
    end
    M = max(one(Q), maximum(abs, rows; init = zero(Q)))
    den_bound = ((n + 1) * M)^(n + 1)
    margin = inv(2den_bound)
    function separate(x)
        for i in axes(rows, 1)
            a = @view rows[i, 1:n]
            if LA.dot(a, x) > rows[i, end] + margin
                return a
            end
        end
        return nothing
    end
    # Zero normals witnessing inconsistency are handled before central cuts.
    if any(i -> all(iszero, rows[i, 1:n]) && rows[i, end] < 0, axes(rows, 1))
        return false
    end
    R = max(1, n) * (bound + 1)
    r = margin / (2 * max(n, 1) * M)
    return !isnothing(_ellipsoid(separate, n, R, r))
end
