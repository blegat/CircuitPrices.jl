# Section 5. Sign access to affine forms in approximate logarithms. A proxy
# stores either a short dyadic value or the positive minor whose log it denotes.
struct _Log{T}
    value::T
    logarithmic::Bool
end

function _proxy(a, precision, cutoff_bits)
    a == 1 && return _Log(0, false)
    z = max(a, inv(a))
    two = one(a) + one(a)
    k = big(1)
    for _ in 0:cutoff_bits
        if z < two^k
            low, high = k ÷ 2, k
            while high - low > 1
                mid = (low + high) ÷ 2
                if z < two^mid
                    high = mid
                else
                    low = mid
                end
            end
            z /= two^low
            grid = big(2)^precision
            if !isone(z)
                series(t) = sum(t^j / j for j in 1:(precision+6))
                fraction = series(1 - inv(z)) / series(Q(1//2))
                digit = _last_integer(0, grid) do t
                    2t - 1 <= 2grid * fraction
                end
                low += digit // grid
            end
            return _Log(a < 1 ? -low : low, false)
        end
        k *= 2
    end
    return _Log(a, true)
end

function _logsign(form, proxies)
    coeff = form[2:end]
    constant = form[1]
    for (j, p) in pairs(proxies)
        if !p.logarithmic
            constant += coeff[j] * p.value
            coeff[j] = 0
        end
    end
    all(iszero, coeff) && return sign(constant)
    den = foldl(lcm, denominator.(coeff); init = denominator(constant))
    v = Q(2)^Integer(constant * den)
    for (j, p) in pairs(proxies)
        if !iszero(coeff[j])
            v *= p.value^Integer(coeff[j] * den)
        end
    end
    return sign(v - 1)
end

# Smallest-vector approximation via exact LLL (row basis, δ = 3/4).
function _gram(B)
    m = size(B, 1)
    G = copy(B)
    μ = zeros(Q, m, m)
    norms = zeros(Q, m)
    for i in 1:m
        for j in 1:(i-1)
            μ[i, j] = LA.dot(B[i, :], G[j, :]) / norms[j]
            G[i, :] .-= μ[i, j] .* G[j, :]
        end
        norms[i] = LA.dot(G[i, :], G[i, :])
    end
    return μ, norms
end

function _lll_first(B)
    B = copy(B)
    k = 2
    while k <= size(B, 1)
        for j in (k-1):-1:1
            μ, _ = _gram(B)
            B[k, :] .-= round(BigInt, μ[k, j]) .* B[j, :]
        end
        μ, norms = _gram(B)
        if norms[k] >= (3//4 - μ[k, k-1]^2) * norms[k-1]
            k += 1
        else
            B[k, :], B[k-1, :] = B[k-1, :], B[k, :]
            k = max(2, k - 1)
        end
    end
    return B[1, :]
end

# Largest integer in [lo, hi] satisfying a monotone test, or lo-1.
function _last_integer(test, lo, hi)
    lo, hi = big(lo) - 1, big(hi) + 1
    while hi - lo > 1
        mid = fld(lo + hi, 2)
        if test(mid)
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

function _scales(proxies, C, H, inverse_eps)
    m = length(proxies)
    unit = Matrix{Q}(Identity, m + 1, m + 1)
    rest = [unit[:, j+1] for j in 1:m]
    constant = unit[:, 1]
    scales = Vector{Q}[]
    for _ in 1:m
        X = zero(constant)
        for form in rest
            for candidate in (form, -form)
                if _logsign(candidate - X, proxies) > 0
                    X = candidate
                end
            end
        end
        _logsign(X - H * constant, proxies) <= 0 && break
        grid = 2C * inverse_eps
        u = map(rest) do form
            _last_integer(-grid, grid) do t
                _logsign(grid * form - t * X, proxies) >= 0
            end // grid
        end
        lattice = [unit[1:m, 1:m] zeros(Q, m); transpose(u) 1//C]
        shortest = _lll_first(lattice)
        q = abs(BigInt(shortest[end] * C))
        z = round.(BigInt, q .* u .- sign(shortest[end]) .* shortest[1:m])
        λ = X / q
        push!(scales, λ)
        rest = [form - digit * λ for (form, digit) in zip(rest, z)]
    end
    return scales
end

mutable struct _Compression
    D::BigInt
    M::BigInt
    precision::BigInt
    cutoff_bits::BigInt
    radius::BigInt
    learned::Vector{_Log}
    scales::Vector{Vector{Q}}
end

function _parameters(m, M)
    inverse_eps = 100(m + 1) * M
    C = big(2)^(m * (m + 1)) * (2inverse_eps)^(m + 1)
    H = 100M * C
    base = 10M * (m * H + 3)
    return (; inverse_eps, C, H, base)
end

function _compression(d, K)
    # There are at most binomial(2d,d)-1 nonempty square minors.
    D = min((big(d) + 1)^6, binomial(2big(d), d))
    precision = K + _bits(4D - 1)
    grid = big(2)^precision
    A = big(2)^K * (D + 4) * grid
    M = (D + 1) * A^(D + 1)
    worst = _parameters(D, M)
    U = worst.base^(D + 2) * grid
    F = (D + 2) * (A * U + 1)^(D + 1) * U * grid
    # After clearing denominators ≤ C^D, scale coefficients are ≤ (2C)^D.
    # A digit comparison combines ≤ D+1 such forms with coefficients bounded
    # by the grid, H, mC, 10M and 2C/ε (including final offset rounding).
    cutoff_bits = 10(D + 2) * (_bits(worst.base) + precision + _bits(D))
    return _Compression(D, M, precision, cutoff_bits, 4F, _Log[], Vector{Q}[])
end

function _request!(state, magnitude)
    p = _proxy(magnitude, state.precision, state.cutoff_bits)
    J = state.learned
    m = length(J)
    (; C, H, base) = _parameters(m, state.M)
    proxies = [J; p]
    rest = zeros(Q, m + 2)
    rest[end] = 1
    constant = zeros(Q, m + 2)
    constant[1] = 1
    grid = big(2)^state.precision
    digits = big(0)
    for scale in state.scales
        λ = [scale; 0]
        M10 = 10state.M
        digit = _last_integer(-m*C, m*C) do k
            _logsign(M10 * rest - (M10*k - 1) * λ, proxies) >= 0
        end
        rest -= digit * λ
        if digit < -m*C || _logsign(M10 * rest - λ, proxies) > 0
            return _restart!(state, p)
        end
        digits = (digits + digit) * base
    end
    limit = m * H + 2
    if any(s -> _logsign(s * rest - limit * constant, proxies) > 0, (-1, 1))
        return _restart!(state, p)
    end
    offset = _last_integer(-limit*grid, limit*grid) do t
        _logsign(grid * rest - t * constant, proxies) >= 0
    end
    return digits + offset // grid
end

function _restart!(state, proxy)
    push!(state.learned, proxy)
    params = _parameters(length(state.learned), state.M)
    state.scales = _scales(state.learned, params.C, params.H, params.inverse_eps)
    return nothing
end
