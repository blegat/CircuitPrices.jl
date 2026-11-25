# Column operations are avoided so the free coordinates retain their indices.
function _reduce(A, n = size(A, 2))
    B = copy(A)
    pivots = Int[]
    for j in 1:n
        i = length(pivots) + 1
        k = findfirst(!iszero, @view B[i:end, j])
        isnothing(k) && continue
        k += i - 1
        B[i, :], B[k, :] = B[k, :], B[i, :]
        B[i, :] ./= B[i, j]
        for k in axes(B, 1)
            if k != i && !iszero(B[k, j])
                B[k, :] .-= B[k, j] .* B[i, :]
            end
        end
        push!(pivots, j)
    end
    return B, pivots
end

function _kernel(A)
    T = eltype(A)
    B, pivots = _reduce(A)
    free = setdiff(axes(A, 2), pivots)
    V = zeros(T, size(A, 2), length(free))
    V[free, :] = Matrix{T}(Identity, length(free), length(free))
    V[pivots, :] = -B[eachindex(pivots), free]
    return V
end

# Phase I for M*x = b, x ≥ 0. A dimension-only pivot budget makes this an
# optional shortcut. Reaching it hands the original problem to the price oracle.
function _phase_one(M, b, budget)
    m, n = size(M)
    T = hcat(ifelse.(b .< 0, -M, M), Matrix{eltype(M)}(Identity, m, m), abs.(b))
    basis = collect((n+1):(n+m))
    for step in 0:budget
        artificial = findall(>(n), basis)
        if all(iszero, T[artificial, end])
            return true
        end
        entering = findfirst(1:(n+m)) do j
            (j > n) < sum(@view T[artificial, j])
        end
        isnothing(entering) && return false
        step == budget && break
        rows = findall(>(0), T[:, entering])
        leaving = argmin(i -> (T[i, end] / T[i, entering], basis[i]), rows)
        T[leaving, :] ./= T[leaving, entering]
        for i in axes(T, 1)
            if i != leaving && !iszero(T[i, entering])
                T[i, :] .-= T[i, entering] .* T[leaving, :]
            end
        end
        basis[leaving] = entering
    end
    return nothing
end

# V is a column basis. The positive cone contains a witness at i iff
# V*a ≥ eᵢ is feasible. The nonnegative slacks give a phase-I shortcut.
function _positive_side(V, i)
    T = eltype(V)
    N, k = size(V)
    e = zeros(T, N)
    e[i] = 1
    yes = _phase_one(hcat(V, -V, -Matrix{T}(Identity, N, N)), e, (N + k)^2)
    if isnothing(yes)
        return _price_sides(V)[i] > 0
    end
    return yes
end

# Section 3.2. Every restriction is computed from the original basis.
function _circuit(V, i, positive_side)
    retained = Int[]
    W = V
    for j in axes(V, 1)
        if j == i || all(iszero, @view W[j, :])
            continue
        end
        push!(retained, j)
        trial = V * _kernel(V[retained, :])
        if positive_side(trial, i)
            W = trial
        else
            pop!(retained)
        end
    end
    @assert size(W, 2) == 1
    return W[:, 1] / W[i, 1]
end

function _feasible(A, b, positive_side)
    T = eltype(A)
    m, n = size(A)
    S = [(-A) b; zeros(T, 1, n) 1]
    _, pivots = _reduce(S)
    V = S[:, pivots]
    if !positive_side(V, m + 1)
        y = _circuit(_kernel(Matrix(transpose(V))), m + 1, positive_side)
        return (; feasible = false, y = y[1:m])
    end
    s = _circuit(V, m + 1, positive_side)
    B, columns = _reduce(hcat(A, b - s[1:m]), n)
    x = zeros(T, n)
    x[columns] = B[eachindex(columns), end]
    return (; feasible = true, x)
end
