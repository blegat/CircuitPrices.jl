module CircuitPricesQuadraticModelsExt

import CircuitPrices as CP
import LinearAlgebra as LA
import QuadraticModels as QM

"""
    CircuitPrices.solve(model::QuadraticModels.QuadraticModel)

Solve a matrix-backed linear model with two-sided row and variable bounds.
`rows` maps canonical inequalities (and multipliers) to their original bounds.
Each entry is a pair such as `:constraint_lower => 3` or `:variable_upper => 1`.
Multipliers are nonnegative for the canonical maximization problem.
`objective` uses the model's original sense and constant term.
"""
function CP.solve(model::QM.QuadraticModel; field::Type = CP.Q, kwargs...)
    data, meta = model.data, model.meta
    if !(data.A isa AbstractMatrix && data.H isa AbstractMatrix)
        throw(ArgumentError("the model requires explicit constraint and Hessian matrices"))
    end
    # QuadraticModels evaluates the lower triangle and adds regularization
    # on the selected diagonal. Test the effective Hessian in the field.
    H = LA.LowerTriangular(Matrix{field}(data.H))
    if data.regularize
        for i in data.selected
            H[i, i] += convert(field, data.σ)
        end
    end
    if !iszero(H)
        throw(ArgumentError("the model must have a linear objective"))
    end
    n = meta.nvar
    keys = Pair{Symbol,Int}[]
    bounds = field[]
    vectors = Vector{field}[]
    function append_bound(v, rhs, key, i; lower = false)
        if !isfinite(rhs)
            if rhs == (lower ? -Inf : Inf)
                return
            end
            throw(ArgumentError("invalid bound on $key $i"))
        end
        push!(keys, key => i)
        push!(bounds, lower ? -convert(field, rhs) : convert(field, rhs))
        push!(vectors, lower ? -v : v)
    end
    # Convert before negating to preserve integer extrema and binary floats.
    A = Matrix{field}(data.A)
    for i in axes(A, 1)
        append_bound(A[i, :], meta.ucon[i], :constraint_upper, i)
        append_bound(A[i, :], meta.lcon[i], :constraint_lower, i; lower = true)
    end
    for i in 1:n
        v = zeros(field, n)
        v[i] = 1
        append_bound(v, meta.uvar[i], :variable_upper, i)
        append_bound(v, meta.lvar[i], :variable_lower, i; lower = true)
    end
    isfinite(data.c0) || throw(ArgumentError("the objective constant must be finite"))
    G = isempty(vectors) ? zeros(field, 0, n) : reduce(vcat, transpose.(vectors))
    c = Vector{field}(data.c)
    result = CP.solve(G, bounds, meta.minimize ? -c : c; field, kwargs...)
    if result.status == :optimal
        result =
            merge(result, (; objective = LA.dot(c, result.x) + convert(field, data.c0)))
    end
    return (; result..., rows = keys)
end

end
