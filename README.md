# CircuitPrices.jl

CircuitPrices is a Julia reference implementation of the linear programming algorithm in Sections 3–6 of *Strongly polynomial linear programming*. It solves `maximize cᵀx subject to Ax ≤ b` with free variables and returns exact certificates. The name refers to the coordinate prices used to expose the support of a circuit. [OpenAI preprint, November 24, 2026](https://github.com/openai/math/blob/main/preprints/Strongly-polynomial-linear-programming-November-24-2026/paper.pdf)

From the package directory, start Julia with `julia --project=.`.

```julia
import CircuitPrices

A = [1 1; -1 0; 0 -1]
b = [3, 0, 0]
c = [1, 1]
s = CircuitPrices.solve(A, b, c)

@assert s.status == :optimal
@assert s.objective == 3
@assert A' * s.y == c
@assert all(s.slack + s.y .> 0)
```

| Status | Certificate | Identities to check |
| --- | --- | --- |
| `:optimal` | `x`, `y`, `slack`, `objective` | s = b − Ax ≥ 0, y ≥ 0, Aᵀy = c, sᵀy = 0, s + y > 0 |
| `:unbounded` | `x`, `ray` | Ax ≤ b, A·ray ≤ 0, cᵀray > 0 |
| `:infeasible` | `y` | y ≥ 0, Aᵀy = 0, bᵀy = −1, at most 1 + rank(A) nonzeros |

The default scalar type is `Rational{BigInt}`. Floating-point inputs denote their exact stored values. Use `1//10` for the rational one tenth. The `field` keyword accepts an exact ordered field type with Julia scalar arithmetic, comparisons, conversion from integers and rationals, and generic linear algebra support. The tests exercise Q(√2).

## Implementation

| Paper | Implementation |
| --- | --- |
| §3 | Homogenized slack spaces, coordinate deletion, circuit recovery and the strictly complementary optimality system |
| §4 | Cofactor pivots, reciprocal scaling, dyadic basis approximation and one-sided visibility tests |
| §5 | Online minor-log proxies, exact LLL, simultaneous digit extraction and restarts |
| §6 | Four-slot expert groups, lattice-point counts and first moments, half-integral market prices and recursive wealth updates |

By default each side decision starts with an exact phase-I search with a dimension-dependent pivot limit. Hitting the limit invokes pricing. Set `pricing = true` to begin with a visibility query at zero prices. A proposed partition can terminate the search early when exact nonnegative vectors in the original orthogonal spaces certify every side. For explicit rationals, bounds on the magnitudes of all nonzero minors can also permit direct calibration. The code limits the number of doublings used to find this bound.

The package targets small research examples. It uses dense exact arithmetic. The full price search prescribes 524,289 probes **per non-root internal node**, with greater branching at the root. Tests exercise the certificate path, calibration, compression, market minimization and reduced recursive trees. The full production-size recursion and the implementation's end-to-end complexity bound have not been independently validated.

## JuliaSmoothOptimizers models

Loading `QuadraticModels` activates `CircuitPrices.solve(model)` for a matrix-backed `QuadraticModel` with a linear objective. This interface accepts both objective senses, an objective constant, and two-sided linear and variable bounds. The `rows` field identifies the original bound for each canonical inequality, in the same order as `y` or `slack`. For minimization, those multipliers certify the maximization of the negated linear objective. [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl)

```julia
import CircuitPrices
import QuadraticModels

model = QuadraticModels.QuadraticModel(
    [2.0], zeros(1, 1); lvar = [0.5], uvar = [Inf],
)
s = CircuitPrices.solve(model)
@assert s.objective == 1
@assert only(s.rows) == (:variable_lower => 1)
```

MadNLP's filter line-search IPM and KKT solvers can produce numerical candidates for ordinary LP subproblems. An exact acceptance test and a bounded fallback would be needed to use such candidates in a pricing decision. CircuitPrices uses a rational feasibility test for those decisions. [MadNLP documentation](https://madsuite.org/MadNLP.jl/stable/)

## Tests

Run from the package directory.

```sh
julia --project=test -e 'import Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test test/runtests.jl
```
