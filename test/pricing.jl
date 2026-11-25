import CircuitPrices as CP
import LinearAlgebra as LA
using Test

@testset "Calibration, visibility and exact side certificates" begin
    Q = CP.Q
    B = Q[1//16 1 0 0; 0 0 1 -1//16]
    H = big(2)^400
    zero_basis = CP._direct_basis(B, Q[0, 0], 20)
    @test CP._labels(zero_basis, H) == [-1, 1]
    exposed = CP._direct_basis(B, Q[-4, 0], 20)
    @test CP._labels(exposed, H) == [1, 1]
    half = CP._direct_basis(Q[1 1], Q[1//2], 16)
    @test half == [1//2 1]
    half = CP._direct_basis(Q[1 1 0 0], Q[1//2, 0], 16)
    @test abs(half[1]^2 - 1//2) < 1//2^15
    @test CP._minor_radius(B) >= 5
    @test CP._certifies(B[1:1, 1:2]', [true, true])
    @test !CP._certifies(B[1:1, 1:2]', [false, true])
    @test CP._phase_one(Q[1;;], Q[1], 0) === nothing
    @test CP._bounded_feasible(Q[1; -1;;], Q[1, -1], 2)
    @test !CP._bounded_feasible(Q[1; -1;;], Q[0, -1], 2)
    @test !CP._bounded_feasible(Q[1 0; -1 0], Q[0, -1], 2)
    @test !CP._bounded_feasible(Q[0 0], Q[-1], 1)
    x = CP._ellipsoid(2, Q(4), Q(1//8)) do x
        x[1] < 1 && return Q[-1, 0]
        x[2] < 1 && return Q[0, -1]
        return nothing
    end
    @test all(>=(1), x)
end

@testset "Log proxies, lattice reduction and calibration restarts" begin
    Q = CP.Q
    @test CP._proxy(Q(1), 8, 4).value == 0
    @test abs(CP._proxy(Q(3//7), 10, 5).value - log2(3/7)) < 1/1024
    logs = [CP._proxy(Q(2)^100, 4, 3), CP._Log(Q(1//2), false)]
    @test logs[1].logarithmic
    @test CP._logsign(Q[-201//2, 1, 1], logs) == 0
    @test CP._logsign(Q[-100, 1, 1], logs) > 0
    @test CP._logsign(Q[-1, 0, 1], logs) < 0
    expanded = CP._Compression(9, 2, 4, 30, 100, [CP._Log(Q(1000), false)], [Q[0, 1]])
    @test CP._request!(expanded, Q(2)^1001) == CP._parameters(1, big(2)).base + 1
    @test isnothing(CP._request!(expanded, Q(2)^500))
    # Reduced precision and M exercise the same digit and restart logic.
    state = CP._Compression(9, 2, 4, 30, 100, CP._Log[], Vector{Q}[])
    @test CP._request!(state, Q(2)) == 1
    @test isnothing(CP._request!(state, Q(8)))
    @test CP._request!(state, Q(8)) == 3
    @test CP._request!(state, Q(1//2)) == -1
    scales = CP._scales([CP._Log(100, false), CP._Log(50, false)], big(1024), 4, big(8))
    @test !isempty(scales) && length(scales) <= 2
    @test CP._exp_dyadic(Q(-20), 10) == 0
    @test abs(CP._exp_dyadic(Q(-1//2), 12)^2 - 1//2) < 1//2000
    B = Q[8 -1]
    fresh = CP._Compression(9, 2, 4, 30, 100, CP._Log[], Vector{Q}[])
    @test isnothing(CP._compressed_basis!(B, Q[0], 8, fresh))
    @test CP._compressed_basis!(B, Q[0], 8, fresh) == Q[1 -1//8]
end

@testset "Implicit expert moments and recursive wealth" begin
    Q = CP.Q
    B = CP._bound(CP._box(3, 2), 2, 3, -1)
    B = CP._bound(B, 4, 2, 1)
    tuples = [[i, j, k] for i in -2:2 for j in -2:2 for k in -2:2 if i < j && k <= i + 1]
    @test CP._moments(B) == [length(tuples); sum(tuples)]
    @test CP._bound(B, 3, 2, 0) === nothing
    @test CP._depth(2) == 7
    # Enumerate only in the test, as an independent market oracle.
    cell = CP._Cell((1, -1, 1, -1), 1, -1, CP._box(5, 1), Q[2], Vector{Int}[])
    history = [Q[1//2]]
    p = Q[-1//2]
    expected = sum(Iterators.product(ntuple(_ -> -1:1, 5)...)) do bids
        τ = bids[5] + history[1][1]
        2 * max(0, maximum(bids[i] - τ - sign(cell.slots[i]) * p[1] for i in 1:4))
    end
    @test first(CP._utility([cell], history, p)) == expected
    # Singleton cells with opposite constant bids have a unique minimum at 1.
    cells = [
        CP._Cell(ntuple(_ -> s, 4), 0, 0, CP._box(4, 0), Q[1], Vector{Int}[]) for
        s in (1, -1)
    ]
    for (cell, shift) in zip(cells, (1, -1))
        cell.bounds[1, 2:end] .= -shift
        cell.bounds[2:end, 1] .= shift
    end
    @test CP._proposal(cells, Vector{Q}[], 1, big(1), 2) == [1]
    # A small tree tests median feedback, ties, investment and recombination.
    queries = Vector{Q}[]
    label(x) = (push!(queries, x); [1])
    x, after = CP._node(label, cells, Vector{Q}[], 1, big(1), 3, 2, (; root = 3, child = 1))
    @test x == [1] && length(queries) == 9
    @test all(c -> length(c.wealth) == 1 && isempty(c.gains), after)
    @test sum(c -> only(c.wealth), after) <= 2
    x, _ =
        CP._node(x -> nothing, cells, Vector{Q}[], 1, big(1), 1, 2, (; root = 1, child = 1))
    @test isnothing(x)
end

@testset "Nonzero prices and multivariate market clearing" begin
    Q = CP.Q
    @test CP._price_sides(reshape(Q[1//16, 1], 2, 1)) == [1, 1]
    @test CP._rounded_root(Q(1//1025), big(32)) == 0
    state = CP._compression(1, 1)
    @test state.radius > 0 && CP._request!(state, Q(1)) == 0
    @test CP._exp_dyadic(Q(-2), 10) == 1//4
    # Each coordinate has two hinge terms meeting at its optimum.
    cells = CP._Cell[]
    for s in (1, -1, 2, -2)
        B = CP._box(4, 0)
        B[1, 2:end] .= -s
        B[2:end, 1] .= s
        push!(cells, CP._Cell(ntuple(_ -> s, 4), 0, 0, B, Q[1], Vector{Int}[]))
    end
    @test CP._proposal(cells, Vector{Q}[], 2, big(2), 4) == [1, 2]
end

@testset "Price bounds and restart propagation with production constants" begin
    @test CP._prices(p -> [1, -1], 2, big(10)^30) == [big(10)^30, -big(10)^30]
    @test CP._prices(p -> nothing, 1, big(1)) === nothing
    @test CP._prices(p -> iszero(p) ? [1] : nothing, 1, big(1)) === nothing
    queries = Vector{CP.Q}[]
    result = CP._prices(1, big(1)) do x
        push!(queries, x)
        length(queries) == 1 && return [1]
        length(queries) == 2 && return [-1]
        return nothing
    end
    @test isnothing(result)
    @test queries == [[0], [1], [0]]
end

@testset "Compressed price search across a calibration restart" begin
    # The nonzero minor 8 forces a restart from the empty scale set. The
    # one-coordinate space has a positive witness at every price. Both
    # queries in the second search can therefore certify the positive corner.
    x = CP._compressed_prices(CP.Q[8 0], 1, 8, big(2)^100)
    @test only(x) > 0
end
