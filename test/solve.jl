import CircuitPrices as CP
import LinearAlgebra as LA
using Test

@testset "Strict complementarity, duplicate rows and free variables" begin
    A = [1 1; 2 2; -1 0; 0 -1; 0 0]
    b = [3, 6, 0, 0, 0]
    c = [1, 1]
    s = CP.solve(A, b, c)
    @test s.status == :optimal
    @test s.objective == 3
    @test s.slack == b - A * s.x
    @test all(>=(0), s.slack) && all(>=(0), s.y)
    @test all(>(0), s.slack + s.y)
    @test A' * s.y == c
    @test LA.dot(s.slack, s.y) == 0
end

@testset "Farkas support and improving rays" begin
    A = BigInt[1 2; -1 -2; 2 4]
    b = [-1, -2, -2]
    s = CP.solve(A, b, [1, 2])
    @test s.status == :infeasible
    @test all(>=(0), s.y) && iszero(A' * s.y)
    @test LA.dot(b, s.y) == -1
    @test count(!iszero, s.y) <= 2
    A = [1 -1; 0 -1]
    b = [1//7, -3//2]
    c = [1, 0]
    s = CP.solve(A, b, c)
    @test s.status == :unbounded
    @test all(A * s.x .<= b)
    @test all(<=(0), A * s.ray) && LA.dot(c, s.ray) > 0
end

@testset "Empty systems, binary floats and extreme scales" begin
    @test CP.solve(zeros(Int, 0, 2), Int[], [0, 0]).objective == 0
    @test CP.solve(zeros(Int, 0, 2), Int[], [1, 0]).status == :unbounded
    @test CP.solve(zeros(Int, 2, 0), [-1, 0], Int[]).status == :infeasible
    @test CP.solve(zeros(Int, 0, 0), Int[], Int[]).status == :optimal
    @test CP.solve([1.0;;], [0.1], [1.0]).objective == Rational{BigInt}(0.1)
    u = 1 // (big(2)^2000 + 1)
    @test CP.solve([u;;], [1], [u^2]).objective == u
    @test_throws DimensionMismatch CP.solve([1;;], [0, 0], [0])
    @test_throws ArgumentError CP.solve([NaN;;], [0], [0])
    @test CP.solve([1;;], [2], [1]; pricing = true).objective == 2
    @test CP.solve([1;;], [2], [-1]; pricing = true).ray == [-1]
    s = CP.solve(zeros(Int, 1, 0), [-1], Int[]; pricing = true)
    @test s.status == :infeasible && only(s.y) == 1
end
