import CircuitPrices as CP
import QuadraticModels as QM
using Test

@testset "QuadraticModels LP input" begin
    H = zeros(2, 2)
    m = QM.QuadraticModel(
        [1.0, 2.0],
        H;
        A = [1.0 1.0],
        lcon = [1.0],
        ucon = [Inf],
        lvar = [0.0, 0.0],
        uvar = [1.0, Inf],
        c0 = 0.5,
    )
    s = CP.solve(m)
    @test s.status == :optimal && s.objective == 3//2
    @test s.x == [1, 0]
    @test s.rows == [
        :constraint_lower => 1,
        :variable_upper => 1,
        :variable_lower => 1,
        :variable_lower => 2,
    ]
    q = QM.QuadraticModel([1.0], [1.0;;])
    @test_throws ArgumentError CP.solve(q)
    m = QM.QuadraticModel([1.0], zeros(1, 1); minimize = false)
    @test CP.solve(m).status == :unbounded
end

@testset "Two-sided JSO row bounds" begin
    model = QM.QuadraticModel(
        [2.0],
        Int[],
        Int[],
        Float64[];
        Arows = [1],
        Acols = [1],
        Avals = [3.0],
        lcon = [1.0],
        ucon = [2.0],
        minimize = false,
        c0 = -0.5,
    )
    s = CP.solve(model)
    @test s.objective == 5//6
    @test s.y == [2//3, 0]
    @test s.rows == [:constraint_upper => 1, :constraint_lower => 1]
    model.data.c0 = NaN
    @test_throws ArgumentError CP.solve(model)
end

@testset "Effective JSO Hessian" begin
    model = QM.QuadraticModel(
        [0.0, 0.0],
        [-2.0 9.0; 0.0 0.0];
        regularize = true,
        selected = 1:1,
        σ = 2.0,
    )
    @test CP.solve(model).objective == 0
    model.data.σ = 3.0
    @test_throws ArgumentError CP.solve(model)
end
