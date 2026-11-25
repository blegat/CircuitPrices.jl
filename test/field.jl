module TestOrderedField

import CircuitPrices as CP
using Test

# A copy of Q(√2). Exact comparison uses one rational comparison of squares
# when the signs of a and b differ. No conversion to floating point is defined.
struct Surd <: Real
    a::CP.Q
    b::CP.Q
end
Surd(x::Union{Integer,Rational}) = Surd(x, 0)
Base.convert(::Type{Surd}, x::Union{Integer,Rational}) = Surd(x)
Base.promote_rule(::Type{Surd}, ::Type{T}) where {T<:Union{Integer,Rational}} = Surd
Base.:(+)(x::Surd, y::Surd) = Surd(x.a + y.a, x.b + y.b)
Base.:(-)(x::Surd) = Surd(-x.a, -x.b)
Base.:(-)(x::Surd, y::Surd) = x + (-y)
Base.:(*)(x::Surd, y::Surd) = Surd(x.a*y.a + 2x.b*y.b, x.a*y.b + x.b*y.a)
function Base.inv(x::Surd)
    d = x.a^2 - 2x.b^2
    return Surd(x.a / d, -x.b / d)
end
Base.:(/)(x::Surd, y::Surd) = x * inv(y)
Base.:(==)(x::Surd, y::Surd) = x.a == y.a && x.b == y.b
function Base.:(<)(x::Surd, y::Surd)
    a, b = (x-y).a, (x-y).b
    return a < 0 ? (b <= 0 || a^2 > 2b^2) : (b < 0 && a^2 < 2b^2)
end
Base.:(<=)(x::Surd, y::Surd) = !(y < x)
Base.isfinite(::Surd) = true
Base.abs(x::Surd) = x < 0 ? -x : x

@testset "Ordered field coefficients" begin
    r = Surd(0, 1)
    s = CP.solve([r;;], [3], [1]; field = Surd)
    @test s.status == :optimal
    @test only(s.x) == Surd(0, 3//2)
    @test only(s.y) == Surd(0, 1//2)
    @test s.objective == only(s.x)
    @test CP.solve([-r;;], [-1], [r]; field = Surd).status == :unbounded
    @test CP.solve(zeros(Surd, 1, 0), [-r], Int[]; field = Surd, pricing = true).status ==
          :infeasible
    @test !CP._proxy(r, 8, 2).logarithmic
    @test CP._proxy(r, 8, 2).value == 1//2
    p = CP._proxy(r^20, 8, 2)
    @test p.logarithmic && CP._logsign(CP.Q[-10, 1], [p]) == 0
    @test_throws ArgumentError CP.solve([1;;], [0], [1]; field = Float64)
end

end
