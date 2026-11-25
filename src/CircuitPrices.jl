module CircuitPrices

import LinearAlgebra as LA

const Q = Rational{BigInt}
const Identity = LA.I

include("ellipsoid.jl")
include("compression.jl")
include("calibration.jl")
include("market.jl")
include("prices.jl")
include("circuits.jl")
include("solve.jl")

end
