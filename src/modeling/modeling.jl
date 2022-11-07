using Distributions
using Statistics

export DistributionSketch,
    ImputationStrategy,
    DistributionProjection,
    CategoricalSketch,
    QuantileSketch,
    CategoricalImputation,
    QuantileImputation,
    CramérProjection

include("sketch.jl")
include("imputation.jl")
include("projection.jl")
