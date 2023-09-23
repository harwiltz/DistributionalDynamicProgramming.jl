using Distributions
using Statistics

export DistributionalRepresentation,
    CategoricalRepresentation,
    QuantileRepresentation,
    DistributionSketch,
    ImputationStrategy,
    DistributionProjection,
    CategoricalSketch,
    QuantileSketch,
    CategoricalImputation,
    QuantileImputation,
    CramérProjection,
    Wasserstein1Projection,
    n_params

include("sketch.jl")
include("imputation.jl")
include("projection.jl")

struct DistributionalRepresentation
    sketch::DistributionSketch
    Φ::ImputationStrategy
    Π::DistributionProjection
end

n_params(repr::DistributionalRepresentation) = n_params(repr.sketch)

function CategoricalRepresentation(n_atoms::Int, vmin::Real, vmax::Real)
    sketch = CategoricalSketch(n_atoms, vmin, vmax)
    DistributionalRepresentation(
        sketch,
        CategoricalImputation(sketch),
        CramérProjection(sketch)
    )
end

function QuantileRepresentation(n_atoms::Int)
    DistributionalRepresentation(
        QuantileSketch(n_atoms),
        QuantileImputation(),
        Wasserstein1Projection(n_atoms)
    )
end
