abstract type ImputationStrategy <: Function end

struct CategoricalImputation <: ImputationStrategy
    locs::AbstractArray{<:Real}
end

CategoricalImputation(sketch::CategoricalSketch) = CategoricalImputation(sketch.atoms)

function (Φ::CategoricalImputation)(params::AbstractArray{<:Real})
    DiscreteNonParametric(Φ.locs, params)
end

struct QuantileImputation <: ImputationStrategy end

(::QuantileImputation)(params::Vector{<:Real}) =
    DiscreteNonParametric(params, ones(length(params)) ./ length(params))
