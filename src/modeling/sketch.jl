abstract type DistributionSketch <: Function end

n_params(sketch::DistributionSketch) = 0

struct CategoricalSketch{A <: AbstractArray} <: DistributionSketch
    n_atoms::Integer
    Vmin::Real
    Vmax::Real
    atoms::A
end

function CategoricalSketch(n_atoms, Vmin, Vmax)
    CategoricalSketch(
        n_atoms,
        Vmin,
        Vmax,
        LinRange(Vmin, Vmax, n_atoms)
    )
end

n_params(sketch::CategoricalSketch) = sketch.n_atoms

(sketch::CategoricalSketch)(η::Categorical) = (sketch.atoms, η.p)
(sketch::CategoricalSketch)(η::DiscreteNonParametric) = (sketch.atoms, probs(η))
(sketch::CategoricalSketch)(η::MixtureModel) = (sketch.atoms, η.prior.p)

struct QuantileSketch <: DistributionSketch
    n_atoms::Integer
    τ̂::Vector{<:Real}
end

function QuantileSketch(n_atoms::Integer)
    τ̂ = (Vector{Float64}(1:n_atoms) .- 0.5) ./ n_atoms
    QuantileSketch(n_atoms, τ̂)
end

n_params(sketch::QuantileSketch) = sketch.n_atoms

(sketch::QuantileSketch)(η::UnivariateDistribution) = Statistics.quantile.(η, sketch.τ̂)
