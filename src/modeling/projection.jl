abstract type DistributionProjection <: Function end

const FiniteReprMixture = MixtureModel{Univariate, Discrete, <: DiscreteNonParametric, <: Categorical}

struct CramérProjection <: DistributionProjection
    locs::AbstractArray{<: Real}
end

CramérProjection(sketch::CategoricalSketch) = CramérProjection(sketch.atoms)
CramérProjection(Φ::CategoricalImputation) = CramérProjection(Φ.locs)

function (Π::CramérProjection)(μ::DiscreteNonParametric)
    target_locs = Π.locs
    locs = support(μ)
    p = probs(μ)
    N = length(target_locs)
    ζ = target_locs[2] - target_locs[1]
    vmax = target_locs[end]
    vmin = target_locs[1]
    clamped_locs = clamp.(locs, vmin, vmax)
    tiled_locs = reshape(repeat(clamped_locs, N), :, N)
    weights = clamp.(
        1 .- abs.(tiled_locs .- reshape(target_locs, 1, :)) ./ ζ,
        0,
        1
    ) .* p
    p̂ = reshape(sum(weights, dims = 1), :)
    DiscreteNonParametric(target_locs, p̂)
end

function (Π::DistributionProjection)(μ::CM) where CM <: FiniteReprMixture
    locs = vcat(support.(μ.components)...)
    weights = probs.(μ.components) .* μ.prior.p
    p = vcat(weights...)
    θ = unique(locs)
    pdict = Dict(t => 0.0 for t ∈ θ)
    for (l, p′) in zip(locs, p)
        pdict[l] += p′
    end
    ps = [pdict[t] for t in θ]
    Π(DiscreteNonParametric(θ, ps))
end

struct Wasserstein1Projection <: DistributionProjection
    τ̂::AbstractArray{<: Real}
end

Wasserstein1Projection(sketch::QuantileSketch) = Wasserstein1Projection(sketch.τ̂)
Wasserstein1Projection(n::Integer) = Wasserstein1Projection((collect(1:n) .- 0.5) ./ n)

function (Π::Wasserstein1Projection)(μ::DiscreteNonParametric)
    N = length(Π.τ̂)
    DiscreteNonParametric(Statistics.quantile.(μ, Π.τ̂), ones(N) / N)
end
