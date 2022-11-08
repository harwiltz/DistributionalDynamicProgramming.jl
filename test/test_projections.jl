using LinearAlgebra: I
using Statistics

function test_cramér_projection_basic(l)
    Π = CramérProjection(l)
    locs = cumsum(rand(5))
    p = rand(Dirichlet(5, 1))
    Π(DiscreteNonParametric(locs, p))
end

function test_cramér_projection_identity(l, p)
    Π = CramérProjection(l)
    Π(DiscreteNonParametric(l, p))
end

function test_cramér_projection_mixture(l)
    Π = CramérProjection(l)
    μ = MixtureModel(
        [DiscreteNonParametric(cumsum(rand(5)), rand(Dirichlet(5, 1))) for _ in 1:3],
        rand(Dirichlet(3, 1))
    )
    Π(μ)
end

function test_cramér_projection_mixture_deltas(l, p)
    Π = CramérProjection(l)
    N = length(l)
    one_hot_encoder = Matrix{Float64}(I, N, N)
    μ = MixtureModel(
        [DiscreteNonParametric(l, one_hot_encoder[:, i]) for i in 1:N],
        p
    )
    Π(μ)
end

function test_cramér_projection_mixture_singleton(l, p)
    Π = CramérProjection(l)
    random_comps = [DiscreteNonParametric(cumsum(rand(5)), rand(Dirichlet(5, 1))) for _ in 1:10]
    comps = [DiscreteNonParametric(l, p); random_comps]
    ps = [1.0; zeros(Float64, 10)]
    μ = MixtureModel(comps, ps)
    Π(μ)
end

function test_cramér_projection_mixture_redundant(l, p)
    Π = CramérProjection(l)
    comps = [DiscreteNonParametric(l, p) for _ in 1:10]
    ps = [0.1 for _ in 1:10]
    μ = MixtureModel(comps, ps)
    Π(μ)
end

function test_wasserstein1_projection_basic(n, μ)
    Π = Wasserstein1Projection(n)
    Π(μ)
end

function test_wasserstein1_projection_identity(l)
    N = length(l)
    Π = Wasserstein1Projection(N)
    p = ones(N) / N
    Π(DiscreteNonParametric(l, p))
end

function test_wasserstein1_projection_mixture(n)
    Π = Wasserstein1Projection(n)
    μ = MixtureModel(
        [DiscreteNonParametric(cumsum(rand(5)), rand(Dirichlet(5, 1))) for _ in 1:10],
        rand(Dirichlet(10, 1))
    )
    Π(μ)
end

function test_wasserstein1_projection_mixture_deltas(l)
    N = length(l)
    Π = Wasserstein1Projection(N)
    one_hot_encoder = Matrix{Float64}(I, N, N)
    μ = MixtureModel(
        [DiscreteNonParametric(l, one_hot_encoder[:, i]) for i in 1:N],
        ones(N) / N
    )
    Π(μ)
end

function test_wasserstein1_projection_mixture_singleton(l)
    N = length(l)
    Π = Wasserstein1Projection(N)
    random_comps = [DiscreteNonParametric(cumsum(rand(5)), rand(Dirichlet(5, 1))) for _ in 1:10]
    comps = [DiscreteNonParametric(l, ones(N) / N); random_comps]
    ps = [1.0; zeros(Float64, 10)]
    μ = MixtureModel(comps, ps)
    Π(μ)
end

function test_wasserstein1_projection_mixture_redundant(l)
    N = length(l)
    Π = Wasserstein1Projection(N)
    comps = [DiscreteNonParametric(l, ones(N) / N) for _ in 1:10]
    ps = [0.1 for _ in 1:10]
    μ = MixtureModel(comps, ps)
    Π(μ)
end

cramér_projection_locs = LinRange(0.5, 1.5, 5)
cramér_projection_probs = rand(Dirichlet(5, 1))

cramér_projection_basic = test_cramér_projection_basic(cramér_projection_locs)
@test cramér_projection_basic isa DiscreteNonParametric
@test support(cramér_projection_basic) == cramér_projection_locs

cramér_projection_identity = test_cramér_projection_identity(LinRange(0,1,5), cramér_projection_probs)
@test support(cramér_projection_identity) |> collect  == LinRange(0,1,5)
@test isapprox(probs(cramér_projection_identity), cramér_projection_probs)

cramér_projection_mixture = test_cramér_projection_mixture(cramér_projection_locs)
@test cramér_projection_mixture isa DiscreteNonParametric
@test support(cramér_projection_mixture) == cramér_projection_locs

cramér_projection_mixture_deltas =
    test_cramér_projection_mixture_deltas(cramér_projection_locs, cramér_projection_probs)
@test support(cramér_projection_mixture_deltas) == cramér_projection_locs
@test isapprox(probs(cramér_projection_mixture_deltas), cramér_projection_probs)

cramér_projection_mixture_singleton =
    test_cramér_projection_mixture_singleton(cramér_projection_locs, cramér_projection_probs)
@test support(cramér_projection_mixture_singleton) == cramér_projection_locs
@test isapprox(probs(cramér_projection_mixture_singleton), cramér_projection_probs)

cramér_projection_mixture_redundant =
    test_cramér_projection_mixture_redundant(cramér_projection_locs, cramér_projection_probs)
@test support(cramér_projection_mixture_redundant) == cramér_projection_locs
@test isapprox(probs(cramér_projection_mixture_redundant), cramér_projection_probs)

wasserstein1_locs = cumsum(rand(5))
wasserstein1_μ = DiscreteNonParametric(rand(10), rand(Dirichlet(10, 1)))
wasserstein1_τ̂ = (collect(1:5) .- 0.5) / 5

wasserstein1_projection_basic = test_wasserstein1_projection_basic(5, wasserstein1_μ)
@test support(wasserstein1_projection_basic) == Statistics.quantile.(wasserstein1_μ, wasserstein1_τ̂)
@test isapprox(probs(wasserstein1_projection_basic), ones(5) / 5)

wasserstein1_projection_identity = test_wasserstein1_projection_identity(wasserstein1_locs)
@test support(wasserstein1_projection_identity) == wasserstein1_locs
@test isapprox(probs(wasserstein1_projection_identity), ones(5) / 5)

wasserstein1_projection_mixture = test_wasserstein1_projection_mixture(5)
@test length(support(wasserstein1_projection_mixture)) == 5
@test isapprox(probs(wasserstein1_projection_mixture), ones(5) / 5)

wasserstein1_projection_mixture_deltas = test_wasserstein1_projection_mixture_deltas(wasserstein1_locs)
@test support(wasserstein1_projection_mixture_deltas) == wasserstein1_locs
@test isapprox(probs(wasserstein1_projection_mixture_deltas), ones(5) / 5)

wasserstein1_projection_mixture_singleton = test_wasserstein1_projection_mixture_singleton(wasserstein1_locs)
@test support(wasserstein1_projection_mixture_singleton) == wasserstein1_locs
@test isapprox(probs(wasserstein1_projection_mixture_singleton), ones(5) / 5)

wasserstein1_projection_mixture_redundant = test_wasserstein1_projection_mixture_redundant(wasserstein1_locs)
@test support(wasserstein1_projection_mixture_redundant) == wasserstein1_locs
@test isapprox(probs(wasserstein1_projection_mixture_redundant), ones(5) / 5)
