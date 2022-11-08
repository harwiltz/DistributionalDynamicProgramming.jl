using LinearAlgebra: I

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

cramér_projection_locs = LinRange(0.5, 1.5, 5)
cramér_projection_probs = rand(Dirichlet(5, 1))

cramér_projection_basic = test_cramér_projection_basic(cramér_projection_locs)
@test cramér_projection_basic isa DiscreteNonParametric
@test support(cramér_projection_basic) == cramér_projection_locs

cramér_projection_identity = test_cramér_projection_identity(LinRange(0,1,5), cramér_projection_probs)
@test support(cramér_projection_identity) |> collect  == LinRange(0,1,5)
@test probs(cramér_projection_identity) == cramér_projection_probs

cramér_projection_mixture = test_cramér_projection_mixture(cramér_projection_locs)
@test cramér_projection_mixture isa DiscreteNonParametric
@test support(cramér_projection_mixture) == cramér_projection_locs

cramér_projection_mixture_deltas =
    test_cramér_projection_mixture_deltas(cramér_projection_locs, cramér_projection_probs)
@test support(cramér_projection_mixture_deltas) == cramér_projection_locs
@test probs(cramér_projection_mixture_deltas) == cramér_projection_probs)
