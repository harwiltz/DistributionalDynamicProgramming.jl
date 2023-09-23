using POMDPs

export TabularReturnDistributionFunction,
    distributional_policy_evaluation

const TabularReturnDistributionFunction = Dict{Any, AbstractArray{<:Real}}

abstract type DeterministicPolicy <: Function end
abstract type StochasticPolicy end
const AbstractPolicy = Union{DeterministicPolicy, StochasticPolicy}

initialize_return_distributions(repr::DistributionalRepresentation, mdp::MDP, x) =
    initialize_return_distributions(repr.sketch, mdp::MDP, x)

function initialize_return_distributions(sketch::CategoricalSketch, mdp::MDP, x::Real)
    Π = CramérProjection(sketch)
    μ = DiscreteNonParametric([x], [1.0])
    Dict(s => sketch(Π(μ)) for s in mdp.S)
end

initialize_return_distributions(sketch::QuantileSketch, mdp::MDP, x::Real) =
    Dict(s => x * ones(Float64, sketch.n_atoms) for s in mdp.S)

function distributional_policy_evaluation(
    mdp::MDP,
    repr::DistributionalRepresentation,
    π::DeterministicPolicy,
    η::TabularReturnDistributionFunction
)
end
