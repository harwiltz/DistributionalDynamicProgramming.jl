using DistributionalDynamicProgramming
using Distributions
using Test

@testset "all" begin
    @testset "modeling" begin
        include("modeling/test_modeling.jl")
    end
end
