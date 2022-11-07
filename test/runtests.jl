using DistributionalDynamicProgramming
using Distributions
using Test

@testset "all" begin
    @testset "sketches" begin
        include("test_sketches.jl")
    end
    @testset "imputations" begin
        include("test_imputations.jl")
    end
    @testset "projections" begin
        include("test_projections.jl")
    end
end
