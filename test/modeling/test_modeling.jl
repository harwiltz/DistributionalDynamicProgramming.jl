@testset "representations" begin
    categorical_representation = CategoricalRepresentation(5, 0.0, 1.0)
    @test categorical_representation.sketch isa CategoricalSketch
    @test categorical_representation.Φ isa CategoricalImputation
    @test categorical_representation.Π isa CramérProjection
    @test n_params(categorical_representation) == 5

    quantile_representation = QuantileRepresentation(10)
    @test quantile_representation.sketch isa QuantileSketch
    @test quantile_representation.Φ isa QuantileImputation
    @test quantile_representation.Π isa Wasserstein1Projection
    @test n_params(quantile_representation) == 10
end
@testset "sketches" begin
    include("test_sketches.jl")
end
@testset "imputations" begin
    include("test_imputations.jl")
end
@testset "projections" begin
    include("test_projections.jl")
end
