function test_categorical_imputation_from_atoms(xs, ps)
    Φ = CategoricalImputation(xs)
    Φ(ps)
end

function test_categorical_imputation_from_sketch(xs, ps)
    sketch = CategoricalSketch(length(xs), minimum(xs), maximum(xs), xs)
    Φ = CategoricalImputation(sketch)
    Φ(ps)
end

function test_quantile_imputation(x)
    QuantileImputation()(x)
end

categorical_locs = cumsum(rand(5))
categorical_ps = rand(Dirichlet(5, 1))

categorical_imputation_from_atoms = test_categorical_imputation_from_atoms(categorical_locs, categorical_ps)
@test categorical_imputation_from_atoms isa DiscreteNonParametric
@test support(categorical_imputation_from_atoms) |> collect == categorical_locs
@test probs(categorical_imputation_from_atoms) == categorical_ps
categorical_imputation_from_sketch = test_categorical_imputation_from_sketch(categorical_locs, categorical_ps)
@test categorical_imputation_from_sketch isa DiscreteNonParametric
@test support(categorical_imputation_from_sketch) |> collect == categorical_locs
@test probs(categorical_imputation_from_sketch) == categorical_ps

quantile_locs = cumsum(rand(5))
@test test_quantile_imputation(quantile_locs) isa DiscreteNonParametric
@test support(test_quantile_imputation(quantile_locs)) |> collect == quantile_locs
@test all(probs(test_quantile_imputation(quantile_locs)) .== 1 / 5) == true
