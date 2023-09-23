struct CategoricalSketchParams{L <: AbstractArray{<: Real}, P <: AbstractArray{<: Real}}
    locs::L
    probs::P
end

struct QuantileSketchParams{L <: AbstractArray{<: Real}}
    locs::L
end

function test_categorical_sketch_categorical(p)
    μ = Categorical(p)
    sketch = CategoricalSketch(5, 0, 1)
    xs, ps = sketch(μ)
    CategoricalSketchParams(xs, ps)
end

function test_categorical_sketch_nonparametric(p)
    μ = DiscreteNonParametric(Base.OneTo(5), p)
    sketch = CategoricalSketch(5, 0, 1)
    xs, ps = sketch(μ)
    CategoricalSketchParams(xs, ps)
end

function test_categorical_sketch_mixture(p)
    x = LinRange(0, 1, 6)
    μ = categorical_mixture(x, p)
    sketch = CategoricalSketch(5, 0, 1)
    xs, ps = sketch(μ)
    CategoricalSketchParams(xs, ps)
end

function test_quantile_sketch(x)
    μ = DiscreteNonParametric(x, ones(length(x)) ./ length(x))
    sketch = QuantileSketch(5)
    xs = sketch(μ)
    QuantileSketchParams(xs)
end

categorical_locs = LinRange(0, 1, 5)
categorical_probs = rand(Dirichlet(5, 1))
categorical_mixture(x, p) = MixtureModel(zip(x[1:end - 1], x[2:end]) .|> Base.splat(Uniform), p)

quantile_locs = cumsum(rand(5))

@test test_categorical_sketch_categorical(categorical_probs).locs == categorical_locs
@test test_categorical_sketch_categorical(categorical_probs).probs == categorical_probs
@test test_categorical_sketch_nonparametric(categorical_probs).locs == categorical_locs
@test test_categorical_sketch_nonparametric(categorical_probs).probs == categorical_probs
@test test_categorical_sketch_mixture(categorical_probs).locs == categorical_locs
@test test_categorical_sketch_mixture(categorical_probs).probs == categorical_probs

@test test_quantile_sketch(quantile_locs).locs == quantile_locs
