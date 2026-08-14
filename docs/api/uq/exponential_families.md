# Exponential families

Phydrax represents a regular exponential family by the normalized density

`log p(x; η) = ⟨η, T(x)⟩ - A(η) + log h(x)`

relative to an explicit reference measure. `NaturalCoordinates`, `MeanCoordinates`,
and `StatisticBatch` are different semantic objects even when their arrays have the
same shape:

- `NaturalCoordinates.values` contains `η`.
- `MeanCoordinates.values` contains `μ = Eη[T(X)] = ∇A(η)`.
- `StatisticBatch.values` contains realized sufficient statistics `T(x)` and a
  support-validity mask.

The final axis is always the intrinsic statistic dimension. All preceding axes are
batch axes. A static `ExponentialFamilySignature` records the family, intrinsic
dimension, event shape, density measure, support, and coordinate chart. Operations
such as KL divergence reject mismatched signatures rather than combining arrays that
happen to have equal shapes.

## Supported families

| Family | Reference measure | Natural coordinate | Sufficient statistic |
| --- | --- | --- | --- |
| `BernoulliFamily` | counting | log odds | `x` |
| `CategoricalFamily(K)` | counting | `K - 1` log odds relative to category `K - 1` | first `K - 1` one-hot entries |
| `PoissonFamily` | counting | log rate | `x` |
| `ExponentialRateFamily` | Lebesgue | negative rate | `x` |
| `GammaFamily` | Lebesgue | `(shape - 1, -rate)` | `(log(x), x)` |
| `NormalFamily` | Lebesgue | `(location / variance, -1 / (2 variance))` | `(x, x²)` |
| `MultivariateNormalFamily(d)` | Lebesgue | linear coordinate and orthonormal symmetric quadratic chart | `(x, svec(xxᵀ))` |
| `DirichletFamily(K)` | simplex Hausdorff measure | `concentration - 1` | `log(x)` |

A mean-domain boundary can be a valid empirical sufficient-statistic average while
having no finite natural coordinate in the regular family. All-zero Bernoulli data,
zero-variance Gaussian data, and a simplex point mass are examples. Phydrax returns
`EXPONENTIAL_FAMILY_MEAN_BOUNDARY` and `valid=False`; it does not add pseudocounts,
clip a probability, or jitter a covariance.

## Normalized laws and likelihoods

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

bernoulli = phx.uq.BernoulliFamily()
log_odds = jnp.asarray([0.4])
law = bernoulli.law(log_odds)

log_probability = law.log_prob(jnp.asarray([0.0, 1.0]))
samples = law.sample(jr.key(1), sample_shape=(128,))
mean = law.mean_coordinates
```

`ExponentialFamilyLaw` implements `AbstractProbabilityLaw`. It can therefore be a
factorized `ParameterSpace` prior. A scalar law, with empty batch and event shapes, is
applied independently to every element of an array-valued parameter leaf. A shaped
law must have `batch_shape + event_shape` exactly equal to that leaf's shape.

`ScalarNaturalExponentialFamilyLikelihood` adapts a scalar-event family with one
natural coordinate to the existing elementwise observation-likelihood protocol. The
model output is interpreted as the natural coordinate, not as a conventional mean or
probability:

```python
poisson_likelihood = phx.uq.ScalarNaturalExponentialFamilyLikelihood(
    phx.uq.PoissonFamily()
)
log_rate = jnp.asarray([0.0, jnp.log(2.0)])
counts = jnp.asarray([0.0, 3.0])
log_probability = poisson_likelihood.log_prob(log_rate, counts)
```

Use an explicit link before this adapter if a model emits conventional parameters.
Do not pass categorical logits or the two natural coordinates of `NormalFamily`
through the scalar adapter.

`CategoricalExponentialFamilyLikelihood` takes `CategoricalFamily(K)` plus the
required `prediction_coordinates` convention. Model output may use either `K` full
logits (`"full_logits"`) or the identified `K - 1` natural coordinates
(`"natural"`). Full logits are reduced by subtracting the last-category logit.
Integer labels remain scalar events.

## Weighted projection and maximum likelihood

`project_exponential_family` computes sufficient statistics, combines them with
normalized log weights through the mergeable `LogWeightedAccumulator`, classifies the
resulting mean coordinates, and converts only interior means to natural coordinates.
`fit_exponential_family` is the statistical-weight counterpart.

```python
observations = jnp.asarray([0.0, 1.0, 1.0, 0.0])
log_weights = jnp.asarray([-0.4, 0.2, 0.8, -0.1])

estimate = phx.uq.project_exponential_family(
    phx.uq.BernoulliFamily(),
    observations,
    log_weights=log_weights,
)

assert estimate.valid
probability = estimate.mean_coordinates.values[0]
```

For streaming or distributed data, build
`ExponentialFamilyProjectionAccumulator.from_log_weights(...)` for each aligned
chunk, merge the accumulators, and call `finalize()`. Merging retains the stable
log-weight scale, effective sample size, maximum normalized weight, entropy, and
log-weight range. One-shot and merged projections have the same estimator semantics.

Numerical outcomes are explicit:

- `-inf` log weight means legitimate zero weight.
- `nan` or positive-infinite active weight returns `EXPONENTIAL_FAMILY_NONFINITE`.
- An active observation outside support returns `EXPONENTIAL_FAMILY_INVALID_EVENT`.
- No positive finite weight returns `EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT`.
- An exterior statistic mean returns `EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN`.
- A realizable boundary mean returns `EXPONENTIAL_FAMILY_MEAN_BOUNDARY`.

A mask excludes an observation before these active-observation checks. No invalid
active datum or weight is silently discarded.

## Structured and simplex-valued laws

`GammaFamily` and `DirichletFamily` invert expected sufficient statistics with
family-specific safeguarded solvers. `ExponentialFamilyConversionResult` reports the
iteration count, residual, method identifier, and a distinct
`EXPONENTIAL_FAMILY_NONCONVERGED` status. The solver is never a hidden fallback for
an arbitrary family.

```python
gamma = phx.uq.GammaFamily()
gamma_natural = gamma.natural_from_shape_rate(2.5, 1.7)
gamma_mean = gamma.mean_from_natural(gamma_natural)
gamma_round_trip = gamma.natural_from_mean(gamma_mean)

gaussian = phx.uq.MultivariateNormalFamily(3)
gaussian_natural = gaussian.natural_from_location_covariance(
    jnp.zeros(3),
    jnp.asarray([[1.0, 0.2, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 1.2]]),
)
location, covariance = gaussian.location_covariance_from_natural(gaussian_natural)
```

The multivariate Normal uses the orthonormal `svec` chart: off-diagonal symmetric
entries are scaled by the square root of two. Euclidean dot products in this chart
therefore equal Frobenius products of symmetric matrices. No dense Fisher matrix is
stored. `gaussian_factor_from_multivariate_normal` and
`multivariate_normal_from_gaussian_factor` bridge the family law to the existing
square-root `GaussianFactor` covariance representation without introducing a second
Gaussian algebra.

`DirichletFamily(K)` is normalized relative to the intrinsic Hausdorff measure on
the `K - 1` dimensional simplex, not ambient `K` dimensional Lebesgue measure.
`SimplexBijector(K)` maps `K - 1` additive-log-ratio coordinates to `K` positive
components and returns the corresponding Hausdorff volume factor.

```python
simplex_family = phx.uq.DirichletFamily(4)
simplex_prior = simplex_family.law_from_concentration(jnp.asarray([1.0, 2.0, 1.5, 3.0]))
simplex_space = phx.uq.ParameterSpace(
    jnp.zeros(3),
    priors=simplex_prior,
    bijectors=phx.uq.SimplexBijector(4),
)

assert simplex_space.raw_shapes == ((3,),)
assert simplex_space.physical_shapes == ((4,),)
```

`ParameterSpace` validates raw and physical leaf shapes separately. A
shape-changing bijector is therefore explicit rather than being smuggled through a
shape-preserving transform. Gaussian-prior whitening rejects such a leaf because its
raw-space prior is not Gaussian.

## Explicit conjugate pairs

Conjugacy is represented by likelihood-prior pairs, not by a universal operation on
natural-coordinate arrays:

```python
gamma_poisson = phx.uq.GammaPoissonConjugacy(shape=2.0, rate=1.5)
rate_update = gamma_poisson.update(
    jnp.asarray([0, 2, 3]),
    exposure=jnp.asarray([1.0, 0.5, 2.0]),
)

dirichlet_categorical = phx.uq.DirichletCategoricalConjugacy(jnp.asarray([1.0, 1.0, 1.0]))
composition_update = dirichlet_categorical.update(jnp.asarray([0, 2, 1, 2, 2]))
```

Each update exposes the posterior law, exact ordered-observation log evidence,
posterior predictive probabilities or log probabilities, and mergeable sufficient
statistics. Gamma-Poisson exposure is part of both the posterior rate and the
Poisson base-measure term. Dirichlet-categorical evidence is for an ordered
categorical sequence; it deliberately does not insert the multinomial coefficient
of an unordered count observation.

## Exact matrix-free information geometry

For a regular family, the Fisher information in natural coordinates is the Jacobian
of the mean map: `F(η) v = Jμ(η) v`. `exponential_family_fisher_action` evaluates this
JVP directly. It does not construct, invert, or take a determinant of a dense Fisher
matrix.

```python
family = phx.uq.NormalFamily()
natural = family.natural(jnp.asarray([0.2, -0.7]))
direction = jnp.asarray([0.4, -0.3])

information_direction = phx.uq.exponential_family_fisher_action(
    family,
    natural,
    direction,
)
```

If model parameters `θ` produce natural coordinates `η(θ)`,
`exponential_family_parameter_fisher_action` applies
`Jη(θ)ᵀ F(η(θ)) Jη(θ) v` using one natural-map JVP, the exact family action, and one
transpose pullback. Optional regularization adds exactly the declared parameter-space
`regularization * v`. Both results retain operator, method, approximation, validity,
and status provenance.

## Domains and statuses

Natural and mean domains are different. `natural_domain(...)` classifies whether a
normalized finite law exists. `mean_domain(...)` additionally separates interior,
boundary, and exterior expected-statistic coordinates. `natural_from_mean(...)`
returns an `ExponentialFamilyConversionResult`; it never hides a boundary, invalid
coordinate, or nonfinite analytical result.

Analytical families convert directly. Gamma and Dirichlet use only their explicit,
family-specific numerical inversion hooks. A failed numerical solve returns
`EXPONENTIAL_FAMILY_NONCONVERGED`; it is not conflated with a boundary, an exterior
mean, or nonfinite input.

## Core contracts

::: phydrax.uq.AbstractProbabilityLaw

---

::: phydrax.uq.AbstractExponentialFamily

---

::: phydrax.uq.ExponentialFamilySignature

---

::: phydrax.uq.NaturalCoordinates

---

::: phydrax.uq.MeanCoordinates

---

::: phydrax.uq.StatisticBatch

---

::: phydrax.uq.ExponentialFamilyDomainResult

---

::: phydrax.uq.ExponentialFamilyConversionResult

---

::: phydrax.uq.ExponentialFamilyLaw

## Families

::: phydrax.uq.BernoulliFamily

---

::: phydrax.uq.PoissonFamily

---

::: phydrax.uq.ExponentialRateFamily

---

::: phydrax.uq.NormalFamily

---

::: phydrax.uq.CategoricalFamily

---

::: phydrax.uq.GammaFamily

---

::: phydrax.uq.MultivariateNormalFamily

---

::: phydrax.uq.DirichletFamily

## Conjugacy

::: phydrax.uq.GammaPoissonConjugacy

---

::: phydrax.uq.GammaPoissonStatistics

---

::: phydrax.uq.GammaPoissonUpdate

---

::: phydrax.uq.DirichletCategoricalConjugacy

---

::: phydrax.uq.DirichletCategoricalStatistics

---

::: phydrax.uq.DirichletCategoricalUpdate

## Coordinate and representation bridges

::: phydrax.uq.SimplexBijector

---

::: phydrax.uq.gaussian_factor_from_multivariate_normal

---

::: phydrax.uq.multivariate_normal_from_gaussian_factor

## Projection and fitting

::: phydrax.uq.ExponentialFamilyProjectionAccumulator

---

::: phydrax.uq.ExponentialFamilyEstimateResult

---

::: phydrax.uq.project_exponential_family

---

::: phydrax.uq.fit_exponential_family

## Likelihood and Fisher adapters

::: phydrax.uq.ScalarNaturalExponentialFamilyLikelihood

---

::: phydrax.uq.CategoricalExponentialFamilyLikelihood

---

::: phydrax.uq.exponential_family_fisher_action

---

::: phydrax.uq.exponential_family_parameter_fisher_action

## Status codes

::: phydrax.uq.exponential_family_status_name

---

::: phydrax.uq.EXPONENTIAL_FAMILY_SUCCESS

---

::: phydrax.uq.EXPONENTIAL_FAMILY_NONFINITE

---

::: phydrax.uq.EXPONENTIAL_FAMILY_NONCONVERGED

---

::: phydrax.uq.EXPONENTIAL_FAMILY_INVALID_EVENT

---

::: phydrax.uq.EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN

---

::: phydrax.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN

---

::: phydrax.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY

---

::: phydrax.uq.EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT
