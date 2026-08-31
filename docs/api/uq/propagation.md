# Probability distributions and propagation

## Distributions

::: phydrax.uq.Uniform
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob
            - to_reference
            - from_reference

---

::: phydrax.uq.Normal
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

            - to_reference
            - from_reference
---

::: phydrax.uq.LogNormal
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

            - to_reference
            - from_reference
---

::: phydrax.uq.EmpiricalDistribution
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

## Domains and joint designs

::: phydrax.domain.ProbabilityDomain
    options:
        members:
            - __init__
            - sample
            - fixed
            - supports_reference_transform
            - reference_measure
            - to_reference
            - from_reference

---
::: phydrax.domain.ReferenceDistribution

---


::: phydrax.uq.RandomSampleBatch
    options:
        members:
            - __init__

---

::: phydrax.uq.sample_joint

---

::: phydrax.uq.propagate


## Nonintrusive polynomial chaos

`PolynomialChaosBasis` binds an ordered tuple of independent scalar
`ProbabilityDomain` factors to a graded total-degree orthonormal basis. `Uniform`
factors use normalized Legendre polynomials after their exact affine reference
transform; `Normal` factors use a stable normalized probabilists' Hermite recurrence
after standardization, without materializing factorials. Degree zero is a valid
constant basis. Multiindex count and storage guards fail before allocating an
oversized feature table.

Projection and regression are separate nonintrusive contracts:

- `PolynomialChaosProjectionPlan` preflights product point/replicate counts and
  sample-by-feature basis bytes before materializing the supplied
  `ProductIntegrationPlan`, then projects model values against every basis mode.
- `PolynomialChaosRegressionPlan` uses a native exact linear system for an
  unweighted square design and a diagnosed native least-squares problem for every
  weighted or overdetermined design. The default policies require full rank. A
  deficient design or nonfinite samples fail rather than selecting an undeclared
  pseudoinverse.

Both plans support array, `coordax.Field`, and PyTree outputs. The resulting immutable
`PolynomialChaosExpansion` remains callable under JIT and differentiation; every
coefficient leaf retains its physical output shape instead of flattening that shape
into the polynomial-mode axis. These APIs do **not** implement intrusive stochastic
Galerkin equations.

Projection applies evaluation, accumulation, replicate reduction, and output
precision in that order and rejects nonfinite values at every boundary. Plan
identities include complete quadrature/design and native solver-policy content.

Sparse product projection honors each declared `axis_rules` entry:
`"gauss-hermite"` consumes a standard-normal reference factor with unit-mass
weights, while `"clenshaw-curtis"` uses the bounded uniform canonical map.
Mismatched measures fail before sparse-rule materialization.

::: phydrax.uq.PolynomialMultiIndexSet

---

::: phydrax.uq.PolynomialChaosBasis

---

::: phydrax.uq.PolynomialChaosProjectionPlan

---

::: phydrax.uq.PolynomialChaosRegressionPlan

---

::: phydrax.uq.PolynomialChaosExpansion

---

::: phydrax.uq.PolynomialChaosFitResult


## Matrix-free linearized covariance

`propagate_linearized` evaluates the nominal output once, retains a JVP
pushforward and Hermitian VJP pullback, and represents the first-order output
covariance as the action \(J C_x J^\mathrm{H}\). It never materializes a Jacobian.
Choose the input covariance representation deliberately:

- `DiagonalCovariance` for independent coordinate variances;
- `DenseCovariance` for a small explicit Hermitian positive-semidefinite matrix;
- `FactorCovariance` for \(C_x=B B^\mathrm{H}\), with factor rank on the leading
  axis of every PyTree leaf;
- `CovarianceOperator` for a caller-supplied matrix-free action.

`exact_variance()` is exact under the declared first-order model for diagonal,
dense, and factor inputs. `estimate_variance(...)` is a keyed Hutchinson
estimate, reports its Monte Carlo standard error, and is the only diagonal path
for a generic covariance operator. `materialize_covariance(...)` is guarded by
an explicit output-dimension ceiling. It is a diagnostic for small outputs, not
the default representation.

Real-valued PyTrees and `coordax.Field` outputs retain their structure and named
dimensions. Complex propagation requires `complex_linear=True` and uses the
Hermitian adjoint. Non-holomorphic models must instead expose real and imaginary
coordinates explicitly.

::: phydrax.uq.DiagonalCovariance
    options:
        members:
            - __init__

---

::: phydrax.uq.DenseCovariance
    options:
        members:
            - __init__

---

::: phydrax.uq.FactorCovariance
    options:
        members:
            - __init__

---

::: phydrax.uq.CovarianceOperator
    options:
        members:
            - __init__

---

::: phydrax.uq.covariance_representation

---

::: phydrax.uq.propagate_linearized

---

::: phydrax.uq.propagate_linearized_map

---

::: phydrax.uq.LinearizedPropagationResult
    options:
        members:
            - pushforward
            - pullback
            - covariance_vector_product
            - exact_variance
            - estimate_variance
            - materialize_covariance

---

::: phydrax.uq.LinearizedVarianceEstimate

---

::: phydrax.uq.LinearizedDenseCovariance
    options:
        members:
            - covariance_vector_product