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