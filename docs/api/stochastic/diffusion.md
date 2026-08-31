# Gaussian and structured score diffusions

`phydrax.stochastic` provides prescribed variance-preserving and variance-exploding
Itô processes for continuous-time score learning. These objects are forward stochastic
processes, not neural models and not samplers from a learned law.

The core VP/VE contract uses real vector states, trivial Euclidean coordinates, and
scalar state-independent full-rank noise. Its transition marginal is represented by
`DiagonalGaussianProcessDistribution`, so sampling and log density remain linear in
the state dimension and never materialize an identity covariance matrix. Structured
event, covariance, geometry, and discrete extensions remain separate typed contracts
rather than weakening that core.

## Probability laws and process marginals

`DiagonalNormalLaw` is the normalized Lebesgue law used by terminal references and
continuous transport. It preserves explicit sample, batch, and event axes.
`DiagonalGaussianProcessDistribution` adds process-uncertainty semantics to the same
structured Gaussian calculations.

::: phydrax.uq.DiagonalNormalLaw

---

::: phydrax.stochastic.DiagonalGaussianProcessDistribution

## Forward process contract

An `AbstractGaussianDiffusion` supplies the forward drift, scalar diffusion rate,
exact Gaussian transition mean and scale, conditional transition score, and one
explicit asymptotic terminal reference.

For a forward diffusion

```text
dX_t = f(X_t, t) dt + g(t) dW_t,
```

the transition methods always describe a strict interval `t1 > t0`. A zero-length
transition is singular and is rejected rather than represented as a fictitious
Lebesgue density.

::: phydrax.stochastic.AbstractGaussianDiffusion

---

::: phydrax.stochastic.VariancePreservingDiffusion

---

::: phydrax.stochastic.VarianceExplodingDiffusion

## Terminal references

A finite-time VP or VE marginal is not generally equal to its Gaussian asymptotic
reference. `DiffusionTerminalReference` therefore records whether a supplied source
law is `"exact"`, `"asymptotic"`, or `"external"`, together with its process identity
and known residual signal scale. The relationship is provenance, not an automatic
certificate about an unknown data law.

::: phydrax.stochastic.DiffusionTerminalReference

## Structured and discrete extensions

General Euclidean Itô diffusion exposes factor, covariance action, covariance
divergence, reverse drift, and probability-flow drift. Constant matrix coefficients
retain exact affine Gaussian transitions; state-dependent coefficients use exact
automatic differentiation of the covariance divergence.

::: phydrax.stochastic.AbstractItoScoreDiffusion

---

::: phydrax.stochastic.MatrixGaussianDiffusion

---

::: phydrax.stochastic.StateDependentItoDiffusion

Finite schedules and geometric laws have their own reference measures and terminal
semantics.

::: phydrax.stochastic.DiscreteGaussianDiffusionSchedule

---

::: phydrax.stochastic.AncestralGaussianDiffusion

---

::: phydrax.stochastic.DDIMTransport

---

::: phydrax.stochastic.CategoricalDiffusionSchedule

---

::: phydrax.stochastic.CategoricalReverseDiffusion

---

::: phydrax.stochastic.SubspaceGaussianLaw

---

::: phydrax.stochastic.FieldGaussianDiffusion

---

::: phydrax.stochastic.IsotropicRiemannianDiffusion

---

::: phydrax.stochastic.ComplexVariancePreservingDiffusion

---

::: phydrax.stochastic.PathCoefficientDiffusion

See [Advanced generative transport](../transport/generative_expansion.md) for measure,
conditioning, transfer, and composition semantics.
