# Gaussian score diffusions

`phydrax.stochastic` provides prescribed variance-preserving and variance-exploding
Itô processes for continuous-time score learning. These objects are forward stochastic
processes, not neural models and not samplers from a learned law.

The initial public contract is deliberately narrow: real vector states, trivial
Euclidean coordinates, and scalar state-independent full-rank noise. The transition
marginal is represented by `DiagonalGaussianProcessDistribution`, so sampling and log
density remain linear in the state dimension and never materialize an identity
covariance matrix.

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

## Deliberate exclusions

The current Gaussian score process does not claim support for:

- matrix-valued or state-dependent diffusion;
- singular or retained-subspace noise;
- Riemannian or embedded-manifold scores;
- complex, discrete, categorical, or path-valued events;
- mesh-independent random fields.

A low-rank Gaussian has no ambient Lebesgue density. Physical-field diffusion also
requires an explicit mass metric and spatial covariance contract; IID nodal noise is
not silently labeled physical white noise.
