# Differentiable hydrodynamics inference

The complete runnable workflow is
`examples/differentiable_hydrodynamics_inference.py`.
It demonstrates a whitened latent initial field, a fixed finite-volume temporal mesh,
block rematerialization, a physical observation map, and a normalized posterior.

## Model construction

The example prepares periodic one-dimensional Euler dynamics with one conservative face
flux and stage positivity. An adaptive controller is not differentiated. Instead, an
all-active `TemporalMesh` declares the exact physical observation time and every
internal interval.

```text
rollout = phx.solver.ScheduledFiniteVolumeRolloutPlan(
    runtime,
    temporal_mesh,
    replay=phx.solver.FiniteVolumeReplayPolicy("block", block_size=2),
)
```

A proposal that cannot complete the schedule is invalid; it must not be evaluated at a
silently shortened final time.

## Whitened initial field

`SpatialNoiseBasis.from_spectrum` constructs a finite-rank covariance factor. Posterior
coordinates are standard-normal coefficients rather than constrained Fourier phases.
The initial density is a positive transform of the resulting spatial field.

```text
latent ~ Normal(0, I)
fluctuation = diffusion_matrix @ latent
initial_density = exp(scale * fluctuation)
```

The latent prior, physical transformation, rollout, and observation map remain inside
one `PosteriorProblem` log density.

## Gradient and inference semantics

`PosteriorProblem.validate()` evaluates the initial log density and gradient. The same
problem can be passed to Phydrax MAP, Laplace, HMC, or NUTS routines after the schedule
has been qualified over the intended parameter region.

Shock movement, limiters, positivity activation, and schedule boundaries are
branchwise. A stochastic source realization is conditioned when held fixed; that is not
marginalization. Marginal stochastic physics requires explicit latent variables or an
explicit expectation estimator.
