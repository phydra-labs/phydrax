# Differentiable hydrodynamics inference

The complete runnable workflow is
[`examples/differentiable_hydrodynamics_inference.py`](https://github.com/phydra-labs/phydrax/blob/dev/examples/differentiable_hydrodynamics_inference.py).
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

## Compose constrained MHD with source physics

Constrained MHD uses the same adaptive realization and fixed replay as ordinary
finite-volume transport. Prepare one explicit transport adapter, then prepare every
source process against that adapter:

```text
transport = phx.solver.prepare_balance_law_transport(mhd_integrator)
gravity = phx.solver.NewtonianSelfGravityPlan(0.1).prepare(transport)
cooling = phx.solver.RadiativeCoolingProcessPlan(curve).prepare(transport)
forcing = phx.solver.SpectralOUForcingPlan().prepare(transport)

runtime = phx.solver.PreparedBalanceLawRuntime(
    transport,
    (gravity, forcing, cooling),
)
initial = runtime.initialize_state(
    mhd_integrator.initialize(
        full_cell_state,
        face_magnetic_flux,
        step_size=initial_step,
    )
)
```

The adapter reconstructs magnetic cell values for source thermodynamics but retains
face magnetic flux as the authoritative state. Gravity and forcing may modify momentum
and total energy; cooling may modify total energy. Any declared or actual attempt to
modify a `magnetic_*` component is rejected transactionally.

## Gradient and inference semantics

`PosteriorProblem.validate()` evaluates the initial log density and gradient. The same
problem can be passed to Phydrax MAP, Laplace, HMC, or NUTS routines after the schedule
has been qualified over the intended parameter region.

Shock movement, limiters, positivity activation, and schedule boundaries are
branchwise. A stochastic source realization is conditioned when held fixed; that is not
marginalization. Marginal stochastic physics requires explicit latent variables or an
explicit expectation estimator.
