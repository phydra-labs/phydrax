# Astrodynamics

`phydrax.applications.astrodynamics` provides explicit scale, epoch, frame, state,
force, propagation, transfer, event, ephemeris, many-body, and spacecraft contracts.
Numeric solver time is always relative to a two-part reference epoch. State arrays do
not carry runtime quantity objects; every state and product carries a static context ID.

## Two-body propagation

```python
import jax.numpy as jnp
import phydrax as phx

astro = phx.applications.astrodynamics
context = astro.AstrodynamicsContext(
    astro.AstrodynamicsScaleContract.si(),
    astro.ReferenceEpoch(
        astro.TimeInstant(astro.JulianDate(2451545.0), "TT")
    ),
    astro.FrameDefinition("central", "inertial", pseudo_inertial=True),
)
initial = astro.CartesianOrbitState(
    jnp.asarray([1.0, 0.0, 0.0]),
    jnp.asarray([0.0, 1.0, 0.0]),
    context,
)
result = astro.propagate_universal_kepler(initial, 0.25, 1.0)
assert bool(result.valid)
```

Universal propagation covers elliptic, near-parabolic, and hyperbolic conics using
stable Stumpff functions and a bounded universal-anomaly solve. A custom JVP
implicitly differentiates the converged equation; convergence and the chosen conic
regime remain explicit evidence.

## Numerical and geometric propagation

`AstrodynamicsPropagationPlan` lowers a pure force to `ContinuousSystem` and
`DifferentialProblem`. It delegates adaptive execution to `solve_diffrax` and canonical
fixed-step Hamiltonian execution to `StormerVerlet`. The result retains the native
`DifferentialSolution` plus context and energy/angular-momentum diagnostics.

## Elements, Lambert, and events

Modified equinoctial elements are the nonsingular primary chart. Classical elements
return an explicit singular status for circular or equatorial states. `LambertPlan`
returns fixed-capacity zero- and multi-revolution branches without choosing one for the
caller. Orbital events adapt scalar guards and resets to the existing hybrid-event and
saltation substrate.

## External data

Ephemeris and time products require producer, version, checksum, license, frame,
epoch, scale, and differentiability provenance. SPICE, SGP4, and coordinate-system
adapters execute on the host and return native arrays. No provider lookup, network
access, or file parsing occurs inside transformed computation.

## Differentiability boundaries

Smooth fixed branches support JIT, vmap, JVP, and VJP. Branch selection, event bracket
selection, grazing events, collisions, classical-element singularities, external data
loading, and static-capacity transitions do not have an ordinary smooth derivative.
Inspect `valid`, `status`, residuals, and iteration evidence before consuming results.
