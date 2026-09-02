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

## Regularized encounters and native TLE propagation

`CloseEncounterRegularizationPlan` prepares exactly one dominant pair for a bounded
segment. Preparation rejects simultaneous close pairs, tied/grazing selection,
collisions, non-finite state, and invalid mass capacity. Execution uses a Sundman
universal-variable solve, retains the KS coordinate and momentum evidence, and rolls
back the complete state if time closure, perturbation ratio, collision separation, or
iteration capacity fails. Collision merge/bounce remains an event reset rather than
part of the smooth segment. `detect_close_encounter(...,
regularization_prepared=True)` marks the explicit handoff; it never silently enables
regularization.

`TLEPropagationPlan` is the clean-cutover native TLE surface. The parsed period fixes a
`near-earth` or `deep-space` route at preparation. The deep-space route includes
solar/lunar secular and long-period terms plus fixed-capacity synchronous and
twelve-hour resonance scans. Results retain the UTC epoch, TEME frame, WGS-72 constant
set, regime, resonance kind, range/decay checks, residual, and executed resonance
steps. Requests outside `maximum_minutes` fail rather than truncate. TLE parsing,
regime, resonance, and revolution metadata are discrete; elapsed time remains a
continuous input inside the prepared route.

## External data

Ephemeris and time products require producer, version, checksum, license, frame,
epoch, scale, and differentiability provenance. SPICE, SGP4, and coordinate-system
adapters execute on the host and return native arrays. No provider lookup, network
access, or file parsing occurs inside transformed computation.

`bundled_astronomy_data_store()` resolves the small packaged, content-addressed
astronomy set without networking: UTC/TAI leap seconds with an explicit coverage end,
a bounded historical EOP/CIP interval, low-order Earth gravity, a one-day
Sun/Earth/Moon Chebyshev example, and bounded IAU precession coefficients. The typed
`load_bundled_*` functions verify digest and byte size before constructing the native
product. These data are deterministic examples, not auto-current operational EOP,
ephemeris, gravity, or TLE services; out-of-coverage evaluation fails and callers must
provide an explicit `AstrodynamicsDataStore` for other coverage.

## Differentiability boundaries

Smooth fixed branches support JIT, vmap, JVP, and VJP. Branch selection, event bracket
selection, grazing events, collisions, classical-element singularities, external data
loading, and static-capacity transitions do not have an ordinary smooth derivative.
Inspect `valid`, `status`, residuals, and iteration evidence before consuming results.

## Shared core substrates

`AstrodynamicsScaleContract` is the shared `DimensionalScaleContract` with physical
length coordinates; epoch and frame remain astrodynamics-owned. Direct and hierarchical
gravity now delegate to core `NewtonianPairKernel`, runtime Morton octree, and
`BarnesHutGravityPlan`; the previous application-local nominal FMM and TreePM names were
removed because they did not implement those algorithms. Orbit-determination whitening
uses the core Cholesky covariance action, while range/range-rate/RA-Dec geometry remains
astrodynamics-specific. Artifact manifests use the core checksum/lineage contract.
