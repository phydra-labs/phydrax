# Dry column and Cartesian vertical-slice atmosphere

The dry atmosphere application is a compressible, inviscid reference testbed on a
fixed uniform Cartesian finite-volume mesh. A column uses `(z,)`; a slice uses
`(x, z)`, with height increasing upward. Coordinates, times, pressure,
temperature, velocity, density, and gravity are SI. Column integrals are per unit
horizontal area and slice integrals per unit out-of-plane length.

This method is **not global atmospheric validation**. It has no rotation,
spherical geometry, terrain, moving mesh, radiative transfer, moisture,
turbulence closure, or acoustic filtering. Cold-pool and thermal examples are
inviscid method-qualification problems, not resolved turbulent-front forecasts.

## Thermodynamics and reference states

`DryAir` explicitly models N2, O2, and Ar with molar masses 0.0280134, 0.0319988,
and 0.039948 kg/mol. The default mole amounts 0.78084, 0.20946, and 0.00934 are
normalized after excluding trace gases. Constant molar heat capacities at volume
are 5R/2 for the two diatomic gases and 3R/2 for argon. This is a calorically
perfect dry surrogate with a declared 150–400 K operating interval, not a
high-temperature chemical mechanism. Internal energies use the cv*T reference.

The implementation composes `ChemicalSpeciesSchema`,
`PolynomialSpeciesThermodynamicsPlan`, `HomogeneousHelmholtzPlan`, and
`HomogeneousMixtureEulerSystem`. Pressure, sound speed, caloric recovery,
admissibility, and primitive/conserved conversion remain native thermodynamic
operations. The all-gas dry schema is not a place to insert liquid or ice;
multiphase moist thermodynamics requires its own phase-aware composition.

`DryHydrostaticReference` supports:

- **Isothermal:** T = T0 and p = p0 exp[-g(z-z0)/(Rdry T0)].
- **Isentropic:** T = T0 - g(z-z0)/cp and p = p0 (T/T0)^(cp/Rdry).

The reference must remain within the thermodynamic domain at all cell
quadrature points and faces. Eight-point vertical Gauss quadrature forms
reference cell averages. Faces use the analytic reference. These reference
families have fixed composition and constant positive gravity.

## Equilibrium pairing and energy accounting

The numerical method is a local Rusanov interface flux with either first-order
or monotonized-central MUSCL reconstruction of **conserved departures from the
reference**, not reconstruction of the entire stratified state. The reference
state is added at the face. Its numerical flux is computed by exactly the same
native Rusanov/EOS operation as the actual face flux.

The momentum/continuity operator is

    dU/dt = -div(F(Uface) - F(Ureference_face))
            - g (rho-rho_reference) e_vertical_momentum.

Thus at discrete hydrostatic rest both reconstructed departures and the paired
flux difference vanish. Cancellation is performed **before** divergence, not
between a large independently approximated pressure gradient and `rho*g`.
Equivalently the full momentum source is the discrete reference pressure-flux
divergence minus g times the density departure. The reference pressure drop
agrees with integrated column weight to EOS/quadrature roundoff.

Each face has one mass flux: the sum of its species fluxes. With Phi = g*z,
gas-energy evolution uses

    dEgas/dt = -div(Fenergy + Phi_face Fmass) + Phi_cell div(Fmass).

The continuity equation therefore cancels the potential-energy exchange in
`Egas + rho*Phi_cell`. No independent `rho*u*g` evaluation is used. Open-boundary
budgets use the same `Fenergy + Phi_face Fmass`; closed walls reflect face normal
momentum and have zero mass/energy flux. Static gravity does no net work on this
gas-plus-potential-energy budget. Momentum has the explicit integrated gravity
source; it is not claimed globally conserved in a gravitational box.

The spatial route is nominally second order in smooth regions at `order=2`,
with limiter order reduction at extrema/discontinuities and first-order
exterior ghost reconstruction. Qualification reports measured refinement,
not an assumed order. Time integration is SSPRK(3,3), using its 1/6, 1/6, 2/3
weights for the accepted boundary/source ledger.

## Boundaries and stability

Boundary entries are one `(lower, upper)` pair per coordinate axis:

- `("periodic", "periodic")`: horizontal periodicity only. Gravity is not
  periodic, so a vertically periodic domain is rejected.
- `"closed"`: stationary inviscid wall. The exterior *face* state reflects the
  interior normal momentum; tangential momentum is unchanged.
- `"prescribed"`: stationary exterior **conserved state**, not a specified normal
  flux. The matching `prescribed` entry broadcasts to that boundary face batch.
  When omitted, the analytic hydrostatic face state is prescribed. Actual
  inflow/outflow is determined by the numerical interface problem.

The acoustic step bound is CFL divided by the maximum cell sum of adjacent-face
signal speed over cell width. A gravity bound sqrt(CFL*dz/g) also applies.
CFL must be in `(0, 0.5]`; the default is 0.35. Each SSPRK stage checks its actual
reconstructed-face admissibility, trial-state admissibility, and stability
bound. A failing stage rejects the entire macro step, preserving content, time,
accepted-step count, and ledgers. There is no vacuum clipping, silent fallback,
or automatic step reduction. `stable_step` returns zero for inadmissible face
reconstruction. Supplied near-vacuum initial conditions are rejected.

## Usage and continuation

```python
import equinox as eqx
import jax.numpy as jnp
from phydrax.applications.atmosphere import (
    DryAtmospherePlan, DryHydrostaticReference,
)

prepared = DryAtmospherePlan(
    (16, 24), ((0., 0.), (20000., 10000.)),
    reference=DryHydrostaticReference("isentropic", temperature=300.),
).prepare()
state = prepared.initial_state()  # hydrostatic cell averages
step = 0.5 * eqx.filter_jit(prepared.stable_step)(state)
result = eqx.filter_jit(prepared.rollout)(state, jnp.full((4,), step))
assert bool(result.successful)
restart = prepared.checkpoint(result.state)
continued_state = prepared.restore(restart)
```

Use `thermal_state(delta_temperature, velocity=...)` for pressure-balanced
thermal perturbations, or `initial_state(conserved)` for explicit Euler data.
The conserved component layout is three species densities, Cartesian momenta,
and gas total energy. `budget` returns species masses, momenta, gas energy,
potential energy, total energy, cumulative boundary/source integrals, and
closure `current - initial + boundary - source`.

`DryAtmosphereState.content` is the native
`FiniteVolumeConservativeContentState`: authoritative extensive values,
geometry/precision identities, and time. `PreparedDryAtmosphere` implements
`AbstractFixedStepMethod`, making atomic accepted steps available to native
fixed-step production continuation. Generic `PreparedFiniteVolumeRuntime`
fallback dynamics are intentionally not wrapped: rebuilding its generic
fallback would remove this method's equilibrium pairing. Atmospheric rollout
uses a prescribed step vector and freezes after its first rejection. Its stored
trajectory contains accepted endpoints, excluding the initial state.

A restart contains the complete accepted content and budget state, its prepared
identity, and a content fingerprint. Restoring into changed gravity,
thermodynamics, grid, boundary values, precision, or method fails. Native Equinox
PyTree serialization can persist this object with its matching template; no
separate file-format or schema-version contract is introduced.

## Executable evidence

Run from the repository with double precision:

```sh
JAX_ENABLE_X64=1 python -m examples.dry_atmosphere --case column
JAX_ENABLE_X64=1 python -m examples.dry_atmosphere --case rising_thermal
JAX_ENABLE_X64=1 python -m tools.dry_atmosphere_qualification --nx 16 --nz 12 --steps 4
pytest -n auto tests/unit/applications/test_dry_atmosphere.py
```

The qualification driver reports both hydrostatic families, actual nonlinear
steady-shear residual refinement at 12/24/48 vertical cells, gravity-wave,
rising-thermal and density-current onset, closed-box energy accounting,
restart agreement, near-vacuum admissibility, atomic unstable-step rejection,
and separate first-run versus warm execution time. The stationary manufactured
shear has hydrostatic p/rho, arbitrary horizontal u(z), and zero vertical
velocity; no numerical-residual-cancelling manufactured source is inserted.
The regression suite additionally exercises prescribed-boundary mass and
gas-plus-potential-energy transport with restart-complete budget ledgers.

The gravity-wave example is a small buoyancy perturbation of stable isothermal
stratification advected by 20 m/s horizontal flow. Warm and cold cosine bubbles
start in isentropic reference atmospheres. These scripts expose actual measured
values and fail on rejected schedules/budget or refinement failures. Short
runs verify onset and conservation, not full-time published benchmark
agreement, global circulation, or climate skill.
