# Hydrostatic primitive-equation ocean modeling

Phydrax provides a separate hydrostatic primitive-equation product alongside the
Cartesian rigid-lid Boussinesq process model. The hydrostatic product owns prognostic
free-surface elevation, integrated layer transports, conservative tracer inventories,
column geometry, external-mode evolution, vertical mixing, coastal boundaries, and
wetting/drying.

## Authoritative state

`HydrostaticOceanState` contains:

- cell-centered free-surface elevation `eta`;
- x/y face-located integrated layer volume transports;
- cell tracer inventories, including `absolute_salinity` and
  `conservative_temperature`;
- typed TKE inventory.

Velocity and concentration are derived only on positive support. Dry cells have zero
volume and inventory; dry faces have zero transport.

For every geometry epoch:

`sum_k V[k] = A (H + eta)`.

The same signed integrated face fluxes drive free-surface continuity, diagnosed vertical
flux, tracer advection, boundaries, freshwater, and accepted ledgers.

## Geometry

`TensorZHydrostaticGridPlan` prepares Cartesian tensor-z geometry with either:

- `zstar`: every layer follows the current column thickness;
- `partial-z`: fixed z levels intersect static bed and free surface, with small bottom
  cells merged upward.

`LatitudeLongitudeHydrostaticGridPlan` remains the one-block bounded spherical C-grid.
For global or polar work, `polar_cap`, `tripolar`, and
`equiangular_cubed_sphere` prepare `SphericalHydrostaticMosaicPlan` products backed by
the canonical `PreparedMultiblockGrid`. Every spherical block exposes a
`PreparedHydrostaticGrid`; `prepare_ocean` and `prepare_oceans` pass those grids through
the standard hydrostatic preparation workflow. Prepared topology interfaces,
`seam_traces`, and `seam_transport_traces` provide executable block-face routes for
cell fields and hydrostatic face transports, including trace orientation.
`scatter_seam_flux` applies the equal/opposite integrated-flux convention. Blocks
retain finite spherical cell areas, directional great-circle metric scales, tangent
coordinates, Coriolis, and vertical capacity. Atlas topology, seam route, orientation,
and block shapes remain static.

A metric epoch contains current layer volumes, face apertures, active support, total
depth, and finite/positivity evidence.

## Hydrostatic pressure and vertical continuity

Density is evaluated at cell centers. Hydrostatic pressure anomaly is integrated from
the surface downward on the same layer interfaces used by horizontal pressure forces.
The horizontal force is integrated over the common wet face aperture.

Vertical volume flux is diagnosed bottom-up from the exact horizontal layer-flux
divergence. Bottom flux is zero and the top flux equals minus the depth-integrated
horizontal divergence.

`TEOS10GSW75EOS` implements the named official 75-term specific-volume polynomial for
already supplied Absolute Salinity, Conservative Temperature, and sea pressure.
Density, thermal expansion, haline contraction, and the density pressure derivative per
Pa are JAX-native derivatives of the same polynomial. Funnel violations return
`valid=False`; hydrostatic views preserve EOS validity, finiteness, and success
evidence, and a step cannot commit unless both evaluated and accepted-state EOS
evidence succeeds. Inputs are not clipped and never fall back to
`NonlinearSeawaterPolynomialEOS`. This is not the complete GSW API and does not perform
Practical/Absolute Salinity atlas conversion, freezing/ice calculations, or arbitrary
Gibbs derivatives.

## Free surface

`LinearImplicitFreeSurfacePlan` solves a matrix-free Helmholtz equation for eta and
applies the resulting depth-uniform pressure correction to every active layer.
The operator has an identity mass term and therefore does not use a pressure gauge.
The result reports continuity residual, CG iterations, finiteness, and success.

`HydrostaticPrimitiveEquationPlan(external_mode="split-explicit")` uses an explicit
`ExternalModeSubcyclePolicy`. A fixed policy reproduces a declared count; an
adaptive-CFL policy computes a masked count inside a fixed `maximum_substeps`
capacity. The complete schedule, active mask, count, maximum Courant number, and
capacity status remain in continuation/evidence. Every active fast step evaluates
boundaries, accumulates the time-integrated transport register, and reconciles accepted
layer transports. Capacity overflow fails the candidate rather than truncating it.

## Wetting and drying

`wetting_and_drying=True` activates a conservative barotropic donor limiter during
every split-explicit substep. A donor cannot export more column volume than available.
Free surface is never clipped after the update. Newly wet tracer inventory arrives only
through conservative flux and declared source composition.

`WetDryEpochPolicy` and `HydrostaticWetDryEventPlan` make the topology boundary
explicit. Smooth differentiation is valid while the wet mask is fixed. An isolated,
transverse wet/dry transition returns reset/saltation evidence; simultaneous or grazing
changes report `derivative_available=False`. Retained dry tracer/TKE inventory and
zeroed inactive-face transports are observable. No global smooth shoreline gradient is
claimed.

## Open boundaries

`HydrostaticOpenBoundary` supports:

- closed;
- prescribed elevation;
- prescribed transport;
- Flather-like external-wave relation;
- radiation-like outgoing relation.

Each boundary carries exterior SA/CT composition for inflow. Outflow uses the interior
upwind state. The identical normal transport is used by continuity and tracer fluxes.

## Freshwater

`FreshwaterVolumeFluxPlan` is a real volume source. It changes eta/column volume and
injects explicit SA/CT composition. Zero-salinity input dilutes concentration while
conserving existing salt inventory.

It is distinct from the rigid-lid product's virtual scalar flux.

## Rotation and spherical metrics

Cartesian rotation uses `f = f0 + beta * y`. Latitude-longitude geometry instead uses
`f = 2 Omega sin(latitude)` with metric areas and edge lengths. Coriolis is constructed
as an operator/weighted-transpose pair for zero discrete work under the transport
pairing.

Beta-plane remains Cartesian and is not called spherical geometry.

## Seawater thermodynamics

`LinearHydrostaticEOS` provides the linear Boussinesq compatibility model.
`NonlinearSeawaterPolynomialEOS` provides a pure-JAX nonlinear SA/CT/sea-pressure
polynomial with analytic alpha, beta, and pressure derivative. Pressure input is sea
pressure in dbar; the returned pressure derivative is per Pa. The polynomial enforces a
declared oceanographic input range and does not claim exact TEOS-10 equivalence.

## Vertical and mesoscale closures

`HydrostaticMixingPlan` supports:

- prescribed nonnegative vertical coefficients;
- smooth Richardson/convective adjustment;
- KPP-like local and nonlocal vertical mixing;
- prognostic TKE-based coefficients and inventory;
- Redi symmetric tracer transport plus GM-like skew transport.

All vertical implicit solves use the checked batched tridiagonal line solver. Inactive
segments are decoupled, zero-flux mixing is conservative, and solve residuals enter the
accepted evidence.

## Passive trajectories

`lower_ocean_trajectories` lowers padded observations into canonical
`TrajectoryData`. `PassiveOceanTrajectoryPlan` samples a prepared hydrostatic snapshot
and advances a fixed-capacity particle batch with explicit executed masks, exits,
capacity status, and source identity. Diagnosed vertical volume transport is divided
by horizontal cell area before sampling, so all three sampled components have velocity
units. Spherical motion converts tangent velocity to longitude/latitude rates; topology
and interpolation routing remain fixed.

## Time stepping and accepted ledgers

`HydrostaticIMEXMidpointMethod` evaluates an explicit midpoint state and performs the
free-surface and vertical implicit corrections at each stage. Accepted continuation
stores:

- physical state;
- volume/freshwater/open-boundary ledger;
- tracer changes and sources;
- kinetic/free-surface energy changes;
- Coriolis work;
- mixing residual/dissipation;
- limiter, filter, and reconciliation corrections;
- the complete masked split-subcycle schedule and phase;

Rejected steps preserve state, geometry-dependent history, and ledger.

## Restart and output

`write_hydrostatic_checkpoint` stores model, method, geometry, EOS, closure, boundary,
and state identities plus time, accepted step, continuation, and ledger. Rigid-lid
checkpoints are rejected.

`hydrostatic_diagnostic_view` and `write_hydrostatic_output` expose eta, depth, layer
volumes, velocity, tracers, density, hydrostatic pressure, vertical flux, wet support,
energies, inventories, and ledger evidence without making derived fields authoritative.

## Deliberate limits

- mosaic topology and seam routing are fixed during an execution epoch;
- the GSW75 surface is the documented SA/CT/sea-pressure subset, not all TEOS-10;
- KPP/TKE/Redi/GM coefficients are native configurable closures, not calibrated global
  climate defaults;
- distributed vertical decomposition is unsupported;
- global smooth wet/dry topology gradients are unsupported;
- passive trajectories do not choose a new grid or provider inside transformed
  execution.

## Minimal construction

```python
import jax.numpy as jnp
import phydrax as phx

grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(3, periodic=True),
        phx.discretization.UniformCellAxisSpec(3, periodic=True),
        phx.discretization.UniformCellAxisSpec(3, periodic=False),
    ),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray(((0.0, 0.0, -10.0), (3.0, 3.0, 0.0))))
discretization = phx.discretization.FiniteVolumePlan(
    grid, component_names=("hydrostatic",)
).prepare()
geometry = phx.discretization.TensorZHydrostaticGridPlan(
    discretization, jnp.full((3, 3), 10.0)
).prepare()
ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
    geometry
).prepare()
state = ocean.initialize_state(jnp.zeros((3, 3)))
continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
    ocean, state
)
```
