# Cartesian ocean process modeling

Phydrax provides a three-dimensional Cartesian rigid-lid nonhydrostatic Boussinesq
process model. It is intended for idealized rotating and stratified process studies,
not basin, coastal, hydrostatic, free-surface, or global circulation modeling.

## State and reference convention

The authoritative numerical state remains the native coupled MAC coordinates:

`[face-normal velocity, named cell scalars]`.

`CartesianBoussinesqOceanPlan` declares temperature and salinity semantics without
creating another state layout. `LinearSeawaterReference` defines reference density,
heat capacity, gravity, reference T/S, thermal expansion, and haline contraction:

`rho' / rho0 = -alpha (T - T0) + beta (S - S0)`.

`OceanAxisConvention` declares the vertical axis and sign. The initial model requires
exactly three dimensions and a bounded vertical axis. Gravity along a periodic axis is
rejected.

Pressure is diagnostic kinematic pressure. Reference density is used for physical-unit
conversion and surface-stress acceleration, not as a second projected density field.

## Coupled numerical method

The model reuses `CompiledMACScalarBuoyancyDynamics`:

1. reconstruct temperature and salinity on the MAC faces;
2. compute conservative scalar fluxes;
3. evaluate buoyancy from those same face values;
4. compute skew momentum transport, viscosity, f-plane rotation, stress, and buoyancy;
5. project the complete momentum rate with the current boundary stage;
6. return projected velocity and scalar rates.

The coupling reports normalized kinetic/potential buoyancy-exchange evidence. A finite
but excessive exchange defect is not accepted.

## Coriolis and surface stress

`PreparedMACOceanForcing` builds one staggered cross-component interpolation and uses
its exact mass-weighted transpose for the opposite component. Consequently the
discrete f-plane map satisfies the MAC weighted skew relation and reports normalized
Coriolis work.

The rigid-lid surface remains impermeable. Tangential stress is applied to the upper or
lower surface layer selected by the axis convention and divided by reference density
and local surface-layer thickness. Stress work is reported separately.

The current implementation supports constant f only. Beta-plane and spherical rotation
are not implied.

## Scalar transport, diffusion, and surface flux

`MACScalarTransport` accepts either one isotropic diffusivity or one diagonal value per
grid axis. T and S may use different coefficients. The explicit diffusive restriction
uses the prepared conservative diffusion diagonal.

`MACScalarBoundaryCondition("flux", value)` prescribes outward scalar-content flux. A
positive upper-surface value removes scalar content from the domain. The value may be a
static array or a stage-time callable with a stable `function_id`.

`LinearSeawaterReference.temperature_flux_from_heat_flux` converts outward heat flux to
outward temperature-content flux by dividing by `rho0 cp`.

The rigid-lid model has fixed volume. It does not support real freshwater volume flux.
No implicit virtual-salt convention is applied.

## Time stepping and budgets

`OceanBoussinesqSSPRK33Method` evaluates the full projected coupled stage at all three
SSPRK stages. A step is accepted only when projection, scalar transport, buoyancy, and
ocean-forcing evidence all succeed.

`OceanBoussinesqContinuationState` stores the packed physical coordinates plus
accepted cumulative:

- T and S diffusive boundary content;
- Coriolis work;
- surface-stress work;
- buoyancy exchange defect;
- energy-balance defect.

Rejected steps leave both state and budgets unchanged. The method composes with
`FixedStepProblem` and `solve_fixed_step`; pass an explicit
`EuclideanStateGeometry` because the continuation is a structured PyTree.

The stable-step contract combines:

- momentum advection and viscosity;
- oriented scalar face-volume-flux CFL;
- scalar diffusion/reaction;
- f-plane inertial frequency;
- a conservative state-dependent stratification frequency.

## Restart and output

`write_ocean_checkpoint` and `read_ocean_checkpoint` use the strict pickle-free array
archive and require the exact prepared ocean identity. They preserve the full
continuation state, time, and accepted-step index.

`ocean_diagnostic_view` derives named velocity, T, S, density anomaly, buoyancy,
pressure, inventories, energy, projection evidence, Coriolis/stress power, and budget
defects. `write_ocean_output` writes those derived fields without making them
restart-authoritative.

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
).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
discretization = phx.discretization.FiniteVolumePlan(
    grid, component_names=("ocean",)
).prepare()
axes = phx.applications.ocean.OceanAxisConvention(2)
reference = phx.applications.ocean.LinearSeawaterReference()
plan = phx.applications.ocean.CartesianBoussinesqOceanPlan(
    axes,
    reference,
    coriolis_parameter=1.0e-4,
    temperature_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
    salinity_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
)
ocean = plan.prepare(discretization)
velocity = tuple(
    jnp.zeros(layout.shape) for layout in discretization.face_layouts
)
temperature = jnp.full(discretization.cell_shape, 10.0)
salinity = jnp.full(discretization.cell_shape, 35.0)
coordinates = ocean.initial_state(velocity, temperature, salinity)
continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
    coordinates
)
```

## Runnable examples

```text
python examples/ocean_inertial_oscillation.py
python examples/ocean_stratified_adjustment.py
python examples/ocean_surface_flux_column.py
```

## Deliberate limitations

Not supported:

- hydrostatic primitive equations;
- prognostic free-surface elevation;
- bathymetry or partial cells;
- wetting/drying;
- open/radiation boundaries;
- freshwater volume flux;
- beta-plane or spherical metrics;
- vertically implicit mixing or turbulence closures;
- nonlinear pressure-dependent seawater EOS;
- distributed end-to-end ocean execution;
- passive trajectory ingestion or online particle coupling.

The existing liquid-mask free-surface projection and cell-centered shallow-water solver
are separate models and are not used as a primitive-equation external mode.

## Qualification and benchmarks

Machine-readable qualification cases are available through:

```text
python tools/ocean_qualification.py --case rest
python tools/ocean_qualification.py --case inertial
python tools/ocean_qualification.py --case stratified
python tools/ocean_qualification.py --case surface-flux
```

`tools/ocean_benchmarks.py` separates JIT compilation from steady coupled-RHS and
accepted-SSPRK throughput for isotropic/directional diffusion and rotation on/off.
