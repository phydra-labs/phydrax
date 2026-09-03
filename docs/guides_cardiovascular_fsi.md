# Cardiovascular fluid–structure interaction

PhydraX provides two deliberately separate cardiovascular FSI routes:

- **Immersed LBM–FEM coupling** for fixed-grid flow coupled to a compliant wall or
  leaflet through sparse Lagrangian markers.
- **Conforming ALE flow** while every valve clearance remains above a declared
  minimum gap.

True leaflet contact is a third, explicit workflow layered over the existing solid
contact and either an immersed or cut-cell fluid geometry. A minimum-gap ALE route
cannot silently become a contact route.

## Units and signs

The cardiovascular kernel uses millimetres, milliseconds, milligrams, millivolts,
kilopascals, and cubic millimetres. Convert external data at the case boundary.
Marker force passed to `PreparedSparseMarkerTransfer.spread` is the force **on the
fluid**. Its negative is the load on the structural body. Positive contact gap is
open; negative gap is penetration. Record the torque origin and body centres in the
same physical coordinate frame as the marker positions.

## Sparse immersed transfer

`SparseMarkerTransferPlan` requires explicit stable marker IDs. `prepare` performs
cell search on the host and fixes one bounded-width route table. For a spatial
dimension `d` and stencil width `w`, runtime storage and gather/scatter work are
proportional to `capacity * w**d`; spreading additionally initializes one grid-sized
output. No cell-by-marker matrix is constructed.

```python
import jax.numpy as jnp
import phydrax as phx

from phydrax.applications.cardiovascular import (
    ImmersedDirectForcingPlan,
    SparseMarkerTransferPlan,
)

grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(64, periodic=True),
        phx.discretization.UniformCellAxisSpec(64, periodic=True),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray([[0.0, 0.0], [32.0, 32.0]]))
lattice = phx.discretization.LatticeBoltzmannPlan(
    grid, phx.discretization.D2Q9()
).prepare()

marker_ids = jnp.arange(128, dtype=jnp.int32)
angle = 2.0 * jnp.pi * marker_ids / marker_ids.size
position = jnp.stack(
    (16.0 + 6.0 * jnp.cos(angle), 16.0 + 6.0 * jnp.sin(angle)), axis=-1
)
transfer = SparseMarkerTransferPlan(
    lattice,
    marker_ids,
    stencil_width=4,
    minimum_coverage=1.0,
).prepare(position)
relation = transfer.relation(position)
```

The route indices and validity masks are nondifferentiable preparation data. Within
that fixed topology, `relation(new_position)` recomputes smooth kernel weights and
first-moment correction. If a marker leaves its prepared support, coverage evidence
fails closed instead of rebuilding topology inside a differentiated computation.
Reprepare at an explicit geometry epoch.

For a grid velocity `u` and total marker force `F`, use:

```python
marker_velocity = transfer.interpolate(relation, u)
fluid_force_density = transfer.spread(relation, F)
evidence = transfer.diagnostics(
    relation,
    u,
    F,
    marker_velocity=wall_velocity,
    torque_origin=torque_origin,
    body_indices=marker_body,
    body_centers=body_centres,
)
```

Qualification requires:

- partition of unity and first-moment reproduction on every active marker;
- prepared support coverage at or above `minimum_coverage`;
- integrated Eulerian force equal to the marker resultant;
- Eulerian and marker torque equal about the declared origin;
- interpolation power equal to integrated spreading power.

`interface_power_residual` additionally compares fluid power with the equal-and-
opposite body work at the supplied wall velocity. It approaches zero as no slip is
resolved; it is not substituted for the exact transpose identity.

Inactive capacity slots retain fixed shapes, contribute zero, and keep their stable
IDs. Resource limits are checked before route arrays are materialized.

## Sparse direct forcing and compliant walls

`ImmersedDirectForcingPlan` applies a fixed number of multi-direct-forcing corrections
through the sparse transfer. Marker quadrature measures convert acceleration into
total marker force, and spreading divides by the Cartesian cell measure. The result
retains corrected velocity, total marker force, force density, no-slip residual, and
transpose evidence.

```python
forcing = ImmersedDirectForcingPlan(
    transfer,
    iteration_count=8,
    convergence_tolerance=1.0e-5,
)
result = forcing.apply(
    fluid_velocity,
    density,
    wall_position,
    wall_velocity,
    marker_measure,
    time_step,
)
if not bool(result.evidence.successful):
    raise RuntimeError("immersed forcing candidate failed qualification")
```

The convenience `build_immersed_fsi_participants` constructs native partitioned
coupling participants:

- the LBM participant receives separate wall-position and wall-velocity ports,
  performs sparse direct forcing, advances the fluid callback, reinterpolates the
  actual post-LBM candidate velocity, and outputs the equal-and-opposite marker load;
- the FEM participant receives marker load and outputs separate updated wall-position
  and wall-velocity ports.

The returned graph uses bounded damped fixed-point iteration for the added-mass
loop. Position, velocity, and load have independent spaces, physical reference
scales, and convergence tolerances. It is a native `PreparedCoupling`, so a failed
fluid solve, structural solve, post-LBM no-slip qualification, or interface
convergence leaves the accepted state unchanged.

```python
bundle = build_immersed_fsi_participants(
    forcing,
    fluid_fields,
    advance_lbm,
    advance_fem,
    marker_measure,
    position_reference=10.0,  # mm
    velocity_reference=0.1,   # mm / ms
    force_reference=1.0,      # mg mm / ms²
    position_tolerance=1.0e-4,
    velocity_tolerance=1.0e-5,
    force_tolerance=1.0e-4,
    damping=0.5,
    maximum_iterations=30,
)
prepared = bundle.prepare(
    initial_fluid_state,
    initial_solid_state,
    initial_wall_position,
    initial_wall_velocity,
)
```

The participant callbacks return `ImmersedLBMAdvanceResult` and
`ImmersedFEMAdvanceResult`. Their candidate state must have the same PyTree shape as
the corresponding initial participant state. Discrete route changes are outside the
coupling derivative boundary; this fixed-point bundle therefore declares no implicit
coupling derivative.

## Conforming noncontact ALE

Wrap the existing `MACALEGeometryPlan` with an `ALEMinimumGapRoute`. The native ALE
substrate remains authoritative for mapped metrics, wall kinematics, relative flux,
projection, and geometric conservation law (GCL) evidence.

```python
motion = phx.solver.MACALEGeometryPlan(
    finite_volume,
    coordinate_map,
    grid_velocity,
    mapping_id="patient-case-ale-motion",
)
gap_route = ALEMinimumGapRoute(
    leaflet_clearances,
    swept_leaflet_clearance_bound,
    minimum_gap=0.05,  # mm
    route_id="noncontact-clearance",
)
ale = CardiovascularALEPlan(
    motion,
    gap_route,
    gcl_tolerance=1.0e-9,
).prepare()
state = CardiovascularALEState(face_velocity, pressure)
transition = ale.advance(state, start_time, time_step, case_args)
state = transition.accepted_state
```

A window is accepted only when cell volumes, face measures, velocity dual measures,
and oriented dual distances remain positive above their configured thresholds; the
map velocity, boundary kinematics, free stream, mapped adjoint, and GCL residuals
also must pass. `swept_leaflet_clearance_bound` must return a conservative clearance
lower bound over the entire supplied start/end time window, not merely endpoint
samples. The transition records `swept_certified` and rejects a swept bound below the
minimum gap. Any failure rolls velocity and pressure back atomically.

Do not use `ALEMinimumGapRoute` once true contact is possible. End the ALE geometry
epoch before the clearance threshold and start a prepared leaflet contact route.

## Leaflet contact with immersed or cut-cell fluid geometry

`LeafletContactWorkflowPlan` consumes a native
`DeformableContactResidualPlan`. Its structural callback receives the assembled
contact residual, advances the fixed-shape solid state, and returns
`LeafletStructuralAdvanceResult`.

Choose exactly one fluid route type:

- `ImmersedLeafletRoute` evaluates frozen sparse-marker coverage plus an explicitly
  supplied resolved leakage probe.
- `CutCellLeafletRoute` evaluates `MACDiffuseSDFGeometryPlan`, measures the open-area
  fraction on prepared leakage faces, checks its geometric-conservation residual,
  and reports the fraction of small cells requiring refinement.

```python
cut_cell_route = CutCellLeafletRoute(
    diffuse_sdf_geometry,
    geometry_arguments,
    leakage_face_masks,
    maximum_leakage_proxy=0.02,
    maximum_gcl_residual=1.0e-7,
    maximum_small_cell_fraction=0.01,
)
workflow = LeafletContactWorkflowPlan(
    deformable_contact_residual,
    cut_cell_route,
    advance_leaflet,
    maximum_penetration=0.01,
).prepare()
state = workflow.initialize(configuration, velocity, start_time, case_args)
transition = workflow.advance(state, start_time, time_step, case_args)
state = transition.accepted_state
```

The transaction retains gap, maximum penetration, active contact count, normal
pressure, native contact success, action–reaction residual, contact power,
dissipation, leakage proxy, cut-cell GCL residual, and refinement evidence. Both the
accepted start contact state and the candidate contact state must satisfy the native
contact result. Structural, contact, leakage, and fluid-geometry state commit
together or all roll back.

The cut-cell leakage value is a grid-resolved physical open-face-area ratio: both
open and reference areas use the finite-volume face measures, not velocity dual
measures. The immersed leakage callback is likewise a declared numerical observable.
Neither route certifies an exact seal, and `exact_sealing_certified` is always false.
Refine the geometry, repeat the qualification, and compare leakage across resolution
before interpreting it physically.

## Qualification and benchmark

Run the focused unit contract with the repository test policy, then execute:

```text
python benchmarks/cardiovascular_fsi.py \
  --grid-counts 16 32 64 \
  --marker-counts 16 64 256
```

The benchmark records cell count, marker capacity, fixed route count, relation and
workspace bytes, compile and execution time, transpose residuals, direct-forcing
residual, and ALE GCL/admissibility evidence. `dense_matrix_entries` is always zero;
route count changes with marker capacity and stencil width, not with the product of
cell and marker counts.
