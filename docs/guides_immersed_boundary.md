# Immersed-boundary and marker-flow coupling

Phydrax exposes two enforcement families:

- regularized-delta marker coupling on Cartesian, nonuniform, mapped, AMR-composite,
  and distributed MAC layouts;
- sharp cut-cell coupling through `MACSharpInterfaceProjectionPlan`.

They share accepted-state, conservation, failure, differentiation, checkpoint, and
qualification conventions. They are not interchangeable numerically: regularized
markers enforce a smoothed velocity constraint, while cut cells retain a sharp
geometric interface and integrate pressure/viscous traction on that interface.

## Marker measure and force convention

`LagrangianMarkerSetPlan` owns stable marker IDs, reference positions, a static active
mask, and positive material quadrature weights. Current positions and velocities are
temporal state. `ImmersedMarkerQuadraturePlan` materializes differentiable positions,
surface Jacobians, physical quadrature weights, and source-entity IDs from any
`BoundaryAtlas`.

`BoundaryAtlasSurfelPlan` preserves the same stable marker identity and physical
quadrature while adding oriented tangent footprints. Its materialization can
produce compatible marker kinematics for the immersed solve. Marker kernel
support, surfel footprint, and physical quadrature weight remain independent;
see [Surfels](guides_surfels.md).

`PreparedMACMarkerTransfer.interpolation_operator(relation)` is J. Its Hilbert adjoint
is S:

```text
S = W_E⁻¹ Jᵀ W_L
```

`W_E` is the MAC face-dual measure and `W_L` is marker material quadrature.
`spread(relation, value)` accepts force density per marker measure; callers must not
multiply by marker quadrature first. Diagnostics expose partition, first-moment,
gradient-sum, force, torque, and virtual-work defects.

`MACMarkerKernelPlan` selects cubic B-spline, Peskin four-point, or Roma three-point
assignment. Uniform axes use their native regularized-delta formula. Nonuniform axes
use local affine-constrained weights. `MappedMACMarkerTransferPlan` reconstructs
physical vector velocity from mapped normal-face values and applies its exact
measure-adjoint spread.

`MACMarkerRouteState` makes the nondifferentiable routing state explicit. Newton and
implicit solves recompute smooth weights on the fixed route and fail closed if the
canonical accepted route differs. Route refresh is an outer accepted-step policy, not
a hidden derivative.

## Exact fixed and prescribed coupling

`MACImmersedBoundaryProjectionPlan` enforces the shared physical-boundary correction,
incompressibility, and prescribed marker velocity in one pressure-plus-marker solve:

```text
J u = U_b
```

The returned marker multiplier is force density exerted by the body on the fluid. The
fluid receives its spread; a body receives the exact negative marker-measure adjoint
load.

The projection consumes an actual stage inverse momentum operator. Available
implementations are:

- `MACDiagonalStageInverseMomentum` for diagonal stages;
- `MACHelmholtzStageInverseMomentum` for repeated homogeneous Helmholtz solves;
- `MACVariableDensityStageInverseMomentum` for face-density mass stages;
- `MACOperatorStageInverseMomentum` for a certified SPD variable-coefficient momentum
  operator.

`MACImmersedBoundaryIMEXEulerMethod` and `MACImmersedBoundarySBDF2Method` construct
their inverse from the same Helmholtz stage used for the tentative velocity. Pressure,
multiplier, route state, and multistep history commit atomically or retain the previous
accepted state.

The default coupled solve is matrix-free GMRES with a right pressure/marker block
factorization and Jacobi approximations on the two diagonal blocks.
Rank certification and a configurable mobility condition limit are mandatory by
default. Problems larger than the configured exact rank-audit capacity require an
explicit rank-certification policy; they are not silently accepted.

## Physical boundaries and variable coefficients

`MACBoundaryCorrectionDescriptor` is the shared correction-space contract. Tentative
momentum, pressure gradient, KKT correction, and diagnostics use the same prepared
boundary identity and stage. Periodic, wall, inflow/outflow, and mixed pressure closure
therefore do not acquire a second boundary interpretation inside the immersed solve.

Variable density is advanced by the native conservative MAC density/momentum path.
For explicit viscosity, use `MACVariableDensityStageInverseMomentum`.
`MACVariableViscosityStagePlan` constructs a native SPD momentum action from the
discrete variable-viscosity strain energy; `MACOperatorStageInverseMomentum` also
accepts another certified SPD momentum operator.

## Fixed immersed MAC LES

`FixedImmersedMACLESPlan` is the integrated static LES route for prescribed
regularized markers. It binds one `MACAlgebraicLESPlan`, the exact
`MACImmersedBoundaryProjectionPlan`, stationary `LagrangianMarkerKinematics`,
caller-owned fixed cell fluid fractions, and a canonical geometry identity.
`compile_fixed_immersed_mac_les_flow` installs its prepared action in the standard
compiled MAC equation. The exact filter/stress convention remains the one defined
by the [LES equations API](api/equations/les.md#backend-support-and-refusals).

Preparation accepts only single-device, three-dimensional, unit-density flow with
stationary active markers, fixed marker identities and transfer routes, successful
untruncated support, and periodic/free-slip/symmetry outer boundaries. Cell fluid
fractions must be floating, finite, lie in `[0, 1]`, and contain active fluid.
Directional filter widths are scaled by the cube root of active fluid fraction;
eddy viscosity, deviatoric stress, and energy transfer are weighted by that
fraction, and solid-cell SGS stress is zero. Moving, deforming, distributed, open,
or changed-topology requests are refused.

Without `wall_stress`, the pressure/marker solve enforces full no-slip. With a
prepared `VectorEquilibriumWallStressPlan`, it enforces only the marker-normal
constraint and spreads the attached equilibrium law's tangential wall-on-fluid
traction. Active marker normals, sample distances, optional roughness, positive
molecular viscosity, wall-law convergence, and dissipative power must all pass.
The wall law still has no adverse-pressure-gradient, separation, or moving-wall
claim.

`plan.imex_euler_method(...)` and `plan.sbdf2_method(...)` construct the matching
immersed accepted-time methods. Their SGS rate is explicit; pressure, marker
constraint, and accepted history remain atomic. Stage output separates bulk SGS
and modeled-wall rates/work. `admission_regime()` binds the existing
prescribed-marker support tuple, so this route does not create a seventh immersed
owner or inherit a release.

Qualification case `immersed-mac-wall-stress` covers only the normal-constraint
IMEX wall-on/off tuple: slip, nonzero wall traction, normal defect, power/rate,
trajectory effect, wall-run impulse/work ledgers, and execution. Its
`sbdf2_evidence=not-claimed` field is local to that tuple.

Separate case `immersed-mac-sbdf2-restart` covers the full-vector-constraint SBDF2
route, serialized continuation history, startup and advanced balance ledgers, and
restart equivalence. Neither candidate evidence set broadens the other. See the
[LES guide](guides_large_eddy_simulation.md#wall-stress-inflow-and-immersed-coupling).

## Rigid and deformable accepted-time coupling

`RigidMarkerMapPlan` rotates body-frame markers with SO(2) or SO(3), constructs marker
velocity, and exposes the paired generalized force/torque pullback.

- `MACRigidImmersedEulerMethod` is the baseline synchronized contact-free step.
- `MACRigidImmersedBackwardEulerMethod` iterates the body pose and fluid constraint at
  accepted time.
- `MACRigidImmersedMidpointMethod` uses a pose-centred second-order update.
- `MACRigidImmersedJointMethod` couples native rigid-joint dynamics.
- `MACRigidImmersedContactMethod` couples native hard-contact complementarity.

Rigid results include accepted and attempted time, status, fluid/rigid kinetic energy,
coupling power, external work, and total energy change.

`FiniteElementImmersedMarkerMapPlan` binds a fixed FE interpolation H and material
adjoint H*. `MACDeformableImmersedBackwardEulerMethod` supplies the first-order
implicit baseline. `MACDeformableImmersedNewmarkMethod` solves configuration,
velocity, acceleration, fluid momentum, pressure, and marker no-slip in one
monolithic Newmark nonlinear system. `structural_contact_residual` admits the native
deformable-contact residual inside either solve. Acceptance includes nonlinear/KKT,
divergence, slip, gauge, route, transfer, coupling-work, and finite-state gates.

`ResolvedLubricationCorrectionPlan` adds only the asymptotic near-gap normal
resistance not already resolved by the grid. It uses a finite minimum gap, smooth
cutoff, nonnegative residual resistance, and an explicit dissipation report.

## Mapped grids, ALE, remeshing, AMR, and distribution

`MappedMACGeometryPlan` provides compatible physical-space divergence, gradient,
pressure action, velocity reconstruction, and metric evidence. The solver-level
`MACALEGeometryPlan` and `MACRemeshEpochPlan` own accepted-time geometric-conservation
and conservative remap contracts.

`CompositeMACMarkerTransferPlan` selects the finest valid owner level, enforces local
partition and first-moment constraints, applies a measure-adjoint spread across all
levels, and records accepted substep impulse. `CompositeMACProjectionPlan` projects
through caller-supplied compatible composite divergence, gradient, inverse momentum,
and gauge operators.

`DistributedMarkerOwnershipPlan` assigns every stable marker ID one deterministic
owner and records all support ranks. `DistributedMACMarkerTransfer` separates
owner-computes gather/spread, explicit halo exchange, and global force/work reduction.

`MarkerEpochPlan` and `MarkerEpochTransferPlan` make activation, deactivation,
refinement, coarsening, and migration explicit topology events. The primal map
preserves constants and weighted integrals; the dual is its material-measure adjoint.
Differentiation is either a frozen event schedule or an explicitly certified event
map.

## Sharp-interface family

`MACSharpInterfaceProjectionPlan` consumes one qualified sharp-geometry realization,
not raw fractions. The realization carries absolute fluid cell volumes and open face
measures, lower/upper measure bounds, source fidelity, topology/component identities,
and acceptance evidence. Exact source, certified bounded source, exact
piecewise-linear surrogate, and diffuse/unqualified data are distinct contracts.

Preparation constructs compact active cell/face spaces and an explicit weighted
divergence/gradient adjoint pair. Neumann gauges are applied once per disconnected
active-fluid component; Dirichlet pressure closure does not receive a mean gauge.
Small cells below the accepted lower-volume threshold fail closed rather than gaining
an unqualified denominator floor.

Interface area, centroid, normal, and traction have independent moment evidence.
Projection remains available when qualified fluid measures exist but unqualified
traction is never reported as a physical zero. `MACMovingSharpInterfaceEpochPlan`
refreshes numeric measures only while support and active topology remain fixed; a
topology change requests a host epoch rebuild.

## Fluctuating hydrodynamics and FIB

`MACFluctuatingHydrodynamicsPlan` samples thermal momentum through a supplied noise
factor B and verifies the declared covariance action B B* = D against the dissipation
operator. `MACInertialStochasticStepPlan` applies the result through the inertial mass
inverse.

`FIBOverdampedPlan` advances marker position with deterministic mobility, a Krylov
square-root Brownian increment, and random-finite-difference stochastic drift. Random
streams use `StochasticReplayKey(seed, accepted_step, stage, sample)`. Pathwise
derivatives are certified only for fixed routes/topology and fixed semantic random
keys; weak derivatives are reported as a separate policy.

## Checkpoint, replay, output, and qualification

`MarkerFlowCheckpointPayload` stores accepted fluid, pressure, marker, multiplier,
rigid, deformable, contact, route, topology, AMR, stochastic, and solver-history
state. `MarkerFlowReplayPlan` replays accepted events and states its route, topology,
event-map, and pathwise-noise derivative boundary.

`HydrodynamicLoadPlan` combines pressure, viscous, marker, lubrication, and contact
force/torque/impulse into one interval power/work record.
`MarkerFlowAdaptiveStepPlan` selects the strictest advection, diffusion, marker,
contact, lubrication, geometry, stochastic, or maximum-step restriction.
`MarkerFlowTrajectoryAdapter` exposes accepted states to trajectory consumers.
`marker_flow_artifact_reference` binds checkpoint/output/benchmark paths to content
digests. `MarkerFlowCompiledExportPlan` rejects dynamic route, topology, random
schedule, or state-shape export.

`MarkerFlowOutputPlan` writes accepted HDF5/XDMF time series and optional VTK point
snapshots for Eulerian, marker, rigid, deformable, contact, and diagnostic fields.

`MarkerFlowQualificationPlan` reports route-local divergence/slip,
force/torque/work/energy, order, covariance, interface, lubrication/contact, and replay
evidence. It does not publish a release. `ImmersedDNSQualificationProfile` is the
current unsigned candidate envelope; its nested `CapabilityProfile` and the profile
itself both retain `released=False`.

Fixed immersed MAC LES reuses the prescribed-marker admission regime while adding
its own filter, model, cell-fraction, wall-law, and compiled-action identities.
Passing the DNS admission does not qualify those LES-specific coordinates.

## Candidate regimes and runtime admission

| Candidate support tuple | Numerical owner and boundary |
| --- | --- |
| Prescribed marker | MAC pressure–marker KKT, prescribed motion, fixed marker identities, marker-reaction loads, fixed-route JVP/VJP |
| Free rigid marker | Simultaneous fluid/rigid/marker KKT, optional hard contact, marker loads, fixed-route JVP/VJP only without active contact |
| Fixed-topology sharp | Qualified absolute cut-cell measures, pressure/viscous traction, fixed active topology, fixed-topology JVP/VJP, no distributed admission |
| Deformable/contact | Accepted-time monolithic FSI with explicit contact state and optional lubrication; derivatives only on fixed routes without active contact |
| LBM body | Iterated direct forcing and its force ledger, fixed marker count, no derivative or distributed admission |
| Resolved CFD–DEM | MAC penalty markers plus soft-sphere contact/lubrication ledgers, fixed-capacity contact graph, no derivative admission |

`ImmersedBodyRegimePlan` never evaluates these owners. It binds the already prepared
owner to exact marker, geometry, route, topology, motion, and geometry-epoch identities,
plus an optional `DistributedMACMarkerTransfer` and lubrication policy where admitted.
Its gap classifier only labels resolved-grid, lubrication, contact, or inadmissible
separations; it does not apply a force.

`ImmersedReferenceCampaignPlan` consumes measured reference values for the named
manufactured-load, cylinder, sphere, added-mass, settling, flexible-contact, and sharp
cases without rerunning a flow solve. `ImmersedRuntimeAdmissionPlan` then separates
preflight from use:

```python
admission = phx.applications.incompressible_flow.ImmersedRuntimeAdmissionPlan(
    profile,
    regime,
    maximum_resource_bytes=resource_limit,
    derivative_mode="none",
)
prepared_admission = admission.prepare(preflight_evidence)
decision = prepared_admission.admit(runtime_evidence)
```

Preflight gates campaign evidence, owner identity, marker rank and condition, memory,
and derivative scope. Runtime admission additionally gates all epochs, route,
untruncated support, distributed force/work reductions, gap classification, sharp
certificate, frozen differentiation routes, and optional load provenance. Nothing
falls back to another immersed owner. Owner-computes marker transfer is an explicit
reduction path, not a claim of arbitrary multi-device or multi-host DNS support.

## Failure semantics

Exact paths fail closed on nonfinite state, invalid geometry, truncated support, rank
or condition failure, linear/nonlinear failure, divergence, gauge, slip, KKT residual,
route inconsistency, contact failure, or failed conservation evidence. Mathematical
solve differentiation is certified only when the primal and adjoint solve contracts,
route/topology policy, event policy, and stochastic-key policy all pass.
