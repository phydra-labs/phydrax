# Material point method

## Spatial method and runtime state

::: phydrax.discretization.mpm
    options:
      show_root_heading: false
      members_order: source

## Canonical boundary loads and viscous absorption

MPM reuses the mechanical-load vocabulary owned by
`phydrax.applications.solid_mechanics`; it does not introduce a separate traction
or pressure law. Prepare an explicit boundary quadrature with
`PreparedMPMBoundaryLoad(splat, reference_coordinates, reference_area_vectors,
canonical_load, /, *, geometry=None, field_slot=0)`. Coordinates have shape
`(boundary_points, spatial_dimension)`. Each oriented area vector includes its
complete quadrature weight: its magnitude is the physical reference area and
its direction is the outward reference normal. Boundary points need not coincide
with volume particles. The adapter evaluates the canonical load and transfers its
weighted forces to the grid using the same splat assignment.

Alternatively,
`PreparedMPMBoundaryLoad.from_surface_particles(splat, particle_slots,
reference_coordinates, reference_area_vectors, canonical_load, /, **kwargs)`
selects coordinates from the **complete** reference particle-position array;
area vectors correspond only to the selected slots. Selection happens during
preparation. This constructs the same boundary quadrature as passing those
selected coordinates directly, so equal coordinates, areas, load, and field slot
give equal forces and ledgers. Surface-particle selection is a preparation
convenience, not a second surface-measure convention or a live attachment to
particle history.

Point assignments (multilinear and cardinal B-splines) admit an independent
boundary-point count. Domain-bearing assignments require matching prepared
source-domain extents; surface-particle preparation selects those extents along
with the coordinates. Cartesian CPDI and evolving GIMP use the canonical
assignment update with boundary position and deformation gradient, preserving
their moving support rather than replacing it with a point assignment.
Axisymmetric assignment admission follows the owning geometry plan's support
restrictions.

Pass tuples as `MaterialPointProblemIR(..., boundary_loads=(adapter, ...),
dashpots=(dashpot, ...))`; compilation forwards them to
`PreparedMPMDynamics`. Supply the corresponding canonical controller states with
`MaterialPointArguments(material_parameters, external_arguments=None, /, *,
mechanical_load_states=(MechanicalLoadState(...), ...), boundary_arguments=None)`.
The controller states are immutable evaluation inputs, distinct from evolving
boundary geometry. `field_slot` identifies the receiving velocity field.

### Frames, measures, and potentials

`ReferenceDeadTraction` is a spatial traction density per **reference** area;
`CurrentSurfaceTraction` is a spatial density per **current** area. The adapter
retains the canonical load's configuration, measure-frame, load-frame, and
potential/virtual-work semantics. Current area and orientation come from the
canonical `DeformedMeasureState` and Nanson transformation, not from particle
volume or an independently chosen normal. Quadrature weights are applied once.

For normal-dependent pressure use a canonical `GeneralFollowerLoad` with
`support="boundary"` and `measure_frame="current"`: its law can obtain the
current unit normal as `measure.normal("current")`. A law returning `p * n`
applies outward pressure; an inward compression uses the opposite sign.
The sign belongs to the canonical load, not to the MPM adapter. A vector-valued
`CurrentSurfaceTraction` is not implicitly interpreted as a pressure.

Canonical potential density, when provided, is integrated with its declared
measure and returned as `MPMBoundaryLoadResult.potential`. For example, reference
dead traction has a certified potential. General follower traction need not have
one: `None` remains `None`, not a fabricated zero energy, and the implicit solver
does not assume a potential-based merit for virtual-work loads.

For axisymmetric `(r, z)` geometry, supply **ring-integrated physical areas**,
just as particle masses and volumes are ring-integrated. Preparation coordinates
and area vectors remain two-dimensional. Canonical load evaluation is physical
three-dimensional: the adapter embeds `(r, z)` as `(r, z, 0)` and supplies
three-component normals and the full deformation, including hoop stretch.
Therefore axisymmetric canonical traction vectors and field outputs must have
three components; only their meridional force components transfer to the grid.
This preserves the canonical three-dimensional pressure/volume semantics rather
than substituting a planar pressure law. The adapter never inserts a hidden
`2*pi*r` factor. Multiply a meridional line quadrature weight by the reference
circumference during preparation, not again during force assembly.

### Evaluation, ledgers, and atomic state

`MPMBoundaryState(position, deformation_gradient)` records evolving boundary
geometry. `initialize_state()` supplies the reference geometry and
`build_routes(state)` constructs its grid support. Standalone
`evaluate(state, routes, grid_velocity, step_size, mechanical_load_state,
args=None)` accepts flattened nodal velocities `(nodes, spatial_dimension)` and
returns `MPMBoundaryLoadResult` with `force`, `impulse`, `work`, `dissipation`,
`potential`, `candidate_state`, `successful`, and the canonical `evaluation` and
`measure` evidence. With zero step size it evaluates the supplied
geometry; with a positive step it evaluates the trial geometry from the gathered
velocity and velocity gradient. Its work uses the velocity supplied to that
evaluation. External traction contributes zero dashpot dissipation.

The explicit dynamics evaluates traction on the **committed pre-step geometry**.
Its ledger records nodal `impulse = dt * force`; summing nodes gives the total
applied impulse. Work uses the midpoint velocity of the force substep:
`work = dt * sum_nodes(force dot (v_before + v_forced)/2)`.
Prescribed-velocity/contact corrections retain their separate work ledgers.
Boundary positions and deformation gradients advance separately using the
actual accepted nodal velocity. Implicit backward Euler instead evaluates the
trial boundary geometry and endpoint velocity inside the pure residual.
Its JVP includes boundary motion, Nanson measure/normal changes, canonical
follower-law dependence, and dashpot velocity dependence; no frozen-force
approximation replaces these terms.

`MPMRuntimeState.boundary_states` is initialized automatically and committed
atomically with particle/material/runtime state. An unsuccessful attempt retains
the committed states. Explicit and implicit diagnostics expose separate
`boundary_loads` and `dashpots` tuples with force, impulse, work, dissipation, and
success evidence. Candidate geometry is not permission to commit a failed step.
Omitting both tuples preserves the no-load calculations.

### Lysmer–Kuhlemeyer dashpot

Construct `LysmerKuhlemeyerDashpot(nodal_area_vectors, density,
compressional_speed, shear_speed, /, *, field_slot=0)`, with oriented nodal
physical areas shaped as `(grid_shape..., spatial_dimension)`. For each nonzero
area `A`, normal `n`, and velocity `v`, its nodal force is

```text
f = -A rho [c_p (v dot n) n + c_s (v - (v dot n) n)]
impulse = dt sum_nodes(f)
work = dt sum_nodes(f dot v)
dissipation = -work
```

Zero-area nodes have no absorber. This is a stateless impedance boundary:
**there is no spring**, displacement-restoring force, or stored absorber energy.
The equation follows [Lysmer and Kuhlemeyer (1969), “Finite Dynamic Model for
Infinite Media,” Journal of the Engineering Mechanics Division 95(4):859–877,
doi:10.1061/JMCEA3.0001144](https://doi.org/10.1061/JMCEA3.0001144).
The matched normal/shear impedance is not a claim of exact absorption for every
incidence angle or finite-grid wave.

Standalone `evaluate(grid_velocity, dt, /, *, logical_node_ids=None,
node_valid=None)` returns
`MPMDashpotResult(force, impulse, work, dissipation, successful)` at the supplied
velocity. Indexed evaluation uses logical node identities for compact storage.
The explicit step solves the nodal midpoint impedance equation with the native
small-solve substrate and evaluates damping at `(v_before + v_after)/2`; its
dashpot work is nonpositive. The implicit residual uses endpoint damping.
Neither route adds a spring.

Run `python -m examples.material_point_boundary_loads` for a small compiled
traction-plus-dashpot step that prints the measured impulse, work, dissipation,
and acceptance result. It is a runtime example, not a wave-reflection
qualification.

## Volumetric stabilization

`MPMVolumetricStabilizationKind` is the closed selector `"none"`, `"b_bar"`,
or `"f_bar"`. Construct `MPMVolumetricStabilizationPlan(kind="none", /, *,
f_bar_blend=1.0)` and pass it as `ExplicitMPMMethodPlan(volumetric_stabilization=...)`.
Omitting the plan and explicitly selecting `"none"` preserve the same numerical
operations and bitwise results. The blend is finite and lies in `[0, 1]`.

### Cell-averaged B-bar

For the background cell containing particle `p`, the same assignment functions
define the averaged gradient and modified velocity gradient:

```text
Gbar_ip = (1 / V_cell) integral_cell grad(N_i) dV
L_p = sum_i v_i outer grad(N_ip)
Lbar_p = L_p + [sum_i v_i dot Gbar_ip - trace(L_p)] I / d
```

Internal force is the **exact transpose** of this modified operator:

```text
tau_p = P_p F_p^T
f_i = -sum_p V0_p [
    tau_p grad(N_ip) + trace(tau_p) (Gbar_ip - grad(N_ip)) / d
]
```

Thus `sum_i v_i dot f_i = -sum_p V0_p tau_p : Lbar_p`, including
arbitrary nonsymmetric stress. Ordinary and averaged gradients share one canonical
route per background node; Gauss-point contributions and periodic node aliases are
combined before transfer. A route may have zero interpolation weight but nonzero
averaged gradient, contributing internal-force virtual work without mass or momentum.
Two-point tensor Gauss quadrature integrates linear and cubic cardinal assignments;
quadratic cardinal splines use composite quadrature split at their half-cell knot.
Particle-independent linear/B-spline averages are prepared once as factorized
per-axis cell tables, then gathered by particle cell. This also preserves virtual
block-sparse grids without materializing a dense table over the logical grid.
For translated GIMP/CPDI particle domains, fixed composite quadrature approximates
the cell integral while preserving the exact discrete force/kinematic adjoint.

This is a background-cell adaptation of Hughes' mean-dilatation/B-bar construction,
not an implementation of Coombs' cell F-bar formulation:
[Hughes (1980), IJNME 15:1413–1418](https://doi.org/10.1002/nme.1620150914).
For the separate cell-F-bar MPM approach and locking discussion, see
[Coombs et al. (2018), CMAME 333:1–21](https://doi.org/10.1016/j.cma.2018.01.010).

### Incremental F-bar

The assumed material deformation `Fbar` and physical deformation `Fphysical`
are distinct. `MPMParticleState.deformation_gradient` stores `Fbar`;
`physical_deformation_gradient` stores the raw physical update for `"f_bar"` and
is `None` for the other modes. The published Zhao–Jiang–Choo projection uses
**current physical volume**, not particle mass:

```text
DeltaF_p = I + dt L_p
q_p = det(Fbar_p^n) det(DeltaF_p)
V_p^n = V0_p det(Fphysical_p^n)
q_i = sum_p N_ip V_p^n q_p / sum_p N_ip V_p^n
qbar_p = sum_i N_ip q_i
qhat_p = (1 - eta) q_p + eta qbar_p
DeltaFbar_p = (qhat_p / q_p)^(1/d) DeltaF_p
Fbar_p^(n+1) = DeltaFbar_p Fbar_p^n
Fphysical_p^(n+1) = DeltaF_p Fphysical_p^n
```

Here `eta = f_bar_blend`. At `eta=0`, the assumed state takes the unmodified
increment; at `eta=1`, the total assumed trial determinant is projected. This
implements [Zhao, Jiang, and Choo, arXiv:2209.02466v3, Eq. (27) and
Algorithm 1](https://arxiv.org/html/2209.02466v3), with an optional blend.
A mass-weighted projection of `det(DeltaF)` alone is **not** this algorithm.
The internal force retains original assignment gradients, including implicit
residuals, and uses physical volume times Cauchy stress:

```text
f_i = -sum_p V0_p [det(Fphysical_p) / det(Fbar_p)] [P_p Fbar_p^T] grad(N_ip)
```

This volume ratio matters when assumed and physical determinants differ.
Evolving GIMP/CPDI assignment shapes use the physical deformation.

### Implicit residual and route sensitivities

`ImplicitMPMMethodPlan(..., *, volumetric_stabilization=None)` selects the
same plan vocabulary. When the implicit method is omitted,
`PreparedImplicitMPMDynamics` inherits the explicit dynamics' stabilization.
An explicitly supplied implicit method must match the **complete plan identity**
(including the F-bar blend); otherwise preparation raises `ValueError`.

`PreparedImplicitMPMDynamics.prepare_residual(state, dt, arguments)` returns
`PreparedImplicitMPMResidual`: `initial_velocity` supplies the nodal initial
iterate and `evaluate(grid_velocity)` returns `(residual, valid)` without
committing state. Its topology is frozen for the attempt. Newton uses that same
pure residual, so its JVP differentiates the full F-bar projection or B-bar
adjoint, not an unstabilized approximation.

For geometry sensitivities, construct `MPMRouteSupersetPlan(prepared, *,
dynamics=owner, minimum_margin=...)` with the owning dynamics to include its
stabilization. A B-bar route linearization accepts four cotangents in order:
weights, ordinary gradients, offsets, and cell-average gradients. It reports
`primal_cell_average_gradients` and `cell_average_gradient_jvp`; those optional
fields are absent for ordinary routes, whose cotangent contract remains the
ordinary three-tuple. The admitted fixed support and minimum-margin contract
still governs differentiation across particle motion.

### Support and failure boundaries

Local structured Cartesian routes support dense and fixed-capacity block-sparse
storage, and multiple velocity fields; F-bar projects separately within each
field. Unsupported/zero-volume nodes are masked before division. B-bar quadrature
support extending beyond a rejecting grid is rejected, not truncated or
renormalized. Domain-based assignments retain their translated current shape
during cell quadrature; their integral is approximate, unlike the admitted
polynomial cardinal cases. Distributed halo and AMR restriction projection are
not implemented: distributed/AMR plans explicitly refuse non-`"none"` stabilization.

Neither treatment smooths pressure, changes material parameters, nor repairs an
inverted physical increment. Existing material, support, Jacobian, transfer, and
transactional acceptance evidence still governs whether a step commits.

## Problem declaration and compilation

`AbstractMPMConstitutivePlan` and `AbstractImplicitMPMConstitutivePlan` are neutral
`MODEL`-authority slots: built-in analytic plans are FIXED, wrapper plans hold
their base plan as a child, and a learned plan keeps its model trainable.

::: phydrax.equations.AbstractMPMConstitutivePlan

---

::: phydrax.equations.AbstractImplicitMPMConstitutivePlan

---

::: phydrax.equations.MPMConstitutiveResponse

---

::: phydrax.equations.MPMConstitutiveCapabilities

---

::: phydrax.equations.MPMLinearizedConstitutiveResponse

---

::: phydrax.nonlinear.LocalRootPlan

---

::: phydrax.equations.MaterialPointArguments

---

::: phydrax.equations.MaterialPointProblemIR

---

::: phydrax.equations.CompiledMaterialPointProblem

---

::: phydrax.equations.compile_material_point_problem

## Fixed-temporal rollout

::: phydrax.solver.MPMReplayPolicy

---

::: phydrax.solver.ScheduledMPMRolloutPlan

---

::: phydrax.solver.MPMRolloutResult

---

::: phydrax.solver.MPMGradientReport

## Adaptive, implicit, and fracture solvers

::: phydrax.solver.MPMAdaptivePolicy

---

::: phydrax.solver.AdaptiveMPMRolloutPlan

---

::: phydrax.solver.ImplicitMPMMethodPlan

---

::: phydrax.solver.PreparedImplicitMPMDynamics

---

::: phydrax.solver.MPMPhaseFieldFracturePlan

---

::: phydrax.solver.PreparedMPMPhaseFieldDynamics

## Baseline solid material

::: phydrax.applications.solid_mechanics.NeoHookeanParameters

---

::: phydrax.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.PlaneStressMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.FiniteStrainJ2Parameters

---

::: phydrax.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan

---

J2 and Drucker–Prager store seven history entries: the plastic metric packed as
`(11, 22, 33, 23, 13, 12)` and equivalent plastic strain. Off-diagonals have no
Mandel scaling. `initialize_state` initializes the metric to identity, not zeros.
Their `initialize_state_from_stress` methods return explicit prestress admissibility
and yield evidence. Stress is tension-positive and `p = -trace(tau)/3` is
compression-positive; Drucker–Prager yield is `q - M_phi p - k`.

::: phydrax.applications.solid_mechanics.PrestressedMaterialState

---

::: phydrax.applications.solid_mechanics.FiniteStrainConstitutiveModel

---

::: phydrax.applications.solid_mechanics.MPMPhaseFieldParameters

---

::: phydrax.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan

## Commercial claims, execution, and derivatives

::: phydrax.discretization.mpm.MPMIntendedUse

---

::: phydrax.discretization.mpm.MPMClaimTuple

---

::: phydrax.discretization.mpm.MPMSupportMatrix

---

::: phydrax.discretization.mpm.MPMExecutionPlan

---

::: phydrax.discretization.mpm.MPMGradientResult

---

::: phydrax.discretization.mpm.KWayMPMContactPlan

---

::: phydrax.discretization.mpm.MPMDistributedPlan

---

::: phydrax.discretization.mpm.MPMParticleLifecyclePlan

---

::: phydrax.discretization.mpm.MPMAMRPlan

## Commercial runtime

::: phydrax.solver.MPMCheckpointPlan

---

::: phydrax.solver.MPMOutputPlan

---

::: phydrax.solver.MPMRunSupervisor

---

::: phydrax.solver.MPMImplicitTopologyPlan

---

::: phydrax.solver.MPMCompactImplicitOperator

## Commercial materials and coupled fields

::: phydrax.applications.solid_mechanics.GeneralPlaneStressMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan

---

### Mohr–Coulomb

The finite-strain multiplicative Hencky material uses
`MohrCoulombParameters(G, K, c, phi, psi, H=0)`, angles in radians,
and cohesion hardening `c + H alpha`. History has seven entries:
`Cp_inverse(11,22,33,23,13,12), alpha`; `alpha` accumulates plane multipliers.
`initialize_state` gives an identity metric. The prestress initializer returns
`PrestressedMaterialState` and must be checked for admissibility.
The ordered closed-form return reports `0` elastic, `1` face, `2` edge,
`3` apex, `-1` failure, with `edge_side=-1/+1` for left/right edges.
`evaluate_linearized` reports tangent validity separately, excluding branch
boundaries. All six yield planes and ordered plastic increments are diagnostics.
See the [Mohr–Coulomb guide](../../guides_mpm_commercial_geomechanics.md#mohrcoulomb-ordered-return).

::: phydrax.applications.solid_mechanics.MohrCoulombParameters

::: phydrax.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan

---

### Modified Cam-Clay

The canonical material is the Borja–Tamagnini finite-strain hyperelastic MCC
model ([DOI: 10.1016/S0045-7825(97)00141-2](https://doi.org/10.1016/S0045-7825(97)00141-2)).
Use `ModifiedCamClayParameters(mu0, alpha0, M, lambda_raw, kappa_raw)` and the
no-reference-fields `ModifiedCamClayMPMConstitutivePlan()`. Raw compression and
swelling slopes are normalized by fixed `1 + e0`. `mu0` is additive; the
zero-shear initial modulus is `mu0 + alpha0 p0`.

With `a = trace(epsilon_e)`, `b = sqrt(2/3 * dev(epsilon_e):dev(epsilon_e))`,
`R = kappa_raw/(1 + e0)`, and `B = p0 exp(-a/R)`, stored energy is exactly
`psi = B R + (3/2) (mu0 + alpha0 B) b^2` (`has_free_energy=True`).
The elastic reference volume strain is fixed at zero. Width-nine history packs
`Cp_inverse` as `(11, 22, 33, 23, 13, 12)`, then `pc, p0, e0`.

`initialize_state` provides only an identity metric and zero reference scalars,
an inadmissible placeholder. Required initialization is
`initialize_state_from_stress(tau, parameters, preconsolidation_pressure=pc,
initial_void_ratio=e0)`, accepting batched 3×3 effective Kirchhoff stresses and
returning `PrestressedMaterialState`. It sets `p0` from initial mean pressure,
not from `pc`; check `admissible` before committing `state`.
The material refuses `p <= 0` without pressure floors.

`evaluate` and `evaluate_linearized` use the standard MPM positional arguments.
The latter differentiates the full discrete first-Piola update by AD, composing
the three-unknown `VectorLocalRootPlan` implicit derivative with spectral and
finite-strain kinematics. Diagnostics include `p`/`mean_pressure`,
`q`/`equivalent_stress`, `preconsolidation_pressure`, `reference_pressure`,
`initial_void_ratio`, `elastic_volume_strain`, `elastic_shear_strain`,
`yield_residual`, `plastic_multiplier`, `root_successful`, and
`tangent_successful`. See the [commercial material guide](../../guides_mpm_commercial_geomechanics.md#modified-cam-clay-energy-and-initialization)
and [constitutive extension guide](../../guides_mpm_constitutive_extensions.md#finite-strain-modified-cam-clay).

::: phydrax.applications.solid_mechanics.ModifiedCamClayParameters

---

::: phydrax.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan

---

::: phydrax.applications.solid_mechanics.PreparedMPMCoupledFieldOperator
