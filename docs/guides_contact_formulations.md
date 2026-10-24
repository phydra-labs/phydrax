# Contact formulations and closure

Phydrax separates contact geometry, local interface kinematics, constitutive
closure, mathematical enforcement, and nonlinear solution. These layers are not
interchangeable contact modes.

## Participants and kinematics

`AbstractContactParticipant` maps one mechanics state to world-space collision
vertices and maps contact forces back to generalized forces. Linear finite
element, rod, shell, static, rigid, articulated, and meshfree participants can
share one `ContactParticipantScene` while retaining independent state spaces.

These are **geometry** participants, not mechanics residual owners.
`LinearContactParticipant(PreparedCollisionSurface)` only supplies positions,
tangents, and transpose effort. `AbstractImplicitContactParticipant` separately
supplies physical residual/JVP, trial history, optional consistent potential,
and a domain step limit. A `MonolithicContactPlan` binds scene surfaces to
mechanics-block indices or explicit fixed wall states. Several surfaces may
share one block. Its single residual retains contact cross blocks and nonlinear
geometry-curvature terms; it does not exchange forces between private solves.
See [Monolithic implicit contact](guides_monolithic_contact.md) for the potential
gate, nonsymmetric GMRES route, frozen envelope, and atomic transaction.

`ContactKinematicsEpoch` lowers candidate stencils to law-independent local
quantities: gap, relative normal velocity, tangential velocity/slip, normal and
tangent frame, quadrature weight, material pair, stable route key, and branch
margins.

## Contact material pairs

`ContactMaterialPairTable` stores explicit pair parameters. No hidden material
mixing rule or fallback is applied. Each pair declares normal stiffness,
static/dynamic friction, restitution, adhesion energy, thermal/electrical
conductance, wear coefficient, hardness, roughness, and availability masks.

## Closure components

`ContactClosurePlan` composes:

- one `AbstractNormalContactLaw`;
- one `AbstractTangentialContactLaw`;
- one `AbstractInterfaceEvolutionLaw`;
- one `AbstractContactTransportLaw`.

The closure returns local tractions, potential density, dissipation, consistent
normal tangent, candidate history, and transport fluxes. A formulation checks
closure capabilities before use.

Available normal laws include physical clamped barriers, prefiltered geometric
inverse-power barriers, compliant Hunt-Crossley-style response, and cohesive
adhesive barriers. Tangential laws include frictionless, regularized
static/dynamic Coulomb, anisotropic elliptic Coulomb, and rate-state friction.
`FrictionWearEvolutionLaw` advances slip, adhesion damage, wear, state-variable,
and film-thickness histories transactionally.

## Smooth primal formulation

`assemble_smooth_contact` converts a potential-capable closure to balanced
surface forces. The stencil coefficients are the exact local Jacobian transpose,
so participant pullbacks preserve force and moment. Smooth formulations are
appropriate for incremental-potential equilibrium and dynamics.

## Cone contact and impact

`ContactConeProgram` represents normal and tangential impulses in local contact
coordinates over a native response operator `W`, an `AbstractLinearOperator`
from the contact impulse dual to the contact velocity space. Dense and local
block responses are explicit operators (`dense_contact_response`,
`local_block_contact_response`); articulated and composite rod responses are the
matrix-free composition `G M⁻¹ G*`. No dense Delassus matrix is formed on the
ordinary path.

A fixed-width `ContactBlockLayout` mixes Coulomb contact blocks with normal-only
blocks (for example coupled joint stops) in one solve. Padded components of a
normal-only block carry the explicit equation `p = 0`, and block keys keep
typed ownership (`ContactBlockIdentity`). The free velocity is the physical
contact velocity; restitution targets and position-level gap rates enter only
the separate normal law bias, so the result reports the physical post-impulse
velocity `W p + v` separately from the law velocity `W p + C p + v + b e_n`.

`ContactCertificatePolicy` declares unit families with a timestep-independent
reference velocity (used for the congruent scaling `S W S`), impulse,
law-velocity and slip-speed tolerances, plus complementarity-work and
tangential-work energy tolerances. Every certificate is evaluated in physical
units: impulse sign and Coulomb cone in impulse units, Signorini feasibility in
law-velocity units, `|p_n w_n|` and `p_t · v_t` in energy units, and the
sticking/sliding law with its separate slip-speed tolerance. Solver
tolerances are iteration controls only; their single-precision floor is
sixteen units in the last place of the scaled law-velocity magnitude, and the
selected-friction stage continues iterating within its budget until the
physical certificate holds.

`solve_contact_cone` iterates projected gradients whose step is accepted only
when the quadratic descent inequality holds on the actual iterate (bounded
backtracking); no power-iteration spectral estimate is used. Exhausting the
step budget is a failure. `ContactConeNumericRevision` snapshots the response
operator's array leaves, the response owner's binding (for example a prepared
articulated mass), the law data, layout, certificate tolerances and solver
controls; any change invalidates a stored result, and failed solves apply zero
impulse while retaining the candidate for diagnosis.

`assemble_contact_impulses` maps local impulses to participant surface impulses.
`RollingSpinningResistancePlan` adds bounded rolling and spinning impulses with
an explicit dissipation certificate.

## Mortar and Nitsche

`ContactInterfacePlan` is a fixed quadrature map between nonmatching plus/minus
nodal traces. Its affine trace maps and transpose traction assembly conserve
interface action exactly.

- `MortarContactPlan` updates normal and tangential multipliers through an
  augmented-Lagrangian projection.
- `OneSidedNitscheContactPlan` uses one consistency stress trace.
- `UnbiasedNitscheContactPlan` averages both traces and is invariant to the
  declared participant order when the traces are exchanged.
- `MeshTiePlan` enforces vector continuity and can release routes beyond a
  declared tension limit.

## Guarantee semantics

Every collision backend declares one ordered guarantee:

```text
UNAVAILABLE
HEURISTIC
PRACTICAL_CONSERVATIVE
ENCLOSURE_CONSERVATIVE
ANALYTIC_CONSERVATIVE
ROUNDING_CERTIFIED
```

The ordinary inclusion backend provides an enclosure-conservative guarantee.
`CertifiedAABBCCDPlan` uses outward-rounded swept primitive boxes and reports
unresolved intervals at their lower endpoint, yielding a deliberately
conservative certified backend. A requested guarantee stronger than a backend
can provide rejects during preparation.

## Declared trajectories and path admissibility

A contact trajectory owns its path over normalized time `s ∈ [0, 1]` and
declares its own guarantee; callers cannot upgrade it. `enclosure(lo, hi)`
returns a `ContactPathEnclosure` (per-vertex boxes, interval, guarantee,
dynamic `revision`, static `trajectory_id`) and `midpoint_bound(lo, hi)`
returns midpoint positions with per-vertex Hausdorff displacement bounds.
Search consumes the enclosure as its path coverage, so the epoch records the
trajectory identity, revision and interval it covers.

- `LinearContactTrajectory` and `CubicHermiteContactTrajectory` are analytic;
  cubic subinterval bounds use the exact Bernstein control polygon restricted to
  the subinterval, so they contract with the interval without sampling.
- `RigidSweepContactTrajectory` takes an explicit rotation increment: the path
  is `exp(s[w]) R0`. Endpoint rotations alone do not define full turns.
- `ArticulatedContactTrajectory` binds body-attached spheres and capsules to
  native forward kinematics of the lifted path `q(s) = q0 + s h v`. Its bounds
  intersect an interval forward-kinematics box (sine/cosine enclosures of every
  ancestor hinge, including interior extrema and full turns) with a midpoint
  ball bounded by the Jacobian-column speed. Its guarantee is
  `ENCLOSURE_CONSERVATIVE`: the floating-point allowance is modeled, not
  directed-rounding certified.

`collision_free_step_limit` keeps the strict-separation contract of barrier
consumers and refuses initially touching states. Its enclosure predicate uses
the larger of box separation and box-center distance minus box half-diagonals,
which certifies oblique separated features whose boxes overlap.

Hard contact uses `certify_admissible_path(AdmissiblePathPlan(gap_floor), scene,
epoch, trajectory)`. It proves `g(s) >= -gap_floor` for every candidate pair
over the whole interval, including legal initial resting or sliding contact,
from a directed separating-axis lower bound of the midpoint core distance minus
both features' displacement bounds (or the directed box/plane support bound)
minus physical radii. The separating-axis bound is valid for any trial axis, so
an ill-conditioned closest-point solve (for example determinant cancellation of
nearly parallel crossing segments) can only weaken it; violations are proved
with actual witness points. Intervals are processed in time order with bounded
work. The result distinguishes `ADMISSIBLE`, `INITIALLY_INADMISSIBLE`,
`CERTIFIED_PREFIX` (a later violation was found; `certified_prefix` marks the
certified start), `UNRESOLVED_INTERVAL`, `INVALID_GEOMETRY` (including a
non-unit or nonfinite plane normal anywhere in the scene), `WORK_LIMIT`,
`SEARCH_INCOMPLETE` and `COVERAGE_MISMATCH` (the epoch was searched for a
different trajectory, revision, interval, contact extents or planes, or does not
cover the trajectory's full-path enclosure). An epoch from a foreign catalogue,
topology or routing policy is rejected outright. A certified prefix is never a
successful full step. `collision_free_step_limit` applies the same scene binding
and plane validation, reporting `INVALID_GEOMETRY` and `SEARCH_INCOMPLETE`.

Analytic primitive witnesses: capsule-plane pairs emit both endpoint gaps as
separate witness slots; parallel capsule axes with overlapping axial projection
emit the two extremal closest-point pairs along the left capsule's
first-to-second axis (`segment_segment_support_distance`); all other pairs use
the unique closest feature. Support rows report zero feature margin, so their
derivatives are not admitted while the primal support remains valid.
`offset_surface_witnesses` converts core witnesses into the physical gap and the
surface points where tangential friction acts. Coincident capsule axes and
touching sphere/capsule cores have no defined normal and refuse rather than
acquire one. Plane gaps are affine with a fixed normal, so penetrating trial
configurations keep their signed gap and Jacobian; refusing an initial overlap
is the job of the path certificate, not the distance kernel.

## State ownership

`ContactRouteState` is keyed by physical routes and stores stick/slip mode,
accumulated slip, adhesion damage, wear, rate-state variable, and film thickness.
Route remapping, proxy refinement, repartition, and remeshing produce candidate
state with explicit transition evidence. Only an accepted physical solve commits
candidate interface state.
