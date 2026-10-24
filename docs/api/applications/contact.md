# Contact applications

## Closure and materials

::: phydrax.applications.contact.ContactMaterialPairTable

---

::: phydrax.applications.contact.ContactClosurePlan

---

::: phydrax.applications.contact.ContactRouteState

Route remapping uses stable key ordering and binary-search ranges with linear workspace. The first valid old slot supplies inherited state when old keys repeat; duplicate old matches and duplicate valid new keys retain explicit refusal flags. Candidate ordering, defaults for invalid/unmatched slots, lifecycle counts, commit, and rollback are unchanged.

---

::: phydrax.applications.contact.BarrierNormalContactLaw

---

::: phydrax.applications.contact.GeometricContactNormalLaw

---

::: phydrax.applications.contact.CompliantNormalContactLaw

---

::: phydrax.applications.contact.AdhesiveBarrierNormalLaw

---

::: phydrax.applications.contact.RegularizedCoulombContactLaw

---

::: phydrax.applications.contact.AnisotropicCoulombContactLaw

---

::: phydrax.applications.contact.RateStateFrictionContactLaw

---

::: phydrax.applications.contact.FrictionWearEvolutionLaw

## Operator-valued Signorini--Coulomb cone

`ContactConeProgram(response, free_velocity, compliance, dynamic_friction,
route_keys, valid, layout, certificate, program_id, ...)` owns one fixed-route
Signorini--Coulomb program over a native linear response operator `W` mapping
the contact dual (impulses) to contact velocity. It stores no dense
effective-mass array: the former `effective_mass` fields and the
`(tangent_free_velocity, tangent_effective_mass)` JVP arguments are removed.
Explicit dense or block-diagonal responses are supplied as operators through
`dense_contact_response(matrix, contact_cone_space(count, width, dtype))` and
`local_block_contact_response(blocks)`; a dense authority is never an implicit
fallback. Actions, recertification and revision matching use operator actions
only. Restitution and compliance remain available to the fixed-route APIs.

`ContactBlockLayout` is a fixed-width block layout prepared on the host. Each
block declares a `ContactBlockLaw` (`COULOMB` or `NORMAL_ONLY`), a typed
`ContactBlockIdentity` (`CONTACT_ROUTE` or `JOINT_STOP`; keys of different
owners never compare), an active-component mask and a physical unit family.
`ContactBlockLayout.mixed(contacts, stop_unit_families, family_count=...)`
places Coulomb contact blocks before normal-only stop blocks with width three.
A normal-only stop is not a Coulomb block with zero friction: padded components
satisfy `p_component = 0` and `project_contact_blocks` zeroes them.

`ContactCertificatePolicy` declares per-family reference velocities and
impulse, law-velocity and slip-speed tolerances plus complementarity-work and
tangential-work tolerances; all are positive and dimensional. The physical
velocity is `W p + v_free`; the law velocity adds compliance and the normal
bias. `certify_contact_cone_impulse` evaluates impulse sign, law-normal
feasibility, energy-form complementarity `|p_n g/h|`, the Coulomb cone,
sticking/sliding laws, padding, tangential work, numeric validity and material
completeness against current actions. The iteration tolerance of a solver plan
is an iteration control, not a physical certificate.
`classify_contact_cone_branches` labels each valid block `SEPARATING`,
`STICKING`, `SLIDING`, `NORMAL_ACTIVE` (a normal-only block such as a joint stop
in contact) or `FRICTIONLESS_BOUNDARY` (a touching Coulomb block with zero
friction coefficients) with dimensionless margins in tolerance units; boundary
and apex cases have zero margin and invalid blocks are `UNCLASSIFIED`.

`ContactConeResult.impulse` is the accepted impulse and is exactly zero when
convergence, the step budget or any certificate fails; `candidate_impulse`,
`candidate_post_relative_velocity` and `candidate_contact_law_velocity` keep the
proposal for diagnosis. `post_relative_velocity` is physical and
`contact_law_velocity` is the biased/compliant law velocity.
`ContactConeNumericRevision` snapshots the response operator's array leaves, the
owner binding, law data, layout, certificate and solver parameters, so changed
geometry, inertia, friction, compliance, route mask/order or solver parameters
invalidate a stale result.

::: phydrax.applications.contact.ContactConeProgram

---

::: phydrax.applications.contact.ContactConeSolverPlan

---

::: phydrax.applications.contact.ContactBlockLayout

---

::: phydrax.applications.contact.ContactBlockLaw

---

::: phydrax.applications.contact.ContactBlockIdentity

---

::: phydrax.applications.contact.ContactCertificatePolicy

---

::: phydrax.applications.contact.ContactConeCertificate

---

::: phydrax.applications.contact.ContactConeBranch

---

::: phydrax.applications.contact.certify_contact_cone_impulse

---

::: phydrax.applications.contact.classify_contact_cone_branches

---

::: phydrax.applications.contact.project_contact_blocks

---

::: phydrax.applications.contact.contact_cone_space

---

::: phydrax.applications.contact.dense_contact_response

---

::: phydrax.applications.contact.local_block_contact_response

## Formulations

::: phydrax.applications.contact.ConvergentContactPotentialPlan

---

::: phydrax.applications.contact.MortarContactPlan

---

::: phydrax.applications.contact.OneSidedNitscheContactPlan

---

::: phydrax.applications.contact.UnbiasedNitscheContactPlan

---

::: phydrax.applications.contact.MeshTiePlan

## Patch and multiphysics contact

::: phydrax.applications.contact.HydroelasticMaterialPlan

---

::: phydrax.applications.contact.HomogenizedRoughContactPlan

---

::: phydrax.applications.contact.PeriodicRoughContactPlan

---

::: phydrax.applications.contact.CoupledContactTransportLaw

---

::: phydrax.applications.contact.LubricationContactPlan

---

::: phydrax.applications.contact.ReynoldsFilmPlan

---

::: phydrax.applications.contact.ReynoldsPressureBoundaryConditions

## Solvers, coupling, and derivatives

::: phydrax.applications.contact.ContactSolvePolicy

---

::: phydrax.applications.contact.MonolithicContactPlan

---

::: phydrax.applications.contact.CrossDiscretizationContactResult

---

::: phydrax.applications.contact.ContactStateTransferPlan

---

::: phydrax.applications.contact.ContactBlockPreconditionerPlan

---

::: phydrax.applications.contact.contact_closure_gap_jvp

---

::: phydrax.applications.contact.contact_cone_solution_jvp

---

::: phydrax.applications.contact.contact_cone_solution_vjp

---

::: phydrax.applications.contact.prepare_contact_cone_sensitivity

---

::: phydrax.applications.contact.PreparedContactConeSensitivity

---

::: phydrax.applications.contact.ContactConeTangent

---

::: phydrax.applications.contact.ContactConeCotangent

---

::: phydrax.applications.contact.ContactConeVJP

---

::: phydrax.applications.contact.ContactConeFixedBranch

---

::: phydrax.applications.contact.fix_contact_cone_branch

---

::: phydrax.applications.contact.contact_cone_fixed_branch_residual

---

::: phydrax.applications.contact.contact_cone_fixed_branch_admitted

Cone derivatives are admitted only on a certified, provably regular fixed
branch. `prepare_contact_cone_sensitivity(program, result, regularity=...)`
recertifies the result, classifies every valid block and prepares one
matrix-free fixed-branch residual linearization, solved with restarted GMRES
because the implicit Jacobian can be nonsymmetric; the JVP and its transpose VJP
share it. Regularity is proved, not inferred from a successful linear solve:
`regularity` is an explicit bounded `phydrax.linalg.svd.SVDSolvePolicy` under
which `phydrax.linalg.certify_rectangular_rank` must prove full, locally fixed
rank of the fixed-branch Jacobian. Without that policy, or when the certificate
does not prove it (for example sticking blocks whose tangential response rows
vanish, or duplicated support witnesses), every derivative is refused.
`PreparedContactConeSensitivity` keeps its `fixed_branch`, the certificate
(`regularity`), `jacobian_regular` and `regularity_margin`, and is bound to the
exact numeric revision and impulse of the result it linearizes; a stale
preparation yields refused derivatives. `regularity` is fixed at preparation:
passing it again together with a prepared `sensitivity` raises `ValueError`.

The fixed-branch law is a shared helper, not private to the cone sensitivity.
`fix_contact_cone_branch(program, result)` classifies a certified result without
linearizing it and returns a `ContactConeFixedBranch`: per-block branch labels
and margins, the minimum `cone_margin`, the congruent block scale, the active
component mask, the contacting, sticking, sliding and frictionless-boundary
block masks, and the current certification of the solve.
`contact_cone_fixed_branch_residual(fixed, scaled_impulse, scaled_law_velocity,
friction)` evaluates that branch's block law in scaled variables: contacting
blocks impose zero normal law velocity, sticking blocks zero tangential law
velocity, sliding blocks the dynamic Coulomb boundary map
`p_t = mu a_n a_t / |a_t|` with `a = p - w`, and every other, padded or invalid
component `p = 0`. The caller supplies the law velocity, so the same law serves
the cone response and endpoint laws such as the articulated step's `g(q1)/h`.
`contact_cone_fixed_branch_admitted(program, fixed, margin_factor=...,
friction_derivative=..., geometry_admissible=...)` requires the current
certification, a classified branch for every valid block, geometric admission,
margins above `margin_factor` certificate tolerances and, for friction
derivatives, no frictionless Coulomb block in contact. Jacobian regularity is a
separate certificate.

Branch admission is applied on every JVP/VJP call, including reuse of a
prepared sensitivity: geometric admission (`geometry_admissible`) and a minimum
dimensionless branch margin above `margin_factor`. Tangents are structured:
`ContactConeTangent.response_action` is the action `(dW) p*`, not a matrix, and
the remaining fields perturb free velocity, normal bias, compliance and
friction. A JVP whose friction tangent touches a `FRICTIONLESS_BOUNDARY` block
is refused, because friction changes the tangential impulse one-sidedly there.
A VJP on such blocks is refused unless `friction_cotangent=False`, in which case
`ContactConeCotangent.static_friction` and `.dynamic_friction` are `None` and
`friction_computed` is false. For cone derivatives `ContactDerivativeEvidence`
additionally reports `linear_successful`, `jacobian_regular`,
`regularity_margin` and `friction_boundary_refused`; its `successful` requires a
regular Jacobian, a successful linear solve and no refused friction boundary.
Apex, stick/slip boundaries, stale or uncertified
results, irregular Jacobians, geometric inadmissibility, insufficient margin and
failed linear solves return zero derivatives with refusal evidence. Masks,
route keys and branch decisions are discrete.

---

::: phydrax.applications.contact.mortar_gap_jvp

---

::: phydrax.applications.contact.SAPContactSolverPlan

---

::: phydrax.applications.contact.SemismoothContactSolverPlan

---

::: phydrax.applications.contact.PrimalDualContactSolverPlan

---

::: phydrax.applications.contact.RigidContactParticipant

---

::: phydrax.applications.contact.prepare_point_contact_participant

---

::: phydrax.applications.contact.prepare_mpm_contact_participant

## Articulated fixed-route response

`prepare_articulated_contact` composes the true-dual response `G M⁻¹ G*`
without forming a dense Delassus matrix. A `PreparedArticulatedMass` supplies
its factor evidence and numerical binding. An explicit inverse-mass operator
instead needs `certify_inverse_mass_operator`: an existing `DenseLinearOperator`
is checked from its matrix; another operator requires the caller's bounded
`inverse_mass_materialization` policy.

The certificate retains native `DensePropertyEvidence` for symmetry and positive
semidefiniteness, including its tolerances. The original response is not repaired.
A zero-dimensional map is vacuously valid and has no dense property evidence.

::: phydrax.applications.contact.prepare_articulated_contact

::: phydrax.applications.contact.PreparedArticulatedContact

::: phydrax.applications.contact.ArticulatedContactPreparationEvidence

::: phydrax.applications.contact.certify_inverse_mass_operator

::: phydrax.applications.contact.InverseMassOperatorCertificate

## Mechanical contact history

`ContactManifoldState` is the fixed-capacity, key-sorted mechanical history of
one route catalogue: occupied/active masks, route keys, witnesses, normal and
tangent frame, warm impulses, branch labels, slip, age/retention, material and
geometry revisions and a transaction revision. It is distinct from the
closure-law `ContactRouteState` and from monolithic lagged-friction history.
`update_contact_manifold(state, observation, policy=...)` joins current
`ContactManifoldObservation` routes into history with a device sorted-key join
(no contact-by-contact matching): deterministic birth, death, resurrection and
retention, reset of material-changed routes, and tangential warm starts
transported through the world impulse into the new tangent frame
(`ContactFrameTransport.NORMAL_PARALLEL`, or `MATERIAL_FRAME` for a common
material carrier; an exactly reversed normal resets that route). Active
overflow, duplicate keys, stale geometry revisions and frame residuals make the
`ContactManifoldTransition` unsuccessful. `commit()` returns the candidate and
`rollback()` the previous state without testing `successful`; the caller
commits only after the complete requested step succeeds.
`record_contact_response` stores accepted impulses, branches and integrated slip
for keys already in history.

The rod lifecycle uses the same state: the former `RodContactManifoldState` is
removed. `empty_rod_contact_manifold`, `rod_contact_manifold_observation` and
`update_rod_contact_manifold` adapt rod witnesses with material-frame transport;
the rod transition exposes the generic transition as `.manifold`.

::: phydrax.applications.contact.ContactManifoldState

---

::: phydrax.applications.contact.ContactManifoldObservation

---

::: phydrax.applications.contact.ContactManifoldPolicy

---

::: phydrax.applications.contact.ContactFrameTransport

---

::: phydrax.applications.contact.ContactManifoldTransition

---

::: phydrax.applications.contact.update_contact_manifold

---

::: phydrax.applications.contact.record_contact_response

---

::: phydrax.applications.contact.rod_contact_manifold_observation

## Native articulated contact step

`PreparedArticulatedContactStep(articulation, coordinates, policy, ...)` owns the
coupled contact/limit step of one prepared fixed-base reduced articulation.
Its admitted profile is exactly:

- three-dimensional fixed-base trees of fixed, hinge and prismatic joints
  (`nq == nv`), with lifted coordinates from `PreparedLiftedJointCoordinates`;
- body-attached positive-radius spheres and capsules and static planes
  (`ArticulatedCollisionGeometry`) with an explicit table of unordered material
  pairs and isotropic static/dynamic Coulomb coefficients, `0 <= dynamic <= static`;
- the first-order, position-level, inelastic hard-contact law: zero
  restitution, zero compliance;
- normal-only unilateral hinge/prismatic stops (`PreparedReducedJointStops`) in
  the same generalized response, never a separate correction;
- an explicit `ArticulatedActuationMode`: `AUTONOMOUS` (no input),
  `MAPPED_MOTORS` (one command per stateless scalar motor) or
  `GENERALIZED_EFFORT` (a generalized effort covector in velocity order, N·m for
  hinges and N for prismatic joints); executed damping is added in every mode.

Other laws, geometry and actuation are refused at preparation rather than
approximated. Search uses `search_backend` (`ArticulatedContactSearchBackend`):
`"exhaustive"` (`CompiledContactSearchPlan`) or `"lbvh"` with an explicit
`maximum_traversal_visits`; route capacities and a positive history capacity
are explicit whenever legal contact routes exist.

For one segment of duration `h` from a provisional source `(q0, v0)`,
`M0 = M(q0)` is prepared once, `v_free = v0 + h a0`, and the binding lifted path
is `q(s) = q0 + s h v1` with `q1 = q0 + h v1` and `v1 = v_free + M0⁻¹ G(q1)* p`.
Each row obeys `p_n >= 0`, `g(q1)/h >= 0` and `p_n g(q1)/h = 0` within
dimensional tolerances; friction acts on the physical tangential velocity
`G_t(q1) v1`. The coupled root uses the native fixed-point iteration over
frozen-route cone programs relinearized at `q_hat` with
`b_n = (g(q_hat) - G_n(q_hat)(q_hat - q0)) / h`, and routes are rebuilt from the
actual path enclosure under a bounded outer epoch. The result is independently
recertified: kinematic update, momentum, actual gaps, cone law, material
completeness, physical work, energy ledger and full-path admissibility of the
identical trajectory. When the native smooth Euler trajectory is completely
admissible and zero impulse satisfies every law, that native result is returned
unchanged.

Unresolved segments subdivide time dyadically within
`maximum_subdivision_depth`, `maximum_segment_attempts` and
`minimum_segment_duration`. Accepted segments commit only when they cover the
entire request; otherwise `ArticulatedContactStepResult.accepted` is the
complete original source (configuration, winding, velocity, contact history,
stop impulses and branches) and `candidate` may hold a provisional prefix. A
certified prefix is never a successful step.

`ArticulatedContactSegmentEvidence` keeps the physical velocity `G(q1) v1`
separate from the normal law velocity `g(q1)/h`, and requires two cone
certificates: `response_certificate` at the response velocity
`v_free + M0⁻¹ G* p` and `cone_certificate` at the accepted endpoint `v1`.
Contact blocks precede stop
blocks. `ArticulatedContactCertificatePolicy` declares timestep-independent
contact, hinge-stop and prismatic-stop families: reference velocities, impulse
and gap tolerances, slip speed, complementarity/friction/contact work, contact
kinetic energy, ledger identity, energy surplus, momentum-energy residual and a
discretization energy budget. Per segment, impulse and work allowances are
allocated by duration fraction and only the gap tolerance becomes a law-velocity
tolerance `epsilon_gap / h`; subdivision cannot loosen impulse or work checks.
The `AdmissiblePathPlan` gap floor must equal the contact gap tolerance.

`ArticulatedContactEnergyLedger` records, in joules,
`E1 - E0 - W_s - W_contact + D_contact = r_f + R_geometry + R_work` with the
physical contact work `W_contact = p · G(q1) v1` (not the biased law work), the
inelastic correction loss `D_contact = 0.5 (v1 - v_free) · M0 (v1 - v_free)`, the
source-force work, the native predictor defect, and the geometry and work
residuals; positive parts are accumulated over accepted segments against the
request budgets. `D_contact` is a mechanical correction-loss record, not a
thermodynamic heat assignment.

| `ArticulatedContactStepStatus` | First failing stage |
| --- | --- |
| `SUCCESS` | Every segment certificate and request budget passed with full coverage |
| `INVALID_INPUT` | Invalid effort, nonfinite state/gravity/duration, or nonpositive duration |
| `FREE_DYNAMICS_FAILED` | Native smooth dynamics or the prepared mass failed |
| `SEARCH_INCOMPLETE` | Initial, root or final-coverage search incomplete |
| `NARROW_PHASE_FAILED` | A narrow-phase row failed |
| `INITIALLY_INADMISSIBLE` | Initial contact path or stop gap already violated |
| `INVALID_GEOMETRY` | Path geometry invalid |
| `WINDING_RANGE_EXCEEDED` | Lifted winding outside its admitted range |
| `HISTORY_FAILED` | Warm-start or endpoint manifold transition failed |
| `MASS_RESPONSE_FAILED` | Mass correction failed or the inverse action was not current |
| `ROOT_NOT_CONVERGED` | Coupled root did not converge |
| `ROUTE_REBUILD_EXHAUSTED` | Routes did not stabilize within the rebuild budget |
| `PATH_NOT_ADMISSIBLE` | Contact/stop full-path certificate or endpoint features failed |
| `CONE_FAILED` | Endpoint cone certificate failed |
| `MOMENTUM_INVALID` | Momentum residual energy or coordinate advance inconsistent |
| `GAP_INVALID` | An endpoint gap is below its negative tolerance |
| `CONTACT_WORK_INVALID` | Physical contact work exceeds its allocated allowance |
| `ENERGY_INVALID` | Contact energy, correction loss, ledger, predictor defect, surplus or discretization check failed |
| `REQUEST_BUDGET_EXCEEDED` | A fully covered request violated accumulated budgets |
| `SUBSTEP_BUDGET_EXHAUSTED` | Segment attempts exhausted before coverage |
| `MINIMUM_SEGMENT_REACHED` | Retry would exceed subdivision depth or minimum duration |

`ROOT_NOT_CONVERGED` through `ENERGY_INVALID` trigger bounded subdivision; they
never shrink the requested time. `ArticulatedContactRootEvidence.status` is a
nonlinear status (`-1` when the exact contact-free bypass made no attempt).
`admissibility(source)` certifies one accepted configuration as a legal resting
state for reset and refresh.

`step_jvp` and `step_vjp` share one fixed-branch linearization. The result's
`fixed_branch_candidate` is only the primal condition (one accepted segment,
classified branches with margin above `derivative_margin_factor`, no
activation, feature or winding transition); it is not derivative admission.
Each derivative call additionally compares the current step, actuation and
gravity exactly with the request's `binding` and refuses any mismatch. With
contact or stop blocks the derivative solves one full coupled fixed-branch
linearization in `(v1, p)`, not a velocity Schur complement with an inner
solve: the momentum residual `v1 - v_free - M0⁻¹ G(q1)* p` together with the
cone owner's fixed-branch law residual (`contact_cone_fixed_branch_residual`)
at the accepted endpoint (`g(q1)/h` normal, `G_t(q1) v1` tangential).
Before any solve, `ArticulatedContactStepPolicy.derivative_regularity` (an
explicit bounded `SVDSolvePolicy`) must let
`phydrax.linalg.certify_rectangular_rank` prove that coupled Jacobian
nonsingular with fixed rank; the JVP is one GMRES solve and the VJP one
transposed solve on that same certified operator. With `None`, or a refused or
over-budget certificate, no contact derivative is admitted and no solve runs.
Without contact or stop blocks the step is the explicit Euler map and needs no
certificate. The JVP/VJP results expose `fixed_branch_candidate`,
`binding_current`, `branch_admitted` (the cone owner's branch admission),
`coupled_regular`, `coupled_regularity` and `linear_successful`; refusals
return zero tangents with `successful` false. The JVP step tangent and the VJP
step cotangent are PyTrees of the prepared step's inexact leaves (geometry,
inertia, friction); `step_vjp(..., friction_cotangent=False)` leaves the
friction tables of `step_cotangent` as `None` rather than zero. Search, route
birth/death, winding, contact or stop activation and subdivision are discrete
events.

The step stores its articulation and collision geometry once. With contact
routes, `step.scene` holds the only participant. `step.participant`,
`step.articulation` and `step.geometry` are read through that participant, so a
leaf updated through `scene` reaches native prediction, the prepared mass,
route kinematics, contact trajectories and search together. Without contact
routes, `participant` and `geometry` are `None` and the step owns its sole
articulation privately. The constructor and the public read attributes are
unchanged. The prepared PyTree is not: it no longer carries separate
articulation, geometry and participant subtrees. That affects `eqx.filter`
partitions, the JVP step tangent and VJP step cotangent, and any
`PreparedArticulatedPlant` that contains the step. Select leaves through the
public attributes, e.g. `eqx.tree_at(lambda step: step.articulation, ...)`.
Array-leaf sequences, executables and checkpoints made from steps prepared
before this change do not line up with current ones. Scientific identities
(`step_id`, `program_id`, `catalogue_id`, geometry and articulation IDs) are
unchanged.

::: phydrax.applications.contact.ArticulatedActuationMode

---

::: phydrax.applications.contact.ArticulatedContactCertificatePolicy

---

::: phydrax.applications.contact.ArticulatedContactStepPolicy

---

::: phydrax.applications.contact.PreparedArticulatedContactStep

---

::: phydrax.applications.contact.ArticulatedContactSource

---

::: phydrax.applications.contact.ArticulatedContactStepResult

---

::: phydrax.applications.contact.ArticulatedContactStepBinding

---

::: phydrax.applications.contact.ArticulatedContactStepStatus

---

::: phydrax.applications.contact.ArticulatedContactSegmentEvidence

---

::: phydrax.applications.contact.ArticulatedContactRootEvidence

---

::: phydrax.applications.contact.ArticulatedContactEnergyLedger

---

::: phydrax.applications.contact.ArticulatedRouteGeometry

---

::: phydrax.applications.contact.ArticulatedContactAdmissibility

---

::: phydrax.applications.contact.ArticulatedContactStepJVP

---

::: phydrax.applications.contact.ArticulatedContactStepVJP

## Shared residual participants and affine bodies

See [Monolithic implicit contact](../../guides_monolithic_contact.md) for
equations, the potential-consistency gate, conservative envelopes, and atomic
acceptance. Geometry participants remain distinct from mechanics participants.

::: phydrax.applications.contact.AbstractImplicitContactParticipant

::: phydrax.applications.contact.ContactParticipantEvaluation

::: phydrax.applications.contact.ContactParticipantStepLimit

::: phydrax.applications.contact.ContactKinematicLinearization

::: phydrax.applications.contact.make_implicit_contact_participant

::: phydrax.applications.contact.prepare_finite_element_contact_participant

::: phydrax.applications.contact.prepare_implicit_mpm_contact_participant

::: phydrax.applications.contact.prepare_iga_contact_participant

::: phydrax.applications.contact.prepare_implicit_reduced_rod_contact_geometry

::: phydrax.applications.contact.AffineBodyPlan

::: phydrax.applications.contact.AffineBodyHistory

::: phydrax.applications.contact.AffineBodyParticipant

::: phydrax.applications.contact.prepare_affine_body_participant

## Monolithic lifecycle and local derivatives

`MonolithicContactPlan(participants, scene, contact, search, ccd,
geometry_owner_indices=..., fixed_geometry_states=..., friction=None, policy=None)`
binds an ordered tuple of native mechanics spaces to geometry. Policy fields are
`merit`, `method`, `termination`, `maximum_lag_iterations`, and `lag_tolerance`.
Coordinates/history are mechanics-block tuples; fixed walls are explicit geometry
states. Results retain the frozen candidate epoch for local sensitivity.

::: phydrax.applications.contact.MonolithicContactState

::: phydrax.applications.contact.initialize_monolithic_contact_state

::: phydrax.applications.contact.PreparedMonolithicContactStep

::: phydrax.applications.contact.prepare_monolithic_contact_step

::: phydrax.applications.contact.solve_monolithic_contact_step

::: phydrax.applications.contact.PreparedMonolithicContactEquilibrium

::: phydrax.applications.contact.prepare_monolithic_contact_equilibrium

::: phydrax.applications.contact.solve_monolithic_contact_equilibrium

::: phydrax.applications.contact.ContactSolveDiagnostics

::: phydrax.applications.contact.MonolithicContactResult

::: phydrax.applications.contact.monolithic_contact_solution_jvp

::: phydrax.applications.contact.monolithic_contact_solution_vjp
