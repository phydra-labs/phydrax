# Particle methods

## Material support and execution

`ParticleSetPlan.prepare(measure_accumulation_dtype=...)` and direct
`ParticleDiscretization` construction choose material-mass accumulation precision.
The default is float64, independently of coordinate storage; explicitly select
float32 when the complete prepared object must contain only single-precision
floating arrays. `mass_measure` is the canonical accessor. The physical weights'
measure identity is unchanged, while nondefault prepared arithmetic participates
in the prepared/container identity.

::: phydrax.discretization.ParticleSetPlan

---

::: phydrax.discretization.ParticleDiscretization

---

::: phydrax.discretization.ParticleBox

---

::: phydrax.discretization.ParticlePrecisionPolicy

---

::: phydrax.discretization.ParticleExecutionPolicy

## Neighborhoods and pair geometry

::: phydrax.discretization.AbstractParticleNeighborhoodPlan

---

::: phydrax.discretization.AbstractPreparedParticleNeighborhood

---

::: phydrax.discretization.ParticleNeighborhoodState

---

::: phydrax.discretization.DenseParticleNeighborhoodPlan

---

::: phydrax.discretization.PreparedDenseParticleNeighborhood
---

::: phydrax.discretization.CellListParticleNeighborhoodPlan

---

::: phydrax.discretization.PreparedCellListParticleNeighborhood


---

::: phydrax.discretization.ParticlePairRelation

---

::: phydrax.discretization.ParticlePairGeometry

---

::: phydrax.discretization.particle_pair_geometry

---

::: phydrax.discretization.scatter_pair_sum

---

::: phydrax.discretization.scatter_pair_exchange

---

::: phydrax.discretization.particle_graph_view

## Periodic cells and image-aware neighborhoods

`PeriodicCell` stores row lattice vectors `H`. Its nearest-image stencil serves
classical pair-once relations (`ParticlePairRelation`), which keep their
unique-image guards. `PeriodicCell.image_stencil(radius, maximum_image_count=...)`
is a separate, complete enumeration: it returns a `PeriodicImageStencil` holding
every integer translation `n` for which `d = x_receiver - x_source + n @ H` can
satisfy `|d| < radius`. It bounds the extent on each periodic axis by
`floor(fractional_excursion + radius * ||H^+[:, a]||)` using the host right
inverse, with the axis reach scaled outward by a small relative roundoff headroom
(64 float32 epsilons) for sizing only. Where that outward interval crosses an
integer the stencil holds the larger extent, so the reference lattice keeps a positive
completeness margin under the traced solve. `axis_reach` records the raw reach.
The headroom is not an error certificate for arbitrary conditioning. The image
route searches size their stencils with the same rule. Nonperiodic axes have zero
extent. The cell's condition certificate
applies, and a stencil larger than `maximum_image_count` is refused rather than
truncated. `lattice_right_inverse_with_status` and `lattice_measure` are the
traced, non-raising lattice solve and `sqrt(det(H @ H.T))` measure; both return
a status flag in place of a substitute value.

`ParticleImageRelation` is directed and image-aware. Route `e` is identified by
the stable tuple `(source id, receiver id, n, case)` and has displacement
`d_e = x[receiver] - x[source] + n_e @ H`. Only `(source == receiver, n == 0)` is
excluded. Every nonzero self image and every repeated image of a pair is a
distinct route. `reversed()` maps `(source, receiver, n)` to
`(receiver, source, -n)`. `with_representation_offsets` updates `n` by the exact
image-count difference when positions are rewrapped, without rebuilding or
re-sorting the relation. Shifts, wrap counts and offsets live in the symmetric
int32 range `|n| <= 2**31 - 1`; wider integers are refused before storage, never
clipped or wrapped. This relation is not pair-once. Classical pair terms do not
consume it.

`CellListParticleImageNeighborhoodPlan` is the scalable fractional cell-list
search. It is not limited by the unique-image radius, and it charges
`maximum_candidate_slots` from the scalar stencil size before enumerating any
offsets. `DenseParticleImageNeighborhoodPlan` is a bounded, named dense
reference for validation and small systems, guarded by `maximum_dense_routes`.
`ParticleImageCapacity` charges cell occupancy, stored edges, receiver degree,
and image count separately. `ParticleImageRelationEvidence` keeps capacity
failures (cell, image, edge, degree) separate from scientific failures (stencil
envelope, domain, nonfinite, representation overflow). A singular or non-finite
runtime cell is a case-level `nonfinite` failure even when no particle is
active. `representation_overflow` reports an active wrap count or route shift
outside the int32 image range, for example coordinates billions of cells from
the origin; translated systems inside that range keep their exact routes.
`ParticleImageCapacityLadder.select` advances to the next declared covering
entry only for a capacity-only failure. It refuses scientific failures and
raises once the ladder is exhausted.

`ParticleImageNeighborhoodState` stores the build-frame `reference_positions`
(flat case-major) alongside `cell_vectors`, `wrap_counts` and
`stencil_extents`. Its `with_representation_offsets` moves routes, reference
positions and wrap counts together, so image certificates are unchanged, and
marks `representation_overflow` instead of raising.

`ImageVerletParticleNeighborhoodPlan` caches an image relation searched at
`interaction_radius + skin`. `ImageCertificate` (from `image_certificate`) keeps
the cache valid while two conditions hold:

- `2 max|dx| + sum_i K_i |dH_i| <= skin`, charged over the complete stencil
  extents `K`. This covers stored routes and images absent from the cached
  relation.
- The fractional-spread coverage margin stays positive under the current cell.

Otherwise `PreparedImageVerletParticleNeighborhood.update` re-enumerates. Wraps
across a periodic face reuse the cached epoch through exact image-count
offsets; a count difference or re-expressed shift outside the int32 image range
forces a rebuild, and unrepresentable active image counts make the state
unsuccessful. An optional `StreamedRelationPlan` prepares the receiver-major
schedule once per rebuild epoch.

`FractionalOwnerPartition` splits the fractional coordinates of a periodic,
triclinic, or partially periodic `PeriodicCell` into an owner grid. On
nonperiodic axes the outer owner regions extend to infinity. `alias_mask`
conservatively selects the image aliases that can reach each owner's receivers,
and `local_alias_exchange` packs them into fixed-capacity `ImageAliasPackets`
inside one mapped owner region. Completeness over translations is the caller's
`PeriodicImageStencil` contract. See the
[distributed atomistic guide](../../guides_atomistic_distributed_execution.md#domain-plan-and-ownership).

::: phydrax.discretization.PeriodicCell

---

::: phydrax.discretization.PeriodicImageStencil

---

::: phydrax.discretization.lattice_right_inverse_with_status

---

::: phydrax.discretization.lattice_measure

---

::: phydrax.discretization.ParticleImageRelation

---

::: phydrax.discretization.ParticleImageRelationEvidence

---

::: phydrax.discretization.ParticleImageCapacity

---

::: phydrax.discretization.ParticleImageCapacityLadder

---

::: phydrax.discretization.AbstractImageRouteSearch

---

::: phydrax.discretization.CellListImageRouteSearch

---

::: phydrax.discretization.DenseImageRouteSearch

---

::: phydrax.discretization.ImageRouteSearchResult

---

::: phydrax.discretization.AbstractParticleImageNeighborhoodPlan

---

::: phydrax.discretization.AbstractPreparedParticleImageNeighborhood

---

::: phydrax.discretization.CellListParticleImageNeighborhoodPlan

---

::: phydrax.discretization.PreparedCellListParticleImageNeighborhood

---

::: phydrax.discretization.DenseParticleImageNeighborhoodPlan

---

::: phydrax.discretization.PreparedDenseParticleImageNeighborhood

---

::: phydrax.discretization.ParticleImageNeighborhoodState

---

::: phydrax.discretization.ImageVerletParticleNeighborhoodPlan

---

::: phydrax.discretization.PreparedImageVerletParticleNeighborhood

---

::: phydrax.discretization.ParticleImageVerletState

---

::: phydrax.discretization.image_certificate

---

::: phydrax.discretization.ImageCertificate

---

::: phydrax.discretization.FractionalOwnerPartition

---

::: phydrax.discretization.ImageAliasPackets

## Discrete element method

::: phydrax.discretization.ParticlePairKeySpace

---

::: phydrax.discretization.RigidSphereSetPlan

---

::: phydrax.discretization.PreparedRigidSphereSet

---

::: phydrax.discretization.DEMContactModelPlan

---

::: phydrax.discretization.LinearSpringDashpotNormalPlan

---

::: phydrax.discretization.CundallStrackTangentialPlan

---

::: phydrax.discretization.HertzNormalContactPlan

---

::: phydrax.discretization.MindlinTangentialContactPlan

---

::: phydrax.discretization.ImplicitDEMBarrier

---

::: phydrax.discretization.CompliantDEMMethodPlan

---

::: phydrax.discretization.PreparedCompliantDEMDynamics

---

::: phydrax.discretization.CompliantDEMBodyProperties

---

::: phydrax.discretization.AbstractRigidContactShapePlan

---

::: phydrax.discretization.AbstractPreparedRigidContactShape

---

::: phydrax.discretization.RigidContactGeometryBatch

---

::: phydrax.equations.DEMMaterialTable

---

::: phydrax.equations.DiscreteElementProblemIR

---

::: phydrax.equations.compile_discrete_element_problem

---

::: phydrax.solver.DEMFixedStepMethod


### Energy, qualification, and execution

::: phydrax.discretization.DEMEnergyLedgerState

---

::: phydrax.discretization.DEMStepEnergyLedger

---

::: phydrax.discretization.DEMQualificationProfile

---

::: phydrax.discretization.VerletParticleNeighborhoodPlan

---

::: phydrax.discretization.DEMSensitivityPolicy

### Contact extensions

::: phydrax.discretization.ConstantRollingResistancePlan

---

::: phydrax.discretization.ElasticRollingTorsionalResistancePlan

---

::: phydrax.discretization.DMTContactCohesionPlan

---

::: phydrax.discretization.LinearCapillaryBridgePlan

---

::: phydrax.discretization.NearContactLubricationPlan

---

::: phydrax.discretization.CompositeDEMCohesionPlan

---

::: phydrax.discretization.ThorntonLinearPlasticNormalPlan

---

::: phydrax.discretization.ElasticHalfSpaceMulticontactPlan

### Rigid bodies, shapes, and bonds

::: phydrax.discretization.RigidBodySetPlan

---

#### Holonomic rigid-body constraints

::: phydrax.discretization.BallJointSetPlan

---

::: phydrax.discretization.FixedJointSetPlan

---

::: phydrax.discretization.HingeJointSetPlan

---

::: phydrax.discretization.RigidJointGraphPlan

---

::: phydrax.discretization.PreparedRigidJointGraph

---

::: phydrax.discretization.RigidConstraintSolverPlan

---

::: phydrax.discretization.RigidConstraintDynamicsPlan

---

::: phydrax.discretization.PreparedRigidConstraintDynamics

---

::: phydrax.discretization.RigidConstraintState

---

::: phydrax.discretization.RigidConstraintDiagnostics

---

::: phydrax.discretization.RigidConstraintEvaluation

---

::: phydrax.discretization.RigidConstraintStepResult

---

::: phydrax.discretization.RigidConstraintRejectionReason

---

::: phydrax.discretization.PrismaticJointSetPlan

---

::: phydrax.discretization.DistanceJointSetPlan

---

::: phydrax.discretization.PreparedRigidJointCoordinates

---

::: phydrax.discretization.CompliantRigidJointLawPlan

---

::: phydrax.discretization.DissipativeRigidJointLawPlan

---

::: phydrax.discretization.RigidJointEffortMotorPlan

---

::: phydrax.discretization.RigidJointPDServoPlan

---

::: phydrax.discretization.JointLimitPlan

---

::: phydrax.discretization.HardContactRoutePlan

---

::: phydrax.discretization.RigidTopologyPlan

---

::: phydrax.discretization.RigidMPMCouplingPlan

---

::: phydrax.discretization.SphereClumpTemplatePlan

---

::: phydrax.discretization.RigidContactGeometry

---

::: phydrax.discretization.TriangleWallPlan

---

::: phydrax.discretization.FinnieWearPlan

---

::: phydrax.discretization.FixedBondGraphPlan

---

::: phydrax.discretization.TopologyEventPlan

---

::: phydrax.discretization.ConvexShapePlan

---

::: phydrax.discretization.ImplicitRigidShapePlan

---

::: phydrax.discretization.SuperquadricSetPlan

---

::: phydrax.discretization.SuperquadricContactPlan

---

::: phydrax.discretization.RigidSphereClumpSetPlan

---

::: phydrax.discretization.ConvexShapeSetPlan

---

::: phydrax.discretization.PreparedConvexShapeSet

#### Reduced scalar-joint coordinates, actuation, and stops

These owners serve prepared fixed-base reduced articulations with fixed, hinge
and prismatic joints. Rows address scientific joint IDs, never source order;
fixed edges and unknown IDs are refused.

`PreparedLiftedJointCoordinates` keeps the standalone principal-angle reduced
state unchanged and owns the integer hinge winding that lifts it.
`advance(q0, state, v, h)` follows the actual increment `q0 + h v`, so a full
turn with identical principal endpoints still advances the winding. The counter
range is explicit (`maximum_turns` in `[1, 2**29]`); an out-of-range candidate,
nonzero prismatic winding or excessive lift residual is refused and
`accepted_state` retains the source winding. FK periodicity does not make stop
or servo coordinates periodic.

`ScalarJointMotorPlan` maps a separate command space to generalized effort in
the fixed order: clip the command to `[command_lower, command_upper]`, multiply
by the nonzero signed `gear`, then clip to `±effort_limit`. An effort limit of
exactly zero is an explicitly disabled motor. Evaluations report margins,
saturation flags and the actuator source power `tau · v`; `successful` is
finiteness, and outputs are not zeroed on failure. `ScalarJointDampingPlan`
executes `-c v` with `c >= 0` and reports the dissipation rate `sum(c v²)`. Source
effort or velocity bounds are not motors and are not stops.

`ReducedJointStopPlan` bounds lifted coordinates (radians with full winding, or
meters); at least one bound per joint is finite and `lower < upper`. Prepared
rows are ordered `(joint, lower/upper)` with signs `+1/-1` and units `rad`/`m`;
infinite sides are masked. Gaps are affine in lifted coordinates, so
`path_minimum_gaps` is exact on a linear lifted path. `jacobian_operator()`
returns the normal-only stop rows `G` whose exact transpose feeds the same
generalized contact response as geometric contacts. The stop owner never clips
`q` or `v` and performs no separate correction. Activation, saturation and stop
kinks are discrete events for derivatives.

::: phydrax.discretization.PreparedLiftedJointCoordinates

---

::: phydrax.discretization.LiftedJointCoordinateState

---

::: phydrax.discretization.LiftedJointCoordinateUpdate

---

::: phydrax.discretization.ScalarJointMotorPlan

---

::: phydrax.discretization.PreparedScalarJointMotors

---

::: phydrax.discretization.ScalarJointMotorEvaluation

---

::: phydrax.discretization.ScalarJointDampingPlan

---

::: phydrax.discretization.PreparedScalarJointDamping

---

::: phydrax.discretization.ScalarJointDampingEvaluation

---

::: phydrax.discretization.ReducedJointStopPlan

---

::: phydrax.discretization.PreparedReducedJointStops

#### Rigid-constraint rank routes

The rank audit concerns the scaled, unweighted mobile-coordinate Jacobian of the
retracted rigid-joint residuals, with the existing characteristic-length scaling
and strict relative cutoff. It is not the mass-weighted Schur operator.
`RigidConstraintSolverPlan(rank_audit=...)` selects the route; the default
`RigidConstraintRankAudit()` is `"general-dense-svd"` and keeps the existing
dense SVD audit for every maximal-coordinate graph. The kind selector is
`RigidConstraintRankAuditKind = Literal["general-dense-svd", "certified-forest"]`;
`dense_fallback` and `workspace_bytes` are refused on the general route.

`RigidConstraintRankAudit("certified-forest", dense_fallback=..., workspace_bytes=...)`
is opt-in. Host preparation admits a forest: no cycles or parallel edges, at
most one fixed body per connected component (the root when present; a free
component is rooted at its tree center, ties broken by the smallest stable body
ID), isolated mobile bodies allowed. Setting a free root increment to zero is
part of constructing a right inverse, not a physical constraint. Local blocks
come from the audited constraint action and must reproduce it on reference
probes. The generic `phydrax.linalg.certify_block_forest_rank` then proves full
row rank only when its verified lower singular-value bound exceeds the
threshold. Unsupported topologies, rounding profiles and insufficient resource
budgets raise during preparation rather than becoming runtime results. An
explicit `dense_fallback` runs the dense SVD only after an inconclusive
certificate, with its memory planned before tracing; under `vmap` both
branches may execute.

`RigidConstraintDiagnostics.rank` is a `RigidConstraintRankEvidence`. It
replaces the former ambiguous `constraint_rank` and `constraint_condition`
scalars:

| Question | Field |
| --- | --- |
| Outcome | `status` (`RigidConstraintRankStatus`) |
| Route actually executed | `executed_route` (`RigidConstraintRankRoute`) |
| Full row rank accepted | `full_row_rank` |
| Rank bounds | `rank_lower_bound`, `rank_upper_bound` (equal when certified or measured; `[0, row_count]` when inconclusive or failed) |
| Dense condition estimate | `measured_condition` (NaN when no SVD ran; `1` for zero rows) |
| Forest condition upper bound | `certificate.condition_upper_bound` |

Only dense measurements report `RANK_DEFICIENT_MEASURED`. `INCONCLUSIVE` means
the sufficient forest test failed; it is never evidence of deficiency, and a
step rejects it distinctly. A forest bound is never labeled a computed
condition number, and no SVD is reported unless one executed.
`PreparedRigidConstraintDynamics.constraint_rank_evidence(kinematics)` returns
the evidence at one configuration alone. Full row rank of the velocity Jacobian
does not establish regularity of the nonlinear position-root KKT system.

::: phydrax.discretization.RigidConstraintRankAudit

---

::: phydrax.discretization.RigidConstraintRankRoute

---

::: phydrax.discretization.RigidConstraintRankStatus

---

::: phydrax.discretization.RigidConstraintRankEvidence

---

::: phydrax.discretization.PreparedRigidConstraintRank

---

::: phydrax.discretization.prepare_rigid_constraint_rank

### Internal particle state and processes

::: phydrax.discretization.RadialShellMeshPlan

---

::: phydrax.discretization.ParticleInternalBatchPlan

---

::: phydrax.discretization.ParticleInternalBatchState

---

::: phydrax.discretization.ParticleConversionState

---

::: phydrax.discretization.DensityPorosityMorphologyPlan

---

::: phydrax.discretization.ReciprocalPairRadiationPlan

---

::: phydrax.discretization.ReactiveParticleTemplatePlan

---

::: phydrax.discretization.ReactiveParticleTemplateDistributionPlan

---

::: phydrax.discretization.ParticleInsertionPlan

---

::: phydrax.discretization.insert_reactive_particles

---

::: phydrax.discretization.ParticleRegionPlan

---

::: phydrax.discretization.MassFlowSurfacePlan

### CFD--DEM

The unresolved coupling owns conservative fluid–particle exchange and a typed
`drag_law`; pressure-gradient force remains a separate
`include_pressure_gradient` option. See the [CFD–DEM guide](../../guides_cfd_dem.md)
for equations, source provenance, and the clean closure-to-drag cutover.

`InterphaseDragLawPlan(correlation, /, *, maximum_reynolds=None)` selects
`stokes`, `schiller_naumann`, `di_felice`, `ergun`, `wen_yu`, `gidaspow`,
`huilin_gidaspow`, or `beetstra`. Its positional evaluation contract is
`evaluate(porosity, relative_speed, particle_diameter, dynamic_viscosity,
fluid_density, particle_volume, /)`.

`beta` is bulk momentum exchange (kg m⁻³ s⁻¹), while
`particle_drag_coefficient` multiplies one particle's slip velocity (kg s⁻¹).
Do not infer one from the other using a supplied particle volume for the
isolated-sphere laws. The reported Reynolds number is interstitial ρdw/μ;
Di Felice, Wen–Yu, and Beetstra use ερdw/μ inside their fits, and a declared
Reynolds cap applies to that fit convention. Porosity, regime, branch margin,
derivative validity, admission, numerical success, and static law identity
remain visible in `CFDEMCouplingEvaluation.drag`. Hard switches and zero-speed
value limits do not automatically qualify derivatives.

::: phydrax.equations.AbstractInterphaseDragLaw

---

::: phydrax.equations.InterphaseDragLawPlan

---

::: phydrax.equations.InterphaseDragEvaluation

---

::: phydrax.discretization.PreparedMeshParticleGridSplat

---

::: phydrax.discretization.ParticleContactExchangePlan

---

::: phydrax.equations.UnresolvedCFDEMCouplingPlan

---

::: phydrax.equations.MACPenaltyIBCFDEMCouplingPlan

---

::: phydrax.equations.ParticleContinuumExchangePlan

---

::: phydrax.equations.ReactiveCFDDEMCouplingPlan

### Wet barrier and periodic rheology

::: phydrax.discretization.DEMBarrierCapillaryPlan

---

::: phydrax.discretization.PeriodicNeighborhoodEnvelope

---

::: phydrax.discretization.DEMBulkStressPlan

### Runtime SPH sources

::: phydrax.discretization.SPHParticleSourcePlan

---

::: phydrax.discretization.emit_sph_particles

## Adaptive particle runtime

::: phydrax.discretization.ParticleCapacityGrowthPolicy

---

::: phydrax.discretization.ParticleCapacityRequest

---

::: phydrax.discretization.ParticleExecutionEpoch

---

::: phydrax.discretization.grow_particle_execution_epoch

---

::: phydrax.discretization.insert_reactive_particles_with_growth

---

::: phydrax.discretization.UnstructuredParticleInternalMeshPlan

---

::: phydrax.discretization.ParticleInternalAdaptationPolicy

---

::: phydrax.discretization.adapt_particle_internal_mesh

---

::: phydrax.discretization.SuperquadricTriangleContactPlan

---

::: phydrax.discretization.superquadric_triangle_contact_geometry

## SPH kernels and dynamics

::: phydrax.discretization.AbstractSPHSmoothingKernel

---

::: phydrax.discretization.WendlandC2SPHKernel

---

::: phydrax.discretization.CubicSplineSPHKernel

---

::: phydrax.discretization.BarotropicSPHMethodPlan

---

::: phydrax.discretization.PreparedBarotropicSPHDynamics

---

::: phydrax.discretization.BarotropicSPHDiagnostics

---

::: phydrax.discretization.BarotropicSPHStepRestriction

## Weakly compressible SPH

::: phydrax.discretization.AbstractSPHDensityPlan

---

::: phydrax.discretization.SummationDensityPlan

---

::: phydrax.discretization.ContinuityDensityPlan

---

::: phydrax.discretization.WeaklyCompressibleSPHStateLayout

---

::: phydrax.discretization.MorrisViscosityPlan

---

::: phydrax.discretization.WeaklyCompressibleSPHMethodPlan

---

::: phydrax.discretization.PreparedWeaklyCompressibleSPHDynamics

---

::: phydrax.discretization.WeaklyCompressibleSPHDiagnostics

---

::: phydrax.discretization.WeaklyCompressibleSPHStepRestriction

## Advanced particle methods

::: phydrax.discretization.ParticleAssemblyPlan

---

::: phydrax.discretization.DenseBipartiteParticleNeighborhoodPlan

---

::: phydrax.discretization.WallParticleGenerationPlan

---

::: phydrax.discretization.AdamiWallBoundaryPlan

---

::: phydrax.discretization.FreeSurfaceDetectionPlan

---

::: phydrax.discretization.AntuonoDeltaSPHDiffusionPlan

---

::: phydrax.discretization.MonaghanArtificialViscosityPlan

---

::: phydrax.discretization.TransportVelocitySPHMethodPlan

---

::: phydrax.discretization.AlgebraicSmoothingLengthPlan

---

::: phydrax.discretization.MultiphaseWCSPHPlan

---

::: phydrax.discretization.IISPHMethodPlan

---

::: phydrax.discretization.DFSPHMethodPlan

## Qualification

::: phydrax.discretization.ParticleMethodMaturity

---

::: phydrax.discretization.ParticleQualificationClaim

---

::: phydrax.discretization.ParticleConstraintResiduals

---

::: phydrax.discretization.ParticleQualificationProfile

---

::: phydrax.discretization.ParticleQualificationResult

---

::: phydrax.discretization.particle_constraint_residuals

## Production hardening

::: phydrax.discretization.MultiPopulationCellPlan

---

::: phydrax.linalg.SmallLinearSolvePlan

---

::: phydrax.discretization.AdaptiveHRootPlan

---

::: phydrax.discretization.FreeSurfaceReconstructionPlan

---

::: phydrax.discretization.BalancedInterfaceForcePlan

---

::: phydrax.discretization.IISPHAssembledOracle

---

::: phydrax.discretization.ProductionProjectedSolvePlan

---

::: phydrax.discretization.ParticleDomainDecompositionPlan

---

::: phydrax.discretization.ParticleBenchmarkRegistry

---

::: phydrax.discretization.ParticleReplayPacket

## Materials and compilation

::: phydrax.equations.AbstractBarotropicMaterial

---

::: phydrax.equations.TaitBarotropicMaterial

---

::: phydrax.equations.BarotropicFluidProblemIR

---

::: phydrax.equations.CompiledBarotropicSPHProblem

---

::: phydrax.equations.compile_barotropic_sph_problem

---

::: phydrax.equations.WeaklyCompressibleFluidProblemIR

---

::: phydrax.equations.CompiledWeaklyCompressibleSPHProblem

---

::: phydrax.equations.compile_weakly_compressible_sph_problem

## Learned conservative pair exchanges

`ParticleExchangeLedger` records same-population total exchange, action-reaction
defect, torque, relative power, pair count, and finiteness. It complements the
two-population `ParticleInteractionLedger`.

The neural-operator adapters live in `phydrax.nn.operator.adapters`:

- `particle_pair_operator_batch` represents one fixed-capacity pair relation as
  a masked point-cloud operator batch;
- `PairwiseExchangeFeatureSchema` binds feature order, units, dtype, and
  relation schema;
- `PairwiseExchangeBindingPlan` binds one trained artifact and exchange kind;
- `PreparedPairwiseExchangeBinding` predicts once per canonical pair and
  deposits equal and opposite endpoint values with `scatter_pair_exchange`.

An arbitrary vector exchange guarantees linear momentum. A central scalar force
also guarantees zero internal torque. A scalar flux conserves the exchanged
scalar. None of these constructions alone guarantees energy conservation,
dissipation, rotational equivariance, or smooth derivatives through a changed
neighbor relation.

## Level-set rigid grains and packing

Level-set fields are approximate `LEVEL_SET` interpolants, not exact SDFs.
The common compliant engine integrates symmetric two-sided half physical-area
quadrature. Per-area normal/tangential stiffness is N/m³; point-force laws
cannot be reused on these routes. See [the level-set guide](../../guides_level_set_dem.md)
for mass-property evidence, source enclosures, history, and atomic refusal.

::: phydrax.discretization.LevelSetShapeTemplatePlan

::: phydrax.discretization.LevelSetRigidShapeSetPlan

::: phydrax.discretization.PreparedLevelSetRigidShapeSet

::: phydrax.discretization.LevelSetMassPropertyEvidence

::: phydrax.discretization.LevelSetPenaltyNormalPlan

::: phydrax.discretization.LevelSetTangentialContactPlan

::: phydrax.discretization.evaluate_level_set_point_query

::: phydrax.discretization.LevelSetPointQueryResult

::: phydrax.discretization.PosedSignedFieldQueryResult

::: phydrax.discretization.RandomSequentialPackingPlan

::: phydrax.discretization.RigidDEMPackingResult

::: phydrax.discretization.spatial.SparseVoxelDifferentialQueryResult

::: phydrax.applications.contact.MPMLevelSetGrainContactParticipant

::: phydrax.applications.contact.MPMLevelSetGrainContactEvaluation
