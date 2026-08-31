# Particle methods

## Material support and execution

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

::: phydrax.discretization.SoftSphereDEMMethodPlan

---

::: phydrax.discretization.PreparedSoftSphereDEMDynamics

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

::: phydrax.discretization.SuperquadricDEMPlan

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

::: phydrax.discretization.ConservativeParticleGridTransferPlan

---

::: phydrax.discretization.ParticleContactExchangePlan

---

::: phydrax.equations.UnresolvedCFDEMCouplingPlan

---

::: phydrax.equations.ResolvedIBCFDEMCouplingPlan

---

::: phydrax.equations.ParticleContinuumExchangePlan

---

::: phydrax.equations.ReactiveCFDDEMCouplingPlan

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
