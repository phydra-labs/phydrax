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
