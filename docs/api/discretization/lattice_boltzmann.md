# Lattice Boltzmann and discrete velocity

## Lattices, quadratures, scaling, and precision

::: phydrax.discretization.LatticeBoltzmannVelocitySet

---

::: phydrax.discretization.D2Q9

---

::: phydrax.discretization.D3Q19

---

::: phydrax.discretization.D3Q27

---

::: phydrax.discretization.LatticeBoltzmannCapabilityEvidence

---

::: phydrax.discretization.LatticeBoltzmannScaling

---

::: phydrax.discretization.LatticeBoltzmannPrecisionPolicy

---

::: phydrax.discretization.CertifiedDiscreteVelocityQuadrature

---

::: phydrax.discretization.PreparedOffLatticeSemiLagrangianDVM

## Prepared discretization

::: phydrax.discretization.LatticeBoltzmannPlan

---

::: phydrax.discretization.LatticeBoltzmannDiscretization

## Collision, moments, and forcing

::: phydrax.discretization.BGKCollisionPlan

---

::: phydrax.discretization.TRTCollisionPlan

---

::: phydrax.discretization.MRTCollisionPlan

---

::: phydrax.discretization.RegularizedCollisionPlan

---

::: phydrax.discretization.CentralMomentCollisionPlan

---

::: phydrax.discretization.CumulantCollisionPlan

---

::: phydrax.discretization.KBCCollisionPlan

---

::: phydrax.discretization.EntropicCollisionPlan

---

::: phydrax.discretization.SmagorinskyCollisionPlan

---

::: phydrax.discretization.MomentBasisPlan

---

::: phydrax.discretization.RelaxationSpectrumPlan

---

::: phydrax.discretization.GuoForcingPlan

---

::: phydrax.discretization.VelocityDependentAccelerationPlan

---

::: phydrax.discretization.DampedLocalRootSolver

---

::: phydrax.discretization.LatticeBoltzmannMethodPlan

---

::: phydrax.discretization.PreparedLatticeBoltzmannMethodPlan

## Boundaries and geometry

::: phydrax.discretization.LatticeBoltzmannGeometrySnapshot

---

::: phydrax.discretization.LatticeBoltzmannBoundaryPlan

---

::: phydrax.discretization.StagedLatticeBoltzmannBoundaryPlan

---

::: phydrax.discretization.compile_staged_lattice_boltzmann_boundary

---

::: phydrax.discretization.CompiledLatticeBoltzmannLinkTopology

---

::: phydrax.discretization.PreparedStagedLatticeBoltzmannBoundary

---

::: phydrax.discretization.LatticeBoltzmannBoundaryParameters

---

::: phydrax.discretization.FixedSDFLinkGeometry

---

::: phydrax.discretization.prepare_lattice_boltzmann_link_geometry

---

::: phydrax.discretization.LatticeBoltzmannGeometryImportEvidence

---

::: phydrax.discretization.PreparedLatticeBoltzmannLinkGeometry

---

::: phydrax.discretization.MovingSDFGeometryPlan

---

::: phydrax.discretization.LatticeBoltzmannGeometryEpoch

---

::: phydrax.discretization.LatticeBoltzmannGeometryRefresh

---

::: phydrax.discretization.LatticeBoltzmannPopulationTransferPlan

---

::: phydrax.discretization.LatticeBoltzmannGeometryTransaction

---

::: phydrax.discretization.ImmersedBoundaryForcingPlan

## Multiblock, refinement, and mapped grids

::: phydrax.discretization.LatticeBoltzmannBlockInterfacePlan

---

::: phydrax.discretization.LatticeBoltzmannMultiblockCouplingPlan

---

::: phydrax.discretization.LatticeBoltzmannAMRPlan

---

::: phydrax.discretization.LatticeBoltzmannAMRTransferPlan

---

::: phydrax.discretization.PreparedLatticeBoltzmannAMRTransfer

---

::: phydrax.discretization.LatticeBoltzmannAMRTemporalInterfacePlan

---

::: phydrax.discretization.LatticeBoltzmannAMRInterfaceEvidence

---

::: phydrax.discretization.LatticeBoltzmannCollisionAwareAMRAdvanceResult

---

::: phydrax.discretization.MappedLatticeBoltzmannPlan

## Shared thermodynamic closure

::: phydrax.equations.AbstractKineticThermodynamicClosure

---

::: phydrax.equations.BinaryThermodynamicParameters

---

::: phydrax.equations.BinaryPhaseThermodynamicClosure

---

::: phydrax.equations.ThermodynamicForceRepresentation

---

::: phydrax.discretization.PreparedBinaryKineticThermodynamics

---

::: phydrax.discretization.BinaryKineticThermodynamicFields

## Vascular target profiles

::: phydrax.equations.ParabolicVelocityParameters

---

::: phydrax.equations.ParabolicVelocityProfilePlan

---

::: phydrax.equations.WomersleyVelocityParameters

---

::: phydrax.equations.WomersleyVelocityProfilePlan

## Multiphysics distributions

::: phydrax.discretization.ColourGradientLBMMethod

---

::: phydrax.discretization.FreeEnergyLBMMethod

---

::: phydrax.discretization.ThermalLatticeBoltzmannPlan

---

::: phydrax.discretization.SpeciesLatticeBoltzmannPlan

## Program manifest and checkpoint

::: phydrax.discretization.KineticFieldRole

---

::: phydrax.discretization.KineticFailureScope

---

::: phydrax.discretization.KineticProgramManifest

---

::: phydrax.discretization.KineticFieldSpec

---

::: phydrax.discretization.KineticStageSpec

---

::: phydrax.discretization.KineticCheckpoint

---

::: phydrax.discretization.KineticCheckpointPlan

---

::: phydrax.discretization.write_kinetic_checkpoint

---

::: phydrax.discretization.read_kinetic_checkpoint

## Runtime and diagnostics

::: phydrax.discretization.LatticeBoltzmannRuntimeParameters

---

::: phydrax.discretization.LatticeBoltzmannMacroscopicState

---

::: phydrax.discretization.LatticeBoltzmannDiagnostics

---

::: phydrax.discretization.PreparedLatticeBoltzmannDynamics

## Execution

::: phydrax.discretization.ShardedLatticeBoltzmannExecutionPlan

---

::: phydrax.discretization.LatticeBoltzmannHaloSchedule

---

::: phydrax.discretization.PreparedDistributedLatticeBoltzmannDynamics

---

::: phydrax.discretization.AALatticeBoltzmannPlan

---

::: phydrax.discretization.FusedLatticeBoltzmannExecutionPlan

## Equation compilation

::: phydrax.equations.LatticeBoltzmannProblem

---

::: phydrax.equations.compile_lattice_boltzmann_problem

---

::: phydrax.equations.ColourGradientLatticeBoltzmannProblem

---

::: phydrax.equations.FreeEnergyLatticeBoltzmannProblem

---

::: phydrax.equations.ThermalLatticeBoltzmannProblemIR

---

::: phydrax.equations.SpeciesLatticeBoltzmannProblemIR

---

::: phydrax.equations.SmoothCompressibleD2VKineticMethod

---

::: phydrax.equations.FixedConformingFVKineticInterfacePlan

## Fixed-step execution and export

::: phydrax.solver.LatticeBoltzmannFixedStepMethod

---

::: phydrax.solver.ReactiveSpeciesCouplingSchedulePlan

---

::: phydrax.solver.FixedStepReplayPolicy

---

::: phydrax.solver.FixedStepRolloutPlan

---

::: phydrax.export.LatticeBoltzmannIREEForwardContract
