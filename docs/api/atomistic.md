# Atomistic learning and dynamics

`phydrax.atomistic` owns finite molecular learning and conservative atomistic dynamics.
`phydrax.nn.atomistic` owns PaiNN and low-degree Cartesian NequIP model architectures.
Material particles, sparse relations, graph IR, precision, replay, and qualification remain
shared native substrates.

## Structures, units, systems, and topology

::: phydrax.atomistic.AtomisticScaleContract

::: phydrax.atomistic.AtomisticUnitSystem

::: phydrax.atomistic.AtomicStructure

::: phydrax.atomistic.AtomisticBatch

::: phydrax.atomistic.AtomisticSystemPlan

::: phydrax.atomistic.PreparedAtomisticSystem

::: phydrax.atomistic.MolecularTopologyPlan

::: phydrax.atomistic.PreparedMolecularTopology

## Graph realization and learned potentials

::: phydrax.atomistic.AtomisticGraphExecutionPlan

::: phydrax.atomistic.AtomisticGraph

::: phydrax.atomistic.realize_atomistic_graph

::: phydrax.atomistic.realize_particle_atomistic_graph

::: phydrax.atomistic.AbstractAtomisticPotential

::: phydrax.atomistic.checkpoint_atomistic_potential

::: phydrax.nn.atomistic.PaiNNPotential

::: phydrax.nn.atomistic.NequIPPotential

::: phydrax.atomistic.AtomisticPrediction

::: phydrax.atomistic.AtomisticProvenance

::: phydrax.atomistic.energy_and_forces

## Potential programs and classical terms

::: phydrax.atomistic.AtomisticPotentialProgram

::: phydrax.atomistic.PreparedAtomisticPotentialProgram

::: phydrax.atomistic.LearnedGraphPotentialTerm

::: phydrax.atomistic.HarmonicBondPotential

::: phydrax.atomistic.HarmonicAnglePotential

::: phydrax.atomistic.PeriodicTorsionPotential

::: phydrax.atomistic.LennardJonesPotential

::: phydrax.atomistic.DirectCoulombPotential

::: phydrax.atomistic.EwaldReferencePotential

::: phydrax.atomistic.ParticleMeshEwaldPotential

## Dynamics, constraints, and thermodynamics

::: phydrax.atomistic.VelocityVerletPlan

::: phydrax.atomistic.BAOABLangevinPlan

::: phydrax.atomistic.DistanceConstraintPlan

::: phydrax.atomistic.AtomisticDynamicsPlan

::: phydrax.atomistic.PreparedAtomisticDynamics

::: phydrax.atomistic.AtomisticDynamicsState

::: phydrax.atomistic.AtomisticDynamicsDiagnostics

::: phydrax.atomistic.AtomisticStepEvaluation

::: phydrax.atomistic.AtomisticStepRejectionReason

::: phydrax.atomistic.ThermodynamicAccumulator

::: phydrax.atomistic.RadialDistributionPlan

## Rollout, replay, checkpoints, and stress

::: phydrax.atomistic.AtomisticTrajectoryPlan

::: phydrax.atomistic.AtomisticRolloutPlan

::: phydrax.atomistic.AtomisticReplayPolicy

::: phydrax.atomistic.AtomisticCheckpointPlan

::: phydrax.atomistic.write_atomistic_checkpoint

::: phydrax.atomistic.read_atomistic_checkpoint

::: phydrax.atomistic.atomistic_cell_energy_and_stress

::: phydrax.atomistic.IsotropicMonteCarloBarostatPlan

## Hybrid and specialized methods

::: phydrax.atomistic.AlchemicalScaledPotential

::: phydrax.atomistic.RegionMaskedPotential

::: phydrax.atomistic.RESPAPlan

::: phydrax.atomistic.AbstractExternalAtomisticProvider

::: phydrax.atomistic.BornOppenheimerVelocityVerletPlan

::: phydrax.atomistic.RingPolymerPlan

::: phydrax.atomistic.PreparedRingPolymerDynamics

::: phydrax.atomistic.VarianceConstrainedSemiGrandPlan

## Typed training and rMD17

::: phydrax.atomistic.AtomisticTrainingProblem

::: phydrax.atomistic.AtomisticTrainingPolicy

::: phydrax.atomistic.AtomisticTrainingResult

::: phydrax.atomistic.fit_atomistic_potential

::: phydrax.atomistic.RMD17Dataset

::: phydrax.atomistic.load_rmd17_npz

::: phydrax.atomistic.split_rmd17

## Qualification

::: phydrax.atomistic.AtomisticDynamicsQualificationClaim

::: phydrax.atomistic.AtomisticDynamicsQualificationProfile

::: phydrax.atomistic.AtomisticDynamicsQualificationResult
