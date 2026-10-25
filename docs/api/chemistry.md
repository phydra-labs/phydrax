# Computational chemistry API

## Electronic models, tasks, numerics, and contexts

::: phydrax.chemistry.MolecularElectronicSectorPlan

::: phydrax.chemistry.PreparedMolecularElectronicSector

::: phydrax.chemistry.AbstractElectronicMethodPlan

::: phydrax.chemistry.HartreeFockMethodPlan

::: phydrax.chemistry.KohnShamMethodPlan

::: phydrax.chemistry.DensityFunctionalPlan

::: phydrax.chemistry.ElectronicModelChemistryPlan

::: phydrax.chemistry.GroundStateTaskPlan

::: phydrax.chemistry.CorrelationTaskPlan

::: phydrax.chemistry.LinearResponseTaskPlan

::: phydrax.chemistry.ExcitedManifoldTaskPlan

::: phydrax.chemistry.NonadiabaticCouplingTaskPlan

::: phydrax.chemistry.BandStructureTaskPlan

::: phydrax.chemistry.ElectronicNumericalPlan

::: phydrax.chemistry.ElectronicEvaluationContext

::: phydrax.chemistry.ElectronicInitialGuessState

::: phydrax.chemistry.ExternalFieldState

## Providers, calculations, results, and lifecycle

::: phydrax.chemistry.ElectronicProviderCapabilities

::: phydrax.chemistry.AbstractElectronicProvider

::: phydrax.chemistry.ElectronicCalculationPlan

::: phydrax.chemistry.ElectronicEnergyLedger

::: phydrax.chemistry.ElectronicEnergyEvaluation

::: phydrax.chemistry.ElectronicEnergyForceEvaluation

::: phydrax.chemistry.ElectronicEnergyForceHessianEvaluation

::: phydrax.chemistry.ElectronicGroundStatePropertyEvaluation

::: phydrax.chemistry.ChemistryResultCodec

::: phydrax.chemistry.ProductionChemistryArchivePlan

## Molecular mean field and response

::: phydrax.chemistry.MolecularHartreeFockPlan

The `integrals` keyword takes
`phydrax.operators.quantum.gaussian.MolecularIntegralPlan`; native exact dense
ERIs remain the default. Native screened direct J/K, fixed factorized ERIs, and
optional cuEST DF are explicit sources of that owner, not separate HF keyword
arguments. Unsupported gradients, response, stability, or correlation must not
substitute a different integral definition.

::: phydrax.chemistry.MolecularRHFProvider

::: phydrax.chemistry.PreparedMolecularRHFCalculation

The typed RHF provider admits native exact or optional cuEST integer,
all-electron, nonperiodic RHF energy/first-force tasks. Bind a model chemistry
and use `numerical=provider.numerical` when constructing the calculation.
Results use the system's energy and length units; atomic-unit HF calls use
Hartree and bohr. Forces include Pulay and nuclear terms and require stationary
density/orbital evidence, not merely a finite converged energy. The provider
does not admit AD, batching, Hessians, external-field/embedding contexts, or
fixed-factor/direct analytic-force substitution. See the
[typed energy/force/PES example](../cookbook/production_computational_chemistry.md).

::: phydrax.chemistry.ElectronicPotentialEnergySurface

::: phydrax.chemistry.PotentialEnergySurfaceEvaluation

`AtomisticPotentialEnergySurface(system, potential)` presents a native prepared
atomistic Hamiltonian, which owns its neighborhood bundle, as a chemistry surface.
Its host `evaluate` grows declared capacities when a point does not fit, and
reports an exhausted ladder or a resource failure as an unsuccessful evaluation.
Device NEB uses its `potential` directly. `ExternalAtomisticPotentialEnergySurface`
keeps an external atomistic provider on the host.

::: phydrax.chemistry.AtomisticPotentialEnergySurface

::: phydrax.chemistry.ExternalAtomisticPotentialEnergySurface


::: phydrax.chemistry.MolecularKohnShamPlan

::: phydrax.chemistry.SCFConvergencePlan

::: phydrax.chemistry.SCFAccelerationPlan

::: phydrax.chemistry.SCFDIISEvidence

`state.evidence.diis` retains the most recent constrained native SVD solve,
including minimum-norm coefficients, affine-constraint residual/tolerance,
original error scale, native status, rank, rank cutoff, and normal residual.
Rank-deficient histories do not trigger diagonal jitter or silent DIIS removal.
If DIIS has not run, this field is `None`.


::: phydrax.chemistry.ElectronicOccupationPlan

::: phydrax.chemistry.InitialGuessPlan

::: phydrax.chemistry.SCFStabilityPlan

::: phydrax.chemistry.MeanFieldResponsePlan

::: phydrax.chemistry.ContinuumSolvationPlan

::: phydrax.chemistry.RelativisticOneElectronPlan

## Correlation and excited states

::: phydrax.chemistry.CorrelatedOrbitalPartition

::: phydrax.chemistry.MolecularIntegralTransformationPlan

::: phydrax.chemistry.MP2Plan

::: phydrax.chemistry.CoupledClusterPlan

::: phydrax.chemistry.CoupledClusterCheckpoint

::: phydrax.chemistry.CASCIPlan

::: phydrax.chemistry.CASSCFPlan

::: phydrax.chemistry.ElectronicManifoldResult

::: phydrax.chemistry.HartreeFockExcitedResponsePlan

::: phydrax.chemistry.KohnShamExcitedResponsePlan

::: phydrax.chemistry.RandomPhaseApproximationPlan

::: phydrax.chemistry.StateTrackingResult

::: phydrax.chemistry.MinimumEnergyCrossingPlan

::: phydrax.chemistry.FewestSwitchesSurfaceHoppingPlan

## Molecular workflows and spectra

::: phydrax.chemistry.MolecularGeometryOptimizationPlan

::: phydrax.chemistry.MolecularHessianPlan

::: phydrax.chemistry.VibrationalAnalysisPlan

::: phydrax.chemistry.HarmonicThermochemistryPlan

::: phydrax.chemistry.MolecularCoordinateSystemPlan

::: phydrax.chemistry.InternalCoordinateOptimizationPlan

::: phydrax.chemistry.DimerSaddleRefinementPlan

::: phydrax.chemistry.NudgedElasticBandPlan

`NudgedElasticBandPlan` is the host-provider route. Each round evaluates every
interior image through `surface.evaluate` and consumes the result with one native
optimizer step. `AtomisticNudgedElasticBandPlan` is the bounded device route over
an `AtomisticPotentialEnergySurface`. It moves `G` whole fixed-cell bands as
groups of one optimizer layout, commits them band-atomically, and replays
capacity-only failures after `grow`. Both routes share
`nudged_elastic_band_forces`, report images, energies, physical forces, and NEB
forces from one accepted iterate, and refuse a downhill-checked FIRE. Periodic
paths take an explicit image branch or the certified shortest-image policy.
Variable-cell NEB is not provided. See
[native reaction paths](../guides_atomistic_dynamics.md#native-reaction-paths).

::: phydrax.chemistry.reaction.AtomisticNudgedElasticBandPlan

::: phydrax.chemistry.reaction.PreparedAtomisticNudgedElasticBand

::: phydrax.chemistry.reaction.ReactionPathBranch

::: phydrax.chemistry.reaction.ReactionPathBranchCertificate

::: phydrax.chemistry.reaction.resolve_reaction_path_branch

::: phydrax.chemistry.reaction.AtomisticReactionPathState

::: phydrax.chemistry.reaction.AtomisticPathHamiltonianTransitionResult

::: phydrax.chemistry.reaction.ReactionPathResult

::: phydrax.chemistry.reaction.nudged_elastic_band_forces

::: phydrax.chemistry.reaction.NudgedElasticBandProjection

::: phydrax.chemistry.IntrinsicReactionCoordinatePlan

::: phydrax.chemistry.TransitionStateRatePlan

::: phydrax.chemistry.ReactionNetworkPlan

::: phydrax.chemistry.SpectralProfilePlan

::: phydrax.chemistry.IRSpectrumPlan

::: phydrax.chemistry.RamanSpectrumPlan

::: phydrax.chemistry.ResonanceRamanPlan

::: phydrax.chemistry.DuschinskyFranckCondonPlan

::: phydrax.chemistry.AnharmonicForceFieldPlan

::: phydrax.chemistry.VibrationalPerturbationPlan

::: phydrax.chemistry.VibrationalConfigurationPlan

::: phydrax.chemistry.HinderedRotorPlan

::: phydrax.chemistry.ConformationalEnsemblePlan

## Interchange providers

::: phydrax.chemistry.interchange.ASECalculatorProvider

::: phydrax.chemistry.interchange.QCEngineProvider

::: phydrax.chemistry.interchange.PySCFProvider

::: phydrax.chemistry.interchange.PySCFCoupledClusterProvider

::: phydrax.chemistry.interchange.PySCFMolecularCoupledClusterGradientProvider

::: phydrax.chemistry.interchange.import_basis_set_exchange

::: phydrax.chemistry.interchange.electronic_calculation_to_qcschema

::: phydrax.chemistry.interchange.electronic_evaluation_from_qcschema
