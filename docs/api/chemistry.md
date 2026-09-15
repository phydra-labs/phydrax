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

::: phydrax.chemistry.MolecularKohnShamPlan

::: phydrax.chemistry.SCFConvergencePlan

::: phydrax.chemistry.SCFAccelerationPlan

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
