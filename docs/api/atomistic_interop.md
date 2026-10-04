# Atomistic frames and interoperability

## Frames and reporting

::: phydrax.atomistic.AtomisticFrame

::: phydrax.atomistic.AtomisticFrameFields

::: phydrax.atomistic.AtomisticMetadata

::: phydrax.atomistic.AtomisticSelectionPlan

::: phydrax.atomistic.AtomisticReporterPlan

::: phydrax.atomistic.AtomisticRerunPlan

::: phydrax.atomistic.AtomisticRerunResult

## Trajectories

::: phydrax.atomistic.AbstractAtomisticTrajectorySourcePlan

::: phydrax.atomistic.AbstractAtomisticTrajectorySinkPlan

::: phydrax.atomistic.InMemoryTrajectorySourcePlan

::: phydrax.atomistic.interchange.H5MDTrajectoryPlan

::: phydrax.atomistic.interchange.ExtendedXYZTrajectoryPlan

## ASE structures

::: phydrax.atomistic.interchange.from_ase_atoms

::: phydrax.atomistic.interchange.to_ase_atoms

::: phydrax.atomistic.interchange.is_ase_available

::: phydrax.atomistic.interchange.require_ase

`ASE_PARTICLE_ID_ARRAY` and `ASE_SOURCE_ID_INFO` name the reserved ASE array and info
field used for stable material-atom identity and source provenance.

## MDAnalysis

::: phydrax.atomistic.interchange.atomistic_frame_from_mdanalysis

::: phydrax.atomistic.interchange.atomistic_metadata_from_mdanalysis

::: phydrax.atomistic.interchange.mdanalysis_selection

## Native ASE calculator

Both names resolve lazily from `phydrax.atomistic.interchange` and require ASE when
accessed. Stress is the tensile ASE Voigt vector `(xx, yy, zz, yz, xz, xy)` in eV/Å³;
an unavailable stress raises `PropertyNotImplementedError`.

::: phydrax.atomistic.interchange.NativeASECalculatorPlan

::: phydrax.atomistic.interchange.NativeASECalculator

## i-PI

`IPITransportMode` is `"unix"` or `"tcp"`. `IPIVirialPolicy` is `"required"` or
`"optional"`; neither substitutes a zero virial. `IPIInverseCellPolicy` is `"verify"` or
`"ignore"`. `IPIListener.accept()` returns an `IPISession`. `IPITransportStatus` is one of
`READY`, `HAVE_DATA`, `CLOSED`, `PROTOCOL_ERROR`, or `PROVIDER_ERROR`. Wire values are
Hartree atomic units; the reply carries the configurational virial `W = -V σ`, never the
tensile stress `σ`.

::: phydrax.atomistic.interchange.IPITransportPlan

::: phydrax.atomistic.interchange.IPISession

::: phydrax.atomistic.interchange.IPIRequest

::: phydrax.atomistic.interchange.serve_ipi_once

::: phydrax.atomistic.interchange.TransportedExternalAtomisticProvider

## MACE checkpoint conversion

`MACE_PROVIDER_RELEASES` maps the admitted provider distributions to their exact
releases (`mace-torch` 0.3.16, `e3nn` 0.4.4). `MACESourceKind` is `"torch-state-dict"` or
`"torch-full-model"`; `MACEEvaluationDtype` is `"float32"` or `"float64"`. The provider
subprocess is a resource boundary, not a security sandbox. No checkpoint weights are
bundled and nothing is downloaded.

::: phydrax.atomistic.interchange.convert_mace_checkpoint

::: phydrax.atomistic.interchange.MACESource

::: phydrax.atomistic.interchange.TrustedTorchPickleSource

::: phydrax.atomistic.interchange.MACEProviderRuntime

::: phydrax.atomistic.interchange.MACEConversionLimits

::: phydrax.atomistic.interchange.MACECheckpointConversion

::: phydrax.atomistic.interchange.MACESourceProvenance

::: phydrax.atomistic.interchange.MACESourceRefusedError

::: phydrax.atomistic.interchange.evaluate_mace_source

::: phydrax.atomistic.interchange.mace_source_gradients

::: phydrax.atomistic.interchange.MACEProviderConfiguration

::: phydrax.atomistic.interchange.MACEProviderCase

::: phydrax.atomistic.interchange.MACEProviderEvaluation

::: phydrax.atomistic.interchange.MACEProviderGradients

::: phydrax.atomistic.interchange.create_mace_provider_fixture

::: phydrax.atomistic.interchange.MACEProviderFixture

## External boundaries

::: phydrax.atomistic.interchange.PackmolAssemblyPlan

::: phydrax.atomistic.interchange.AtomisticInterchangeBundle

::: phydrax.atomistic.interchange.AtomisticInterchangeReport

::: phydrax.atomistic.interchange.from_openmm_system

::: phydrax.atomistic.interchange.from_openff_interchange

::: phydrax.atomistic.interchange.from_parmed_structure

::: phydrax.atomistic.interchange.to_openmm_system

::: phydrax.atomistic.interchange.to_openff_interchange
