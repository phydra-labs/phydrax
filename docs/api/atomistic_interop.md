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

`phydrax.atomistic.interchange.H5MDTrajectoryPlan(path, source_id=...,
sink_id=...)` opens the bounded H5MD reader by default and the writer when
`append` is supplied. The HDF5 dependency is loaded only at that I/O boundary;
stream manifests retain frame fields, units, committed-frame count, and source
or sink identity. `ExtendedXYZTrajectoryPlan` is the corresponding
text-trajectory plan. Both remain public through
`phydrax.atomistic.interchange`; they are not re-exported as alternate root
spellings.

## ASE structures

Public entry points are `from_ase_atoms`, `to_ase_atoms`,
`is_ase_available`, and `require_ase`.

`ASE_PARTICLE_ID_ARRAY` and `ASE_SOURCE_ID_INFO` name the reserved ASE array and info
field used for stable material-atom identity and source provenance.

## MDAnalysis

Public entry points are `atomistic_frame_from_mdanalysis`,
`atomistic_metadata_from_mdanalysis`, and `mdanalysis_selection`.

## Native ASE calculator

Both names resolve lazily from `phydrax.atomistic.interchange` and require ASE when
accessed. Stress is the tensile ASE Voigt vector `(xx, yy, zz, yz, xz, xy)` in eV/Å³;
an unavailable stress raises `PropertyNotImplementedError`.

The public lazy types are `NativeASECalculatorPlan` and
`NativeASECalculator`.

## i-PI

`IPITransportMode` is `"unix"` or `"tcp"`. `IPIVirialPolicy` is `"required"` or
`"optional"`; neither substitutes a zero virial. `IPIInverseCellPolicy` is `"verify"` or
`"ignore"`. `IPIListener.accept()` returns an `IPISession`. `IPITransportStatus` is one of
`READY`, `HAVE_DATA`, `CLOSED`, `PROTOCOL_ERROR`, or `PROVIDER_ERROR`. Wire values are
Hartree atomic units; the reply carries the configurational virial `W = -V σ`, never the
tensile stress `σ`.

The public i-PI surface is `IPITransportPlan`, `IPISession`, `IPIRequest`,
`serve_ipi_once`, and `TransportedExternalAtomisticProvider`.

## MACE checkpoint conversion

`MACE_PROVIDER_RELEASES` maps the admitted provider distributions to their exact
releases (`mace-torch` 0.3.16, `e3nn` 0.4.4). `MACESourceKind` is `"torch-state-dict"` or
`"torch-full-model"`; `MACEEvaluationDtype` is `"float32"` or `"float64"`. The provider
subprocess is a resource boundary, not a security sandbox. No checkpoint weights are
bundled and nothing is downloaded.

The public conversion/evaluation surface comprises `convert_mace_checkpoint`,
`MACESource`, `TrustedTorchPickleSource`, `MACEProviderRuntime`,
`MACEConversionLimits`, `MACECheckpointConversion`, `MACESourceProvenance`,
`MACESourceRefusedError`, `evaluate_mace_source`, `mace_source_gradients`,
`MACEProviderConfiguration`, `MACEProviderCase`, `MACEProviderEvaluation`,
`MACEProviderGradients`, `create_mace_provider_fixture`, and
`MACEProviderFixture`.

## External boundaries

External host boundaries are `PackmolAssemblyPlan`,
`AtomisticInterchangeBundle`, `AtomisticInterchangeReport`,
`from_openmm_system`, `from_openff_interchange`, `from_parmed_structure`,
`to_openmm_system`, and `to_openff_interchange`. Missing optional providers are
reported at invocation; no alternate provider is selected silently.
