# Atomistic learning and dynamics

`phydrax.atomistic` owns finite and periodic atomistic learning, prediction, training,
conservative dynamics, and native model artifacts. `phydrax.nn.atomistic` owns the
PaiNN, low-degree Cartesian NequIP, and standard MACE model architectures together with
MACE's radial, symmetric-contraction, preparation, and accelerated-coupling owners.
Material particles, sparse and streamed relations, image neighborhoods, graph IR,
precision, replay, and qualification remain shared native substrates.

Classical force fields, interaction sites, trajectory interoperability, MACE
checkpoint conversion, ASE and i-PI deployment, enhanced sampling, uncertainty,
advanced ensembles, polarization, quantum nuclei, many-body methods, and owner-local
distributed execution are documented on the split
[force-field](atomistic_force_fields.md), [interoperability](atomistic_interop.md),
[sampling](atomistic_sampling.md), and [advanced-method](atomistic_advanced.md) API pages.
Image-aware particle neighborhoods are on the
[particle API page](discretization/particle.md), and frozen atomistic IREE export is on
the [export API page](export.md). The [atomistic learning guide](../guides_atomistic.md)
and [native MACE execution guide](../guides_mace_execution.md) describe the contracts.
Every MACE route is an unreleased candidate.

## Structures, units, systems, and topology

::: phydrax.atomistic.AtomisticScaleContract

::: phydrax.atomistic.AtomisticUnitSystem

::: phydrax.atomistic.AtomicStructure

::: phydrax.atomistic.AtomisticBatch

::: phydrax.atomistic.AtomisticSystemPlan

::: phydrax.atomistic.PreparedAtomisticSystem

::: phydrax.atomistic.MolecularTopologyPlan

::: phydrax.atomistic.PreparedMolecularTopology

## Graph topology and realization

Topology preparation is a host boundary over concrete positions and cells. It
discovers directed routes, explicit integer image shifts, stable receiver-major route
IDs, and the streamed schedule once per topology epoch. Binding only computes
`d_e = x[receiver] - x[sender] + n_e @ H[case_e]`, cutoff masks, and certificates, and it
is safe under `jit` and differentiation. `AtomisticGraphExecutionPlan.backend` is
`"dense"` (the bounded named reference) or `"particle"` (case-batched cell-list image
search).

::: phydrax.atomistic.AtomisticGraphExecutionPlan

::: phydrax.atomistic.AtomisticGraphTopology

::: phydrax.atomistic.prepare_atomistic_graph_topology

::: phydrax.atomistic.particle_atomistic_graph_topology

::: phydrax.atomistic.bind_atomistic_graph

::: phydrax.atomistic.AtomisticGraph

::: phydrax.atomistic.realize_atomistic_graph

::: phydrax.atomistic.realize_particle_atomistic_graph

## Potentials, prediction, and stress

::: phydrax.atomistic.AbstractAtomisticPotential

::: phydrax.atomistic.AbstractPreparedAtomisticPotential

::: phydrax.atomistic.AtomisticPotentialCapabilities

::: phydrax.atomistic.atomistic_potential_revision

::: phydrax.nn.atomistic.PaiNNPotential

::: phydrax.nn.atomistic.NequIPPotential

::: phydrax.atomistic.AtomisticPrediction

::: phydrax.atomistic.AtomisticProvenance

::: phydrax.atomistic.energy_and_forces

::: phydrax.atomistic.AtomisticEnergyDerivatives

::: phydrax.atomistic.atomistic_energy_derivatives

::: phydrax.atomistic.AtomisticStressConvention

## Native MACE

`MACEArchitecture.interactions` admits `"real-agnostic"`, `"real-agnostic-residual"`,
`"real-agnostic-density"`, and `"real-agnostic-density-residual"`. Other literal
selectors:

- `distance_transform`: `"none"` or `"agnesi"`;
- `cutoff_placement`: `"embedding"` or `"weights"`;
- `energy_scaling`: `"unscaled"` or `"scale-shift"`;
- `prepare_mace_potential(radial=...)`: `"exact"` or `"tabulated"`.

`MACEPotential` is trainable. `PreparedMACEPotential` is inference-only and bound to
one parameter revision. Acceleration is an execution policy that changes neither
`architecture_id` nor the numeric revision.

::: phydrax.nn.atomistic.MACEArchitecture

::: phydrax.nn.atomistic.MACEPotential

::: phydrax.nn.atomistic.MACELayerUpdate

::: phydrax.nn.atomistic.MACESourceRows

::: phydrax.nn.atomistic.mace_potential_from_source

::: phydrax.nn.atomistic.prepare_mace_potential

::: phydrax.nn.atomistic.PreparedMACEPotential

::: phydrax.nn.atomistic.StaleMACEPreparation

### Radial realization and tables

`RadialMLP` postprocessing is `"none"` or `"tanh-square"`. Table layouts are
`"projected-width"` or `"embedding-width"`, and table objectives are
`"minimum-table-bytes"` or `"minimum-edge-work"`. Tables admit first coordinate
derivatives only; Hessians and force/stress training use the exact radial network.

::: phydrax.nn.atomistic.BesselRadialBasis

::: phydrax.nn.atomistic.PolynomialCutoff

::: phydrax.nn.atomistic.AgnesiTransform

::: phydrax.nn.atomistic.RadialEmbedding

::: phydrax.nn.atomistic.RadialMLP

::: phydrax.nn.atomistic.normalized_silu_scale

::: phydrax.nn.atomistic.RadialTableDeclaration

::: phydrax.nn.atomistic.RadialSpeciesBinding

::: phydrax.nn.atomistic.prepare_radial_tables

::: phydrax.nn.atomistic.PreparedRadialTables

::: phydrax.nn.atomistic.radial_source_revision

::: phydrax.nn.atomistic.radial_table_resources

::: phydrax.nn.atomistic.RadialTableResources

::: phydrax.nn.atomistic.select_radial_table_layout

::: phydrax.nn.atomistic.RadialTableLayoutChoice

::: phydrax.nn.atomistic.RadialTableQualificationPolicy

::: phydrax.nn.atomistic.qualify_radial_tables

::: phydrax.nn.atomistic.RadialTableQualification

::: phydrax.nn.atomistic.RadialTableQualificationError

::: phydrax.nn.atomistic.StaleRadialTableBinding

### Symmetric contraction

A `SymmetricContractionBasis` origin is `"native-coupling"` or `"imported-source"`.
The trainable contraction keeps the original weights `W`, and merged coefficients are
fixed data bound to one `W` revision.

::: phydrax.nn.atomistic.SymmetricContractionBasis

::: phydrax.nn.atomistic.SymmetricContractionPlan

::: phydrax.nn.atomistic.SymmetricContraction

::: phydrax.nn.atomistic.symmetric_contraction_revision

::: phydrax.nn.atomistic.MergedSymmetricContraction

::: phydrax.nn.atomistic.StaleSymmetricContractionBinding

### Accelerated edge coupling

Target admission (`"cuda"` with compute capability at least 9.0, or the
`"cpu_interpret"` Mosaic GPU interpreter used only for semantic reference
verification) is owned by the [backend API](backends.md). No CUDA hardware
qualification has been recorded.

::: phydrax.nn.atomistic.MACEKernelPlan

::: phydrax.nn.atomistic.MACEAcceleratedCoupling

::: phydrax.nn.atomistic.MACEEdgeCouplingSpec

::: phydrax.nn.atomistic.MACECouplingRow

::: phydrax.nn.atomistic.MACELayoutBlock

::: phydrax.nn.atomistic.MACEFragmentExtent

::: phydrax.nn.atomistic.fragment_extent

::: phydrax.nn.atomistic.require_fragment_routing

::: phydrax.nn.atomistic.MACEFragmentResources

::: phydrax.nn.atomistic.MACEKernelEvidence

::: phydrax.nn.atomistic.MACECouplingDerivativeError

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

::: phydrax.atomistic.retry_atomistic_step_with_capacity

::: phydrax.atomistic.ThermodynamicAccumulator
::: phydrax.atomistic.AtomisticPhaseSpaceMeasurePlan

::: phydrax.atomistic.AtomisticThermodynamicStatePlan

::: phydrax.atomistic.PreparedThermodynamicStateTable


::: phydrax.atomistic.RadialDistributionPlan

## Rollout, replay, checkpoints, restarts, and stress

A runtime checkpoint requires an explicitly matching prepared model. A restart
additionally bundles the pickle-free native model artifact for fresh-process
continuation.

::: phydrax.atomistic.AtomisticTrajectoryPlan

::: phydrax.atomistic.AtomisticRolloutPlan

::: phydrax.atomistic.AtomisticReplayPolicy

::: phydrax.atomistic.AtomisticCheckpointPlan

::: phydrax.atomistic.write_atomistic_checkpoint

::: phydrax.atomistic.read_atomistic_checkpoint

::: phydrax.atomistic.write_atomistic_restart

::: phydrax.atomistic.read_atomistic_restart_model

::: phydrax.atomistic.read_atomistic_restart

::: phydrax.atomistic.atomistic_cell_energy_and_stress

::: phydrax.atomistic.IsotropicMonteCarloBarostatPlan

## Native model artifacts

Archives are bounded by `ATOMISTIC_MODEL_ARTIFACT_LIMITS` before allocation.
Restoring an archive reruns the registered model's scientific validators and
recomputes every identity. No optional provider package is imported.

::: phydrax.atomistic.write_atomistic_model_artifact

::: phydrax.atomistic.read_atomistic_model_artifact

::: phydrax.atomistic.AtomisticModelArtifact

::: phydrax.atomistic.AtomisticModelArtifactManifest

::: phydrax.atomistic.AtomisticModelArtifactError

::: phydrax.atomistic.atomistic_model_identity

::: phydrax.atomistic.register_atomistic_model_artifact

## Controlled Hamiltonians, providers, and specialized methods

`NativeAtomisticProviderPlan` prepares one learned model as a native provider. Its
neighborhood is an image-aware Verlet cache for periodic systems or a Verlet cache
over the declared finite neighborhood otherwise. Energy, forces, and stress come
from one evaluation of the same scalar energy. Stress is available exactly when
the system is a fully periodic 3D cell and every program term owns cell
derivatives.

::: phydrax.atomistic.AlchemicalControlSchedulePlan

::: phydrax.atomistic.AlchemicalInteractionPartitionPlan

::: phydrax.atomistic.ControlledHamiltonianPlan

::: phydrax.atomistic.PreparedControlledHamiltonian

::: phydrax.atomistic.RegionMaskedPotential

::: phydrax.atomistic.RESPAPlan

::: phydrax.atomistic.AbstractExternalAtomisticProvider

::: phydrax.atomistic.NativeAtomisticProviderPlan

::: phydrax.atomistic.NativeAtomisticProvider

::: phydrax.atomistic.NativeAtomisticRequest

::: phydrax.atomistic.NativeAtomisticEvaluation

::: phydrax.atomistic.BornOppenheimerVelocityVerletPlan

::: phydrax.atomistic.RingPolymerPlan

::: phydrax.atomistic.PreparedRingPolymerDynamics

::: phydrax.atomistic.VarianceConstrainedSemiGrandPlan

## Typed training and rMD17

`fit_atomistic_potential` runs every full-batch Adam update as one attempt of the
shared accepted-update training kernel, with `MODEL` root authority and one
energy/force/stress data-fit objective. Each training and validation split is an
`AtomisticSupervisionSplit` with role `"training"` or `"validation"`. A split binds
its batch to a frozen candidate topology. The problem prepares that topology from
`cutoff` and `skin`, or the caller supplies it.

A nonfinite training loss or gradient, including capacity overflow, rolls the
attempt back. An update whose post-update training loss is nonfinite is discarded
too. Either way the run ends with `AtomisticStatus.NONFINITE`, and `potential` is
the last finite accepted state.

`AtomisticTrainingResult.training_state` holds the committed kernel state:
parameters, Adam state, root key, and cursors. A continuation resumes it after
checking the problem, policy, normalization, capability, and kernel identities.
Training checkpoints and restarts persist that state without pickle.

::: phydrax.atomistic.AtomisticSupervisionSplit

::: phydrax.atomistic.AtomisticTrainingProblem

::: phydrax.atomistic.AtomisticTrainingPolicy

::: phydrax.atomistic.AtomisticTrainingNormalization

::: phydrax.atomistic.AtomisticTrainingResult

::: phydrax.atomistic.fit_atomistic_potential

::: phydrax.atomistic.write_atomistic_training_checkpoint

::: phydrax.atomistic.read_atomistic_training_checkpoint

::: phydrax.atomistic.write_atomistic_training_restart

::: phydrax.atomistic.read_atomistic_training_restart

::: phydrax.atomistic.RMD17Dataset

::: phydrax.atomistic.load_rmd17_npz

::: phydrax.atomistic.split_rmd17

## Qualification

::: phydrax.atomistic.AtomisticDynamicsQualificationClaim

::: phydrax.atomistic.AtomisticDynamicsQualificationProfile

::: phydrax.atomistic.AtomisticDynamicsQualificationResult
