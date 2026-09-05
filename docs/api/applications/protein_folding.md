# Protein folding applications

`phydrax.applications.protein_folding` is the public application surface for
protein identity, explicit physical handoff, experimental inference, conditional
rotamer energies, coordinate proposals, mixed-resolution mechanics, and
cotranslation. Numeric engines remain owned by the existing atomistic, PGM,
rigid, dynamics, stochastic, and UQ packages.

Read the [protein guide](../../guides_protein_folding.md) for scientific admission
and approximation limits and the [cookbook](../../cookbook/protein_folding.md) for
the source-pinned 1L2Y/Amber14 workflow. The [source disposition](../../biophysical_sources.md)
distinguishes conceptual references from licensed data, parameters, and weights.

## Public layout

```python
from phydrax.applications import protein_folding as protein

construct = protein.ProteinConstruct(("A",), ("NLYIQWLKDGGPSSGRPPPS",))
physical_binding = protein.bind_protein
joint_inference = protein.experiments.prepare_protein_experiments
paired_enthalpy = protein.thermodynamics.paired_state_enthalpy
conditional_rotamers = protein.potentials.RotamerFreeEnergyTerm
coordinate_training = protein.generation.fit_coordinate_model
mixed_mechanics = protein.hybrid.PreparedHybridModel
activation_protocol = protein.cotranslation.CotranslationProtocol
```

These are distinct contracts, not interchangeable routes to an accurate-folding
claim. Static hypotheses have no physical time; experimental conditions are
paired observation rows; MD has native model time; flow generation uses
pseudotime; cotranslation requires separate biological timing calibration.

## P1: construct, hypothesis, chemistry, binding, and geometry

`ResidueKey` is a zero-based construct-local sequence identity. `ProteinSourceAtom`
retains author/model/alternate-location/occupancy/source-row provenance beside the
chemical atom key. A `ProteinStructureHypothesis` is one static source model;
`ProteinHypothesisView` retains all originals while naming a selection policy.

The all-atom physical profile of `ResolvedProteinChemistry` is exactly
`canonical-L-single-chain-explicit`: uncapped, isolated, nonperiodic canonical-L
protein with explicit heavy atoms, hydrogens, residue chemical states, and
amine/carboxyl termini. It excludes solvent, ions, disulfides, caps, PTMs,
covalent ligands, virtual DOFs, and multichain physical binding. No pH-based
protonation assignment or atom completion occurs.

`bind_protein` requires a complete source-to-chemistry-to-native stable-ID map,
caller-prepared force field, exact parameter energy scale, and source/parameter
rights. `PreparedProteinBinding.evaluate` consumes a realized native
neighborhood state and returns conservative energy/forces, including fixed-atom
reactions. Geometry qualification checks explicit bond bounds, chirality,
clashes, peptide planarity, and finiteness, not experimental accuracy.

::: phydrax.applications.protein_folding
    options:
      members:
        - ProteinConstruct
        - ResidueKey
        - ProteinAtomKey
        - ProteinSourceAtom
        - ProteinStructureHypothesis
        - ProteinHypothesisView
        - ResolvedProteinChemistry
        - ProteinMappingCoverage
        - protein_mapping_coverage
        - PreparedProteinBinding
        - bind_protein
        - ProteinGeometryEvidence
        - PreparedProteinQualification

## P1: explicit interchange

`protein_hypothesis_from_pdb_records` consumes an explicitly selected neutral PDB
model and an author-residue-to-construct map. It preserves source identity even
when `canonical_atom_names` resolves naming differences.
`bind_protein_openmm` consumes an already parameterized OpenMM `System` and a
bijective particle-to-source-record map. OpenMM remains optional/lazy, and the
native converter's completeness report and bounded-NoCutoff warning are retained.
`accept_bounded_no_cutoff=True` is explicit approximation admission, not permission
to discard unsupported force terms.

::: phydrax.applications.protein_folding.interchange
    options:
      members:
        - ProteinOpenMMBinding
        - protein_hypothesis_from_pdb_records
        - bind_protein_openmm

## P1/P3: physical dynamics, free energy, and kinetic workflows

`prepare_protein_dynamics` and `run_protein_dynamics` accept native
`VelocityVerletPlan` (NVE) or `BAOABLangevinPlan` (NVT). The run requires qualified
initial geometry, explicit velocity units and PRNG key, and retains initial/final
geometry, rollout, ensemble label, bias identity, and separate trajectory
lineage. Check native rollout success, geometry success, and artifact status.
`ProteinDynamicsResult.trajectory_data()` preserves native time and sample masks.

`ProteinFreeEnergyWorkflow` composes native `fep`, `bar`, `ti`, `mbar`, and
`targeted` operations using explicitly identified states, composition,
temperature, ensemble/decorrelation evidence, and bias IDs. Inputs are physical
energies/work, not optimizer losses. Inspect the wrapped native estimator's
support, overlap, convergence, and uncertainty evidence.

`ProteinKineticWorkflow` exposes `vamp`, `vac`, `tica`, `markov`,
`chapman_kolmogorov`, and `committor`. It requires unbiased declared physical
`TrajectoryData`, matching physical time units, conditions, calibration evidence,
and valid uniform lags after resets/missing samples. Static predictions,
optimizer traces, MC sweeps, and configuration-biased trajectories cannot be
relabeled as unbiased kinetics. `ProteinBasinDefinitions` supplies independent
region predicates and first-passage basin pairs; committor fitting needs actual
physical shooting outcomes and censoring/source evidence.

::: phydrax.applications.protein_folding.workflows
    options:
      members:
        - ProteinDynamicsResult
        - prepare_protein_dynamics
        - run_protein_dynamics
        - ProteinFreeEnergyEstimate
        - ProteinFreeEnergyWorkflow
        - ProteinBasinDefinitions
        - ProteinKineticWorkflow

## P2: conditions and named observation models

`ThermodynamicConvention` separates molar energy/R from single-system energy/kB,
fixes reference temperature and standard concentration, and makes concentration
units explicit. `ExperimentConditions` uses paired finite rows, T > 0 Kelvin,
nonnegative denaturant/concentration, and positive total monomer concentration
for normalized dimer fluorescence. `celsius_to_kelvin` is the explicit offset
adapter.

The delivered equilibrium family is reversible monomer two-/three-state,
dimer two-/three-state, and heterogeneous open-chain repeat transfer models.
Thermal free energies use constant ΔCp, linear denaturant and a thermal m-value;
ΔG is unfolded minus folded. `repeat_transfer_statistics` instead accepts
formation energies where negative favors folding/contact and returns the
log partition and folded marginals. Kinetic models are isothermal at the
reference temperature and observe summed relaxation rates, not resolved
parallel pathways or general intermediate mechanisms.

::: phydrax.applications.protein_folding.experiments
    options:
      members:
        - ThermodynamicConvention
        - ExperimentConditions
        - celsius_to_kelvin
        - thermal_unfolding_free_energy
        - two_state_log_populations
        - dimer_log_populations
        - repeat_transfer_statistics
        - TwoStateUnfolding
        - ThreeStateUnfolding
        - DimerTwoStateUnfolding
        - DimerThreeStateUnfolding
        - RepeatTransferUnfolding
        - ChevronKinetics
        - ParallelPathKinetics

## P2: joint fitting, likelihood rank, and actual posterior sampling

`FluorescenceExperiment` carries group/state nuisance baselines, reversible
applicability evidence, calibrated Gaussian errors, masks, and an optional
active-row covariance Cholesky factor. `KineticRateExperiment` uses calibrated
Gaussian log-rate errors with an explicit rate time unit. Experimental sources
require rights and measured reference uncertainty; unknown uncertainty is `None`,
not zero. Irreversible/ramp-dependent/aggregating measurements are outside the
equilibrium contract.

Use `parameter_slots()` and explicit `bindings` to share named physical
parameters. `ExperimentParameter` scales free coordinates as initial + scale × z.
`prepare_protein_experiments` creates one joint residual/likelihood, and
`fit_protein_experiments` executes native nonlinear least squares for the entire
named model family. Inspect optimizer acceptance separately from
`ExperimentIdentifiability` and its likelihood-only null directions. Fit
covariance exists only for accepted full-rank fits and is not a credible interval.

`protein_experiment_posterior_problem` requires explicit Gaussian priors on free
z coordinates; `sample_protein_experiments` runs native NUTS and retains native
chain diagnostics. Priors do not erase likelihood rank deficiency or global
pathway symmetries. Posterior `predictive_samples()` contains conditional mean
draws, not measurement-noise draws. `phi_posterior` uses paired WT/mutant samples
and reports denominator-invalid draws, including NaN intervals when any draw is
invalid; Φ is not clipped to [0, 1].

::: phydrax.applications.protein_folding.experiments
    options:
      members:
        - ExperimentParameter
        - NamedParameterMap
        - FluorescenceExperiment
        - KineticRateExperiment
        - PreparedProteinObservation
        - PreparedProteinExperiments
        - prepare_protein_experiments
        - ExperimentIdentifiability
        - protein_experiment_identifiability
        - ProteinExperimentFit
        - fit_protein_experiments
        - protein_experiment_posterior_problem
        - ProteinExperimentPosterior
        - sample_protein_experiments
        - PhiPosterior
        - phi_posterior

## P3: paired-state enthalpy and experimental closure

`native_enthalpy_series` forms total H = U + K + pV using explicit pressure and
volume, preserving native physical time and validity. `ProteinEnsembleComposition`
compares the entire box, not just protein coordinates. `EnthalpyReplica` requires
equilibration/correlation evidence, uninterrupted physical time, complete blocks
spanning at least five correlation-time bounds, and independent realization IDs.

`paired_state_enthalpy` requires matched whole-box composition/pressure,
independently declared folded/unfolded basins, matching temperatures, and at least
two independent replicas per state/temperature. It reports unfolded-minus-folded
enthalpy and conditional sampling uncertainty. `fit_heat_capacity_slope` requires
at least three temperatures and retains residuals/covariance over the sampled
interval.

`close_free_energy_at_reference` requires measured `(T_ref, ΔG_ref)` and their
2 × 2 covariance, rights and uncertainty, with either
`measured-melting-temperature` (ΔG = 0) or `experimental-thermodynamic-cycle`
closure. A ligand Kd alone is insufficient. The closure point must be inside the
sampled interval; evaluation outside it is invalid/NaN. The result preserves its
experimental dependencies. Analytical controls do not satisfy real experimental
closure merely by using the same API.

::: phydrax.applications.protein_folding.thermodynamics
    options:
      members:
        - ProteinEnsembleComposition
        - EnthalpyReplica
        - PairedStateEnthalpyEstimate
        - HeatCapacitySlopeEstimate
        - ExperimentallyClosedFreeEnergy
        - native_enthalpy_series
        - paired_state_enthalpy
        - fit_heat_capacity_slope
        - close_free_energy_at_reference

## P4: conditional rotamer potential

`RotamerGeometryPlan` binds three-atom backbone frames and heterogeneous local
point states; `RotamerParameterPlan` supplies unary and fixed-pair Gaussian
energies in exact single-system units with rights and fixed temperature.
`RotamerFreeEnergyTerm` contributes G = −kBT ln Z to the native conservative
potential. Exact enumeration has a preparation cap; `bethe` uses implicit BP
with deterministic zero initialization and a sufficient loop-contraction gate.
Trees are exact when inference succeeds; loop normalizers remain Bethe
approximations even at a converged root.

`RotamerFreeEnergyEvaluation` distinguishes inference, geometry, branch
contraction, derivative qualification, and status. Failed geometry/root/branch
or nonfinite energy yields NaN energy and failed success. Differentiation applies
only on the accepted fixed-support branch. Attribution weights partition G and
do not define unique physical atom energies. These caller-defined single-site
states are not bundled calibrated rotamers or generated all-atom side chains.

::: phydrax.applications.protein_folding.potentials
    options:
      members:
        - RotamerGeometryPlan
        - RotamerParameterPlan
        - RotamerFreeEnergyTerm
        - PreparedRotamerFreeEnergyTerm
        - RotamerFreeEnergyEvaluation
        - RotamerFreeEnergyStatus

## G1: provider boundary and fixed-chemistry coordinate generation

`CoordinateProviderProvenance` separates output, weight, prepared-input and code
rights, explicit input lineage, and authorized egress. Offline import requires
raw source digest/license matching and retains every hypothesis and its
provider-specific confidence. Learned outputs require separate weight and
prepared-input authorization. No provider runtime, download, or automatic
restricted-output training is supplied.

`prepare_protein_coordinate_support` and `map_protein_hypothesis` bind protein
identity to fixed native atomistic support. The shared `PreparedCoordinateSupport`,
resource/geometry policies, training/model/sampler types and provider provenance
are public through both biological generation leaves, not through a universal
biomolecule module. Heavy-atom proposal support is not P1 all-atom chemistry.

Training requires cleared source manifests per record, named conditions,
nonempty disjoint caller-declared groups, and qualified conformers; canonical
duplicates across training/validation are refused. The dense model has a fixed
support/order ABI. Sampling retains raw and canonical positions, all failures,
solver status, geometry masks, sample IDs, and inherited rights. Generation
pseudotime is 0 to 1, never molecular time. Likelihood, Boltzmann weighting,
pretrained accuracy, and calibrated confidence are not available.

Save/load uses native pickle-free artifacts, checksum-bound weights, the same
support, and retained parent restrictions. Source availability does not grant
training, commercial, redistribution, or export permission.

::: phydrax.applications.protein_folding.generation
    options:
      members:
        - CoordinateProviderProvenance
        - ProteinProviderHypotheses
        - import_protein_hypotheses
        - CoordinateResourcePolicy
        - CoordinateGeometryPolicy
        - PreparedCoordinateSupport
        - prepare_protein_coordinate_support
        - map_protein_hypothesis
        - CoordinateTrainingData
        - prepare_coordinate_training_data
        - ConditionalCoordinateVelocity
        - CoordinateFitResult
        - fit_coordinate_model
        - PreparedCoordinateSampler
        - prepare_coordinate_sampler
        - CoordinateProposalBatch
        - sample_coordinate_proposals
        - CoordinateProposalQualification
        - qualify_coordinate_proposals
        - save_coordinate_model
        - load_coordinate_model

## X1: mixed protein–nucleotide mechanics

`PreparedHybridModel` combines an elastic protein network, native rigid nucleotide
model, and sparse cross-site soft steric/linker/screened-electrostatic terms in
one exact unit system. Parameter/reference rights are checked; periodicity,
Cartesian holonomic constraints, inactive/frame-only cross sites, and coincident
coupled sites are refused. Native Cartesian and rigid owners retain their own
state and stable IDs, with `HybridSupportMap` providing reversible disjoint
namespaces.

Energy is conservative with equal/opposite cross-site forces and native
force/torque pullback. `HybridForceEvaluation` retains full/mobile/reaction loads.
`step` returns an explicit no-heat-bath kick/drift/kick candidate and success
rather than silently admitting failed mechanics. Finite-step anisotropic rigid
rotation is not exact free-rotor flow or a biological calibration.

::: phydrax.applications.protein_folding.hybrid
    options:
      members:
        - HybridSupportMap
        - HybridCrossInteractionPlan
        - HybridState
        - HybridForceEvaluation
        - HybridStepResult
        - PreparedHybridModel

## X2: cotranslation, boundary forces, observations, and replay

`CotranslationProtocol` requires a complete one-bead-per-residue single-chain
schedule, stable capacity ordering, and per-epoch native dynamics with the active
nascent prefix plus persistent environment. Each later stage inserts one residue
with a matching RNA sense codon and explicit position/momentum, or switches
protocol at the same length with `codon=None`. Future material is dormant.

`RibosomeBoundaryPotential` is a source-declared harmonic tether/soft-sphere
boundary, not ribosome chemistry. Changes occur at native topology epochs with
insertion/work ledgers. Runs return separate stage series and a replayable
cursor; no lag/interpolation crosses an activation. Rejected activation retains
the preactivation state. Native checkpoint methods preserve protocol and epoch
identity.

Biological dwell timing requires `timing_calibration` with uncertainty plus
`timing_calibration_scope`; non-reference-conditioned claims require independent
`non_native_qualification`. `NascentChainObservations` retains future-contact
unavailability, geometric Gauss entanglement, separation and quadrature evidence.
Open-curve values are not integer linking numbers/knot classes, and intersecting
or degenerate curves fail.

::: phydrax.applications.protein_folding.cotranslation
    options:
      members:
        - CotranslationStage
        - CotranslationProtocol
        - CotranslationCursor
        - CotranslationRun
        - RibosomeBoundaryPotential
        - PreparedRibosomeBoundaryPotential
        - NascentChainObservations
        - NascentObservation
