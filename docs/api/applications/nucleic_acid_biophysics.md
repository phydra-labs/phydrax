# Nucleic-acid biophysics applications

`phydrax.applications.nucleic_acid_biophysics` owns explicit DNA/RNA constructs,
nucleotide-to-atom identity, source-preserving hypotheses and full base-interaction
graphs. Its named leaves own structural observations, chemical-mapping inference,
rigid-nucleotide mechanics, secondary-state kinetics, electronic dynamics and coordinate
generation. Generic atomistic, rigid-body, sampled-series, jump and quantum solvers
retain their existing public owners.

```python
from phydrax.applications import nucleic_acid_biophysics as nucleic

construct = nucleic.NucleicAcidConstruct(
    strand_ids=("sense", "antisense"),
    sequences=("ACGT", "ACGT"),
    polymer_types=("DNA", "DNA"),
    circular=(False, False),
)
first = nucleic.NucleotideKey("sense", 0)
directed_connectivity = construct.directed_edges
```

This declares chemistry and identity, not coordinates, charges, force-field parameters,
secondary pairing, or an electronic Hamiltonian. See the
[scientific guide](../../guides_nucleic_acid_biophysics.md) and the
[complete repository workflows](../../cookbook/nucleic_acid_biophysics.md).

## Construct, source and interaction contracts

Keys are strand-local zero-based identities. Only explicit canonical DNA/RNA is
admitted; directed 5′→3′ edges include declared circular closure. Atom mappings retain
stable atom IDs and coordinate coverage independently of inactive capacity.
Normalization changes only requested units/order, preserving raw hypotheses and lineage.
A full interaction graph can retain richer annotations than the restricted,
linear/canonical/one-partner/noncrossing `to_dot_bracket()` export.

::: phydrax.applications.nucleic_acid_biophysics.NucleotideKey

---

::: phydrax.applications.nucleic_acid_biophysics.NucleicAcidConstruct

---

::: phydrax.applications.nucleic_acid_biophysics.NucleotideAtomMapping

---

::: phydrax.applications.nucleic_acid_biophysics.PreparedNucleotideBinding

---

::: phydrax.applications.nucleic_acid_biophysics.prepare_nucleotide_binding

---

::: phydrax.applications.nucleic_acid_biophysics.NucleicStructureHypothesis

---

::: phydrax.applications.nucleic_acid_biophysics.NormalizedNucleicHypothesis

---

::: phydrax.applications.nucleic_acid_biophysics.normalize_nucleic_hypothesis

---

::: phydrax.applications.nucleic_acid_biophysics.NucleicRecordHypothesis

---

::: phydrax.applications.nucleic_acid_biophysics.nucleic_hypothesis_from_pdb_records

---

::: phydrax.applications.nucleic_acid_biophysics.BaseInteraction

---

::: phydrax.applications.nucleic_acid_biophysics.BaseInteractionGraph

## Structural observations (N1)

`structure` consumes explicit nonperiodic or already-unwrapped atom coordinates.
`NucleotideGDescriptor` converts declared lengths to Å and uses dimensionless ellipsoidal
coordinates. Published eRMSD uses `smooth_width=0`; a positive width is a distinct
C2-tapered descriptor. The squared directed-pair sum is normalized by nucleotide count.
A selected sparse support is not automatically full eRMSD. Coverage, validity and branch
margins qualify results; masked zeros are not observations.

Torsion outputs are α, β, γ, δ, ε, ζ, χ and ν₀–ν₄ in radians. Sugar pseudorotation is a
five-ring Fourier fit with retained harmonic residual, not an empirical conformer label.
Series methods preserve support, resets and missingness without inventing physical time.
Contact geometry is not a hydrogen-bond assignment, and ring/backbone qualification is
not complete physical validity.

::: phydrax.applications.nucleic_acid_biophysics.structure.base_frames

---

::: phydrax.applications.nucleic_acid_biophysics.structure.BaseFrameEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.NucleotideGDescriptor

---

::: phydrax.applications.nucleic_acid_biophysics.structure.GFeatureEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.ERMSDEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.ERMSDCollectiveVariableProgram

---

::: phydrax.applications.nucleic_acid_biophysics.structure.NucleotideTorsionProgram

---

::: phydrax.applications.nucleic_acid_biophysics.structure.NucleotideTorsionEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.sugar_pseudorotation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.SugarPseudorotationEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.GeometricContactCriteria

---

::: phydrax.applications.nucleic_acid_biophysics.structure.GeometricContactEvaluation

---

::: phydrax.applications.nucleic_acid_biophysics.structure.geometric_contacts

---

::: phydrax.applications.nucleic_acid_biophysics.structure.contact_interaction_graph

---

::: phydrax.applications.nucleic_acid_biophysics.structure.NucleotideStructureQualifier

---

::: phydrax.applications.nucleic_acid_biophysics.structure.NucleotideStructureQualification

## Chemical mapping and distance reconstruction (N2)

`observations` retains processed reactivities, measured SDs, masks, replicate/condition
identity and source rights. Optional covariance factors describe observed rows only;
SD-only input declares a diagonal approximation. Negative corrected values are valid.
The RDAT importer supports an explicit 0.34 profile, retains source bytes and per-row
mutant hypotheses, and requires explicit standard-deviation semantics for supplied
errors. Unsupported fields are refused.

The affine accessibility law fits one reagent/preprocessing profile with declared
baseline sharing and condition effects. Accessibility is a supplied hypothesis feature,
not pairing inferred from an assay. Fit success, identifiability and held-out predictive
performance are separate. Distance reconstruction consumes separately admitted
intervals and an existing atomistic support; reflection/chirality qualification and
local optimizer success are separate from physical validity.

::: phydrax.applications.nucleic_acid_biophysics.observations.ChemicalMappingCondition

---

::: phydrax.applications.nucleic_acid_biophysics.observations.ChemicalMappingObservation

---

::: phydrax.applications.nucleic_acid_biophysics.observations.AccessibilityReactivityModel

---

::: phydrax.applications.nucleic_acid_biophysics.observations.ChemicalMappingFit

---

::: phydrax.applications.nucleic_acid_biophysics.observations.import_processed_rdat

---

::: phydrax.applications.nucleic_acid_biophysics.observations.ProcessedRDAT

---

::: phydrax.applications.nucleic_acid_biophysics.observations.ProcessedRDATEntry

---

::: phydrax.applications.nucleic_acid_biophysics.observations.IntervalDistanceReconstruction

---

::: phydrax.applications.nucleic_acid_biophysics.observations.IntervalReconstructionResult

---

::: phydrax.applications.nucleic_acid_biophysics.observations.ChiralityEvaluation

## Rigid-nucleotide mechanics (N3)

`coarse` admits caller-cleared exact coefficient bytes for `average-dna`,
`groove-salt-dna`, `sequence-dna`, `rna`, or `dna-rna-hybrid`. No calibrated tables are
bundled. Each family requires its full geometry/interaction profile, sequence strengths
and conditions; hybrid coefficients are not inferred by averaging. Salt is mol/litre;
temperature, energy, length, masses, inertia and time follow the declared
`AtomisticUnitSystem`.

Five physical sites and three differential frame markers belong to each stable rigid
body. `evaluate` preserves mobile and fixed-body reaction loads; `step` and `heat_bath`
use the generic rigid machinery. Periodic support is fully orthorhombic, with unwrapped
bonded geometry and one unique COM image per nonbonded pair. Mechanical trajectories
and thermalization are not an experimentally calibrated folding clock.

::: phydrax.applications.nucleic_acid_biophysics.coarse.NucleotideParameterArtifact

---

::: phydrax.applications.nucleic_acid_biophysics.coarse.nucleotide_reference_sites

---

::: phydrax.applications.nucleic_acid_biophysics.coarse.NucleotideModelPlan

---

::: phydrax.applications.nucleic_acid_biophysics.coarse.PreparedNucleotideModel

---

::: phydrax.applications.nucleic_acid_biophysics.coarse.NucleotideForceEvaluation

## Secondary-state CTMCs (N4)

`secondary_kinetics` compiles exhaustive bounded, linear, labelled-strand,
pseudoknot-free states. `pair_loop` and `nearest_neighbor_loop` use admitted dimensionless
molar G/(RT) tables at one temperature. Standard-state and fixed-volume association
are distinct. Named rate laws supply independent inverse-time prefactors; only the
fixed-volume `association_metropolis` elementary join has the stated dilute
bimolecular conversion. Capacity excess refuses rather than truncating legal states.

`PreparedSecondaryKinetics.process` composes with `phydrax.solver.solve_direct_ssa`.
`phydrax.solver.event_first_hit` scans event post-states and separates exact hits,
right-censored successful non-hits, and incomplete failed non-hits.
`phydrax.solver.finite_generator_hitting` requires closed support and retains unreachable
or non-almost-sure targets with zero hitting probability or infinite unconditional MFPT
as appropriate. These generic solvers are not re-exported from the application leaf.

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.StrandComplexPartition

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.SecondaryStructureState

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.SecondaryMove

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.AssociationConvention

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.SecondaryEnergyModel

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.SecondaryRateLaw

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.prepare_secondary_kinetics

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.PreparedSecondaryKinetics

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.SecondaryJumpProcess

---

::: phydrax.applications.nucleic_acid_biophysics.secondary_kinetics.CompiledSecondaryTarget

## Electronic states and dynamics (N5)

`electronics` binds orbital/site IDs, not atom IDs. Parameters are source-pinned
single-system energies, Hermitian couplings, orbital gauge and explicit Markovian
channels. `hamiltonian` is already H/ℏ in inverse native time. Dimensionless `jumps`
retain separate inverse-time `rates`; `collapse_operators` applies √rate exactly once.
No cyclic-frequency factor or second rate multiplication belongs in the solver handoff.

Dense Lindblad and finite-CPTP density routes are distinct from native fixed-step
quantum trajectories, which are not event-exact SSA. Electron–hole support and vacuum
must be explicit. Reduced surviving-carrier densities are not renormalized after loss.
Populations/coherences are electronic observables, not atom charges or lesion yields.
All source/structure/weight-use restrictions and finite resource bounds remain active.

::: phydrax.applications.nucleic_acid_biophysics.electronics.ElectronicSiteGraph

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.ElectronicChannel

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.ElectronicParameterArtifact

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.prepare_electronics

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.prepare_electron_hole

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.PreparedElectronicModel

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.evolve_electronics

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.ElectronicEvolution

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.evolve_electronic_jumps

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.ElectronicJumpEvolution

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.electronic_reduced_density

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.electronic_populations

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.electronic_coherences

---

::: phydrax.applications.nucleic_acid_biophysics.electronics.nucleotide_electronic_populations

## Coordinate proposals and provider admission

`generation` publicly exposes shared coordinate contracts alongside DNA/RNA-specific
mapping and provider functions. Import these types here (or through the protein
generation leaf), not through private implementation paths.

Provider import is offline admission, not provider execution or checkpoint download.
Output, input and learned-weight rights and authorized egress remain separate.
Native training consumes actual fixed-support, geometry-qualified conformers with
caller-declared disjoint groups. Support/order/chemistry/condition features are ABI.
The model is not permutation equivariant and does not complete missing material atoms.

Training/persistence and all-sample materialization are host operations. The learned
velocity and prepared sampler are numeric PyTrees; pseudotime 0–1 is not molecular time.
Proposal records retain raw/canonical coordinates, all solver statuses and per-sample
geometry checks without hidden rejection or repair. No independent pretrained weights,
calibrated confidence, coordinate likelihood or Boltzmann measure is supplied.

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateProviderProvenance

---

::: phydrax.applications.nucleic_acid_biophysics.generation.NucleicProviderHypotheses

---

::: phydrax.applications.nucleic_acid_biophysics.generation.import_nucleic_hypotheses

---

::: phydrax.applications.nucleic_acid_biophysics.generation.prepare_nucleic_coordinate_support

---

::: phydrax.applications.nucleic_acid_biophysics.generation.map_nucleic_hypothesis

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateResourcePolicy

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateGeometryPolicy

---

::: phydrax.applications.nucleic_acid_biophysics.generation.PreparedCoordinateSupport

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateProposalQualification

---

::: phydrax.applications.nucleic_acid_biophysics.generation.qualify_coordinate_proposals

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateTrainingData

---

::: phydrax.applications.nucleic_acid_biophysics.generation.prepare_coordinate_training_data

---

::: phydrax.applications.nucleic_acid_biophysics.generation.ConditionalCoordinateVelocity

---

::: phydrax.applications.nucleic_acid_biophysics.generation.fit_coordinate_model

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateFitResult

---

::: phydrax.applications.nucleic_acid_biophysics.generation.prepare_coordinate_sampler

---

::: phydrax.applications.nucleic_acid_biophysics.generation.PreparedCoordinateSampler

---

::: phydrax.applications.nucleic_acid_biophysics.generation.sample_coordinate_proposals

---

::: phydrax.applications.nucleic_acid_biophysics.generation.CoordinateProposalBatch

---

::: phydrax.applications.nucleic_acid_biophysics.generation.save_coordinate_model

---

::: phydrax.applications.nucleic_acid_biophysics.generation.load_coordinate_model

## Qualification and runnable evidence

The [cookbook](../../cookbook/nucleic_acid_biophysics.md) includes an actual retained-RMDB
fit with whole-construct holdout and separately runnable structural, mechanical, CTMC,
and electronic numerical benchmarks. Their outputs have different scientific meanings.
Numerical success does not supply absent coefficient calibration, measured uncertainty,
experimental time scales, or pretrained-generation accuracy. See the
[source dispositions](../../biophysical_sources.md) for provenance and rights boundaries.
