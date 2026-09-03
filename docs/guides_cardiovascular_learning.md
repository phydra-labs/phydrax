# Cardiovascular cohorts, random fields, and learned proposals

The cardiovascular learning layer is an adapter over PhydraX ROM, UQ, stochastic,
and neural-operator substrates. It does not define another dataset, covariance,
operator-training, geometry, or solver framework. Its governing rule is that a
learned value is a proposal only. Scientific output is accepted only after a
complete native electrophysiology, mechanics, circulation, and hemodynamics
reanalysis.

## Authoritative truth cohorts

`CardiovascularTruthCase` binds a `DeidentifiedCohortIdentity`, site grouping,
`CardiovascularCaseManifest`, fixed topology, parameter record, probability mass,
operator input/target pair, ROM `TruthSample`, and authoritative execution-manifest
identity. The cohort identity requires deidentification policy and receipt IDs and
rejects PHI/linkable markers. A case with any status other than
`CohortCaseStatus.COMPLETE` cannot carry training arrays or truth.

`batch_fixed_topology_cohort` requires every complete case to have the same topology
and exact tensor layout. It delegates collation to `nn.operator` and stores
conditional valid-case weights in the resulting `OperatorDataset`. Invalid cases
are not silently renormalized away: `valid_probability`, `invalid_probability`, and
`invalid_probability_by_status` retain their mass under the original cohort law.
`adapt_complete_truth_to_rom` exposes the same complete cases through the existing
`ROMCorpus` contract without invoking or replacing the truth solver.

## Leakage-safe splits and preprocessing

The split routes are distinct policy types:

- `SubjectSplitPolicy` keeps all cases for a subject in one partition.
- `SiteSplitPolicy` reserves named sites as an external-site OOD test partition and
  also prevents a subject from crossing the retained/held-out boundary.
- `OODSplitPolicy` reserves explicitly tagged phenotypes or acquisition regimes
  before splitting the remaining subjects.

Partition ordering is a stable content hash of the seed and deidentified cohort
identity. `split_cardiovascular_cohort` therefore produces the same split on every
replay and records a content-sensitive `split_id`. `prepare_learning_cohort`
revalidates subject disjointness even for manually constructed splits before
touching any arrays.

Preparation fits `OperatorNormalizationPolicy` only from the training operator
batch. Its `TrainOnlyFeaturePreprocessor` likewise fits parameter location,
regularized `DenseCovariance`, scales, and the Mahalanobis support boundary only
from `split.train_ids`. Calibration, interpolation-test, and OOD arrays never enter
those statistics.

```python
split = split_cardiovascular_cohort(
    cases,
    OODSplitPolicy(("rare-geometry",), seed=17),
)
prepared = prepare_learning_cohort(cases, split)

assert prepared.features.training_case_ids == split.train_ids
assert prepared.ood_test is not None
print(prepared.cohort.invalid_probability)
```

## Canonical-coordinate random fields

`CanonicalCardiacCoordinates` separates mesh identity and physical quadrature from
canonical coordinates such as transmural, apicobasal, and circumferential position.
A `CardiacRandomFieldRecipe` constructs a Matérn-3/2 covariance in those coordinates,
solves the weighted Karhunen–Loève problem with the native dense self-adjoint
eigensolver, and creates a stochastic `SpatialBasisSynthesis` /
`StaticGaussianRandomField`. This gives one canonical latent recipe across registered
fixed-topology cases rather than correlating raw scanner coordinates.

Every draw is a `GaussianCoefficientRealization` with stable mode and coupling IDs.
Reusing it gives bit-exact replay. `CanonicalRandomField.diagnostics` delegates to
the stochastic Gaussian-field diagnostics for latent covariance, nodal variance,
and replay evidence. Use `PositiveExponentialFieldTransform` for positive
coefficients or `BoundedLogisticFieldTransform` for explicitly bounded fields; the
transform and kernel-unit `CardiovascularQuantitySpec` are part of recipe identity.

```python
coordinates = CanonicalCardiacCoordinates(
    canonical_points,
    quadrature_weights,
    (
        CanonicalCoordinateAxis("transmural", 0.0, 1.0),
        CanonicalCoordinateAxis("apicobasal", 0.0, 1.0),
    ),
    topology_id,
)
field = CardiacRandomFieldRecipe(
    "conductivity-heterogeneity",
    cardiovascular_quantity("electrical_conductivity"),
    latent_mean,
    latent_standard_deviation,
    (0.25, 0.4),
    rank=24,
).instantiate(coordinates)
realization = field.realize(key, sample_count=1024)
samples = field.sample(realization)
evidence = field.diagnostics(realization)
```

## Calibrated proposal and refusal

`CardiacSurrogateCalibration.fit` accepts a `PreparedLearningCohort` and requires
its arrays to match that prepared calibration partition exactly. It uses the native
`GaussianScaleCalibrator` for predictive-scale correction and `SplitConformal` for
a finite-sample simultaneous standardized residual radius. A
`CardiacSurrogateProposalManifest` binds the operator artifact and contract, truth
corpus, prepared split, train-only preprocessing, calibration, topology, and output
quantity specifications.

Call `assess_surrogate_input` before operator inference. It checks manifest,
topology, and train-fitted Mahalanobis support without requiring a prediction, so
an OOD case can be refused without evaluating the learned operator at all.

`propose_cardiac_surrogate` refuses before reanalysis when any of these conditions
holds:

1. artifact, preprocessing, calibration, split, or topology identities disagree;
2. the parameter Mahalanobis distance exceeds the train-fitted support boundary;
3. output or scale is non-finite, empty, non-positive, or shape-inconsistent;
4. the calibrated conformal interval half-width exceeds the declared limit; or
5. required generated anatomy/motion evidence is absent or invalid.

Even `SurrogateProposalStatus.QUALIFIED_FOR_REANALYSIS` does not mean accepted:
`CardiacSurrogateProposal.accepted` is always false.

## Generated anatomy and motion

`FixedTopologyReferenceGeometry` binds reference coordinates and line, surface
triangle, or volume-simplex connectivity. A `GenerativeGeometryCandidate` may supply
new coordinates and a fixed sequence of motion frames, but never new connectivity.
`qualify_generative_geometry` checks exact topology/layout identity, finite values,
relative cell orientation and measure, displacement from the reference, and
frame-to-frame motion increments. The resulting `GeometryQualificationEvidence`
can only qualify the candidate as a reanalysis input; it cannot make generated
anatomy authoritative.

## Mandatory full native reanalysis

A `FullNativeReanalysisPlan` contains four concrete route types rather than a string
fidelity mode:

- `ElectrophysiologyReanalysisRoute`
- `MechanicsReanalysisRoute`
- `CirculationReanalysisRoute`
- `HemodynamicsReanalysisRoute`

`FullNativeReanalysisRequest.from_proposal` accepts only a calibrated, in-support,
geometry-qualified proposal. The predicted state is recorded solely as
`initial_guess`. The callback passed to `run_full_native_reanalysis` must return a
`NativeReanalysisCandidate` containing all declared output quantities, fixed
topology, and one `NativeDomainSolveReceipt` per route. Receipts accept only exact
successful native result types: `AlievPanfilovCandidate`, `HyperelasticResponse`,
`ConsistentInitializationResult`, and `FixedWallLBMCandidate`. Each receipt also
binds a concrete `CardiovascularExecutionManifest`; domain-result, output-artifact,
and receipt identities are derived from the native result contents.

Acceptance fails closed on missing receipts, route/solver/execution-manifest
mismatch, unsuccessful native domain evidence, topology or quantity mismatch, and
non-finite output. `FullNativeReanalysisResult.accepted_fields` is `None` on every
failure and never references the learned initial guess.

## Qualification and performance

Run the focused scientific qualification campaign from the repository root:

```text
python tools/cardiovascular_learning_qualification.py
```

The campaign checks deterministic splitting, train-only fitting, invalid probability
mass, covariance and replay, geometry qualification, and calibrated OOD refusal.
It executes a phenomenological EP reaction, passive hyperelastic response,
circulation DAE initialization, and fixed-wall LBM candidate; only their exact
successful result receipts permit full native reanalysis acceptance.

The benchmark reports KL preparation, random-field sampling throughput, cohort
preparation, geometry qualification, OOD assessment, and native acceptance-gate
overhead:

```text
python benchmarks/cardiovascular_learning.py --point-counts 16 64 --sample-count 2048
```
