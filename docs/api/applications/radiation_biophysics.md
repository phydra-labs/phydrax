# Radiation biophysics applications

`phydrax.applications.radiation_biophysics` owns source-linked initial-lesion
analysis (R1) and fixed-candidate two-cause calibration with staged qualification
(R2). Transport, spatial chemistry, repair, survival, and clinical dose response
are not native application models. Host import, geometry mapping, thresholding,
realization, and clustering are separate from the differentiable fixed-support
expected-lesion map.

Read the [guide](../../guides_radiation_biophysics.md) for the exact source schema,
units, refusals, dose denominators, and scientific gates, or run the
[cookbook](../../cookbook/radiation_biophysics.md). Provider/data/parameter rights
remain governed by retained artifacts; see
[biophysical source dispositions](../../biophysical_sources.md).

## External source and raw identities

Primary identities include source artifact, run, primary, and dose fraction.
Physical deposition, kinetic-energy loss, and carried energy are distinct fields;
unreported values remain `None`. The source's rights tuple retains governing
parents and its configuration references must address real retained artifacts.

::: phydrax.applications.radiation_biophysics.RadiationSource

---

::: phydrax.applications.radiation_biophysics.PrimaryHistoryKey

---

::: phydrax.applications.radiation_biophysics.RadiationEventKey

---

::: phydrax.applications.radiation_biophysics.PhysicalInteraction

---

::: phydrax.applications.radiation_biophysics.InteractionLedger

---

::: phydrax.applications.radiation_biophysics.ChemicalReaction

---

::: phydrax.applications.radiation_biophysics.ReactionLedger

## Pinned interchange

The only implemented external reader profile is
`DNADAMAGE1_PROFILE = "Geant4-dnadamage1-ROOT-11.3.0"`, with
`DNADAMAGE1_REVISION = "v11.3.0"`. `NANOMETER` is its exact length unit; energy is eV.
The column entrypoint requires original tree-entry IDs and an admitted canonical
column payload. The binary reader verifies retained ROOT bytes and loads `uproot`
only at invocation. The `radiation-interop` extra installs `uproot>=5.6,<6`; no
optional ROOT library is imported by the application root.

Both readers return declared-loss evidence: the pinned ntuples omit event time,
track/parent/process/physical-species/carried-energy semantics and record only
selected OH/deoxyribose damage reactions. Required omitted semantics cause refusal.
A real reader and synthetic format fixtures do not qualify a real transport or
radiolysis campaign.

::: phydrax.applications.radiation_biophysics.interchange.ImportedRadiationLedgers

---

::: phydrax.applications.radiation_biophysics.interchange.dnadamage1_column_payload

---

::: phydrax.applications.radiation_biophysics.interchange.import_dnadamage1_columns

---

::: phydrax.applications.radiation_biophysics.interchange.import_dnadamage1_root

## Derived scoring targets

Target IDs are neither source volume/copy IDs nor atom IDs. Explicit source routes
allocate deposition; coordinate routes use transformed scoring spheres. Topology
requires caller-aligned strand contours and declared circularity. Unknown routes,
overlap, unmapped records, and material mismatch follow explicit refusal policies.

::: phydrax.applications.radiation_biophysics.TargetMolecule

---

::: phydrax.applications.radiation_biophysics.TargetSite

---

::: phydrax.applications.radiation_biophysics.SourceTargetRoute

---

::: phydrax.applications.radiation_biophysics.RadiationTargetGeometry

---

::: phydrax.applications.radiation_biophysics.PreparedRadiationTargets

---

::: phydrax.applications.radiation_biophysics.prepare_radiation_targets

---

::: phydrax.applications.radiation_biophysics.TargetHit

---

::: phydrax.applications.radiation_biophysics.TargetMapping

---

::: phydrax.applications.radiation_biophysics.map_radiation_targets

## Candidates and initial lesions

Direct candidates use inclusive cumulative deposited-energy thresholds per
history/target; indirect candidates use retained reaction channels and a matched
chemistry endpoint. One realized lesion can retain both causes. The Bernoulli
realization preserves stable candidate-addressed randomness and parent events.

::: phydrax.applications.radiation_biophysics.IndirectLesionRule

---

::: phydrax.applications.radiation_biophysics.LesionPolicy

---

::: phydrax.applications.radiation_biophysics.LesionCandidate

---

::: phydrax.applications.radiation_biophysics.LesionCandidates

---

::: phydrax.applications.radiation_biophysics.candidate_radiation_lesions

---

::: phydrax.applications.radiation_biophysics.InitialLesion

---

::: phydrax.applications.radiation_biophysics.InitialLesionLedger

---

::: phydrax.applications.radiation_biophysics.realize_radiation_lesions

## Contour clusters and physical normalization

Clusters cannot cross molecules, histories, or fractions. A DSB needs an actual
opposite-strand backbone pair within the inclusive contour gap, including circular
closure. These are initial-lesion classifications, not repair or survival states.

`GRAY` is J/kg. Yield uses the whole scored-mass deposition and explicitly selected
normalization, retaining independent-primary sampling uncertainty separately from
dose-normalization uncertainty. Unknown uncertainty is `None`, not zero; direct
and indirect cause yields overlap when the same lesion has both causes.

::: phydrax.applications.radiation_biophysics.LesionCluster

---

::: phydrax.applications.radiation_biophysics.RadiationClusters

---

::: phydrax.applications.radiation_biophysics.contour_distance

---

::: phydrax.applications.radiation_biophysics.cluster_radiation_lesions

---

::: phydrax.applications.radiation_biophysics.HistoryExposure

---

::: phydrax.applications.radiation_biophysics.RadiationYield

---

::: phydrax.applications.radiation_biophysics.radiation_yield

## Calibration and stage evidence

Calibration consumes unthinned unit-probability candidate support, not realized
lesions. The differentiable union map predicts initial-site-lesion yield, not DSB
cluster yield. The proper Gaussian logit prior defines a Laplace posterior
approximation; it does not repair missing likelihood identifiability. Result rank,
held-out residuals, stage evidence, and `gates` remain separate.

Quantitative comparison requires declared positive observation errors and known
reference uncertainty. Training/held-out observations and physical conditions must
be disjoint. Scientific qualification additionally requires experimental lesion
yields and accepted non-synthetic `transport`, `chemical-G`, `target-reactions`, and
`lesion-yields` evidence. Missing real provider/campaign data is an explicit gate,
not experimental success inferred from a synthetic fit.

::: phydrax.applications.radiation_biophysics.LesionExpectationSupport

---

::: phydrax.applications.radiation_biophysics.prepare_lesion_expectation

---

::: phydrax.applications.radiation_biophysics.expected_initial_lesion_yield

---

::: phydrax.applications.radiation_biophysics.RadiationCondition

---

::: phydrax.applications.radiation_biophysics.RadiationCalibrationData

---

::: phydrax.applications.radiation_biophysics.RadiationStageEvidence

---

::: phydrax.applications.radiation_biophysics.RadiationCalibrationResult

---

::: phydrax.applications.radiation_biophysics.calibrate_radiation_lesions
