# Omics, spatial biology, and spectrometry

These domains share experimental-design obligations but not a data model. Counts,
continuous assays, spatial measurements, spectra, chromatograms, and identifications
remain distinct public contracts.

## Omics assays and measurement state

`CountAssay` and `ContinuousAssay` use samples-by-features orientation. Dense and
fixed-sparse storage preserve four different states: observed value, observed zero,
missing measurement, and structural absence. Sparse routes have an explicit implicit
state; an absent stored route is not automatically a biological zero.

Feature identity belongs in a versioned `FeatureDictionary`; one-to-many or uncertain
routes use `FeatureMapping`. Do not align assays by column position after filtering or
conversion. Ontology propagation uses a validated `OntologyGraph`, not free-form names.

Normalization functions return transformed data, offsets, and evidence rather than
mutating the assay. Library-size and median-ratio assumptions are visible. Size factors,
feature filters, and transformation parameters used for predictive evaluation must be
estimated on training data only.

## Experimental design and pseudobulk

A cell, nucleus, spot, aliquot, or technical lane is not automatically an experimental
unit. `pseudobulk_counts` sums technical count rows into caller-declared unit indices
and reports contributing cells and observed measurements. Empty units remain invalid.
It does not manufacture biological replication.

`build_experimental_design` encodes condition, donor pairing, batch, nested batch,
interactions, and covariates. The resulting `ExperimentalDesign` exposes rank,
condition number, valid rows, and coefficient projection. `DesignContrast` and
`pairwise_condition_contrast` diagnose estimability in that coefficient space.

The native count model is NB2. `fit_negative_binomial_glm` fits each feature under a
supplied dispersion and optional log offset. The model likelihood is exact for the NB2
assumption; finite optimization is iterative. Feature dispersion estimation,
mean–dispersion trend fitting, and log-scale shrinkage are approximate modeling steps.
`wald_test` and `likelihood_ratio_test` use asymptotic approximations. Report their
method contracts, fit validity, degrees of freedom, dispersion procedure, estimability,
and multiple-testing correction.

`benjamini_hochberg` and `benjamini_yekutieli` control different dependence settings;
the correction does not rescue an invalid design or pseudoreplication. Composition
inference retains donor/group/exchangeability evidence. Single-cell QC thresholds,
doublet-like scoring, cell labels, and RNA-velocity parameters are model- and
workflow-dependent, not universal biological truth.

## Transcript abundance, pathways, and multiomics

Transcript abundance operates on a supplied equivalence-class relation and effective
lengths. Differential transcript usage retains the gene-to-transcript relation and
group definition; transcript ambiguity is not silently collapsed to genes.

Ontology feature-set testing uses explicit feature/ontology identity, a bounded
permutation plan, and input provenance. A pathway label does not establish mechanism.
Kinetic RNA velocity requires named spliced/unspliced layers and fitted kinetic
assumptions. Transport integration and multiomic alignment carry training masks and
provenance and are exploratory learned/relaxed mappings; they must not be fitted on the
evaluation rows or interpreted as exact biological correspondences.

## Spatial frames, assays, and graphs

`SpatialFrame` names ordered axes and one `SpatialUnit`. `MICROMETRE`, `MILLIMETRE`, and
`NANOMETRE` have physical scale. `PIXEL` deliberately has none. A pixel-to-physical
conversion therefore requires an explicit calibrated `AffineSpatialTransform`.
`SpatialCoordinates` carries numeric points and the frame code into JAX.

`SpatialAssay` links host `SpatialSampleRecord` lineage to packed
`SpatialAssayData`. Every valid spot has a positive sampling weight. Sample, donor, and
section routes remain separately available; unequal spot density must not be mistaken
for biological replication.

`SpatialNeighborPlan` builds a deterministic radius or k-nearest-neighbor graph with
binary, inverse-distance, or Gaussian weights. Distance is Euclidean in one explicit
common frame. Sections are disconnected graph components. The complete required row
capacity is preflighted; overflow invalidates the graph rather than truncating high
degree rows. Tied distances are recorded and graph routing is nondifferentiable.

`spatial_autocorrelation_test` estimates Moran's I or Geary's C significance by
restricted randomization. `assay_autocorrelation_test` derives donor, section, and
sampling weights from the assay. Inference requires at least two donors, nonconstant
values, a valid graph, and valid exchangeability. Permutations default to section
blocks unless a more explicit `ExchangeabilityPlan` or block vector is supplied.
Spots contribute to a statistic, but donors provide biological replication.

Rigid OT registration is iterative and requires explicit source/target frames and a
`SpatialRegistrationPlan`; registration uncertainty is not biological variability.
Image-pyramid classes represent metadata, shapes, chunks, transforms, tile bounds, and
bounded patches. They do not read arbitrary Zarr arrays. Morphology summaries retain
pixel size, origin, object capacity, and persistence/topology evidence; segmentation
itself is supplied, not learned or inferred by these summaries.

## Spectrometry units and acquisition

`MassSpectrum`, `SpectrumBatch`, `Chromatogram`, and `AcquisitionRun` keep acquisition
and units explicit. m/z, intensity, retention time, collision energy, polarity,
dissociation, analyzer, precursor/product roles, and ion mobility are not interchangeable
numeric columns. Padded peaks and spectra use masks; zero intensity remains distinct
from padding.

Calibration uses declared calibrants and model form. `fit_mass_calibration` and
`apply_mass_calibration` report residual and validity evidence. Binning uses a
`MassBinningPlan`; a binned profile is not a centroid list. `lookup_spectrum` performs
bounded acquisition lookup, not spectral identification.

LC–MS feature matching is conditioned on `FeatureMatchPlan` tolerances and supplied
features. Formula/isotope and adduct candidates remain bounded candidate sets.
Proteomics preserves target/decoy construction, PSM competition level, q-value
estimation, peptide evidence, shared-peptide protein groups, digestion specificity,
and quantification evidence. Protein inference over shared peptides is not proof that
every member is present. FDR applies to the declared competition and decoy construction,
not automatically to downstream proteins or metabolites.

Metabolomics confidence classification follows the supplied reference evidence and
cannot create authentic-standard evidence. Isotope/adduct reasoning, library matches,
and formula candidates must retain tolerance, unit, and candidate provenance.
Quantitative results require declared normalization/calibration assumptions and cannot
be upgraded to absolute concentration without an appropriate quantitative standard.

## Leakage and resource policy

- Split by donor or higher biological lineage before learned preprocessing,
  normalization fitting, feature selection, integration, threshold tuning, or model
  selection. Keep technical replicates in the same biological split.
- Spatial sections from one donor are not independent donors. Restrict permutations by
  the experimental design rather than shuffling all spots.
- Fit spectral calibration, library thresholds, decoy policy, and learned classifiers
  without evaluation-set reuse. Keep acquisition batch and instrument identity.
- Declare assay sparsity semantics, graph capacity, permutations, features, spectra,
  peaks, precursors, candidates, and auxiliary solve capacity before computation.
- Inspect status/evidence for every feature and every bounded batch. Do not filter
  invalid results after looking at effects and then report an unchanged error rate.

## Unsupported boundaries

No public omics call reads `AnnData` or `MuData` implicitly, no spatial call downloads or
opens an OME-NGFF store, and no spectrometry call downloads libraries or calibrants.
The native layer does not perform segmentation, spot calling, raw vendor-file decoding,
peak picking from proprietary raw signals, de novo peptide sequencing, or clinical
biomarker validation. The optional Pyteomics mzML reader is a host adapter only; it
does not change the native scientific contracts.
