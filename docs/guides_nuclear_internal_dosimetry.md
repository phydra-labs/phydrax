# Nuclear internal dosimetry

`phydrax.nuclear.dosimetry` provides research-only time-activity integration and MIRD-style regional or voxel S-value calculations. It does not reconstruct PET/SPECT data, infer biological kinetics, prescribe activity, calculate biological effect, write DICOM, or make a clinical dose claim.

The facade is a separate submodule rather than an eager `phydrax.nuclear` export because medical-grid dosimetry depends on imaging while core evaluated nuclear data is also imported during equation initialization.

## Time-activity series

`TimeActivitySeries` binds a governed `MeasurementAsset` to an existing decay-only `InventoryTransition`. The source must:

- represent `activity` or `activity_concentration`;
- use instantaneous temporal sampling;
- carry an explicit `SampleTimeAxis`;
- include the transition data artifact among its references;
- contain nonnegative values on valid samples;
- declare research use.

`TimeActivityIntegrationPlan` performs piecewise-linear trapezoidal integration over a closed interval wholly inside measured support. It never extrapolates or fits a kinetic model.

For samples A_i at times t_i, the full-support integral is the sum of 0.5 × (A_i + A_(i+1)) × (t_(i+1) − t_i). For A = [4, 2, 1] Bq at t = [0, 1, 3] s, the result is 6 Bq·s.

The result removes the time axis while preserving the original spatial sampling meaning. Required invalid samples invalidate the output. Input uncertainty is not collapsed to independent output uncertainty without a covariance model; the derivation records that propagation was not performed.

## Regional S values

`RegionalSValueTable` stores a target-by-source matrix for exactly one radionuclide transition, source-region set, target-region set, dose meaning, and retained nuclear-data artifact. The required S-value unit is Gy/(Bq·s).

`RegionalSValuePlan` evaluates:

```text
D_target = sum_source A_tilde_source × S(target <- source)
```

The source support and region-set identity must match exactly. Dose-to-water, dose-to-medium, absorbed dose, and kerma are distinct; a table declares one admitted dose meaning. No patient-mass scaling is inferred.

## Spatial S-value kernels

`SpatialSValueKernel` binds an odd-sized nonperiodic kernel to exact source and target `MedicalImageSupport` identities. Source and target grids must have the same shape and affine. The route accepts time-integrated activity concentration, multiplies by physical voxel volume to obtain Bq·s per source voxel, then applies the kernel in Gy/(Bq·s).

There is no wraparound, hidden resampling, spacing interpolation, or cross-grid transfer. Outside-grid activity is explicitly zero under the current profile. Invalid source samples propagate through the nonzero kernel footprint. Output uncertainty remains unknown unless an independently justified covariance model is added.

## Data and qualification

S-value and decay data require `NuclearDataProvenance`, retained rights, checksums, processing lineage, and uncertainty disposition. PhydraX does not bundle tables from MIRDCalculation or any other upstream repository.

`tools/nuclear_internal_dosimetry_qualification.py` checks analytic integration, regional matrix arithmetic, and nonperiodic spatial convolution. `benchmarks/nuclear_internal_dosimetry.py` records performance only. Synthetic agreement cannot close source-admission, locked-reference, external-transfer, or clinical-validation gates; all internal-dosimetry capability profiles remain unreleased.
