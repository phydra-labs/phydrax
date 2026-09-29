# Scientific measurements

`phydrax.measurement` is the admission boundary between external data and numerical observation operators. Its central object is one `QuantityField`: physical meaning, component layout, sampling support, values, validity, optional independent uncertainty, and quality evidence.

## Data authority

Keep acquisition and derivation stages distinct:

```text
raw → calibrated → derived → reconstructed → inferred
```

`DataOrigin` separately distinguishes external and synthetic products. A synthetic asset requires a generator identity. A product derived inside Phydrax requires parent identities and a transformation identity. A reconstructed or inferred field is never relabeled as a direct measurement.

`MeasurementAsset` binds one field to acquisition identity, governed `ReferenceArtifactManifest` values, intended use, metadata, and a `DerivationRecord`. Rights are checked before the asset is admitted.

## Collections, clocks, and frames

`MeasurementCollection` groups heterogeneous assets through explicit roles and
relations without implying alignment. Clock mappings and time-dependent frame
routes are calibrated, bounded, and prepared before JAX execution. See
[Measurement collections, clocks, and frames](guides_measurement_collections.md).

## Quantity and value layout

`QuantitySpec` answers what values mean. Its compatibility identity includes the explicit semantic key, physical dimension, sign convention, support association, and reference configuration. Equal dimensions alone are insufficient.

`ValueLayout` answers how components transform. Scalars, complex scalars, vectors, covectors, tensors, categorical values, probability vectors, and counts have different validation. Geometric component values require a frame.

## Sample support

A support owns sample identity and geometry, not measured values:

- `IndexSampleSupport`: named dense axes without an asserted physical embedding;
- `PointSampleSupport`: irregular physical points with stable sample IDs;
- `RaySampleSupport`: origins, normalized directions, bounds, time, and stable pulse/sample IDs;
- image-specific supports implement the same minimal `SampleSupport` protocol.

`SamplingSemantics` distinguishes point values, cell averages, cell integrals, surface averages, path integrals, detector bins, and events. `TemporalSampling` distinguishes instantaneous values, interval means, interval integrals, and cumulative values.

## Validity, quality, and uncertainty

The validity mask has the sample shape. Component axes remain separate. Invalid payloads are retained but cannot contribute to numerical comparison. Zero is valid data.

`QualityFlag` is evidence and never changes validity implicitly. `IndependentStandardUncertainty` is optional; absence means unquantified, not exact. Correlated covariance remains an explicit observation/UQ operator rather than a dense asset field.

## Compiled use

`QuantityField.prepare()` creates `PreparedQuantityField`: fixed-shape JAX arrays plus static quantity, layout, support, sampling, unit, and field identities. Unit conversion is explicit at preparation.

`MeasurementComparisonPlan` accepts only compatible prepared observed and predicted fields. Different supports require an explicit spatial observation operator; no automatic interpolation occurs.

Discretized fields of a coupled spatial problem are observed through explicit observation bindings of `phydrax.solver.coupling`. `FieldPointObservation` samples point values or one derivative, `FieldBoundaryObservation` a boundary trace, average, or path integral, and `FieldFluxObservation` a flux content. Each binding declares its `QuantitySpec`, layout, support, and sampling, and construction refuses a sampling kind or unit dimension that differs from the operation it performs. Every `CoupledSolution` returns the prediction as a `PreparedQuantityField` (`solution.observation(binding_id)`) that compares with data prepared from the same records; it is evidence only when the solution is accepted. Virtual-element point values are labeled H1 projections, not field values. See [Observation bindings](guides_numerical_interoperability.md#observation-bindings).

## Comparison noise and likelihoods

A comparison's noise model is declared, never implied. `MeasurementComparisonResult.noise_model` records which declaration applies:

| `noise_model` | Declaration | Result |
|---|---|---|
| `"unquantified"` | no covariance, no uncertainty, no reference scale | residual only; no quadratic, log determinant, or likelihood |
| `"reference_weighting"` | `reference_scale=` on unquantified data | reference-weighted residual and least-squares quadratic; no likelihood |
| `"independent_uncertainty"` | observed `IndependentStandardUncertainty` | diagonal whitening, quadratic, log determinant, normalized log likelihood on active values |
| `"covariance"` | explicit `covariance=` action | quadratic, log determinant, normalized log likelihood; whitened residual only when a factor exists |

`whitened_residual` is present only when an actual factor `W` with `W^T W` equal to the precision backs it, so its squared norm equals `quadratic`. `whitening` names the factor: diagonal, dense Cholesky, Kronecker Cholesky, or circulant. Dense precision, diagonal-plus-low-rank, and matrix-free precision covariances report `whitening="unavailable"` with the quadratic and log determinant; no dense square root is invented. `log_likelihood` is `-(quadratic + logdet_covariance + active_value_count * log(2 pi)) / 2` for real-valued data. Independent uncertainty on complex data carries no implied circular or per-component normalization, so its likelihood fields are absent.

Independent uncertainty marginalizes exactly by dropping invalid samples, so its active set follows both validity masks at runtime. A correlated covariance does not: masking entries of a dense correlated residual changes the likelihood. The correlated active set is therefore fixed by the observed validity mask at construction, and the covariance must be declared on exactly those values. `restrict_observation_covariance(covariance, active)` returns the exact Gaussian marginal: diagonal and diagonal-plus-low-rank structure is kept, a dense Cholesky factor is refactored, and a dense precision uses its Schur complement. Kronecker, circulant, and matrix-free precision covariances refuse a partial restriction because no structure-preserving marginal exists; declare the active-set covariance explicitly. A prediction invalid on a correlated active value sets `active_set_consistent` and `successful` to false.

Least-squares objectives may use `quadratic` or `whitened_residual` under an explicit reference weighting. Posterior code requires a declared noise model and consumes `log_likelihood`; noise that depends on inferred parameters belongs in a normalized likelihood such as `phydrax.uq.FixedObservationLikelihood` or `phydrax.uq.LinearizedGaussianMeasurementLikelihood`, whose values and derivatives include the covariance log determinant.

## Radiation quantity meanings

`RadiationQuantityKind` and `resolve_radiation_quantity` are the single shared
catalog for deposited energy, absorbed dose, dose to water, dose to medium,
kerma, dose rate, relative dose, particle and energy fluence, LET, lineal
energy, activity, activity concentration, and their time integrals. The factory
returns ordinary `QuantitySpec` values; consumers still use the same field and
asset substrate.

Equal dimensions do not collapse meanings. Absorbed dose and kerma both use Gy;
LET and lineal energy both use energy per length; those pairs remain
incompatible. Reference medium, normalization, particle/site convention, and
spatial association must be non-generic where the quantity requires them.
Correlated Monte Carlo estimator error stays in explicit estimator evidence and
is not mislabeled `IndependentStandardUncertainty`.
