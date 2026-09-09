# Scientific measurements

`phydrax.measurement` is the admission boundary between external data and numerical observation operators. Its central object is one `QuantityField`: physical meaning, component layout, sampling support, values, validity, optional independent uncertainty, and quality evidence.

## Data authority

Keep acquisition and derivation stages distinct:

```text
raw → calibrated → derived → reconstructed → inferred
```

`DataOrigin` separately distinguishes external and synthetic products. A synthetic asset requires a generator identity. A product derived inside Phydrax requires parent identities and a transformation identity. A reconstructed or inferred field is never relabelled as a direct measurement.

`MeasurementAsset` binds one field to acquisition identity, governed `ReferenceArtifactManifest` values, intended use, metadata, and a `DerivationRecord`. Rights are checked before the asset is admitted.

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
