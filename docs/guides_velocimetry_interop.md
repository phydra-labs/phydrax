# Velocimetry persistence and interoperability

The native archive is the canonical persistence form. It stores arrays without pickle, verifies checksums, and validates exact current manifests and payload names. It preserves raw values, masks, status/evidence, validation, replacement, uncertainty, coordinates, frames, units, capacities, and provenance.

## Adapter reports

Every external conversion returns a `phydrax.interchange.AdapterReport` containing source/target identities and explicit transformed, synthesized, unsupported, or dropped semantics. A conversion that cannot represent a required distinction fails before writing a deceptive result.

Examples of scientifically relevant loss include:

- replacing missing vectors with zeros;
- merging raw, rejected, and filled vectors;
- dropping image-versus-physical coordinate convention;
- omitting calibration or frame timing;
- losing track identity, gaps, or camera detection IDs;
- replacing covariance with a scalar quality score.

## Formats

- OpenPIV text is accepted only when columns and the rectilinear coordinate layout are unambiguous.
- Supported PIVlab MAT/HDF5 layouts retain calibration, timing, vector type, and masks when present; unsupported object/cell layouts are rejected.
- xarray/pivpy conversion is optional and lazy. Named dimensions are validated, and object dtype or implicit materialization is rejected.
- OpenPTV-style targets, reconstructed particles, link records, and tracks remain separate record kinds with explicit index-base, time, frame, and unit mapping.
- Common image formats use an optional image reader. Bit-depth normalization and grayscale conversion are explicit policies.

## Downstream Phydrax adapters

A calibrated rectilinear PIV result may adapt to a compatible `PreparedTensorGrid`. Nonrectilinear physical positions are not silently resampled. Invalid vectors become observation masks in state-space data.

A PTV `TrackResult` may adapt to `TrajectoryData`. Each track is one case; valid timestamps are increasing; gaps break transition validity and establish reset boundaries. Tracking covariance and association alternatives remain on the original result.

PIV never adapts to `TrajectoryData`, because interrogation-grid vectors are not identity-bearing particle paths.
