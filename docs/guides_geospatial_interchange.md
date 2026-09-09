# Qualified geospatial interchange

Phydrax keeps numerical coordinates and geospatial interpretation separate. A
`SpatialCoordinateContract` identifies the numerical length unit, coordinate kind,
coordinate system, and reference frame. `GeospatialContract` composes that identity
with horizontal, vertical, epoch, axis, registration, and transformation metadata.
It does not change the canonical serialization of `SpatialCoordinateContract`.

## Numerical admission

`GeospatialContract.require_cartesian()` is the boundary used by physical kernels. It
requires an explicitly qualified local Cartesian length frame, compatible axes and
units, a horizontal datum, and—when three dimensions are requested—an upward vertical
axis with a declared vertical datum. Geographic longitude/latitude coordinates are not
silently interpreted as Cartesian distances.

`require_compatible()` compares the qualified meanings of two coordinate systems.
Unit equality alone is insufficient: ellipsoidal height, orthometric height, and depth
may all be stored in metres while denoting different positions.

`GeospatialTransform` remains the provenance record for an externally established
operation. `CoordinateTransformPlan` additionally executes one explicit PROJ pipeline
on the host. The caller pins the expected pyproj/PROJ version strings and SHA-256 of
`proj.db` plus every named grid. Execution disables PROJ network access, verifies
those resources, preserves an explicit validity mask, and returns a transform record.
It never asks PROJ to choose an operation from source/target CRS labels. Version and
data pins do not content-pin the loaded native shared-library bytes; that environment
remains part of qualification evidence.

## Qualified grids

`QualifiedGeospatialGrid` stores regular x/y vectors, a `(y, x)` value array, a
validity mask, value unit and role, exact resource provenance, and one qualified
coordinate contract. It accepts ascending or descending axes without reordering the
native arrays.

Registration is physical metadata:

- `gridline`: samples lie on the declared region bounds;
- `pixel`: samples represent cells whose bounds extend half a spacing beyond the
  sample coordinates.

Unknown registration is rejected. Periodic longitude seams require an explicit seam
contract; duplicated gridline endpoints must agree in values and validity. No
resampling, seam repair, missing-value imputation, or CRS inference occurs.

## SEG-Y profiles

`SEGYRev1IEEEProfile` and `read_segy` implement the bounded revision-1 profile:

- explicit big- or little-endian selection;
- explicit ASCII or EBCDIC cp500 textual encoding;
- fixed-length IEEE binary32 traces;
- metre or foot linear header coordinates;
- source/group XY and positive-up elevation-minus-depth Z mapping;
- seismic, pressure-sensor, and dead trace identifiers;
- explicit pressure calibration and polarity when pressure units are claimed.

`SEGYRev2IEEEProfile` separately supports revision-2 fixed or variable IEEE binary32
traces, 16-bit or extended trace sample counts, extended textual headers, and an
explicit padded invalid mask for variable trace lengths. It does not broaden the
revision-1 class or alias the two profiles.

Sequence gaps remain gaps. Dead or padded samples remain invalid. Source-relative
clocks are not relabelled as UTC. Unsupported mute, angular-coordinate, transduction,
or sample-format semantics fail closed. `ResourceLimits` account for bytes, decoded
nodes, attributes, and losses before sample allocation.

## Clocks, boreholes, rasters, waveforms, and domain formats

`TimeReferenceContract` converts TAI, GPS, UTC, instrument, and source-relative clocks
through declared epoch, offset, and drift. UTC requires a checksum-pinned
`LeapSecondTable`; nominal UTC cannot silently represent the inserted positive leap
second. `BoreholeTrajectory`, `BoreholeInterval`, and `PreparedBoreholeSampling` keep
measured depth and Cartesian position as distinct coordinates with an explicit
transpose.

Additional bounded adapters cover:

- genuine miniSEED 3 through `pymseed`, SAC, and StationXML;
- GeoTIFF, CF-NetCDF, and consolidated ZIP-Zarr regular grids;
- electrical survey CSV, EDI, and EMTF XML impedance;
- ICGEM static `gfc` and geomagnetic coefficient records;
- SINEX positions and RINEX observation arrays;
- selected LAS curves tied to a qualified borehole trajectory.

Optional parsers are imported only at the host boundary. Every normalized result
retains exact source bytes, a `ResourceManifest`, format profile, assumptions,
preserved fields, capabilities, and declared losses. Unselected headers or curves are
reported as dropped from the normalized view even though the exact source resource is
retained.

## PyGMT boundary

`export_geospatial_grid` creates a host-only xarray export. With `for_pygmt=True`, its
region and registration are adapted explicitly for GMT while the native grid remains
unchanged. `render_geospatial_grid` accepts only local arrays, built-in palette names,
and local PNG/PDF/EPS paths. It runs the real optional PyGMT/GMT stack, validates the
rendered file signature and size, and returns a hashed `ResourceManifest` plus an
`AdapterReport`.

PyGMT, xarray, GMT, and Ghostscript are not numerical dependencies. Install the
`geospatial-render` extra and a compatible GMT runtime only for this host boundary.
Remote `@` resources, automatic reprojection of Cartesian grids, and fabricated output
on missing dependencies are rejected.

## Deliberate limits

This module is not a GIS, seismic processing suite, or remote data catalog. It does
not geocode, choose datum grids, access map tiles, impute missing samples, infer
physical units from a label, remove instrument responses, reconstruct borehole
trajectories, or resample between grids. Explicit pinned PROJ execution and the
bounded format adapters are the supported host operations; every other transformation
must be performed and qualified separately.
