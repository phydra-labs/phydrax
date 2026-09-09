# Geophysical data: a strict host boundary

Geophysical interchange converts selected external scientific variables into
immutable native values, explicit Boolean validity masks, and self-contained
scientific descriptors. It does not introduce a data execution engine or a
second archive format. Runtime storage remains a native `DiscreteFieldSpace`,
`StateLayout`, or `OperatorTask`, identified by `GeophysicalFieldBinding`.

## Optional dependencies

Nothing imports xarray, a NetCDF backend, Zarr, or a GRIB decoder until the host
interoperability function is called.

| Route | Optional dependencies | API |
| --- | --- | --- |
| Host dataset | `xarray` | `from_cf_dataset`, `to_cf_dataset` |
| NetCDF3 | `xarray`, `scipy` | `read_cf(..., format="netcdf")`, `write_cf` |
| NetCDF4/HDF5 | `xarray`, `h5netcdf`, `h5py` | `engine="h5netcdf"` |
| NetCDF4 alternate backend | `xarray`, `netCDF4` | `engine="netcdf4"` |
| Zarr | `xarray`, `zarr` | `format="zarr"` |
| GRIB decoding | `xarray`, `cfgrib`, `eccodes` and ecCodes runtime | `read_cf(..., format="grib")` |
| Native persistence | Existing PHYDRAX dependencies only | `write_geophysical_archive`, `read_geophysical_archive` |

A missing optional module raises `AdapterError` with
`OPTIONAL_DEPENDENCY_UNAVAILABLE`. Decoding errors remain real errors: there is
no fake fallback dataset, alternate format guess, or GRIB encoder.

## Explicit source-to-storage binding

```python
from phydrax.applications.geophysics import (
    GeophysicalFieldBinding, GeophysicalQuantity,
    from_cf_dataset, write_geophysical_archive, read_geophysical_archive,
    to_cf_dataset,
)
from phydrax.dynamics import StateLayout
from phydrax.units import KELVIN

layout = StateLayout((1,), component_names=("temperature",))
binding = GeophysicalFieldBinding(
    GeophysicalQuantity("temperature", "temperature", KELVIN,
                        axes=("time", "latitude", "longitude")),
    state_layout=layout, components=("temperature",),
)
# dataset is an xarray Dataset containing raw numeric CF variables.
# Open external data with decode_cf=False; this boundary owns semantic decoding.
data, report = from_cf_dataset(dataset, bindings={"tas": binding})
write_geophysical_archive("weather.zip", data, run_id="source-analysis",
                          evidence_ids=(report.report_id,))
restored = read_geophysical_archive("weather.zip", bindings={"tas": binding})
exported, export_report = to_cf_dataset(restored)
```

Source local names need not match native names. The explicit CF `standard_name`
map checks physical kind (`air_temperature` maps to `temperature`, for example).
Unsupported names are rejected, not inferred from a variable's spelling.
Declared native axes permit an exact dimension permutation, never interpolation,
reshaping, or implicit remapping. Omit axes to retain source dimension order.
Directional wind and flux standard names additionally require
`quantity.reference_configuration` to equal that exact CF standard name and
`sign_convention="positive"`. This deliberately refuses to infer east/north or
up/down meaning from a component's local storage label. `geopotential_height`
likewise requires its exact reference configuration and is not silently treated
as geometric height. Potential temperature is not admitted as air temperature.

Only explicitly bound scientific variables and their supported dependency
closure are selected. Coordinate variables, auxiliary coordinates, bounds,
grid-mapping variables, and hybrid formula terms are preserved. The native
product's field descriptors retain quantity and binding identities and complete
binding descriptors. A detached archive does not invent resolved geometry:
rebind a descriptor using `GeophysicalFieldBinding.from_dict(..., field_space=...)`
or another actual original native storage owner.

`data.values(name)` and `data.valid(name)` return read-only host arrays.
`data.descriptor` returns a fresh JSON-safe mapping; mutating it cannot change
the product. `data.arrays` contains canonical payload names referenced by that
descriptor. Moving selected values into a model's existing native storage is an
explicit application operation, not an external-format side effect.

## Scientific semantics

- **Units:** supported CF spellings are parsed at the boundary into the native
  units substrate. Examples include `K`, `Pa`, `hPa`, `m s-1`, `kg m-3`,
  `kg m-2`, `kg m-2 s-1`, `W m-2`, `J kg-1`, and `kg kg-1`.
  Celsius absolute temperature is an explicit affine boundary conversion.
  This does not extend the generic multiplicative unit API. Unknown required
  units fail closed. Latitude/longitude retain angular units and frame labels.
  CF export converts bound quantities and their bounds to supported reference
  units rather than emitting arbitrary native unit symbols. Native archives keep
  the original full unit descriptor. `lwe_precipitation_rate` is a volume flux
  (length/time), not a mass flux; water density is never implicitly applied.
- **Missing and packed data:** fill/missing sentinels and valid ranges are tested
  in the packed domain before scale/offset decoding. Invalid values remain
  masked, not replaced by physical zero. Decoded values use host float64.
  Original packing attributes remain source provenance. Export writes logical
  decoded values with NaN missing values; it does not claim original packed-byte
  equivalence. Conversion reports enumerate packing and unit transformations.
- **Time:** numeric CF `<unit> since <epoch>` coordinates are resolved through
  `GeophysicalTimeSpec`, including mixed Gregorian, proleptic Gregorian, no-leap,
  all-leap, and 360-day calendars. Calendar aliases normalize to native identity.
  No implicit calendar conversion, month/year-duration assumption, leap-second
  interpretation, or conversion through NumPy's Gregorian datetime is performed.
  Scalar time coordinates and auxiliary `time(t)` coordinates associate through
  their support dimensions, not a requirement that names equal dimension labels.
- **Cell methods:** a single `time: point`, `time: mean`, or `time: sum` is
  supported. Means and accumulations require finite positive-width bounds
  containing each sample; start/midpoint/end positioning is recorded when exact.
  Sums remain amounts: a six-hour accumulation is never silently turned into a
  per-second rate. A supplied binding's temporal support must match exactly.
- **Vertical coordinates:** pressure coordinates retain pressure units,
  orientation, and bounds. Hybrid `p=a*p0+b*ps` and `p=ap+b*ps` formulas require
  explicit layer-coordinate and coefficient bounds. Contiguous coefficient
  bounds define native interfaces; midpoint coefficients are never mistaken
  for interfaces. Surface pressure must be positive/unmasked; interfaces must
  increase top-to-bottom in every source column and bracket sample pressures.
  A supplied native hybrid binding must match the derived interface identity.
- **Grid/frame:** geographic and rotated latitude/longitude mapping descriptors
  are preserved. This is not a projection or vector rotation routine. Unknown
  mappings are rejected, as are grid-relative GRIB winds labeled as earth-relative
  winds. Native callers remain responsible for selecting compatible geometry.
- **Provenance:** local file imports record existing `ResourceManifest` content
  checksums, trusted-root information, resource limits, and source paths. Native
  descriptor and collection identities bind this provenance to scientific values.

Unsupported required features include arbitrary compound/spatial cell methods,
climatologies, compressed/ragged coordinates, mesh/geometry conventions,
ancillary-variable semantics, cell measures, unsigned packed reinterpretation,
and unsupported parametric vertical coordinates. They produce explicit adapter
failure instead of silently discarding scientific meaning. Applications can
also declare `required_semantics`; unknown requested semantics are rejected.
Arbitrary JSON-compatible descriptive attributes are preserved, not executed.

## Bounded local resources

`read_cf` accepts local resources and an optional `trusted_root`. NetCDF and
GRIB files are read with the existing descriptor-relative bounded-resource
reader, then decoded from private snapshots. GRIB index sidecar creation is
disabled. Zarr stores are copied member-by-member through the same bounded reader,
with aggregate byte/member/depth bounds and no directory symlinks. No remote
store, executable provider plugin, or automatic distributed/lazy compute is
introduced.

`ResourceLimits` bounds source bytes, selected logical float64 values plus
Boolean masks, metadata depth/count/bytes, and conversion-loss count. These are
resource and logical-payload limits, not an operating-system peak RSS quota:
external decoders and temporary host arrays also consume bounded working memory.
In-memory lazy arrays require explicit `materialize_lazy=True`; shape/dtype
bounds are checked before materialization. The default source/logical payload
limit is 256 MiB. Choose lower finite limits for untrusted inputs.

The GRIB profile accepts one scalar forecast reference/lead pair per dataset.
It derives a CF valid-time coordinate, preserves forecast reference/lead
coordinates and source provenance, and maps instantaneous, accumulation, or mean step types to
the same CF converter. Interval statistics require actual decoded
`startStep`/`endStep`/`stepUnits`, not guessed bounds. Multi-reference forecasts,
ambiguous step statistics, and hybrid GRIB without full CF formula terms/bounds
must be explicitly prepared upstream or are rejected.

## Native result products versus exact restarts

`GeophysicalData` purpose is `initialization` or `sampled-result`. External
imports cannot claim `exact-restart`. Native persistence is exact for the
converted product, including masks, quantities, time/vertical descriptors,
source provenance, and canonical payload references. It does not preserve solver
history, RNG state, distributed partition state, or an external execution graph.
Those belong to existing checkpoint and restart-admission owners.

Persistence uses the existing lifecycle `ResultManifest` and array archive.
The optional `ResultManifest.sampled_semantics` metadata binds the canonical
geophysical descriptor into the manifest identity. Empty sampled semantics
preserve existing generic result identities and generic query/export routes.
`lifecycle.query` still selects named field values and units; its archive manifest
exposes sampled semantics. Generic NPZ export remains a values-only inspection
route, not a complete geophysical round trip. Use `write_cf` or the native
geophysical archive functions for complete scientific products.

Every native open checks outer archive/manifest identities, payload checksums,
field references, masks/shapes, quantity and binding descriptor identities,
calendar identity, and hybrid descriptor consistency. Native checksums detect
corruption/inconsistency; they are not cryptographic signatures against an
attacker who can rewrite a complete coherent artifact.

## Runnable qualification

```sh
python tools/geophysical_data_qualification.py
python tools/geophysical_data_qualification.py --format zarr
python tools/geophysical_data_qualification.py --output /tmp/geophysical-data-proof
```

The script exercises analytic packed Celsius temperatures, a missing sample,
360-day time coordinates, and precipitation accumulations through host import,
native lifecycle persistence, CF export, and file reimport. It reports physical
round-trip errors, mask count, calendar, and native archive identity. Optional
codec tests also construct an actual ecCodes GRIB sample and decode it through
cfgrib. Run the focused regression file with the repository's selected-test policy:

```sh
pytest -n auto tests/unit/applications/test_geophysical_data.py
```
