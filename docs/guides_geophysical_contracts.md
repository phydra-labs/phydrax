# Geophysical quantity, time, and storage contracts

`phydrax.applications.geophysics` binds physical meaning to existing numerical
owners. It is not a field container, model state, time integrator, or coupled
simulation runtime.

## Quantities and exchange compatibility

`GeophysicalQuantity` follows the native application-quantity resolver and exact
multiplicative units. `quantity_id` includes the local name, axes and support.
`compatibility_id` excludes those local storage choices and unit scale while
retaining physical kind, reference configuration, sign and reference unit.

Consequently pressure and pressure anomaly, specific humidity and dry-air vapor
mixing ratio, or temperature and temperature anomaly remain distinct even when
their dimensions match. Equal compatibility IDs permit explicit unit conversion;
they do not prove compatible storage, time support or spatial transfer.

```python
from phydrax.applications.geophysics import GeophysicalQuantity
from phydrax.units import KILOPASCAL, PASCAL

source = GeophysicalQuantity("station_pressure", "pressure", KILOPASCAL)
target = GeophysicalQuantity("cell_pressure", "pressure", PASCAL)
assert source.compatibility_id == target.compatibility_id
pressure_pa = target.from_si(source.to_si(100.0))
```

Celsius offsets, moist/dry humidity-basis conversion and energy-reference changes
are explicit scientific transformations at the data/application boundary, not
implicit `UnitDefinition` conversions. A flux sign and thermodynamic reference
must be selected by the application; generic defaults do not certify an exchange.

## Storage binding

`GeophysicalFieldBinding` accepts exactly one native owner:

- `DiscreteFieldSpace`: the complete field space;
- `StateLayout`: an exact nonempty component-name selection;
- `OperatorTask`: one exact task field name.

It retains IDs and host descriptors, not the owner's arrays. A descriptor includes
quantity, storage reference, scientific role, optional vertical coordinate and
temporal support. Descriptors are canonical and content-addressed. Restoring a
binding with `from_dict` requires the actual native storage owner and verifies its
identity. An archived descriptor alone does not reconstruct a mesh or constitute
an exact model restart.

## Model calendars

`GeophysicalTimeSpec` converts ISO dates to numerical durations before compiled
execution. Supported calendars are:

- `standard` / `gregorian`: Julian through 1582-10-04, Gregorian from 1582-10-15;
- `proleptic_gregorian`;
- `noleap` / `365_day`;
- `all_leap` / `366_day`;
- `360_day`.

The transition gap is invalid in `standard`. Years 1 through 9999 are admitted;
dates before year 1 and timezone offsets are not implicitly interpreted. Epochs
accept date-only or date-time ISO strings, optional `Z`, and microsecond precision.
The clock is continuous model time, not a leap-second-aware UTC clock. Leap seconds
require an external explicit time-scale conversion. Encoding rejects timestamps
whose microseconds cannot survive the numerical duration roundtrip; choose a
closer epoch rather than silently losing precision.

```python
from phydrax.applications.geophysics import GeophysicalTimeSpec

clock = GeophysicalTimeSpec("360_day", "2001-02-30", "d")
assert clock.encode(["2001-03-01"])[0] == 1.0
```

Only `s`, `min`, `h`, and `d` are fixed-duration units. A calendar month/year is not
silently replaced by a fixed number of seconds. Calendar aliases canonicalize to
one identity. Changing calendar or epoch changes identity; no automatic cross-
calendar conversion occurs. Bind `time_id` into application plans and continuation
metadata while passing ordinary numerical coordinates to `TimeGrid` or
`TemporalMesh`.

## Temporal support

`TemporalSupport` distinguishes instantaneous, mean, accumulation, minimum and
maximum quantities. Every interval quantity requires finite, strictly increasing
bounds per sample. Bounds need not be disjoint: rolling-window products may
legitimately overlap. An operation requiring disjoint windows must check that
additional condition explicitly. Sample position is point, start, midpoint or end.
Bounds use the associated time specification's numerical unit; support alone does
not supply a clock. Instantaneous precipitation rate and accumulated precipitation
are not interchangeable.

## Hybrid pressure

`HybridPressureCoordinate(a, b)` prepares top-to-bottom interfaces:

```
p_interface = a * reference_pressure + b * surface_pressure
layer_mass_per_area = pressure_thickness / gravity
```

With `a_is_pressure=True`, `a` is already pressure-valued and is not multiplied by
the reference pressure. Arrays place the vertical axis last. Coefficients are
fixed nontrainable preparation state; surface pressure remains differentiable.
Constructor validation checks the reference column. `valid(surface_pressure)`
reports positive finite realized pressure thicknesses; consuming models must admit
each actual column rather than clipping crossed layers. Top pressure may be zero;
algorithms evaluating logarithms must use their own finite-layer quadrature rather
than taking an unregularized logarithm of that interface.

The coordinate does not imply hydrostatic balance, a dynamical core, surface
boundary conditions or a conservation theorem. Atmospheric models own those
requirements. Ocean moving-depth geometry remains with the existing ocean owner.
