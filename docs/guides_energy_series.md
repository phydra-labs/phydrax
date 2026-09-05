# Energy quantities and interval data

`phydrax.applications.EnergySeries` contains a native `SampledSeries`; it does not introduce a second numerical-series implementation. Quantity and native unit identities, the time unit, the time basis, sign/reference metadata, and provenance accompany the numerical samples. Absolute time requires a declared origin. Civil calendars and source-specific quality flags remain ingress-adapter responsibilities; timezone labels do not themselves convert clock coordinates.

## Samples and intervals

- `instantaneous`, `schedule`, and `cumulative` data are node-aligned.
- `interval_average` and `interval_integral` data are edge-aligned.
- Average power and integrated energy are different meanings even when the same timestamps are present.
- Invalid components remain invalid. A valid neighboring channel does not make an incomplete channel valid.

```python
import jax.numpy as jnp
from phydrax.applications import EnergySeries, rebin_energy_series, integrate_energy_series
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import JOULE, SECOND, derived_unit

watt = derived_unit("W", ((JOULE, 1), (SECOND, -1)))
support = SeriesSupport(jnp.array([0., 2., 5.]), coordinate_id="elapsed-time")
load = EnergySeries(
    SampledSeries(support, jnp.array([3., 5.]), alignment="edge", series_id="load"),
    quantity="active_power", unit=watt, meaning="interval_average",
)
target = SeriesSupport(jnp.array([0., 1., 3., 5.]), coordinate_id="elapsed-time")
aligned = rebin_energy_series(load, target)
energy, energy_unit = integrate_energy_series(aligned)
# Aligned averages: [3, 4, 5] W. Integral: 21 J.
```

`rebin_energy_series` uses cumulative integrals and ordered searches, not a dense all-pairs interval-overlap matrix. Average samples assume held values within source intervals; integral samples assume uniform within-interval density. It is differentiable in numerical payloads within the declared alignment regime. It does not claim to reconstruct unknown subinterval variations.

The target must use the same coordinate identity, time unit/origin, and leading series axes. Clock-reset/overlapping episodes must be selected independently before rebinning. Missing source intervals or target coverage outside the source produce invalid output components, not extrapolation. Integer missing-interval accounting prevents small uncovered intervals from being hidden by a floating-point coverage tolerance.

`integrate_energy_series` refuses incomplete values on active intervals. It sums only explicitly declared active intervals: it does not assert that disconnected support covers the gaps. Average-valued quantities gain an SI-second factor after exact time-unit conversion; integral-valued quantities retain their unit.

## Cumulative counters

`counter_to_intervals` rejects a counter decrease unless the caller supplies one of:

- a finite positive single-wrap `rollover` modulus, with readings inside its range;
- authoritative edge-aligned `reset_increments` for decreasing intervals.

A reading after a reset does not establish how much consumption occurred before the reset. The operation therefore does not guess reset-to-zero behavior. Source validity is preserved in the differentiated interval samples.

## Units and physical references

Use `phydrax.units` for multiplicative conversion. Temperature offsets belong to explicit ingress transformations, not DAE potential equality. Electrical per-unit values also require a power-base contract; shared dimensions do not equate active/reactive/apparent power or different energy carriers. Planning currency and heating-value assumptions must be declared by the respective domain.

See [building energy](guides_building_energy.md), [energy planning](guides_energy_planning.md), and [energy interchange](guides_energy_interchange.md).

## API

::: phydrax.applications.EnergySeries

::: phydrax.applications.rebin_energy_series

::: phydrax.applications.integrate_energy_series

::: phydrax.applications.counter_to_intervals
