# Joint correlated ratios of means

`correlated_ratio_of_means` estimates `mean(numerator) / mean(denominator)` from aligned `(stream, draw)` arrays. It does **not** average instantaneous ratios or estimate numerator/denominator errors independently. This generic UQ owner has no dependency on projector solver types. See [projector analysis](../solver/projector_monte_carlo.md) for its physical estimator composition.

## Stream identity and complete common blocks

Declare unique `stream_ids`, one `dependence_id` per stream, and `sampling_origin_id`. Equal dependence IDs declare synchronously dependent streams; different groups declare independence, which the caller must justify. A `valid` mask retains original aligned positions. Gaps split contiguous complete runs; dependent streams are synchronized before blocking. Every active record must be finite, even if an incomplete tail or synchronization excludes it. Padding is not a draw.

Correlation selection uses initial-positive-sequence diagnostics for raw real/imaginary channels and ratio-influence channels. A still-positive tail at the admitted lag limit is unresolved, not truncated into a reassuring error bar. One common block length is applied to synchronous complete batch means. Incomplete tails are discarded explicitly; independent group contributions use their squared sample weights in the mean covariance. Insufficient draws/blocks and observed stochastic zero variation refuse qualification. No PSD clipping or new bootstrap is used to repair the result.

## Covariance and denominator evidence

Real channel order is `Re X`, optional `Im X`, `Re Y`, optional `Im Y`. `mean_covariance` retains all cross-channel terms (two to four channels); `ratio_covariance` uses the corresponding real one- or two-component division Jacobian. `standard_error` alone is not the full complex uncertainty record.

For real data, `fieller_set` distinguishes bounded, unbounded, disconnected, all-real, empty, singleton, and not-applicable confidence sets. Qualified ratios require bounded denominator-safe evidence and any declared denominator-sign/magnitude policy. Complex denominators have no sign contract: an asymptotic confidence ball must exclude the origin, including when covariance is singular. A finite nonzero sample denominator alone does not satisfy either gate. The confidence multiplier controls an asymptotic Gaussian approximation, not guaranteed finite-history coverage.

`exploratory_value` is distinct from `value`; unsafe qualified values/covariances are NaN. Inspect `statistically_valid`, composable `CorrelatedRatioStatus`, denominator radius/lower bound, Fieller class/bounds, retained/block masks, block counts/group weights, and correlation closure/variation diagnostics. Do not replace a refusal with denominator clipping, missing covariance terms, or an instantaneous ratio.

## Deterministic sources are explicit

An observed constant **stochastic** history is unresolved, not proof of zero sampling uncertainty. Only explicit deterministic source evidence permits deterministic singleton handling. `deterministic_ratio` is a separately declared exact relation `X = ratio * Y`, verified against every active record; it is not fitted from the data or a bypass for denominator safety. For example, a known proportionality with a stochastic denominator does not make that denominator safely nonzero.

This small example has genuinely deterministic source records; it is not a template for declaring stochastic projector output deterministic:

```python
import jax
import jax.numpy as jnp

from phydrax.uq import CorrelatedRatioPolicy, correlated_ratio_of_means

jax.config.update("jax_enable_x64", True)
result = correlated_ratio_of_means(
    jnp.asarray([[6.0]], dtype=jnp.float64),
    jnp.asarray([[3.0]], dtype=jnp.float64),
    policy=CorrelatedRatioPolicy(denominator_sign="positive"),
    stream_ids=("analytic-control",),
    dependence_ids=("analytic-control",),
    sampling_origin_id="known-deterministic-source",
    deterministic=True,
)
print(result.value, result.statistically_valid, result.fieller_set)
```

Temporal correlation evidence is not weight ESS, and neither diagnoses finite-population, timestep, stationarity, guide-support, or sign-resolution systematics. In projector composition, ordered replica pairs sharing a replica are aggregated per time rather than passed as independent streams.

::: phydrax.uq.CorrelatedRatioPolicy

::: phydrax.uq.CorrelatedRatioResult

::: phydrax.uq.CorrelatedRatioStatus

::: phydrax.uq.correlated_ratio_of_means
