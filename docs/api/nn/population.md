# Physical LIF population codes

`phydrax.nn.population` constructs a fixed, interpretable population response and
fits linear decoders with Phydrax's existing weighted-SVD least-squares substrate.
It does not introduce another optimizer, graph compiler, or spiking runtime.
Physical LIF neurons come from `phydrax.applications.electrophysiology`; artificial
recurrent LIF cells are a separate `phydrax.nn.layers` model.

## End-to-end approximation

```python
import jax.numpy as jnp
import jax.random as jr
from phydrax.applications import electrophysiology as ep
from phydrax.domain import HyperRectangle
from phydrax.nn import population as pc

population_key, fit_key, assessment_key = jr.split(jr.key(42), 3)
neuron = ep.LeakyIntegrateAndFire(
    0.2, 0.01, -65.0, -50.0, -62.0, refractory_ms=2.0
)
population = pc.prepare_lif_population(
    HyperRectangle([-1.0], [1.0]), neuron, 64, key=population_key,
    intercept_range=(-0.9, 0.9), maximum_rate_range_hz=(50.0, 150.0),
)
training = pc.sample_population_points(population, 512, key=fit_key)
held_out = pc.sample_population_points(population, 256, key=assessment_key)

def target(point):
    return jnp.asarray([point[0] ** 2, jnp.sin(2.0 * point[0])])

code = pc.fit_population_decoder(population, training, target, ridge=1e-6)
prediction = code(held_out)
report = pc.assess_population_code(code, held_out, target)
print(code.least_squares.valid, code.least_squares.rank, report.rmse)
```

Every stochastic construction call requires an explicit key. Evaluation points
use the native `HyperRectangle.sample_interior` sampler; use different keys for
fit and assessment. The API also accepts caller-supplied evaluation points and
weights. Targets may be pointwise callables, sample-leading arrays, or `None`
(identity reconstruction). Scalar and tensor-valued target shapes are retained.
Assessment is intentionally explicit: the library does not secretly split,
resample, or reuse training samples as held-out evidence.

`LIFPopulation`, `PopulationCode`, diagnostics, and filtered state are
`NonTrainableState`: fitting is intentionally frozen under the native solver's
trainable partition. Decoder application remains differentiable with respect to
inputs; frozen training ownership is not a `stop_gradient` operation.

## Encoding and physical units

The domain is the existing native `HyperRectangle`, not a second geometry type.
Coordinates map componentwise to [-1, 1]. Encoders are normalized to Euclidean
unit length. Their projection is divided by the encoder's L1 norm, so its range
over the **whole box**, including multidimensional corners, is [-1, 1]. Points
outside the box are evaluated without clipping, but approximation quality there
has not been established by an in-domain fit.

`prepare_lif_population` samples or accepts `encoders`, `intercepts`, and
`maximum_rates_hz`, then solves for inward-current gain and bias. Each intercept
is the projection at the neuron's rheobase, and each maximum rate is attained at
projection +1. Intercepts must lie in [-1, 1); maximum rates must be positive and
strictly below the refractory ceiling. Direct construction is available for
explicit calibrated parameters:

```python
population = pc.LIFPopulation(
    HyperRectangle([-1.0], [1.0]), neuron,
    [[1.0], [-1.0]], [0.2, 0.2], [0.15, 0.15],
)
current_nA = population.currents(jnp.asarray([0.4]))
rate_hz = population.rates(jnp.asarray([0.4]))
```

Capacitance is nF, conductance uS, voltage mV, current nA, and physical time ms.
Rates and decoder activity inputs are **Hz**, not spikes/ms or binary events.
The physical response is

```text
I_threshold = g * (threshold - resting)
charge_ms = (C / g) * log1p(g * (threshold - reset) / (I - I_threshold))
rate_hz = 1000 / (refractory_ms + charge_ms), when I > I_threshold
rate_hz = 0, otherwise
```

The reset voltage therefore affects both the measured period and gain inversion.
This is not the artificial timestep-based LIF activation. Population construction
requires positive capacitance, nonnegative leak conductance, reset below threshold,
and nonnegative refractory duration. At zero leak the exact perfect-integrator
limit is `charge_ms = C * (threshold - reset) / I` for positive current. It does
not infer a rate response for AdEx, conductance-driven, or time-varying states
from this constant-input LIF law.

## Diagnosing the fit

`fit_population_decoder` delegates to `solve_weighted_least_squares`, which uses
the native `DenseSVD` least-squares solve rather than normal equations. No hidden
centering, column scaling, or intercept neuron changes the decoder model.
`code.least_squares` exposes:

- `rank`, `singular_values`, and `condition_number`;
- `valid`, `status`, and `normal_equation_error`;
- `sample_count`, `valid_rows`, and coefficients in the physical Hz coordinates.

`code.silent_neurons` identifies selected zero-activity neurons over the effective
training samples. `code.activity_rms_hz` reports their weighted activity scale.
Duplicate/dependent neurons remain in place and create an observable rank
deficit. With positive ridge, a finite regularized fit may be valid despite that
deficit; its unregularized rank and infinite deficient condition number remain
visible. Without ridge, deficient fits are marked invalid even though their
minimum-norm coefficients remain inspectable. Check diagnostics before accepting
a scientific approximation; validity alone is not held-out accuracy evidence.

A `neuron_mask` selects the decoder's active neurons without changing capacity.
Inactive neurons receive zero coefficients and are excluded when decoding, even
if their supplied rate values are nonfinite. Silent active neurons are not
silently removed from the rank requirement.

Sample masks and weights define the evaluation measure. Following the native
least-squares contract, zero, negative, nonfinite, masked, and nonfinite-data rows
do not contribute. The remaining positive weights are normalized **after** row
selection. This makes global positive rescaling, weighted sample replication,
and masked padding invariant, including with ridge and total raw weight below
one. `code.training_weight_sum` preserves the original effective weight sum;
`code.least_squares.weight_sum` refers to the normalized measure. An empty
effective fit is invalid; empty assessments return invalid evidence and NaN
errors, never a misleading zero-error result.

Held-out assessments report output-wise weighted RMSE, target RMS, relative
RMSE, and maximum absolute error. A zero target has relative error zero only for
zero absolute error, otherwise infinity.

## Explicit filtered-spike decoding

A rate-fitted decoder must not multiply raw binary spikes or counts as though
they were Hz. Convert externally generated uniform-bin spike counts explicitly:

```python
counts = jnp.zeros((4, code.population.neuron_count))
counts = counts.at[0, 0].set(1).at[1, 0].set(1).at[3, 0].set(2)
filtered = pc.filter_population_spikes(
    counts, dt_ms=1.0, time_constant_ms=20.0
)
output = pc.decode_filtered_spikes(code, filtered)
next_filtered = pc.filter_population_spikes(
    counts, dt_ms=1.0, time_constant_ms=20.0,
    initial_rate_hz=filtered.final_rate_hz,
)
```

Counts have shape `(time, ..., neurons)`, may exceed one, and must be finite and
nonnegative. Fractional counts are allowed for expected-count calculations.
The API does not manufacture event times or infer a clock. Each bin's
`1000 * count / dt_ms` Hz is treated as zero-order-held forcing of a first-order
low-pass filter, integrated exactly over that bin. Values are sampled at the
right endpoint. This preserves the DC rate and keeps the binning approximation
explicit; it is not an assertion that all spikes occurred at the bin endpoint.
Filter time constants and dt must be finite and positive. Filtering is causal;
chunking with `final_rate_hz` exactly preserves the state convention.

`decode_filtered_spikes` requires the typed `FilteredSpikeRates` result; raw
arrays belong only in the explicitly named `code.decode_rates` Hz API.

For a time-aligned point trajectory, `assess_population_spikes` reports four
weighted assessments under one shared valid-sample measure:

1. `approximation`: unfiltered decoded physical rates minus the function target;
2. `filtering`: filtered deterministic rate decoding minus unfiltered decoding;
3. `spike_variability`: filtered spike decoding minus the matched filtered
   deterministic rate decoding;
4. `total`: filtered spike decoding minus the target.

The three signed residuals add to the total residual. Their RMSEs **do not add**,
because the errors can be correlated. The deterministic baseline uses the
recorded filter initial state, so startup history is not quietly attributed to
spike noise. Assessment masks exclude measurements, not causal filter evolution.
This separation does not certify that externally supplied spikes came from the
physical model or that a steady-rate response captures a transient neuron.

## Runnable evidence and cost

- `python examples/population_code.py` fits a two-output nonlinear function,
  assesses independent points, and decodes analytically periodic physical LIF
  spikes under constant input with random stationary phases. It reports all four
  temporal errors, including filter startup rather than hiding a burn-in.
- `python benchmarks/population_code.py --neurons 64 --samples 512 --queries 256`
  measures weighted-SVD fitting and rate-plus-decoder application separately,
  reports synchronized warmups/repeats and compilation phases, and includes
  held-out approximation and rank evidence. Increase `--neurons` and `--samples`
  to inspect scaling; no benchmark outcome is assumed by the API.

The dense design costs O(samples × neurons) storage and uses one native economy
SVD. Decoder application costs O(neurons × output size) per point. Filtering is
a scan with one carried activity array, not a second event runtime.

::: phydrax.nn.population
