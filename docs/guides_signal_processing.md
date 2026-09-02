# Differentiable signal processing

`phydrax.signal` provides JAX-native signal operators with explicit sample axes,
immutable streaming state, and gradients through signal values and numerical
parameters. Core functions operate on ordinary arrays; physical units, labeled
domains, and application metadata remain with their existing Phydrax owners.

## Array and axis contract

Every one-dimensional signal operator accepts a sample `axis`, defaulting to the
last axis. All remaining axes are independent streams. Initial FIR and
resampling kernels are one-dimensional and shared across those streams; use
`jax.vmap` when streams require distinct kernels.

Framing replaces the sample axis by adjacent frame and within-frame axes:

```python
import jax.numpy as jnp
import phydrax as phx

values = jnp.arange(2 * 32 * 3, dtype=float).reshape((2, 32, 3))
frames = phx.signal.frame(values, 8, 4, axis=1)
assert frames.shape == (2, 7, 8, 3)
```

`overlap_add` performs summation only. It does not normalize overlapping
samples, fill uncovered gaps, center the record, or restore a dropped tail.
Those policies belong to higher-level transforms.

## Finite convolution and causal FIR filtering

`convolve` is a finite zero-extended operation. `full`, `same`, and `valid`
have exact crops, and `direct` and one-shot `fft` use the same definition. No
runtime size heuristic changes the selected method.

`fir_filter` is instead causal and returns one output per input:

```python
taps = jnp.asarray((0.25, 0.5, 0.25))
filtered = phx.signal.fir_filter(values, taps, axis=1)
```

For streaming execution, coefficients remain explicit differentiable inputs
while the plan and carried history are separate:

```python
plan = phx.signal.FIRFilterPlan(taps.size, axis=1)
state = plan.initial_state((2, 16, 3), dtype=values.dtype)
state, block = plan.step(state, values[:, :16], taps)
state, tail = plan.flush(state, taps)
```

`block.active` is a one-dimensional prefix mask. An optional scalar
`valid_length` marks a padded terminal chunk without consuming its inactive
suffix. `flush` emits the finite response tail and returns reset state.

## Three distinct rate-conversion contracts

### Periodic Fourier resampling

`fourier_resample` treats transformed axes as periodic grids and globally
transfers representable Fourier modes. When `axes` is omitted, the trailing
`len(output_shape)` axes are transformed:

```python
periodic = jnp.sin(2.0 * jnp.pi * jnp.arange(16) / 16)
fine = phx.signal.fourier_resample(periodic, (31,))
```

For channel-last fields, pass spatial axes explicitly:

```python
field = jnp.ones((4, 32, 48, 2))
resized = phx.signal.fourier_resample(field, (40, 60), axes=(1, 2))
```

This operation is not a finite-record anti-aliasing resampler.

### Raw polyphase `upfirdn`

`upfirdn` applies the supplied taps literally. It does not reduce the integer
ratio or multiply taps by the upsampling factor. Its output begins at high-rate
phase zero and includes the finite zero-extended filter tail:

```python
raw = phx.signal.upfirdn(periodic, taps, up=3, down=2)
```

The implementation phase-packs the taps and never materializes a zero-inserted
input.

### Aligned finite-record resampling

`resample_poly` reduces the ratio, applies the required rate gain, compensates
the linear-phase filter delay, and returns exactly the duration-aligned finite
record:

```python
prototype = phx.signal.kaiser_sinc_resampling_filter(3, 2)
aligned = phx.signal.resample_poly(periodic, 3, 2, taps=prototype)
```

Custom centered prototypes must have odd length. Finite records use zero
extension. Other boundary policies are deliberately not inferred.

### Causal streaming resampling

A streaming plan fixes the reduced ratio, tap count, chunk capacity, and sample
axis. Full chunk lengths must be divisible by the reduced down factor:

```python
prototype = phx.signal.kaiser_sinc_resampling_filter(3, 2)
plan = phx.signal.RationalResamplingPlan(3, 2, prototype.size, 16)
state = plan.initial_state((16,), dtype=jnp.float64)

state, output = plan.step(state, periodic, prototype)
state, tail = plan.flush(state, prototype)
```

The causal stream is intentionally not centered. Concatenating active outputs
and the flush tail equals raw `upfirdn(values, prototype * up, up, down)` for
the reduced ratio. Results carry an absolute output offset, making replay and
time-coordinate construction deterministic.

## Differentiability and static topology

| Quantity | Differentiable | Reason |
| --- | --- | --- |
| Signal values | Yes | Ordinary JAX array operands |
| FIR/resampling taps | Yes | Passed to each numerical call |
| FIR/resampling history | Yes | Dynamic `StrictModule` state |
| Kaiser beta and Tukey alpha | Yes | Continuous window parameters |
| Axis, frame length, hop length | No | Determine layout and output shape |
| Integer up/down ratio | No | Determines polyphase topology |
| Chunk capacity and valid length | No | Static capacity or discrete prefix count |
| Active masks and absolute counters | No | Discrete stream metadata |

Plans inherit `NonTrainableState`; carried states do not. This prevents static
topology from entering optimizer parameter trees without blocking gradients
through history.

Complex arrays use native complex dtypes. For a real scalar objective, JAX's
ordinary conjugate-aware JVP/VJP conventions apply.

## Wavelet transforms

Fixed critically sampled wavelet transforms are public through
`phydrax.signal`:

```python
transform = phx.signal.DiscreteWaveletTransform(
    (-1,),
    levels=3,
    wavelet="db4",
    boundary="symmetric",
)
coefficients = transform.analysis(periodic)
restored = transform.synthesis(coefficients)
```

The filter bank and transform plan are fixed numerical state. Trainable custom
filtering should use the FIR primitives instead. Alpert multiwavelets retain
their existing private layout until they support an explicit transformed axis.

## Deliberate boundaries

The core does not provide a universal signal container, hidden mutable state,
a processor graph, implicit channel axes, arbitrary missing-sample policies,
or automatic direct/FFT selection. Spherical transforms, control-system
frequency response, stochastic state-space inference, and physical signal
metadata remain with their existing Phydrax modules.
