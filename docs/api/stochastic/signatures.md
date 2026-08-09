# Signatures and path kernels

`phydrax.stochastic` provides one tensor-algebra implementation for rough controls,
path features, and online recurrent updates. `phydrax.kernels` builds exact finite
feature kernels and a pure-JAX signature-PDE kernel on that same convention.

## Conventions

A path-value array has shape
`batch_shape + (num_knots, dimension)`. `path_signature(values, depth)` computes the
piecewise-linear signature of the increments between adjacent knots. It returns tensor
levels one through `depth`; the scalar level is deliberately external. A one-knot path
therefore has zero nonscalar levels.

With `stream=True`, each level has an additional knot axis and contains one value per
input knot. The first value is the identity signature and each later value is the
signature of the corresponding path prefix. Terminal mode returns only the final
levels and does not retain prefix history.

`flatten_signature(..., include_scalar=True)` prepends the scalar level. The flattened
size is

`1 + dimension + dimension**2 + ... + dimension**depth`.

`SignatureFeatures` packages these rules as an Equinox feature map.
`LogSignatureFeatures` applies the tensor logarithm and converts it to the
standard-bracket Lyndon basis represented by `PrimitiveBasis`. Its coordinate order is
exactly `primitive_basis.words`.

```python
import jax.numpy as jnp
import phydrax as phx

path = jnp.asarray(
    [[0.0, 0.0], [0.3, -0.1], [0.5, 0.4], [0.8, 0.2]]
)
paths = jnp.stack((path, -path, 0.5 * path))
features = phx.stochastic.SignatureFeatures(
    2,
    4,
    include_scalar=True,
)
terminal = features(path)
prefixes = phx.stochastic.SignatureFeatures(
    2,
    4,
    include_scalar=True,
    stream=True,
)(path)
```

## Ragged paths and physical time

Path kernels do not infer masks, normalize coordinates, interpolate missing values, or
repair invalid samples.

Use `repeat_last_path_padding(values, lengths)` to replace only the suffix after each
valid prefix by its final valid knot. Zero terminal increments then make the signature
independent of the padded capacity. Every declared valid prefix must already be finite.

Use `time_augment_path(times, values, lengths=...)` to prepend physical time as a
channel. Shared time vectors and per-case time arrays are supported. Valid times must
be finite and strictly increasing; repeat-last padding is applied jointly to time and
state when `lengths` is supplied.

```python
padded_paths = jnp.asarray(
    [
        [[0.0, 0.0], [0.4, 0.2], [0.7, 0.5], [jnp.nan, jnp.nan]],
        [[0.0, 0.0], [0.3, -0.2], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
    ]
)
lengths = jnp.asarray([3, 2])
times = jnp.asarray(
    [[0.0, 0.5, 1.0, jnp.nan], [0.0, 0.75, jnp.nan, jnp.nan]]
)
clean = phx.stochastic.repeat_last_path_padding(padded_paths, lengths)
space_time = phx.stochastic.time_augment_path(times, clean, lengths=lengths)
```

## Online recurrent signatures

`SignatureRecurrentCell` implements the same Chen updates through the canonical
`phydrax.nn.layers.run_recurrent` contract. The first valid point establishes a
basepoint and emits the identity signature. Later valid points update from the previous
point. Invalid padding preserves state and produces the recurrent runner's masked zero
output; a valid reset starts a new path segment. Passing `result.final_state` as the
next call's `initial_state` gives chunked streaming without changing the result.

```python
valid = jnp.ones(path.shape[:-1], dtype=bool)
batch = phx.nn.layers.RecurrentBatch(path, valid)
cell = phx.stochastic.SignatureRecurrentCell(2, 4, include_scalar=True)
result = phx.nn.layers.run_recurrent(cell, batch)
assert jnp.allclose(result.final_output, features(path))
```

## Exact feature kernels

For small feature spaces, pull `LinearKernel` back through `SignatureFeatures`. This
constructs the exact truncated signature Gram matrix and exposes the feature map for
reuse in linear models:

```python
features = phx.stochastic.SignatureFeatures(2, 4, include_scalar=True)
exact_kernel = phx.kernels.InputTransformedKernel(
    phx.kernels.LinearKernel(),
    features,
    transform_id=features.feature_id,
    input_ndim=2,
)
gram = exact_kernel.matrix(paths, paths)
```

The tensor feature size grows exponentially with dimension and depth. Prefer this route
when the resulting vectors are small or will be reused many times.

## Signature-PDE kernel

`SignaturePDEKernel(static_kernel, polynomial_order=m)` evaluates the Goursat-PDE
recurrence over every pair of path intervals without constructing tensor features. The
monomial boundary recurrence carries an additional global Picard-level axis. Truncating
that axis after `m` makes the result exactly the inner product of signatures through
tensor level `m`, rather than a local numerical approximation. Positive definiteness is
therefore inherited from a real feature inner product.

For a nonlinear `static_kernel`, each interval pairing is the four-point RKHS increment
of the knot Gram matrix. For `LinearKernel`, it reduces to the ordinary dot product of
path increments. One-knot paths return the scalar kernel value one, duplicate knots
contribute zero increments, and left and right knot counts may differ.

```python
kernel = phx.kernels.SignaturePDEKernel(
    phx.kernels.LinearKernel(),
    polynomial_order=5,
    pair_block_size=32,
)
gram = kernel.matrix(paths, paths)
```

The PDE route costs polynomial work in `polynomial_order` and quadratic work in the two
path lengths, but its recurrence does not grow exponentially with channel dimension.
Increase the order until the quantity of interest is stable. `pair_block_size` controls
matrix-evaluation memory only; it does not change kernel values. No normalization,
time augmentation, or ragged masking is implicit.

The recurrence follows the Goursat formulation of the signature kernel and the
monomial boundary propagation developed in
[The Signature Kernel is the solution of a Goursat PDE](https://arxiv.org/abs/2006.14794)
and
[Numerical Schemes for Signature Kernels](https://arxiv.org/abs/2502.08470).
Phydrax uses global level truncation to retain an exact positive-definite kernel at each
finite order.

The repository benchmark separates compilation, steady-state forward execution,
and reverse-mode execution:

```console
python tools/signature_benchmarks.py --quick
```

It always compares tensor features with the PDE recurrence and automatically
adds Signax and iisignature rows when those optional packages are installed.
Sigkax is not retained as a reference backend: its archived custom-call bridge
targets obsolete JAX/XLA extension APIs, while the native recurrence is portable
pure JAX on CPU and GPU.

## API

::: phydrax.stochastic.path_signature

---

::: phydrax.stochastic.path_logsignature

---

::: phydrax.stochastic.flatten_signature

---

::: phydrax.stochastic.repeat_last_path_padding

---

::: phydrax.stochastic.time_augment_path

---

::: phydrax.stochastic.SignatureFeatures

---

::: phydrax.stochastic.LogSignatureFeatures

---

::: phydrax.stochastic.SignatureRecurrentState

---

::: phydrax.stochastic.SignatureRecurrentCell
