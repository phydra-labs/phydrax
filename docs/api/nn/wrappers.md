# Wrappers

Composable model transforms that add structure or change output interpretation.

!!! note
    Key notes:

    - `EquinoxModel` / `EquinoxStructuredModel` adapt arbitrary Equinox/JAX callables into Phydrax models by attaching `in_size` / `out_size`.
    - `RaggedSeriesModel` adapts encoders for `RaggedSeriesDatasetDomain` payloads.
    - `ComplexOutputModel` packs/unpacks real/imag parts into complex outputs.
    - `Sequential` chains models so outputs of stage `i` feed stage `i+1`.

## Equinox adapters

Use these wrappers when you already have an `equinox.Module` (or any JAX callable) and
want it to participate in Phydrax's solver/training APIs.

`layout="value"` (default for `EquinoxModel`) treats `in_size/out_size` as the **value shape**
of a single (unbatched) sample. Inputs are flattened to a vector, the wrapped module is called,
and outputs are reshaped back to the declared value shape.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

key = jr.key(0)

mlp = eqx.nn.MLP(
    in_size=4,
    out_size=6,
    width_size=64,
    depth=2,
    activation=jax.nn.tanh,
    key=key,
)

# Declare value shapes: 2×2 -> 3×2 (both flatten to lengths 4 and 6 internally).
model = phx.nn.EquinoxModel(mlp, in_size=(2, 2), out_size=(3, 2))

x = jnp.zeros((2, 2))
y = model(x, key=key)
assert y.shape == (3, 2)
```

`layout="passthrough"` forwards inputs/outputs unchanged (the wrapper only supplies metadata).
This is useful if your wrapped module already owns its input/output layout.

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

key = jr.key(0)

drop = eqx.nn.Dropout(p=0.1)
model = phx.nn.EquinoxModel(drop, in_size=4, out_size=4, layout="passthrough")

x = jnp.zeros((4,))
y = model(x, key=key, inference=True)
```

For structured inputs (e.g. product domains), use `EquinoxStructuredModel`. With
`layout="passthrough"` it forwards tuples unchanged:

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

key = jr.key(0)

def stack_pair(inp, *, key=None):
    del key
    a, b = inp
    return jnp.stack([a, b])

model = phx.nn.EquinoxStructuredModel(stack_pair, in_size=2, out_size=2, layout="passthrough")
y = model((1.0, 2.0), key=key)
assert y.shape == (2,)
```

With `layout="value"`, tuple parts are concatenated into a single vector before calling the
wrapped module:

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

key = jr.key(0)

lin = eqx.nn.Linear(in_features=5, out_features=4, key=key)
model = phx.nn.EquinoxStructuredModel(lin, in_size=5, out_size=4, layout="value")

x = (jnp.ones((2,)), jnp.ones((3,)))
y = model(x, key=key)
assert y.shape == (4,)
```

!!! note
    - These wrappers are pointwise by default; use `jax.vmap` for batching.
    - `iter_=` is accepted for interface compatibility but is not forwarded to the wrapped callable.

::: phydrax.nn.EquinoxModel
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.EquinoxStructuredModel
    options:
        members:
            - __init__
            - __call__

---

## Ragged series adapters

Use `RaggedSeriesModel` to wrap a callable that consumes
`RaggedSeriesBatchInput`. The wrapper extracts the sampled case payload from a
`RaggedSeriesDatasetDomain` batch and returns a `coordax.Field` with the case
axis preserved.

`MaskedSeriesPoolingModel` is a small baseline encoder. It applies a per-step
model to each valid timestep, masks padded entries, pools over time, then applies
a readout model.

```python
import jax.random as jr
import phydrax as phx

key_step, key_readout = jr.split(jr.key(0))

encoder = phx.nn.MaskedSeriesPoolingModel(
    step_model=phx.nn.MLP(
        in_size=5,
        out_size=16,
        width_size=32,
        depth=2,
        key=key_step,
    ),
    readout_model=phx.nn.MLP(
        in_size=18,
        out_size=2,
        width_size=32,
        depth=2,
        key=key_readout,
    ),
)

u = domain.Function("data")(phx.nn.RaggedSeriesModel(encoder))
```

The `step_model` input size should match the flattened series channel count plus
one time channel when `include_time=True`, plus static channels when
`include_static_in_steps=True`. The `readout_model` input size should match the
pooled latent size plus static channels when `include_static_in_readout=True`.

::: phydrax.nn.RaggedSeriesBatchInput

---

::: phydrax.nn.RaggedSeriesModel
    options:
        members:
            - __init__
            - __call_batch__

---

::: phydrax.nn.MaskedSeriesPoolingModel
    options:
        members:
            - __init__
            - __call__

---

## Model transforms

`Sequential` is useful for embedded pipelines, for example
`RandomFourierFeatureEmbeddings -> MLP`, then reused inside separable wrappers.

```python
import jax.random as jr
import phydrax as phx

branch = phx.nn.Sequential(
    (
        phx.nn.RandomFourierFeatureEmbeddings(
            in_size="scalar",
            out_size=64,
            key=jr.key(0),
        ),
        phx.nn.MLP(
            in_size=64,
            out_size=16,
            width_size=64,
            depth=2,
            key=jr.key(1),
        ),
    )
)
```

::: phydrax.nn.Sequential
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.MagnitudeDirectionModel
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.ComplexOutputModel
    options:
        members:
            - __init__
            - __call__
