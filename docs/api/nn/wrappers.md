# Wrappers

Composable model transforms that add structure or change output interpretation.

!!! note
    Key notes:

    - `EquinoxModel` / `EquinoxStructuredModel` adapt arbitrary Equinox/JAX callables into Phydrax models by attaching `in_size` / `out_size`.
    - `ComplexOutputModel` packs/unpacks real/imag parts into complex outputs.

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

## Model transforms

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
