# Parameter geometry

`phydrax.nn.parameters` owns reusable maps from unconstrained optimizer coordinates to physical parameter spaces, plus explicit model-PyTree selection. These objects are model infrastructure rather than uncertainty-method policy; UQ consumes them without owning them.

## Transforms

A transform is pure: raw leaves remain ordinary trainable arrays and the physical value is constructed on demand. No optimizer-side clipping, hidden mutation, or post-update repair occurs.

- Shape-preserving array transforms: `IdentityTransform`, `PositiveTransform`,
  `IntervalTransform`, `SymmetricTransform`, `SkewSymmetricTransform`, and
  `StiefelTransform`.
- Shape-changing transforms: `SimplexTransform`, which maps `k - 1` additive
  log-ratio coordinates to an interior `k`-simplex point;
  `PositiveDefiniteTransform`, which consumes packed lower-triangular
  coordinates; and the tuple-valued `HurwitzTransform` and
  `SchurStableTransform`.
- Unsupported complex inputs and impossible matrix shapes fail before numerical execution.

```python
import jax.numpy as jnp
import phydrax as phx

positive_layer = phx.nn.layers.Linear(
    in_size=4,
    out_size=3,
    rwf=False,
    weight_transform=phx.nn.parameters.PositiveTransform(minimum=1e-6),
)

raw = jnp.zeros((6,))
spd = phx.nn.parameters.PositiveDefiniteTransform()(raw)
```

`Linear.weight_transform` accepts only shape-preserving transforms and is mutually exclusive with Random Weight Factorization. MLP KFAC metadata labels transformed and RWF blocks explicitly; the KFAC solver rejects both rather than silently treating nonlinear parameterizations as direct affine weights.

::: phydrax.nn.parameters.AbstractParameterTransform

---

::: phydrax.nn.parameters.IdentityTransform

---

::: phydrax.nn.parameters.PositiveTransform

---

::: phydrax.nn.parameters.IntervalTransform

---

::: phydrax.nn.parameters.SimplexTransform

---

::: phydrax.nn.parameters.SymmetricTransform

---

::: phydrax.nn.parameters.SkewSymmetricTransform

---

::: phydrax.nn.parameters.PositiveDefiniteTransform
    options:
        members:
            - __init__
            - factor
            - __call__

---

::: phydrax.nn.parameters.HurwitzTransform

---

::: phydrax.nn.parameters.SchurStableTransform

---

::: phydrax.nn.parameters.StiefelTransform

## Transformed parameters

`TransformedParameter` stores raw coordinates and one static transform. It is an Equinox PyTree: optimizers see `raw`, while model evaluation calls `value()` or the object itself to obtain the constrained value.

::: phydrax.nn.parameters.TransformedParameter
    options:
        members:
            - __init__
            - value
            - __call__

## Explicit model subspaces

`ParameterSubspace` partitions selected inexact array leaves from a frozen complement and reconstructs the original model topology exactly. It also records deterministic leaf paths, shapes, exact dtypes, and total dimension. `pack()` and `unpack()` provide the canonical vector coordinate system; `reconstruct_vector()` rebuilds a complete model directly from that vector. Use exact leaf paths or named subtree paths for branched architectures. `last_layer(...)` means the globally final array leaves in deterministic PyTree order; it is not architecture-aware and does not select one output head per branch.

```python
subspace = phx.nn.parameters.ParameterSubspace.from_subtree_paths(
    model,
    (".projection",),
)
selected = subspace.initial
vector = subspace.pack()
assert vector.shape == (subspace.total_dimension,)
reconstructed_from_vector = subspace.reconstruct_vector(vector)
reconstructed = subspace.reconstruct(selected)
```

::: phydrax.nn.parameters.ParameterSubspace
    options:
        members:
            - __init__
            - reconstruct
            - array_leaf_paths
            - pack
            - unpack
            - reconstruct_vector
            - from_leaf_paths
            - from_subtree_paths
            - last_layer
