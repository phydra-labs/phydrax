# Parameter geometry

`phydrax.nn.parameters` owns reusable maps from unconstrained optimizer coordinates to physical parameter spaces, plus explicit model-PyTree selection. These objects are model infrastructure rather than uncertainty-method policy; UQ consumes them without owning them.

## Transforms

A transform is pure: raw leaves remain ordinary trainable arrays and the physical value is constructed on demand. No optimizer-side clipping, hidden mutation, or post-update repair occurs.

- Shape-preserving array transforms: `IdentityTransform`, `PositiveTransform`,
  `IntervalTransform`, `SymmetricTransform`, `SkewSymmetricTransform`, and
  `StiefelTransform`.
- Shape-changing transforms: `SimplexTransform`, which maps `k - 1` additive
  log-ratio coordinates to an interior `k`-simplex point;
  `PackedSkewSymmetricTransform`, which consumes exactly the independent strict
  lower triangle; `PositiveSemidefiniteTransform`, which maps packed lower-triangular
  factors to `L @ L.T` without a diagonal floor; `PositiveDefiniteTransform`,
  which adds a strictly positive diagonal; and the tuple-valued `HurwitzTransform`
  and `SchurStableTransform`.
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

::: phydrax.nn.parameters.PackedSkewSymmetricTransform

---

::: phydrax.nn.parameters.PositiveSemidefiniteTransform
    options:
        members:
            - factor
            - __call__

---


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


## Low-rank adaptation

`LowRankUpdate` represents a real affine weight as
`W_eff = W_0 + scale * left @ right`. The base has shape `(out, in)`,
`left` has shape `(out, rank)`, and `right` has shape `(rank, in)`.
`LowRankSpec.scaling="rank"` uses `scale = alpha / rank`; `"sqrt_rank"`
uses the rank-stabilized `scale = alpha / sqrt(rank)`. `Linear` evaluates the
two factor contractions directly; it does not construct the dense update during
training. `merge_low_rank(...)` performs that construction explicitly for
deployment.

`adapt_low_rank(...)` accepts exact native `Linear.weight` paths only. It
initializes `left` to zero and `right` from a Gaussian, so the adapted model is
exactly the base model before optimization. `alpha=None` resolves to
`alpha=rank`. On an RWF layer the adapter updates the unscaled `V` coordinate
before the frozen row scale `exp(s)`; merging materializes the updated `V` and
preserves RWF. Weight transforms, complex weights, aliased layers, and already
adapted weights fail before model surgery.

```text
import jax.random as jr
import phydrax as phx

paths = phx.nn.parameters.low_rank_sites(model)
specs = {
    path: phx.nn.parameters.LowRankSpec(rank=8)
    for path in paths
}
adapted, report = phx.nn.parameters.adapt_low_rank(
    model,
    specs,
    key=jr.key(0),
)
subspace = phx.nn.parameters.low_rank_parameter_subspace(adapted)

fit = phx.nn.operator.training.fit_operator(
    adapted,
    training_data,
    parameter_subspace=subspace,
    epochs=10,
)
deployed = phx.nn.parameters.merge_low_rank(fit.execution_model)
```

The explicit subspace is mandatory for low-rank training through
`fit_operator` and `FunctionalSolver.solve`. It prevents gradients, optimizer
moments, and decoupled weight decay from reaching the dense base or unrelated
model leaves. The initial implementation supports standard and
extra-argument Optax transformations. KFAC and Phydrax geometric/iterative
optimizers reject explicit subspaces.

Adapter-only artifacts are content-bound to the complete dense base:

```text
phx.nn.parameters.save_low_rank_adapter("task.phxadapter", trained_model)
restored = phx.nn.parameters.read_low_rank_adapter(
    "task.phxadapter",
    exact_base_model,
)
```

Loading verifies the model type, static structure, every base array, site
path, shape, dtype, rank, and factor checksum before reconstruction. A wrong
base fails rather than accepting shape compatibility alone.

The factorization is not identifiable: for invertible `Q`, the pairs
`(left, right)` and `(left @ Q, inverse(Q) @ right)` represent the same dense
update. Deterministic optimization remains valid, but factor-space Hessians,
KFAC blocks, Laplace approximations, and MCMC require a separate gauge-aware
contract and are not claimed here.

::: phydrax.nn.parameters.LowRankSpec

---

::: phydrax.nn.parameters.LowRankUpdate

---

::: phydrax.nn.parameters.LowRankAdaptationReport

---

::: phydrax.nn.parameters.adapt_low_rank

---

::: phydrax.nn.parameters.low_rank_sites

---

::: phydrax.nn.parameters.low_rank_parameter_subspace

---

::: phydrax.nn.parameters.merge_low_rank

---

::: phydrax.nn.parameters.save_low_rank_adapter

---

::: phydrax.nn.parameters.read_low_rank_adapter

## Explicit model subspaces

`ParameterSubspace` partitions selected inexact array leaves from a frozen complement and reconstructs the original model topology exactly. It also records deterministic leaf paths, shapes, exact dtypes, and total dimension. `pack()` and `unpack()` provide the canonical vector coordinate system; `reconstruct_vector()` rebuilds a complete model directly from that vector. Use exact leaf paths or named subtree paths for branched architectures. `last_layer(...)` means the globally final array leaves in deterministic PyTree order; it is not architecture-aware and does not select one output head per branch.

```python
model = positive_layer
subspace = phx.nn.parameters.ParameterSubspace.from_subtree_paths(
    model,
    (".weight",),
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
            - validate_root
            - rebase
            - array_leaf_paths
            - pack
            - unpack
            - reconstruct_vector
            - from_leaf_paths
            - from_subtree_paths
            - last_layer
