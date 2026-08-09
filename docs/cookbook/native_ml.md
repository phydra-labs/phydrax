# Native ML in a scientific workflow

This workflow fits a weighted closure, checks numerical evidence, differentiates a
validation objective through fitting, and chooses whether the fitted arrays remain
fixed or become a solver warm start.

## Fit a weighted closure

Assume a two-coordinate constitutive response is observed with unequal statistical
reliability. `MLBatch` keeps statistical and physical-measure weighting distinct.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

features = jnp.array(
    [
        [-1.0, -0.8],
        [-0.5, 0.2],
        [0.0, 0.0],
        [0.4, -0.3],
        [0.8, 0.7],
        [1.0, 0.9],
    ]
)
targets = 1.2 + 0.7 * features[:, 0] - 0.25 * features[:, 1]
reliability = jnp.array([0.5, 1.0, 1.5, 1.0, 0.75, 0.5])

result = phx.ml.fit(
    phx.ml.linear.RidgeRecipe(alpha=1e-5),
    features,
    targets,
    sample_weight=reliability,
    feature_schema=phx.ml.FeatureSchema(("strain", "temperature")),
    target_schema=phx.ml.TargetSchema("continuous", names=("stress",)),
)

if not bool(result.valid):
    raise RuntimeError((int(result.status), result.diagnostics))

closure = result.model
prediction = jax.jit(closure)(jnp.array([0.25, -0.1]))
```

`result.model` is an immutable `FrozenModel`: ordinary JAX differentiation still
works through its prediction, but Phydrax solvers keep its leaves outside the
trainable partition.

```python
input_sensitivity = jax.grad(closure)(jnp.array([0.25, -0.1]))
```

For quadrature or empirical-measure fitting, pass `measure_weight` and select a
recipe with `weight_policy="measure"` or `"product"`. Do not multiply physical
measure into `sample_weight` implicitly; the two quantities have different
meaning and remain separately inspectable.

## Differentiate through fitting

Ridge fitting is a direct differentiable solve while the reported augmented system
has the required rank. A continuous regularization parameter can therefore be
selected by an outer JAX objective without an estimator callback.

```python
validation_features = jnp.array([[-0.75, 0.1], [0.2, 0.4], [0.9, -0.2]])
validation_targets = (
    1.2
    + 0.7 * validation_features[:, 0]
    - 0.25 * validation_features[:, 1]
)


def validation_loss(alpha):
    fitted = phx.ml.fit(
        phx.ml.linear.RidgeRecipe(alpha=alpha),
        features,
        targets,
        sample_weight=reliability,
    ).as_trainable()
    residual = jax.vmap(fitted)(validation_features) - validation_targets
    return jnp.mean(residual**2)


dloss_dalpha = jax.grad(validation_loss)(jnp.asarray(1e-3))
```

The derivative is a property of the fit map, not merely of prediction. Inspect
`result.gradient_contract.fit_hyperparameters` and every listed condition before
using it in a larger inverse problem. A hard tree split, selected neighbor index,
or selected feature set would not supply the same derivative.

## Bind the fitted model to a Phydrax domain

Array-native ML models implement the same pointwise model and input-binding
contracts as Phydrax neural models. If the two features are the coordinates of a
geometric domain, the frozen closure can be bound directly:

```python
geom = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
)
frozen_field = geom.Model("x")(result.model)
```

`frozen_field` is now a domain-aware function. It can be placed in the `functions`
mapping of a `FunctionalSolver`, used by residual/condition terms, or differentiated
by Phydrax operators according to its pointwise binding. Because the wrapped model
is frozen, a hybrid physics solve can optimize other fields without accidentally
changing the learned closure.

To refine the fitted coefficients jointly with physics, opt in explicitly:

```python
trainable_field = geom.Model("x")(result.as_trainable())
```

This changes solver partitioning only. It does not copy the arrays, make a hard fit
differentiable, or erase the original fit diagnostics.

## Use sparse data deliberately

`SparseFeatures` stores a fixed number of nonzeros per row. A family with a genuine
sparse design path, such as sparse ridge, consumes it directly. A dense-only
algorithm raises `TypeError`; explicit materialization makes the memory decision
visible:

```python
sparse_features = phx.ml.SparseFeatures(
    features,
    jnp.broadcast_to(jnp.arange(features.shape[-1]), features.shape),
    feature_count=features.shape[-1],
)
sparse_batch = phx.ml.MLBatch(sparse_features, targets)
dense_batch = phx.ml.MLBatch(sparse_features.to_dense(), targets)
```

Choose one batch and fit it. Do not catch a sparse rejection and silently retry on
dense data: that would hide a potentially dominant allocation and change the
algorithmic contract.

## Preserve and export the result

A native artifact records registered model structure, schemas, fit metadata,
provenance, and checksums without pickle:

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = phx.ml.artifacts.save_ml_artifact(
        f"{directory}/closure.phxml",
        result.model,
        fit_result=result,
        feature_schema=phx.ml.FeatureSchema(("strain", "temperature")),
        target_schema=phx.ml.TargetSchema("continuous", names=("stress",)),
    )
    restored = phx.ml.artifacts.load_ml_model(path)
```

Selected already-fitted sklearn estimators and saved XGBoost artifacts can instead
cross the one-time fail-closed conversion boundary. The returned model is native;
prediction does not call the source package. ONNX export is available only when the
native JAX primitives used by that model are representable.

See [Native machine learning](../guides/ml.md),
[ML differentiation contracts](../appendix/ml_differentiability.md), and the
[complete ML API](../api/ml/index.md).
