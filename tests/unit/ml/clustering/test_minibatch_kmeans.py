from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.artifacts import read_ml_artifact, save_ml_artifact
from phydrax.ml.clustering import ClusterDiagnostics, KMeans, MiniBatchKMeans


_KEY = jax.random.key(17)
_X = jnp.asarray([[-10.0], [-9.0], [-8.0], [8.0], [9.0], [10.0]], dtype=jnp.float64)


def test_full_batch_matches_two_lloyd_iterations() -> None:
    x = jnp.asarray([[-10.0], [10.0], [-9.0], [9.0], [-8.0], [8.0]], dtype=jnp.float64)
    batch = MLBatch(x)
    mini = MiniBatchKMeans(
        2, epochs=2, batch_size=6, initial_step=1, revival="off", initialization="first"
    ).fit_batch(batch, key=_KEY)
    full = KMeans(
        2, max_iterations=2, initialization="first", empty_policy="retain"
    ).fit_batch(batch)
    np.testing.assert_allclose(
        mini.as_trainable().centers,  # ty: ignore[unresolved-attribute]
        full.as_trainable().centers,  # ty: ignore[unresolved-attribute]
        atol=1e-12,
    )


@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
def test_closing_quantizer_identities(complex_data: bool) -> None:
    x = _X + (1j * _X / 4 if complex_data else 0)
    batch = MLBatch(x, sample_weight=jnp.arange(1, 7, dtype=jnp.float64))
    result = MiniBatchKMeans(2, initialization="first", batch_size=2).fit_batch(
        batch, key=_KEY
    )
    d = result.diagnostics
    w = batch.effective_weight("statistical")
    p = d.cluster_mass / jnp.sum(d.cluster_mass)
    centers = result.as_trainable().centers  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(
        jnp.sum(p[:, None] * centers, axis=0),
        jnp.sum(w[:, None] * x, axis=0) / jnp.sum(w),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jnp.sum(p[:, None] * jnp.abs(centers) ** 2) + d.objective,
        jnp.sum(w[:, None] * jnp.abs(x) ** 2) / jnp.sum(w),
        rtol=1e-10,
    )


def test_one_cluster_is_weighted_mean() -> None:
    weights = jnp.arange(1, 7, dtype=jnp.float64)
    result = MiniBatchKMeans(1).fit_batch(MLBatch(_X, sample_weight=weights), key=_KEY)
    np.testing.assert_allclose(
        result.as_trainable().centers[0],  # ty: ignore[unresolved-attribute]
        jnp.sum(_X * weights[:, None], axis=0) / jnp.sum(weights),
    )


def test_power_of_two_scaling() -> None:
    recipe = MiniBatchKMeans(2, initialization="first", accumulation="deterministic")
    w = jnp.arange(1, 7, dtype=jnp.float64)
    a = recipe.fit_batch(MLBatch(_X, sample_weight=w), key=_KEY)
    b = recipe.fit_batch(MLBatch(_X * 8, sample_weight=w * 4), key=_KEY)
    np.testing.assert_array_equal(b.as_trainable().centers, a.as_trainable().centers * 8)  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(b.model(_X * 8), a.model(_X))


def test_revival_and_translation() -> None:
    x = jnp.concatenate(
        (
            jnp.zeros((4, 1), dtype=jnp.float64),
            jnp.repeat(
                jnp.asarray([[0.0], [20.0], [40.0], [60.0]], dtype=jnp.float64),
                20,
                axis=0,
            ),
        )
    )
    split = MiniBatchKMeans(
        4,
        epochs=5,
        batch_size=8,
        initialization="first",
        revival="split",
        accumulation="deterministic",
    )
    off = MiniBatchKMeans(
        4,
        epochs=5,
        batch_size=8,
        initialization="first",
        revival="off",
        accumulation="deterministic",
    )
    a = split.fit_batch(MLBatch(x), key=_KEY)
    b = split.fit_batch(MLBatch(x + 128), key=_KEY)
    c = off.fit_batch(MLBatch(x), key=_KEY)
    assert a.diagnostics.minibatch.revivals > 0
    assert c.diagnostics.minibatch.revivals == 0
    assert jnp.sum(a.as_trainable().active_clusters) == 4  # ty: ignore[unresolved-attribute]
    assert jnp.sum(c.as_trainable().active_clusters) < 4  # ty: ignore[unresolved-attribute]
    assert a.diagnostics.objective < c.diagnostics.objective * 0.1
    np.testing.assert_allclose(
        b.as_trainable().centers,  # ty: ignore[unresolved-attribute]
        a.as_trainable().centers + 128,  # ty: ignore[unresolved-attribute]
        atol=1e-10,
    )
    np.testing.assert_array_equal(b.model(x + 128), a.model(x))


def test_single_pass_keeps_reviving() -> None:
    x = jnp.concatenate(
        (
            jnp.zeros((4, 1), dtype=jnp.float64),
            jnp.repeat(
                jnp.asarray([[0.0], [20.0], [40.0], [60.0]], dtype=jnp.float64),
                20,
                axis=0,
            ),
        )
    )
    results = {
        revival: MiniBatchKMeans(
            4,
            epochs=1.0,
            batch_size=8,
            initialization="first",
            revival=revival,
            accumulation="deterministic",
        ).fit_batch(MLBatch(x), key=_KEY)
        for revival in ("split", "off")
    }
    split, off = results["split"].diagnostics, results["off"].diagnostics
    assert split.minibatch.revivals > 0
    assert off.minibatch.revivals == 0
    assert split.objective < off.objective


def test_schedule_laws() -> None:
    x = jnp.arange(2048, dtype=jnp.float64)[:, None]
    quality = (
        MiniBatchKMeans(4, epochs=2).fit_batch(MLBatch(x), key=_KEY).diagnostics.minibatch
    )
    throughput = (
        MiniBatchKMeans(4, epochs=2, batch_law="throughput")
        .fit_batch(MLBatch(x), key=_KEY)
        .diagnostics.minibatch
    )
    assert quality.batch_size & (quality.batch_size - 1) == 0
    assert quality.batch_size <= 1024
    assert quality.batch_size * 8 == throughput.batch_size
    assert not quality.in_fitted_domain


def test_explicit_schedule_and_fractional_epochs() -> None:
    result = MiniBatchKMeans(2, epochs=2.5, batch_size=4, initial_step=0.5).fit_batch(
        MLBatch(_X), key=_KEY
    )
    evidence = result.diagnostics.minibatch
    assert evidence.batch_size == 4
    assert evidence.batch_law is None
    assert evidence.initial_step == 0.5
    assert evidence.epochs == 2.5
    assert evidence.epoch_fractions == (1.0, 0.5, 1.0)
    assert evidence.updates == 5
    assert result.diagnostics.iterations == evidence.updates


def test_half_epoch_closes_half_the_rows() -> None:
    result = MiniBatchKMeans(2, epochs=0.5, batch_size=2).fit_batch(MLBatch(_X), key=_KEY)
    assert jnp.sum(result.diagnostics.cluster_mass) == 3
    assert result.diagnostics.minibatch.epoch_objectives.shape == (1,)


def test_masked_contents_do_not_affect_fit() -> None:
    mask = jnp.asarray([True, True, True, True, False, False])
    recipe = MiniBatchKMeans(2, accumulation="deterministic")
    a = recipe.fit_batch(MLBatch(_X.at[4:].set(jnp.nan), sample_mask=mask), key=_KEY)
    b = recipe.fit_batch(MLBatch(_X.at[4:].set(1e9), sample_mask=mask), key=_KEY)
    np.testing.assert_array_equal(a.as_trainable().centers, b.as_trainable().centers)  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(a.diagnostics.objective, b.diagnostics.objective)


def test_case_independence() -> None:
    recipe = MiniBatchKMeans(2, accumulation="deterministic")
    x = jnp.stack((_X, _X))
    a = recipe.fit_batch(MLBatch(x), key=_KEY)
    b = recipe.fit_batch(MLBatch(x.at[1].set(_X * 10)), key=_KEY)
    np.testing.assert_array_equal(
        a.as_trainable().centers[0],  # ty: ignore[unresolved-attribute]
        b.as_trainable().centers[0],  # ty: ignore[unresolved-attribute]
    )
    np.testing.assert_array_equal(a.diagnostics.objective[0], b.diagnostics.objective[0])
    np.testing.assert_array_equal(
        a.diagnostics.minibatch.epoch_objectives[0],
        b.diagnostics.minibatch.epoch_objectives[0],
    )


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("nonfinite", ML_NONFINITE),
        ("insufficient", ML_INSUFFICIENT_DATA),
        ("nonconverged", ML_NONCONVERGED),
        ("success", ML_SUCCESS),
    ],
    ids=["nonfinite", "insufficient", "nonconverged", "success"],
)
def test_status(kind: str, expected: int) -> None:
    x = _X.at[0, 0].set(jnp.nan) if kind == "nonfinite" else _X
    mask = (
        jnp.asarray([True, False, False, False, False, False])
        if kind == "insufficient"
        else None
    )
    recipe = (
        MiniBatchKMeans(2, batch_size=2, tolerance=0.0)
        if kind == "nonconverged"
        else MiniBatchKMeans(2, batch_size=6 if kind == "success" else 2)
    )
    result = recipe.fit_batch(MLBatch(x, sample_mask=mask), key=_KEY)
    assert result.status == expected
    if kind == "success":
        assert result.valid


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0.0},
        {"epochs": -1.0},
        {"epochs": float("inf")},
        {"epochs": float("nan")},
        {"batch_size": 0},
        {"initial_step": 0.0},
        {"initial_step": 1.5},
        {"revival_threshold": 0.0},
        {"revival_threshold": 1.0},
        {"tolerance": -1.0},
        {"tolerance": float("nan")},
        {"workspace_bytes": 0},
    ],
    ids=[
        "zero-epochs",
        "negative-epochs",
        "infinite-epochs",
        "nan-epochs",
        "zero-batch",
        "zero-step",
        "large-step",
        "zero-threshold",
        "unit-threshold",
        "negative-tolerance",
        "nan-tolerance",
        "zero-workspace",
    ],
)
def test_constructor_refuses_invalid_numbers(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        MiniBatchKMeans(2, **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "selector", ["batch_law", "revival", "initialization", "accumulation"]
)
def test_constructor_refuses_invalid_selectors(selector: str) -> None:
    with pytest.raises(ValueError):
        MiniBatchKMeans(2, **{selector: "invalid"})  # ty: ignore[invalid-argument-type]


def test_diagnostics_method_requires_matching_evidence() -> None:
    result = MiniBatchKMeans(2).fit_batch(MLBatch(_X), key=_KEY)
    evidence = result.diagnostics.minibatch
    fields = dict(
        valid=True,
        status=0,
        objective=0.0,
        iterations=1,
        effective_samples=6.0,
        cluster_mass=jnp.ones(2),
        active_clusters=jnp.ones(2, dtype=jnp.bool_),
        empty_clusters_seen=False,
        converged=True,
    )
    with pytest.raises(ValueError, match="Mini-batch evidence"):
        ClusterDiagnostics(**fields, method="ema-mini-batch-k-means")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Mini-batch evidence"):
        ClusterDiagnostics(**fields, method="k-means", minibatch=evidence)  # ty: ignore[invalid-argument-type]
    full = KMeans(2, initialization="first").fit_batch(MLBatch(_X))
    assert full.diagnostics.minibatch is None


def test_key_refusals() -> None:
    recipe = MiniBatchKMeans(2)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(_X))
    with pytest.raises((TypeError, ValueError)):
        recipe.fit_batch(MLBatch(_X), key=jax.random.PRNGKey(0))


@pytest.mark.strict_jax
def test_float32_is_preserved() -> None:
    result = MiniBatchKMeans(2).fit_batch(MLBatch(_X.astype(jnp.float32)), key=_KEY)
    assert result.as_trainable().centers.dtype == jnp.float32  # ty: ignore[unresolved-attribute]
    assert result.diagnostics.objective.dtype == jnp.float32


def test_artifact_round_trip(tmp_path: Path) -> None:
    result = MiniBatchKMeans(2).fit_batch(MLBatch(_X), key=_KEY)
    path = tmp_path / "minibatch.phxml"
    save_ml_artifact(path, result.as_trainable(), fit_result=result)
    restored = read_ml_artifact(path)
    np.testing.assert_array_equal(restored.model(_X), result.model(_X))
    assert restored.manifest.fit["method"] == "ema-mini-batch-k-means"  # ty: ignore[not-subscriptable]


def test_deterministic_replay() -> None:
    recipe = MiniBatchKMeans(2, accumulation="deterministic")
    a = recipe.fit_batch(MLBatch(_X), key=_KEY)
    b = recipe.fit_batch(MLBatch(_X), key=_KEY)
    np.testing.assert_array_equal(a.as_trainable().centers, b.as_trainable().centers)  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(a.diagnostics.objective, b.diagnostics.objective)
