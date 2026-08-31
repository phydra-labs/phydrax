#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError


_KEY0 = jr.key(0)


def _base(*, key=_KEY0):
    return phx.nn.models.MLP(
        in_size=3,
        out_size=2,
        width_size=4,
        depth=2,
        rwf=False,
        key=key,
    )


def _adapted(base):
    paths = phx.nn.parameters.low_rank_sites(base)
    model, _ = phx.nn.parameters.adapt_low_rank(
        base,
        {path: phx.nn.parameters.LowRankSpec(2, scaling="sqrt_rank") for path in paths},
        key=jr.key(1),
    )
    first = model.layers[0].weight
    changed = phx.nn.parameters.LowRankUpdate.from_factors(
        first.base,
        jnp.full_like(first.left, 0.125),
        jnp.full_like(first.right, -0.25),
        alpha=first.alpha,
        scaling=first.scaling,
    )
    return eqx.tree_at(lambda value: value.layers[0].weight, model, changed)


def test_low_rank_adapter_round_trip_binds_base_and_preserves_merge(tmp_path):
    base = _base()
    adapted = _adapted(base)
    destination = tmp_path / "adapter.phx"
    phx.nn.parameters.save_low_rank_adapter(
        destination,
        adapted,
        provenance={"task": "unit"},
    )
    restored = phx.nn.parameters.read_low_rank_adapter(destination, base)
    inputs = jnp.arange(18.0).reshape((2, 3, 3))

    assert jnp.allclose(restored.model(inputs), adapted(inputs))
    assert jnp.allclose(
        phx.nn.parameters.merge_low_rank(restored.model)(inputs),
        phx.nn.parameters.merge_low_rank(adapted)(inputs),
    )
    assert restored.manifest.provenance == {"task": "unit"}
    assert all(site.scaling == "sqrt_rank" for site in restored.manifest.sites)
    assert tuple(site.path for site in restored.manifest.sites) == (
        ".layers[0].weight",
        ".layers[1].weight",
        ".layers[2].weight",
    )
    subspace = phx.nn.parameters.low_rank_parameter_subspace(restored.model)
    assert subspace.total_dimension == sum(
        site.adapter_parameter_count for site in restored.manifest.sites
    )


def test_low_rank_adapter_rejects_wrong_base_content_or_structure(tmp_path):
    base = _base()
    destination = tmp_path / "adapter.phx"
    phx.nn.parameters.save_low_rank_adapter(destination, _adapted(base))
    wrong_content = _base(key=jr.key(2))
    with pytest.raises(ValueError, match="content mismatch"):
        phx.nn.parameters.read_low_rank_adapter(destination, wrong_content)
    wrong_structure = phx.nn.models.MLP(
        in_size=4,
        out_size=2,
        width_size=4,
        depth=2,
        rwf=False,
        key=jr.key(0),
    )
    with pytest.raises(ValueError, match="structure mismatch"):
        phx.nn.parameters.read_low_rank_adapter(destination, wrong_structure)


def test_low_rank_adapter_rejects_payload_corruption_and_dense_save(tmp_path):
    with pytest.raises(ValueError, match="without adapters"):
        phx.nn.parameters.save_low_rank_adapter(tmp_path / "dense.phx", _base())

    destination = tmp_path / "adapter.phx"
    phx.nn.parameters.save_low_rank_adapter(destination, _adapted(_base()))
    payload = bytearray(destination.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    destination.write_bytes(payload)
    with pytest.raises(ArrayArchiveCorruptionError):
        phx.nn.parameters.read_low_rank_adapter(destination, _base())


def test_adapted_model_round_trips_through_native_ml_artifact(tmp_path):
    adapted = _adapted(_base())
    destination = tmp_path / "adapted.phxml"
    phx.ml.artifacts.save_ml_artifact(destination, adapted)
    restored = phx.ml.artifacts.read_ml_artifact(destination).model
    inputs = jnp.arange(9.0).reshape((3, 3))

    assert isinstance(restored.layers[0].weight, phx.nn.parameters.LowRankUpdate)
    assert jnp.allclose(restored(inputs), adapted(inputs))
    assert jnp.allclose(
        phx.nn.parameters.merge_low_rank(restored)(inputs),
        phx.nn.parameters.merge_low_rank(adapted)(inputs),
    )
