#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.parameters import LowRankUpdate


_KEY0 = jr.key(0)
_KEY1 = jr.key(1)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rank": True}, TypeError),
        ({"rank": 0}, ValueError),
        ({"rank": -1}, ValueError),
        ({"rank": 1, "alpha": 0.0}, ValueError),
        ({"rank": 1, "alpha": jnp.inf}, ValueError),
        ({"rank": 1, "stddev": 0.0}, ValueError),
        ({"rank": 1, "scaling": "invalid"}, ValueError),
    ],
)
def test_low_rank_spec_rejects_invalid_coordinates(kwargs, error):
    with pytest.raises(error):
        phx.nn.parameters.LowRankSpec(**kwargs)


def _linear(*, key=_KEY0, rwf=False, weight_transform=None):
    return phx.nn.layers.Linear(
        in_size=3,
        out_size=2,
        rwf=rwf,
        weight_transform=weight_transform,
        key=key,
    )


def _adapt(model, *, rank=2, scaling="rank", key=_KEY1):
    paths = phx.nn.parameters.low_rank_sites(model)
    specs = {
        path: phx.nn.parameters.LowRankSpec(
            rank=rank,
            alpha=rank,
            scaling=scaling,
        )
        for path in paths
    }
    return phx.nn.parameters.adapt_low_rank(model, specs, key=key)


def test_low_rank_linear_preserves_initial_function_and_factorizes_batches():
    base = _linear()
    adapted, report = _adapt(base)
    assert isinstance(adapted.weight, LowRankUpdate)
    assert report.sites[0].path == ".weight"
    assert report.base_parameter_count == 6
    assert report.adapter_parameter_count == 10
    assert report.parameter_ratio == pytest.approx(10.0 / 6.0)

    inputs = jnp.arange(24.0).reshape((2, 4, 3))
    assert jnp.array_equal(adapted(inputs), base(inputs))
    assert jnp.array_equal(jax.jit(adapted)(inputs), base(inputs))
    assert jnp.allclose(jax.vmap(adapted)(inputs[0]), base(inputs[0]))

    update = adapted.weight
    nonzero = LowRankUpdate.from_factors(
        update.base,
        jnp.arange(4.0, dtype=update.dtype).reshape((2, 2)) / 7.0,
        jnp.arange(6.0, dtype=update.dtype).reshape((2, 3)) / 5.0,
        alpha=update.alpha,
        scaling=update.scaling,
    )
    changed = eqx.tree_at(lambda model: model.weight, adapted, nonzero)
    merged = phx.nn.parameters.merge_low_rank(changed)
    assert isinstance(changed.weight, LowRankUpdate)
    assert isinstance(merged.weight, jax.Array)
    assert jnp.allclose(changed(inputs), merged(inputs), rtol=1e-12, atol=1e-12)
    assert jnp.array_equal(base.weight, changed.weight.base)


def test_low_rank_gradients_stop_base_and_reach_both_nonzero_factors():
    adapted, _ = _adapt(_linear())
    update = adapted.weight
    changed_update = LowRankUpdate.from_factors(
        update.base,
        jnp.full_like(update.left, 0.2),
        jnp.full_like(update.right, -0.1),
        alpha=update.alpha,
        scaling=update.scaling,
    )
    changed = eqx.tree_at(lambda model: model.weight, adapted, changed_update)
    inputs = jnp.arange(6.0).reshape((2, 3))

    gradients = eqx.filter_grad(lambda model: jnp.sum(model(inputs) ** 2))(changed)
    assert jnp.array_equal(gradients.weight.base, jnp.zeros_like(update.base))
    assert jnp.any(gradients.weight.left != 0.0)
    assert jnp.any(gradients.weight.right != 0.0)


def test_rank_stabilized_scaling_and_rwf_compose_without_materializing():
    base = _linear(rwf=True)
    adapted, report = _adapt(base, scaling="sqrt_rank")
    assert report.sites[0].scaling == "sqrt_rank"
    assert adapted.weight.scale == pytest.approx(
        adapted.weight.alpha / jnp.sqrt(adapted.weight.rank)
    )
    inputs = jnp.arange(12.0).reshape((4, 3))
    assert jnp.array_equal(adapted(inputs), base(inputs))

    update = adapted.weight
    changed = eqx.tree_at(
        lambda layer: layer.weight,
        adapted,
        LowRankUpdate.from_factors(
            update.base,
            jnp.full_like(update.left, 0.2),
            jnp.full_like(update.right, -0.1),
            alpha=update.alpha,
            scaling=update.scaling,
        ),
    )
    merged = phx.nn.parameters.merge_low_rank(changed)
    assert merged.random_weight_factorization
    assert jnp.allclose(changed(inputs), merged(inputs), rtol=1e-12, atol=1e-12)


def test_low_rank_paths_keys_and_subspace_are_deterministic():
    model = phx.nn.models.MLP(
        in_size=3,
        out_size=2,
        width_size=4,
        depth=3,
        rwf=False,
        key=jr.key(4),
    )
    paths = phx.nn.parameters.low_rank_sites(model)
    assert paths == tuple(f".layers[{index}].weight" for index in range(4))
    specs = {path: phx.nn.parameters.LowRankSpec(2) for path in reversed(paths)}
    first, first_report = phx.nn.parameters.adapt_low_rank(model, specs, key=jr.key(5))
    second, second_report = phx.nn.parameters.adapt_low_rank(model, specs, key=jr.key(5))
    assert first_report == second_report
    for left, right in zip(first.layers, second.layers, strict=True):
        assert jnp.array_equal(left.weight.right, right.weight.right)
    assert not jnp.array_equal(
        first.layers[1].weight.right,
        first.layers[2].weight.right,
    )

    subspace = phx.nn.parameters.low_rank_parameter_subspace(first)
    expected_factor_paths = tuple(
        factor_path
        for weight_path in paths
        for factor_path in (f"{weight_path}.left", f"{weight_path}.right")
    )
    assert subspace.leaf_paths == expected_factor_paths
    subspace.validate_root(first)
    moved = eqx.tree_at(
        lambda adapted: adapted.layers[0].weight.left,
        first,
        first.layers[0].weight.left + 0.5,
    )
    rebased = subspace.rebase(moved)
    assert jnp.array_equal(
        rebased.reconstruct(rebased.initial).layers[0].weight.left,
        moved.layers[0].weight.left,
    )
    with pytest.raises(ValueError, match="does not describe"):
        subspace.validate_root(moved)


def test_low_rank_scan_cache_safely_falls_back_after_model_surgery():
    scanned = phx.nn.models.MLP(
        in_size=3,
        out_size=2,
        width_size=4,
        depth=4,
        rwf=False,
        scan=True,
        key=jr.key(6),
    )
    unrolled = eqx.tree_at(
        lambda model: (model.scan, model._scan_enabled, model._scan_static),
        scanned,
        (False, False, None),
        is_leaf=lambda value: value is None,
    )
    adapted_scan, _ = _adapt(scanned, key=jr.key(7))
    adapted_loop, _ = _adapt(unrolled, key=jr.key(7))
    value = jnp.arange(3.0)
    assert jnp.allclose(adapted_scan(value), adapted_loop(value))
    scan_grad = eqx.filter_grad(lambda model: jnp.sum(model(value)))(adapted_scan)
    loop_grad = eqx.filter_grad(lambda model: jnp.sum(model(value)))(adapted_loop)
    assert jnp.allclose(
        scan_grad.layers[1].weight.left,
        loop_grad.layers[1].weight.left,
    )


def test_low_rank_adaptation_rejects_unsupported_or_ambiguous_sites():
    with pytest.raises(ValueError, match="weight transform"):
        phx.nn.parameters.adapt_low_rank(
            _linear(
                weight_transform=phx.nn.parameters.PositiveTransform(),
            ),
            {".weight": phx.nn.parameters.LowRankSpec(1)},
        )
    complex_layer = eqx.tree_at(
        lambda layer: layer.weight,
        _linear(),
        _linear().weight.astype(jnp.complex128),
    )
    with pytest.raises(TypeError, match="real inexact"):
        phx.nn.parameters.adapt_low_rank(
            complex_layer,
            {".weight": phx.nn.parameters.LowRankSpec(1)},
        )
    layer = _linear()
    with pytest.raises(ValueError, match="aliased"):
        phx.nn.parameters.low_rank_sites((layer, layer))
    with pytest.raises(ValueError, match="Unknown"):
        phx.nn.parameters.adapt_low_rank(
            layer,
            {".missing": phx.nn.parameters.LowRankSpec(1)},
        )


def test_mlp_kfac_metadata_rejects_adapter_until_merged():
    model = phx.nn.models.MLP(
        in_size=2,
        out_size=1,
        hidden_sizes=(),
        rwf=False,
        key=jr.key(8),
    )
    adapted, _ = _adapt(model, rank=1, key=jr.key(9))
    assert adapted.kfac_affine_blocks()[0].parameterization == "low_rank_update"
    merged = phx.nn.parameters.merge_low_rank(adapted)
    assert merged.kfac_affine_blocks()[0].parameterization == "direct"
