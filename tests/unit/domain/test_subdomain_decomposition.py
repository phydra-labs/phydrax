#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _batch(domain, count=9):
    return domain.component().sample(
        phx.domain.PointSampling(
            count,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
        key=jr.key(0),
    )


def test_cartesian_cover_has_exact_local_domains_pairing_and_audit():
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        3,
        overlap_fraction=0.2,
    )

    assert cover.patch_ids == ("patch-0000", "patch-0001", "patch-0002")
    assert cover.pairing_ids == (
        "interface-axis-0-patch-0000-patch-0001",
        "interface-axis-0-patch-0001-patch-0002",
    )
    assert cover.adjacency == (
        ("patch-0000", ("patch-0001",)),
        ("patch-0001", ("patch-0000", "patch-0002")),
        ("patch-0002", ("patch-0001",)),
    )
    assert cover.structural_evidence().verified

    evidence = cover.audit(_batch(domain, 32))
    assert evidence.scope == "sampled"
    assert evidence.uncovered_points == 0
    assert evidence.max_coverage <= 2
    assert evidence.verified

    first = cover.patches[0].domain
    middle = cover.patches[1].domain
    np.testing.assert_allclose((first.start, first.end), (0.0, 0.4))
    np.testing.assert_allclose(
        (middle.start, middle.end),
        (1.0 / 3.0 - 1.0 / 15.0, 2.0 / 3.0 + 1.0 / 15.0),
    )


def test_cartesian_cover_splits_one_scalar_factor_in_a_product_domain():
    space = phx.domain.Interval1d(-1.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    domain = space @ time
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "t",
        2,
        overlap_fraction=0.1,
    )
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            patch.patch_id: patch.domain.Function("t")(lambda value: value)
            for patch in cover.patches
        },
    )
    pairing = cover.pairings[0]
    batch = pairing.component.sample(
        phx.domain.PointSampling(5),
        key=jr.key(4),
    )

    assert all(patch.domain.labels == ("x", "t") for patch in cover.patches)
    np.testing.assert_allclose(
        pairing.trace(family.fields[0], side="left")(batch).data,
        0.5,
    )
    np.testing.assert_allclose(
        pairing.trace(family.fields[1], side="right")(batch).data,
        0.5,
    )


def test_partition_of_unity_uses_sparse_active_fields_and_differentiable_windows():
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        2,
        overlap_fraction=0.2,
    )
    left, right = cover.patches
    fields = {
        left.patch_id: left.domain.Function()(0.0),
        right.patch_id: right.domain.Function()(1.0),
    }
    family = phx.domain.LocalFieldFamily("u", cover, fields)
    assembled = phx.domain.partition_of_unity_field(family)

    left_window = left.window
    right_window = right.window
    assert left_window is not None and right_window is not None

    def reference(value):
        point = jnp.asarray([value])
        left_value = left_window.func(point, key=jr.key(0))
        right_value = right_window.func(point, key=jr.key(0))
        return right_value / (left_value + right_value)

    value = 0.47
    observed = assembled.func(jnp.asarray([value]), key=jr.key(0))
    observed_gradient = jax.grad(
        lambda coordinate: assembled.func(
            jnp.asarray([coordinate]),
            key=jr.key(0),
        )
    )(value)
    observed_curvature = jax.grad(
        jax.grad(
            lambda coordinate: assembled.func(
                jnp.asarray([coordinate]),
                key=jr.key(0),
            )
        )
    )(value)

    np.testing.assert_allclose(observed, reference(value), atol=1.0e-12)
    np.testing.assert_allclose(
        observed_gradient,
        jax.grad(reference)(value),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        observed_curvature,
        jax.grad(jax.grad(reference))(value),
        atol=1.0e-9,
    )

    safe_fields = {
        left.patch_id: left.domain.Function()(2.0),
        right.patch_id: right.domain.Function()(jnp.nan),
    }
    sparse = phx.domain.partition_of_unity_field(
        phx.domain.LocalFieldFamily("safe", cover, safe_fields)
    )
    assert jnp.isfinite(sparse.func(jnp.asarray([0.1]), key=jr.key(1)))


def test_normalized_coordinate_is_unclipped_and_broken_field_is_side_aware():
    domain = phx.domain.Interval1d(-2.0, 2.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2)
    left, right = cover.patches
    normalized = phx.domain.normalized_patch_coordinate(left, "x")

    assert normalized.func(jnp.asarray([-2.0]), key=jr.key(0))[0] == pytest.approx(-1.0)
    assert normalized.func(jnp.asarray([0.0]), key=jr.key(0))[0] == pytest.approx(1.0)
    assert normalized.func(jnp.asarray([0.5]), key=jr.key(0))[0] > 1.0

    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            left.patch_id: left.domain.Function()(1.0),
            right.patch_id: right.domain.Function()(3.0),
        },
    )
    broken = phx.domain.broken_field(family)
    pairing = cover.pairings[0]
    batch = pairing.component.sample(
        phx.domain.PointSampling(4),
        key=jr.key(2),
    )

    np.testing.assert_allclose(
        broken.trace(pairing.pairing_id, side="left")(batch).data, 1.0
    )
    np.testing.assert_allclose(
        broken.trace(pairing.pairing_id, side="right")(batch).data,
        3.0,
    )
    with pytest.raises(TypeError):
        broken.as_domain_function()
    owned = broken.as_domain_function(ownership="first")
    np.testing.assert_allclose(
        owned.func(jnp.asarray([-1.0]), key=jr.key(3)),
        1.0,
    )
    evaluate = jax.jit(
        lambda coordinate: owned.func(
            jnp.asarray([coordinate]),
            key=jr.key(3),
        )
    )
    np.testing.assert_allclose(evaluate(-1.0), 1.0)
