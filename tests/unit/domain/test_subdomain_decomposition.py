#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _batch(domain: Any, count: Any = 9) -> Any:
    return domain.component().sample(
        phx.domain.PointSampling(
            count,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
        key=jr.key(0),
    )


def test_subdomain_decomposition_scenario_1() -> None:
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


def test_partition_of_unity_uses_sparse_active_fields_and_differentiable_windows() -> (
    None
):
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

    def reference(value: Any) -> Any:
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


@pytest.mark.parametrize(
    ("domain", "count", "periodic", "identified"),
    (
        pytest.param(phx.domain.Interval1d(0.0, 2.0), 1, True, ((0, 0.0, 2.0),), id="1d"),
        pytest.param(
            phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 3.0])),
            (2, 1),
            (False, True),
            ((1, 0.0, 3.0), (1, 0.0, 3.0)),
            id="box-one-patch-per-periodic-column",
        ),
    ),
)
def test_single_partition_periodic_axis_produces_a_self_seam(
    domain: Any, count: Any, periodic: Any, identified: Any
) -> None:
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", count, periodic=periodic)
    seams = tuple(
        pairing for pairing in cover.pairings if pairing.topology == "periodic-interface"
    )

    assert len(seams) == len(identified)
    for seam, (axis, lower, upper) in zip(seams, identified, strict=True):
        assert seam.self_seam
        assert seam.identification is not None
        assert (seam.identification.component, seam.identification.period) == (
            axis,
            upper - lower,
        )
        patch = cover.patch(seam.left_patch_id)
        batch = seam.component.sample(phx.domain.PointSampling(5), key=jr.key(5))
        assert seam.audit(batch, patch, patch).verified
        coordinate = patch.domain.Function("x")(lambda x, axis=axis: x[axis])
        np.testing.assert_allclose(
            seam.trace(coordinate, side=seam.source_side)(batch).data, lower
        )
        np.testing.assert_allclose(
            seam.trace(coordinate, side=seam.target_side)(batch).data, upper
        )
    # A self-seam is not a neighbor relation.
    for patch_id, neighbors in cover.adjacency:
        assert patch_id not in neighbors
    assert cover.structural_evidence().verified


def test_single_partition_without_periodicity_has_no_seam() -> None:
    cover = phx.domain.cartesian_subdomain_cover(phx.domain.Interval1d(0.0, 2.0), "x", 1)

    assert cover.pairings == ()
    assert cover.adjacency == (("patch-0000", ()),)


def test_only_identified_periodic_pairings_may_join_a_patch_to_itself() -> None:
    domain = phx.domain.Interval1d(0.0, 2.0)
    identification = phx.domain.PeriodicIdentification(domain, "x")
    face = identification.face("lower")
    upper = identification.face_map("upper")
    lower = identification.face_map("lower")

    def pairing(**options: Any) -> Any:
        return phx.domain.PairedSupport(
            face,
            upper,
            lower,
            pairing_id="seam",
            left_patch_id="patch",
            right_patch_id="patch",
            **options,
        )

    with pytest.raises(ValueError, match="two distinct patches"):
        pairing()
    with pytest.raises(TypeError, match="requires a PeriodicIdentification"):
        pairing(topology="periodic-interface")
    with pytest.raises(ValueError, match="Only periodic-interface pairings"):
        pairing(identification=identification)
    assert pairing(topology="periodic-interface", identification=identification).self_seam


def test_point_interfaces_have_counting_support_without_dummy_sampling_axes() -> None:
    domain = phx.domain.Interval1d(0.0, 2.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2, periodic=True)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4))
    for seam in cover.pairings:
        batch = seam.component.sample(phx.domain.PointSampling(9))
        assert batch.structure.blocks == ()
        assert seam.component.mass.value == pytest.approx(1.0)
        left = cover.patch(seam.left_patch_id)
        right = cover.patch(seam.right_patch_id)
        assert seam.audit(batch, left, right).verified
        value = phx.integration.integrate(3.0, phx.integration.over(seam.component), plan)
        assert value.value.data == pytest.approx(3.0)


def test_periodic_audit_rejects_reversed_wrapped_self_seam() -> None:
    cover = phx.domain.cartesian_subdomain_cover(
        phx.domain.Interval1d(0.0, 2.0), "x", 1, periodic=True
    )
    seam = cover.pairings[0]
    reversed_seam = phx.domain.PairedSupport(
        seam.component,
        dict(seam.right_coordinates),
        dict(seam.left_coordinates),
        pairing_id="reversed",
        left_patch_id=seam.left_patch_id,
        right_patch_id=seam.right_patch_id,
        normal=seam.normal,
        topology="periodic-interface",
        identification=seam.identification,
    )
    batch = seam.component.sample(phx.domain.PointSampling(4))
    (patch,) = cover.patches
    assert seam.audit(batch, patch, patch).verified
    evidence = reversed_seam.audit(batch, patch, patch)
    assert not evidence.verified
    assert evidence.maximum_map_mismatch == pytest.approx(2.0)


def test_nonuniform_cover_capacity_counts_wide_support_crossing_narrow_cells() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(
        domain, "x", boundaries=(0.0, 0.1, 0.2, 1.0), overlap_fraction=0.4
    )
    evidence = cover.audit(
        domain.component().points({"x": jnp.asarray([0.09], dtype=jnp.float64)})
    )
    assert evidence.max_coverage == 3
    assert cover.maximum_overlap == 3
    assert evidence.verified


def test_cover_endpoints_must_match_bounds_without_tolerance_gaps() -> None:
    with pytest.raises(ValueError, match="endpoints must match"):
        phx.domain.cartesian_subdomain_cover(
            phx.domain.Interval1d(0.0, 2.0), "x", boundaries=(0.0, 2.000001)
        )


def test_cover_revision_binds_native_static_map_geometry_and_canonical_order() -> None:
    domain = phx.domain.HyperRectangle(
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([2.0, 3.0], dtype=jnp.float64),
    )
    cover = phx.domain.cartesian_subdomain_cover(
        domain, "x", (1, 2), periodic=(True, False), cover_id="named-cover"
    )
    reordered = phx.domain.SubdomainCover(
        domain,
        tuple(reversed(cover.patches)),
        tuple(reversed(cover.pairings)),
        cover_id=cover.cover_id,
        exact_coverage=True,
        maximum_overlap=cover.maximum_overlap,
    )
    assert cover.revision == reordered.revision
    seam = next(
        pairing for pairing in cover.pairings if pairing.topology == "periodic-interface"
    )
    changed = phx.domain.PairedSupport(
        seam.component,
        dict(seam.right_coordinates),
        dict(seam.left_coordinates),
        pairing_id=seam.pairing_id,
        left_patch_id=seam.left_patch_id,
        right_patch_id=seam.right_patch_id,
        normal=seam.normal,
        topology=seam.topology,
        identification=seam.identification,
    )
    changed_cover = phx.domain.SubdomainCover(
        domain,
        cover.patches,
        tuple(changed if pairing is seam else pairing for pairing in cover.pairings),
        cover_id=cover.cover_id,
        exact_coverage=True,
        maximum_overlap=cover.maximum_overlap,
    )
    assert cover.revision != changed_cover.revision


def test_periodic_audit_uses_ambient_endpoints_for_normalized_local_charts() -> None:
    ambient = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    local = phx.domain.ScalarInterval(-1.0, 1.0, label="xi")
    patch = phx.domain.SubdomainPatch(
        local,
        local.component(),
        ambient.Function()(1.0),
        {"xi": ambient.Function("x")(lambda x: x - 1.0)},
        {"x": local.Function("xi")(lambda xi: xi + 1.0)},
        patch_id="normalized",
    )
    identification = phx.domain.PeriodicIdentification(ambient, "x")
    seam = phx.domain.PairedSupport(
        identification.face("lower"),
        {"xi": ambient.Function()(1.0)},
        {"xi": ambient.Function()(-1.0)},
        pairing_id="normalized-seam",
        left_patch_id=patch.patch_id,
        right_patch_id=patch.patch_id,
        normal=identification.normal(),
        topology="periodic-interface",
        identification=identification,
    )
    batch = seam.component.sample(phx.domain.PointSampling(3))
    evidence = seam.audit(batch, patch, patch)
    assert evidence.verified
    assert evidence.maximum_map_mismatch == pytest.approx(0.0)
