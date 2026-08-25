from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
import jax.numpy as jnp

from phydrax.discretization.finite_volume._contact_angle import (
    ContactAngleCondition,
    ContactAngleStatus,
    EmbeddedBoundaryContactAngleSet,
    reconstruct_wall_interface_normal,
)


def test_condition_validates_angle_tolerance_and_body_tag():
    condition = ContactAngleCondition(7, np.pi / 2.0, 1.0e-5, "wall-7")
    assert condition.body_tag == 7
    assert condition.condition_id == "wall-7"
    for angle in (0.0, np.pi, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="angle"):
            ContactAngleCondition(7, angle, 1.0e-5, "wall-7")
    with pytest.raises(ValueError, match="tolerance"):
        ContactAngleCondition(7, np.pi / 2.0, -1.0, "wall-7")
    with pytest.raises(ValueError, match="tolerance"):
        ContactAngleCondition(7, np.pi / 2.0, np.inf, "wall-7")
    with pytest.raises(ValueError, match="body_tag"):
        ContactAngleCondition(-1, np.pi / 2.0, 1.0e-5, "wall-7")
    with pytest.raises(TypeError, match="body_tag"):
        ContactAngleCondition(True, np.pi / 2.0, 1.0e-5, "wall-7")


def test_oblique_wall_preserves_norm_and_declared_cosine():
    wall = np.asarray((1.0, 2.0))
    plic = np.asarray((-2.0, 1.0))
    condition = ContactAngleCondition(3, np.pi / 3.0, 1.0e-5, "oblique")
    result = reconstruct_wall_interface_normal(plic, wall, condition)
    wall_unit = wall / np.linalg.norm(wall)
    normal = np.asarray(result.normal)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(
        normal @ wall_unit,
        np.cos(condition.angle),
        atol=condition.tolerance,
    )
    assert int(np.asarray(result.status)) == int(ContactAngleStatus.SUCCESS)
    assert result.body_tag == condition.body_tag
    assert result.condition_id == condition.condition_id


@pytest.mark.parametrize(
    "angle",
    (1.0e-6, np.pi / 2.0, np.pi - 1.0e-6),
)
def test_limiting_contact_angles(angle):
    condition = ContactAngleCondition(7, angle, 1.0e-5, f"limit-{angle}")
    result = reconstruct_wall_interface_normal(
        np.asarray((1.0, 0.2)),
        np.asarray((-0.3, 1.0)),
        condition,
    )
    wall_unit = np.asarray((-0.3, 1.0)) / np.linalg.norm((-0.3, 1.0))
    normal = np.asarray(result.normal)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(
        normal @ wall_unit,
        np.cos(angle),
        atol=condition.tolerance,
    )
    assert int(np.asarray(result.status)) == int(ContactAngleStatus.SUCCESS)


def test_contact_angle_set_requires_exact_coverage_and_fresh_ids():
    first = ContactAngleCondition(3, np.pi / 2.0, 1.0e-5, "first")
    second = ContactAngleCondition(8, np.pi / 4.0, 1.0e-5, "second")
    policies = EmbeddedBoundaryContactAngleSet(
        {3: first, 8: second}, geometry_id="geometry-1", plic_id="plic-1"
    )
    policies.validate_body_tags(np.asarray((3, 8, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="missing policy tags"):
        policies.validate_body_tags(np.asarray((3,), dtype=np.int32))
    with pytest.raises(ValueError, match="extra policy tags"):
        policies.validate_body_tags(np.asarray((3, 8, 9), dtype=np.int32))
    with pytest.raises(ValueError, match="stale"):
        policies.validate_bindings("geometry-stale", "plic-1")
    with pytest.raises(ValueError, match="stale"):
        policies.validate_bindings("geometry-1", "plic-stale")
    result = policies.reconstruct(np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0)), 3)
    assert result.geometry_id == "geometry-1"
    assert result.plic_id == "plic-1"


def test_contact_angle_set_rejects_mismatched_or_duplicate_policies():
    with pytest.raises(ValueError, match="key"):
        EmbeddedBoundaryContactAngleSet(
            {4: ContactAngleCondition(3, np.pi / 2.0, 1.0e-5, "wrong")},
            geometry_id="geometry",
            plic_id="plic",
        )
    condition = ContactAngleCondition(3, np.pi / 2.0, 1.0e-5, "same")
    with pytest.raises(ValueError, match="unique"):
        EmbeddedBoundaryContactAngleSet(
            (condition, condition), geometry_id="geometry", plic_id="plic"
        )


def test_degenerate_tangent_projection_fails():
    condition = ContactAngleCondition(3, np.pi / 2.0, 1.0e-5, "degenerate")
    with pytest.raises(Exception, match="degenerate projection"):
        reconstruct_wall_interface_normal((1.0, 0.0), (2.0, 0.0), condition)


def test_valid_path_is_jittable_and_differentiable():
    condition = ContactAngleCondition(3, np.pi / 3.0, 1.0e-4, "smooth")
    wall = jnp.asarray((0.4, 0.9))

    @jax.jit
    def normal_from_plic(plic):
        return reconstruct_wall_interface_normal(plic, wall, condition).normal

    plic = jnp.asarray((-0.8, 0.6))
    normal = normal_from_plic(plic)
    assert bool(jnp.all(jnp.isfinite(normal)))
    jacobian = jax.jacfwd(normal_from_plic)(plic)
    assert bool(jnp.all(jnp.isfinite(jacobian)))

    def scalar_path(plic):
        return jnp.sum(normal_from_plic(plic) * jnp.asarray((0.7, -0.2)))

    gradient = jax.grad(scalar_path)(plic)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_rotated_wall_zero_tolerance_records_failed_evidence_eager_and_jit():
    condition = ContactAngleCondition(7, np.pi / 3.0, 0.0, "strict-rotated-wall")
    plic = jnp.asarray((-0.8, 0.2), dtype=jnp.float32)
    wall = jnp.asarray((0.3, 0.7), dtype=jnp.float32)

    def reconstruct(plic_normal, wall_normal):
        return reconstruct_wall_interface_normal(
            plic_normal,
            wall_normal,
            condition,
        )

    for result in (reconstruct(plic, wall), jax.jit(reconstruct)(plic, wall)):
        assert not bool(np.asarray(result.evidence.passed))
        assert int(np.asarray(result.evidence.status)) == int(ContactAngleStatus.FAILED)
        assert int(np.asarray(result.status)) == int(ContactAngleStatus.FAILED)
