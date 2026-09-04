#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.linalg import ArraySpace
from phydrax.metrix._lie_group import (
    LieGroupStateGeometry,
    RightLieGroupStateGeometry,
    SpecialEuclideanGroup,
    SpecialOrthogonalGroup,
)
from phydrax.metrix._product_state_geometry import (
    ProductStateGeometry,
    ProductStateGeometryBlock,
)
from phydrax.metrix._quaternion_state_geometry import (
    QuaternionPoseStateGeometry,
    ScalarFirstQuaternionStateGeometry,
)
from phydrax.metrix._state_geometry import EuclideanStateGeometry


def test_scalar_first_quaternion_is_antipodal_invariant_inside_one_chart():
    geometry = ScalarFirstQuaternionStateGeometry(convention="body")
    anchor = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    local = jnp.asarray([0.3, -0.2, 0.1])
    target = geometry.retract(anchor, local)

    assert target.shape == (4,)
    assert geometry.inverse_retract(anchor, target) == pytest.approx(local)
    assert geometry.inverse_retract(anchor, -target) == pytest.approx(local)
    assert geometry.cut_locus_margin(anchor, target) > 0.0

    pi_target = jnp.asarray([0.0, 1.0, 0.0, 0.0])
    assert geometry.cut_locus_margin(anchor, pi_target) == pytest.approx(0.0)
    with pytest.raises(Exception, match="pi cut locus"):
        geometry.inverse_retract(anchor, pi_target)


def test_quaternion_four_space_maps_are_finite_and_exact_at_chart_origin():
    quaternion = ScalarFirstQuaternionStateGeometry(convention="body")
    quaternion_source = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    quaternion_direction = jnp.asarray([0.2, -0.1, 0.3])
    quaternion_zero = jnp.zeros((3,))
    quaternion_point = quaternion.retract(quaternion_source, quaternion_zero)
    quaternion_pushed = quaternion.retraction_jvp(
        quaternion_source, quaternion_zero, quaternion_direction
    )

    assert jnp.array_equal(quaternion_point, quaternion_source)
    assert jnp.all(jnp.isfinite(quaternion_pushed))
    assert quaternion_pushed == pytest.approx(quaternion_direction)
    assert quaternion.retraction_inverse_jvp(
        quaternion_source, quaternion_point, quaternion_pushed
    ) == pytest.approx(quaternion_direction)
    assert quaternion.retraction_inverse_jvp(
        quaternion_source, -quaternion_point, quaternion_pushed
    ) == pytest.approx(quaternion_direction)
    assert quaternion.retraction_vjp(
        quaternion_source, quaternion_zero, quaternion_direction
    ) == pytest.approx(quaternion_direction)

    pose = QuaternionPoseStateGeometry(convention="body")
    pose_source = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.7])
    pose_direction = jnp.asarray([0.2, -0.1, 0.3, -0.4, 0.25, 0.15])
    pose_zero = jnp.zeros((6,))
    pose_point = pose.retract(pose_source, pose_zero)
    pose_pushed = pose.retraction_jvp(pose_source, pose_zero, pose_direction)
    equivalent_pose = pose_point.at[:4].multiply(-1.0)

    assert jnp.array_equal(pose_point, pose_source)
    assert jnp.all(jnp.isfinite(pose_pushed))
    assert pose_pushed == pytest.approx(pose_direction)
    assert pose.retraction_inverse_jvp(
        pose_source, pose_point, pose_pushed
    ) == pytest.approx(pose_direction)
    assert pose.retraction_inverse_jvp(
        pose_source, equivalent_pose, pose_pushed
    ) == pytest.approx(pose_direction)
    assert pose.retraction_vjp(pose_source, pose_zero, pose_direction) == pytest.approx(
        pose_direction
    )


def test_quaternion_pose_body_and_spatial_conventions_are_distinct_and_exact():
    body = QuaternionPoseStateGeometry(convention="body")
    spatial = QuaternionPoseStateGeometry(convention="spatial")
    anchor = jnp.asarray([jnp.sqrt(0.5), jnp.sqrt(0.5), 0.0, 0.0, 1.0, -0.5, 0.25])
    local = jnp.asarray([0.2, -0.1, 0.3, 0.1, 0.25, -0.15])
    direction = jnp.asarray([-0.3, 0.2, 0.1, 0.15, -0.05, 0.25])
    cotangent = jnp.asarray([0.4, -0.2, 0.3, -0.1, 0.5, 0.2])

    body_target = body.retract(anchor, local)
    spatial_target = spatial.retract(anchor, local)
    assert not jnp.allclose(body_target, spatial_target)
    assert body.inverse_retract(anchor, body_target) == pytest.approx(local)
    assert spatial.inverse_retract(anchor, spatial_target) == pytest.approx(local)

    pushed = body.retraction_jvp(anchor, local, direction)
    recovered = body.retraction_inverse_jvp(anchor, body_target, pushed)
    pulled = body.retraction_vjp(anchor, local, cotangent)
    assert recovered == pytest.approx(direction)
    assert jnp.vdot(cotangent, pushed) == pytest.approx(jnp.vdot(pulled, direction))


def test_matrix_so3_and_se3_lie_geometries_use_hat_vee_coordinates():
    rotation_group = SpecialOrthogonalGroup(3)
    rotation_geometry = LieGroupStateGeometry(rotation_group)
    rotation_state = rotation_group.identity()
    rotation_local = jnp.asarray([0.2, -0.1, 0.3])
    rotation_target = rotation_geometry.retract(rotation_state, rotation_local)
    assert rotation_target.shape == (3, 3)
    assert rotation_geometry.inverse_retract(rotation_state, rotation_target).shape == (
        3,
    )
    assert rotation_geometry.retraction_jvp(
        rotation_state, rotation_local, rotation_local
    ).shape == (3,)

    rigid_group = SpecialEuclideanGroup(3)
    body = LieGroupStateGeometry(rigid_group)
    spatial = RightLieGroupStateGeometry(rigid_group)
    state_coordinates = jnp.asarray([0.4, -0.1, 0.2, 0.3, 0.2, -0.15])
    state = rigid_group.exp(rigid_group.hat(state_coordinates))
    local = jnp.asarray([0.2, 0.1, -0.3, -0.1, 0.25, 0.2])
    assert body.retract(state, local).shape == (4, 4)
    assert body.inverse_retract(state, body.retract(state, local)).shape == (6,)
    assert not jnp.allclose(body.retract(state, local), spatial.retract(state, local))


def test_product_keeps_role_offsets_and_certifies_vjp_and_transport_duality():
    pose_geometry = QuaternionPoseStateGeometry(convention="body")
    product = ProductStateGeometry(
        (
            ProductStateGeometryBlock(
                pose_geometry,
                (7,),
                block_id="pose",
                local_space=ArraySpace((6,)),
                tangent_space=ArraySpace((6,)),
            ),
            ProductStateGeometryBlock(
                EuclideanStateGeometry(),
                (6,),
                block_id="velocity",
            ),
        )
    )
    state = product.combine_point(
        (
            jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.3]),
            jnp.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
        )
    )
    local = jnp.linspace(-0.15, 0.15, 12)
    direction = jnp.linspace(0.2, -0.1, 12)
    cotangent = jnp.linspace(-0.3, 0.25, 12)

    assert product.point_size == 13
    assert product.local_size == 12
    assert product.tangent_size == 12
    assert product.cotangent_size == 12
    assert product.point_offsets == (0, 7, 13)
    assert product.local_offsets == (0, 6, 12)
    assert product.tangent_offsets == (0, 6, 12)
    assert product.cotangent_offsets == (0, 6, 12)

    target = product.retract(state, local)
    pushed = product.retraction_jvp(state, local, direction)
    pulled = product.retraction_vjp(state, local, cotangent)
    assert jnp.vdot(cotangent, pushed) == pytest.approx(
        jnp.vdot(pulled, direction), abs=1.0e-10
    )

    transported = product.transport_tangent(state, target, pushed)
    recovered = product.transport_tangent(target, state, transported)
    pulled_transport_cotangent = product.transport_cotangent_pullback(
        state, target, cotangent
    )
    assert recovered == pytest.approx(pushed)
    assert jnp.vdot(cotangent, transported) == pytest.approx(
        jnp.vdot(pulled_transport_cotangent, pushed), abs=1.0e-10
    )
