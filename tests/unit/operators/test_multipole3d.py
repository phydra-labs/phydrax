import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


SOURCES = jnp.asarray(
    [[-0.7, -0.6, -0.5], [-0.5, 0.6, 0.4], [0.55, -0.45, 0.5], [0.7, 0.65, -0.4]]
)
TARGETS = jnp.asarray([[0.05, 0.1, 0.2], [0.2, -0.1, -0.15], [-0.1, 0.2, -0.2]])
STRENGTHS = jnp.asarray([0.8, -0.4, 0.3, 1.1])


def _direct(kernel, parameter=0.0):
    radii = jnp.linalg.norm(TARGETS[:, None, :] - SOURCES[None, :, :], axis=-1)
    if kernel == "laplace":
        numerator = jnp.ones_like(radii)
    elif kernel == "helmholtz":
        numerator = jnp.exp(1j * parameter * radii)
    else:
        numerator = jnp.exp(-parameter * radii)
    return jnp.sum(numerator * STRENGTHS[None, :] / (4.0 * jnp.pi * radii), axis=1)


@pytest.mark.parametrize(
    ("plan_type", "keyword", "kernel"),
    [
        (phx.operators.LaplaceMultipolePlan3D, {}, "laplace"),
        (phx.operators.HelmholtzMultipolePlan3D, {"wavenumber": 0.7}, "helmholtz"),
        (
            phx.operators.ModifiedHelmholtzMultipolePlan3D,
            {"decay": 0.7},
            "modified-helmholtz",
        ),
    ],
)
def test_complete_multipole_pipelines_match_direct(plan_type, keyword, kernel):
    prepared = plan_type(
        SOURCES,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=3,
        **keyword,
    ).prepare()
    result = prepared.evaluate(SOURCES, STRENGTHS, TARGETS)
    expected = _direct(kernel, 0.7)
    np.testing.assert_allclose(result.values, expected, rtol=4e-3, atol=4e-4)
    assert bool(result.successful)
    assert int(result.p2m_count) == SOURCES.shape[0]
    assert int(result.m2m_count) > 0
    assert int(result.m2l_count) > 0
    assert int(result.l2l_count) > 0
    assert int(result.l2p_count) == TARGETS.shape[0]
    assert int(result.p2p_count) > 0
    assert bool(result.capacity.successful)
    assert bool(result.truncation.well_separated)


def test_laplace_elementary_translations_compose():
    prepared = phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        [-2.0, -2.0, -2.0],
        [2.0, 2.0, 2.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=4,
    ).prepare()
    child = jnp.asarray([-0.2, 0.1, 0.0])
    parent = jnp.asarray([0.0, 0.0, 0.0])
    multipole = prepared.p2m(SOURCES, STRENGTHS, child)
    translated = prepared.m2m(multipole, child, parent)
    exterior = jnp.asarray([[2.5, -0.2, 0.4], [2.2, 1.4, -0.8]])
    direct = jnp.sum(
        STRENGTHS[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(exterior[:, None, :] - SOURCES[None, :, :], axis=-1)
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        prepared.multipole_to_point(translated, parent, exterior),
        direct,
        rtol=6e-3,
        atol=2e-5,
    )

    local_center = jnp.asarray([2.4, 0.1, -0.1])
    local = prepared.m2l(translated, parent, local_center)
    nearby = local_center + jnp.asarray([[0.02, 0.01, -0.01], [-0.03, 0.02, 0.01]])
    direct_nearby = jnp.sum(
        STRENGTHS[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(nearby[:, None, :] - SOURCES[None, :, :], axis=-1)
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        prepared.l2p(local, local_center, nearby),
        direct_nearby,
        rtol=4e-3,
        atol=4e-5,
    )


def test_plane_dual_laplace_matches_level_octree_and_direct_completion():
    source = jnp.asarray(
        [
            [-0.80, -0.75, -0.70],
            [-0.74, -0.70, -0.69],
            [-0.15, 0.25, 0.30],
            [-0.08, 0.29, 0.27],
            [0.68, 0.72, 0.70],
            [0.75, 0.68, 0.74],
        ]
    )
    target = jnp.asarray(
        [
            [-0.70, -0.68, -0.66],
            [-0.10, 0.22, 0.26],
            [0.70, 0.72, 0.73],
            [0.25, -0.35, 0.10],
        ]
    )
    strengths = jnp.asarray([0.8, -0.3, 0.4, 1.1, -0.7, 0.2])
    keywords = {
        "reference_targets": target,
        "depth": 3,
        "expansion_order": 4,
        "maximum_reference_displacement": 0.08,
    }
    level = phx.operators.LaplaceMultipolePlan3D(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        **keywords,
    ).prepare()
    plane = phx.operators.LaplaceMultipolePlan3D(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        execution="plane_dual",
        source_leaf_occupancy=2,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
        **keywords,
    ).prepare()
    level_result = level.evaluate(source, strengths, target)
    plane_result = plane.evaluate(source, strengths, target)
    direct = jnp.sum(
        strengths[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
        ),
        axis=1,
    )
    np.testing.assert_allclose(plane_result.values, direct, rtol=8e-3, atol=8e-5)
    np.testing.assert_allclose(
        plane_result.values,
        level_result.values,
        rtol=8e-3,
        atol=8e-5,
    )
    assert bool(plane_result.successful)
    assert int(plane_result.m2l_count) > 0
    assert int(plane_result.p2p_count) > 0
    assert bool(plane_result.capacity.successful)


def test_plane_laplace_accepts_fixed_envelope_motion_and_jits():
    reference_sources = jnp.asarray(
        [[-0.70, -0.70, -0.70], [-0.65, -0.68, -0.66], [0.70, 0.70, 0.70]]
    )
    reference_targets = jnp.asarray([[0.10, 0.10, 0.10], [0.60, 0.62, 0.64]])
    prepared = phx.operators.LaplaceMultipolePlan3D(
        reference_sources,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=reference_targets,
        depth=3,
        expansion_order=3,
        maximum_reference_displacement=0.08,
        execution="plane_dual",
        source_leaf_occupancy=1,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
    ).prepare()
    sources = reference_sources + jnp.asarray(
        [[0.04, -0.03, 0.02], [-0.03, 0.02, -0.01], [0.01, -0.02, 0.03]]
    )
    targets = reference_targets + jnp.asarray([[-0.02, 0.03, -0.01], [0.03, -0.01, 0.02]])
    strengths = jnp.asarray([1.0, -0.5, 0.25])
    compiled = eqx.filter_jit(
        lambda source, strength, target: prepared.evaluate(source, strength, target)
    )
    result = compiled(sources, strengths, targets)
    direct = jnp.sum(
        strengths[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
        ),
        axis=1,
    )
    assert bool(result.successful)
    assert not bool(result.stale_topology)
    np.testing.assert_allclose(result.values, direct, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize(
    ("plan_type", "keyword", "direct_factor"),
    [
        (
            phx.operators.ModifiedHelmholtzMultipolePlan3D,
            {"decay": 0.7},
            lambda radius: jnp.exp(-0.7 * radius),
        ),
        (
            phx.operators.HelmholtzMultipolePlan3D,
            {"wavenumber": 0.7},
            lambda radius: jnp.exp(1j * 0.7 * radius),
        ),
    ],
)
def test_radial_plane_execution_is_wave_resolved_and_differentiable(
    plan_type,
    keyword,
    direct_factor,
):
    source = jnp.asarray(
        [
            [-0.75, -0.70, -0.68],
            [-0.68, -0.66, -0.65],
            [-0.10, 0.25, 0.30],
            [-0.04, 0.28, 0.26],
            [0.68, 0.72, 0.70],
            [0.74, 0.68, 0.73],
        ]
    )
    target = jnp.asarray([[-0.68, -0.65, -0.64], [-0.08, 0.22, 0.28], [0.70, 0.70, 0.72]])
    strengths = jnp.asarray([0.8, -0.3, 0.4, 1.1, -0.7, 0.2])
    keywords = {
        "reference_targets": target,
        "depth": 3,
        "expansion_order": 3,
    }
    level = plan_type(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        **keywords,
        **keyword,
    ).prepare()
    plane = plan_type(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        execution="plane_dual",
        source_leaf_occupancy=2,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
        maximum_plane_node_argument=0.5,
        **keywords,
        **keyword,
    ).prepare()
    plane_result = plane.evaluate(source, strengths, target)
    level_result = level.evaluate(source, strengths, target)
    radius = jnp.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    direct = jnp.sum(
        direct_factor(radius) * strengths[None, :] / (4.0 * jnp.pi * radius),
        axis=1,
    )
    np.testing.assert_allclose(plane_result.values, direct, rtol=1.2e-2, atol=1e-4)
    np.testing.assert_allclose(
        plane_result.values,
        level_result.values,
        rtol=1.2e-2,
        atol=1e-4,
    )
    assert bool(plane_result.successful)
    assert int(plane_result.p2p_count) > 0

    weights = jnp.asarray([0.4, -0.2, 0.7])

    def loss(position):
        value = plane.evaluate(position, strengths, target).values
        return jnp.real(jnp.vdot(weights, value))

    gradient = jax.grad(loss)(source)

    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_plane_laplace_position_and_strength_gradients_match_level_route():
    source = jnp.asarray(
        [[-0.72, -0.68, -0.66], [-0.64, -0.70, -0.65], [0.70, 0.68, 0.72]]
    )
    target = jnp.asarray([[0.10, 0.10, 0.10], [0.55, 0.50, 0.52]])
    strengths = jnp.asarray([0.8, -0.4, 1.1])
    weights = jnp.asarray([0.3, -0.7])
    common = {
        "reference_targets": target,
        "depth": 3,
        "expansion_order": 3,
    }
    level = phx.operators.LaplaceMultipolePlan3D(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        **common,
    ).prepare()
    plane = phx.operators.LaplaceMultipolePlan3D(
        source,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        execution="plane_dual",
        source_leaf_occupancy=1,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
        **common,
    ).prepare()

    def level_loss(position, weight):
        return jnp.real(
            jnp.sum(weights * level.evaluate(position, weight, target).values)
        )

    def plane_loss(position, weight):
        return jnp.real(
            jnp.sum(weights * plane.evaluate(position, weight, target).values)
        )

    level_gradient = jax.grad(level_loss, argnums=(0, 1))(source, strengths)
    plane_gradient = jax.grad(plane_loss, argnums=(0, 1))(source, strengths)
    np.testing.assert_allclose(plane_gradient[0], level_gradient[0], rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(plane_gradient[1], level_gradient[1], rtol=1e-2, atol=1e-4)
