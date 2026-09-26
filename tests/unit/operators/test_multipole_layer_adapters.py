import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


SOURCES = jnp.asarray(
    [[-0.7, -0.6, -0.5], [-0.5, 0.6, 0.4], [0.55, -0.45, 0.5], [0.7, 0.65, -0.4]]
)
TARGETS = jnp.asarray([[0.05, 0.1, 0.2], [0.2, -0.1, -0.15], [-0.1, 0.2, -0.2]])
DENSITY = jnp.asarray([0.8, -0.4, 0.3, 1.1])
WEIGHTS = jnp.asarray([0.7, 1.2, 0.8, 0.9])


def _prepared():
    return phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=3,
    ).prepare()


def test_weighted_single_and_double_layer_adapters_match_direct_kernels():
    prepared = _prepared()
    strengths = DENSITY * WEIGHTS
    single = phx.operators.evaluate_laplace_layer_multipole_3d(
        prepared, SOURCES, DENSITY, WEIGHTS, TARGETS
    )
    difference = TARGETS[:, None, :] - SOURCES[None, :, :]
    radius = jnp.linalg.norm(difference, axis=-1)
    expected_single = jnp.sum(strengths[None, :] / (4.0 * jnp.pi * radius), axis=1)
    np.testing.assert_allclose(single.values, expected_single, rtol=4e-3, atol=4e-4)

    normals = SOURCES / jnp.linalg.norm(SOURCES, axis=-1, keepdims=True)
    double = phx.operators.evaluate_laplace_layer_multipole_3d(
        prepared,
        SOURCES,
        DENSITY,
        WEIGHTS,
        TARGETS,
        source_normals=normals,
    )
    expected_double = jnp.sum(
        strengths[None, :]
        * jnp.sum(difference * normals[None, :, :], axis=-1)
        / (4.0 * jnp.pi * radius**3),
        axis=1,
    )
    np.testing.assert_allclose(double.values, expected_double, rtol=3.5e-2, atol=1e-3)


def _clustered_layer():
    rng = np.random.default_rng(29)
    sources = np.concatenate(
        (
            [-0.55, -0.5, -0.45] + 0.04 * rng.standard_normal((14, 3)),
            [0.5, 0.45, 0.5] + 0.08 * rng.standard_normal((8, 3)),
            rng.uniform(-0.9, 0.9, (6, 3)),
        )
    )
    targets = np.concatenate(
        (
            [-0.45, -0.4, -0.35] + 0.05 * rng.standard_normal((4, 3)),
            rng.uniform(-0.9, 0.9, (5, 3)),
        )
    )
    density = rng.standard_normal(sources.shape[0])
    weights = rng.uniform(0.5, 1.5, sources.shape[0])
    return (
        jnp.asarray(sources),
        jnp.asarray(targets),
        jnp.asarray(density),
        jnp.asarray(weights),
    )


def test_qbx_adapter_far_locals_reproduce_far_field_at_centers():
    sources, targets, density, weights = _clustered_layer()
    prepared = phx.operators.LaplaceMultipolePlan3D(
        sources,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=targets,
        depth=4,
        expansion_order=3,
        source_leaf_occupancy=2,
    ).prepare()
    far = phx.operators.prepare_laplace_qbx_far_local_3d(
        prepared, sources, density, weights, targets
    )
    evaluation = phx.operators.evaluate_laplace_layer_multipole_3d(
        prepared, sources, density, weights, targets
    )
    local_values = jax.vmap(prepared.l2p)(far.coefficients, targets, targets)

    assert bool(far.successful)
    assert int(evaluation.m2p_count) > 0
    assert float(jnp.max(jnp.abs(evaluation.far_values))) > 1e-3
    np.testing.assert_allclose(local_values, evaluation.far_values, rtol=1e-9, atol=1e-11)


def test_layer_and_qbx_adapters_accept_plane_far_execution():
    prepared = phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=3,
        execution="plane_dual",
        source_leaf_occupancy=1,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
    ).prepare()
    strengths = DENSITY * WEIGHTS
    radius = jnp.linalg.norm(TARGETS[:, None, :] - SOURCES[None, :, :], axis=-1)
    single = phx.operators.evaluate_laplace_layer_multipole_3d(
        prepared,
        SOURCES,
        DENSITY,
        WEIGHTS,
        TARGETS,
    )
    expected = jnp.sum(strengths[None, :] / (4.0 * jnp.pi * radius), axis=1)
    np.testing.assert_allclose(single.values, expected, rtol=4e-3, atol=4e-4)

    normals = SOURCES / jnp.linalg.norm(SOURCES, axis=-1, keepdims=True)
    double = phx.operators.evaluate_laplace_layer_multipole_3d(
        prepared,
        SOURCES,
        DENSITY,
        WEIGHTS,
        TARGETS,
        source_normals=normals,
    )
    difference = TARGETS[:, None, :] - SOURCES[None, :, :]
    double_expected = jnp.sum(
        strengths[None, :]
        * jnp.sum(difference * normals[None, :, :], axis=-1)
        / (4.0 * jnp.pi * radius**3),
        axis=1,
    )
    np.testing.assert_allclose(
        double.values,
        double_expected,
        rtol=3.5e-2,
        atol=1e-3,
    )

    far = phx.operators.prepare_laplace_qbx_far_local_3d(
        prepared,
        SOURCES,
        DENSITY,
        WEIGHTS,
        TARGETS,
    )
    local_values = jax.vmap(prepared.l2p)(far.coefficients, TARGETS, TARGETS)
    np.testing.assert_allclose(local_values, single.far_values, rtol=3e-11, atol=3e-12)
