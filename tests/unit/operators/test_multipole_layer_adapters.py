from typing import Any

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


def _prepared() -> Any:
    return phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=3,
    ).prepare()


def test_multipole_layer_adapters_scenario_1() -> None:
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


def _clustered_layer() -> Any:
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


def test_rotation_layer_payload_normals_and_qbx() -> None:
    density = jnp.stack((DENSITY + 0.3j * DENSITY, -0.2j * DENSITY), axis=-1)
    normals = SOURCES / jnp.linalg.norm(SOURCES, axis=-1, keepdims=True)
    difference = TARGETS[:, None, :] - SOURCES[None, :, :]
    radius = jnp.linalg.norm(difference, axis=-1)
    strength = density * WEIGHTS[:, None]
    expected_single = jnp.tensordot(
        1.0 / (4.0 * jnp.pi * radius), strength, axes=((1,), (0,))
    )
    expected_double = jnp.tensordot(
        jnp.sum(difference * normals[None, :, :], axis=-1) / (4.0 * jnp.pi * radius**3),
        strength,
        axes=((1,), (0,)),
    )
    for execution in ("level_octree", "plane_dual"):
        prepared = phx.operators.LaplaceMultipolePlan3D(
            SOURCES,
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            reference_targets=TARGETS,
            depth=3,
            expansion_order=4,
            translation_route="rotation",
            execution=execution,
            source_leaf_occupancy=1,
            target_leaf_occupancy=1,
            plane_coarsening_factor=2,
            plane_target_top_nodes=1,
        ).prepare()
        single = phx.operators.evaluate_laplace_layer_multipole_3d(
            prepared, SOURCES, density, WEIGHTS, TARGETS
        )
        double = phx.operators.evaluate_laplace_layer_multipole_3d(
            prepared, SOURCES, density, WEIGHTS, TARGETS, source_normals=normals
        )
        assert bool(single.successful)
        assert bool(double.successful)
        np.testing.assert_allclose(single.values, expected_single, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(double.values, expected_double, rtol=2e-2, atol=2e-4)
        reversed_normals = phx.operators.evaluate_laplace_layer_multipole_3d(
            prepared, SOURCES, density, WEIGHTS, TARGETS, source_normals=-normals
        )
        np.testing.assert_allclose(
            reversed_normals.values, -double.values, rtol=3e-13, atol=3e-15
        )
        far = phx.operators.prepare_laplace_qbx_far_local_3d(
            prepared, SOURCES, density, WEIGHTS, TARGETS, source_normals=normals
        )
        assert bool(far.successful)
        np.testing.assert_allclose(
            jax.vmap(prepared.l2p)(far.coefficients, TARGETS, TARGETS),
            double.far_values,
            rtol=3e-11,
            atol=3e-13,
        )
        direction = 0.1 * normals
        _, tangent = jax.jvp(
            lambda values: (
                phx.operators.evaluate_laplace_layer_multipole_3d(
                    prepared, SOURCES, density, WEIGHTS, TARGETS, source_normals=values
                ).values
            ),
            (normals,),
            (direction,),
        )
        np.testing.assert_allclose(tangent, 0.1 * double.values, rtol=3e-12, atol=3e-14)
