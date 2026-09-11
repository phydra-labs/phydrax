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


def test_qbx_adapter_translates_far_locals_to_requested_centers():
    prepared = _prepared()
    raw = prepared.far_local(SOURCES, DENSITY * WEIGHTS, TARGETS)
    shifted = phx.operators.prepare_laplace_qbx_far_local_3d(
        prepared,
        SOURCES,
        DENSITY,
        WEIGHTS,
        TARGETS,
    )
    hierarchy = prepared.topology.hierarchy
    leaves = hierarchy.logical_point_leaf_slots[
        prepared.plan.source_capacity : prepared.plan.source_capacity
        + prepared.plan.target_capacity
    ]
    leaf_centers = hierarchy.node_centers[jnp.maximum(leaves, 0)]
    raw_values = jax.vmap(prepared.l2p)(raw.coefficients, leaf_centers, TARGETS)
    shifted_values = jax.vmap(prepared.l2p)(shifted.coefficients, TARGETS, TARGETS)
    np.testing.assert_allclose(shifted_values, raw_values, rtol=3e-11, atol=3e-12)
    assert int(shifted.l2l_count) == int(raw.l2l_count) + TARGETS.shape[0]
