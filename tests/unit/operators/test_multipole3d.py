from typing import Any, NotRequired, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


class _PlanKeywords(TypedDict):
    reference_targets: jax.Array
    depth: int
    expansion_order: int
    maximum_reference_displacement: NotRequired[float]


SOURCES = jnp.asarray(
    [[-0.7, -0.6, -0.5], [-0.5, 0.6, 0.4], [0.55, -0.45, 0.5], [0.7, 0.65, -0.4]],
    dtype=jnp.float64,
)
TARGETS = jnp.asarray(
    [[0.05, 0.1, 0.2], [0.2, -0.1, -0.15], [-0.1, 0.2, -0.2]], dtype=jnp.float64
)
STRENGTHS = jnp.asarray([0.8, -0.4, 0.3, 1.1], dtype=jnp.float64)


def _clustered_cloud() -> Any:
    rng = np.random.default_rng(17)
    sources = np.concatenate(
        (
            [-0.55, -0.5, -0.45] + 0.04 * rng.standard_normal((16, 3)),
            [0.5, 0.45, 0.5] + 0.08 * rng.standard_normal((10, 3)),
            rng.uniform(-0.9, 0.9, (6, 3)),
        )
    )
    targets = np.concatenate(
        (
            [-0.45, -0.4, -0.35] + 0.05 * rng.standard_normal((4, 3)),
            rng.uniform(-0.9, 0.9, (6, 3)),
        )
    )
    strengths = rng.standard_normal(sources.shape[0])
    return (
        jnp.asarray(sources, dtype=jnp.float64),
        jnp.asarray(targets, dtype=jnp.float64),
        jnp.asarray(strengths, dtype=jnp.float64),
    )


def _direct(
    kernel: Any, sources: Any, strengths: Any, targets: Any, parameter: Any = 0.0
) -> Any:
    radii = jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
    if kernel == "laplace":
        numerator = jnp.ones_like(radii)
    elif kernel == "helmholtz":
        numerator = jnp.exp(1j * parameter * radii)
    else:
        numerator = jnp.exp(-parameter * radii)
    return jnp.sum(numerator * strengths[None, :] / (4.0 * jnp.pi * radii), axis=1)


def test_multipole3d_scenario_1() -> None:
    for kernel in ("laplace", "helmholtz", "modified-helmholtz"):
        sources, targets, strengths = _clustered_cloud()
        keywords: _PlanKeywords = {
            "reference_targets": targets,
            "depth": 4,
            "expansion_order": 4,
        }
        lower = jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64)
        upper = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)
        if kernel == "laplace":
            plan = phx.operators.LaplaceMultipolePlan3D(
                sources, lower, upper, source_leaf_occupancy=2, **keywords
            )
        elif kernel == "helmholtz":
            plan = phx.operators.HelmholtzMultipolePlan3D(
                sources, lower, upper, wavenumber=0.7, source_leaf_occupancy=2, **keywords
            )
        else:
            plan = phx.operators.ModifiedHelmholtzMultipolePlan3D(
                sources, lower, upper, decay=0.7, source_leaf_occupancy=2, **keywords
            )
        prepared = plan.prepare()
        result = prepared.evaluate(sources, strengths, targets)
        expected = _direct(kernel, sources, strengths, targets, 0.7)
        # Expansion truncation scales with the absolute potential sum(|q| / (4 pi r)),
        # not with the signed sum, which cancels about 450-fold at one screened target.
        # At order 4 the Laplace tail bound sum(|q| rho**5 / (4 pi R (1 - rho))) of the
        # realized V, W, and X routes stays below 7.5e-3 of that scale at every target;
        # screened and low-frequency Helmholtz expansions share this geometric tail.
        scale = _direct("laplace", sources, jnp.abs(strengths), targets)
        np.testing.assert_array_less(jnp.abs(result.values - expected), 1e-2 * scale)
        assert bool(result.successful)
        assert int(result.p2m_count) == sources.shape[0]
        assert int(result.m2m_count) > 0
        assert int(result.m2l_count) > 0
        assert int(result.p2l_count) > 0
        assert int(result.l2l_count) > 0
        assert int(result.l2p_count) == targets.shape[0]
        assert int(result.m2p_count) > 0
        assert int(result.p2p_count) > 0
        assert bool(result.capacity.successful)
        assert bool(result.truncation.well_separated)
    sources, _, strengths = _clustered_cloud()
    prepared = phx.operators.LaplaceMultipolePlan3D(
        sources,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
        depth=4,
        expansion_order=3,
        source_leaf_occupancy=2,
        target_leaf_occupancy=2,
        maximum_reference_displacement=0.02,
    ).prepare()
    moved = sources + 0.019 * jnp.asarray(
        np.random.default_rng(3).uniform(-1.0, 1.0, sources.shape), dtype=jnp.float64
    ) / jnp.sqrt(3.0)
    result = prepared.evaluate(moved, strengths)
    radii = jnp.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=-1)
    radii = jnp.where(jnp.eye(sources.shape[0], dtype=jnp.bool_), jnp.inf, radii)
    expected = jnp.sum(strengths[None, :] / (4.0 * jnp.pi * radii), axis=1)
    assert bool(result.successful)
    assert not bool(result.stale_topology)
    np.testing.assert_allclose(result.values, expected, rtol=4e-3, atol=4e-4)

    far = sources + jnp.asarray([0.03, 0.0, 0.0], dtype=jnp.float64)
    with pytest.raises(eqx.EquinoxRuntimeError, match="frozen-topology envelope"):
        prepared.evaluate(far, strengths)
    prepared = phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        jnp.asarray([-2.0, -2.0, -2.0], dtype=jnp.float64),
        jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float64),
        reference_targets=TARGETS,
        depth=2,
        expansion_order=4,
    ).prepare()
    child = jnp.asarray([-0.2, 0.1, 0.0], dtype=jnp.float64)
    parent = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
    multipole = prepared.p2m(SOURCES, STRENGTHS, child)
    translated = prepared.m2m(multipole, child, parent)
    exterior = jnp.asarray([[2.5, -0.2, 0.4], [2.2, 1.4, -0.8]], dtype=jnp.float64)
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

    local_center = jnp.asarray([2.4, 0.1, -0.1], dtype=jnp.float64)
    local = prepared.m2l(translated, parent, local_center)
    nearby = local_center + jnp.asarray(
        [[0.02, 0.01, -0.01], [-0.03, 0.02, 0.01]], dtype=jnp.float64
    )
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


def test_plane_contracts() -> None:
    source = jnp.asarray(
        [
            [-0.80, -0.75, -0.70],
            [-0.74, -0.70, -0.69],
            [-0.15, 0.25, 0.30],
            [-0.08, 0.29, 0.27],
            [0.68, 0.72, 0.70],
            [0.75, 0.68, 0.74],
        ],
        dtype=jnp.float64,
    )
    target = jnp.asarray(
        [
            [-0.70, -0.68, -0.66],
            [-0.10, 0.22, 0.26],
            [0.70, 0.72, 0.73],
            [0.25, -0.35, 0.10],
        ],
        dtype=jnp.float64,
    )
    strengths = jnp.asarray([0.8, -0.3, 0.4, 1.1, -0.7, 0.2], dtype=jnp.float64)
    keywords: _PlanKeywords = {
        "reference_targets": target,
        "depth": 3,
        "expansion_order": 4,
        "maximum_reference_displacement": 0.08,
    }
    level = phx.operators.LaplaceMultipolePlan3D(
        source,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
        **keywords,
    ).prepare()
    plane = phx.operators.LaplaceMultipolePlan3D(
        source,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
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
    reference_sources = jnp.asarray(
        [[-0.70, -0.70, -0.70], [-0.65, -0.68, -0.66], [0.70, 0.70, 0.70]],
        dtype=jnp.float64,
    )
    reference_targets = jnp.asarray(
        [[0.10, 0.10, 0.10], [0.60, 0.62, 0.64]], dtype=jnp.float64
    )
    prepared = phx.operators.LaplaceMultipolePlan3D(
        reference_sources,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
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
        [[0.04, -0.03, 0.02], [-0.03, 0.02, -0.01], [0.01, -0.02, 0.03]],
        dtype=jnp.float64,
    )
    targets = reference_targets + jnp.asarray(
        [[-0.02, 0.03, -0.01], [0.03, -0.01, 0.02]], dtype=jnp.float64
    )
    strengths = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)
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


def test_radial_plane_execution_is_wave_resolved_and_differentiable() -> None:
    for kernel in ("modified-helmholtz", "helmholtz"):
        source = jnp.asarray(
            [
                [-0.75, -0.70, -0.68],
                [-0.68, -0.66, -0.65],
                [-0.10, 0.25, 0.30],
                [-0.04, 0.28, 0.26],
                [0.68, 0.72, 0.70],
                [0.74, 0.68, 0.73],
            ],
            dtype=jnp.float64,
        )
        target = jnp.asarray(
            [[-0.68, -0.65, -0.64], [-0.08, 0.22, 0.28], [0.70, 0.70, 0.72]],
            dtype=jnp.float64,
        )
        strengths = jnp.asarray([0.8, -0.3, 0.4, 1.1, -0.7, 0.2], dtype=jnp.float64)
        keywords: _PlanKeywords = {
            "reference_targets": target,
            "depth": 3,
            "expansion_order": 3,
        }
        lower = jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64)
        upper = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)
        if kernel == "modified-helmholtz":
            level_plan = phx.operators.ModifiedHelmholtzMultipolePlan3D(
                source, lower, upper, decay=0.7, **keywords
            )
            plane_plan = phx.operators.ModifiedHelmholtzMultipolePlan3D(
                source,
                lower,
                upper,
                decay=0.7,
                execution="plane_dual",
                source_leaf_occupancy=2,
                target_leaf_occupancy=1,
                plane_coarsening_factor=2,
                plane_target_top_nodes=1,
                maximum_plane_node_argument=0.5,
                **keywords,
            )
        else:
            level_plan = phx.operators.HelmholtzMultipolePlan3D(
                source, lower, upper, wavenumber=0.7, **keywords
            )
            plane_plan = phx.operators.HelmholtzMultipolePlan3D(
                source,
                lower,
                upper,
                wavenumber=0.7,
                execution="plane_dual",
                source_leaf_occupancy=2,
                target_leaf_occupancy=1,
                plane_coarsening_factor=2,
                plane_target_top_nodes=1,
                maximum_plane_node_argument=0.5,
                **keywords,
            )
        level = level_plan.prepare()
        plane = plane_plan.prepare()
        plane_result = plane.evaluate(source, strengths, target)
        level_result = level.evaluate(source, strengths, target)
        radius = jnp.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
        direct_factor = (
            jnp.exp(-0.7 * radius)
            if kernel == "modified-helmholtz"
            else jnp.exp(1j * 0.7 * radius)
        )
        direct = jnp.sum(
            direct_factor * strengths[None, :] / (4.0 * jnp.pi * radius),
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

        weights = jnp.asarray([0.4, -0.2, 0.7], dtype=jnp.float64)

        def loss(position: Any) -> Any:
            value = plane.evaluate(position, strengths, target).values
            return jnp.real(jnp.vdot(weights, value))

        gradient = jax.grad(loss)(source)

        assert bool(jnp.all(jnp.isfinite(gradient)))


def test_plane_laplace_position_and_strength_gradients_match_level_route() -> None:
    source = jnp.asarray(
        [[-0.72, -0.68, -0.66], [-0.64, -0.70, -0.65], [0.70, 0.68, 0.72]],
        dtype=jnp.float64,
    )
    target = jnp.asarray([[0.10, 0.10, 0.10], [0.55, 0.50, 0.52]], dtype=jnp.float64)
    strengths = jnp.asarray([0.8, -0.4, 1.1], dtype=jnp.float64)
    weights = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    common: _PlanKeywords = {
        "reference_targets": target,
        "depth": 3,
        "expansion_order": 3,
    }
    level = phx.operators.LaplaceMultipolePlan3D(
        source,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
        **common,
    ).prepare()
    plane = phx.operators.LaplaceMultipolePlan3D(
        source,
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64),
        execution="plane_dual",
        source_leaf_occupancy=1,
        target_leaf_occupancy=1,
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
        **common,
    ).prepare()

    def level_loss(position: Any, weight: Any) -> Any:
        return jnp.real(
            jnp.sum(weights * level.evaluate(position, weight, target).values)
        )

    def plane_loss(position: Any, weight: Any) -> Any:
        return jnp.real(
            jnp.sum(weights * plane.evaluate(position, weight, target).values)
        )

    level_gradient = jax.grad(level_loss, argnums=(0, 1))(source, strengths)
    plane_gradient = jax.grad(plane_loss, argnums=(0, 1))(source, strengths)
    np.testing.assert_allclose(plane_gradient[0], level_gradient[0], rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(plane_gradient[1], level_gradient[1], rtol=1e-2, atol=1e-4)
