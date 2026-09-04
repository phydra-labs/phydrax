#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.optics.geometric._interface import OpticalRayState
from phydrax.optics.geometric._nonsequential import (
    NonSequentialBranchMode,
    NonSequentialOpticsPlan,
    NonSequentialOpticsStatus,
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
    prepare_nonsequential_optics,
    trace_nonsequential_optics,
)


def _planes(z_values):
    vertices = []
    triangles = []
    for z in z_values:
        offset = len(vertices)
        vertices.extend(
            [[-10.0, -10.0, z], [10.0, -10.0, z], [10.0, 10.0, z], [-10.0, 10.0, z]]
        )
        triangles.extend(
            [[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]]
        )
    return jnp.asarray(vertices), jnp.asarray(triangles, dtype=jnp.int32)


def _single_ray():
    return OpticalRayState(
        jnp.asarray([[0.0, 0.0, -1.0]]),
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([1.0]),
    )


def _slab_surfaces(*, kinds=None, modes=None):
    vertices, triangles = _planes((0.0, 1.0))
    if kinds is None:
        kinds = jnp.full((4,), int(NonSequentialSurfaceKind.DIELECTRIC))
    if modes is None:
        modes = jnp.full((4,), int(NonSequentialBranchMode.BOTH))
    return NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0, 1, 1]),
        jnp.asarray([1, 1, 0, 0]),
        jnp.asarray([1.0, 1.5]),
        surface_ids=jnp.asarray([0, 0, 1, 1]),
        surface_kinds=kinds,
        branch_modes=modes,
    )


def test_fresnel_tree_candidates_and_complete_energy_ledger():
    prepared = prepare_nonsequential_optics(
        NonSequentialOpticsPlan(
            _slab_surfaces(),
            maximum_interactions=3,
            branch_capacity=8,
            record_history=True,
            power_tolerance=0.0,
        )
    )
    result = trace_nonsequential_optics(
        prepared, _single_ray(), jnp.asarray([1.0]), jnp.asarray([0])
    )
    first_powers = np.asarray(result.history_powers[0, 1])
    live_first = np.asarray(result.history_live[0, 1])
    np.testing.assert_allclose(first_powers[live_first], [0.04, 0.96], atol=2e-6)
    np.testing.assert_allclose(result.power_ledger_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(
        result.absorbed_power
        + jnp.sum(result.detected_power, axis=-1)
        + result.escaped_power
        + result.discarded_power
        + result.ambiguous_power
        + result.truncated_power
        + result.live_power,
        result.launched_power,
        atol=2e-6,
    )
    assert int(result.status[0]) in {
        int(NonSequentialOpticsStatus.SUCCESS),
        int(NonSequentialOpticsStatus.INTERACTION_CAPACITY_EXHAUSTED),
    }


def test_history_enabled_and_disabled_have_identical_terminal_trace():
    surface = _slab_surfaces()
    common = dict(maximum_interactions=4, branch_capacity=12, power_tolerance=0.0)
    without_history = prepare_nonsequential_optics(
        NonSequentialOpticsPlan(surface, record_history=False, **common)
    )
    with_history = prepare_nonsequential_optics(
        NonSequentialOpticsPlan(surface, record_history=True, **common)
    )
    args = (_single_ray(), jnp.asarray([1.0]), jnp.asarray([0]))
    result_without = trace_nonsequential_optics(without_history, *args)
    result_with = trace_nonsequential_optics(with_history, *args)

    np.testing.assert_allclose(result_without.rays.origins, result_with.rays.origins)
    np.testing.assert_allclose(
        result_without.rays.directions, result_with.rays.directions
    )
    np.testing.assert_allclose(result_without.powers, result_with.powers)
    np.testing.assert_array_equal(result_without.live, result_with.live)
    np.testing.assert_array_equal(result_without.status, result_with.status)
    assert result_without.history_origins.shape[-3] == 0
    assert result_with.history_origins.shape[-3] == 5


def test_mirror_detector_and_branch_capacity_ledgers():
    vertices, triangles = _planes((0.0,))
    mirror = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0]),
        jnp.asarray([0, 0]),
        jnp.asarray([1.0]),
        surface_ids=jnp.asarray([0, 0]),
        surface_kinds=jnp.asarray([int(NonSequentialSurfaceKind.MIRROR)] * 2),
    )
    mirror_result = trace_nonsequential_optics(
        prepare_nonsequential_optics(
            NonSequentialOpticsPlan(mirror, maximum_interactions=2, branch_capacity=2)
        ),
        _single_ray(),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
    )
    np.testing.assert_allclose(mirror_result.escaped_power, 1.0)
    np.testing.assert_allclose(mirror_result.power_ledger_residual, 0.0)

    detector = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0]),
        jnp.asarray([0, 0]),
        jnp.asarray([1.0]),
        surface_ids=jnp.asarray([0, 0]),
        surface_kinds=jnp.asarray([int(NonSequentialSurfaceKind.DETECTOR)] * 2),
        detector_indices=jnp.asarray([0, 0]),
        detector_acceptance_cosines=jnp.asarray([0.5, 0.5]),
    )
    detector_result = trace_nonsequential_optics(
        prepare_nonsequential_optics(
            NonSequentialOpticsPlan(detector, maximum_interactions=1, branch_capacity=1)
        ),
        _single_ray(),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
    )
    np.testing.assert_allclose(detector_result.detected_power, [[1.0]])
    assert bool(detector_result.successful[0])

    capacity_result = trace_nonsequential_optics(
        prepare_nonsequential_optics(
            NonSequentialOpticsPlan(
                _slab_surfaces(),
                maximum_interactions=1,
                branch_capacity=1,
                power_tolerance=0.0,
            )
        ),
        _single_ray(),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
    )
    assert int(capacity_result.status[0]) == int(
        NonSequentialOpticsStatus.BRANCH_CAPACITY_EXHAUSTED
    )
    np.testing.assert_allclose(capacity_result.truncated_power, 0.96, atol=2e-6)
    np.testing.assert_allclose(capacity_result.live_power, 0.04, atol=2e-6)
    np.testing.assert_allclose(capacity_result.power_ledger_residual, 0.0, atol=2e-6)


def test_transmission_only_records_omitted_fresnel_branch():
    modes = jnp.full(
        (4,), int(NonSequentialBranchMode.TRANSMISSION_ONLY), dtype=jnp.int32
    )
    result = trace_nonsequential_optics(
        prepare_nonsequential_optics(
            NonSequentialOpticsPlan(
                _slab_surfaces(modes=modes),
                maximum_interactions=1,
                branch_capacity=2,
                power_tolerance=0.0,
            )
        ),
        _single_ray(),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
    )
    np.testing.assert_allclose(result.discarded_power, 0.04, atol=2e-6)
    np.testing.assert_allclose(result.live_power, 0.96, atol=2e-6)
    np.testing.assert_allclose(result.power_ledger_residual, 0.0, atol=2e-6)


def test_ambiguous_coincident_interfaces_stop_without_continuation():
    vertices = jnp.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = jnp.asarray([[0, 1, 2], [0, 1, 2]], dtype=jnp.int32)
    surfaces = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0]),
        jnp.asarray([1, 1]),
        jnp.asarray([1.0, 1.5]),
        surface_ids=jnp.asarray([0, 1]),
    )
    result = trace_nonsequential_optics(
        prepare_nonsequential_optics(
            NonSequentialOpticsPlan(surfaces, maximum_interactions=2, branch_capacity=2)
        ),
        _single_ray(),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
    )
    assert int(result.status[0]) == int(NonSequentialOpticsStatus.AMBIGUOUS_INTERSECTION)
    np.testing.assert_allclose(result.ambiguous_power, 1.0)
    np.testing.assert_allclose(result.live_power, 0.0)
    np.testing.assert_allclose(result.power_ledger_residual, 0.0)
