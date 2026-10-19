"""Periodic high-order geometry preserves quotient-node seam traces."""

from __future__ import annotations

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    FiniteElementSpec,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.meshing._curving import apply_periodic, periodic_nodes, straight_geometry


# Unchanged empirical allowance from the pre-refactor straight-edge, seam, plane-distance and zero-mean reference assertions.
CURVING_COORDINATE_ROUNDOFF_ATOL = 1e-12


@pytest.mark.parametrize("degree", (3, 4))
@pytest.mark.parametrize("rotation", (False, True))
def test_quotient_curving_nodes_preserve_oriented_high_order_seam_traces(
    degree: int,
    rotation: bool,
) -> None:

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    block = phx.discretization.CellBlock(
        "wall", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    )
    generators = np.repeat(np.eye(4)[None], 2, axis=0)
    generators[0, 0, 3] = 1.0
    generators[1, 1, 3] = 1.0
    representatives = np.zeros(4, dtype=np.int64)
    shifts = points[:, :2].astype(np.int64)
    if rotation:
        points = np.asarray(
            ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0))
        )
        generators = np.asarray(
            (
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )
        representatives = np.asarray((0, 1, 1, 0), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
    lifted = phx.discretization.CellMesh(points, (block,))
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(generators),
        representatives,
        shifts,
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    straight = straight_geometry(wall, degree)
    pairs = periodic_nodes(wall, straight, ())
    moved = np.asarray(straight.coordinates).copy()
    moved[4:, 2] = 0.02 * np.sin(np.arange(4, moved.shape[0]))
    coordinates = apply_periodic(moved, pairs)
    elements, routes, _ = straight.resolve(wall)
    t = np.linspace(0.05, 0.95, 9)
    element = elements[0]
    if not isinstance(element, FiniteElementSpec):
        raise AssertionError(
            "curving_reference.element must retain a finite-element basis; got another coordinate element"
        )
    route = np.asarray(routes[0])
    if rotation:
        source = (
            np.asarray(element.tabulate(np.stack((t, np.zeros_like(t)), axis=1))[0])
            @ coordinates[route[0]]
        )
        target = (
            np.asarray(element.tabulate(np.stack((t, 1.0 - t), axis=1))[0])
            @ coordinates[route[1]]
        )
        np.testing.assert_allclose(
            target,
            source @ generators[0, :3, :3].T,
            atol=CURVING_COORDINATE_ROUNDOFF_ATOL,
        )
        assert np.max(np.abs(source[:, 2])) > 1e-4
    else:
        left = (
            np.asarray(element.tabulate(np.stack((np.zeros_like(t), t), axis=1))[0])
            @ coordinates[route[1]]
        )
        right = (
            np.asarray(element.tabulate(np.stack((1.0 - t, t), axis=1))[0])
            @ coordinates[route[0]]
        )
        bottom = (
            np.asarray(element.tabulate(np.stack((t, np.zeros_like(t)), axis=1))[0])
            @ coordinates[route[0]]
        )
        top = (
            np.asarray(element.tabulate(np.stack((t, 1.0 - t), axis=1))[0])
            @ coordinates[route[1]]
        )
        np.testing.assert_allclose(
            right - left,
            np.tile((1.0, 0.0, 0.0), (t.size, 1)),
            atol=CURVING_COORDINATE_ROUNDOFF_ATOL,
        )
        np.testing.assert_allclose(
            top - bottom,
            np.tile((0.0, 1.0, 0.0), (t.size, 1)),
            atol=CURVING_COORDINATE_ROUNDOFF_ATOL,
        )
        assert np.max(np.abs(left[:, 2])) > 1e-4
        assert len(set(topology.entity_keys(1))) == 3
