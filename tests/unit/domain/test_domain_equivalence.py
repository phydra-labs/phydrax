#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

import phydrax as phx
from phydrax.domain import (
    DatasetDomain,
    HyperRectangle,
    Interval1d,
    ProductDomain,
    TimeInterval,
)


def _permute_mesh(points: np.ndarray, faces: np.ndarray, perm: np.ndarray) -> Any:
    pts2 = points[perm]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])
    faces2 = inv[faces]
    return pts2, faces2


def test_domain_equivalence_scenario_1() -> None:
    a = TimeInterval(0.0, 1.0)
    b = TimeInterval(0.0, 1.0)
    c = TimeInterval(0.0, 2.0)
    assert a.same_support(b)
    assert not a.same_support(c)
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 1.0)
    c = Interval1d(0.0, 2.0)
    assert a.same_support(b)
    assert not a.same_support(c)
    data1 = {
        "a": jnp.zeros((4, 2), dtype="float64"),
        "b": jnp.ones((4,), dtype="float64"),
    }
    data2 = {
        "a": jnp.ones((4, 2), dtype="float64") * 3.0,
        "b": jnp.arange(4.0, dtype="float64"),
    }
    dom1 = DatasetDomain(data1, label="data", measure="probability")
    dom2 = DatasetDomain(data2, label="data", measure="probability")
    assert dom1.same_support(dom2)

    dom3 = DatasetDomain(data2, label="data", measure="count")
    assert not dom1.same_support(dom3)

    dom4 = DatasetDomain(data2, label="other", measure="probability")
    assert not dom1.same_support(dom4)

    data3 = {"a": jnp.zeros((4, 3), dtype="float64")}
    dom5 = DatasetDomain(data3, label="data", measure="probability")
    assert not dom1.same_support(dom5)


@pytest.mark.parametrize(
    "make",
    [
        lambda: HyperRectangle(np.zeros(2), np.ones(2)),
        lambda: Interval1d(0.0, 1.0),
        lambda: TimeInterval(0.0, 1.0),
    ],
    ids=("hyperrectangle", "interval1d", "scalar-interval"),
)
def test_traced_support_is_decided_by_identity_never_by_traced_values(
    make: Any,
) -> None:
    first, copy = make(), make()
    decisions = []

    @eqx.filter_jit
    def stage(factor: Any, other: Any) -> Any:
        decisions.append(factor.same_support(factor))
        with pytest.raises(ValueError, match="support-defining values are traced"):
            factor.same_support(other)
        return jnp.zeros(())

    stage(first, copy)
    assert decisions == [True]
    assert first.same_support(copy)


def test_domain_equivalence_scenario_2() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float64",
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh1 = meshio.Mesh(points=points, cells=[("triangle", faces)])

    perm = np.array([1, 2, 0], dtype=np.int64)
    points2, faces2 = _permute_mesh(points, faces, perm)
    mesh2 = meshio.Mesh(points=points2, cells=[("triangle", faces2)])

    geom1 = phx.domain.GeometryDomain(
        phx.geometry.planar_region_from_source(mesh1, recenter=False).compile()
    )
    geom2 = phx.domain.GeometryDomain(
        phx.geometry.planar_region_from_source(mesh2, recenter=False).compile()
    )
    assert geom1.same_support(geom2)

    points3 = points.copy()
    points3[0, 0] += 1e-3
    mesh3 = meshio.Mesh(points=points3, cells=[("triangle", faces)])
    geom3 = phx.domain.GeometryDomain(
        phx.geometry.planar_region_from_source(mesh3, recenter=False).compile()
    )
    assert not geom1.same_support(geom3)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float64",
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    mesh1 = (points, faces)

    perm = np.array([2, 0, 3, 1], dtype=np.int64)
    points2, faces2 = _permute_mesh(points, faces, perm)
    mesh2 = (points2, faces2)

    geom1 = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(mesh1, recenter=False).compile()
    )
    geom2 = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(mesh2, recenter=False).compile()
    )
    assert geom1.same_support(geom2)

    points3 = points.copy()
    points3[1, 2] += 1e-3
    mesh3 = (points3, faces)
    geom3 = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(mesh3, recenter=False).compile()
    )
    assert not geom1.same_support(geom3)
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 1.0)
    dom = ProductDomain(a, b)
    assert dom.labels == ("x",)
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 2.0)
    with pytest.raises(ValueError):
        ProductDomain(a, b)
