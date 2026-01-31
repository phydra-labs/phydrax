#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import meshio
import numpy as np
import pytest
import trimesh

from phydrax.domain import (
    DatasetDomain,
    Geometry2DFromCAD,
    Geometry3DFromCAD,
    Interval1d,
    ProductDomain,
    TimeInterval,
)


def _permute_mesh(points: np.ndarray, faces: np.ndarray, perm: np.ndarray):
    pts2 = points[perm]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])
    faces2 = inv[faces]
    return pts2, faces2


def test_time_interval_equivalence():
    a = TimeInterval(0.0, 1.0)
    b = TimeInterval(0.0, 1.0)
    c = TimeInterval(0.0, 2.0)
    assert a.equivalent(b)
    assert not a.equivalent(c)


def test_interval1d_equivalence():
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 1.0)
    c = Interval1d(0.0, 2.0)
    assert a.equivalent(b)
    assert not a.equivalent(c)


def test_dataset_domain_equivalence():
    data1 = {"a": jnp.zeros((4, 2), dtype=float), "b": jnp.ones((4,), dtype=float)}
    data2 = {
        "a": jnp.ones((4, 2), dtype=float) * 3.0,
        "b": jnp.arange(4.0, dtype=float),
    }
    dom1 = DatasetDomain(data1, label="data", measure="probability")
    dom2 = DatasetDomain(data2, label="data", measure="probability")
    assert dom1.equivalent(dom2)

    dom3 = DatasetDomain(data2, label="data", measure="count")
    assert not dom1.equivalent(dom3)

    dom4 = DatasetDomain(data2, label="other", measure="probability")
    assert not dom1.equivalent(dom4)

    data3 = {"a": jnp.zeros((4, 3), dtype=float)}
    dom5 = DatasetDomain(data3, label="data", measure="probability")
    assert not dom1.equivalent(dom5)


def test_geometry2d_equivalence_strong():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh1 = meshio.Mesh(points=points, cells=[("triangle", faces)])

    perm = np.array([1, 2, 0], dtype=np.int64)
    points2, faces2 = _permute_mesh(points, faces, perm)
    mesh2 = meshio.Mesh(points=points2, cells=[("triangle", faces2)])

    geom1 = Geometry2DFromCAD(mesh1, recenter=False)
    geom2 = Geometry2DFromCAD(mesh2, recenter=False)
    assert geom1.equivalent(geom2)

    points3 = points.copy()
    points3[0, 0] += 1e-3
    mesh3 = meshio.Mesh(points=points3, cells=[("triangle", faces)])
    geom3 = Geometry2DFromCAD(mesh3, recenter=False)
    assert not geom1.equivalent(geom3)


def test_geometry3d_equivalence_strong():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    mesh1 = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    perm = np.array([2, 0, 3, 1], dtype=np.int64)
    points2, faces2 = _permute_mesh(points, faces, perm)
    mesh2 = trimesh.Trimesh(vertices=points2, faces=faces2, process=False)

    geom1 = Geometry3DFromCAD(mesh1, recenter=False)
    geom2 = Geometry3DFromCAD(mesh2, recenter=False)
    assert geom1.equivalent(geom2)

    points3 = points.copy()
    points3[1, 2] += 1e-3
    mesh3 = trimesh.Trimesh(vertices=points3, faces=faces, process=False)
    geom3 = Geometry3DFromCAD(mesh3, recenter=False)
    assert not geom1.equivalent(geom3)


def test_product_domain_label_collision_equivalent():
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 1.0)
    dom = ProductDomain(a, b)
    assert dom.labels == ("x",)


def test_product_domain_label_collision_not_equivalent():
    a = Interval1d(0.0, 1.0)
    b = Interval1d(0.0, 2.0)
    with pytest.raises(ValueError):
        ProductDomain(a, b)
