"""Prepared FE/FV meshes shared by facet graph contracts."""

import numpy as np
from jax.typing import ArrayLike

import phydrax as phx


VERTICES = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
    dtype=np.float64,
)


def finite_volume(
    triangles: ArrayLike,
) -> phx.discretization.UnstructuredFiniteVolumeDiscretization:
    return phx.discretization.UnstructuredFiniteVolumePlan(
        VERTICES, triangles=np.asarray(triangles, dtype=np.int32)
    ).prepare()


def finite_element(
    mesh: phx.discretization.CellMesh,
) -> phx.discretization.FiniteElementDiscretization:
    element = phx.discretization.lagrange_element("triangle", 1)
    return phx.discretization.FiniteElementPlan(
        mesh, (phx.discretization.FiniteElementFieldSpec("u", element),)
    ).prepare()
