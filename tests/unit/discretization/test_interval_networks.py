import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import IntervalConnectivity
from phydrax.discretization.fem import (
    discontinuous_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)


def _y_junction() -> phx.discretization.CellMesh:
    # Hub vertex 0 joins three branches ending at tips 1, 2 and 3.
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (-0.5, 0.8), (-0.5, -0.8)))
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "branches",
                "interval",
                np.asarray(((0, 1), (0, 2), (3, 0)), dtype=np.int32),
            ),
        ),
    )


def test_y_junction_interval_mesh_classifies_tips_as_boundary() -> None:
    mesh = _y_junction()
    connectivity = mesh.connectivity

    assert isinstance(connectivity, IntervalConnectivity)
    np.testing.assert_array_equal(connectivity.vertex_cell_counts, (3, 1, 1, 1))
    vertices = mesh.entity_set(0)
    np.testing.assert_array_equal(
        vertices.subset("boundary").mask, (False, True, True, True)
    )
    boundary = mesh.topology.incidences[0].scipy_boundary().toarray()
    np.testing.assert_array_equal(
        boundary,
        ((-1.0, -1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    )


def test_manifold_interval_chain_keeps_endpoint_boundary() -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(((0.0,), (1.0,), (2.0,))),
        (
            phx.discretization.CellBlock(
                "chain", "interval", np.asarray(((0, 1), (1, 2)), dtype=np.int32)
            ),
        ),
    )

    connectivity = mesh.connectivity
    assert isinstance(connectivity, IntervalConnectivity)
    np.testing.assert_array_equal(connectivity.vertex_cell_counts, (1, 2, 1))
    np.testing.assert_array_equal(
        mesh.entity_set(0).subset("boundary").mask, (True, False, True)
    )


def test_finite_element_facet_pairing_refuses_junction_meshes() -> None:
    mesh = _y_junction()
    plan = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", discontinuous_element("interval", 0))
    )

    with pytest.raises(ValueError, match="at most two cells"):
        plan.prepare()
