import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    BlockDofLayout,
    CellMesh,
    EntityDofLayout,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization.vem import (
    conforming_h1_virtual_element,
    VirtualElementFieldSpec,
    VirtualElementPlan,
)
from phydrax.discretization.vem._operator import FactorizedVirtualElementOperator


# Empirical `test_factorized_scatter_transpose_with_repeated_quotient_dofs`: matched max |delta|=2.22045e-16; joint allowance fraction=2.01857e-09, margin=100%.
_QUOTIENT_SCATTER_VALUES_ATOL = 1e-12
# Empirical `test_factorized_scatter_transpose_with_repeated_quotient_dofs`: matched max |delta|=2.22045e-16; joint allowance fraction=0.000222045, margin=99.9778%.
_QUOTIENT_SCATTER_DUAL_PAIRING_ABS = 1e-12


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_scalar_h1_winding_quotient_routes_and_physical_boundary(
    degree: int,
) -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lifted = CellMesh.from_polygons(points, (np.asarray((0, 1, 2, 3), dtype=np.int64),))
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(2)),
        np.zeros(4, dtype=np.int32),
        points.astype(np.int32),
    )
    mesh = CellMesh(points, lifted.blocks, periodic_topology=topology)
    prepared = VirtualElementPlan(
        mesh,
        VirtualElementFieldSpec("u", conforming_h1_virtual_element(degree)),
    ).prepare()
    dofs = prepared.dof_map
    assert prepared.field_space.vector_space.size == dofs.global_dof_count
    layout = prepared.field_space.layout
    if (
        not isinstance(layout, BlockDofLayout)
        or not layout.layouts
        or not isinstance(layout.layouts[0], EntityDofLayout)
    ):
        raise AssertionError(
            f"fixture.layout must contain a quotient EntityDofLayout in a BlockDofLayout; got {layout!r}"
        )
    assert layout.layouts[0].entity_set_id == topology.quotient.entities(0).entity_set_id
    trace = np.asarray(prepared.edge_trace_routes(np.arange(4)))
    np.testing.assert_array_equal(trace[:, (0, -1)], 0)
    route = np.asarray(dofs.cell_dofs[0])[0]
    np.testing.assert_array_equal(route[:4], 0)
    assert dofs.vertex_dof_count == 1
    # Horizontal and vertical edges remain separate winding entities.
    assert dofs.edge_dof_count == 2 * (degree - 1)
    assert not np.any(dofs.boundary_dof_mask)
    if degree > 1:
        np.testing.assert_array_equal(
            route[4 : 4 + degree - 1],
            route[4 + 2 * (degree - 1) : 4 + 3 * (degree - 1)][::-1],
        )


def test_factorized_scatter_transpose_with_repeated_quotient_dofs() -> None:
    coefficient = np.eye(3)[None]
    matrix = np.asarray([[[2.0, -1.0, 3.0], [4.0, 5.0, 0.0], [-2.0, 1.0, 7.0]]])
    stabilization = np.asarray([[[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0]]])
    gather = np.asarray([[0, 1, 0]], dtype=np.int32)
    operator = FactorizedVirtualElementOperator(
        (coefficient,), (matrix,), (stabilization,), (gather,), 2
    )
    x, y = jnp.asarray([0.3, -0.2]), jnp.asarray([-0.4, 0.7])
    assert float(jnp.vdot(x, operator.mv(y))) == pytest.approx(
        float(jnp.vdot(operator.transpose_mv(x), y)),
        abs=_QUOTIENT_SCATTER_DUAL_PAIRING_ABS,
    )
    lift = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    expected = lift.T @ (matrix[0] + stabilization[0]) @ lift
    np.testing.assert_allclose(
        operator.mv(y), expected @ y, atol=_QUOTIENT_SCATTER_VALUES_ATOL
    )
