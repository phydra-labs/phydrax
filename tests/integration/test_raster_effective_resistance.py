#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Raster sheet conductance to grounded-solve effective resistance, end to end.

The oracle assembles the four-neighbour two-point-flux Laplacian directly from
the raster with NumPy (series face conductance on a uniform pixel grid) and
solves the grounded system with SciPy's sparse LU; it shares no code with the
finite-volume, graph, or linear-algebra paths under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import phydrax as phx
from phydrax.interchange import GeospatialContract, QualifiedGeospatialGrid
from phydrax.units import METER, SIEMENS


_SPACING_X = 2.0
_SPACING_Y = 0.5


def _contract() -> GeospatialContract:
    return GeospatialContract.local_cartesian(
        phx.SpatialCoordinateContract(METER, reference_frame="survey-site"),
        vertical_datum="survey-benchmark-A",
        registration="pixel",
        mask_semantics="valid_true",
    )


def _raster(values: np.ndarray, valid: np.ndarray) -> QualifiedGeospatialGrid:
    rows, columns = values.shape
    x = 10.0 + _SPACING_X * np.arange(columns)
    # North-up raster: row 0 is the northernmost row, so y decreases.
    y = 50.0 - _SPACING_Y * np.arange(rows)
    return QualifiedGeospatialGrid(
        x, y, values, _contract(), value_unit=SIEMENS, valid=valid
    )


def _oracle_resistance(
    values: np.ndarray, valid: np.ndarray, sites: list[tuple[int, int]]
) -> np.ndarray:
    """Dense-free grounded oracle in raster (row, column) order."""
    rows, columns = values.shape
    index = -np.ones(values.shape, dtype=np.int64)
    index[valid] = np.arange(int(valid.sum()))
    count = int(valid.sum())
    entries: list[tuple[int, int, float]] = []
    for row in range(rows):
        for column in range(columns):
            if not valid[row, column]:
                continue
            for d_row, d_column, length, width in (
                (0, 1, _SPACING_X, _SPACING_Y),
                (1, 0, _SPACING_Y, _SPACING_X),
            ):
                other_row, other_column = row + d_row, column + d_column
                if other_row >= rows or other_column >= columns:
                    continue
                if not valid[other_row, other_column]:
                    continue
                left = values[row, column]
                right = values[other_row, other_column]
                conductance = width / (0.5 * length / left + 0.5 * length / right)
                entries.append(
                    (index[row, column], index[other_row, other_column], conductance)
                )
    laplacian = sp.lil_matrix((count, count))
    for u, v, w in entries:
        laplacian[u, u] += w
        laplacian[v, v] += w
        laplacian[u, v] -= w
        laplacian[v, u] -= w
    adjacency = sp.csr_matrix(
        ([1.0] * len(entries), ([u for u, _, _ in entries], [v for _, v, _ in entries])),
        shape=(count, count),
    )
    _, labels = sp.csgraph.connected_components(adjacency, directed=False)
    grounds = [int(np.flatnonzero(labels == label)[0]) for label in np.unique(labels)]
    keep = np.ones(count, dtype=np.bool_)
    keep[grounds] = False
    kept = np.flatnonzero(keep)
    reduced = sp.csc_matrix(laplacian.tocsr()[kept, :][:, kept])
    reduced_index = -np.ones(count, dtype=np.int64)
    reduced_index[kept] = np.arange(kept.size, dtype=np.int64)
    nodes = [int(index[row, column]) for row, column in sites]
    rhs = np.zeros((kept.size, len(nodes)))
    for column, node in enumerate(nodes):
        if reduced_index[node] >= 0:
            rhs[reduced_index[node], column] = 1.0
    solution = spla.splu(reduced).solve(rhs)
    potentials = np.zeros((len(nodes), len(nodes)))
    for a, node in enumerate(nodes):
        if reduced_index[node] >= 0:
            potentials[a] = solution[reduced_index[node]]
    diagonal = np.diag(potentials)
    result = diagonal[:, None] + diagonal[None, :] - potentials - potentials.T
    same = labels[nodes][:, None] == labels[nodes][None, :]
    return np.where(same, result, np.inf)


def _pipeline(
    raster: QualifiedGeospatialGrid,
    sites: list[tuple[int, int]],
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[
    phx.graph.EffectiveResistancePlan,
    phx.discretization.finite_volume.StructuredFaceConductancePlan,
    phx.interchange.GeospatialCellField,
]:
    field = phx.interchange.geospatial_cell_field(
        raster, phx.SpatialCoordinateContract(METER, reference_frame="survey-site")
    )
    conductance = phx.discretization.finite_volume.StructuredFaceConductancePlan(
        field.grid, np.asarray(field.valid)
    )
    raster_cells = np.zeros(raster.values.shape, dtype=np.int64)
    raster_cells[tuple(np.asarray(sites).T)] = 1 + np.arange(len(sites))
    tensor_cells = np.asarray(field.raster_to_tensor(raster_cells))
    order = np.argsort(tensor_cells[tensor_cells > 0])
    cells = np.argwhere(tensor_cells > 0)[order]
    nodes = conductance.node_indices(cells)
    plan = phx.graph.EffectiveResistancePlan(conductance.relation, nodes, policy=policy)
    return plan, conductance, field


@pytest.fixture(scope="module")
def landscape() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261011)
    values = rng.uniform(0.2, 3.0, size=(9, 12))
    valid = np.ones(values.shape, dtype=bool)
    valid[4, :10] = False  # barrier with a gap at the east edge
    valid[7:, 9:] = False
    valid[8, 11] = True  # isolated cell: its own component
    return values, valid


def test_raster_resistance_matches_independent_grounded_oracle(
    landscape: tuple[np.ndarray, np.ndarray],
) -> None:
    values, valid = landscape
    sites = [(0, 0), (8, 0), (2, 7), (6, 3), (8, 11)]
    plan, conductance, field = _pipeline(_raster(values, valid), sites)
    result = plan.solve(conductance.conductances(field.values))

    assert int(result.status) == int(phx.graph.EffectiveResistanceStatus.SUCCESS)
    expected = _oracle_resistance(values, valid, sites)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(
        np.asarray(result.resistance)[finite], expected[finite], rtol=1e-7, atol=1e-10
    )
    assert np.all(np.isposinf(np.asarray(result.resistance)[~finite]))
    np.testing.assert_array_equal(np.asarray(result.connected), finite)


def test_raster_resistance_cell_sensitivity_matches_finite_differences(
    landscape: tuple[np.ndarray, np.ndarray],
) -> None:
    values, valid = landscape
    sites = [(0, 0), (8, 0)]
    # Finite differences need solves tighter than the default tolerance: with
    # step h the central quotient carries solve noise of order tol * R / h.
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-11, absolute=1e-14, max_steps=4096
        ),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        ),
    )
    plan, conductance, field = _pipeline(_raster(values, valid), sites, policy)

    def resistance(cells: jax.Array) -> jax.Array:
        return plan.solve(conductance.conductances(cells)).resistance[0, 1]

    cells = jnp.asarray(field.values)
    gradient = np.asarray(jax.jit(jax.grad(resistance))(cells))
    probes = [(1, 1), (5, 11), (3, 4), (6, 2)]
    step = 1e-3
    for probe in probes:
        bump = jnp.zeros_like(cells).at[probe].set(step)
        upper = plan.solve(conductance.conductances(cells + bump))
        lower = plan.solve(conductance.conductances(cells - bump))
        success = int(phx.graph.EffectiveResistanceStatus.SUCCESS)
        assert int(upper.status) == success and int(lower.status) == success
        central = (upper.resistance[0, 1] - lower.resistance[0, 1]) / (2 * step)
        np.testing.assert_allclose(gradient[probe], float(central), rtol=5e-5, atol=1e-12)
    # Masked cells carry no conductance and therefore no sensitivity.
    assert np.all(gradient[~np.asarray(field.valid)] == 0.0)
