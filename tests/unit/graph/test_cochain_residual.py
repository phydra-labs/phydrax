#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import simplicial_cell_geometry
from tests._support.cochain import reoriented_lowering, triangle_cochain_lowering


def _square_complex() -> Any:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return triangle_cochain_lowering(vertices, faces)


def test_cochain_field_masks_other_degrees_and_preserves_compatible_metadata() -> None:
    complex_ir = _square_complex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    all_cells = domain.component({"graph": phx.domain.Nodes()}).sample(
        phx.domain.PointSampling(complex_ir.num_cells, layout=structure)
    )
    zero_spec = phx.exterior.FormType(2, 0)
    one_spec = phx.exterior.FormType(2, 1)

    @domain.Function("graph")
    def raw(cell: Any) -> Any:
        return 1.0 + cell["local_index"]

    zero_form = phx.domain.as_cochain_field(raw, zero_spec, representation="cochain")
    another_zero_form = phx.domain.as_cochain_field(
        2.0 * raw, zero_spec, representation="cochain"
    )
    one_form = phx.domain.as_cochain_field(raw, one_spec, representation="cochain")
    values = zero_form(all_cells).data
    degree = all_cells["graph"]["cell_dim"].data

    assert jnp.all(values[degree != 0] == 0.0)
    assert phx.domain.cochain_form_type(3.0 * zero_form - another_zero_form) == zero_spec
    with pytest.raises(ValueError):
        phx.domain.cochain_form_type(zero_form + one_form)


def test_domain_cochain_dec_matches_independent_incidence_oracle() -> None:
    complex_ir = _square_complex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    edge_batch = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[1], layout=structure)
    )
    face_batch = domain.component({"graph": phx.domain.CochainCells(2)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[2], layout=structure)
    )
    zero_spec = phx.exterior.FormType(2, 0)

    @domain.Function("graph")
    def raw(cell: Any) -> Any:
        return 0.25 + cell["local_index"]

    zero_form = phx.domain.as_cochain_field(raw, zero_spec, representation="cochain")
    derivative = phx.operators.cochain_exterior_derivative(zero_form)
    second_derivative = phx.operators.cochain_exterior_derivative(derivative)
    laplacian = phx.operators.cochain_hodge_laplacian(zero_form)
    cells, _ = simplicial_cell_geometry(complex_ir.discretization.topology)
    # Independent endpoint oracle, keyed by global vertex identities rather than
    # the storage position of an edge in the canonical topology.
    endpoint_rows = {
        (0, 1): [-1.0, 1.0, 0.0, 0.0],
        (0, 2): [-1.0, 0.0, 1.0, 0.0],
        (0, 3): [-1.0, 0.0, 0.0, 1.0],
        (1, 2): [0.0, -1.0, 1.0, 0.0],
        (2, 3): [0.0, 0.0, -1.0, 1.0],
    }
    incidence = np.asarray([endpoint_rows[tuple(edge)] for edge in cells[1]])
    coefficients = np.asarray([0.25, 1.25, 2.25, 3.25], dtype=np.float64)
    expected_derivative = np.zeros((complex_ir.num_cells,), dtype=np.float64)
    expected_derivative[4:9] = incidence @ coefficients
    realization = complex_ir.discretization
    mass_zero = np.asarray(realization.hodge_diagonal(0))
    mass_one = np.asarray(realization.hodge_diagonal(1))
    expected_laplacian = incidence.T @ (mass_one * (incidence @ coefficients)) / mass_zero
    edge_indices = edge_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data
    vertex_batch = domain.component({"graph": phx.domain.CochainCells(0)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[0], layout=structure)
    )
    vertex_indices = vertex_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data

    assert phx.domain.cochain_form_type(derivative).degree == 1
    assert phx.domain.cochain_form_type(second_derivative).degree == 2
    assert jnp.allclose(
        jnp.asarray(derivative(edge_batch).data), expected_derivative[edge_indices]
    )
    assert jnp.allclose(jnp.asarray(second_derivative(face_batch).data), 0.0, atol=1e-12)
    assert jnp.allclose(
        jnp.asarray(laplacian(vertex_batch).data), expected_laplacian[vertex_indices]
    )


def test_domain_cochain_laplacian_is_equivariant_to_cell_reorientation() -> None:
    complex_ir = _square_complex()
    signs = (
        np.ones((complex_ir.cell_counts[0],), dtype=np.float64),
        np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0]),
    )
    reoriented = reoriented_lowering(complex_ir, signs)
    values = jnp.asarray([0.4, -0.1, 0.8, -0.3, 0.6])
    transformed_values = phx.discretization.reorient_cochain(
        values, signs[1], cell_axis=0
    )
    one_spec = phx.exterior.FormType(2, 1)
    structure = phx.domain.SampleLayout((("graph",),))

    def laplacian_values(bundle: Any, coefficients: Any) -> Any:
        domain = phx.domain.GraphDomain(bundle.graph)

        @domain.Function("graph")
        def raw(cell: Any) -> Any:
            index = jnp.where(cell["cell_dim"] == 1, cell["local_index"], 0)
            return coefficients[index]

        one_form = phx.domain.as_cochain_field(raw, one_spec, representation="cochain")
        batch = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
            phx.domain.PointSampling(bundle.cell_counts[1], layout=structure)
        )
        return phx.operators.cochain_hodge_laplacian(one_form)(batch).data

    original = laplacian_values(complex_ir, values)
    transformed = laplacian_values(reoriented, transformed_values)

    assert jnp.allclose(
        transformed,
        phx.discretization.reorient_cochain(original, signs[1], cell_axis=0),
        atol=1e-12,
    )


def test_cochain_metric_reductions_ignore_padding_and_compose_segment_weights() -> None:
    values = jnp.asarray([1.0, 3.0, 1000.0, 2.0])
    metric = jnp.asarray([1.0, 3.0, 1000.0, 2.0])
    graph_index = jnp.asarray([0, 0, -1, 1], dtype=jnp.int32)
    active = jnp.asarray([True, True, False, True])

    graph_mean = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="graph_mean",
        entity_mask=active,
    )
    metric_mean = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_mean",
        entity_mask=active,
    )
    metric_sum = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_sum",
        entity_mask=active,
    )
    weighted_sum = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_sum",
        segment_weight=jnp.asarray([2.0, 2.0, 1000.0, 4.0]),
        entity_mask=active,
    )

    assert jnp.allclose(graph_mean, 2.0)
    assert jnp.allclose(metric_mean, 2.25)
    assert jnp.allclose(metric_sum, 7.0)
    assert jnp.allclose(weighted_sum, 18.0)


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("time_integral_average", 1.5),
        ("time_integral_sum", 3.0),
    ],
)
def test_cochain_residual_constraint_composes_graph_and_time_measures(
    measure: Any,
    expected: Any,
) -> None:
    complex_ir = _square_complex()
    base = phx.domain.GraphTrajectoryDatasetDomain(
        (complex_ir.graph, complex_ir.graph),
        jnp.asarray([2, 3], dtype=jnp.int32),
        dt=1.0,
        measure=measure,
    )
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=4))
    component = domain.component(
        {
            "graph": phx.domain.CochainCells(0),
            "t": phx.domain.Interior(),
        }
    )
    structure = phx.domain.SampleLayout((("graph", "t"),))
    zero_spec = phx.exterior.FormType(2, 0)

    @domain.Function("graph", "t")
    def unit_residual(cell: Any, time: Any) -> Any:
        return jnp.ones_like(time) + 0.0 * cell["local_index"]

    residual = phx.domain.as_cochain_field(
        unit_residual, zero_spec, representation="cochain"
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=structure,
    )
    constraint = phx.terms.CochainResidualTerm(
        component=component,
        residual=lambda functions: functions["u"],
        fields=("u",),
        sampling=phx.domain.PointSampling(2, layout=structure),
        reduction="graph_mean",
    )

    assert jnp.allclose(
        constraint.loss({"u": residual}, batch=batch),
        expected,
    )
