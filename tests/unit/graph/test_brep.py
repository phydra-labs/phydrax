"""Independent sphere incidence and differentiable scalar-consumer smoke."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.geometry.brep._constructors import brep_sphere
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._differentiable import (
    _FixedTopologyBRepKernel,
    FixedTopologyBRepSource,
)
from phydrax.geometry.brep._edge_classification import brep_edge_classification
from phydrax.geometry.brep._measure import brep_mass_properties
from phydrax.geometry.brep._query import BRepQueryPolicy
from phydrax.geometry.design._naming import RecipeEntityName, RecipeEntityNames
from phydrax.graph._brep import brep_incidence_graph, prepare_brep_graph_samples
from tests._support.cad_models import native_solid


# Empirical `test_scalar_graph_sample_design_gradient`: matched max |delta|=2.8535e-10; joint allowance fraction=0.000951165, margin=99.9049%.
_SPHERE_GRAPH_DESIGN_GRADIENT_RTOL = 1e-8


def test_sphere_incidence_seams_and_poles() -> None:
    model = native_solid("sphere")
    record = brep_incidence_graph(model)
    assert record.model_id == model.model_id
    assert sum(entity.kind == "face" for entity in record.entity_ids) == 1
    types = np.asarray(record.graph.nodes["coedge_type"])
    assert np.count_nonzero(types == 1) == 2
    assert np.count_nonzero(types == 2) == 2
    relations = np.asarray(record.graph.edges["type"])
    assert np.count_nonzero(relations == record.relation_kinds.index("seam_mate")) == 2


def test_scalar_graph_sample_design_gradient() -> None:
    model = native_solid("sphere")
    compiled = FixedTopologyBRepSource(model).compile(
        coordinate_unit=model.coordinate_contract.length_unit
    )
    kernel = compiled.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    radius_index = next(
        index
        for index, spec in enumerate(compiled.schema.specs)
        if spec.parameter_id.name == "radius"
    )
    with coordinate_source_request((model, 100)) as budget:
        plan = prepare_brep_graph_samples(
            model,
            grid_size=3,
            coedge_samples=3,
            policy=BRepQueryPolicy(
                maximum_operations=100_000,
                maximum_points=10_000,
                maximum_scratch_bytes=1 << 20,
            ),
            budget=budget,
        )

    def scalar(radius: jax.Array) -> jax.Array:
        state = compiled.state.replace_at(radius_index, radius)
        realization = kernel.realize(state)
        samples = plan.evaluate(realization)
        weights = samples.face_quadrature[0].quadrature_weights
        assert weights is not None
        return jnp.sum(samples.face_grids[0][:3] ** 2) + jnp.sum(weights) / (4 * jnp.pi)

    radius = jnp.asarray(1.5, dtype=jnp.float64)
    derivative = jax.grad(scalar)(radius)
    step = 1e-5
    finite_difference = (scalar(radius + step) - scalar(radius - step)) / (2 * step)
    assert float(derivative) == pytest.approx(
        float(finite_difference), rel=_SPHERE_GRAPH_DESIGN_GRADIENT_RTOL
    )
    assert float(derivative) == pytest.approx(30.0)


def test_graph_recipe_face_identity_survives_radius_edit() -> None:
    original = native_solid("sphere")
    edited = brep_sphere(2.0, coordinate_contract=original.coordinate_contract)
    name = RecipeEntityName("face", "body", "sphere", ("primitive", "boundary"))
    first = brep_incidence_graph(
        original, recipe_names=RecipeEntityNames(original, (name,), None)
    )
    second = brep_incidence_graph(
        edited, recipe_names=RecipeEntityNames(edited, (name,), None)
    )
    assert first.model_id != second.model_id
    assert first.recipe_names is not None and second.recipe_names is not None
    assert first.recipe_names.face_names == second.recipe_names.face_names


def test_graph_attaches_certified_owner_descriptors() -> None:
    model = native_solid("box")
    geometry = model.require_geometry()
    with coordinate_source_request((model, 100)) as budget:
        mass = brep_mass_properties(
            model, geometry, subdivisions=1, coordinate_budget=budget
        )
        edges = brep_edge_classification(
            model, geometry, maximum_cells=8, coordinate_budget=budget
        )
    graph = brep_incidence_graph(model, mass_properties=mass, edge_classifications=edges)
    assert graph.mass_properties is mass
    assert graph.edge_classifications == edges
    assert graph.mass_properties.face_area_enclosures.shape == (6, 2)
    assert all(record.status == "convex" for record in graph.edge_classifications)


def test_graph_sample_budget_refuses_before_materializing_grid() -> None:
    model = native_solid("sphere")
    with coordinate_source_request((model,)) as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            prepare_brep_graph_samples(
                model,
                grid_size=budget.maximum_work_units + 2,
                coedge_samples=2,
                policy=BRepQueryPolicy(),
                budget=budget,
            )
