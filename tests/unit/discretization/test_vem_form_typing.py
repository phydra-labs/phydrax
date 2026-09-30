#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._model._ports import PortVariance, ValuePort
from phydrax.discretization import CellMesh, PolygonalConnectivity
from phydrax.discretization.vem import (
    conforming_h1_virtual_element,
    conforming_hcurl_virtual_element,
    conforming_hdiv_virtual_element,
    discontinuous_l2_virtual_element,
    prepare_polyhedral_h1_virtual_element_3d,
    VirtualElementDiscretization,
    VirtualElementFieldSpec,
    VirtualElementPlan,
    VirtualElementSpec,
)
from phydrax.equations.vem import (
    prepare_virtual_element_field_reconstruction,
    VirtualElementReconstructionChannel,
)
from phydrax.exterior import FormProxy, FormTwist, FormType, FormValueSpec


def _square_space(
    factory: Callable[[int], VirtualElementSpec],
) -> VirtualElementDiscretization:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    mesh = CellMesh.from_polygons(coordinates, (np.arange(4, dtype=np.int32),))
    return VirtualElementPlan(mesh, VirtualElementFieldSpec("u", factory(1))).prepare()


def _polynomial_dofs(space: VirtualElementDiscretization) -> tuple[Array, np.ndarray]:
    """Independent scalar means and affine vector moments on the unit square."""
    if space.field.element.value_spec.proxy == "scalar":
        value = np.asarray(2.5, dtype=np.float64)
        if space.field.element.family == "ConformingH1":
            return jnp.full((4,), 2.5, dtype=jnp.float64), value
        # The centered linear monomials have zero mean on the unit square.
        return jnp.asarray((2.5, 0.0, 0.0), dtype=jnp.float64), value
    value = np.asarray((0.7, -1.3), dtype=np.float64)
    slopes = np.asarray(((2.0, -1.0), (1.0, 3.0)), dtype=np.float64)
    connectivity = space.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("The square fixture requires polygonal connectivity.")
    endpoints = np.asarray(connectivity.edges)
    coordinates = np.asarray(space.mesh.coordinates)
    edge_vectors = coordinates[endpoints[:, 1]] - coordinates[endpoints[:, 0]]
    tangents = edge_vectors / np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    direction = (
        np.stack((tangents[:, 1], -tangents[:, 0]), axis=1)
        if space.field.element.value_spec.proxy == "flux"
        else tangents
    )
    midpoints = 0.5 * (coordinates[endpoints[:, 0]] + coordinates[endpoints[:, 1]])
    midpoint_values = value + midpoints @ slopes.T
    changes = edge_vectors @ slopes.T
    moments = np.stack(
        (
            np.sum(direction * midpoint_values, axis=1),
            np.sum(direction * changes, axis=1) / 6.0,
        ),
        axis=1,
    )
    cell_means = value + slopes @ np.asarray((0.5, 0.5), dtype=np.float64)
    return jnp.asarray(np.concatenate((moments.ravel(), cell_means))), value


@pytest.mark.parametrize(
    ("factory", "channel", "degree", "twist", "proxy", "variance"),
    (
        pytest.param(
            conforming_h1_virtual_element,
            "h1-projection",
            0,
            "untwisted",
            "scalar",
            "neutral",
            id="h1-energy",
        ),
        pytest.param(
            conforming_h1_virtual_element,
            "l2-projection",
            0,
            "untwisted",
            "scalar",
            "neutral",
            id="h1-l2",
        ),
        pytest.param(
            discontinuous_l2_virtual_element,
            "l2-projection",
            0,
            "untwisted",
            "scalar",
            "neutral",
            id="scalar-l2-not-density",
        ),
        pytest.param(
            conforming_hdiv_virtual_element,
            "l2-projection",
            1,
            "twisted",
            "flux",
            "contravariant",
            id="hdiv-flux",
        ),
        pytest.param(
            conforming_hcurl_virtual_element,
            "l2-projection",
            1,
            "untwisted",
            "circulation",
            "covariant",
            id="hcurl-circulation",
        ),
    ),
)
def test_projected_values_reproduce_declared_scalar_and_compatible_forms(
    factory: Callable[[int], VirtualElementSpec],
    channel: VirtualElementReconstructionChannel,
    degree: int,
    twist: FormTwist,
    proxy: FormProxy,
    variance: PortVariance,
) -> None:
    space = _square_space(factory)
    reconstruction = prepare_virtual_element_field_reconstruction(space, channel=channel)
    expected = FormValueSpec(FormType(2, degree, twist=twist), proxy=proxy)
    declared = reconstruction.value_port.form
    assert declared is not None
    assert declared.value_spec_id == expected.value_spec_id
    field_form = space.field_space.form_type
    assert field_form is not None
    assert field_form.form_type_id == expected.form_type.form_type_id
    assert space.dof_map.value_spec.value_spec_id == expected.value_spec_id
    assert reconstruction.value_port.variance == variance
    state, value = _polynomial_dofs(space)
    points = np.asarray(((0.21, 0.37), (0.73, 0.62)), dtype=np.float64)
    query = reconstruction.prepare_query(points)
    if expected.value_shape:
        slopes = np.asarray(((2.0, -1.0), (1.0, 3.0)), dtype=np.float64)
        reference = value + points @ slopes.T
        reference_derivative = np.broadcast_to(slopes[:, 0], (2, 2))
    else:
        reference = np.full((2,), value.item(), dtype=np.float64)
        reference_derivative = np.zeros((2,), dtype=np.float64)
    np.testing.assert_allclose(query.apply(state), reference, atol=1e-11)
    derivative = reconstruction.prepare_query(points, derivative=(1, 0))
    np.testing.assert_allclose(derivative.apply(state), reference_derivative, atol=1e-11)


@pytest.mark.parametrize(
    ("family", "value_spec"),
    (
        pytest.param(
            "ConformingHdiv",
            FormValueSpec(FormType(2, 1), proxy="flux"),
            id="flux-needs-twist",
        ),
        pytest.param(
            "ConformingHcurl",
            FormValueSpec(FormType(2, 1, twist="twisted"), proxy="circulation"),
            id="circulation-is-untwisted",
        ),
        pytest.param(
            "ConformingHdiv",
            FormValueSpec(FormType(2, 1, twist="twisted"), proxy="circulation"),
            id="same-shape-wrong-proxy",
        ),
        pytest.param(
            "DiscontinuousL2",
            FormValueSpec(FormType(2, 2, twist="twisted"), proxy="density"),
            id="scalar-l2-is-not-density",
        ),
        pytest.param(
            "ConformingH1",
            FormValueSpec(FormType(3, 0), proxy="scalar"),
            id="planar-spec-not-three-dimensional",
        ),
        pytest.param(
            "ConformingH1",
            FormValueSpec(FormType(2, 0, fiber_shape=(2,)), proxy="scalar"),
            id="no-replicated-scalars",
        ),
    ),
)
def test_element_refuses_incompatible_form_semantics(
    family: str, value_spec: FormValueSpec
) -> None:
    with pytest.raises(ValueError, match="qualified planar"):
        VirtualElementSpec(family, 1, value_spec=value_spec)


@pytest.mark.parametrize(
    "declared", (False, True), ids=("undeclared-form", "wrong-twist")
)
def test_reconstruction_refuses_custom_ports_without_matching_form(
    declared: bool,
) -> None:
    space = _square_space(conforming_hdiv_virtual_element)
    port = ValuePort(
        "velocity",
        event_shape=(2,),
        component_ids=("vx", "vy"),
        representation="cartesian-vector",
        form=FormValueSpec(FormType(2, 1), proxy="flux") if declared else None,
    )
    with pytest.raises(ValueError, match="form"):
        prepare_virtual_element_field_reconstruction(
            space, channel="l2-projection", value_port=port
        )


def test_polyhedral_scalar_form_reproduces_affine_energy_and_limits_degree() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = tuple(
        np.asarray(face, dtype=np.int32)
        for face in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    )
    mesh = CellMesh.from_polyhedra(points, (faces,))
    prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
    assert prepared.value_spec.form_type.form_type_id == FormType(3, 0).form_type_id
    assert prepared.value_spec.proxy == "scalar"
    slope = np.asarray((2.0, -3.0, 0.5), dtype=np.float64)
    affine = jnp.asarray(1.0 + points @ slope)
    # Integral |grad u|^2 over the reference tetrahedron (volume 1/6).
    np.testing.assert_allclose(
        affine @ prepared.mv(affine), slope @ slope / 6.0, atol=1e-11
    )
    with pytest.raises(ValueError, match="degree one"):
        prepare_polyhedral_h1_virtual_element_3d(mesh, degree=2)


def test_obsolete_conformity_selector_is_refused() -> None:
    spec = conforming_hdiv_virtual_element(1)
    with pytest.raises(TypeError):
        VirtualElementSpec(
            "ConformingHdiv",
            1,
            value_spec=spec.value_spec,
            conformity="Hdiv",  # ty: ignore[unknown-argument]
        )
