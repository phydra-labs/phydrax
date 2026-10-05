#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cell-centered finite-volume diffusion published as a coupled trace component.

The coupled scenario joins P1 triangles on ``[0, 1] x [0, 1]`` and a structured
finite-volume grid on ``[1, 2] x [0, 1]`` through an imperfect-contact
conductance ``h`` at ``x = 1``. The closed-form reference carries the same
conormal flux on both sides and the jump ``u_minus - u_plus = Q / h`` the
contact law implies; its sources are exact automatic second derivatives of the
closed form, independent of either discretization.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from tests.unit.solver.coupling._cases import (
    build_region,
    dense_policy,
    interface_binding,
    ManufacturedField,
    nodal_error,
    plate_cover,
    RegionSpec,
)


cpl = phx.solver.coupling
dsc = phx.discretization

type PointField = Callable[[Array], Array]

CONDUCTANCE = 2.0


# --- Closed-form contact field ----------------------------------------------------------


def _base(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return 1.5 + 0.25 * jnp.exp(x) * jnp.sin(y) + 0.1 * x**2 * y


def _jump(heights: Array) -> Array:
    return 0.2 + 0.1 * jnp.cos(jnp.pi * heights)


def _minus(points: Array) -> Array:
    """``g + (x - 1) (-h Delta(y) - dg/dx(1, y))``: outward flux ``-h Delta`` at ``x = 1``."""
    x, y = points[..., 0], points[..., 1]
    slope = 0.25 * jnp.e * jnp.sin(y) + 0.2 * y
    return _base(points) + (x - 1.0) * (-CONDUCTANCE * _jump(y) - slope)


def _plus(points: Array) -> Array:
    return _minus(points) - _jump(points[..., 1])


def _negative_laplacian(value: PointField, /) -> PointField:
    hessian = jax.hessian(lambda point: value(point))

    def source(points: Array) -> Array:
        flat = points.reshape((-1, 2))
        laplacian = jax.vmap(lambda point: jnp.trace(hessian(point)))(flat)
        return -laplacian.reshape(points.shape[:-1])

    return source


def _minus_field() -> ManufacturedField:
    # du_minus/dx(1, y) = -h Delta(y); its moment against y (1 - y) by 16-point Gauss.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    heights = 0.5 * (nodes + 1.0)
    flux = -CONDUCTANCE * np.asarray(_jump(jnp.asarray(heights)))
    moment = float(np.sum(0.5 * weights * flux * heights * (1.0 - heights)))
    return ManufacturedField(
        "contact-minus", _minus, _negative_laplacian(_minus), moment, None
    )


def _jump_l2() -> float:
    nodes, weights = np.polynomial.legendre.leggauss(16)
    heights = 0.5 * (nodes + 1.0)
    return float(
        np.sqrt(np.sum(0.5 * weights * np.asarray(_jump(jnp.asarray(heights))) ** 2))
    )


def _host(value: PointField, points: np.ndarray, /) -> np.ndarray:
    return np.asarray(value(jnp.asarray(points, dtype=jnp.float64)))


# --- Finite-volume region -------------------------------------------------------------


def _grid(cells: int, /) -> dsc.PreparedTensorGrid:
    return dsc.TensorGridPlan(
        (dsc.UniformCellAxisSpec(cells), dsc.UniformCellAxisSpec(cells)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[1.0, 0.0], [2.0, 1.0]]))


def _owner(
    cells: int,
    /,
    *,
    lower: dsc.ConservativeBoundaryKind = "neumann",
    lower_target: float = 0.0,
    value: PointField = _plus,
) -> tuple[cpl.FiniteVolumeComponent, dsc.FiniteVolumeDiscretization]:
    """FV owner of ``-Laplace(u) = f`` with Dirichlet data of ``value`` off ``x = 1``."""
    grid = _grid(cells)
    discretization = dsc.FiniteVolumePlan(grid, field_name="u").prepare()
    diffusion = dsc.ConservativeDiffusionPlan(
        grid,
        boundaries={"x": (lower, "dirichlet"), "y": ("dirichlet", "dirichlet")},
    ).prepare(1.0)
    centers = np.asarray(discretization.cell_centers)
    xs, ys = centers[:, 0, 0], centers[0, :, 1]

    def edge(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray:
        first_, second_ = np.broadcast_arrays(first, second)
        return _host(value, np.stack((first_, second_), axis=-1))

    component = cpl.FiniteVolumeComponent(
        "right",
        discretization,
        diffusion,
        source=_host(_negative_laplacian(value), centers),
        boundary_values={
            "x": (lower_target, edge(2.0, ys)),
            "y": (edge(xs, 0.0), edge(xs, 1.0)),
        },
    )
    return component, discretization


def _interface(discretization: dsc.FiniteVolumeDiscretization, /) -> IntegrationDomain:
    """Exterior facets of the grid's lower ``x`` boundary (local facet 0)."""
    exterior = discretization.integration_domain("exterior_facet")
    count = sum(int(np.prod(layout.shape)) for layout in discretization.face_layouts)
    mask = np.zeros((count,), dtype=np.bool_)
    lower = np.asarray(exterior.owner_local_entities) == 0
    mask[np.asarray(exterior.entity_indices)[lower]] = True
    selection = EntitySelection(
        exterior.entity_set_id, mask, active_mask=np.ones((count,), dtype=np.bool_)
    )
    return discretization.integration_domain("exterior_facet", selection)


def _trace(
    component: cpl.FiniteVolumeComponent, discretization: dsc.FiniteVolumeDiscretization
) -> dsc.PreparedTraceAction:
    return component.prepare_side_trace(
        "u",
        _interface(discretization),
        rule=FacetTraceRule("gauss-lobatto-legendre", points=2),
    )


# --- Coupled FE--FV contact -------------------------------------------------------------


def _contact_level(cells: int, /) -> tuple[float, cpl.CoupledSolution]:
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, cells, 1), _minus_field())
    right, discretization = _owner(cells)
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.field_space_id("u")
    )
    law = cpl.ConservativeFluxLaw(
        "contact",
        binding,
        (
            cpl.TransmissionSide("left", "left", "u", left.interface),
            cpl.TransmissionSide("right", "right", "u", _interface(discretization)),
        ),
        cpl.InterfaceConductance(CONDUCTANCE),
    )
    plan = cpl.CoupledProblemPlan(
        "fe-fv-contact",
        components=(left.component, right),
        bindings=(binding,),
        laws=(law,),
    )
    prepared = cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
    assert prepared.execution == "linear"
    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    exact = _host(_plus, np.asarray(discretization.cell_centers))
    right_error = float(np.max(np.abs(np.asarray(solution.field("right", "u")) - exact)))
    left_error = nodal_error(left, solution.field("left", "u"), _minus_field())
    return max(left_error, right_error), solution


def test_fe_fv_conductance_contact_converges_with_the_implied_jump() -> None:
    errors, solution = [], None
    for cells in (4, 8):
        error, solution = _contact_level(cells)
        assert bool(solution.native_successful) and bool(solution.accepted)
        errors.append(error)
    # The cell-average face state sits half a cell from the contact, so the
    # contact flux is first-order consistent: halving the cells at least ~halves
    # the error.
    assert errors[0] / errors[1] > 1.7
    assert solution is not None
    report = solution.interface("contact")
    # The one shared density leaves the triangles exactly as it enters the cells.
    assert float(report.value("flux-conservation")) <= 1.0e-12 * float(
        report.scales[report.names.index("flux-conservation")]
    )
    # The solved traces carry the physically implied jump u_minus - u_plus = Q / h,
    # up to the first-order face-state offset.
    assert float(report.value("trace-jump-l2")) == pytest.approx(_jump_l2(), rel=1.0e-1)


# --- Owner contracts --------------------------------------------------------------------


def test_pointwise_flux_and_stability_are_refused() -> None:
    component, discretization = _owner(4)
    trace = _trace(component, discretization)
    reaction = component.prepare_conormal_flux(trace)
    assert reaction.descriptor.representation == "residual-reaction"
    with pytest.raises(ValueError, match="no exact pointwise"):
        component.prepare_pointwise_flux(trace)
    with pytest.raises(ValueError, match="no exact pointwise"):
        component.certify_flux_stability(reaction)


@pytest.mark.parametrize(
    ("lower", "target", "message"),
    [
        ("dirichlet", 0.0, "native dirichlet condition"),
        ("robin", 0.0, "native robin condition"),
        ("neumann", 0.5, "nonzero native Neumann flux"),
    ],
    ids=["dirichlet", "robin", "inhomogeneous-neumann"],
)
def test_coupled_face_with_a_native_flux_law_is_refused(
    lower: dsc.ConservativeBoundaryKind, target: float, message: str
) -> None:
    component, discretization = _owner(4, lower=lower, lower_target=target)
    trace = _trace(component, discretization)
    # The owner reports the law on those facets, so coupling laws refuse them too.
    assert any(
        imposition.kind != "strong" and imposition.overlaps(trace)
        for imposition in component.boundary_impositions()
    )
    with pytest.raises(ValueError, match=message):
        component.prepare_conormal_flux(trace)


def test_homogeneous_neumann_face_is_free_for_coupling_laws() -> None:
    component, discretization = _owner(4)
    trace = _trace(component, discretization)
    assert not any(
        imposition.overlaps(trace) for imposition in component.boundary_impositions()
    )


def test_mismatched_owners_are_refused() -> None:
    discretization = dsc.FiniteVolumePlan(_grid(4), field_name="u").prepare()
    other = dsc.ConservativeDiffusionPlan(_grid(5)).prepare(1.0)
    with pytest.raises(ValueError, match="different grids"):
        cpl.FiniteVolumeComponent("right", discretization, other)
    grid = _grid(4)
    vector = dsc.FiniteVolumePlan(
        grid, field_name="u", component_names=("first", "second")
    ).prepare()
    with pytest.raises(ValueError, match="one-component"):
        cpl.FiniteVolumeComponent(
            "right", vector, dsc.ConservativeDiffusionPlan(grid).prepare(1.0)
        )


@pytest.mark.parametrize("coefficient", [0.0, -1.0], ids=["zero", "negative"])
def test_capacity_rejects_nonpositive_coefficients(coefficient: float) -> None:
    component, _ = _owner(4)
    with pytest.raises(ValueError, match="finite positive scalar"):
        component.prepare_capacity("u", coefficient=coefficient)


def test_capacity_is_the_cell_volume_diagonal() -> None:
    component, _ = _owner(4)
    operator = component.prepare_capacity("u", coefficient=3.0).operator(None)
    ones = jnp.ones((4, 4), dtype=jnp.float64)
    # Uniform 4 x 4 cells of [1, 2] x [0, 1] have volume 1/16.
    np.testing.assert_allclose(operator.mv(ones), 3.0 / 16.0, rtol=1.0e-15)
    np.testing.assert_allclose(operator.transpose_mv(ones), 3.0 / 16.0, rtol=1.0e-15)


def test_constant_state_balances_interior_cells() -> None:
    component, _ = _owner(5, value=lambda points: jnp.zeros(points.shape[:-1]))
    rows = component.residual((jnp.full((5, 5), 1.75, dtype=jnp.float64),), None)[0]
    np.testing.assert_allclose(np.asarray(rows)[1:-1, 1:-1], 0.0, atol=1.0e-14)
    # The homogeneous Dirichlet faces pull the constant back at the boundary cells.
    assert np.all(np.abs(np.asarray(rows)[-1, :]) > 1.0e-3)
