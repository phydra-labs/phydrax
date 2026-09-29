#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-region scalar Poisson scenarios for the spatial coupled-problem tests.

The minus region is ``[0, 1] x [0, h]`` and the plus region ``[1, 2] x [0, h]``;
they meet on the interface ``x = 1``. Every region is a native finite-element
(triangle Lagrange) or virtual-element (conforming H1 on quadrilateral
polygons) owner of ``-Laplace(u) = f`` with strong Dirichlet data from a
manufactured field. Case rows state the expected layout and exactness
independently of the coupled assembler under test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import (
    EntitySelection,
    FacetTraceRule,
    IntegrationDomain,
)
from phydrax.equations import (
    CompiledFiniteElementProblem,
    CompiledVirtualElementProblem,
)


cpl = phx.solver.coupling

type Method = Literal["fe", "vem"]
type ImpositionKind = Literal["matching", "mortar-side-trace", "mortar-discontinuous"]
type DirichletRegion = Literal["exterior", "whole-boundary", "none", "tied-corners"]
type PointField = Callable[[Array], Array]

INTERFACE_X = 1.0


@dataclass(frozen=True, slots=True)
class ManufacturedField:
    """Exact solution of ``-Laplace(u) = f`` with host-evaluated interface data.

    ``flux_moment`` is the analytic ``int_0^1 du/dx(1, y) y (1 - y) dy`` of the
    outward conormal flux of the minus region against a weight that vanishes
    at the interface end points.
    """

    field_id: str
    value: PointField
    source: PointField
    flux_moment: float
    polynomial_degree: int | None

    def host(self, points: np.ndarray, /) -> np.ndarray:
        return np.asarray(self.value(jnp.asarray(points, dtype=jnp.float64)))


def _smooth_value(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return jnp.sin(jnp.pi * x / 3.0) * jnp.exp(y)


def _smooth_source(points: Array) -> Array:
    return (jnp.pi**2 / 9.0 - 1.0) * _smooth_value(points)


def _quadratic_value(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return 1.0 + 0.5 * x - y + x**2 + 0.5 * x * y + 1.5 * y**2


def _quadratic_source(points: Array) -> Array:
    return jnp.full(points.shape[:-1], -5.0, dtype=points.dtype)


def _neumann_value(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return x**2 - x**3 / 3.0 + y**2 - 2.0 * y**3 / 3.0


def _neumann_source(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return 2.0 * x + 4.0 * y - 4.0


# u = sin(pi x / 3) e^y: du/dx(1, y) = (pi / 6) e^y, and
# int_0^1 e^y y (1 - y) dy = 3 - e.
SMOOTH = ManufacturedField(
    "smooth",
    _smooth_value,
    _smooth_source,
    float(np.pi / 6.0 * (3.0 - np.e)),
    None,
)
# du/dx(1, y) = 5/2 + y/2 and int_0^1 (5/2 + y/2) y (1 - y) dy = 5/12 + 1/24.
QUADRATIC = ManufacturedField(
    "quadratic", _quadratic_value, _quadratic_source, 5.0 / 12.0 + 1.0 / 24.0, 2
)
# Homogeneous Neumann data on the whole boundary of [0, 2] x [0, 1]: du/dx =
# x (2 - x) and du/dy = 2 y (1 - y). The linear source integrates to zero over
# the plate and is integrated exactly, so the discrete load is compatible.
# du/dx(1, y) = 1, whose moment against y (1 - y) is 1/6.
NEUMANN = ManufacturedField("neumann", _neumann_value, _neumann_source, 1.0 / 6.0, 3)


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """One native owner of ``-div(kappa grad u) = f`` on ``[x0, x1] x [0, height]``."""

    name: str
    method: Method
    x0: float
    x1: float
    cells: int
    degree: int
    height: float = 1.0
    dirichlet: DirichletRegion = "exterior"
    interface_load: bool = False
    diffusivity: float = 1.0


@dataclass(frozen=True, slots=True)
class Region:
    """Prepared owner, its interface facets, and host facts of its full rows."""

    spec: RegionSpec
    component: cpl.VariationalComponent
    problem: CompiledFiniteElementProblem | CompiledVirtualElementProblem
    interface: IntegrationDomain
    dof_points: np.ndarray
    point_rows: np.ndarray
    free_rows: np.ndarray


def _grid(x0: float, x1: float, nx: int, ny: int, height: float) -> np.ndarray:
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def _quad_vertices(nx: int, ny: int) -> list[tuple[int, int, int, int]]:
    cells: list[tuple[int, int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            cells.append((a, a + 1, a + nx + 2, a + nx + 1))
    return cells


def triangle_mesh(
    x0: float, x1: float, nx: int, ny: int, height: float = 1.0
) -> phx.discretization.CellMesh:
    """Structured right-triangle mesh of ``[x0, x1] x [0, height]``."""
    triangles = [
        triangle
        for a, b, c, d in _quad_vertices(nx, ny)
        for triangle in ((a, b, c), (a, c, d))
    ]
    return phx.discretization.CellMesh(
        _grid(x0, x1, nx, ny, height),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )


def quad_polygon_mesh(
    x0: float, x1: float, nx: int, ny: int, height: float = 1.0
) -> phx.discretization.CellMesh:
    """Structured quadrilateral polygon mesh of ``[x0, x1] x [0, height]``."""
    cells = tuple(np.asarray(cell, dtype=np.int32) for cell in _quad_vertices(nx, ny))
    return phx.discretization.CellMesh.from_polygons(
        jnp.asarray(_grid(x0, x1, nx, ny, height)), cells
    )


def _dirichlet_mask(
    boundary: np.ndarray, points: np.ndarray, spec: RegionSpec
) -> np.ndarray:
    interior = (
        np.isclose(points[:, 0], INTERFACE_X)
        & (points[:, 1] > 1.0e-12)
        & (points[:, 1] < spec.height - 1.0e-12)
    )
    match spec.dirichlet:
        case "exterior":
            return boundary & ~interior
        case "whole-boundary" | "none" | "tied-corners":
            return boundary


type FiniteElementChart = (
    phx.discretization.FiniteElementDirichletConstraint | phx.linalg.ConstraintMap | None
)


def _fe_chart(
    space: phx.discretization.FiniteElementDiscretization,
    points: np.ndarray,
    spec: RegionSpec,
) -> tuple[FiniteElementChart, np.ndarray]:
    """Owner chart and the full rows it solves for (all rows unless Dirichlet).

    ``"tied-corners"`` identifies the two far corners (a multipoint chart
    that is not a row selection) and imposes no strong data.
    """
    rows = np.arange(points.shape[0])
    boundary = np.asarray(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    match spec.dirichlet:
        case "none":
            return None, rows
        case "tied-corners":
            far = spec.x0 if np.isclose(spec.x1, INTERFACE_X) else spec.x1
            corners = np.flatnonzero(
                np.isclose(points[:, 0], far)
                & (np.isclose(points[:, 1], 0.0) | np.isclose(points[:, 1], spec.height))
            )
            kept = np.setdiff1d(rows, corners[1:])
            prolongation = np.zeros((rows.size, kept.size), dtype=np.float64)
            prolongation[kept, np.arange(kept.size)] = 1.0
            prolongation[corners[1], np.searchsorted(kept, corners[0])] = 1.0
            chart = phx.discretization.affine_dof_constraint(space, "u", prolongation)
            return chart, rows
        case "exterior" | "whole-boundary":
            fixed = _dirichlet_mask(boundary, points, spec)
            chart = phx.discretization.dirichlet_constraint(
                space, "u", boundary_mask=fixed
            )
            return chart, np.flatnonzero(~fixed)


def _source_coefficient(field: ManufacturedField) -> phx.equations.VariationalCoefficient:
    def source(points: Array, args: object) -> Array:
        del args
        return field.source(points)

    return phx.equations.coefficient(source, coefficient_id=f"f-{field.field_id}")


def _dirichlet_values(field: ManufacturedField) -> Callable[[Array], Array]:
    def values(points: Array) -> Array:
        return field.value(jnp.asarray(points))

    return values


def _interface_load(
    spec: RegionSpec, domain: IntegrationDomain
) -> tuple[phx.equations.BoundaryLoadAction, ...]:
    if not spec.interface_load:
        return ()
    return (
        phx.equations.BoundaryLoadAction(
            "u", 1.0, action_id="interface-load", domain=domain
        ),
    )


def _fe_region(spec: RegionSpec, field: ManufacturedField) -> Region:
    space = phx.discretization.FiniteElementPlan(
        triangle_mesh(spec.x0, spec.x1, spec.cells, spec.cells, spec.height),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", spec.degree)
        ),
    ).prepare()
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    on = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    entities = space.mesh.topology.entity_sets[1]
    mask = np.zeros((entities.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    interface = space.integration_domain(
        "exterior_facet", EntitySelection(entities, mask)
    )
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    constraint, free_rows = _fe_chart(space, points, spec)
    form = phx.equations.FiniteElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", spec.diffusivity),
            phx.equations.SourceAction("u", _source_coefficient(field)),
            *_interface_load(spec, interface),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=None
        if spec.dirichlet in ("none", "tied-corners")
        else _dirichlet_values(field),
    )
    return Region(
        spec,
        cpl.VariationalComponent(spec.name, problem, field="u"),
        problem,
        interface,
        points,
        np.arange(points.shape[0]),
        free_rows,
    )


def _polygon_connectivity(
    mesh: phx.discretization.CellMesh,
) -> phx.discretization.PolygonalConnectivity:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError("Virtual-element regions are polygon meshes.")
    return connectivity


def _vem_boundary_rows(
    connectivity: phx.discretization.PolygonalConnectivity, rows: int, degree: int
) -> np.ndarray:
    """Boundary rows: vertex rows first, then ``degree - 1`` rows per edge."""
    boundary = np.zeros((rows,), dtype=np.bool_)
    vertices = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    boundary[: vertices.shape[0]] = vertices
    per_edge = degree - 1
    for edge in np.flatnonzero(np.asarray(connectivity.boundary_edges)):
        start = vertices.shape[0] + edge * per_edge
        boundary[start : start + per_edge] = True
    return boundary


def _vem_region(spec: RegionSpec, field: ManufacturedField) -> Region:
    if spec.dirichlet not in ("exterior", "whole-boundary"):
        raise ValueError("The virtual-element scenarios always anchor Dirichlet data.")
    mesh = quad_polygon_mesh(spec.x0, spec.x1, spec.cells, spec.cells, spec.height)
    connectivity = _polygon_connectivity(mesh)
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(spec.degree)
        ),
    ).prepare()
    exterior = space.exterior_facet_domain
    facets = np.asarray(exterior.entity_indices)
    edges = np.asarray(connectivity.edges)
    ends = np.asarray(mesh.coordinates)[edges[facets]]
    on = np.all(np.isclose(ends[..., 0], INTERFACE_X), axis=1)
    entities = mesh.topology.entity_sets[1]
    mask = np.zeros((entities.count,), dtype=np.bool_)
    mask[facets[on]] = True
    interface = space.integration_domain(
        "exterior_facet", EntitySelection(entities, mask)
    )
    points = np.asarray(space.dof_map.default_dof_points)
    boundary = _vem_boundary_rows(connectivity, points.shape[0], spec.degree)
    fixed = _dirichlet_mask(boundary, points, spec)
    constraint = phx.discretization.virtual_element_dirichlet_constraint(
        space, "u", boundary_mask=fixed
    )
    form = phx.equations.VirtualElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", spec.diffusivity),
            phx.equations.SourceAction("u", _source_coefficient(field)),
        ),
    )
    problem = phx.equations.compile_virtual_element_problem(
        form, space, constraint=constraint, dirichlet_values=_dirichlet_values(field)
    )
    vertex_and_edge_rows = mesh.coordinates.shape[0] + (spec.degree - 1) * int(
        edges.shape[0]
    )
    return Region(
        spec,
        cpl.VariationalComponent(spec.name, problem, field="u"),
        problem,
        interface,
        points,
        np.arange(vertex_and_edge_rows),
        np.flatnonzero(~fixed),
    )


def build_region(spec: RegionSpec, field: ManufacturedField, /) -> Region:
    """Compile one native owner of the manufactured Poisson problem."""
    match spec.method:
        case "fe":
            return _fe_region(spec, field)
        case "vem":
            return _vem_region(spec, field)


def plate_cover() -> phx.domain.SubdomainCover:
    """Analytic ``[0, 2] x [0, 1]`` cover whose one pairing is the cut ``x = 1``."""
    domain = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    return phx.domain.cartesian_subdomain_cover(domain, "x", (2, 1), cover_id="plate")


def interface_binding(
    cover: phx.domain.SubdomainCover,
    minus_field: str,
    plus_field: str,
    /,
    *,
    interface_id: str = "cut",
    roles: tuple[str, str] = ("left", "right"),
) -> cpl.InterfaceBinding:
    """Two-sided paired-support binding ``(minus, plus)`` of the cut."""
    pairing = cover.pairings[0]
    points = pairing.component.sample(phx.domain.PointSampling(8), key=jax.random.key(1))
    minus = cpl.PairedSupportAttachment(
        cover, pairing.pairing_id, pairing.left_patch_id, points
    )
    plus = cpl.PairedSupportAttachment(
        cover, pairing.pairing_id, pairing.right_patch_id, points
    )
    return cpl.InterfaceBinding(
        interface_id,
        cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        (
            cpl.InterfaceEndpoint(roles[0], minus, fields={"value": minus_field}),
            cpl.InterfaceEndpoint(roles[1], plus, fields={"value": plus_field}),
        ),
    )


def imposition(
    kind: ImpositionKind,
    /,
    *,
    side: str = "right",
    multiplier_degree: int | None = None,
) -> cpl.TransmissionImposition:
    """The declared numerical imposition of the scalar transmission law."""
    match kind:
        case "matching":
            return cpl.MatchingElimination(eliminated=side)
        case "mortar-side-trace":
            return cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side=side))
        case "mortar-discontinuous":
            return cpl.MortarImposition(
                cpl.MortarMultiplier(
                    "discontinuous-polynomial", side=side, degree=multiplier_degree
                )
            )


def transmission_law(
    left: Region,
    right: Region,
    binding: cpl.InterfaceBinding,
    kind: ImpositionKind,
    /,
    *,
    law_id: str = "gamma",
    side: str = "right",
    multiplier_degree: int | None = None,
) -> cpl.ScalarTransmissionLaw:
    """``u_minus = u_plus`` and flux balance across the cut, minus side first."""
    return cpl.ScalarTransmissionLaw(
        law_id,
        binding,
        (
            cpl.TransmissionSide("left", left.spec.name, "u", left.interface),
            cpl.TransmissionSide("right", right.spec.name, "u", right.interface),
        ),
        imposition(kind, side=side, multiplier_degree=multiplier_degree),
    )


@dataclass(frozen=True, slots=True)
class TransmissionCase:
    """Two-region transmission scenario and its expected layout and accuracy.

    ``multiplier_side`` is the plus region for side-trace multipliers and the
    minus region for discontinuous ones; ``expected_exact`` states whether
    the declared discretization reproduces the manufactured field exactly.
    """

    case_id: str
    left_method: Method
    right_method: Method
    left_cells: int
    right_cells: int
    degree: int
    imposition: ImpositionKind
    field: ManufacturedField
    multiplier_degree: int | None = None
    expected_exact: bool = False

    @property
    def multiplier_side(self) -> str:
        return "left" if self.imposition == "mortar-discontinuous" else "right"

    def regions(self) -> tuple[RegionSpec, RegionSpec]:
        return (
            RegionSpec("left", self.left_method, 0.0, 1.0, self.left_cells, self.degree),
            RegionSpec(
                "right", self.right_method, 1.0, 2.0, self.right_cells, self.degree
            ),
        )

    def expected_multiplier_size(self) -> int | None:
        """Law unknowns: free interface trace rows or discontinuous moments."""
        match self.imposition:
            case "matching":
                return None
            case "mortar-side-trace":
                return self.degree * self.right_cells - 1
            case "mortar-discontinuous":
                degree = self.multiplier_degree
                if degree is None:
                    raise ValueError("A discontinuous multiplier declares its degree.")
                return self.left_cells * (degree + 1)

    def expected_eliminated_rows(self) -> int:
        """Interior interface rows of the eliminated plus region."""
        return self.degree * self.right_cells - 1


@dataclass(frozen=True, slots=True)
class Coupled:
    """Declared and prepared two-region problem."""

    left: Region
    right: Region
    cover: phx.domain.SubdomainCover
    binding: cpl.InterfaceBinding
    law: cpl.ScalarTransmissionLaw
    plan: cpl.CoupledProblemPlan
    prepared: cpl.PreparedCoupledProblem


def couple(
    left: Region,
    right: Region,
    kind: ImpositionKind,
    /,
    *,
    side: str = "right",
    multiplier_degree: int | None = None,
    gauge: cpl.CoupledGauge | None = None,
) -> Coupled:
    """Bind, declare, and prepare the transmission problem of two regions."""
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    law = transmission_law(
        left, right, binding, kind, side=side, multiplier_degree=multiplier_degree
    )
    plan = cpl.CoupledProblemPlan(
        "two-regions",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(law,),
        gauge=gauge,
    )
    prepared = cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
    return Coupled(left, right, cover, binding, law, plan, prepared)


def build_case(case: TransmissionCase, /) -> Coupled:
    left_spec, right_spec = case.regions()
    return couple(
        build_region(left_spec, case.field),
        build_region(right_spec, case.field),
        case.imposition,
        side=case.multiplier_side,
        multiplier_degree=case.multiplier_degree,
    )


def dense_policy() -> phx.linalg.LinearSolvePolicy:
    """Explicit bounded dense LU of the small assembled saddle system."""
    return phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())


def nodal_error(region: Region, values: ArrayLike, field: ManufacturedField, /) -> float:
    """Maximum error at the rows whose coefficients are point values."""
    rows = region.point_rows
    exact = field.host(region.dof_points[rows])
    return float(np.max(np.abs(np.asarray(values)[rows] - exact)))


def flux_weight(heights: np.ndarray, /) -> np.ndarray:
    """Interface weight ``y (1 - y)``; it vanishes at the Dirichlet end points."""
    return heights * (1.0 - heights)


def interface_rows(region: Region, /) -> np.ndarray:
    """Free full rows of point-value coefficients on the interior of the interface."""
    points = region.dof_points[region.free_rows]
    on = np.isclose(points[:, 0], INTERFACE_X) & np.isin(
        region.free_rows, region.point_rows
    )
    return region.free_rows[on]


def reaction_moment(region: Region, full: Array, /) -> float:
    """Owner reaction ``sum_i R_i(u) w(y_i)`` over the free interface rows.

    With ``R(u) = A u - b`` and Lagrange/virtual edge traces this is the
    owner's discrete ``int_Gamma (du/dn) w ds`` for the interpolant of ``w``.
    """
    reduced = jnp.asarray(np.asarray(full)[region.free_rows])
    residual = np.asarray(region.problem.residual(reduced, None))
    rows = np.searchsorted(region.free_rows, interface_rows(region))
    heights = region.dof_points[region.free_rows[rows], 1]
    return float(np.dot(residual[rows], flux_weight(heights)))


def multiplier_moment(region: Region, multiplier: Array, /) -> float:
    """``int_Gamma lambda w ds`` of a side-trace multiplier of ``region``.

    The multiplier coefficients belong to the region's free interface rows in
    increasing row order. Each Dirichlet end row's trace function is merged
    equally into the free rows of its end facet (the crosspoint-modified
    mortar space), so the multiplier's end value is the mean of the
    coefficients of the ``degree`` free nodes on that facet. The trace is the
    piecewise Lagrange interpolant of degree ``region.spec.degree``.
    """
    degree = region.spec.degree
    rows = interface_rows(region)
    heights = region.dof_points[rows, 1]
    coefficients = np.asarray(multiplier)[np.argsort(heights)]
    ends = np.asarray([np.mean(coefficients[:degree]), np.mean(coefficients[-degree:])])
    nodes = np.concatenate((np.sort(heights), np.asarray([0.0, region.spec.height])))
    return interface_integral(nodes, np.concatenate((coefficients, ends)), degree)


def interface_integral(nodes: np.ndarray, values: np.ndarray, degree: int, /) -> float:
    """``int v(y) w(y) dy`` of the piecewise Lagrange interpolant ``v`` of degree ``degree``.

    ``nodes`` are the interface coordinates ``y`` of the trace coefficients
    (end points included), so consecutive groups of ``degree + 1`` sorted
    nodes span one facet of a Lagrange or virtual-element edge trace.
    """
    order = np.argsort(nodes)
    heights, samples = nodes[order], values[order]
    gauss, gauss_weights = np.polynomial.legendre.leggauss(degree + 4)
    total = 0.0
    for start in range(0, heights.size - 1, degree):
        local = heights[start : start + degree + 1]
        coefficients = np.polyfit(local, samples[start : start + degree + 1], degree)
        low, high = local[0], local[-1]
        points = 0.5 * (high - low) * (gauss + 1.0) + low
        density = np.polyval(coefficients, points) * flux_weight(points)
        total += 0.5 * (high - low) * float(np.dot(gauss_weights, density))
    return total


def observed_rate(sizes: np.ndarray, errors: np.ndarray, /) -> float:
    """Least-squares slope of ``log(error)`` against ``log(h)``."""
    slope, _ = np.polyfit(np.log(sizes), np.log(errors), 1)
    return float(slope)
