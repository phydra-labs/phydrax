#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One Laplace field solved across spectral-element, virtual-element, and boundary-element owners.

The square annulus ``0.5 <= |x|_inf <= 1.5`` surrounds a square hole that
contains the origin. Conforming virtual elements on perturbed polygons
discretize the inner ring ``0.5 <= |x|_inf <= 1``; high-order spectral
elements (tensor Lagrange quadrilaterals on Gauss--Lobatto nodes) discretize
the outer ring ``1 <= |x|_inf <= 1.5``; a 2-D Galerkin boundary operator on the
outer square represents the unbounded exterior. Two explicitly bound
interfaces join the three owners in ONE coupled solve:

- the internal square ``|x|_inf = 1`` carries a ``ScalarTransmissionLaw``
  imposed by a mortar (declared multiplier family, side, and degree, with
  rank, inf-sup, and coverage evidence) or, on coincident facets, by matching
  elimination;
- the outer square carries a ``BoundaryIntegralTransmissionLaw`` (bordered
  Johnson--Nedelec coupling): spectral-element rows receive ``-int q v``, and
  the boundary equation receives ``(M/2 - K) phi`` for the DECLARED L2
  projection ``phi`` of the spectral-element trace onto continuous P1.

The reference ``u = x / (x^2 + y^2)`` is harmonic outside the origin, decays,
and has zero net exterior flux; the hole carries its Dirichlet data and
``kappa = 1``. A separate heterogeneous case uses ``kappa = 4`` on the virtual
elements, ``kappa = 4 + (x^2 - 1)(y^2 - 1)`` on the spectral elements, the unit
exterior, the independently prescribed forcing ``-div(kappa grad u)``, and
the prescribed conormal jump ``kappa d_n u - d_n u`` on the outer square.

Method substitutions change only component preparation: finite-element
triangles or explicit polygons replace the spectral elements, and
finite-element triangles replace the virtual elements, under the same
declarations and the same point observation. The same boundary law also
couples each of those volume owners and a single-patch isogeometric owner on
the full square (``-Laplace u + u = f`` inside, a smooth dipole with a compact
cap as reference); the isogeometric patch cannot mesh the annulus (one
untrimmed tensor patch), so it substitutes on the square only. Without the
reaction the square has the coupled kernel ``u = 1, phi = 1, q = 0, c = 1``,
which preparation detects through the law-owned and boundary unknowns and
refuses until a gauge is declared; a declared decaying exterior gates the
solved far-field constant.

Refinement campaigns (h, spectral p, virtual-element degree, interface
multiplier, boundary panels, quadrature) print tables that identify the
active error floor. The example raises when a solve is not accepted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, iga, IntegrationDomain
from phydrax.solver.coupling import (
    BoundaryIntegralEvidence,
    BoundaryIntegralSide,
    BoundaryIntegralTransmissionLaw,
    CoupledGauge,
    CoupledProblemPlan,
    CoupledResourcePolicy,
    CoupledSolution,
    EliminationEvidence,
    GalerkinBoundaryComponent,
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    MatchingElimination,
    MortarEvidence,
    MortarImposition,
    MortarMultiplier,
    PairedSupportAttachment,
    prepare_coupled_problem,
    PreparedCoupledProblem,
    ScalarTransmissionLaw,
    solve_coupled_problem,
    TransmissionSide,
    VariationalComponent,
)


jax.config.update("jax_enable_x64", True)

HOLE, INTERFACE, OUTER = 0.5, 1.0, 1.5
INNER_KAPPA = 4.0
CAP = 1.0

type InnerMethod = Literal["vem", "fe"]
type OuterMethod = Literal["sem", "fe", "polygon"]
type SquareMethod = Literal["sem", "fe", "polygon", "iga"]
type Imposition = Literal["mortar-side-trace", "mortar-discontinuous", "matching"]
type Coefficient = Literal["unit", "heterogeneous"]
type TraceOwner = (
    phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization
    | phx.discretization.ExplicitPolygonH1Discretization
)


# --- Independent references ----------------------------------------------------------------


def exact(points: ArrayLike) -> np.ndarray:
    """Host reference ``u = x / (x^2 + y^2)``."""
    values = np.asarray(points, dtype=np.float64)
    return values[..., 0] / np.sum(values**2, axis=-1)


def exact_gradient(points: ArrayLike) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    x, y = values[..., 0], values[..., 1]
    radius = x * x + y * y
    return np.stack(((y * y - x * x) / radius**2, -2.0 * x * y / radius**2), axis=-1)


def square_normal(points: np.ndarray, /) -> np.ndarray:
    """Outward unit normal of an origin-centered square at points on its sides."""
    axis = np.argmax(np.abs(points), axis=-1)[..., None]
    normal = np.zeros_like(points)
    np.put_along_axis(
        normal, axis, np.sign(np.take_along_axis(points, axis, axis=-1)), axis=-1
    )
    return normal


def outer_kappa(points: Array, args: object) -> Array:
    """Heterogeneous outer diffusivity ``4 + (x^2 - 1)(y^2 - 1)``; 4 on ``|x|_inf = 1``."""
    del args
    return INNER_KAPPA + (points[..., 0] ** 2 - 1.0) * (points[..., 1] ** 2 - 1.0)


def outer_forcing(points: Array, args: object) -> Array:
    """``-div(kappa grad u) = -grad kappa . grad u`` for the harmonic reference."""
    del args
    x, y = points[..., 0], points[..., 1]
    radius = x * x + y * y
    gradient_u = ((y * y - x * x) / radius**2, -2.0 * x * y / radius**2)
    gradient_kappa = (2.0 * x * (y * y - 1.0), 2.0 * y * (x * x - 1.0))
    return -(gradient_kappa[0] * gradient_u[0] + gradient_kappa[1] * gradient_u[1])


def conormal_jump(points: Array) -> Array:
    """Prescribed ``kappa d_n u - d_n u`` of the heterogeneous case on the outer square."""
    host = np.asarray(points, dtype=np.float64)
    flux = np.sum(exact_gradient(host) * square_normal(host), axis=-1)
    kappa = np.asarray(outer_kappa(jnp.asarray(host), None))
    return jnp.asarray((kappa - 1.0) * flux)


def _cap_profile(radius2: Array, /) -> tuple[Array, Array, Array]:
    """``g, g', g''`` of the dipole profile: ``1/s`` outside ``CAP``, its cubic Taylor cap inside."""
    s0 = CAP * CAP
    inside = radius2 < s0
    shift = radius2 - s0
    taylor = (
        1.0 / s0 - shift / s0**2 + shift**2 / s0**3 - shift**3 / s0**4,
        -1.0 / s0**2 + 2.0 * shift / s0**3 - 3.0 * shift**2 / s0**4,
        2.0 / s0**3 - 6.0 * shift / s0**4,
    )
    safe = jnp.where(inside, s0, radius2)
    outside = (1.0 / safe, -1.0 / safe**2, 2.0 / safe**3)
    value, first, second = (
        jnp.where(inside, cap, tail) for cap, tail in zip(taylor, outside, strict=True)
    )
    return value, first, second


def dipole_value(points: Array) -> Array:
    """``x g(r^2)``: ``x / r^2`` outside the cap, a ``C^3`` cubic cap inside."""
    profile, _, _ = _cap_profile(jnp.sum(points**2, axis=-1))
    return points[..., 0] * profile


def dipole(points: ArrayLike) -> np.ndarray:
    """Host evaluation of the dipole reference."""
    return np.asarray(dipole_value(jnp.asarray(points, dtype=jnp.float64)))


def dipole_source(points: Array, args: object) -> Array:
    """``-Laplace(x g(r^2)) = -x (4 r^2 g'' + 8 g')``; it vanishes outside the cap."""
    del args
    radius2 = jnp.sum(points**2, axis=-1)
    _, first, second = _cap_profile(radius2)
    return -points[..., 0] * (4.0 * radius2 * second + 8.0 * first)


# --- Geometry authority and interface bindings ---------------------------------------------


def _identity(domain: phx.domain.Domain, /) -> dict[str, phx.domain.DomainFunction]:
    return {"x": domain.Function("x")(lambda x: x)}


def _square(half_width: float, /) -> phx.domain.HyperRectangle:
    return phx.domain.HyperRectangle(np.full(2, -half_width), np.full(2, half_width))


def _band(inner: float, outer: float, /) -> Callable[[Array], Array]:
    def support(x: Array) -> Array:
        size = jnp.max(jnp.abs(x))
        return ((size >= inner) & (size <= outer)).astype(jnp.float64)

    return support


def _square_pairing(
    half_width: float, pairing_id: str, left: str, right: str, /
) -> phx.domain.PairedSupport:
    square = _square(half_width)

    def normal(x: Array) -> Array:
        axis = jnp.argmax(jnp.abs(x))
        return jnp.where(jnp.arange(2) == axis, jnp.sign(x), 0.0)

    return phx.domain.PairedSupport(
        square.component({"x": phx.domain.Boundary()}),
        _identity(square),
        _identity(square),
        pairing_id=pairing_id,
        left_patch_id=left,
        right_patch_id=right,
        normal=square.Function("x")(normal),
    )


def nested_square_cover(
    bands: tuple[tuple[str, float, float], ...], cover_id: str, /
) -> phx.domain.SubdomainCover:
    """Analytic authority of concentric square bands; consecutive bands are paired.

    The ambient window is bounded; the last band only witnesses the plus side
    of the outermost square, whereas the unbounded exterior itself belongs to
    the boundary-integral owner.
    """
    window = _square(bands[-1][2])
    patches = tuple(
        phx.domain.SubdomainPatch(
            window,
            window.component(),
            window.Function("x")(_band(inner, outer)),
            _identity(window),
            _identity(window),
            patch_id=patch_id,
        )
        for patch_id, inner, outer in bands
    )
    pairings = tuple(
        _square_pairing(first[2], f"{first[0]}|{second[0]}", first[0], second[0])
        for first, second in zip(bands[:-1], bands[1:], strict=True)
    )
    return phx.domain.SubdomainCover(window, patches, pairings, cover_id=cover_id)


def annulus_cover() -> phx.domain.SubdomainCover:
    return nested_square_cover(
        (
            ("inner-ring", HOLE, INTERFACE),
            ("outer-ring", INTERFACE, OUTER),
            ("exterior", OUTER, 3.0),
        ),
        "square-annulus",
    )


def square_binding(
    cover: phx.domain.SubdomainCover,
    pairing_id: str,
    endpoints: tuple[tuple[str, dict[str, str]], tuple[str, dict[str, str]]],
    /,
) -> InterfaceBinding:
    """Two-sided binding ``(minus, plus)`` of one square: its normal points outward."""
    pairing = cover.pairing(pairing_id)
    witness = pairing.component.sample(phx.domain.PointSampling(16), key=jr.key(3))
    patches = (pairing.left_patch_id, pairing.right_patch_id)
    return InterfaceBinding(
        pairing_id,
        InterfaceSource.paired_support(cover, pairing_id),
        "two-sided",
        tuple(
            InterfaceEndpoint(
                role,
                PairedSupportAttachment(cover, pairing_id, patch, witness),
                fields=fields,
            )
            for (role, fields), patch in zip(endpoints, patches, strict=True)
        ),
    )


# --- Meshes --------------------------------------------------------------------------------


def annulus_grid(
    inner: float, outer: float, cells: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Counterclockwise quadrilaterals of ``inner <= |x|_inf <= outer``.

    ``cells`` quadrilaterals span ``outer - inner`` (``inner = 0`` meshes the
    full square); every square side is a union of whole cell edges.
    """
    spacing = (outer - inner) / cells
    count = round(2.0 * outer / spacing)
    axis = np.linspace(-outer, outer, count + 1)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1
    ).reshape(-1, 4)
    quads = quads[np.max(np.abs(np.mean(grid[quads], axis=1)), axis=1) > inner]
    used, local = np.unique(quads, return_inverse=True)
    return grid[used], local.reshape(quads.shape).astype(np.int32)


def perturbed(
    points: np.ndarray, inner: float, outer: float, spacing: float, /
) -> np.ndarray:
    """Deterministic jitter of the vertices strictly inside the ring."""
    size = np.max(np.abs(points), axis=1)
    interior = (size > inner + 1.0e-9) & (size < outer - 1.0e-9)
    jitter = np.random.default_rng(20260928).uniform(-0.2, 0.2, points.shape) * spacing
    return points + np.where(interior[:, None], jitter, 0.0)


def triangulated(cells: np.ndarray, /) -> np.ndarray:
    """Split each quadrilateral along its first diagonal."""
    return np.concatenate((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]), axis=0)


def on_square(points: np.ndarray, half_width: float, /) -> np.ndarray:
    return np.isclose(np.max(np.abs(points), axis=-1), half_width)


def facets_on(space: TraceOwner, half_width: float, /) -> IntegrationDomain:
    """The owner's exterior facets on the square ``|x|_inf = half_width``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    facets = np.all(on_square(np.asarray(probe.sites), half_width), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


# --- Volume owners -------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One compiled volume owner, its interface facet domains, and sample points.

    ``point_rows`` are the coefficient rows that are point values at
    ``points`` (nodal owners); ``query`` instead evaluates the owner's exact
    field reconstruction at ``points`` (isogeometric control values are not
    point values).
    """

    component: VariationalComponent
    internal: IntegrationDomain | None
    boundary: IntegrationDomain | None
    points: np.ndarray
    point_rows: np.ndarray
    query: Callable[[Array], Array] | None = None

    def point_values(self, coefficients: Array, /) -> np.ndarray:
        """Field values at ``points[point_rows]``."""
        if self.query is None:
            return np.asarray(coefficients)[self.point_rows]
        return np.asarray(self.query(coefficients))


type Actions = tuple[
    phx.equations.DiffusionAction | phx.equations.MassAction | phx.equations.SourceAction,
    ...,
]


def _actions(
    kappa: float | Callable[[Array, object], Array],
    forcing: Callable[[Array, object], Array] | None,
    /,
) -> Actions:
    diffusion = phx.equations.DiffusionAction(
        "u",
        kappa
        if isinstance(kappa, float)
        else phx.equations.coefficient(kappa, coefficient_id="kappa"),
    )
    if forcing is None:
        return (diffusion,)
    return (
        diffusion,
        phx.equations.SourceAction(
            "u", phx.equations.coefficient(forcing, coefficient_id="forcing")
        ),
    )


def _outer_actions(coefficient: Coefficient, /) -> Actions:
    match coefficient:
        case "unit":
            return _actions(1.0, None)
        case "heterogeneous":
            return _actions(outer_kappa, outer_forcing)


def _polygon_space(
    points: np.ndarray, quads: np.ndarray, /
) -> phx.discretization.ExplicitPolygonH1Discretization:
    return phx.discretization.ExplicitPolygonH1Plan(
        phx.discretization.CellMesh.from_polygons(
            jnp.asarray(points), tuple(np.asarray(cell) for cell in quads)
        ),
        phx.discretization.ExplicitPolygonH1FieldSpec("u"),
    ).prepare()


def _lagrange_space(
    points: np.ndarray, quads: np.ndarray, degree: int, tensor: bool, /
) -> phx.discretization.FiniteElementDiscretization:
    """Tensor GLL quadrilaterals (spectral elements) or triangles of one degree."""
    cells, kind = (
        (quads, "quadrilateral") if tensor else (triangulated(quads), "triangle")
    )
    return phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh(
            jnp.asarray(points),
            (phx.discretization.CellBlock("cells", kind, jnp.asarray(cells)),),
        ),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(kind, degree)
        ),
    ).prepare()


def _coupled_owner(
    name: str,
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.ExplicitPolygonH1Discretization,
    actions: Actions,
    internal: IntegrationDomain | None,
    /,
) -> Region:
    """A volume owner coupled on all its boundaries (no Dirichlet rows)."""
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("laplace", "u", actions), space
    )
    match space:
        case phx.discretization.FiniteElementDiscretization():
            dofs = np.asarray(space.dof_maps[0].dof_coordinates)
        case _:
            dofs = np.asarray(space.dof_map.default_dof_points)
    return Region(
        VariationalComponent(name, problem, field="u"),
        internal,
        facets_on(space, OUTER),
        dofs,
        np.arange(dofs.shape[0]),
    )


def outer_region(
    name: str, method: OuterMethod, cells: int, degree: int, coefficient: Coefficient, /
) -> Region:
    """Spectral elements, Lagrange triangles, or explicit polygons on the outer ring."""
    points, quads = annulus_grid(INTERFACE, OUTER, cells)
    match method:
        case "sem":
            space = _lagrange_space(points, quads, degree, True)
        case "fe":
            space = _lagrange_space(points, quads, degree, False)
        case "polygon":
            space = _polygon_space(points, quads)
    return _coupled_owner(
        name, space, _outer_actions(coefficient), facets_on(space, INTERFACE)
    )


def _vem_boundary_rows(
    mesh: phx.discretization.CellMesh, rows: int, degree: int, /
) -> np.ndarray:
    """Boundary rows: vertex rows first, then ``degree - 1`` rows per edge."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError("Virtual elements live on polygon meshes.")
    boundary = np.zeros((rows,), dtype=np.bool_)
    vertices = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    boundary[: vertices.shape[0]] = vertices
    for edge in np.flatnonzero(np.asarray(connectivity.boundary_edges)):
        start = vertices.shape[0] + edge * (degree - 1)
        boundary[start : start + degree - 1] = True
    return boundary


def _hole_values(points: Array) -> Array:
    return jnp.asarray(exact(np.asarray(points)))


def inner_vem(name: str, cells: int, degree: int, coefficient: Coefficient, /) -> Region:
    """Conforming virtual elements on perturbed polygons of the inner ring."""
    points, quads = annulus_grid(HOLE, INTERFACE, cells)
    points = perturbed(points, HOLE, INTERFACE, (INTERFACE - HOLE) / cells)
    mesh = phx.discretization.CellMesh.from_polygons(
        jnp.asarray(points), tuple(np.asarray(cell) for cell in quads)
    )
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(degree)
        ),
    ).prepare()
    dofs = np.asarray(space.dof_map.default_dof_points)
    hole = _vem_boundary_rows(mesh, dofs.shape[0], degree) & on_square(dofs, HOLE)
    kappa = 1.0 if coefficient == "unit" else INNER_KAPPA
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm("laplace", "u", _actions(kappa, None)),
        space,
        constraint=phx.discretization.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=hole
        ),
        dirichlet_values=_hole_values,
    )
    return Region(
        VariationalComponent(name, problem, field="u"),
        facets_on(space, INTERFACE),
        None,
        dofs,
        np.arange(points.shape[0]),
    )


def inner_fe(name: str, cells: int, degree: int, coefficient: Coefficient, /) -> Region:
    """Lagrange triangles on the same perturbed inner ring (the virtual-element substitute)."""
    points, quads = annulus_grid(HOLE, INTERFACE, cells)
    points = perturbed(points, HOLE, INTERFACE, (INTERFACE - HOLE) / cells)
    space = _lagrange_space(points, quads, degree, False)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    hole = np.asarray(space.dof_maps[0].boundary_dof_mask) & on_square(dofs, HOLE)
    kappa = 1.0 if coefficient == "unit" else INNER_KAPPA
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("laplace", "u", _actions(kappa, None)),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=hole
        ),
        dirichlet_values=_hole_values,
    )
    return Region(
        VariationalComponent(name, problem, field="u"),
        facets_on(space, INTERFACE),
        None,
        dofs,
        np.arange(dofs.shape[0]),
    )


def exterior_owner(
    panels_per_side: int,
    regular_order: int,
    /,
    *,
    decay_tolerance: float | None = None,
) -> GalerkinBoundaryComponent:
    """Galerkin boundary operator on the outer square with counterclockwise panels.

    The exterior is bounded (free far-field constant) unless ``decay_tolerance``
    declares it decaying, which gates the solved ``|c|`` at that tolerance.
    """
    corners = np.asarray(
        [[-OUTER, -OUTER], [OUTER, -OUTER], [OUTER, OUTER], [-OUTER, OUTER]]
    )
    fractions = np.arange(panels_per_side)[:, None] / panels_per_side
    vertices = np.concatenate(
        [
            corners[side] + fractions * (corners[(side + 1) % 4] - corners[side])
            for side in range(4)
        ]
    )
    galerkin = phx.operators.prepare_scalar_laplace_galerkin_2d(
        phx.operators.ClosedPolygonalCurve2D(vertices, source_id="outer-square"),
        policy=phx.operators.ScalarLaplaceGalerkinPolicy2D(regular_order=regular_order),
    )
    if decay_tolerance is None:
        return GalerkinBoundaryComponent("exterior", galerkin)
    return GalerkinBoundaryComponent(
        "exterior", galerkin, far_field="decaying", far_field_tolerance=decay_tolerance
    )


def boundary_law(
    binding: InterfaceBinding,
    volume: str,
    domain: IntegrationDomain,
    /,
    *,
    projection_order: int | None = None,
    interface_source: Callable[[Array], Array] | None = None,
) -> BoundaryIntegralTransmissionLaw:
    """The one boundary-integral law every volume method reuses unchanged."""
    return BoundaryIntegralTransmissionLaw(
        "boundary",
        binding,
        TransmissionSide("volume", volume, "u", domain),
        BoundaryIntegralSide("exterior", "exterior"),
        projection_order=projection_order,
        interface_source=interface_source,
    )


def _exterior_binding(
    cover: phx.domain.SubdomainCover,
    pairing_id: str,
    volume: Region,
    exterior: GalerkinBoundaryComponent,
    /,
) -> InterfaceBinding:
    return square_binding(
        cover,
        pairing_id,
        (
            ("volume", {"value": volume.component.field_space_id("u")}),
            ("exterior", {"conormal": exterior.field_space_id("conormal")}),
        ),
    )


# --- Flagship declaration and solve -------------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    """One discretization of the flagship; every refinement axis is independent."""

    outer_method: OuterMethod = "sem"
    outer_cells: int = 2
    outer_degree: int = 4
    inner_method: InnerMethod = "vem"
    inner_cells: int = 3
    inner_degree: int = 2
    panels_per_facet: int = 2
    imposition: Imposition = "mortar-side-trace"
    multiplier_degree: int | None = None
    regular_order: int = 8
    projection_order: int | None = None
    coefficient: Coefficient = "unit"

    @property
    def label(self) -> str:
        return (
            f"{self.outer_method}(cells {self.outer_cells}, p {self.outer_degree}) "
            f"{self.inner_method}(cells {self.inner_cells}, k {self.inner_degree}) "
            f"panels/facet {self.panels_per_facet} {self.imposition}"
            + ("" if self.multiplier_degree is None else f"({self.multiplier_degree})")
            + f" order {self.regular_order}/{self.projection_order or 'min'}"
        )


@dataclass(frozen=True)
class Flagship:
    configuration: Configuration
    inner: Region
    outer: Region
    exterior: GalerkinBoundaryComponent
    prepared: PreparedCoupledProblem


def _internal_imposition(
    configuration: Configuration, /
) -> MatchingElimination | MortarImposition:
    match configuration.imposition:
        case "mortar-side-trace":
            return MortarImposition(MortarMultiplier("side-trace", side="inner"))
        case "mortar-discontinuous":
            return MortarImposition(
                MortarMultiplier(
                    "discontinuous-polynomial",
                    side="outer",
                    degree=configuration.multiplier_degree,
                )
            )
        case "matching":
            return MatchingElimination(eliminated="outer")


def _inner_region(configuration: Configuration, /) -> Region:
    match configuration.inner_method:
        case "vem":
            build = inner_vem
        case "fe":
            build = inner_fe
    return build(
        "inner",
        configuration.inner_cells,
        configuration.inner_degree,
        configuration.coefficient,
    )


def _internal_law(
    configuration: Configuration, binding: InterfaceBinding, inner: Region, outer: Region
) -> ScalarTransmissionLaw:
    if inner.internal is None or outer.internal is None:
        raise ValueError("Both rings must publish their internal-square facets.")
    return ScalarTransmissionLaw(
        "internal",
        binding,
        (
            TransmissionSide("inner", "inner", "u", inner.internal),
            TransmissionSide("outer", "outer", "u", outer.internal),
        ),
        _internal_imposition(configuration),
    )


SENSORS = np.asarray(
    [[0.75, 0.1], [-0.7, 0.3], [0.2, 0.8], [-0.3, -0.75], [0.8, -0.6], [-0.9, -0.9]]
)


def sensor_observation() -> phx.solver.coupling.FieldPointObservation:
    """Point values of the inner ring's potential: the same binding for any inner owner."""
    measurement = phx.measurement
    return phx.solver.coupling.FieldPointObservation(
        "sensors",
        "inner",
        "u",
        quantity=measurement.QuantitySpec(
            "inner-potential", "potential", "potential", phx.units.ONE, "potential"
        ),
        support=measurement.PointSampleSupport(
            np.concatenate((SENSORS, np.zeros((SENSORS.shape[0], 1))), axis=1),
            tuple(f"sensor-{index}" for index in range(SENSORS.shape[0])),
            phx.SpatialCoordinateContract(phx.units.METER),
        ),
        sampling=measurement.SamplingSemantics(measurement.SpatialSamplingKind.POINT),
        field_unit=phx.units.ONE,
    )


def prepare_flagship(configuration: Configuration, /) -> Flagship:
    """Components, bindings, laws, and one prepared coupled problem."""
    inner = _inner_region(configuration)
    outer = outer_region(
        "outer",
        configuration.outer_method,
        configuration.outer_cells,
        configuration.outer_degree,
        configuration.coefficient,
    )
    facets_per_side = round(2.0 * OUTER * configuration.outer_cells / (OUTER - INTERFACE))
    exterior = exterior_owner(
        facets_per_side * configuration.panels_per_facet, configuration.regular_order
    )
    if outer.boundary is None:
        raise ValueError("The outer ring must publish its outer-square facets.")
    cover = annulus_cover()
    internal = square_binding(
        cover,
        "inner-ring|outer-ring",
        (
            ("inner", {"value": inner.component.field_space_id("u")}),
            ("outer", {"value": outer.component.field_space_id("u")}),
        ),
    )
    boundary = _exterior_binding(cover, "outer-ring|exterior", outer, exterior)
    heterogeneous = configuration.coefficient == "heterogeneous"
    plan = CoupledProblemPlan(
        "sem-vem-bem",
        components=(inner.component, outer.component, exterior),
        bindings=(internal, boundary),
        observations=(sensor_observation(),),
        laws=(
            _internal_law(configuration, internal, inner, outer),
            boundary_law(
                boundary,
                "outer",
                outer.boundary,
                projection_order=configuration.projection_order,
                interface_source=conormal_jump if heterogeneous else None,
            ),
        ),
        resources=dense_resources(),
    )
    prepared = prepare_coupled_problem(plan, interface_owners=(cover,))
    return Flagship(configuration, inner, outer, exterior, prepared)


DENSE_BUDGET = phx.linalg.MaterializationPolicy(
    max_entries=64_000_000, max_bytes=512 * 1024 * 1024
)


def dense_resources() -> CoupledResourcePolicy:
    """The dense budget for kernel detection.

    The spectral-element ring publishes its constant kernel; preparation
    completes it through the mortar multipliers, the trace projection, and the
    boundary unknowns, whose dense operator images this budget bounds.
    """
    return CoupledResourcePolicy(materialization=DENSE_BUDGET)


def dense_policy() -> phx.linalg.LinearSolvePolicy:
    """Bounded dense LU of the assembled coupled system (a few thousand unknowns)."""
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(), materialization=DENSE_BUDGET
    )


# --- Measured errors ------------------------------------------------------------------------


@dataclass(frozen=True)
class Errors:
    """Errors against the host reference.

    ``volumes`` are max nodal errors of the owners whose coefficients are point
    values (isogeometric control values are not), then the DP0 conormal L2
    error, the far-field max error, and ``|c|``.
    """

    volumes: tuple[float, ...]
    conormal: float
    far_field: float
    constant: float


FAR_TARGETS = np.concatenate(
    [
        radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        for radius, angles in (
            (2.5, np.linspace(0.1, 2.0 * np.pi, 24, endpoint=False)),
            (10.0, np.linspace(0.3, 2.0 * np.pi, 12, endpoint=False)),
        )
    ]
)


def panel_conormal(
    galerkin: phx.operators.ScalarLaplaceGalerkin2D,
    gradient: Callable[[np.ndarray], np.ndarray],
    /,
) -> np.ndarray:
    """Exact panel means of ``d_n u`` along the interior-to-exterior normal."""
    nodes, weights = np.polynomial.legendre.leggauss(12)
    vertices = np.asarray(galerkin.curve.vertices)
    ends = np.asarray(galerkin.curve.panel_vertices)
    start, stop = vertices[ends[:, 0]], vertices[ends[:, 1]]
    points = start[:, None] + 0.5 * (nodes + 1.0)[None, :, None] * (stop - start)[:, None]
    normals = np.asarray(galerkin.curve.normals)
    return 0.5 * np.sum(gradient(points) * normals[:, None, :], axis=-1) @ weights


def measure(
    solution: CoupledSolution,
    volumes: tuple[tuple[str, Region], ...],
    exterior: GalerkinBoundaryComponent,
    reference: Callable[[ArrayLike], np.ndarray],
    /,
) -> Errors:
    galerkin = exterior.galerkin
    conormal = np.asarray(solution.field("exterior", "conormal"))
    constant = np.asarray(solution.field("exterior", "far_field_constant"))
    (phi,) = solution.law_state("boundary")
    field = galerkin.evaluate_field(
        FAR_TARGETS,
        side="exterior",
        dirichlet=phi,
        conormal=conormal,
        far_field_constant=constant[0],
    )
    if not bool(field.accepted):
        raise RuntimeError("Far-field targets are not on the exterior side.")
    lengths = np.asarray(galerkin.curve.lengths)
    exact_conormal = panel_conormal(galerkin, exact_gradient)
    return Errors(
        volumes=tuple(
            float(
                np.max(
                    np.abs(
                        region.point_values(solution.field(name, "u"))
                        - reference(region.points[region.point_rows])
                    )
                )
            )
            for name, region in volumes
            if region.point_rows.size
        ),
        conormal=float(np.sqrt(np.sum(lengths * (conormal - exact_conormal) ** 2))),
        far_field=float(np.max(np.abs(np.asarray(field.values) - exact(FAR_TARGETS)))),
        constant=float(abs(constant[0])),
    )


def _accepted(label: str, solution: CoupledSolution, /) -> None:
    if not bool(solution.accepted):
        raise RuntimeError(
            f"{label} was not accepted: native {bool(solution.native_successful)}, "
            f"interfaces {[np.asarray(report.values) for report in solution.interfaces]}."
        )


@dataclass(frozen=True)
class Outcome:
    flagship: Flagship
    solution: CoupledSolution
    errors: Errors
    seconds: float


_OUTCOMES: dict[Configuration, Outcome] = {}


def solve(configuration: Configuration, /) -> Outcome:
    """Prepare, solve, certify, and measure one configuration (memoized per run)."""
    if configuration in _OUTCOMES:
        return _OUTCOMES[configuration]
    start = time.perf_counter()
    flagship = prepare_flagship(configuration)
    solution = solve_coupled_problem(flagship.prepared, policy=dense_policy())
    _accepted(configuration.label, solution)
    errors = measure(
        solution,
        (("outer", flagship.outer), ("inner", flagship.inner)),
        flagship.exterior,
        exact,
    )
    outcome = Outcome(flagship, solution, errors, time.perf_counter() - start)
    _OUTCOMES[configuration] = outcome
    return outcome


# --- Every volume owner on the full square --------------------------------------------------


def square_forcing(points: Array, args: object) -> Array:
    """``-Laplace(u) + u`` of the dipole: the reaction removes the constant kernel.

    A pure-Neumann interior next to a bounded exterior with a free far-field
    constant has the coupled kernel ``u = 1, phi = 1, q = 0, c = 1`` (see
    ``neumann_kernel``); the flagship's hole Dirichlet data removes that
    kernel, and here a unit reaction does.
    """
    return dipole_source(points, args) + dipole_value(points)


def _square_actions(reaction: bool, /) -> Actions:
    source = phx.equations.SourceAction(
        "u",
        phx.equations.coefficient(
            square_forcing if reaction else dipole_source, coefficient_id="forcing"
        ),
    )
    diffusion = phx.equations.DiffusionAction("u")
    if not reaction:
        return (diffusion, source)
    return (diffusion, phx.equations.MassAction("u", 1.0), source)


SQUARE_SAMPLES = np.stack(
    np.meshgrid(*(np.linspace(-OUTER, OUTER, 13),) * 2, indexing="ij"), axis=-1
).reshape((-1, 2))


def _isogeometric_owner(cells: int, degree: int, /) -> Region:
    """One affine B-spline patch of the square with ``2 cells`` spans per side.

    Control values are not point values, so the owner is sampled through its
    exact field reconstruction.
    """
    grid = iga.BSplineGrid.open_uniform(degree, 2 * cells, interval=(-OUTER, OUTER))
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    space = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        iga.NURBSGeometryState(
            jnp.stack((xx, yy), axis=-1),
            jnp.ones((grid.coefficient_count, grid.coefficient_count)),
        ),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(degree + 2),
    ).prepare(numeric_version="square-patch")
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("laplace", "u", _square_actions(True)), space
    )
    component = VariationalComponent("square", problem, field="u")
    query = component.prepare_field_reconstruction("u").prepare_query(SQUARE_SAMPLES)
    return Region(
        component,
        None,
        space.exterior_facet_domain,
        SQUARE_SAMPLES,
        np.arange(SQUARE_SAMPLES.shape[0]),
        query.apply,
    )


def square_owner(
    method: SquareMethod, cells: int, degree: int, /, *, reaction: bool = True
) -> Region:
    """Spectral elements, triangles, explicit polygons, or a spline patch on ``[-1.5, 1.5]^2``."""
    points, quads = annulus_grid(0.0, OUTER, cells)
    match method:
        case "sem":
            space = _lagrange_space(points, quads, degree, True)
        case "fe":
            space = _lagrange_space(points, quads, degree, False)
        case "polygon":
            space = _polygon_space(points, quads)
        case "iga":
            return _isogeometric_owner(cells, degree)
    return _coupled_owner("square", space, _square_actions(reaction), None)


def _square_plan(
    method: str,
    volume: Region,
    exterior: GalerkinBoundaryComponent,
    /,
    *,
    gauge: CoupledGauge | None = None,
) -> tuple[CoupledProblemPlan, phx.domain.SubdomainCover]:
    cover = nested_square_cover(
        (("square", 0.0, OUTER), ("exterior", OUTER, 3.0)), "full-square"
    )
    binding = _exterior_binding(cover, "square|exterior", volume, exterior)
    if volume.boundary is None:
        raise ValueError("The square owner must publish its boundary facets.")
    plan = CoupledProblemPlan(
        f"{method}-bem-square",
        components=(volume.component, exterior),
        bindings=(binding,),
        laws=(boundary_law(binding, "square", volume.boundary),),
        gauge=gauge,
        resources=dense_resources(),
    )
    return plan, cover


def square_solution(
    method: SquareMethod,
    cells: int,
    degree: int,
    panels_per_facet: int,
    /,
    *,
    decay_tolerance: float | None = None,
) -> tuple[Region, GalerkinBoundaryComponent, CoupledSolution]:
    """Prepare and solve one square declaration; acceptance is left to the caller."""
    volume = square_owner(method, cells, degree)
    exterior = exterior_owner(
        2 * cells * panels_per_facet, 8, decay_tolerance=decay_tolerance
    )
    plan, cover = _square_plan(method, volume, exterior)
    prepared = prepare_coupled_problem(plan, interface_owners=(cover,))
    return volume, exterior, solve_coupled_problem(prepared, policy=dense_policy())


def solve_square(
    method: SquareMethod, cells: int, degree: int, panels_per_facet: int, /
) -> tuple[Errors, CoupledSolution]:
    """The same boundary law and exterior owner around any volume method."""
    volume, exterior, solution = square_solution(method, cells, degree, panels_per_facet)
    _accepted(f"{method} square", solution)
    return measure(solution, (("square", volume),), exterior, dipole), solution


def _kernel_blocks(
    prepared: PreparedCoupledProblem, basis: Array, /
) -> dict[str, np.ndarray]:
    """One kernel column split into its named ``owner/block`` solve blocks."""
    state = prepared.chart.state_space
    vector = state.unflatten(basis[:, 0])
    values: dict[str, np.ndarray] = {}
    for index, (owner, member) in enumerate(zip(state.names, state.spaces, strict=True)):
        if not isinstance(member, phx.linalg.BlockSpace):
            raise TypeError("Solve owners are block spaces.")
        for position, name in enumerate(member.names):
            values[f"{owner}/{name}"] = np.asarray(vector[index][position])
    return values


def neumann_kernel() -> None:
    """The pure-Neumann spectral-element square: its coupled kernel at preparation.

    The owner publishes only its constant; preparation completes it through the
    law's trace projection and the exterior's unknowns and refuses the plan
    without a gauge. With a gauge the prepared nullspace policy holds the right
    kernel ``u = phi = c``, ``q = 0`` and the different left kernel ``u = c``
    (rows) of the nonsymmetric Johnson--Nedelec coupling.
    """
    volume = square_owner("sem", 2, 4, reaction=False)
    exterior = exterior_owner(8, 8)
    plan, cover = _square_plan("sem-neumann", volume, exterior)
    # The refusal is the demonstrated result; any other failure propagates.
    try:
        prepare_coupled_problem(plan, interface_owners=(cover,))
    except ValueError as refusal:
        print(f"  without a gauge: {refusal}")
    else:
        raise RuntimeError("The pure-Neumann square was expected to be refused.")
    gauged, cover = _square_plan("sem-neumann", volume, exterior, gauge=CoupledGauge())
    prepared = prepare_coupled_problem(gauged, interface_owners=(cover,))
    policy = prepared.nullspace_policy
    if policy is None or policy.right is None or policy.left is None:
        raise RuntimeError("The gauged square lost its kernel pair.")
    for side, basis in (("right", policy.right.basis), ("left", policy.left.basis)):
        ranges = ", ".join(
            f"{name} [{np.min(value):+.3f}, {np.max(value):+.3f}]"
            for name, value in _kernel_blocks(prepared, basis).items()
        )
        print(f"  {side} kernel: {ranges}")


def decaying_gate() -> None:
    """A declared decaying exterior gates the solved far-field constant."""
    for tolerance in (1.0e-2, 1.0e-3):
        _, _, solution = square_solution("fe", 2, 2, 2, decay_tolerance=tolerance)
        report = solution.interface("boundary")
        excess = float(report.values[report.names.index("far-field-decay")])
        constant = abs(
            float(np.asarray(solution.field("exterior", "far_field_constant"))[0])
        )
        print(
            f"  fe square, tolerance {tolerance:.0e}: |c| {constant:.2e} decay excess "
            f"{excess:.2e} accepted {bool(solution.accepted)}"
        )


# --- Reports --------------------------------------------------------------------------------


def report_interfaces(solution: CoupledSolution, /) -> None:
    for interface in solution.interfaces:
        for name, gated, value, scale in zip(
            interface.names,
            interface.gated,
            np.asarray(interface.values),
            np.asarray(interface.scales),
            strict=True,
        ):
            role = "gated" if gated else "evidence"
            print(
                f"  {interface.law_id:<8} {name:<26} {value:.3e} "
                f"(scale {scale:.3e}, {role})"
            )


def report_evidence(flagship: Flagship, /) -> None:
    for law in flagship.prepared.laws:
        evidence = law.evidence
        match evidence:
            case MortarEvidence():
                print(
                    f"  mortar: {evidence.multiplier_family} multiplier on "
                    f"{evidence.multiplier_side!r}, degree {evidence.multiplier_degree}, "
                    f"dimension {evidence.multiplier_dimension}, rank "
                    f"{evidence.numerical_rank}, inf-sup {evidence.inf_sup:.3f}, "
                    f"trace degrees {evidence.trace_degrees}, coverage gap "
                    f"{evidence.coverage.maximum_gap:.1e}"
                )
            case EliminationEvidence():
                print(
                    f"  matching elimination of {evidence.eliminated_role!r}: rows "
                    f"{evidence.eliminated_rows}, relation residual "
                    f"{evidence.relation_residual:.1e}, trace rank {evidence.trace_rank}"
                )
            case BoundaryIntegralEvidence():
                print(
                    f"  boundary: {evidence.formulation}, trace degree "
                    f"{evidence.trace_degree}, projection order "
                    f"{evidence.projection_order} (exact through degree "
                    f"{evidence.projection_exact_degree}), panels "
                    f"{evidence.panel_count}, coverage gap "
                    f"{evidence.coverage.maximum_gap:.1e}, normal defect "
                    f"{evidence.coverage.maximum_normal_defect:.1e}"
                )
            case _:
                print(f"  {law.law_id}: {type(evidence).__name__}")


def report_flagship(configuration: Configuration, /) -> None:
    outcome = solve(configuration)
    solution, errors = outcome.solution, outcome.errors
    print(
        f"{configuration.coefficient} flagship {configuration.label}: unknowns "
        f"{outcome.flagship.prepared.state_space.size}, {outcome.seconds:.1f} s"
    )
    report_evidence(outcome.flagship)
    report_interfaces(solution)
    for certificate in solution.components:
        print(
            f"  component {certificate.component:<9} residual "
            f"{float(certificate.residual_norm):.3e} (scale "
            f"{float(certificate.scale):.3e}) accepted {bool(certificate.accepted)}"
        )
    print(
        f"  errors: outer {errors.volumes[0]:.3e}, inner {errors.volumes[1]:.3e}, "
        f"conormal {errors.conormal:.3e}, far field {errors.far_field:.3e}, |c| "
        f"{errors.constant:.3e}; native {bool(solution.native_successful)}, "
        f"accepted {bool(solution.accepted)}"
    )
    report_sensors(outcome)


def report_sensors(outcome: Outcome, /) -> None:
    """The inner-ring observation binding, whatever owner discretizes the ring."""
    predicted = outcome.solution.observation("sensors")
    label = outcome.flagship.prepared.observation("sensors").approximation
    error = np.max(np.abs(np.asarray(predicted.values) - exact(SENSORS)))
    print(
        f"  sensors ({label} reconstruction of {outcome.flagship.configuration.inner_method}): "
        f"max error {error:.3e}, all valid {bool(np.all(np.asarray(predicted.valid_mask)))}"
    )


def campaign(name: str, configurations: tuple[Configuration, ...], /) -> None:
    print(f"{name}:")
    for configuration in configurations:
        outcome = solve(configuration)
        errors = outcome.errors
        print(
            f"  {configuration.label}: outer {errors.volumes[0]:.3e} inner "
            f"{errors.volumes[1]:.3e} q {errors.conormal:.3e} far "
            f"{errors.far_field:.3e} |c| {errors.constant:.2e} "
            f"({outcome.seconds:.1f} s)"
        )


def square_substitutions() -> None:
    print("one boundary law around every volume owner of the full square:")
    methods: tuple[tuple[SquareMethod, int], ...] = (
        ("sem", 4),
        ("fe", 2),
        ("polygon", 1),
        ("iga", 2),
    )
    for method, degree in methods:
        errors, solution = solve_square(method, 2, degree, 2)
        print(
            f"  {method:<8} degree {degree}: points {errors.volumes[0]:.3e} q "
            f"{errors.conormal:.3e} far {errors.far_field:.3e} |c| "
            f"{errors.constant:.2e} accepted {bool(solution.accepted)}"
        )
    print("pure-Neumann square (coupled kernel through law and boundary unknowns):")
    neumann_kernel()
    print("declared decaying exterior (|c| is the fe square's discretization error):")
    decaying_gate()


BASE = Configuration()

CAMPAIGNS: tuple[tuple[str, tuple[Configuration, ...]], ...] = (
    (
        "h (spectral p = 2, virtual k = 1, one panel per facet)",
        tuple(
            Configuration(
                outer_cells=cells,
                outer_degree=2,
                inner_cells=2 * cells,
                inner_degree=1,
                panels_per_facet=1,
            )
            for cells in (1, 2, 4)
        ),
    ),
    (
        "spectral p at fixed h (boundary panels fixed)",
        tuple(replace(BASE, outer_degree=degree) for degree in (2, 4, 6)),
    ),
    (
        "virtual-element degree",
        tuple(replace(BASE, inner_degree=degree) for degree in (1, 2, 3)),
    ),
    (
        "boundary panels per spectral facet",
        tuple(replace(BASE, panels_per_facet=panels) for panels in (1, 2, 4, 8)),
    ),
    (
        "interface multiplier",
        (
            BASE,
            replace(BASE, imposition="mortar-discontinuous", multiplier_degree=2),
            replace(BASE, imposition="mortar-discontinuous", multiplier_degree=4),
            replace(
                BASE,
                inner_cells=2,
                inner_degree=4,
                imposition="matching",
            ),
        ),
    ),
    (
        "quadrature (panel rule, projection rule)",
        (
            BASE,
            replace(BASE, regular_order=16),
            replace(BASE, projection_order=8),
        ),
    ),
)


if __name__ == "__main__":
    report_flagship(BASE)
    report_flagship(replace(BASE, coefficient="heterogeneous"))
    campaign(
        "substitutions (same declarations, other owners)",
        (
            BASE,
            replace(BASE, outer_method="fe", outer_degree=2),
            replace(BASE, outer_method="polygon", outer_degree=1),
            replace(BASE, inner_method="fe", inner_degree=2),
        ),
    )
    report_sensors(solve(replace(BASE, inner_method="fe", inner_degree=2)))
    square_substitutions()
    for title, configurations in CAMPAIGNS:
        campaign(title, configurations)
