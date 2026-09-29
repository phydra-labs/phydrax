#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One coupled SEM--VEM--BEM Laplace solve on a square annulus and its substitutions.

Virtual elements discretize the inner ring ``0.5 <= |x|_inf <= 1`` (hole
Dirichlet data), spectral elements (GLL tensor Lagrange quadrilaterals) the
outer ring ``1 <= |x|_inf <= 1.5``, and a Galerkin boundary operator the
unbounded exterior of the outer square. A ``ScalarTransmissionLaw`` (mortar or
matching elimination) joins the rings and a ``BoundaryIntegralTransmissionLaw``
joins the spectral elements to the exterior; everything is ONE coupled solve.

The reference ``u = x / (x^2 + y^2)`` is harmonic, decays, and has zero net
exterior flux; it is evaluated on the host and never from the discrete system.
The heterogeneous case prescribes its forcing and conormal jump independently.
The builders mirror ``examples/sem_vem_bem_transmission.py`` without importing
it. Thresholds cite the refinement campaigns measured on these fixtures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver.coupling import (
    BoundaryIntegralEvidence,
    BoundaryIntegralSide,
    BoundaryIntegralTransmissionLaw,
    CoupledProblemPlan,
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


HOLE, INTERFACE, OUTER = 0.5, 1.0, 1.5
INNER_KAPPA = 4.0

type InnerMethod = Literal["vem", "fe"]
type Imposition = Literal["mortar-side-trace", "mortar-discontinuous", "matching"]
type Coefficient = Literal["unit", "heterogeneous"]
type TraceOwner = (
    phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization
)
type Tree = tuple[tuple[Array, ...], ...]


# --- Independent host references ----------------------------------------------------------


def _exact(points: ArrayLike) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values[..., 0] / np.sum(values**2, axis=-1)


def _exact_gradient(points: ArrayLike) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    x, y = values[..., 0], values[..., 1]
    radius = x * x + y * y
    return np.stack(((y * y - x * x) / radius**2, -2.0 * x * y / radius**2), axis=-1)


def _square_normal(points: np.ndarray, /) -> np.ndarray:
    axis = np.argmax(np.abs(points), axis=-1)[..., None]
    normal = np.zeros_like(points)
    np.put_along_axis(
        normal, axis, np.sign(np.take_along_axis(points, axis, axis=-1)), axis=-1
    )
    return normal


def _outer_kappa(points: Array, args: object) -> Array:
    """``4 + (x^2 - 1)(y^2 - 1)``: continuous with the inner ``kappa = 4`` on ``|x|_inf = 1``."""
    del args
    return INNER_KAPPA + (points[..., 0] ** 2 - 1.0) * (points[..., 1] ** 2 - 1.0)


def _outer_forcing(points: Array, args: object) -> Array:
    """``-div(kappa grad u) = -grad kappa . grad u`` for the harmonic reference."""
    del args
    x, y = points[..., 0], points[..., 1]
    radius = x * x + y * y
    gradient_u = ((y * y - x * x) / radius**2, -2.0 * x * y / radius**2)
    gradient_kappa = (2.0 * x * (y * y - 1.0), 2.0 * y * (x * x - 1.0))
    return -(gradient_kappa[0] * gradient_u[0] + gradient_kappa[1] * gradient_u[1])


def _conormal_jump(points: Array) -> Array:
    """``kappa d_n u - d_n u`` of the heterogeneous reference on the outer square."""
    host = np.asarray(points, dtype=np.float64)
    flux = np.sum(_exact_gradient(host) * _square_normal(host), axis=-1)
    kappa = np.asarray(_outer_kappa(jnp.asarray(host), None))
    return jnp.asarray((kappa - 1.0) * flux)


# --- Geometry authority and bindings ------------------------------------------------------


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


def _annulus_cover() -> phx.domain.SubdomainCover:
    """Analytic concentric square bands; consecutive bands are paired."""
    bands = (
        ("inner-ring", HOLE, INTERFACE),
        ("outer-ring", INTERFACE, OUTER),
        ("exterior", OUTER, 3.0),
    )
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
    return phx.domain.SubdomainCover(window, patches, pairings, cover_id="square-annulus")


def _square_binding(
    cover: phx.domain.SubdomainCover,
    pairing_id: str,
    endpoints: tuple[tuple[str, dict[str, str]], tuple[str, dict[str, str]]],
    /,
) -> InterfaceBinding:
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


# --- Meshes and owners --------------------------------------------------------------------


def _annulus_grid(
    inner: float, outer: float, cells: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Counterclockwise quadrilaterals of ``inner <= |x|_inf <= outer``."""
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


def _perturbed(points: np.ndarray, spacing: float, /) -> np.ndarray:
    """Deterministic jitter of the vertices strictly inside the inner ring."""
    size = np.max(np.abs(points), axis=1)
    interior = (size > HOLE + 1.0e-9) & (size < INTERFACE - 1.0e-9)
    jitter = np.random.default_rng(20260928).uniform(-0.2, 0.2, points.shape) * spacing
    return points + np.where(interior[:, None], jitter, 0.0)


def _on_square(points: np.ndarray, half_width: float, /) -> np.ndarray:
    return np.isclose(np.max(np.abs(points), axis=-1), half_width)


def _facets_on(space: TraceOwner, half_width: float, /) -> IntegrationDomain:
    """The owner's exterior facets on the square ``|x|_inf = half_width``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    facets = np.all(_on_square(np.asarray(probe.sites), half_width), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


@dataclass(frozen=True)
class _Region:
    """One compiled volume owner, its interface facets, and nodal comparison rows."""

    component: VariationalComponent
    space: TraceOwner
    internal: IntegrationDomain
    boundary: IntegrationDomain | None
    points: np.ndarray
    point_rows: np.ndarray


type _Actions = tuple[phx.equations.DiffusionAction | phx.equations.SourceAction, ...]


def _actions(
    kappa: float | Callable[[Array, object], Array],
    forcing: Callable[[Array, object], Array] | None,
    /,
) -> _Actions:
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


def _lagrange_space(
    points: np.ndarray, cells: np.ndarray, kind: str, degree: int, /
) -> phx.discretization.FiniteElementDiscretization:
    return phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh(
            jnp.asarray(points),
            (phx.discretization.CellBlock("cells", kind, jnp.asarray(cells)),),
        ),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(kind, degree)
        ),
    ).prepare()


def _hole_values(points: Array) -> Array:
    return jnp.asarray(_exact(np.asarray(points)))


def _outer_sem(
    cells: int, degree: int, coefficient: Coefficient, *, dirichlet: bool = False
) -> _Region:
    """Spectral elements of the outer ring, coupled on both squares.

    ``dirichlet`` additionally imposes the reference strongly on the outer
    square: an unsupported boundary-data declaration for the boundary law.
    """
    points, quads = _annulus_grid(INTERFACE, OUTER, cells)
    space = _lagrange_space(points, quads, "quadrilateral", degree)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    actions = (
        _actions(1.0, None)
        if coefficient == "unit"
        else _actions(_outer_kappa, _outer_forcing)
    )
    form = phx.equations.FiniteElementForm("laplace", "u", actions)
    if dirichlet:
        mask = np.asarray(space.dof_maps[0].boundary_dof_mask) & _on_square(dofs, OUTER)
        problem = phx.equations.compile_finite_element_problem(
            form,
            space,
            constraint=phx.discretization.dirichlet_constraint(
                space, "u", boundary_mask=mask
            ),
            dirichlet_values=_hole_values,
        )
    else:
        problem = phx.equations.compile_finite_element_problem(form, space)
    return _Region(
        VariationalComponent("outer", problem, field="u"),
        space,
        _facets_on(space, INTERFACE),
        _facets_on(space, OUTER),
        dofs,
        np.arange(dofs.shape[0]),
    )


def _vem_boundary_rows(
    mesh: phx.discretization.CellMesh, rows: int, degree: int, /
) -> np.ndarray:
    """Boundary rows of a conforming VEM space: vertex rows, then edge rows."""
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


def _inner_vem(cells: int, degree: int, coefficient: Coefficient, /) -> _Region:
    """Conforming virtual elements on perturbed quadrilaterals of the inner ring."""
    points, quads = _annulus_grid(HOLE, INTERFACE, cells)
    points = _perturbed(points, (INTERFACE - HOLE) / cells)
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
    hole = _vem_boundary_rows(mesh, dofs.shape[0], degree) & _on_square(dofs, HOLE)
    kappa = 1.0 if coefficient == "unit" else INNER_KAPPA
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm("laplace", "u", _actions(kappa, None)),
        space,
        constraint=phx.discretization.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=hole
        ),
        dirichlet_values=_hole_values,
    )
    # Vertex rows are point values; edge and moment rows are compared nowhere.
    return _Region(
        VariationalComponent("inner", problem, field="u"),
        space,
        _facets_on(space, INTERFACE),
        None,
        dofs,
        np.arange(points.shape[0]),
    )


def _inner_fe(cells: int, degree: int, coefficient: Coefficient, /) -> _Region:
    """Lagrange triangles on the same perturbed inner ring (the VEM substitute)."""
    points, quads = _annulus_grid(HOLE, INTERFACE, cells)
    points = _perturbed(points, (INTERFACE - HOLE) / cells)
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
    space = _lagrange_space(points, triangles, "triangle", degree)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    hole = np.asarray(space.dof_maps[0].boundary_dof_mask) & _on_square(dofs, HOLE)
    kappa = 1.0 if coefficient == "unit" else INNER_KAPPA
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("laplace", "u", _actions(kappa, None)),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=hole
        ),
        dirichlet_values=_hole_values,
    )
    return _Region(
        VariationalComponent("inner", problem, field="u"),
        space,
        _facets_on(space, INTERFACE),
        None,
        dofs,
        np.arange(dofs.shape[0]),
    )


def _exterior(
    panels_per_side: int,
    /,
    *,
    half_width: float = OUTER,
    policy: phx.operators.ScalarLaplaceGalerkinPolicy2D | None = None,
) -> GalerkinBoundaryComponent:
    """Galerkin boundary operator on a square with counterclockwise panels."""
    corners = half_width * np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
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
        policy=phx.operators.ScalarLaplaceGalerkinPolicy2D(regular_order=8)
        if policy is None
        else policy,
    )
    return GalerkinBoundaryComponent("exterior", galerkin)


# --- The flagship declaration -------------------------------------------------------------


@dataclass(frozen=True)
class _Setup:
    """One discretization of the flagship; every refinement axis is independent."""

    outer_cells: int = 1
    outer_degree: int = 4
    inner_method: InnerMethod = "vem"
    inner_cells: int = 2
    inner_degree: int = 2
    panels_per_facet: int = 2
    imposition: Imposition = "mortar-side-trace"
    multiplier_degree: int | None = None
    coefficient: Coefficient = "unit"

    @property
    def facets_per_side(self) -> int:
        return round(2.0 * OUTER * self.outer_cells / (OUTER - INTERFACE))


# Spectral p = 4 on one cell across the outer ring, k = 2 virtual elements on
# two perturbed cells, two boundary panels per spectral facet: 769 unknowns,
# one dense LU.
COARSE = _Setup()

SENSORS = np.asarray(
    [[0.75, 0.1], [-0.7, 0.3], [0.2, 0.8], [-0.3, -0.75], [0.8, -0.6], [-0.9, -0.9]]
)


def _sensor_observation() -> phx.solver.coupling.FieldPointObservation:
    """Point values of the inner ring's potential, whatever owner discretizes it."""
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


SENSOR_OBSERVATION = _sensor_observation()

# A degree-6 discontinuous multiplier on the 16 spectral facets of |x|_inf = 1
# has 16 x 7 = 112 functions but the free traces admit only rank 96.
RANK_DEGREE = 6


def _imposition(setup: _Setup, /) -> MatchingElimination | MortarImposition:
    match setup.imposition:
        case "mortar-side-trace":
            return MortarImposition(MortarMultiplier("side-trace", side="inner"))
        case "mortar-discontinuous":
            return MortarImposition(
                MortarMultiplier(
                    "discontinuous-polynomial",
                    side="outer",
                    degree=setup.multiplier_degree,
                )
            )
        case "matching":
            return MatchingElimination(eliminated="outer")


@dataclass(frozen=True)
class _Flagship:
    setup: _Setup
    inner: _Region
    outer: _Region
    exterior: GalerkinBoundaryComponent
    plan: CoupledProblemPlan
    cover: phx.domain.SubdomainCover


def _declare(
    setup: _Setup,
    /,
    *,
    outer: _Region | None = None,
    exterior: GalerkinBoundaryComponent | None = None,
    projection_order: int | None = None,
    interface_source: Callable[[Array], Array] | None = None,
) -> _Flagship:
    """Components, bindings, laws, and the one coupled plan of a setup."""
    match setup.inner_method:
        case "vem":
            inner = _inner_vem(setup.inner_cells, setup.inner_degree, setup.coefficient)
        case "fe":
            inner = _inner_fe(setup.inner_cells, setup.inner_degree, setup.coefficient)
    outer_ = (
        _outer_sem(setup.outer_cells, setup.outer_degree, setup.coefficient)
        if outer is None
        else outer
    )
    exterior_ = (
        _exterior(setup.facets_per_side * setup.panels_per_facet)
        if exterior is None
        else exterior
    )
    if outer_.boundary is None:
        raise ValueError("The outer ring publishes its outer-square facets.")
    cover = _annulus_cover()
    internal = _square_binding(
        cover,
        "inner-ring|outer-ring",
        (
            ("inner", {"value": inner.component.field_space_id("u")}),
            ("outer", {"value": outer_.component.field_space_id("u")}),
        ),
    )
    boundary = _square_binding(
        cover,
        "outer-ring|exterior",
        (
            ("volume", {"value": outer_.component.field_space_id("u")}),
            ("exterior", {"conormal": exterior_.field_space_id("conormal")}),
        ),
    )
    source = (
        _conormal_jump
        if interface_source is None and setup.coefficient == "heterogeneous"
        else interface_source
    )
    plan = CoupledProblemPlan(
        "sem-vem-bem",
        components=(inner.component, outer_.component, exterior_),
        bindings=(internal, boundary),
        observations=(SENSOR_OBSERVATION,),
        laws=(
            ScalarTransmissionLaw(
                "internal",
                internal,
                (
                    TransmissionSide("inner", "inner", "u", inner.internal),
                    TransmissionSide("outer", "outer", "u", outer_.internal),
                ),
                _imposition(setup),
            ),
            BoundaryIntegralTransmissionLaw(
                "boundary",
                boundary,
                TransmissionSide("volume", "outer", "u", outer_.boundary),
                BoundaryIntegralSide("exterior", "exterior"),
                projection_order=projection_order,
                interface_source=source,
            ),
        ),
    )
    return _Flagship(setup, inner, outer_, exterior_, plan, cover)


def _prepare(flagship: _Flagship, /) -> PreparedCoupledProblem:
    return prepare_coupled_problem(flagship.plan, interface_owners=(flagship.cover,))


def _dense_policy() -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=64_000_000, max_bytes=512 * 1024 * 1024
        ),
    )


@dataclass(frozen=True)
class _Solved:
    flagship: _Flagship
    prepared: PreparedCoupledProblem
    solution: CoupledSolution


def _solve(setup: _Setup, /) -> _Solved:
    flagship = _declare(setup)
    prepared = _prepare(flagship)
    return _Solved(
        flagship, prepared, solve_coupled_problem(prepared, policy=_dense_policy())
    )


# --- Measured errors ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Errors:
    """Max nodal errors, DP0 conormal L2 error, far-field max error, and ``|c|``."""

    outer: float
    inner: float
    conormal: float
    far_field: float
    constant: float


# Errors of the fixtures measured with this module's builders (max nodal
# errors, DP0 L2 conormal error, far-field max error at r = 2.5 and 10, |c|).
# The coarse inner virtual elements are the active volume floor: they set the
# nodal errors of both rings (spectral p = 4 alone would be far smaller), while
# the conormal error is panel-limited (see the boundary-panel campaign). The
# h-campaign (spectral p = 2, virtual k = 1, one panel per facet) measured
# outer 1.05e-2, 2.76e-3, 7.44e-4; inner 1.57e-2, 5.23e-3, 1.50e-3; conormal
# 3.86e-2, 1.17e-2, 3.58e-3; far field 2.65e-3, 4.05e-4, 1.10e-4 for
# outer_cells 1, 2, 4. Tests allow twice the measured value: tight enough that
# a one-level regression of any method fails, loose enough for platform BLAS.
COARSE_ERRORS = _Errors(
    outer=1.34e-2, inner=1.68e-2, conormal=1.36e-2, far_field=4.67e-4, constant=1.73e-4
)
COARSE_SENSOR_ERROR = 1.03e-2
HETEROGENEOUS_ERRORS = _Errors(
    outer=1.35e-2, inner=1.68e-2, conormal=1.33e-2, far_field=6.57e-4, constant=1.79e-4
)
MATCHING_ERRORS = _Errors(
    outer=1.18e-2, inner=9.26e-3, conormal=1.33e-2, far_field=9.28e-5, constant=8.4e-12
)
# P2 triangles on the same perturbed inner ring resolve it better than k = 2
# virtual elements on its quadrilaterals, so the substitution lowers the floor.
FE_ERRORS = _Errors(
    outer=2.05e-3, inner=4.93e-3, conormal=1.31e-2, far_field=1.04e-4, constant=5.2e-6
)
FE_SENSOR_ERROR = 2.33e-3

FAR_TARGETS = np.concatenate(
    [
        radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        for radius, angles in (
            (2.5, np.linspace(0.1, 2.0 * np.pi, 24, endpoint=False)),
            (10.0, np.linspace(0.3, 2.0 * np.pi, 12, endpoint=False)),
        )
    ]
)


def _panel_conormal(galerkin: phx.operators.ScalarLaplaceGalerkin2D, /) -> np.ndarray:
    """Exact panel means of ``d_n u`` along the interior-to-exterior normal."""
    nodes, weights = np.polynomial.legendre.leggauss(12)
    vertices = np.asarray(galerkin.curve.vertices)
    ends = np.asarray(galerkin.curve.panel_vertices)
    start, stop = vertices[ends[:, 0]], vertices[ends[:, 1]]
    points = start[:, None] + 0.5 * (nodes + 1.0)[None, :, None] * (stop - start)[:, None]
    normals = _square_normal(0.5 * (start + stop))
    return 0.5 * np.sum(_exact_gradient(points) * normals[:, None, :], axis=-1) @ weights


def _exterior_field(solved: _Solved, targets: np.ndarray, /) -> np.ndarray:
    solution = solved.solution
    (phi,) = solution.law_state("boundary")
    field = solved.flagship.exterior.galerkin.evaluate_field(
        targets,
        side="exterior",
        dirichlet=phi,
        conormal=solution.field("exterior", "conormal"),
        far_field_constant=solution.field("exterior", "far_field_constant")[0],
    )
    assert bool(field.accepted)
    return np.asarray(field.values)


def _nodal_error(solution: CoupledSolution, name: str, region: _Region, /) -> float:
    values = np.asarray(solution.field(name, "u"))[region.point_rows]
    return float(np.max(np.abs(values - _exact(region.points[region.point_rows]))))


def _errors(solved: _Solved, /) -> _Errors:
    solution = solved.solution
    galerkin = solved.flagship.exterior.galerkin
    conormal = np.asarray(solution.field("exterior", "conormal"))
    lengths = np.asarray(galerkin.curve.lengths)
    return _Errors(
        outer=_nodal_error(solution, "outer", solved.flagship.outer),
        inner=_nodal_error(solution, "inner", solved.flagship.inner),
        conormal=float(
            np.sqrt(np.sum(lengths * (conormal - _panel_conormal(galerkin)) ** 2))
        ),
        far_field=float(
            np.max(np.abs(_exterior_field(solved, FAR_TARGETS) - _exact(FAR_TARGETS)))
        ),
        constant=float(abs(solution.field("exterior", "far_field_constant")[0])),
    )


def _assert_certified(solution: CoupledSolution, /) -> None:
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    for certificate in solution.components:
        assert bool(certificate.accepted), certificate.component
    for report in solution.interfaces:
        values, scales = np.asarray(report.values), np.asarray(report.scales)
        for name, gated, value, scale in zip(
            report.names, report.gated, values, scales, strict=True
        ):
            # Every gated defect is algebraic. DenseLU meets it at <= 3e-13 of
            # its scale on every fixture except the matching variant, whose
            # degree-4 virtual element (component scale 1e6) leaves 6.7e-11 in
            # its flux balance; 1e-9 bounds roundoff, not discretization.
            if gated:
                assert value <= 1.0e-9 * scale, (report.law_id, name, value, scale)


# --- Fixtures -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unit() -> _Solved:
    return _solve(COARSE)


@pytest.fixture(scope="module")
def heterogeneous() -> _Solved:
    return _solve(replace(COARSE, coefficient="heterogeneous"))


# --- One coupled solve --------------------------------------------------------------------


def _members(space: phx.linalg.BlockSpace, /) -> tuple[phx.linalg.BlockSpace, ...]:
    members: list[phx.linalg.BlockSpace] = []
    for member in space.spaces:
        if not isinstance(member, phx.linalg.BlockSpace):
            raise TypeError("Coupled owners are named block spaces.")
        members.append(member)
    return tuple(members)


def _paths(space: phx.linalg.BlockSpace, /) -> tuple[tuple[str, str], ...]:
    """``(owner, block)`` paths of a coupled state or row space, in order."""
    return tuple(
        (owner, block)
        for owner, member in zip(space.names, _members(space), strict=True)
        for block in member.names
    )


def _positions(space: phx.linalg.BlockSpace, /) -> tuple[tuple[int, int], ...]:
    return tuple(
        (owner, block)
        for owner, member in enumerate(_members(space))
        for block in range(len(member.names))
    )


def test_flagship_is_one_accepted_coupled_solve(unit: _Solved) -> None:
    prepared, solution = unit.prepared, unit.solution
    assert set(_paths(prepared.state_space)) == {
        ("inner", "u"),
        ("outer", "u"),
        ("exterior", "conormal"),
        ("exterior", "far_field_constant"),
        ("internal", "multiplier"),
        ("boundary", "dirichlet-trace"),
    }
    assert set(_paths(prepared.row_space)) == {
        ("inner", "u"),
        ("outer", "u"),
        ("exterior", "exterior_boundary_equation"),
        ("exterior", "total_conormal"),
        ("internal", "constraint"),
        ("boundary", "trace-projection"),
    }
    _assert_certified(solution)
    assert {report.law_id for report in solution.interfaces} == {"internal", "boundary"}
    mortar = next(law.evidence for law in prepared.laws if law.law_id == "internal")
    assert isinstance(mortar, MortarEvidence)
    assert mortar.numerical_rank == mortar.multiplier_dimension
    # The side-trace mortar is nonconforming: the (evidence-only) L2 trace
    # mismatch across |x|_inf = 1 is a discretization error, measured at
    # 1.40e-3 of its scale; twice that bounds it.
    internal = solution.interface("internal")
    mismatch = internal.names.index("trace-mismatch-l2")
    assert not internal.gated[mismatch]
    assert float(internal.values[mismatch]) <= 2.8e-3 * float(internal.scales[mismatch])
    boundary = next(law.evidence for law in prepared.laws if law.law_id == "boundary")
    assert isinstance(boundary, BoundaryIntegralEvidence)
    assert boundary.trace_degree == COARSE.outer_degree
    assert boundary.panel_count == 4 * COARSE.facets_per_side * COARSE.panels_per_facet
    assert boundary.projection_exact_degree >= boundary.trace_degree
    assert not boundary.interface_source


def test_hole_dirichlet_data_is_imposed_strongly(unit: _Solved) -> None:
    inner = unit.flagship.inner
    values = np.asarray(unit.solution.field("inner", "u"))
    space = inner.space
    assert isinstance(space, phx.discretization.VirtualElementDiscretization)
    hole = _vem_boundary_rows(
        space.mesh, inner.points.shape[0], COARSE.inner_degree
    ) & _on_square(inner.points, HOLE)
    # Vertices and edge nodes of the hole: 4 sides x 4 edges x (1 + (k - 1)) rows.
    assert np.count_nonzero(hole) == 16 * COARSE.inner_degree
    np.testing.assert_allclose(
        values[hole], _exact(inner.points[hole]), rtol=0.0, atol=1.0e-13
    )


def test_fields_and_sensors_match_the_independent_reference(unit: _Solved) -> None:
    errors = _errors(unit)
    # Measured on COARSE: see COARSE_ERRORS. Thresholds are twice the measured
    # values; one coarser h-level multiplies them by ~3-4 (h-campaign ratios).
    assert errors.outer <= 2.0 * COARSE_ERRORS.outer
    assert errors.inner <= 2.0 * COARSE_ERRORS.inner
    assert errors.conormal <= 2.0 * COARSE_ERRORS.conormal
    predicted = unit.solution.observation("sensors")
    assert bool(np.all(np.asarray(predicted.valid_mask)))
    sensor_error = np.max(np.abs(np.asarray(predicted.values) - _exact(SENSORS)))
    assert sensor_error <= 2.0 * COARSE_SENSOR_ERROR


def test_exterior_relation_compatibility_and_far_field(unit: _Solved) -> None:
    solution = unit.solution
    galerkin = unit.flagship.exterior.galerkin
    (phi,) = solution.law_state("boundary")
    conormal = np.asarray(solution.field("exterior", "conormal"))
    constant = float(solution.field("exterior", "far_field_constant")[0])
    lengths = np.asarray(galerkin.curve.lengths)
    ends = np.asarray(galerkin.curve.panel_vertices)
    # Host DP0 x P1 mass: each panel pairs with its two hats by half its length.
    mixed = np.zeros((ends.shape[0], np.asarray(phi).shape[0]))
    np.add.at(mixed, (np.arange(ends.shape[0]), ends[:, 0]), 0.5 * lengths)
    np.add.at(mixed, (np.arange(ends.shape[0]), ends[:, 1]), 0.5 * lengths)
    terms = (
        0.5 * mixed @ np.asarray(phi),
        -np.asarray(galerkin.double_layer.mv(phi)),
        np.asarray(galerkin.single_layer.mv(jnp.asarray(conormal))),
        -lengths * constant,
    )

    def dual_norm(rows: np.ndarray, /) -> float:
        return float(np.sqrt(np.sum(rows * rows / lengths)))

    # The relation holds to the dense solve's roundoff (measured 4e-15 of the
    # term norms); 1e-11 is far below any discretization effect.
    assert dual_norm(sum(terms)) <= 1.0e-11 * sum(dual_norm(term) for term in terms)
    # Zero net flux of the bounded exterior field, to roundoff.
    assert abs(lengths @ conormal) <= 1.0e-12 * (lengths @ np.abs(conormal))
    assert abs(constant) <= 2.0 * COARSE_ERRORS.constant
    far = _exterior_field(unit, FAR_TARGETS)
    near = np.linalg.norm(FAR_TARGETS, axis=1) < 5.0
    assert np.max(np.abs(far - _exact(FAR_TARGETS))) <= 2.0 * COARSE_ERRORS.far_field
    # Decay: r |u| <= |cos| <= 1 for the reference; the discrete field adds
    # r (|c| + error) at r = 10.
    radii = np.linalg.norm(FAR_TARGETS[~near], axis=1)
    assert np.max(radii * np.abs(far[~near])) <= 1.0 + 20.0 * COARSE_ERRORS.far_field
    assert np.max(np.abs(far[~near])) < 0.5 * np.max(np.abs(far[near]))


def _trace_load(
    region: _Region,
    spacing: float,
    galerkin: phx.operators.ScalarLaplaceGalerkin2D,
    conormal: np.ndarray,
    /,
) -> np.ndarray:
    """``int q gamma v ds`` on every spectral row, from GLL nodal Lagrange traces.

    Each boundary panel lies inside one spectral facet of length ``spacing``;
    the facet's trace basis is the 1-D Lagrange basis on its ``p + 1``
    collinear dof nodes, integrated by an 8-point Gauss rule (exact for p <= 15).
    """
    load = np.zeros(region.points.shape[0])
    nodes, weights = np.polynomial.legendre.leggauss(8)
    vertices = np.asarray(galerkin.curve.vertices)
    for panel, (first, second) in enumerate(np.asarray(galerkin.curve.panel_vertices)):
        start, stop = vertices[first], vertices[second]
        middle = 0.5 * (start + stop)
        normal_axis = int(np.argmax(np.abs(middle)))
        tangent_axis = 1 - normal_axis
        low = -OUTER + spacing * np.floor((middle[tangent_axis] + OUTER) / spacing)
        tangent = region.points[:, tangent_axis]
        rows = np.flatnonzero(
            np.isclose(region.points[:, normal_axis], middle[normal_axis])
            & (tangent >= low - 1.0e-12)
            & (tangent <= low + spacing + 1.0e-12)
        )
        sites = tangent[rows]
        samples = start[tangent_axis] + 0.5 * (nodes + 1.0) * (
            stop[tangent_axis] - start[tangent_axis]
        )
        length = float(np.linalg.norm(stop - start))
        for index, row in enumerate(rows):
            others = np.delete(sites, index)
            basis = np.prod(
                (samples[:, None] - others[None, :]) / (sites[index] - others[None, :]),
                axis=1,
            )
            load[row] += conormal[panel] * 0.5 * length * (weights @ basis)
    return load


def test_flux_balance_matches_the_owner_reaction(unit: _Solved) -> None:
    outer = unit.flagship.outer
    field = unit.solution.field("outer", "u")
    (reaction,) = outer.component.residual((field,), None)
    reaction = np.asarray(reaction)
    conormal = np.asarray(unit.solution.field("exterior", "conormal"))
    injected = _trace_load(
        outer,
        (OUTER - INTERFACE) / COARSE.outer_cells,
        unit.flagship.exterior.galerkin,
        conormal,
    )
    rows = _on_square(outer.points, OUTER)
    assert np.count_nonzero(rows) == 4 * COARSE.facets_per_side * COARSE.outer_degree
    # The spectral rows on the outer square carry exactly the injected work of
    # the exterior conormal (the law's rule integrates the degree-p trace
    # exactly); measured agreement 4e-14 of the largest reaction, i.e. the
    # dense solve's roundoff.
    scale = np.max(np.abs(reaction[rows]))
    assert scale > 0.0
    np.testing.assert_allclose(
        reaction[rows], injected[rows], rtol=0.0, atol=1.0e-10 * scale
    )
    # Partition of unity: the total reaction is the exterior's zero net flux
    # (measured 1.5e-15 of the summed magnitudes).
    assert abs(np.sum(reaction[rows])) <= 1.0e-11 * np.sum(np.abs(reaction[rows]))


def _flat(space: phx.linalg.BlockSpace, tree: Tree, /) -> np.ndarray:
    return np.asarray(space.flatten(tree))


def _only(tree: Tree, owner: int, block: int, /) -> Tree:
    return tuple(
        tuple(
            value if (index, position) == (owner, block) else jnp.zeros_like(value)
            for position, value in enumerate(values)
        )
        for index, values in enumerate(tree)
    )


def test_complete_block_transpose_is_exact(unit: _Solved) -> None:
    prepared = unit.prepared
    operator = prepared.weak_operator(None)
    states, rows = prepared.state_space, prepared.row_space
    keys = jr.split(jr.key(20260928), 2)
    x = states.unflatten(jr.normal(keys[0], (states.size,), dtype=jnp.float64))
    y = rows.unflatten(jr.normal(keys[1], (rows.size,), dtype=jnp.float64))
    state_blocks, state_paths = _positions(states), _paths(states)
    row_blocks, row_paths = _positions(rows), _paths(rows)
    images = {path: operator.mv(_only(x, *path)) for path in state_blocks}
    pullbacks = {path: operator.transpose_mv(_only(y, *path)) for path in row_blocks}
    coupled: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for row, row_path in zip(row_blocks, row_paths, strict=True):
        restricted_y = _flat(rows, _only(y, *row))
        for state, state_path in zip(state_blocks, state_paths, strict=True):
            restricted_x = _flat(states, _only(x, *state))
            image = _flat(rows, _only(images[state], *row))
            pullback = _flat(states, _only(pullbacks[row], *state))
            forward, transposed = restricted_y @ image, pullback @ restricted_x
            scale = np.linalg.norm(restricted_y) * np.linalg.norm(image) + np.linalg.norm(
                pullback
            ) * np.linalg.norm(restricted_x)
            # Each block and its transpose are the same bilinear form: equal up
            # to float64 accumulation over a few hundred rows (measured 3e-17).
            assert abs(forward - transposed) <= 1.0e-12 * scale, (row_path, state_path)
            # A block present in the forward action is present in the transpose.
            assert (np.linalg.norm(image) == 0.0) == (np.linalg.norm(pullback) == 0.0)
            if np.linalg.norm(image) > 0.0:
                coupled.add((row_path, state_path))
    # Owner diagonals plus every law coupling, and nothing else.
    assert coupled == {
        (("inner", "u"), ("inner", "u")),
        (("outer", "u"), ("outer", "u")),
        (("exterior", "exterior_boundary_equation"), ("exterior", "conormal")),
        (("exterior", "exterior_boundary_equation"), ("exterior", "far_field_constant")),
        (("exterior", "total_conormal"), ("exterior", "conormal")),
        # Internal mortar: multiplier loads on both rings, constraint rows on both.
        (("inner", "u"), ("internal", "multiplier")),
        (("outer", "u"), ("internal", "multiplier")),
        (("internal", "constraint"), ("inner", "u")),
        (("internal", "constraint"), ("outer", "u")),
        # Boundary law: -int q gamma v, M phi - L gamma u, and (M/2 - K) phi.
        (("outer", "u"), ("exterior", "conormal")),
        (("boundary", "trace-projection"), ("outer", "u")),
        (("boundary", "trace-projection"), ("boundary", "dirichlet-trace")),
        (("exterior", "exterior_boundary_equation"), ("boundary", "dirichlet-trace")),
    }
    full_x, full_y = _flat(states, x), _flat(rows, y)
    forward = full_y @ _flat(rows, operator.mv(x))
    transposed = _flat(states, operator.transpose_mv(y)) @ full_x
    assert abs(forward - transposed) <= 1.0e-12 * abs(forward)


def _rate(sizes: tuple[float, ...], errors: tuple[float, ...], /) -> float:
    slope, _ = np.polyfit(np.log(np.asarray(sizes)), np.log(np.asarray(errors)), 1)
    return float(slope)


def test_h_refinement_converges_at_second_order() -> None:
    cells = (1, 2, 4)
    campaign = [
        _solve(
            _Setup(
                outer_cells=count,
                outer_degree=2,
                inner_cells=2 * count,
                inner_degree=1,
                panels_per_facet=1,
            )
        )
        for count in cells
    ]
    for solved in campaign:
        _assert_certified(solved.solution)
    errors = [_errors(solved) for solved in campaign]
    sizes = tuple((OUTER - INTERFACE) / count for count in cells)
    # Spectral p = 2 and virtual k = 1 nodal errors converge at O(h^2); the
    # DP0 conormal (panel means) and far field follow. The measured errors
    # (see COARSE_ERRORS) give least-squares rates outer 1.91, inner 1.69,
    # conormal 1.72, far field 2.30; the thresholds sit ~0.2 below them.
    assert _rate(sizes, tuple(error.outer for error in errors)) >= 1.7
    assert _rate(sizes, tuple(error.inner for error in errors)) >= 1.5
    assert _rate(sizes, tuple(error.conormal for error in errors)) >= 1.5
    assert _rate(sizes, tuple(error.far_field for error in errors)) >= 1.7


def test_boundary_panels_are_the_floor_only_when_coarse() -> None:
    coarse, fine = (
        _errors(_solve(replace(COARSE, panels_per_facet=panels))) for panels in (1, 4)
    )
    # Measured (1 -> 4 panels per spectral facet at spectral p = 4): conormal
    # 4.58e-2 -> 5.72e-3, far field 6.28e-4 -> 4.67e-4, volume nodal errors
    # outer 1.30e-2 -> 1.35e-2 and inner 1.66e-2 -> 1.68e-2. The DP0 panel
    # conormal is the only panel-limited quantity; the volume floor belongs to
    # the inner virtual elements, so refining panels alone cannot lower it.
    assert fine.conormal <= 0.25 * coarse.conormal
    assert fine.far_field <= coarse.far_field
    assert abs(fine.outer - coarse.outer) <= 0.1 * coarse.outer
    assert abs(fine.inner - coarse.inner) <= 0.1 * coarse.inner


def test_heterogeneous_coefficient_case(heterogeneous: _Solved) -> None:
    _assert_certified(heterogeneous.solution)
    errors = _errors(heterogeneous)
    assert errors.outer <= 2.0 * HETEROGENEOUS_ERRORS.outer
    assert errors.inner <= 2.0 * HETEROGENEOUS_ERRORS.inner
    assert errors.conormal <= 2.0 * HETEROGENEOUS_ERRORS.conormal
    assert errors.far_field <= 2.0 * HETEROGENEOUS_ERRORS.far_field
    internal = heterogeneous.solution.interface("internal")
    # The kappa-weighted internal flux is nontrivial (kappa = 4 on both sides).
    assert float(internal.scales[internal.names.index("flux-balance")]) > 0.0
    evidence = next(
        law.evidence for law in heterogeneous.prepared.laws if law.law_id == "boundary"
    )
    assert isinstance(evidence, BoundaryIntegralEvidence)
    assert evidence.interface_source


def test_vem_to_fe_substitution_keeps_declarations_and_observation(
    unit: _Solved,
) -> None:
    substituted = _solve(replace(COARSE, inner_method="fe"))
    for solved in (unit, substituted):
        assert solved.flagship.plan.observations == (SENSOR_OBSERVATION,)
    first, second = unit.prepared, substituted.prepared
    assert [c.name for c in first.components] == [c.name for c in second.components]
    assert [law.law_id for law in first.laws] == [law.law_id for law in second.laws]
    assert _paths(first.state_space) == _paths(second.state_space)
    assert _paths(first.row_space) == _paths(second.row_space)
    assert first.observation("sensors").approximation == "h1-projection"
    assert second.observation("sensors").approximation == "exact"
    _assert_certified(substituted.solution)
    predicted = substituted.solution.observation("sensors")
    assert bool(np.all(np.asarray(predicted.valid_mask)))
    errors = _errors(substituted)
    assert errors.inner <= 2.0 * FE_ERRORS.inner
    assert errors.outer <= 2.0 * FE_ERRORS.outer
    sensor_error = np.max(np.abs(np.asarray(predicted.values) - _exact(SENSORS)))
    assert sensor_error <= 2.0 * FE_SENSOR_ERROR


def test_matching_elimination_variant() -> None:
    # One cell of degree 4 on both sides of |x|_inf = 1: coincident facets.
    solved = _solve(replace(COARSE, inner_cells=1, inner_degree=4, imposition="matching"))
    _assert_certified(solved.solution)
    evidence = next(
        law.evidence for law in solved.prepared.laws if law.law_id == "internal"
    )
    assert isinstance(evidence, EliminationEvidence)
    assert evidence.eliminated_role == "outer"
    assert evidence.trace_degrees == (4, 4)
    # The two traces are the same GLL degree-4 polynomials on coincident facets.
    assert evidence.relation_residual <= 1.0e-12
    internal = solved.solution.interface("internal")
    continuity = internal.names.index("trace-continuity-l2")
    assert float(internal.values[continuity]) <= 1.0e-12 * float(
        internal.scales[continuity]
    )
    errors = _errors(solved)
    assert errors.outer <= 2.0 * MATCHING_ERRORS.outer
    assert errors.inner <= 2.0 * MATCHING_ERRORS.inner
    assert errors.far_field <= 2.0 * MATCHING_ERRORS.far_field


def _wrong_shape_source(points: Array) -> Array:
    """One value per panel instead of one per ``(panel, sample)`` point."""
    return jnp.zeros(points.shape[:1])


type _Refusal = Callable[[], object]

REFUSALS: tuple[tuple[str, _Refusal, str], ...] = (
    (
        "mortar-rank-deficient",
        lambda: _prepare(
            _declare(
                replace(
                    COARSE,
                    imposition="mortar-discontinuous",
                    multiplier_degree=RANK_DEGREE,
                )
            )
        ),
        "rank deficient",
    ),
    (
        "panels-coarser-than-spectral-facets",
        lambda: _prepare(_declare(COARSE, exterior=_exterior(3))),
        "span several facets",
    ),
    (
        "curve-off-the-volume-boundary",
        lambda: _prepare(
            _declare(
                COARSE,
                exterior=_exterior(COARSE.facets_per_side, half_width=1.6),
            )
        ),
        "share no segment",
    ),
    (
        "strong-rows-on-the-boundary-trace",
        lambda: _prepare(
            _declare(
                COARSE,
                outer=_outer_sem(
                    COARSE.outer_cells, COARSE.outer_degree, "unit", dirichlet=True
                ),
            )
        ),
        "imposes trace rows strongly",
    ),
    (
        "interface-source-shape",
        lambda: _prepare(_declare(COARSE, interface_source=_wrong_shape_source)),
        "one real value per projection point",
    ),
    (
        "projection-order-below-minimum",
        lambda: _prepare(_declare(COARSE, projection_order=2)),
        "at least 3 points per panel",
    ),
    (
        "galerkin-resident-bytes",
        lambda: _exterior(
            COARSE.facets_per_side,
            policy=phx.operators.ScalarLaplaceGalerkinPolicy2D(
                regular_order=8, max_resident_bytes=4096
            ),
        ),
        r"\[resident-bytes\]",
    ),
)


@pytest.mark.parametrize(
    ("refuse", "match"),
    [pytest.param(refuse, match, id=name) for name, refuse, match in REFUSALS],
)
def test_unsupported_declarations_are_refused_at_preparation(
    refuse: _Refusal, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        refuse()


def test_failed_native_solve_is_visible(unit: _Solved) -> None:
    # Two FGMRES steps cannot reach 1e-12 on the unpreconditioned saddle system.
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=5),
        tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, max_steps=2),
        failure=phx.linalg.FailurePolicy("status"),
    )
    solution = solve_coupled_problem(unit.prepared, policy=policy)
    assert not bool(solution.native_successful)
    assert not bool(solution.accepted)


def test_dense_materialization_excess_is_refused_at_solve(unit: _Solved) -> None:
    # The coupled operator has 769^2 = 591361 entries, beyond a 1000-entry bound.
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(max_entries=1000),
    )
    with pytest.raises(ValueError, match="materialization requires 591361 entries"):
        solve_coupled_problem(unit.prepared, policy=policy)
