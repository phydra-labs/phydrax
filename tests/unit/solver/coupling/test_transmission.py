#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar and boundary-integral transmission across method substitutions.

One ``ScalarTransmissionLaw`` declaration couples ``-Laplace(u) = f`` on
``[0, 1] x [0, 1]`` and ``[1, 2] x [0, 1]`` across ``x = 1`` for every method
pair (finite elements, virtual elements, explicit polygons); only the declared
imposition changes. One ``BoundaryIntegralTransmissionLaw`` declaration
couples every volume method on ``[-1.5, 1.5]^2`` to the unbounded Laplace
exterior. References are manufactured fields evaluated on the host, their
analytic interface fluxes, and the analytic dipole ``x / |x|^2``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import assert_never, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import (
    EntitySelection,
    FacetTraceRule,
    iga,
    IntegrationDomain,
)
from phydrax.equations import CompiledFiniteElementProblem
from tests.unit.solver.coupling._cases import (
    build_case,
    build_region,
    couple,
    dense_policy,
    imposition,
    ImpositionKind,
    interface_binding,
    INTERFACE_X,
    ManufacturedField,
    Method,
    multiplier_moment,
    nodal_error,
    observed_rate,
    plate_cover,
    quad_polygon_mesh,
    QUADRATIC,
    reaction_moment,
    Region,
    RegionSpec,
    SMOOTH,
    transmission_law,
    TransmissionCase,
)


cpl = phx.solver.coupling

# Roundoff bound for exactly reproduced quadratics: the dense saddle solves
# have O(10^3) unknowns with condition numbers below ~1e5, so 1e-10 leaves
# two orders of margin over eps * cond while any consistency error of the
# interface coupling (>= 1e-4 on these meshes) would be caught.
_EXACT = 1.0e-10

_EXACT_CASES = (
    TransmissionCase(
        "fe-fe-p2-matching",
        "fe",
        "fe",
        3,
        3,
        2,
        "matching",
        QUADRATIC,
        expected_exact=True,
    ),
    TransmissionCase(
        "fe-vem-k2-matching",
        "fe",
        "vem",
        3,
        3,
        2,
        "matching",
        QUADRATIC,
        expected_exact=True,
    ),
    TransmissionCase(
        "fe-fe-p2-side-trace",
        "fe",
        "fe",
        2,
        4,
        2,
        "mortar-side-trace",
        QUADRATIC,
        expected_exact=True,
    ),
    TransmissionCase(
        "fe-vem-k2-side-trace",
        "fe",
        "vem",
        2,
        4,
        2,
        "mortar-side-trace",
        QUADRATIC,
        expected_exact=True,
    ),
    TransmissionCase(
        "fe-fe-p2-discontinuous-p1",
        "fe",
        "fe",
        3,
        4,
        2,
        "mortar-discontinuous",
        QUADRATIC,
        multiplier_degree=1,
        expected_exact=True,
    ),
    TransmissionCase(
        "fe-vem-k2-discontinuous-p1",
        "fe",
        "vem",
        3,
        4,
        2,
        "mortar-discontinuous",
        QUADRATIC,
        multiplier_degree=1,
        expected_exact=True,
    ),
)


def _interface_names(kind: ImpositionKind) -> tuple[str, ...]:
    match kind:
        case "matching":
            return ("trace-continuity-l2", "flux-balance")
        case "mortar-side-trace" | "mortar-discontinuous":
            return ("weak-continuity", "flux-balance", "trace-mismatch-l2")


@pytest.mark.parametrize("case", _EXACT_CASES, ids=lambda case: case.case_id)
def test_quadratic_transmission_is_reproduced_exactly(case: TransmissionCase) -> None:
    """Fields, interface defects, fluxes, and certificates of one exact solve.

    The expensive two-owner preparation is shared by every assertion.
    """
    coupled = build_case(case)
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    assert case.expected_exact
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    for region in (coupled.left, coupled.right):
        values = solution.field(region.spec.name, "u")
        assert nodal_error(region, values, case.field) <= _EXACT, region.spec.name

    report = solution.interface("gamma")
    values = np.asarray(report.values)
    scales = np.asarray(report.scales)
    gated = np.asarray(report.gated)
    assert report.names == _interface_names(case.imposition)
    assert bool(report.accepted(solution.tolerance))
    assert np.all(values[gated] <= _EXACT * scales[gated])
    # An exactly reproduced field is continuous: even the ungated mismatch vanishes.
    assert np.all(values <= _EXACT * np.maximum(scales, 1.0))

    # Owner reactions: the minus side's outward flux and its negative on the plus side.
    minus = reaction_moment(coupled.left, solution.field("left", "u"))
    plus = reaction_moment(coupled.right, solution.field("right", "u"))
    assert minus == pytest.approx(case.field.flux_moment, rel=_EXACT, abs=_EXACT)
    assert plus == pytest.approx(-case.field.flux_moment, rel=_EXACT, abs=_EXACT)

    size = case.expected_multiplier_size()
    law_state = dict(solution.law_states)["gamma"]
    assert tuple(value.shape for value in law_state) == (
        () if size is None else ((size,),)
    )
    if case.imposition == "mortar-side-trace":
        # The weight y (1 - y) lies in the plus side's free P2 trace space, so
        # the multiplier's moment against it is the exact minus-side flux moment.
        measured = multiplier_moment(coupled.right, law_state[0])
        assert measured == pytest.approx(case.field.flux_moment, rel=_EXACT, abs=_EXACT)

    certificates = {item.component: item for item in solution.components}
    assert tuple(certificates) == ("left", "right")
    for region in (coupled.left, coupled.right):
        certificate = certificates[region.spec.name]
        assert certificate.owner_id == region.problem.compilation_id
        assert bool(certificate.accepted)
        assert float(certificate.residual_norm) <= _EXACT * float(certificate.scale)
        assert float(certificate.scale) > 0.0


@dataclass(frozen=True, slots=True)
class RefinementCase:
    """Refinement ladder of a P1 / VEM k = 1 transmission problem.

    Uniform refinement of both regions keeps the mesh ratio fixed, so nodal
    and interface errors of the smooth field decay like ``h^2`` (P1/VEM k=1
    nodal and L2 superconvergence on structured meshes; the side-trace
    multiplier space merges each Dirichlet end row into its facet's free
    rows, so it reproduces the nonzero end flux). Errors must decrease on
    every level; the rate is fitted over the finest pair, and
    ``minimum_rate`` sits below 2 for the pre-asymptotic nodal maximum of the
    coarse minus side (observed 1.6 between h = 1/8 and 1/16).
    """

    case_id: str
    left_method: Method
    right_method: Method
    levels: tuple[tuple[int, int], ...]
    imposition: ImpositionKind
    minimum_rate: float = 1.75


_REFINEMENT_CASES = (
    RefinementCase("fe-fe-matching", "fe", "fe", ((4, 4), (8, 8), (16, 16)), "matching"),
    RefinementCase(
        "fe-vem-matching", "fe", "vem", ((4, 4), (8, 8), (16, 16)), "matching"
    ),
    RefinementCase(
        "fe-fe-side-trace",
        "fe",
        "fe",
        ((4, 6), (8, 12), (16, 24)),
        "mortar-side-trace",
        minimum_rate=1.5,
    ),
    RefinementCase(
        "fe-vem-side-trace",
        "fe",
        "vem",
        ((4, 3), (8, 6), (16, 12)),
        "mortar-side-trace",
        minimum_rate=1.5,
    ),
)


@dataclass(frozen=True, slots=True)
class LevelResult:
    size: float
    errors: tuple[float, float]
    flux_error: float
    multiplier_error: float | None
    mismatch: float | None
    accepted: bool
    native: bool


def _level(case: RefinementCase, left_cells: int, right_cells: int) -> LevelResult:
    transmission = TransmissionCase(
        case.case_id,
        case.left_method,
        case.right_method,
        left_cells,
        right_cells,
        1,
        case.imposition,
        SMOOTH,
    )
    coupled = build_case(transmission)
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    left = solution.field("left", "u")
    errors = (
        nodal_error(coupled.left, left, SMOOTH),
        nodal_error(coupled.right, solution.field("right", "u"), SMOOTH),
    )
    flux = abs(reaction_moment(coupled.left, left) - SMOOTH.flux_moment)
    multiplier: float | None = None
    mismatch: float | None = None
    if case.imposition == "mortar-side-trace":
        (values,) = solution.law_state("gamma")
        multiplier = abs(multiplier_moment(coupled.right, values) - SMOOTH.flux_moment)
        mismatch = float(solution.interface("gamma").value("trace-mismatch-l2"))
    return LevelResult(
        1.0 / left_cells,
        errors,
        flux,
        multiplier,
        mismatch,
        bool(solution.accepted),
        bool(solution.native_successful),
    )


_CONSTANT = 3.7
_CONSTANT_FIELD = ManufacturedField(
    "constant",
    lambda points: jnp.full(points.shape[:-1], _CONSTANT, dtype=points.dtype),
    lambda points: jnp.zeros(points.shape[:-1], dtype=points.dtype),
    0.0,
    0,
)


@pytest.mark.parametrize("kind", ["matching", "mortar-side-trace"])
def test_exact_state_whose_owner_terms_cancel_is_certified(kind: ImpositionKind) -> None:
    """``u = 3.7`` with zero source: the pure-Neumann owner's rows sum to roundoff.

    Every row of the right owner's residual is a sum of stiffness terms of the
    constant that cancel exactly; the certificate must measure that roundoff
    against the magnitude of the summed terms, not against the cancelled sums.
    """
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 2), _CONSTANT_FIELD)
    right = build_region(
        RegionSpec("right", "fe", 1.0, 2.0, 3, 2, dirichlet="none"), _CONSTANT_FIELD
    )
    solution = cpl.solve_coupled_problem(
        couple(left, right, kind).prepared, policy=dense_policy()
    )

    assert bool(solution.native_successful)
    for name in ("left", "right"):
        np.testing.assert_allclose(
            np.asarray(solution.field(name, "u")), _CONSTANT, rtol=0.0, atol=_EXACT
        )
    for certificate in solution.components:
        assert bool(certificate.accepted), certificate.component
        assert float(certificate.residual_norm) <= 1.0e-12 * float(certificate.scale)


@pytest.mark.parametrize("case", _REFINEMENT_CASES, ids=lambda case: case.case_id)
def test_refinement_converges_at_second_order(case: RefinementCase) -> None:
    """Nodal errors, minus-side flux, multiplier flux, and trace mismatch.

    One refinement campaign per method pair; every rate is a least-squares
    fit over all levels.
    """
    levels = tuple(_level(case, left, right) for left, right in case.levels)
    sizes = np.asarray([level.size for level in levels])
    assert all(level.native and level.accepted for level in levels)
    for side in range(2):
        errors = np.asarray([level.errors[side] for level in levels])
        assert np.all(np.diff(errors) < 0.0), errors
        assert observed_rate(sizes[-2:], errors[-2:]) >= case.minimum_rate, errors
    # The coarsest level already resolves the smooth field to a few 1e-3.
    assert max(levels[0].errors) <= 1.0e-2

    flux = np.asarray([level.flux_error for level in levels])
    assert np.all(np.diff(flux) < 0.0), flux
    assert observed_rate(sizes[-2:], flux[-2:]) >= case.minimum_rate, flux
    if case.imposition != "mortar-side-trace":
        return
    multiplier = np.asarray(
        [level.multiplier_error for level in levels], dtype=np.float64
    )
    mismatch = np.asarray([level.mismatch for level in levels], dtype=np.float64)
    assert np.all(np.diff(multiplier) < 0.0), multiplier
    # The multiplier functional superconverges on the middle level and then
    # approaches its asymptotic regime, so its rate is fitted over all levels.
    assert observed_rate(sizes, multiplier) >= case.minimum_rate, multiplier
    assert np.all(np.diff(mismatch) < 0.0), mismatch
    assert observed_rate(sizes[-2:], mismatch[-2:]) >= case.minimum_rate, mismatch


# --- Explicit polygons across a nonmatching mortar ----------------------------------------


def _linear_value(points: Array) -> Array:
    return 0.5 + 0.75 * points[..., 0] - 0.5 * points[..., 1]


def _linear_source(points: Array) -> Array:
    return jnp.zeros(points.shape[:-1], dtype=points.dtype)


# u = 1/2 + 3x/4 - y/2 is harmonic with du/dx = 3/4, and
# int_0^1 (3/4) y (1 - y) dy = 1/8.
LINEAR = ManufacturedField("linear", _linear_value, _linear_source, 0.125, 1)
_LINEAR_FLUX = 0.75


@dataclass(frozen=True, slots=True)
class PolygonRegion:
    """Explicit-polygon (P1) owner of the minus plate ``[0, 1] x [0, 1]``.

    Its coefficients are vertex values; ``free_rows`` are the vertices off the
    strongly imposed exterior facets (every exterior facet except the cut).
    """

    component: phx.solver.coupling.VariationalComponent
    problem: CompiledFiniteElementProblem
    interface: IntegrationDomain
    points: np.ndarray
    free_rows: np.ndarray


def _polygon_region(cells: int, field: ManufacturedField, /) -> PolygonRegion:
    mesh = quad_polygon_mesh(0.0, INTERFACE_X, cells, cells)
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError("Explicit polygon regions are polygon meshes.")
    space = phx.discretization.ExplicitPolygonH1Plan(
        mesh, phx.discretization.ExplicitPolygonH1FieldSpec("u")
    ).prepare()
    facets = np.asarray(space.exterior_facet_domain.entity_indices)
    edges = np.asarray(connectivity.edges)[facets]
    ends = np.asarray(mesh.coordinates)[edges]
    on_cut = np.all(np.isclose(ends[..., 0], INTERFACE_X), axis=1)
    entities = mesh.topology.entity_sets[1]

    def select(chosen: np.ndarray, /) -> IntegrationDomain:
        mask = np.zeros((entities.count,), dtype=np.bool_)
        mask[facets[chosen]] = True
        return space.integration_domain("exterior_facet", EntitySelection(entities, mask))

    def source(points: Array, args: object) -> Array:
        del args
        return field.source(points)

    def dirichlet(points: Array) -> Array:
        return field.value(jnp.asarray(points))

    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "poisson",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        source, coefficient_id=f"f-{field.field_id}"
                    ),
                ),
            ),
        ),
        space,
        constraint=phx.discretization.explicit_polygon_h1_dirichlet_constraint(
            space, "u", domain=select(~on_cut)
        ),
        dirichlet_values=dirichlet,
    )
    points = np.asarray(space.dof_map.default_dof_points)
    return PolygonRegion(
        phx.solver.coupling.VariationalComponent("left", problem, field="u"),
        problem,
        select(on_cut),
        points,
        np.setdiff1d(np.arange(points.shape[0]), np.unique(edges[~on_cut])),
    )


def _couple_polygon(
    left: PolygonRegion,
    right: Region,
    kind: ImpositionKind,
    side: str,
    multiplier_degree: int | None,
    /,
) -> cpl.PreparedCoupledProblem:
    """The plate's one transmission declaration with a polygon minus region."""
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    law = cpl.ScalarTransmissionLaw(
        "gamma",
        binding,
        (
            cpl.TransmissionSide("left", "left", "u", left.interface),
            cpl.TransmissionSide("right", "right", "u", right.interface),
        ),
        imposition(kind, side=side, multiplier_degree=multiplier_degree),
    )
    plan = cpl.CoupledProblemPlan(
        "polygon-plate",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(law,),
    )
    return cpl.prepare_coupled_problem(plan, interface_owners=(cover,))


def _cut_reactions(
    problem: CompiledFiniteElementProblem | phx.equations.CompiledVirtualElementProblem,
    full: ArrayLike,
    free_rows: np.ndarray,
    points: np.ndarray,
    /,
) -> np.ndarray:
    """Owner reactions ``A u - b`` at the free vertex rows strictly inside the cut."""
    residual = np.asarray(
        problem.residual(jnp.asarray(np.asarray(full)[free_rows]), None)
    )
    on_cut = np.isclose(points[free_rows, 0], INTERFACE_X)
    return residual[on_cut]


@dataclass(frozen=True, slots=True)
class PolygonCase:
    """Polygon minus region against a P1 finite or k = 1 virtual plus region."""

    case_id: str
    right_method: Method
    left_cells: int
    right_cells: int
    imposition: ImpositionKind
    side: str
    multiplier_degree: int | None = None


# Three against four cells: the common refinement of the cut has six segments,
# so no facet of either side coincides with a facet of the other. The
# multiplier lives on the coarser polygon side (the stable mortar choice).
_POLYGON_CASES = (
    PolygonCase("polygon-vem-side-trace", "vem", 3, 4, "mortar-side-trace", "left"),
    PolygonCase("polygon-fe-side-trace", "fe", 3, 4, "mortar-side-trace", "left"),
    PolygonCase(
        "polygon-vem-discontinuous-p0",
        "vem",
        3,
        4,
        "mortar-discontinuous",
        "left",
        multiplier_degree=0,
    ),
)


@pytest.mark.parametrize("case", _POLYGON_CASES, ids=lambda case: case.case_id)
def test_polygon_region_reproduces_linear_field_across_nonmatching_mortar(
    case: PolygonCase,
) -> None:
    """P1 polygons, P1 triangles, and k = 1 virtual elements all contain linear
    fields, and a constant flux lies in every declared multiplier space, so the
    nonmatching mortar reproduces the linear field, its interface defects, and
    both owners' interface reactions to roundoff.
    """
    left = _polygon_region(case.left_cells, LINEAR)
    right = build_region(
        RegionSpec("right", case.right_method, 1.0, 2.0, case.right_cells, 1), LINEAR
    )
    prepared = _couple_polygon(
        left, right, case.imposition, case.side, case.multiplier_degree
    )
    (law,) = prepared.laws
    assert isinstance(law.evidence, cpl.MortarEvidence)
    assert law.evidence.trace_degrees == (1, 1)
    assert law.evidence.coverage.segment_count == case.left_cells + case.right_cells - 1

    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    polygon = np.asarray(solution.field("left", "u"))
    assert np.max(np.abs(polygon - LINEAR.host(left.points))) <= _EXACT
    assert nodal_error(right, solution.field("right", "u"), LINEAR) <= _EXACT

    report = solution.interface("gamma")
    values = np.asarray(report.values)
    scales = np.asarray(report.scales)
    assert bool(report.accepted(solution.tolerance))
    assert np.all(values <= _EXACT * np.maximum(scales, 1.0))

    # Linear u: each owner's reaction at an interior cut vertex is the exact
    # int (du/dn) phi_i ds = +-(3/4) h of its own uniform cut facets h.
    minus = _cut_reactions(left.problem, polygon, left.free_rows, left.points)
    plus = _cut_reactions(
        right.problem, solution.field("right", "u"), right.free_rows, right.dof_points
    )
    assert minus.shape == (case.left_cells - 1,)
    assert plus.shape == (case.right_cells - 1,)
    np.testing.assert_allclose(minus, _LINEAR_FLUX / case.left_cells, atol=_EXACT)
    np.testing.assert_allclose(plus, -_LINEAR_FLUX / case.right_cells, atol=_EXACT)


# --- Binding geometry against the law sides -------------------------------------------------


def _binding_plan(
    left: Region, right: Region, cover: phx.domain.SubdomainCover, /
) -> cpl.CoupledProblemPlan:
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    return cpl.CoupledProblemPlan(
        "binding-geometry",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(transmission_law(left, right, binding, "mortar-side-trace"),),
    )


@pytest.mark.parametrize("cells", [(3, 3), (2, 3)], ids=["matching", "crosspoints"])
def test_binding_must_attach_the_law_sides_geometry(cells: tuple[int, int]) -> None:
    """The plate's cut ``x = 1`` binds the two regions; an unrelated analytic
    interface (the cut ``x = 15`` of ``[10, 20] x [5, 6]``) and a binding whose
    minus patch ``x <= 1`` holds the region on ``[1, 2]`` are refused at
    preparation. Nonmatching facets put crosspoints on the ends of the cut.
    """
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, cells[0], 1), SMOOTH)
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, cells[1], 1), SMOOTH)
    cover = plate_cover()
    prepared = cpl.prepare_coupled_problem(
        _binding_plan(left, right, cover), interface_owners=(cover,)
    )
    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    assert bool(solution.accepted)

    far = phx.domain.cartesian_subdomain_cover(
        phx.domain.HyperRectangle(np.asarray([10.0, 5.0]), np.asarray([20.0, 6.0])),
        "x",
        (2, 1),
        cover_id="elsewhere",
    )
    with pytest.raises(ValueError, match="not on paired support"):
        cpl.prepare_coupled_problem(
            _binding_plan(left, right, far), interface_owners=(far,)
        )

    # The minus endpoint is the patch x <= 1; its law side now lies in x >= 1
    # with its outward normal pointing back into that patch.
    across = build_region(RegionSpec("left", "fe", 1.0, 2.0, cells[1], 1), SMOOTH)
    behind = build_region(RegionSpec("right", "fe", 0.0, 1.0, cells[0], 1), SMOOTH)
    with pytest.raises(ValueError, match="'left'.*not on paired support"):
        cpl.prepare_coupled_problem(
            _binding_plan(across, behind, cover), interface_owners=(cover,)
        )


# --- One boundary-integral law around every volume method ---------------------------------

_HALF = 1.5
_CAP = 1.0

type SquareMethod = Literal["sem", "fe", "polygon", "iga"]


def _cap_profile(radius2: Array, /) -> tuple[Array, Array, Array]:
    """``g, g', g''`` of ``1 / s`` outside ``_CAP^2`` and its cubic Taylor cap inside."""
    s0 = _CAP * _CAP
    inside = radius2 < s0
    shift = radius2 - s0
    taylor = (
        1.0 / s0 - shift / s0**2 + shift**2 / s0**3 - shift**3 / s0**4,
        -1.0 / s0**2 + 2.0 * shift / s0**3 - 3.0 * shift**2 / s0**4,
        2.0 / s0**3 - 6.0 * shift / s0**4,
    )
    safe = jnp.where(inside, s0, radius2)
    outside = (1.0 / safe, -1.0 / safe**2, 2.0 / safe**3)
    return (
        jnp.where(inside, taylor[0], outside[0]),
        jnp.where(inside, taylor[1], outside[1]),
        jnp.where(inside, taylor[2], outside[2]),
    )


def _screened_source(points: Array, args: object) -> Array:
    """``-Laplace(u) + u`` of the ``C^3`` capped dipole ``u = x g(|x|^2)``.

    ``-Laplace(x g) = -x (4 |x|^2 g'' + 8 g')`` vanishes outside the unit disk,
    where ``x g = x / |x|^2`` is harmonic, so the exterior of the square carries
    the exact decaying dipole.
    """
    del args
    radius2 = jnp.sum(points**2, axis=-1)
    value, first, second = _cap_profile(radius2)
    return points[..., 0] * (value - 4.0 * radius2 * second - 8.0 * first)


def _dipole(points: np.ndarray, /) -> np.ndarray:
    """Host ``x / |x|^2``: the capped dipole on and outside the square."""
    return points[..., 0] / np.sum(points**2, axis=-1)


def _dipole_gradient(points: np.ndarray, /) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    radius2 = x * x + y * y
    return np.stack(((y * y - x * x) / radius2**2, -2.0 * x * y / radius2**2), axis=-1)


def _identity(domain: phx.domain.Domain, /) -> dict[str, phx.domain.DomainFunction]:
    return {"x": domain.Function("x")(lambda x: x)}


def _centered_square(half_width: float, /) -> phx.domain.HyperRectangle:
    return phx.domain.HyperRectangle(np.full(2, -half_width), np.full(2, half_width))


def _band(inner: float, outer: float, /) -> Callable[[Array], Array]:
    def support(x: Array) -> Array:
        size = jnp.max(jnp.abs(x))
        return ((size >= inner) & (size <= outer)).astype(jnp.float64)

    return support


def _square_cover() -> phx.domain.SubdomainCover:
    """Analytic authority: the square, a witness band outside it, their pairing."""
    window = _centered_square(3.0)
    patches = tuple(
        phx.domain.SubdomainPatch(
            window,
            window.component(),
            window.Function("x")(_band(inner, outer)),
            _identity(window),
            _identity(window),
            patch_id=patch_id,
        )
        for patch_id, inner, outer in (("square", 0.0, _HALF), ("exterior", _HALF, 3.0))
    )
    square = _centered_square(_HALF)

    def normal(x: Array) -> Array:
        axis = jnp.argmax(jnp.abs(x))
        return jnp.where(jnp.arange(2) == axis, jnp.sign(x), 0.0)

    pairing = phx.domain.PairedSupport(
        square.component({"x": phx.domain.Boundary()}),
        _identity(square),
        _identity(square),
        pairing_id="square|exterior",
        left_patch_id="square",
        right_patch_id="exterior",
        normal=square.Function("x")(normal),
    )
    return phx.domain.SubdomainCover(window, patches, (pairing,), cover_id="full-square")


def _square_binding(
    cover: phx.domain.SubdomainCover,
    endpoints: tuple[tuple[str, dict[str, str]], tuple[str, dict[str, str]]],
    /,
) -> cpl.InterfaceBinding:
    """Two-sided binding ``(minus, plus)`` of the square: its normal points outward."""
    pairing = cover.pairing("square|exterior")
    witness = pairing.component.sample(phx.domain.PointSampling(16), key=jr.key(3))
    patches = (pairing.left_patch_id, pairing.right_patch_id)
    return cpl.InterfaceBinding(
        "square|exterior",
        cpl.InterfaceSource.paired_support(cover, "square|exterior"),
        "two-sided",
        tuple(
            cpl.InterfaceEndpoint(
                role,
                cpl.PairedSupportAttachment(cover, "square|exterior", patch, witness),
                fields=fields,
            )
            for (role, fields), patch in zip(endpoints, patches, strict=True)
        ),
    )


def _square_grid(cells: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Counterclockwise quadrilaterals, ``2 cells`` per side of ``[-1.5, 1.5]^2``."""
    count = 2 * cells
    axis = np.linspace(-_HALF, _HALF, count + 1)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1
    ).reshape(-1, 4)
    return grid, quads.astype(np.int32)


type SquareSpace = (
    phx.discretization.FiniteElementDiscretization
    | phx.discretization.ExplicitPolygonH1Discretization
    | iga.PreparedIsogeometricDiscretization
)


def _lagrange_square(
    cells: int, degree: int, tensor: bool, /
) -> phx.discretization.FiniteElementDiscretization:
    """Tensor GLL quadrilaterals (spectral elements) or split triangles."""
    points, quads = _square_grid(cells)
    blocks, kind = (
        (quads, "quadrilateral")
        if tensor
        else (np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]])), "triangle")
    )
    return phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh(
            jnp.asarray(points),
            (phx.discretization.CellBlock("cells", kind, jnp.asarray(blocks)),),
        ),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(kind, degree)
        ),
    ).prepare()


def _polygon_square(cells: int, /) -> SquareSpace:
    points, quads = _square_grid(cells)
    return phx.discretization.ExplicitPolygonH1Plan(
        phx.discretization.CellMesh.from_polygons(
            jnp.asarray(points), tuple(np.asarray(cell) for cell in quads)
        ),
        phx.discretization.ExplicitPolygonH1FieldSpec("u"),
    ).prepare()


def _iga_square(cells: int, degree: int, /) -> SquareSpace:
    """One affine B-spline patch of ``[-1.5, 1.5]^2`` with unit weights."""
    grid = iga.BSplineGrid.open_uniform(degree, 2 * cells, interval=(-_HALF, _HALF))
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    return iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(degree + 2),
    ).prepare(numeric_version="square-patch")


def _square_space(method: SquareMethod, cells: int, degree: int, /) -> SquareSpace:
    match method:
        case "sem":
            return _lagrange_square(cells, degree, True)
        case "fe":
            return _lagrange_square(cells, degree, False)
        case "polygon":
            return _polygon_square(cells)
        case "iga":
            return _iga_square(cells, degree)


def _square_owner(
    space: SquareSpace,
    /,
    *,
    extra: tuple[phx.equations.BoundaryLoadAction, ...] = (),
    constraint: phx.discretization.FiniteElementDirichletConstraint | None = None,
    source: Callable[[Array, object], Array] = _screened_source,
) -> cpl.VariationalComponent:
    """``-Laplace(u) + u = f`` of the capped dipole, coupled on its whole boundary.

    The mass term removes the constant mode that a pure Neumann volume shares
    with the bounded exterior (``u = 1``, ``c = 1``), so the coupled problem
    has the unique solution with ``c = 0``: the decaying dipole.
    """
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "screened-laplace",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.MassAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(source, coefficient_id="forcing"),
                ),
                *extra,
            ),
        ),
        space,
        constraint=constraint,
        dirichlet_values=None if constraint is None else _square_dirichlet,
    )
    return cpl.VariationalComponent("square", problem, field="u")


def _square_dirichlet(points: Array) -> Array:
    """The capped dipole ``x g(|x|^2)``, finite at every node (the origin included)."""
    value, _, _ = _cap_profile(jnp.sum(points**2, axis=-1))
    return points[..., 0] * value


def _exterior(
    panels_per_side: int, /, *, half_width: float = _HALF
) -> cpl.GalerkinBoundaryComponent:
    """Galerkin Laplace exterior of a counterclockwise square of uniform panels."""
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
        policy=phx.operators.ScalarLaplaceGalerkinPolicy2D(regular_order=8),
    )
    return cpl.GalerkinBoundaryComponent("exterior", galerkin)


def _boundary_law(
    binding: cpl.InterfaceBinding,
    domain: IntegrationDomain,
    /,
    *,
    projection_order: int | None = None,
    interface_source: Callable[[Array], Array] | None = None,
) -> cpl.BoundaryIntegralTransmissionLaw:
    """The one boundary-integral declaration every volume method reuses."""
    return cpl.BoundaryIntegralTransmissionLaw(
        "boundary",
        binding,
        cpl.TransmissionSide("volume", "square", "u", domain),
        cpl.BoundaryIntegralSide("exterior", "exterior"),
        projection_order=projection_order,
        interface_source=interface_source,
    )


def _endpoints(
    volume: cpl.VariationalComponent, exterior: cpl.GalerkinBoundaryComponent, /
) -> tuple[tuple[str, dict[str, str]], tuple[str, dict[str, str]]]:
    return (
        ("volume", {"value": volume.field_space_id("u")}),
        ("exterior", {"conormal": exterior.field_space_id("conormal")}),
    )


def _prepare_square(
    volume: cpl.VariationalComponent,
    exterior: cpl.GalerkinBoundaryComponent,
    cover: phx.domain.SubdomainCover,
    binding: cpl.InterfaceBinding,
    law: cpl.BoundaryIntegralTransmissionLaw,
    /,
) -> cpl.PreparedCoupledProblem:
    plan = cpl.CoupledProblemPlan(
        "bem-square", components=(volume, exterior), bindings=(binding,), laws=(law,)
    )
    return cpl.prepare_coupled_problem(plan, interface_owners=(cover,))


_FAR_TARGETS = np.concatenate(
    [
        radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        for radius, angles in (
            (2.5, np.linspace(0.1, 2.0 * np.pi, 24, endpoint=False)),
            (10.0, np.linspace(0.3, 2.0 * np.pi, 12, endpoint=False)),
        )
    ]
)


def _panel_conormal(galerkin: phx.operators.ScalarLaplaceGalerkin2D, /) -> np.ndarray:
    """Exact panel means of ``d_n u`` along the interior-to-exterior normal (12-point Gauss)."""
    nodes, weights = np.polynomial.legendre.leggauss(12)
    vertices = np.asarray(galerkin.curve.vertices)
    ends = np.asarray(galerkin.curve.panel_vertices)
    start, stop = vertices[ends[:, 0]], vertices[ends[:, 1]]
    points = start[:, None] + 0.5 * (nodes + 1.0)[None, :, None] * (stop - start)[:, None]
    normals = np.asarray(galerkin.curve.normals)
    return 0.5 * np.sum(_dipole_gradient(points) * normals[:, None, :], axis=-1) @ weights


@dataclass(frozen=True, slots=True)
class SquareCase:
    """One volume method of ``[-1.5, 1.5]^2`` and its measured accuracy bounds.

    ``conormal`` bounds the length-weighted L2 error of the DP0 exterior
    conormal against exact panel means; ``far_field`` bounds the maximum error
    of the exterior representation at 36 targets on ``|x| = 2.5`` and
    ``|x| = 10``. Each bound is 1.25x the error measured at this resolution
    (two boundary panels per volume facet). The resolution study behind them
    (conormal / far field; reference norms ``|q| ~ 0.86``, ``max |u| = 0.4``):

    - sem p = 4: one cell per half side 9.83e-2 / 9.53e-3, two 2.60e-2 / 7.17e-4;
    - fe P2: one cell 3.39e-1 / 7.61e-2, two 8.10e-2 / 6.29e-3;
    - polygon: one cell 4.03e-1 / 1.04e-1, two 1.25e-1 / 9.37e-3;
    - iga (quadratic B-splines, 2 cells = 2 spans per half side): one cell
      3.88e-1 / 2.54e-2, two 6.49e-2 / 4.29e-4.

    Every level changes both errors by at least 3x, so a coupling defect that
    costs one resolution level fails the bound, while solver roundoff (gated
    defects ~1e-15) cannot reach it.
    """

    method: SquareMethod
    cells: int
    degree: int
    trace_degree: int
    conormal: float
    far_field: float


_PANELS_PER_FACET = 2

_SQUARE_CASES = (
    SquareCase("sem", 1, 4, 4, 1.25e-1, 1.2e-2),
    SquareCase("fe", 2, 2, 2, 1.0e-1, 8.0e-3),
    SquareCase("polygon", 2, 1, 1, 1.6e-1, 1.2e-2),
    SquareCase("iga", 2, 2, 2, 8.1e-2, 5.4e-4),
)


@pytest.mark.parametrize("case", _SQUARE_CASES, ids=lambda case: case.method)
def test_boundary_law_couples_every_volume_method_unchanged(case: SquareCase) -> None:
    """One boundary-integral declaration around spectral elements, Lagrange
    triangles, explicit polygons, and a single-patch spline owner (whose trace
    acts on its compiled public control layout); only the volume owner's
    preparation differs.
    """
    space = _square_space(case.method, case.cells, case.degree)
    volume = _square_owner(space)
    exterior = _exterior(2 * case.cells * _PANELS_PER_FACET)
    cover = _square_cover()
    binding = _square_binding(cover, _endpoints(volume, exterior))
    prepared = _prepare_square(
        volume,
        exterior,
        cover,
        binding,
        _boundary_law(binding, space.exterior_facet_domain),
    )
    (law,) = prepared.laws
    evidence = law.evidence
    assert isinstance(evidence, cpl.BoundaryIntegralEvidence)
    assert evidence.trace_degree == case.trace_degree
    # The declared rule integrates the P1 load of every trace up to this degree.
    assert evidence.projection_exact_degree >= case.trace_degree
    assert evidence.panel_count == 8 * case.cells * _PANELS_PER_FACET

    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    assert bool(solution.interface("boundary").accepted(solution.tolerance))

    galerkin = exterior.galerkin
    conormal = np.asarray(solution.field("exterior", "conormal"))
    constant = np.asarray(solution.field("exterior", "far_field_constant"))
    (dirichlet,) = solution.law_state("boundary")
    lengths = np.asarray(galerkin.curve.lengths)
    error = np.sqrt(np.sum(lengths * (conormal - _panel_conormal(galerkin)) ** 2))
    assert error <= case.conormal
    far = galerkin.evaluate_field(
        _FAR_TARGETS,
        side="exterior",
        dirichlet=dirichlet,
        conormal=conormal,
        far_field_constant=constant[0],
    )
    assert bool(far.accepted)
    far_error = np.max(np.abs(np.asarray(far.values) - _dipole(_FAR_TARGETS)))
    assert far_error <= case.far_field


def _unit_source(points: Array, args: object) -> Array:
    del args
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


def test_boundary_law_accepts_a_constant_equilibrium_whose_terms_cancel() -> None:
    """``-Laplace(u) + u = 1`` inside, bounded exterior: ``u = phi = c = 1``, ``q = 0``.

    The owner reaction ``(K + M) 1 - f`` and the exterior response to constant
    Dirichlet data both cancel to roundoff, so the gated flux balance, Cauchy
    residual, and logarithmic compatibility are measured against the terms
    that cancel, not against the roundoff-sized sums.
    """
    space = _lagrange_square(2, 2, False)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "screened-constant",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.MassAction("u", 1.0),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_unit_source, coefficient_id="unit")
                ),
            ),
        ),
        space,
    )
    volume = cpl.VariationalComponent("square", problem, field="u")
    exterior = _exterior(8)
    cover = _square_cover()
    binding = _square_binding(cover, _endpoints(volume, exterior))
    prepared = _prepare_square(
        volume,
        exterior,
        cover,
        binding,
        _boundary_law(binding, space.exterior_facet_domain),
    )
    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())

    assert bool(solution.native_successful)
    assert bool(solution.interface("boundary").accepted(solution.tolerance))
    assert bool(solution.accepted)
    np.testing.assert_allclose(solution.field("square", "u"), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        solution.field("exterior", "far_field_constant"), 1.0, atol=1.0e-12
    )
    # Exact q = 0 up to the Galerkin quadrature tolerance (1e-10 per entry).
    assert float(jnp.max(jnp.abs(solution.field("exterior", "conormal")))) < 1.0e-9


# --- Boundary-integral law refusals --------------------------------------------------------


@dataclass(frozen=True)
class SquareDeclaration:
    """Components, binding, and law of one boundary-integral square declaration."""

    volume: cpl.VariationalComponent
    exterior: cpl.GalerkinBoundaryComponent
    binding: cpl.InterfaceBinding
    law: cpl.BoundaryIntegralTransmissionLaw


@dataclass(frozen=True)
class RefusalFixture:
    """The accepted ``fe`` substitution row (P2 triangles, four facets per
    side, two panels per volume facet) and its geometry authority."""

    space: phx.discretization.FiniteElementDiscretization
    volume: cpl.VariationalComponent
    exterior: cpl.GalerkinBoundaryComponent
    cover: phx.domain.SubdomainCover

    def declare(
        self,
        *,
        volume: cpl.VariationalComponent | None = None,
        exterior: cpl.GalerkinBoundaryComponent | None = None,
        domain: IntegrationDomain | None = None,
        swap_roles: bool = False,
        volume_field: str | None = None,
        law_sides: tuple[str, str] = ("square", "exterior"),
        projection_order: int | None = None,
        interface_source: Callable[[Array], Array] | None = None,
    ) -> SquareDeclaration:
        """The valid declaration with exactly the named pieces replaced."""
        volume_ = self.volume if volume is None else volume
        exterior_ = self.exterior if exterior is None else exterior
        endpoints = _endpoints(volume_, exterior_)
        if volume_field is not None:
            endpoints = (("volume", {"value": volume_field}), endpoints[1])
        if swap_roles:
            endpoints = (endpoints[1], endpoints[0])
        binding = _square_binding(self.cover, endpoints)
        law = cpl.BoundaryIntegralTransmissionLaw(
            "boundary",
            binding,
            cpl.TransmissionSide(
                "volume",
                law_sides[0],
                "u",
                self.space.exterior_facet_domain if domain is None else domain,
            ),
            cpl.BoundaryIntegralSide("exterior", law_sides[1]),
            projection_order=projection_order,
            interface_source=interface_source,
        )
        return SquareDeclaration(volume_, exterior_, binding, law)


@pytest.fixture(scope="module")
def refusal_square() -> RefusalFixture:
    space = _lagrange_square(2, 2, False)
    volume = _square_owner(space)
    return RefusalFixture(space, volume, _exterior(8), _square_cover())


def _three_sides(
    space: phx.discretization.FiniteElementDiscretization, /
) -> IntegrationDomain:
    """Exterior facets of the square except its bottom side ``y = -1.5``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    bottom = np.all(np.isclose(np.asarray(probe.sites)[..., 1], -_HALF), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[~bottom]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


def _strong_owner(
    space: phx.discretization.FiniteElementDiscretization, /
) -> cpl.VariationalComponent:
    boundary = np.asarray(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    return _square_owner(
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=boundary
        ),
    )


def _loaded_owner(
    space: phx.discretization.FiniteElementDiscretization, /
) -> cpl.VariationalComponent:
    return _square_owner(
        space,
        extra=(
            phx.equations.BoundaryLoadAction(
                "u", 1.0, action_id="boundary-load", domain=space.exterior_facet_domain
            ),
        ),
    )


def _one_value_per_panel(points: Array) -> Array:
    return jnp.zeros(points.shape[:1], dtype=points.dtype)


type Refusal = Callable[[RefusalFixture], SquareDeclaration]

_REFUSALS: tuple[tuple[str, Refusal, type[Exception], str], ...] = (
    (
        "straddling-panels",
        lambda fixture: fixture.declare(exterior=_exterior(2)),
        ValueError,
        "span several facets",
    ),
    (
        "curve-off-the-volume-boundary",
        lambda fixture: fixture.declare(exterior=_exterior(8, half_width=1.75)),
        ValueError,
        "share no segment",
    ),
    (
        "coverage-gap",
        lambda fixture: fixture.declare(domain=_three_sides(fixture.space)),
        ValueError,
        "do not cover each other",
    ),
    (
        "swapped-binding-roles",
        lambda fixture: fixture.declare(swap_roles=True),
        ValueError,
        "minus role",
    ),
    (
        "mismatched-endpoint-field",
        lambda fixture: fixture.declare(
            volume_field=fixture.exterior.field_space_id("conormal")
        ),
        ValueError,
        "binds value field",
    ),
    (
        "strong-rows-on-the-boundary",
        lambda fixture: fixture.declare(volume=_strong_owner(fixture.space)),
        ValueError,
        "imposes trace rows strongly",
    ),
    (
        "owner-load-on-the-boundary",
        lambda fixture: fixture.declare(volume=_loaded_owner(fixture.space)),
        ValueError,
        "already imposes",
    ),
    (
        "inexact-projection-order",
        lambda fixture: fixture.declare(projection_order=1),
        ValueError,
        "projection_order 1",
    ),
    (
        "interface-source-shape",
        lambda fixture: fixture.declare(interface_source=_one_value_per_panel),
        ValueError,
        "interface_source must return",
    ),
    (
        "non-trace-volume-component",
        lambda fixture: fixture.declare(law_sides=("exterior", "exterior")),
        TypeError,
        "publishes no side traces",
    ),
    (
        "non-galerkin-exterior",
        lambda fixture: fixture.declare(law_sides=("square", "square")),
        TypeError,
        "not a 2-D Galerkin",
    ),
)


@pytest.mark.parametrize(
    ("declare", "error", "match"),
    [row[1:] for row in _REFUSALS],
    ids=[row[0] for row in _REFUSALS],
)
def test_boundary_law_refuses_inconsistent_declarations(
    refusal_square: RefusalFixture, declare: Refusal, error: type[Exception], match: str
) -> None:
    """Each declaration differs from the accepted ``fe`` row by one piece."""
    declaration = declare(refusal_square)
    plan = cpl.CoupledProblemPlan(
        "bem-square",
        components=(declaration.volume, declaration.exterior),
        bindings=(declaration.binding,),
        laws=(declaration.law,),
    )
    with pytest.raises(error, match=match):
        cpl.prepare_coupled_problem(plan, interface_owners=(refusal_square.cover,))


# --- Coupled kernel and far-field declaration ----------------------------------------------


def _laplace_source(points: Array, args: object) -> Array:
    """``-Laplace(u)`` of the capped dipole: odd in ``x``, so its integral vanishes."""
    del args
    radius2 = jnp.sum(points**2, axis=-1)
    _, first, second = _cap_profile(radius2)
    return -points[..., 0] * (4.0 * radius2 * second + 8.0 * first)


@dataclass(frozen=True)
class NeumannSquare:
    """Pure-Neumann ``-Laplace(u) = f`` in the square next to the bounded exterior.

    The spectral-element owner (p = 4, two cells per half side) publishes only
    its constant kernel; the coupled kernel also carries the law's trace
    projection and the exterior's far-field constant. The source is odd in
    ``x`` and the mesh symmetric, so the discrete load is compatible with the
    left kernel to roundoff and the gauge keeps ``compatibility="error"``.
    """

    plan: cpl.CoupledProblemPlan
    cover: phx.domain.SubdomainCover
    space: phx.discretization.FiniteElementDiscretization
    exterior: cpl.GalerkinBoundaryComponent


def _neumann_square(
    gauge: cpl.CoupledGauge | None, /, *, diffusion: float = 1.0
) -> NeumannSquare:
    space = _lagrange_square(2, 4, True)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "laplace",
            "u",
            (
                phx.equations.DiffusionAction("u", diffusion),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(_laplace_source, coefficient_id="forcing"),
                ),
            ),
        ),
        space,
    )
    volume = cpl.VariationalComponent("square", problem, field="u")
    exterior = _exterior(8)
    cover = _square_cover()
    binding = _square_binding(cover, _endpoints(volume, exterior))
    plan = cpl.CoupledProblemPlan(
        "neumann-square",
        components=(volume, exterior),
        bindings=(binding,),
        laws=(_boundary_law(binding, space.exterior_facet_domain),),
        gauge=gauge,
    )
    return NeumannSquare(plan, cover, space, exterior)


def test_kernel_through_law_unknowns_is_refused_without_a_gauge() -> None:
    """``u = 1, phi = 1, q = 0, c = 1`` is found at preparation, not at solve time."""
    square = _neumann_square(None)
    with pytest.raises(ValueError, match="1-dimensional kernel"):
        cpl.prepare_coupled_problem(square.plan, interface_owners=(square.cover,))


@pytest.mark.parametrize("diffusion", [1.0, 1.0e9], ids=["unit", "stiff-volume"])
def test_gauged_kernel_pair_is_the_coupled_constant_mode(diffusion: float) -> None:
    """With a gauge, the prepared nullspace policy carries the exact kernel pair.

    Right: ``u = 1, phi = 1, q = 0, c = 1`` (normalized). Left (row covectors,
    identified with states): ``u = 1, c = 1`` with vanishing projection and
    boundary-equation rows, which differs from the right kernel because the
    Johnson--Nedelec coupling is nonsymmetric. Scaling the volume operator by
    ``1e9`` leaves the kernel pair unchanged, although 66 directions of the
    boundary-integral images then fall below ``1e-8`` of the whole operator's
    probe scale (each output block is read at its own gain).
    """
    square = _neumann_square(cpl.CoupledGauge(), diffusion=diffusion)
    prepared = cpl.prepare_coupled_problem(square.plan, interface_owners=(square.cover,))
    policy = prepared.nullspace_policy
    assert policy is not None and policy.right is not None and policy.left is not None
    state = prepared.chart.state_space

    def blocks(basis: Array, /) -> dict[tuple[str, str], np.ndarray]:
        assert basis.shape[1] == 1
        vector = state.unflatten(jnp.asarray(basis[:, 0]))
        values: dict[tuple[str, str], np.ndarray] = {}
        for index, (owner, member) in enumerate(
            zip(state.names, state.spaces, strict=True)
        ):
            assert isinstance(member, phx.linalg.BlockSpace)
            for position, name in enumerate(member.names):
                values[(owner, name)] = np.asarray(vector[index][position])
        return values

    right = blocks(policy.right.basis)
    constant = right[("exterior", "far_field_constant")][0]
    assert abs(constant) > 0.0
    np.testing.assert_allclose(right[("square", "u")], constant, rtol=1.0e-10)
    np.testing.assert_allclose(
        right[("boundary", "dirichlet-trace")], constant, rtol=1.0e-10
    )
    np.testing.assert_allclose(right[("exterior", "conormal")], 0.0, atol=1.0e-10)

    left = blocks(policy.left.basis)
    volume = left[("square", "u")]
    assert abs(volume[0]) > 0.0
    np.testing.assert_allclose(volume, volume[0], rtol=1.0e-10)
    assert abs(left[("exterior", "far_field_constant")][0]) > 0.0
    np.testing.assert_allclose(left[("boundary", "dirichlet-trace")], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(left[("exterior", "conormal")], 0.0, atol=1.0e-10)


@pytest.mark.parametrize(
    ("tolerance", "accepted"),
    [(1.0e-2, True), (1.0e-3, False)],
    ids=["within-tolerance", "beyond-tolerance"],
)
def test_decaying_exterior_gates_the_far_field_constant(
    refusal_square: RefusalFixture, tolerance: float, accepted: bool
) -> None:
    """The accepted ``fe`` row solves ``|c| = 2.70e-3`` (its discretization
    error; the dipole decays). A decaying declaration accepts it within a
    ``1e-2`` tolerance and refuses it beyond ``1e-3``; every other gated
    defect is unchanged."""
    exterior = cpl.GalerkinBoundaryComponent(
        "exterior",
        refusal_square.exterior.galerkin,
        far_field="decaying",
        far_field_tolerance=tolerance,
    )
    declaration = refusal_square.declare(exterior=exterior)
    plan = cpl.CoupledProblemPlan(
        "bem-square",
        components=(declaration.volume, declaration.exterior),
        bindings=(declaration.binding,),
        laws=(declaration.law,),
    )
    prepared = cpl.prepare_coupled_problem(plan, interface_owners=(refusal_square.cover,))
    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    report = solution.interface("boundary")
    constant = abs(float(np.asarray(solution.field("exterior", "far_field_constant"))[0]))
    decay = report.names.index("far-field-decay")

    assert bool(solution.native_successful)
    assert 1.0e-3 < constant < 1.0e-2
    assert report.gated[decay]
    np.testing.assert_allclose(
        float(report.values[decay]), max(constant - tolerance, 0.0), rtol=1.0e-12
    )
    others = [
        index for index, gated in enumerate(report.gated) if gated and index != decay
    ]
    values, scales = np.asarray(report.values), np.asarray(report.scales)
    assert np.all(values[others] <= solution.tolerance * scales[others])
    assert bool(solution.accepted) is accepted


@dataclass(frozen=True)
class BEMSquareSystem:
    """The accepted ``fe`` row's coupled system and its dense matrix, assembled
    column by column from forward actions only."""

    system: phx.linalg.LinearSystem
    state: phx.linalg.BlockSpace
    rhs: Array
    dense: np.ndarray


@pytest.fixture(scope="module")
def bem_square_system(refusal_square: RefusalFixture) -> BEMSquareSystem:
    declaration = refusal_square.declare()
    plan = cpl.CoupledProblemPlan(
        "bem-square",
        components=(declaration.volume, declaration.exterior),
        bindings=(declaration.binding,),
        laws=(declaration.law,),
    )
    prepared = cpl.prepare_coupled_problem(plan, interface_owners=(refusal_square.cover,))
    system, rhs = prepared.linear_system()
    state = prepared.state_space

    def action(column: Array, /) -> Array:
        return state.flatten(system.operator.mv(state.unflatten(column)))

    dense = np.asarray(
        jax.vmap(action, in_axes=1, out_axes=1)(jnp.eye(state.size, dtype=jnp.float64))
    )
    return BEMSquareSystem(system, state, state.flatten(rhs), dense)


def test_bem_square_transpose_is_the_dense_transpose(
    bem_square_system: BEMSquareSystem,
) -> None:
    """The Riesz-identified exterior rows transpose to ``A^T``, not ``A`` with
    the Riesz map untransposed."""
    square = bem_square_system
    covector = np.random.default_rng(3).standard_normal(square.state.size)
    transposed = square.state.flatten(
        square.system.operator.transpose_mv(
            square.state.unflatten(jnp.asarray(covector, dtype=jnp.float64))
        )
    )
    expected = square.dense.T @ covector

    np.testing.assert_allclose(
        np.asarray(transposed), expected, rtol=0.0, atol=1.0e-12 * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "method",
    ["fgmres-krylov-derivative", "dense-lu-primal-factors"],
)
def test_bem_square_reverse_derivative_is_the_dense_adjoint_solve(
    bem_square_system: BEMSquareSystem,
    method: Literal["fgmres-krylov-derivative", "dense-lu-primal-factors"],
) -> None:
    """``grad_b (w . A^{-1} b) = A^{-T} w``. FGMRES converges within its full
    declared restart; the default Krylov derivative solve keeps that restart
    rather than a stalling 30-vector one. Dense LU reuses its factors."""
    square = bem_square_system
    size = square.state.size
    match method:
        case "fgmres-krylov-derivative":
            policy = phx.linalg.LinearSolvePolicy(
                phx.linalg.FGMRES(restart=size + 5, stagnation_iterations=size + 5),
                tolerance=phx.linalg.TolerancePolicy(relative=1.0e-13, absolute=0.0),
            )
        case "dense-lu-primal-factors":
            policy = phx.linalg.LinearSolvePolicy(
                phx.linalg.DenseLU(),
                derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
                    route="primal-factors"
                ),
            )
        case _:
            assert_never(method)
    weight = np.random.default_rng(0).standard_normal(size)

    def functional(rhs: Array, /) -> Array:
        result = phx.linalg.solve(
            square.system, square.state.unflatten(rhs), policy=policy
        )
        return jnp.vdot(jnp.asarray(weight), square.state.flatten(result.value))

    gradient = np.asarray(jax.grad(functional)(square.rhs))
    expected = np.linalg.solve(square.dense.T, weight)

    np.testing.assert_allclose(
        gradient, expected, rtol=0.0, atol=1.0e-6 * np.abs(expected).max()
    )


_FORCING_SCALE = "forcing-scale"


def _scaled_screened_source(points: Array, args: object) -> Array:
    """The capped-dipole forcing times the solver argument ``forcing-scale``."""
    assert isinstance(args, phx.equations.FiniteElementExecutionContext)
    user = args.user_args
    assert isinstance(user, Mapping)
    return user[_FORCING_SCALE] * _screened_source(points, None)


def test_bem_square_default_policy_reverse_derivative_is_the_dense_adjoint_solve(
    refusal_square: RefusalFixture, bem_square_system: BEMSquareSystem
) -> None:
    """Without a declared policy the indefinite coupled system is solved and
    differentiated through dense factors: ``d/ds w . u(s) = (A^{-T} w) . b``
    for the forcing ``s b``. The capability-selected Krylov default and its
    GMRES(30) derivative solve returned NaN on this system while reporting
    ``derivative_valid``."""
    square = bem_square_system
    volume = _square_owner(refusal_square.space, source=_scaled_screened_source)
    declaration = refusal_square.declare(volume=volume)
    port = phx.ValuePort(
        _FORCING_SCALE, event_shape=(), component_ids=("value",), representation="scalar"
    )
    plan = cpl.CoupledProblemPlan(
        "bem-square-forcing",
        components=(declaration.volume, declaration.exterior),
        bindings=(declaration.binding,),
        laws=(declaration.law,),
        parameters=(
            cpl.ParameterBinding(
                _FORCING_SCALE,
                port,
                targets=(cpl.RuntimeInput("square", _FORCING_SCALE),),
                role="control",
                derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
            ),
        ),
    )
    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(refusal_square.cover,),
        parameters={_FORCING_SCALE: jnp.asarray(1.0)},
    )
    weight = np.random.default_rng(0).standard_normal(square.state.size)

    def functional(scale: Array, /) -> Array:
        solution = cpl.solve_coupled_problem(prepared, parameters={_FORCING_SCALE: scale})
        return jnp.vdot(jnp.asarray(weight), prepared.state_space.flatten(solution.state))

    def rhs(scale: float, /) -> np.ndarray:
        bound = prepared.bind_arguments(parameters={_FORCING_SCALE: jnp.asarray(scale)})
        return np.asarray(
            prepared.state_space.flatten(prepared.linear_system(bound.arguments)[1])
        )

    gradient = float(jax.grad(functional)(jnp.asarray(0.7)))
    expected = float(np.linalg.solve(square.dense.T, weight) @ (rhs(1.0) - rhs(0.0)))

    assert gradient == pytest.approx(expected, rel=1.0e-8)


@pytest.mark.parametrize(
    ("far_field", "tolerance", "match"),
    [
        ("decaying", None, "requires far_field_tolerance"),
        ("bounded", 1.0e-3, "takes no far_field_tolerance"),
        ("decaying", -1.0e-3, "far_field_tolerance"),
        ("vanishing", None, "far_field"),
    ],
    ids=["decaying-without-tolerance", "bounded-with-tolerance", "negative", "unknown"],
)
def test_far_field_declaration_refuses_incomplete_modes(
    refusal_square: RefusalFixture, far_field: str, tolerance: float | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        cpl.GalerkinBoundaryComponent(
            "exterior",
            refusal_square.exterior.galerkin,
            far_field=far_field,  # ty: ignore[invalid-argument-type]
            far_field_tolerance=tolerance,
        )
