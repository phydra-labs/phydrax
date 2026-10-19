"""Scientific contracts of the 2-D scalar Laplace Galerkin boundary operator.

Independent references: hand-derived closed forms for straight-panel double
integrals (coincident, collinear-adjacent, and right-angle panels), mpmath
tanh-sinh quadrature for general shared-endpoint panels, plain NumPy tensor
Gauss--Legendre quadrature for separated and near panels, and harmonic fields
with analytic Cauchy data for the Green representation identities.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import numpy.typing as npt
import pytest

import phydrax as phx
from phydrax.operators import (
    BoundaryTraceProjection2D,
    ClosedPolygonalCurve2D,
    ExteriorLaplaceDirichletResult2D,
    prepare_boundary_trace_projection_2d,
    prepare_exterior_laplace_dirichlet_2d,
    prepare_scalar_laplace_galerkin_2d,
    PreparedExteriorLaplaceDirichlet2D,
    ScalarLaplaceGalerkin2D,
    ScalarLaplaceGalerkinPolicy2D,
    solve_exterior_laplace_dirichlet_2d,
)


type _Floats = npt.NDArray[np.float64]
type _Field = Callable[[_Floats], tuple[_Floats, _Floats]]

_DENSE = phx.linalg.MaterializationPolicy(max_entries=200_000, max_bytes=16 << 20)
_TWO_PI = 2.0 * np.pi


def _square(panels_per_side: int, half: float = 1.0) -> _Floats:
    steps = np.linspace(-half, half, panels_per_side + 1)[:-1]
    ones = np.full(panels_per_side, half)
    return np.concatenate(
        (
            np.stack((steps, -ones), axis=1),
            np.stack((ones, steps), axis=1),
            np.stack((-steps, ones), axis=1),
            np.stack((-ones, -steps), axis=1),
        )
    )


def _prepare(
    vertices: _Floats,
    source_id: str,
    policy: ScalarLaplaceGalerkinPolicy2D | None = None,
) -> ScalarLaplaceGalerkin2D:
    return prepare_scalar_laplace_galerkin_2d(
        ClosedPolygonalCurve2D(vertices, source_id=source_id), policy=policy
    )


@cache
def _square_galerkin(panels_per_side: int) -> ScalarLaplaceGalerkin2D:
    return _prepare(_square(panels_per_side), f"square-{panels_per_side}")


@cache
def _square_projection(panels_per_side: int) -> BoundaryTraceProjection2D:
    return prepare_boundary_trace_projection_2d(
        _square_galerkin(panels_per_side).spaces, order=8
    )


def _matrices(galerkin: ScalarLaplaceGalerkin2D) -> tuple[_Floats, _Floats]:
    single = np.asarray(phx.linalg.materialize(galerkin.single_layer, _DENSE))
    double = np.asarray(phx.linalg.materialize(galerkin.double_layer, _DENSE))
    return single, double


def _outward_normals(vertices: _Floats) -> _Floats:
    """Shoelace orientation: the exterior lies to the right of a CCW traversal."""
    successors = np.roll(vertices, -1, axis=0)
    area = np.sum(vertices[:, 0] * successors[:, 1] - successors[:, 0] * vertices[:, 1])
    tangents = successors - vertices
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    right = np.stack((tangents[:, 1], -tangents[:, 0]), axis=1)
    return right if area > 0.0 else -right


def _dual_norm(rows: _Floats, lengths: _Floats) -> float:
    return float(np.sqrt(np.sum(rows * rows / lengths)))


def _dipole(points: _Floats) -> tuple[_Floats, _Floats]:
    x, y = points[..., 0], points[..., 1]
    squared = x * x + y * y
    gradient = np.stack(((y * y - x * x) / squared**2, -2.0 * x * y / squared**2), -1)
    return x / squared, gradient


def _logarithm(points: _Floats) -> tuple[_Floats, _Floats]:
    squared = np.sum(points * points, axis=-1)
    return 0.5 * np.log(squared), points / squared[..., None]


def _interior_quadratic(points: _Floats) -> tuple[_Floats, _Floats]:
    x, y = points[..., 0], points[..., 1]
    value = x * x - y * y + 0.5 * x * y
    return value, np.stack((2.0 * x + 0.5 * y, -2.0 * y + 0.5 * x), axis=-1)


def _cauchy_data(panels_per_side: int, field: _Field) -> tuple[jax.Array, jax.Array]:
    """P1 and DP0 L2 projections of analytic Dirichlet and conormal traces."""
    projection = _square_projection(panels_per_side)
    points = np.asarray(projection.sample_points)
    value, gradient = field(points)
    normals = _outward_normals(_square(panels_per_side))[:, None, :]
    dirichlet = projection.project_dirichlet(jnp.asarray(value))
    conormal = projection.project_conormal(jnp.asarray(np.sum(gradient * normals, -1)))
    assert bool(dirichlet.successful) and bool(conormal.successful)
    return dirichlet.coefficients, conormal.coefficients


def test_straight_panel_closed_forms_match_hand_derived_integrals() -> None:
    """Coincident, collinear-adjacent, and right-angle entries are exact.

    ``∫_0^h∫_0^h log|s-t| = h²(log h - 3/2)``;
    ``∫_0^a∫_0^b log(s+t) = [(a+b)² log(a+b) - a² log a - b² log b]/2 - 3ab/2``;
    ``∫_0^1∫_0^1 log(s²+t²)/2 = (log 2 - 3 + π/2)/2``; and on the unit square's
    bottom and right panels the shared-vertex hat moment of ``K`` is
    ``-(1/2π)[∫∫ s/(s²+t²) - ∫∫ st/(s²+t²)] = -(1/2π)(π/4) = -1/8``.
    """
    a, b = 0.6, 1.1
    rectangle = np.asarray(((0.0, 0.0), (a, 0.0), (a + b, 0.0), (a + b, 1.0), (0.0, 1.0)))
    single, _ = _matrices(_prepare(rectangle, "collinear-rectangle"))
    coincident = [length**2 * (1.5 - np.log(length)) / _TWO_PI for length in (a, b)]
    collinear = (
        -(
            0.5 * ((a + b) ** 2 * np.log(a + b) - a * a * np.log(a) - b * b * np.log(b))
            - 1.5 * a * b
        )
        / _TWO_PI
    )
    np.testing.assert_allclose(np.diag(single)[:2], coincident, rtol=1.0e-13)
    np.testing.assert_allclose(single[0, 1], collinear, rtol=1.0e-10)

    unit = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    single, double = _matrices(_prepare(unit, "unit-square"))
    right_angle = -(np.log(2.0) - 3.0 + 0.5 * np.pi) / (2.0 * _TWO_PI)
    np.testing.assert_allclose(single[0, 1], right_angle, rtol=1.0e-10)
    np.testing.assert_allclose(double[0, 1], -0.125, rtol=1.0e-10)


class _Corner(NamedTuple):
    angle: float
    first: float
    second: float


@pytest.mark.parametrize(
    "corner",
    (_Corner(2.0, 0.7, 1.3), _Corner(0.6, 1.0, 0.8), _Corner(1.3, 0.02, 1.0)),
    ids=("obtuse", "acute", "graded"),
)
def test_shared_endpoint_entries_match_mpmath(corner: _Corner) -> None:
    """Panels ``A -> C`` and ``C -> B`` meeting at ``C`` against tanh-sinh quadrature.

    ``K[0, C]`` is the shared-vertex hat moment of panel ``C -> B`` alone
    because the coincident double layer of a straight panel vanishes.
    """
    direction = np.asarray((np.cos(corner.angle), np.sin(corner.angle)))
    vertices = np.stack(
        (np.asarray((corner.first, 0.0)), np.zeros(2), corner.second * direction)
    )
    single, double = _matrices(_prepare(vertices, f"corner-{corner.angle}"))
    normal = _outward_normals(vertices)[1]
    mpmath.mp.dps = 20

    def difference(s: mpmath.mpf, t: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
        return corner.first - s - t * direction[0], -t * direction[1]

    def log_kernel(s: mpmath.mpf, t: mpmath.mpf) -> mpmath.mpf:
        dx, dy = difference(s, t)
        return -mpmath.log(dx * dx + dy * dy) / (2 * _TWO_PI)

    def shared_hat_kernel(s: mpmath.mpf, t: mpmath.mpf) -> mpmath.mpf:
        dx, dy = difference(s, t)
        normal_part = dx * normal[0] + dy * normal[1]
        return normal_part / (_TWO_PI * (dx * dx + dy * dy)) * (1 - t / corner.second)

    bounds = ([0, corner.first], [0, corner.second])
    single_reference = float(mpmath.quad(log_kernel, *bounds))
    double_reference = float(mpmath.quad(shared_hat_kernel, *bounds))
    np.testing.assert_allclose(single[0, 1], single_reference, rtol=1.0e-9, atol=1.0e-13)
    np.testing.assert_allclose(double[0, 1], double_reference, rtol=1.0e-8, atol=1.0e-12)


def _hat_moments(
    vertices: _Floats, target: int, source: int, order: int
) -> tuple[float, float, float]:
    """Plain NumPy tensor Gauss--Legendre ``V`` entry and ``K`` hat moments."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (nodes + 1.0)
    count = vertices.shape[0]
    x0, x1 = vertices[target], vertices[(target + 1) % count]
    y0, y1 = vertices[source], vertices[(source + 1) % count]
    x = x0 + unit[:, None] * (x1 - x0)
    y = y0 + unit[:, None] * (y1 - y0)
    weight = np.outer(
        0.5 * weights * np.linalg.norm(x1 - x0), 0.5 * weights * np.linalg.norm(y1 - y0)
    )
    difference = x[:, None, :] - y[None, :, :]
    squared = np.sum(difference * difference, axis=-1)
    double_kernel = (difference @ _outward_normals(vertices)[source]) / (
        _TWO_PI * squared
    )
    single = float(np.sum(weight * (-np.log(squared) / (2.0 * _TWO_PI))))
    start = float(np.sum(weight * double_kernel * (1.0 - unit)[None, :]))
    end = float(np.sum(weight * double_kernel * unit[None, :]))
    return single, start, end


_SLOT = np.asarray(
    (
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.05, 1.0),
        (1.05, 0.2),
        (0.95, 0.2),
        (0.95, 1.0),
        (0.0, 1.0),
    )
)


def test_separated_and_near_pairs_match_high_order_host_quadrature() -> None:
    """Slot walls 0.1 apart (near) and distant panels against 240-point rules.

    ``K[i, v]`` sums the end-hat moment of panel ``v - 1`` and the start-hat
    moment of panel ``v``; both panels are separated from ``i`` in every row.
    """
    galerkin = _prepare(_SLOT, "slot")
    single, double = _matrices(galerkin)
    assert galerkin.report.pair_counts[2] > 0
    for target, source in ((3, 5), (5, 3), (2, 6), (0, 3), (1, 7)):
        reference = _hat_moments(_SLOT, target, source, 240)[0]
        np.testing.assert_allclose(single[target, source], reference, rtol=1.0e-9)
    for target, vertex in ((3, 6), (5, 3), (0, 4), (1, 7), (6, 2)):
        end = _hat_moments(_SLOT, target, (vertex - 1) % 8, 240)[2]
        start = _hat_moments(_SLOT, target, vertex, 240)[1]
        np.testing.assert_allclose(
            double[target, vertex], end + start, rtol=1.0e-8, atol=1.0e-12
        )


def test_constant_density_double_layer_has_the_declared_jumps() -> None:
    """``D[1] = -1`` inside and ``0`` outside, so ``K 1 = -m/2`` weakly."""
    galerkin = _square_galerkin(4)
    curve = galerkin.curve
    ones = jnp.ones((curve.vertex_count,))
    zeros = jnp.zeros((curve.panel_count,))
    np.testing.assert_allclose(
        galerkin.double_layer.mv(ones), -0.5 * curve.lengths, rtol=0.0, atol=1.0e-12
    )
    rows, total = galerkin.exterior_relation.mv((ones, zeros, jnp.ones((1,))))
    np.testing.assert_allclose(rows, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(total, 0.0, atol=0.0)
    inside = galerkin.evaluate_field(
        [[0.2, -0.3], [0.99, 0.0]], side="interior", dirichlet=ones, conormal=zeros
    )
    outside = galerkin.evaluate_field(
        [[1.01, 0.0], [4.0, 7.0]],
        side="exterior",
        dirichlet=ones,
        conormal=zeros,
        far_field_constant=0.0,
    )
    np.testing.assert_allclose(inside.values, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(outside.values, 0.0, atol=1.0e-12)
    assert bool(inside.accepted) and bool(outside.accepted)
    assert galerkin.convention.double_layer_dirichlet_jump("exterior") == 0.5
    assert galerkin.convention.double_layer_dirichlet_jump("interior") == -0.5
    wrong_side = galerkin.evaluate_field(
        [[0.2, 0.1]],
        side="exterior",
        dirichlet=ones,
        conormal=zeros,
        far_field_constant=0.0,
    )
    assert not bool(wrong_side.accepted)


def test_blocked_actions_match_materialization_transposes_and_adjoints() -> None:
    """Padding-sensitive block size, dense agreement, exact V symmetry, and duality."""
    curve = ClosedPolygonalCurve2D(_square(4), source_id="square-4")
    galerkin = prepare_scalar_laplace_galerkin_2d(
        curve, policy=ScalarLaplaceGalerkinPolicy2D(block_size=5)
    )
    single, double = _matrices(galerkin)
    reference_single, reference_double = _matrices(_square_galerkin(4))
    np.testing.assert_allclose(single, reference_single, rtol=1.0e-13, atol=1.0e-15)
    np.testing.assert_allclose(double, reference_double, rtol=1.0e-13, atol=1.0e-15)
    # Exception entries are mirrored; regular entries differ only by summation order.
    np.testing.assert_allclose(
        single, single.T, rtol=0.0, atol=64 * 2.2e-16 * np.max(single)
    )
    panels = jax.random.normal(jax.random.key(0), (curve.panel_count,))
    vertices = jax.random.normal(jax.random.key(1), (curve.vertex_count,))
    np.testing.assert_allclose(
        galerkin.single_layer.mv(panels), single @ panels, atol=1e-14
    )
    np.testing.assert_allclose(
        galerkin.double_layer.mv(vertices), double @ vertices, atol=1e-14
    )
    np.testing.assert_allclose(
        galerkin.double_layer.transpose_mv(panels), double.T @ panels, atol=1e-14
    )
    operator = galerkin.double_layer
    image = operator.mv(vertices)
    adjoint = operator.adjoint_mv(panels)
    np.testing.assert_allclose(
        operator.target.inner(image, panels),
        operator.source.inner(vertices, adjoint),
        rtol=1.0e-10,
    )
    relation = galerkin.exterior_relation
    unknowns = (vertices, panels, jnp.asarray((0.7,)))
    rows = (panels, jnp.asarray((-1.3,)))
    left = sum(jnp.dot(a, b) for a, b in zip(relation.mv(unknowns), rows, strict=True))
    right = sum(
        jnp.dot(a, b) for a, b in zip(unknowns, relation.transpose_mv(rows), strict=True)
    )
    np.testing.assert_allclose(left, right, rtol=1.0e-12)


def test_orientation_reversal_preserves_physical_operators() -> None:
    """A clockwise traversal keeps outward normals; operators agree after relabeling."""
    count = 16
    forward = _square_galerkin(4)
    reverse = _prepare(_square(4)[::-1].copy(), "square-4-reversed")
    assert (forward.curve.traversal, reverse.curve.traversal) == (
        "counterclockwise",
        "clockwise",
    )
    panel = (count - 2 - np.arange(count)) % count
    vertex = count - 1 - np.arange(count)
    np.testing.assert_allclose(
        reverse.curve.normals, np.asarray(forward.curve.normals)[panel], atol=1.0e-15
    )
    single, double = _matrices(forward)
    reversed_single, reversed_double = _matrices(reverse)
    # Mirrored traversal evaluates the same adaptive rules in mirrored order;
    # agreement is bounded by the normalized pair tolerance 1e-10 * h_i h_j.
    np.testing.assert_allclose(reversed_single, single[np.ix_(panel, panel)], atol=1e-10)
    np.testing.assert_allclose(reversed_double, double[np.ix_(panel, vertex)], atol=1e-10)


def _relative_residual(panels_per_side: int, field: _Field, exterior: bool) -> float:
    galerkin = _square_galerkin(panels_per_side)
    lengths = np.asarray(galerkin.curve.lengths)
    dirichlet, conormal = _cauchy_data(panels_per_side, field)
    trace = np.asarray(0.5 * galerkin.spaces.mixed_mass.mv(dirichlet))
    double = np.asarray(galerkin.double_layer.mv(dirichlet))
    single = np.asarray(galerkin.single_layer.mv(conormal))
    rows = trace - double + single if exterior else trace + double - single
    scale = _dual_norm(trace - double if exterior else trace + double, lengths)
    return _dual_norm(rows, lengths) / (scale + _dual_norm(single, lengths))


@pytest.mark.parametrize(
    ("field", "exterior"),
    ((_dipole, True), (_interior_quadratic, False)),
    ids=("exterior-dipole", "interior-quadratic"),
)
def test_green_identities_hold_with_second_order_consistency(
    field: _Field, exterior: bool
) -> None:
    """Exterior ``(M/2 - K)φ + Vq = 0`` and interior ``(M/2 + K)φ - Vq = 0``.

    The weak residual of projected analytic Cauchy data converges at order two
    (P1 and DP0 projections, smooth data); the opposite-side relation does not.
    """
    residuals = [_relative_residual(n, field, exterior) for n in (8, 16, 32)]
    orders = np.log2(np.asarray(residuals[:-1]) / np.asarray(residuals[1:]))
    assert residuals[-1] < 1.0e-3
    assert np.all(orders > 1.7), orders
    assert _relative_residual(8, field, not exterior) > 0.2


def test_logarithmic_cauchy_data_violate_bounded_compatibility() -> None:
    """``log r`` satisfies the boundary equation but has total conormal ``2π``.

    The bordered solve of its Dirichlet data therefore returns the distinct
    bounded field with zero total conormal instead of ``log r``.
    """
    galerkin = _square_galerkin(8)
    dirichlet, conormal = _cauchy_data(8, _logarithm)
    _, total = galerkin.exterior_relation.mv((dirichlet, conormal, jnp.zeros((1,))))
    assert _relative_residual(8, _logarithm, True) < 1.0e-2
    np.testing.assert_allclose(total, _TWO_PI, rtol=1.0e-12)
    prepared = prepare_exterior_laplace_dirichlet_2d(galerkin)
    result = solve_exterior_laplace_dirichlet_2d(prepared, dirichlet)
    assert bool(result.accepted)
    assert abs(float(result.total_conormal)) < 1.0e-10
    far = galerkin.evaluate_field(
        [[40.0, 0.0]],
        side="exterior",
        dirichlet=dirichlet,
        conormal=result.conormal,
        far_field_constant=result.far_field_constant,
    )
    assert abs(float(far.values[0]) - np.log(40.0)) > 2.0


def test_bordered_exterior_solve_recovers_decaying_and_shifted_fields() -> None:
    """Dipole data decay; a shifted dipole is bounded with ``c_inf = 3``.

    Requesting decay for the shifted data reports the nonzero constant as an
    unsatisfied far field while the solve and equations themselves succeed.
    """
    galerkin = _square_galerkin(8)
    dirichlet, conormal = _cauchy_data(8, _dipole)
    decaying = prepare_exterior_laplace_dirichlet_2d(
        galerkin, far_field="decaying", far_field_tolerance=1.0e-3
    )
    result = solve_exterior_laplace_dirichlet_2d(decaying, dirichlet)
    assert bool(result.accepted) and bool(result.linear.successful)
    assert abs(float(result.far_field_constant)) < 1.0e-3
    assert abs(float(result.total_conormal)) < 1.0e-10
    lengths = np.asarray(galerkin.curve.lengths)
    error = _dual_norm(np.asarray(lengths * (result.conormal - conormal)), lengths)
    assert error < 0.05 * _dual_norm(np.asarray(lengths * conormal), lengths)
    targets = np.asarray(((1.3, 0.4), (3.0, -2.0), (25.0, 10.0)))
    field = galerkin.evaluate_field(
        targets,
        side="exterior",
        dirichlet=dirichlet,
        conormal=result.conormal,
        far_field_constant=result.far_field_constant,
    )
    np.testing.assert_allclose(field.values, _dipole(targets)[0], atol=2.0e-3)
    assert bool(field.accepted)

    shifted = dirichlet + 3.0
    bounded = solve_exterior_laplace_dirichlet_2d(
        prepare_exterior_laplace_dirichlet_2d(galerkin), shifted
    )
    assert bool(bounded.accepted)
    np.testing.assert_allclose(bounded.far_field_constant, 3.0, atol=1.0e-3)
    refused = solve_exterior_laplace_dirichlet_2d(decaying, shifted)
    assert bool(refused.linear.successful) and bool(refused.equations_certified)
    assert not bool(refused.far_field_satisfied)
    assert not bool(refused.accepted)


@pytest.mark.parametrize("panels_per_side", [1, 8])
def test_constant_dirichlet_data_are_accepted_with_a_cancelled_conormal(
    panels_per_side: int,
) -> None:
    """``φ = 1.7`` is the bounded exterior field ``u = 1.7``: ``q = 0``, ``c = 1.7``.

    Its double-layer and identity terms cancel, so the solved conormal and its
    total are at the level of roundoff and of the Galerkin quadrature tolerance
    (``1e-10`` per entry); the certificate measures them against the
    uncancelled Dirichlet terms rather than against that residue itself.
    """
    galerkin = _square_galerkin(panels_per_side)
    count = galerkin.curve.panel_count
    result = solve_exterior_laplace_dirichlet_2d(
        prepare_exterior_laplace_dirichlet_2d(galerkin), jnp.full((count,), 1.7)
    )

    assert bool(result.linear.successful)
    assert bool(result.equations_certified)
    assert bool(result.accepted)
    assert float(jnp.max(jnp.abs(result.conormal))) < 1.0e-9
    np.testing.assert_allclose(result.far_field_constant, 1.7, rtol=1.0e-12)


def test_trace_projection_is_a_declared_l2_projection() -> None:
    """P1 reproduction, Galerkin orthogonality, order-two defect, and adjoint."""
    projection = _square_projection(4)
    spaces = projection.spaces
    points = np.asarray(projection.sample_points)
    linear = 2.0 * points[..., 0] - points[..., 1] + 1.0
    vertices = np.asarray(spaces.curve.vertices)
    exact = projection.project_dirichlet(jnp.asarray(linear))
    np.testing.assert_allclose(
        exact.coefficients, 2.0 * vertices[:, 0] - vertices[:, 1] + 1.0, atol=1.0e-12
    )
    assert float(exact.relative_defect) < 1.0e-12
    means = projection.project_conormal(jnp.asarray(linear))
    midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=0))
    np.testing.assert_allclose(
        means.coefficients, 2.0 * midpoints[:, 0] - midpoints[:, 1] + 1.0, atol=1.0e-12
    )
    assert projection.exact_polynomial_degree == 2 * projection.order - 2

    defects = []
    for panels_per_side in (4, 8):
        refined = _square_projection(panels_per_side)
        samples = np.asarray(refined.sample_points)
        quadratic = jnp.asarray(samples[..., 0] ** 2 + samples[..., 1])
        result = refined.project_dirichlet(quadratic)
        assert bool(result.successful) and result.representation == "continuous-p1"
        orthogonality = refined.load.mv(
            quadratic
        ) - refined.spaces.dirichlet_trace.mass.mv(result.coefficients)
        np.testing.assert_allclose(orthogonality, 0.0, atol=1.0e-13)
        defects.append(float(result.relative_defect))
    assert 3.5 < defects[0] / defects[1] < 4.5

    coefficients = jax.random.normal(jax.random.key(2), (spaces.curve.vertex_count,))
    hats = np.asarray(projection.hat_values)
    panel_vertices = np.asarray(spaces.curve.panel_vertices)
    evaluation = (
        np.asarray(coefficients)[panel_vertices[:, 0]][:, None] * hats[None, :, 0]
        + np.asarray(coefficients)[panel_vertices[:, 1]][:, None] * hats[None, :, 1]
    )
    np.testing.assert_allclose(
        projection.dirichlet_projection.adjoint_mv(coefficients), evaluation, atol=1e-11
    )
    np.testing.assert_allclose(
        projection.dirichlet_projection.mv(jnp.asarray(linear)), exact.coefficients
    )


@pytest.mark.parametrize(
    "vertices",
    (
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 0.0), (0.0, 2.0)),
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0), (np.nan, 1.0)),
    ),
    ids=(
        "two-vertices",
        "zero-length-panel",
        "zero-area",
        "crossing",
        "touching",
        "fold-back",
        "non-finite",
    ),
)
def test_curve_refuses_non_simple_or_degenerate_polygons(
    vertices: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ValueError):
        ClosedPolygonalCurve2D(vertices, source_id="invalid")


def test_curve_requires_an_explicit_canonical_source_identity() -> None:
    unit = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    with pytest.raises(TypeError):
        ClosedPolygonalCurve2D(unit, source_id=3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        ClosedPolygonalCurve2D(unit, source_id=" padded ")
    first = ClosedPolygonalCurve2D(unit, source_id="first")
    second = ClosedPolygonalCurve2D(unit, source_id="second")
    assert first.curve_id != second.curve_id


def test_exterior_formulation_refuses_undeclared_configurations() -> None:
    galerkin = _square_galerkin(4)
    with pytest.raises(ValueError, match="bounded far field"):
        prepare_exterior_laplace_dirichlet_2d(galerkin, far_field_tolerance=1.0e-3)
    with pytest.raises(ValueError, match="requires far_field_tolerance"):
        prepare_exterior_laplace_dirichlet_2d(galerkin, far_field="decaying")
    with pytest.raises(ValueError):
        prepare_exterior_laplace_dirichlet_2d(
            galerkin,
            far_field="radiating",  # ty: ignore[invalid-argument-type]
        )
    mathematical = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(), differentiation=phx.linalg.DifferentiationPolicy()
    )
    with pytest.raises(ValueError, match="geometry derivatives are not qualified"):
        prepare_exterior_laplace_dirichlet_2d(galerkin, linear=mathematical)
    algorithmic = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        differentiation=phx.linalg.DifferentiationPolicy("algorithmic"),
    )
    with pytest.raises(ValueError, match="unrolled Krylov derivative"):
        prepare_exterior_laplace_dirichlet_2d(galerkin, linear=algorithmic)
    with pytest.raises(TypeError):
        prepare_exterior_laplace_dirichlet_2d("galerkin")  # ty: ignore[invalid-argument-type]


def test_resource_and_quadrature_limits_refuse_before_use() -> None:
    square = ClosedPolygonalCurve2D(_square(4), source_id="square-4")
    with pytest.raises(ValueError, match="exception-capacity"):
        prepare_scalar_laplace_galerkin_2d(
            square, policy=ScalarLaplaceGalerkinPolicy2D(max_exception_pairs=40)
        )
    with pytest.raises(ValueError, match="resident-bytes"):
        prepare_scalar_laplace_galerkin_2d(
            square, policy=ScalarLaplaceGalerkinPolicy2D(max_resident_bytes=1024)
        )
    graded = ((1.0e-5, 0.0), (0.0, 0.0), (np.cos(0.3), np.sin(0.3)))
    with pytest.raises(ValueError, match="quadrature-shared-endpoint"):
        _prepare(np.asarray(graded), "graded", ScalarLaplaceGalerkinPolicy2D(max_depth=2))
    with pytest.raises(phx.linalg.LinearCapabilityError):
        phx.linalg.materialize(
            _square_galerkin(4).single_layer,
            phx.linalg.MaterializationPolicy(max_entries=10),
        )


def test_adaptive_pair_quadrature_is_chunked_within_the_workspace_budget() -> None:
    """A small budget streams the adaptive exceptional pairs instead of overrunning.

    Sixteen panels need about 97 KiB for one regular-sweep row, while their 32
    shared-endpoint pairs alone evaluate ``32 x 21 x 3`` samples with their
    temporaries (about 190 KiB) at once. A 120 kB budget therefore admits the
    sweep but must stream the adaptive levels; the declared peak stays within
    it. Chunking changes only the evaluation order of independent intervals,
    so the assembled operators equal the unconstrained ones.
    """
    vertices = _square(4)
    reference = _prepare(vertices, "square-4-reference")
    budget = 120_000
    chunked = _prepare(
        vertices,
        "square-4-chunked",
        ScalarLaplaceGalerkinPolicy2D(max_preparation_workspace_bytes=budget),
    )

    assert 0 < chunked.report.preparation_workspace_bytes <= budget
    for constrained, unconstrained in (
        (chunked.single_layer, reference.single_layer),
        (chunked.double_layer, reference.double_layer),
    ):
        columns = np.eye(unconstrained.source.size)
        # Chunked rule contractions may differ from the batched ones in the
        # last bit only.
        np.testing.assert_allclose(
            np.stack([np.asarray(constrained.mv(column)) for column in columns]),
            np.stack([np.asarray(unconstrained.mv(column)) for column in columns]),
            rtol=1.0e-13,
            atol=1.0e-16,
        )


def test_report_publishes_pair_evidence_and_exact_support() -> None:
    galerkin = _square_galerkin(4)
    report = galerkin.report
    panels = galerkin.curve.panel_count
    assert report.pair_class_names == ("coincident", "shared-endpoint", "near", "regular")
    assert sum(report.pair_counts) == panels * panels
    assert report.pair_counts[:2] == (panels, 2 * panels)
    assert report.exception_count == sum(report.pair_counts[:3])
    assert bool(report.accuracy_supported) and bool(jnp.all(report.pair_class_supported))
    assert bool(jnp.all(report.maximum_errors <= report.tolerance))
    assert bool(jnp.all(report.evaluations > 0))
    assert report.materializable and not report.continuum_discretization_error_estimated
    support = report.support
    assert support.supports("bordered-bounded-exterior-dirichlet-to-neumann")
    assert support.supports("quadrature-error-bound")
    assert support.supports("fixed-geometry-density-derivatives")
    assert support.supports("dirichlet-data-implicit-derivative")
    for nonclaim in (
        "continuum-error",
        "open-curves",
        "curved-geometry",
        "moving-geometry",
        "helmholtz",
        "fast-multipole-galerkin-action",
        "geometry-derivatives",
        "field-target-derivatives",
    ):
        assert not support.supports(nonclaim)
    with pytest.raises(ValueError, match="Unknown boundary qualification claim"):
        support.supports("maxwell")
    assert report.kernel_id == phx.operators.LaplaceLayerKernel2D().kernel_id


_FIELD_TARGETS = np.asarray(((1.3, 0.4), (3.0, -2.0), (25.0, 10.0)))
_INTERIOR_CENTER = np.asarray((0.3, -0.2))


def _interior_dipole(points: _Floats) -> tuple[_Floats, _Floats]:
    """Exterior-decaying dipole centered inside the square: zero far-field constant."""
    return _dipole(points - _INTERIOR_CENTER)


def _linear_policy(
    mode: phx.linalg.DifferentiationMode, size: int, max_steps: int
) -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=size, stagnation_iterations=size),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1.0e-12, absolute=0.0, max_steps=max_steps
        ),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
        failure=phx.linalg.FailurePolicy("status"),
    )


def _exterior_outputs(
    prepared: PreparedExteriorLaplaceDirichlet2D,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    def outputs(dirichlet: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = solve_exterior_laplace_dirichlet_2d(prepared, dirichlet)
        return result.conormal, result.far_field_constant

    return outputs


def _central_difference(
    prepared: PreparedExteriorLaplaceDirichlet2D,
    dirichlet: jax.Array,
    direction: jax.Array,
    step: float,
) -> tuple[_Floats, float]:
    """Central differences of two accepted primal solves."""
    plus, minus = (
        solve_exterior_laplace_dirichlet_2d(prepared, dirichlet + sign * step * direction)
        for sign in (1.0, -1.0)
    )
    assert bool(plus.accepted) and bool(minus.accepted)
    conormal = (np.asarray(plus.conormal) - np.asarray(minus.conormal)) / (2.0 * step)
    constant = (float(plus.far_field_constant) - float(minus.far_field_constant)) / (
        2.0 * step
    )
    return conormal, constant


def _assert_jvp_vjp_match_differences(
    prepared: PreparedExteriorLaplaceDirichlet2D,
    dirichlet: jax.Array,
    direction: jax.Array,
    step: float,
) -> tuple[jax.Array, jax.Array]:
    outputs = _exterior_outputs(prepared)
    _, (conormal, constant) = jax.jvp(outputs, (dirichlet,), (direction,))
    reference_conormal, reference_constant = _central_difference(
        prepared, dirichlet, direction, step
    )
    scale = np.max(np.abs(reference_conormal))
    np.testing.assert_allclose(conormal, reference_conormal, rtol=0.0, atol=1e-7 * scale)
    np.testing.assert_allclose(constant, reference_constant, rtol=0.0, atol=1e-7 * scale)
    rng = np.random.default_rng(7)
    weights = jnp.asarray(rng.normal(size=conormal.shape))
    weight = jnp.asarray(rng.normal())
    _, pullback = jax.vjp(outputs, dirichlet)
    (gradient,) = pullback((weights, weight))
    np.testing.assert_allclose(
        jnp.dot(gradient, direction),
        jnp.dot(weights, conormal) + weight * constant,
        rtol=1.0e-8,
    )
    return conormal, constant


def test_density_actions_have_exact_linear_derivatives() -> None:
    """JVPs of ``V``, ``K``, and field evaluation are the actions; VJPs transpose them."""
    galerkin = _square_galerkin(4)
    capability = galerkin.derivative_capability
    for name in ("dirichlet", "conormal", "far_field_constant"):
        assert capability.require(name) is phx.DerivativeSurface.INPUT
    assert capability.derivative_contract.route is phx.DerivativeRoute.DIRECT
    assert capability.owner_id == galerkin.prepared_id
    rng = np.random.default_rng(3)
    panels, vertices = galerkin.curve.panel_count, galerkin.curve.vertex_count
    phi, phi_tangent = (jnp.asarray(rng.normal(size=vertices)) for _ in range(2))
    q, q_tangent, rows = (jnp.asarray(rng.normal(size=panels)) for _ in range(3))
    for operator, density, tangent in (
        (galerkin.single_layer, q, q_tangent),
        (galerkin.double_layer, phi, phi_tangent),
    ):
        _, image = jax.jvp(operator.mv, (density,), (tangent,))
        np.testing.assert_allclose(image, operator.mv(tangent), rtol=1e-12, atol=1e-14)
        _, pullback = jax.vjp(operator.mv, density)
        np.testing.assert_allclose(
            pullback(rows)[0], operator.transpose_mv(rows), rtol=1e-12, atol=1e-13
        )

    def field(
        dirichlet: jax.Array, conormal: jax.Array, constant: jax.Array
    ) -> jax.Array:
        return galerkin.evaluate_field(
            _FIELD_TARGETS,
            side="exterior",
            dirichlet=dirichlet,
            conormal=conormal,
            far_field_constant=constant,
        ).values

    constant, constant_tangent = jnp.asarray(0.4), jnp.asarray(-1.1)
    _, values = jax.jvp(
        field, (phi, q, constant), (phi_tangent, q_tangent, constant_tangent)
    )
    np.testing.assert_allclose(
        values, field(phi_tangent, q_tangent, constant_tangent), rtol=1e-12, atol=1e-14
    )
    weights = jnp.asarray(rng.normal(size=len(_FIELD_TARGETS)))
    _, pullback = jax.vjp(field, phi, q, constant)
    phi_bar, q_bar, constant_bar = pullback(weights)
    np.testing.assert_allclose(
        jnp.dot(phi_bar, phi_tangent)
        + jnp.dot(q_bar, q_tangent)
        + constant_bar * constant_tangent,
        jnp.dot(weights, values),
        rtol=1e-12,
    )


def test_galerkin_refuses_geometry_kernel_quadrature_and_target_derivatives() -> None:
    galerkin = _square_galerkin(4)
    capability = galerkin.derivative_capability
    for name, reason in (
        ("geometry", "fixed at host preparation"),
        ("kernel", "has no parameters"),
        ("quadrature", "not differentiable"),
        ("targets", "extension line of a panel"),
    ):
        assert not capability.admits(name)
        with pytest.raises(ValueError, match=f"derivative-unsupported.*{reason}"):
            capability.require(name)
    ones = jnp.ones((galerkin.curve.vertex_count,))
    zeros = jnp.zeros((galerkin.curve.panel_count,))

    def field(targets: jax.Array) -> jax.Array:
        return galerkin.evaluate_field(
            targets,
            side="exterior",
            dirichlet=ones,
            conormal=zeros,
            far_field_constant=0.0,
        ).values

    targets = jnp.asarray(_FIELD_TARGETS)
    with pytest.raises(ValueError, match="derivative-unsupported.*'targets'"):
        jax.jvp(field, (targets,), (jnp.ones_like(targets),))

    def moved(vertices: jax.Array) -> jax.Array:
        curve = eqx.tree_at(lambda curve: curve.vertices, galerkin.curve, vertices)
        return (
            eqx.tree_at(lambda model: model.curve, galerkin, curve)
            .evaluate_field(
                targets,
                side="exterior",
                dirichlet=ones,
                conormal=zeros,
                far_field_constant=0.0,
            )
            .values
        )

    with pytest.raises(ValueError, match="derivative-unsupported.*'geometry'"):
        jax.grad(lambda vertices: jnp.sum(moved(vertices)))(galerkin.curve.vertices)


def test_bounded_exterior_dirichlet_derivatives_match_finite_differences() -> None:
    """Implicit Dirichlet-data JVP/VJP against central differences of primal solves."""
    prepared = prepare_exterior_laplace_dirichlet_2d(_square_galerkin(4))
    capability = prepared.derivative_capability
    assert capability.require("dirichlet") is phx.DerivativeSurface.SOLVER_ARGUMENT
    contract = capability.derivative_contract
    assert contract.route is phx.DerivativeRoute.IMPLICIT
    assert contract.conditions == (
        "accepted-result",
        "prepared-geometry-fixed",
        "solve-converged",
    )
    for name in ("geometry", "kernel", "quadrature"):
        with pytest.raises(ValueError, match="derivative-unsupported"):
            capability.require(name)
    dirichlet = _cauchy_data(4, _dipole)[0] + 3.0
    result = solve_exterior_laplace_dirichlet_2d(prepared, dirichlet)
    assert bool(result.accepted) and bool(result.derivative_valid)
    assert result.derivative_capability.capability_id == capability.capability_id
    direction = jnp.asarray(np.random.default_rng(11).normal(size=dirichlet.shape[0]))
    _assert_jvp_vjp_match_differences(prepared, dirichlet, direction, 1.0e-3)


def test_decaying_exterior_derivatives_inside_the_compatibility_regime() -> None:
    """A decaying direction keeps every perturbed solve accepted; FD agree."""
    galerkin = _square_galerkin(8)
    prepared = prepare_exterior_laplace_dirichlet_2d(
        galerkin, far_field="decaying", far_field_tolerance=1.0e-3
    )
    dirichlet, _ = _cauchy_data(8, _dipole)
    direction, _ = _cauchy_data(8, _interior_dipole)
    assert bool(solve_exterior_laplace_dirichlet_2d(prepared, direction).accepted)
    _, constant = _assert_jvp_vjp_match_differences(
        prepared, dirichlet, direction, 1.0e-2
    )
    assert abs(float(constant)) < 1.0e-3


@pytest.mark.parametrize("failure", ["far-field-violated", "solve-failed"])
def test_rejected_exterior_solves_poison_their_derivatives(failure: str) -> None:
    """Inspectable primal values; NaN tangents instead of plausible derivatives."""
    match failure:
        case "far-field-violated":
            galerkin = _square_galerkin(8)
            prepared = prepare_exterior_laplace_dirichlet_2d(
                galerkin, far_field="decaying", far_field_tolerance=1.0e-3
            )
            dirichlet = _cauchy_data(8, _dipole)[0] + 3.0
        case "solve-failed":
            galerkin = _square_galerkin(4)
            size = galerkin.curve.panel_count + 1
            prepared = prepare_exterior_laplace_dirichlet_2d(
                galerkin, linear=_linear_policy("rhs-only", size, 1)
            )
            dirichlet = _cauchy_data(4, _dipole)[0] + 3.0
        case _:
            raise AssertionError(failure)
    result = solve_exterior_laplace_dirichlet_2d(prepared, dirichlet)
    assert not bool(result.accepted) and not bool(result.derivative_valid)
    assert bool(jnp.all(jnp.isfinite(result.conormal)))
    direction = jnp.ones_like(dirichlet)
    _, (conormal, constant) = jax.jvp(
        _exterior_outputs(prepared), (dirichlet,), (direction,)
    )
    assert bool(jnp.all(jnp.isnan(conormal))) and bool(jnp.isnan(constant))


def test_none_differentiation_refuses_dirichlet_derivatives() -> None:
    galerkin = _square_galerkin(4)
    size = galerkin.curve.panel_count + 1
    prepared = prepare_exterior_laplace_dirichlet_2d(
        galerkin, linear=_linear_policy("none", size, 8 * size)
    )
    capability = prepared.derivative_capability
    assert not capability.admits("dirichlet")
    assert capability.derivative_contract.route is phx.DerivativeRoute.STOPPED
    with pytest.raises(ValueError, match="derivative-unsupported.*mode 'none'"):
        capability.require("dirichlet")
    dirichlet = _cauchy_data(4, _dipole)[0] + 3.0
    result = solve_exterior_laplace_dirichlet_2d(prepared, dirichlet)
    assert bool(result.accepted) and not bool(result.derivative_valid)
    readouts: tuple[Callable[[ExteriorLaplaceDirichletResult2D], jax.Array], ...] = (
        lambda solved: solved.far_field_constant,
        lambda solved: solved.equation_residual,
        lambda solved: jnp.sum(solved.linear.value[0]),
    )
    for readout in readouts:
        with pytest.raises(ValueError, match="derivative-unsupported"):
            jax.grad(
                lambda data: readout(solve_exterior_laplace_dirichlet_2d(prepared, data))
            )(dirichlet)
