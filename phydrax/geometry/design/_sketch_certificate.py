# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host certificates over native sketch rank and interval substrates."""

from __future__ import annotations

from typing import final, get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    RankPolicy,
    RHSLayout,
    solve,
)
from ...linalg._rank import (
    numerical_rank_data,
    rank_gap_margin,
    singular_value_backward_error,
)
from ...linalg.svd import DenseSVD, svd, SVDProblem, SVDSolvePolicy
from ...typing import checked, parse
from .._interval_enclosure import (
    interval_add,
    interval_dot,
    interval_subtract,
    IntervalExtensionError,
    prepare_interval_function,
)
from ..brep._constructors import PlanarProfile, ProfilePlane
from ._sketch import (
    _finite_norm,
    AbstractSketchConstraint,
    Coincident,
    EqualLength,
    FixedPoint,
    Horizontal,
    LineAngle,
    Midpoint,
    Parallel,
    Perpendicular,
    PointDistance,
    PointOnLine,
    Radius,
    Sketch,
    SketchConstraint,
    SketchSolution,
    TangentCircles,
    TangentLineCircle,
    Vertical,
)


SketchCertificateStatus: TypeAlias = Literal[
    "certified",
    "rank_unresolved",
    "not_fully_constrained",
    "unsupported",
    "enclosure_unresolved",
]


@final
class SketchComponentDof(StrictModule):
    """One geometric component; dependent IDs follow authored constraint order."""

    variable_indices: tuple[int, ...] = eqx.field(static=True)
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    backward_error: float = eqx.field(static=True)
    rank_certified: bool = eqx.field(static=True)
    underconstrained_ids: tuple[str, ...] = eqx.field(static=True)
    overconstrained_ids: tuple[str, ...] = eqx.field(static=True)
    redundant_ids: tuple[str, ...] = eqx.field(static=True)


@final
class SketchDofCertificate(StrictModule):
    """Numerical rank decisions with their spectral separation evidence."""

    components: tuple[SketchComponentDof, ...]

    @property
    def rank(self) -> int:
        return sum(component.rank for component in self.components)

    @property
    def nullity(self) -> int:
        return sum(component.nullity for component in self.components)

    @property
    def rank_certified(self) -> bool:
        return all(component.rank_certified for component in self.components)

    @property
    def fully_constrained(self) -> bool:
        return (
            self.rank_certified
            and self.nullity == 0
            and all(not component.overconstrained_ids for component in self.components)
        )


@final
class SketchCertifiedSolution(StrictModule):
    """A native solve result and, on success only, a unique-root spatial enclosure."""

    solution: SketchSolution
    dof: SketchDofCertificate
    status: SketchCertificateStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    lower: Array
    upper: Array
    image_lower: Array
    image_upper: Array
    radius: Array

    @checked
    def __init__(
        self,
        solution: SketchSolution,
        dof: SketchDofCertificate,
        status: SketchCertificateStatus,
        reason: str,
        lower: Array,
        upper: Array,
        image_lower: Array,
        image_upper: Array,
        radius: Array,
    ) -> None:
        status = parse(status, SketchCertificateStatus, "SketchCertifiedSolution.status")
        self.solution, self.dof, self.status, self.reason = solution, dof, status, reason
        self.lower, self.upper = lower, upper
        self.image_lower, self.image_upper, self.radius = image_lower, image_upper, radius

    @checked
    def to_profile(self, *, plane: ProfilePlane | None = None) -> SketchCertifiedProfile:
        """Lower with the unique-root enclosure attached as source-fidelity evidence."""
        if self.status != "certified":
            raise ValueError(
                f"SketchCertifiedSolution.status must certify a unique-root enclosure for profile lowering; got {self.status}: {self.reason}"
            )
        profile = self.solution.sketch.to_profile(self.solution, plane=plane)
        return SketchCertifiedProfile(profile, self)


@final
class SketchCertifiedProfile(StrictModule):
    """A lowered profile coupled to its certified-root variable enclosure."""

    profile: PlanarProfile = eqx.field(static=True)
    certificate: SketchCertifiedSolution


def _constraint_variables(
    constraint: AbstractSketchConstraint,
    sketch: Sketch,
) -> tuple[int, ...]:
    lines, centers = jax.device_get((sketch.lines, sketch.circle_centers))
    points: tuple[int, ...] = ()
    line_ids: tuple[int, ...] = ()
    circles: tuple[int, ...] = ()
    match constraint:
        case Coincident() | PointDistance():
            points = (constraint.first_point, constraint.second_point)
        case FixedPoint():
            points = (constraint.point,)
        case Midpoint() | PointOnLine():
            points, line_ids = (constraint.point,), (constraint.line,)
        case Horizontal() | Vertical():
            line_ids = (constraint.line,)
        case Parallel() | Perpendicular() | EqualLength() | LineAngle():
            line_ids = (constraint.first_line, constraint.second_line)
        case Radius():
            circles = (constraint.circle,)
        case TangentLineCircle():
            line_ids, circles = (constraint.line,), (constraint.circle,)
        case TangentCircles():
            circles = (constraint.first_circle, constraint.second_circle)
        case _:
            raise TypeError(
                f"SketchDofCertificate.constraint must belong to the admitted native family; got {type(constraint).__name__}"
            )
    points += tuple(int(point) for line in line_ids for point in lines[line])
    points += tuple(int(centers[circle]) for circle in circles)
    return tuple(
        sorted(
            {2 * point + axis for point in points for axis in (0, 1)}
            | {sketch.points.size + circle for circle in circles}
        )
    )


def _constraint_groups(
    sketch: Sketch,
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    groups = []
    row = 0
    for identity, constraint in zip(
        sketch.constraint_ids, sketch.constraints, strict=True
    ):
        width = constraint.residual(
            sketch.points, sketch.lines, sketch.circle_centers, sketch.circle_radii
        ).size
        groups.append(
            (
                identity,
                tuple(range(row, row + width)),
                _constraint_variables(constraint, sketch),
            )
        )
        row += width
    arcs, centers = jax.device_get((sketch.arcs, sketch.circle_centers))
    for column, label in ((1, "start"), (2, "end")):
        for index, arc in enumerate(arcs):
            circle, point = int(arc[0]), int(arc[column])
            variables = tuple(
                sorted(
                    (
                        2 * point,
                        2 * point + 1,
                        2 * int(centers[circle]),
                        2 * int(centers[circle]) + 1,
                        sketch.points.size + circle,
                    )
                )
            )
            groups.append((f"arc:{index}:{label}", (row,), variables))
            row += 1
    return tuple(groups)


def _rank(
    matrix: Array,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, float, float, bool]:
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return 0, np.inf, 0.0, True
    budget.reserve(16 * max(rows, columns) ** 3, 128 * (rows + columns) ** 2)
    if bool(jnp.all(matrix == 0.0)):
        return 0, np.inf, 0.0, True
    policy = RankPolicy(relative_cutoff=8 * max(rows, columns) * np.finfo(np.float64).eps)
    spectrum = svd(
        SVDProblem(DenseLinearOperator(matrix)),
        policy=SVDSolvePolicy(
            DenseSVD(),
            count=min(rows, columns),
            rank=policy,
            materialization=MaterializationPolicy(
                max_entries=rows * columns,
                max_bytes=matrix.size * 8,
                max_workspace_bytes=128 * (rows + columns) ** 2,
            ),
        ),
    ).singular_values
    data = numerical_rank_data(spectrum, rows, columns, policy)
    retained = jnp.min(jnp.where(data.retained, spectrum, jnp.inf))
    discarded = jnp.max(jnp.where(data.retained, 0.0, spectrum))
    margin = rank_gap_margin(
        data.rank, min(rows, columns), retained, discarded, data.cutoff, data.cutoff
    )
    error = singular_value_backward_error(jnp.max(spectrum), rows, columns)
    rank, margin_, error_, finite = jax.device_get(
        (data.rank, margin, error, data.finite)
    )
    return int(rank), float(margin_), float(error_), bool(finite & (margin_ > error_))


def _components(
    sketch: Sketch,
    groups: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    count = sketch.points.size + sketch.circle_radii.size
    adjacency = [set((index,)) for index in range(count)]
    lines, centers = jax.device_get((sketch.lines, sketch.circle_centers))
    geometric = [
        tuple(2 * int(point) + axis for point in line for axis in (0, 1))
        for line in lines
    ]
    geometric += [
        (2 * int(point), 2 * int(point) + 1, sketch.points.size + circle)
        for circle, point in enumerate(centers)
    ]
    geometric += [(2 * point, 2 * point + 1) for point in range(sketch.points.shape[0])]
    for variables in geometric + [variables for _, _, variables in groups]:
        for variable in variables:
            adjacency[variable].update(variables)
    remaining = set(range(count))
    components = []
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            component.update(frontier)
            frontier = set().union(*(adjacency[index] for index in frontier)) - component
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _dof_certificate(
    solution: SketchSolution,
    vector: Array,
    budget: CoordinateEnclosureBudget,
) -> SketchDofCertificate:
    sketch = solution.sketch
    groups = _constraint_groups(sketch)
    budget.reserve(
        max(1, vector.size**2 * sum(len(rows) for _, rows, _ in groups)),
        32 * max(1, vector.size) * max(1, sum(len(rows) for _, rows, _ in groups)),
    )

    def residual(point: Array) -> Array:
        return sketch.residual(
            point[: sketch.points.size].reshape(sketch.points.shape),
            point[sketch.points.size :],
        )

    jacobian = jax.jacfwd(residual)(vector)
    values = np.asarray(jax.device_get(residual(vector)))
    bound = float(solution.residual_bound)
    components = []
    for variables in _components(sketch, groups):
        selected = tuple(
            group for group in groups if set(group[2]).intersection(variables)
        )
        rows = tuple(row for _, group_rows, _ in selected for row in group_rows)
        matrix = jacobian[jnp.asarray(rows, dtype=jnp.int32)][
            :, jnp.asarray(variables, dtype=jnp.int32)
        ]
        rank, margin, error, certified = _rank(matrix, budget)
        prefix: tuple[int, ...] = ()
        prefix_rank, redundant, over = 0, [], []
        for identity, group_rows, _ in selected:
            next_rows = prefix + group_rows
            next_rank, _, _, next_certified = _rank(
                jacobian[jnp.asarray(next_rows, dtype=jnp.int32)][
                    :, jnp.asarray(variables, dtype=jnp.int32)
                ],
                budget,
            )
            certified &= next_certified
            if np.max(np.abs(values[list(group_rows)]), initial=0.0) > bound:
                over.append(identity)
            elif next_rank - prefix_rank < len(group_rows):
                redundant.append(identity)
            prefix, prefix_rank = next_rows, next_rank
        identities = tuple(identity for identity, _, _ in selected)
        components.append(
            SketchComponentDof(
                variables,
                identities,
                rank,
                len(variables) - rank,
                margin,
                error,
                certified,
                identities if rank < len(variables) else (),
                tuple(over),
                tuple(redundant),
            )
        )
    return SketchDofCertificate(tuple(components))


def _angle_products(
    constraint: LineAngle, sketch: Sketch, points: Array
) -> tuple[Array, Array]:
    first_ends = points[sketch.lines[constraint.first_line]]
    second_ends = points[sketch.lines[constraint.second_line]]
    first, second = first_ends[1] - first_ends[0], second_ends[1] - second_ends[0]
    cross = first[0] * second[1] - first[1] * second[0]
    dot = jnp.sum(first * second)
    sine, cosine = jnp.sin(constraint.angle), jnp.cos(constraint.angle)
    return cross * cosine - dot * sine, dot * cosine + cross * sine


def _certification_residual(sketch: Sketch, vector: Array) -> Array:
    """Replace angle branch cuts by an equivalent sine equation, with a sector proof."""
    points = vector[: sketch.points.size].reshape(sketch.points.shape)
    radii = vector[sketch.points.size :]
    values = []
    for constraint in sketch.constraints:
        if isinstance(constraint, LineAngle):
            angle_residual, _ = _angle_products(constraint, sketch, points)
            values.append(constraint._weighted(angle_residual))
        else:
            values.append(
                constraint.residual(
                    points, sketch.lines, sketch.circle_centers, radii
                ).reshape(-1)
            )
    if sketch.arcs.shape[0]:
        centers = points[sketch.circle_centers[sketch.arcs[:, 0]]]
        values.extend(
            _finite_norm(points[sketch.arcs[:, column]] - centers)
            - radii[sketch.arcs[:, 0]]
            for column in (1, 2)
        )
    return jnp.concatenate(values) if values else jnp.empty((0,), dtype=jnp.float64)


def _angle_sectors(sketch: Sketch, vector: Array) -> Array:
    points = vector[: sketch.points.size].reshape(sketch.points.shape)
    return jnp.asarray(
        [
            _angle_products(constraint, sketch, points)[1]
            for constraint in sketch.constraints
            if isinstance(constraint, LineAngle)
        ],
        dtype=jnp.float64,
    )


def _krawczyk(
    solution: SketchSolution,
    center: Array,
    budget: CoordinateEnclosureBudget,
) -> tuple[Array, Array, Array, Array, Array, str]:
    sketch = solution.sketch
    size = center.size
    budget.reserve(64 * size**3, 512 * size**2)

    def residual(point: Array) -> Array:
        return _certification_residual(sketch, point)

    jacobian = jax.jacfwd(residual)(center)
    if jacobian.shape != (size, size):
        raise IntervalExtensionError(
            f"SketchCertifiedSolution.jacobian must be square and nonredundant for unique-root enclosure; got shape {jacobian.shape}"
        )
    receipt = solve(
        LinearSystem(DenseLinearOperator(jacobian)),
        jnp.eye(size, dtype=jnp.float64),
        policy=LinearSolvePolicy(
            DenseLU(), differentiation=DifferentiationPolicy("none")
        ),
        rhs_layout=RHSLayout((size,)),
    )
    if not bool(jnp.all(receipt.successful)):
        raise IntervalExtensionError(
            f"SketchCertifiedSolution.preconditioner must carry successful native solve evidence; got status {receipt.status}"
        )
    preconditioner = jnp.asarray(receipt.value)

    def correction(point: Array) -> Array:
        return preconditioner @ residual(point)

    def remainder(point: Array) -> Array:
        return jnp.eye(size, dtype=jnp.float64) - preconditioner @ jax.jacfwd(residual)(
            point
        )

    budget.reserve(16 * size**3, 512 * size**2)
    correction_map = prepare_interval_function(correction, size, batch_capacity=1)
    remainder_map = prepare_interval_function(remainder, size, batch_capacity=1)
    host = np.asarray(jax.device_get(center))
    correction_low, correction_high = correction_map.evaluate(host[None], host[None])
    correction_bound = np.maximum(np.abs(correction_low[0]), np.abs(correction_high[0]))
    radius = np.nextafter(
        4 * correction_bound
        + 64 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(host)),
        np.inf,
    )
    lower, upper = (
        np.nextafter(host - radius, -np.inf),
        np.nextafter(host + radius, np.inf),
    )
    if np.any(lower[sketch.points.size :] <= 0.0):
        raise IntervalExtensionError(
            f"SketchCertifiedSolution.lower_radii must remain positive across the Krawczyk box; got {lower[sketch.points.size :]}"
        )
    if any(isinstance(constraint, LineAngle) for constraint in sketch.constraints):

        def sectors(point: Array) -> Array:
            return _angle_sectors(sketch, point)

        sector_map = prepare_interval_function(sectors, size, batch_capacity=1)
        sector_lower, _ = sector_map.evaluate(lower[None], upper[None])
        if not np.all(sector_lower > 0.0):
            raise IntervalExtensionError(
                f"SketchCertifiedSolution.angle_sectors must remain positive across the enclosure; got lower bounds {sector_lower}"
            )
    matrix_low, matrix_high = remainder_map.evaluate(lower[None], upper[None])
    base = interval_subtract((host, host), (correction_low[0], correction_high[0]))
    displacement = interval_subtract((lower, upper), (host, host))
    image_low, image_high = np.empty_like(host), np.empty_like(host)
    for row in range(size):
        contribution = interval_dot(
            (matrix_low[0, row], matrix_high[0, row]), displacement
        )
        low, high = interval_add(
            (np.asarray(base[0])[row], np.asarray(base[1])[row]), contribution
        )
        image_low[row], image_high[row] = low, high
    unique = (
        np.all(np.isfinite(image_low + image_high))
        and np.all(image_low > lower)
        and np.all(image_high < upper)
    )
    published_radius = np.nextafter(np.maximum(host - lower, upper - host), np.inf)
    return (
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(image_low, dtype=jnp.float64),
        jnp.asarray(image_high, dtype=jnp.float64),
        jnp.asarray(published_radius, dtype=jnp.float64),
        "certified" if unique else "enclosure_unresolved",
    )


@checked
def certify_sketch_solution(
    solution: SketchSolution,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SketchCertifiedSolution:
    """Certify rank and a unique root without an accuracy or tolerance option.

    Preparation is host-only. Differentiation uses the existing solution's native
    implicit-root attachment; an enclosure never differentiates a host decision.
    """
    with budget.temporary_scope():
        size = solution.points.size + solution.circle_radii.size
        groups_upper = (
            len(solution.sketch.constraints) + 2 * solution.sketch.arcs.shape[0]
        )
        budget.reserve(
            64 * max(1, size) * max(1, groups_upper),
            512 * max(1, size) * max(1, groups_upper),
        )
        if any(
            type(constraint) not in get_args(SketchConstraint)
            for constraint in solution.sketch.constraints
        ):
            unavailable = SketchComponentDof(
                tuple(range(size)),
                solution.sketch.constraint_ids,
                -1,
                -1,
                np.nan,
                np.nan,
                False,
                (),
                (),
                (),
            )
            empty = jnp.empty((0,), dtype=jnp.float64)
            return SketchCertifiedSolution(
                solution,
                SketchDofCertificate((unavailable,)),
                "unsupported",
                "Custom constraint families have no admitted certification work bound.",
                empty,
                empty,
                empty,
                empty,
                empty,
            )
        vector = jnp.concatenate((solution.points.reshape(-1), solution.circle_radii))
        dof = _dof_certificate(solution, vector, budget)
        empty = jnp.empty((0,), dtype=jnp.float64)
        status: SketchCertificateStatus = "not_fully_constrained"
        reason = "Unique-root enclosure requires a fully constrained, consistent sketch."
        if not dof.rank_certified:
            status, reason = (
                "rank_unresolved",
                "Rank gap does not exceed the SVD backward error.",
            )
        elif dof.fully_constrained and bool(solution.converged):
            try:
                lower, upper, image_low, image_high, radius, decision = _krawczyk(
                    solution, vector, budget
                )
            except IntervalExtensionError as error:
                status, reason = "unsupported", str(error)
            else:
                status = parse(
                    decision, SketchCertificateStatus, "SketchCertifiedSolution.status"
                )
                reason = (
                    "Strict Krawczyk inclusion proves a unique exact root."
                    if status == "certified"
                    else "Krawczyk image is not strictly inside the enclosure."
                )
                return SketchCertifiedSolution(
                    solution,
                    dof,
                    status,
                    reason,
                    lower,
                    upper,
                    image_low,
                    image_high,
                    radius,
                )
        return SketchCertifiedSolution(
            solution, dof, status, reason, empty, empty, empty, empty, empty
        )


__all__ = [
    "SketchCertificateStatus",
    "SketchComponentDof",
    "SketchDofCertificate",
    "SketchCertifiedSolution",
    "SketchCertifiedProfile",
    "certify_sketch_solution",
]
