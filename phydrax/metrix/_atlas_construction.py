#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._ambient import RegularLevelSetManifold
from ._atlas import CoordinateAtlas
from ._atlas_cover import AtlasCover, AtlasOverlap, ChartSupport
from ._chart import ChartTransition, CoordinateChart


class CompactAtlasDomain(StrictModule):
    """Finite cells representing a declared compact subset.

    Representatives are never promoted to a proof. ``certified_cells`` records
    separately established enclosure evidence for each finite cell.
    """

    representatives: Array
    certified_cells: Array
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        representatives: ArrayLike,
        /,
        *,
        certified_cells: ArrayLike | None = None,
        domain_id: str,
    ):
        points = jnp.asarray(representatives)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError(
                "representatives must have shape (cells, ambient_dimension)."
            )
        if not jnp.issubdtype(points.dtype, jnp.inexact):
            raise TypeError("CompactAtlasDomain coordinates must have an inexact dtype.")
        certified = (
            jnp.zeros((points.shape[0],), dtype=bool)
            if certified_cells is None
            else jnp.asarray(certified_cells, dtype=bool)
        )
        if certified.shape != (points.shape[0],):
            raise ValueError(
                "certified_cells must have one entry per compact-domain cell."
            )
        identifier = str(domain_id)
        if not identifier:
            raise ValueError("domain_id must be non-empty.")
        self.representatives = points
        self.certified_cells = certified
        self.domain_id = identifier

    @property
    def ambient_dimension(self) -> int:
        return self.representatives.shape[1]


class AtlasCandidate(StrictModule):
    """One declared chart parameterization and its ambient inverse."""

    chart: CoordinateChart
    parameterization: Callable[[Array], Array]
    inverse: Callable[[Array], Array]
    coordinate_support: Callable[[Array], Array]
    orientation: int = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart: CoordinateChart,
        parameterization: Callable[[Array], Array],
        inverse: Callable[[Array], Array],
        coordinate_support: Callable[[Array], Array],
        /,
        *,
        candidate_id: str,
        orientation: int = 1,
    ):
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if not all(
            callable(value) for value in (parameterization, inverse, coordinate_support)
        ):
            raise TypeError("Atlas candidate maps and support must be callable.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1):
            raise ValueError("orientation must be +1 or -1.")
        identifier = str(candidate_id)
        if not identifier:
            raise ValueError("candidate_id must be non-empty.")
        self.chart = chart
        self.parameterization = parameterization
        self.inverse = inverse
        self.coordinate_support = coordinate_support
        self.orientation = orientation_
        self.candidate_id = identifier

    def coordinates(self, ambient_points: ArrayLike, /) -> Array:
        points = jnp.asarray(ambient_points)
        result = jnp.asarray(self.inverse(points))
        if result.shape != points.shape[:-1] + (self.chart.dimension,):
            raise ValueError("Atlas candidate inverse returned an incompatible shape.")
        return result

    def covers(self, ambient_points: ArrayLike, /) -> Array:
        coordinates = self.coordinates(ambient_points)
        result = jnp.asarray(self.coordinate_support(coordinates), dtype=bool)
        if result.shape != coordinates.shape[:-1]:
            raise ValueError("Atlas candidate support must preserve leading axes.")
        return result & jnp.all(jnp.isfinite(coordinates), axis=-1)


class AtlasConstructionPolicy(StrictModule):
    maximum_charts: int = eqx.field(static=True)
    maximum_overlaps: int = eqx.field(static=True)
    inverse_tolerance: float = eqx.field(static=True)
    cocycle_tolerance: float = eqx.field(static=True)
    require_certified_cover: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_charts: int = 32,
        maximum_overlaps: int = 256,
        inverse_tolerance: float = 1e-8,
        cocycle_tolerance: float = 1e-8,
        require_certified_cover: bool = False,
    ):
        if int(maximum_charts) < 1 or int(maximum_overlaps) < 0:
            raise ValueError(
                "Atlas capacities must be nonnegative and maximum_charts positive."
            )
        if float(inverse_tolerance) <= 0.0 or float(cocycle_tolerance) <= 0.0:
            raise ValueError("Atlas tolerances must be positive.")
        self.maximum_charts = int(maximum_charts)
        self.maximum_overlaps = int(maximum_overlaps)
        self.inverse_tolerance = float(inverse_tolerance)
        self.cocycle_tolerance = float(cocycle_tolerance)
        self.require_certified_cover = bool(require_certified_cover)


class AtlasConstructionCertificate(StrictModule):
    """Finite evidence; sampled representatives never imply certification."""

    covered_cells: Array
    selected_candidates: Array
    maximum_inverse_residual: Array
    maximum_cocycle_residual: Array
    orientation_consistent: Array
    sampled: Array
    certified: Array
    valid: Array
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        covered_cells: ArrayLike,
        selected_candidates: ArrayLike,
        maximum_inverse_residual: ArrayLike,
        maximum_cocycle_residual: ArrayLike,
        orientation_consistent: ArrayLike,
        sampled: ArrayLike,
        certified: ArrayLike,
        valid: ArrayLike,
        domain_id: str,
    ):
        self.covered_cells = jnp.asarray(covered_cells, dtype=bool)
        self.selected_candidates = jnp.asarray(selected_candidates, dtype=jnp.int32)
        self.maximum_inverse_residual = jnp.asarray(maximum_inverse_residual)
        self.maximum_cocycle_residual = jnp.asarray(maximum_cocycle_residual)
        self.orientation_consistent = jnp.asarray(orientation_consistent, dtype=bool)
        self.sampled = jnp.asarray(sampled, dtype=bool)
        self.certified = jnp.asarray(certified, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.domain_id = str(domain_id)


class PreparedAtlasConstruction(StrictModule):
    """Immutable atlas, cover and evidence selected between smooth epochs."""

    cover: AtlasCover
    certificate: AtlasConstructionCertificate
    path_table: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    candidate_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        cover: AtlasCover,
        certificate: AtlasConstructionCertificate,
        /,
        *,
        path_table: Sequence[Sequence[int]],
        candidate_ids: Sequence[str],
    ):
        self.cover = cover
        self.certificate = certificate
        self.path_table = tuple(
            tuple(int(value) for value in path) for path in path_table
        )
        self.candidate_ids = tuple(str(value) for value in candidate_ids)

    @property
    def atlas(self) -> CoordinateAtlas:
        return self.cover.atlas


def _transition(source: AtlasCandidate, target: AtlasCandidate, /) -> ChartTransition:
    def forward(coordinates: Array) -> Array:
        return target.inverse(source.parameterization(coordinates))

    def backward(coordinates: Array) -> Array:
        return source.inverse(target.parameterization(coordinates))

    return ChartTransition(source.chart, target.chart, forward, inverse=backward)


def prepare_atlas(
    candidates: Sequence[AtlasCandidate],
    compact_domain: CompactAtlasDomain,
    policy: AtlasConstructionPolicy | None = None,
    /,
) -> PreparedAtlasConstruction:
    """Prepare a deterministic finite atlas on a declared compact cell family."""

    candidates_ = tuple(candidates)
    policy_ = AtlasConstructionPolicy() if policy is None else policy
    if not isinstance(compact_domain, CompactAtlasDomain):
        raise TypeError("compact_domain must be a CompactAtlasDomain.")
    if not isinstance(policy_, AtlasConstructionPolicy):
        raise TypeError("policy must be an AtlasConstructionPolicy or None.")
    if not candidates_ or any(
        not isinstance(value, AtlasCandidate) for value in candidates_
    ):
        raise TypeError("candidates must contain AtlasCandidate objects.")
    if len(candidates_) > policy_.maximum_charts:
        raise ValueError("Atlas candidate family exceeds maximum_charts.")
    if len({value.candidate_id for value in candidates_}) != len(candidates_):
        raise ValueError("Atlas candidate identifiers must be unique.")

    points = compact_domain.representatives
    coverage_rows = tuple(candidate.covers(points) for candidate in candidates_)
    coverage = jnp.stack(coverage_rows)
    covered_cells = jnp.any(coverage, axis=0)
    if not bool(jnp.all(covered_cells)):
        raise ValueError(
            "Atlas candidates leave a declared compact-domain cell uncovered."
        )

    selected: list[int] = []
    remaining = jnp.ones_like(covered_cells)
    for index, row in enumerate(coverage_rows):
        if bool(jnp.any(row & remaining)):
            selected.append(index)
            remaining = remaining & ~row
    chosen = tuple(candidates_[index] for index in selected)

    transitions: list[ChartTransition] = []
    overlaps: list[AtlasOverlap] = []
    overlap_samples: dict[tuple[int, int], Array] = {}
    for source_index, target_index in combinations(range(len(chosen)), 2):
        shared = chosen[source_index].covers(points) & chosen[target_index].covers(points)
        if not bool(jnp.any(shared)):
            continue
        for first, second in ((source_index, target_index), (target_index, source_index)):
            transition = _transition(chosen[first], chosen[second])
            transitions.append(transition)
            overlap_samples[(first, second)] = points[shared]
            overlaps.append(
                AtlasOverlap(
                    first,
                    second,
                    transition,
                    lambda coordinates, candidate=chosen[first], other=chosen[second]: (
                        jnp.asarray(candidate.coordinate_support(coordinates), dtype=bool)
                        & other.covers(candidate.parameterization(coordinates))
                    ),
                    overlap_id=f"{chosen[first].candidate_id}->{chosen[second].candidate_id}",
                )
            )
    if len(transitions) > policy_.maximum_overlaps:
        raise ValueError("Discovered atlas graph exceeds maximum_overlaps.")

    atlas = CoordinateAtlas(tuple(value.chart for value in chosen), tuple(transitions))
    supports = tuple(
        ChartSupport(
            value.chart,
            value.coordinate_support,
            support_id=f"{value.candidate_id}:support",
        )
        for value in chosen
    )
    cover = AtlasCover(
        atlas, supports, tuple(overlaps), cover_id=f"atlas:{compact_domain.domain_id}"
    )

    inverse_residual = jnp.asarray(0.0, dtype=points.dtype)
    for (first, second), samples in overlap_samples.items():
        coordinates = chosen[first].coordinates(samples)
        transition = atlas.transition(first, second)
        inverse_residual = jnp.maximum(
            inverse_residual,
            jnp.max(jnp.abs(transition.inverse(transition(coordinates)) - coordinates)),
        )

    cocycle_residual = jnp.asarray(0.0, dtype=points.dtype)
    for first in range(len(chosen)):
        for second in range(len(chosen)):
            for third in range(len(chosen)):
                if len({first, second, third}) != 3:
                    continue
                shared = (
                    coverage_rows[selected[first]]
                    & coverage_rows[selected[second]]
                    & coverage_rows[selected[third]]
                )
                if not bool(jnp.any(shared)):
                    continue
                coordinates = chosen[first].coordinates(points[shared])
                direct = atlas.transition(first, third)(coordinates)
                composed = atlas.transition(second, third)(
                    atlas.transition(first, second)(coordinates)
                )
                cocycle_residual = jnp.maximum(
                    cocycle_residual, jnp.max(jnp.abs(direct - composed))
                )

    orientation_consistent = jnp.asarray(
        len({value.orientation for value in chosen}) == 1
    )
    residual_valid = (inverse_residual <= policy_.inverse_tolerance) & (
        cocycle_residual <= policy_.cocycle_tolerance
    )
    all_cells_certified = jnp.all(compact_domain.certified_cells)
    certified = all_cells_certified & residual_valid & orientation_consistent
    sampled = ~all_cells_certified
    valid = jnp.all(covered_cells) & residual_valid & orientation_consistent
    if policy_.require_certified_cover and not bool(certified):
        raise ValueError(
            "Atlas policy requires cell-certified coverage and transition evidence."
        )
    if not bool(valid):
        raise ValueError("Atlas inverse, cocycle, or orientation validation failed.")

    path_table = []
    for first in range(len(chosen)):
        for second in range(len(chosen)):
            path = atlas.transition_path(first, second)
            indices = [first]
            current = first
            for edge in path:
                current = atlas._chart_index(atlas.charts, edge.target)
                indices.append(current)
            path_table.append(tuple(indices))
    certificate = AtlasConstructionCertificate(
        covered_cells=covered_cells,
        selected_candidates=jnp.asarray(selected, dtype=jnp.int32),
        maximum_inverse_residual=inverse_residual,
        maximum_cocycle_residual=cocycle_residual,
        orientation_consistent=orientation_consistent,
        sampled=sampled,
        certified=certified,
        valid=valid,
        domain_id=compact_domain.domain_id,
    )
    return PreparedAtlasConstruction(
        cover,
        certificate,
        path_table=path_table,
        candidate_ids=tuple(value.candidate_id for value in chosen),
    )


def level_set_graph_candidate(
    manifold: RegularLevelSetManifold,
    free_axes: Sequence[int],
    anchor: ArrayLike,
    /,
    *,
    newton_iterations: int = 8,
    rank_tolerance: float = 1e-8,
    candidate_id: str,
) -> AtlasCandidate:
    """Prepare one IFT graph chart from a fixed nonsingular Jacobian minor."""

    if not isinstance(manifold, RegularLevelSetManifold):
        raise TypeError("manifold must be a RegularLevelSetManifold.")
    free = tuple(int(axis) for axis in free_axes)
    ambient = manifold.point_shape[0]
    expected = ambient - manifold.codimension
    if len(free) != expected or len(set(free)) != len(free):
        raise ValueError("free_axes must select every graph coordinate exactly once.")
    if any(axis < 0 or axis >= ambient for axis in free):
        raise ValueError("free_axes contain an ambient axis outside the manifold.")
    dependent = tuple(axis for axis in range(ambient) if axis not in free)
    anchor_ = jnp.asarray(anchor)
    if anchor_.shape != (ambient,) or int(newton_iterations) < 1:
        raise ValueError("Graph anchor/iteration count are invalid.")
    policy = LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status"))
    free_index = jnp.asarray(free, dtype=jnp.int32)
    dependent_index = jnp.asarray(dependent, dtype=jnp.int32)

    def constraint_values(points: Array) -> Array:
        return (
            manifold.constraint(points)
            if points.ndim == 1
            else jax.vmap(manifold.constraint)(points.reshape((-1, ambient))).reshape(
                points.shape[:-1] + (manifold.codimension,)
            )
        )

    def constraint_jacobian(points: Array) -> Array:
        derivative = jax.jacfwd(manifold.constraint)
        return (
            derivative(points)
            if points.ndim == 1
            else jax.vmap(derivative)(points.reshape((-1, ambient))).reshape(
                points.shape[:-1] + (manifold.codimension, ambient)
            )
        )

    def parameterization(coordinates: Array) -> Array:
        values = jnp.asarray(coordinates, dtype=anchor_.dtype)
        point = jnp.broadcast_to(anchor_, values.shape[:-1] + anchor_.shape)
        point = point.at[..., free_index].set(values)
        for iteration in range(int(newton_iterations)):
            jacobian = constraint_jacobian(point)
            minor = jacobian[..., :, dependent_index]
            residual = constraint_values(point)
            operator = DenseLinearOperator(
                minor,
                operator_id=f"{candidate_id}:minor:{iteration}",
            )
            correction = solve(
                LinearSystem(
                    operator,
                    problem_id=f"{candidate_id}:minor-system:{iteration}",
                ),
                residual,
                policy=policy,
            ).value
            point = point.at[..., dependent_index].add(-correction)
        return point

    def inverse(points: Array) -> Array:
        return jnp.asarray(points)[..., free_index]

    def support(coordinates: Array) -> Array:
        point = parameterization(coordinates)
        jacobian = constraint_jacobian(point)
        minor = jacobian[..., :, dependent_index]
        singular = jnp.linalg.svd(minor, compute_uv=False)
        residual = jnp.max(jnp.abs(constraint_values(point)), axis=-1)
        return (
            jnp.all(jnp.isfinite(point), axis=-1)
            & (jnp.min(singular, axis=-1) > float(rank_tolerance))
            & (residual <= manifold.tolerance)
        )

    chart = CoordinateChart(
        f"{candidate_id}:graph",
        tuple(f"x{axis}" for axis in free),
    )
    return AtlasCandidate(
        chart,
        parameterization,
        inverse,
        support,
        candidate_id=candidate_id,
        orientation=manifold.orientation_sign,
    )


__all__ = [
    "AtlasCandidate",
    "AtlasConstructionCertificate",
    "AtlasConstructionPolicy",
    "CompactAtlasDomain",
    "PreparedAtlasConstruction",
    "level_set_graph_candidate",
    "prepare_atlas",
]
