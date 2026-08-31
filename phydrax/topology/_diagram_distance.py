#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..combinatorial import (
    BipartiteAssignmentSpace,
    CombinatorialCertification,
    HungarianAssignment,
    LinearCombinatorialProblem,
)
from ._diagram import PersistenceDiagram


class DiagramDistanceResult(StrictModule, NonTrainableState):
    """Persistence-specific matching distance and assignment evidence."""

    distance: Array
    assignment: Array
    valid: Array
    method: str = eqx.field(static=True)
    source_diagram_id: str = eqx.field(static=True)
    target_diagram_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        distance: Array,
        assignment: Array,
        valid: Array,
        /,
        *,
        method: str,
        source_diagram_id: str,
        target_diagram_id: str,
    ):
        self.distance = jnp.asarray(distance)
        self.assignment = jnp.asarray(assignment, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.method = str(method)
        self.source_diagram_id = str(source_diagram_id)
        self.target_diagram_id = str(target_diagram_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "diagram-distance-result",
                "method": self.method,
                "source": self.source_diagram_id,
                "target": self.target_diagram_id,
            }
        )


def _point_cost(left_birth, left_death, right_birth, right_death, order):
    difference = np.asarray(
        [abs(left_birth - right_birth), abs(left_death - right_death)]
    )
    return float(
        np.max(difference) if np.isinf(order) else np.linalg.norm(difference, ord=order)
    )


def _diagonal_cost(birth, death, order):
    persistence = abs(death - birth)
    return float(
        persistence / 2.0 if np.isinf(order) else persistence / (2.0 ** (1.0 / order))
    )


def _augmented_cost(
    source: PersistenceDiagram,
    target: PersistenceDiagram,
    /,
    *,
    ground_order: float,
    power: float,
):
    source_degrees = np.asarray(source.degrees)
    target_degrees = np.asarray(target.degrees)
    source_births = np.asarray(source.birth_values)
    target_births = np.asarray(target.birth_values)
    source_deaths = np.asarray(source.death_values)
    target_deaths = np.asarray(target.death_values)
    source_finite = np.asarray(source.has_finite_death)
    target_finite = np.asarray(target.has_finite_death)
    source_count = source.interval_count
    target_count = target.interval_count
    size = source_count + target_count
    costs = np.zeros((size, size), dtype=float)
    valid = np.zeros((size, size), dtype=bool)
    for source_index in range(source_count):
        for target_index in range(target_count):
            compatible = (
                source_degrees[source_index] == target_degrees[target_index]
                and source_finite[source_index] == target_finite[target_index]
            )
            if compatible:
                if source_finite[source_index]:
                    distance = _point_cost(
                        source_births[source_index],
                        source_deaths[source_index],
                        target_births[target_index],
                        target_deaths[target_index],
                        ground_order,
                    )
                else:
                    distance = abs(
                        source_births[source_index] - target_births[target_index]
                    )
                costs[source_index, target_index] = distance**power
                valid[source_index, target_index] = True
        if source_finite[source_index]:
            costs[source_index, target_count + source_index] = (
                _diagonal_cost(
                    source_births[source_index], source_deaths[source_index], ground_order
                )
                ** power
            )
            valid[source_index, target_count + source_index] = True
    for target_index in range(target_count):
        row = source_count + target_index
        if target_finite[target_index]:
            costs[row, target_index] = (
                _diagonal_cost(
                    target_births[target_index], target_deaths[target_index], ground_order
                )
                ** power
            )
            valid[row, target_index] = True
        costs[row, target_count:] = 0.0
        valid[row, target_count:] = True
    return costs, valid


def diagram_wasserstein_distance(
    source: PersistenceDiagram,
    target: PersistenceDiagram,
    /,
    *,
    order: float = 2.0,
    ground_order: float = 2.0,
    certification: CombinatorialCertification | None = None,
) -> DiagramDistanceResult:
    """Compute represented-cost finite-p diagram assignment with diagonal copies."""
    order_ = float(order)
    ground_ = float(ground_order)
    if order_ <= 0.0 or (ground_ <= 0.0 and not np.isinf(ground_)):
        raise ValueError("Diagram distance orders must be positive.")
    costs, valid = _augmented_cost(source, target, ground_order=ground_, power=order_)
    if costs.shape[0] == 0:
        return DiagramDistanceResult(
            jnp.asarray(0.0),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.asarray(True),
            method=f"diagram-wasserstein-{order_}",
            source_diagram_id=source.diagram_id,
            target_diagram_id=target.diagram_id,
        )
    space = BipartiteAssignmentSpace(costs.shape[0], costs.shape[1], valid=valid)
    problem = LinearCombinatorialProblem(
        space,
        jnp.asarray(costs),
        problem_id=f"diagram-assignment:{source.diagram_id}:{target.diagram_id}",
    )
    method = HungarianAssignment(maximum_dimension=max(1, costs.shape[0]))
    resolved_certification = (
        CombinatorialCertification() if certification is None else certification
    )
    plan = method.plan(problem, resolved_certification)
    result = method.solve(problem, plan)
    distance = jnp.where(result.valid, result.objective_value ** (1.0 / order_), jnp.nan)
    return DiagramDistanceResult(
        distance,
        result.decision.columns,
        result.valid,
        method=f"diagram-wasserstein-{order_}",
        source_diagram_id=source.diagram_id,
        target_diagram_id=target.diagram_id,
    )


def _perfect_matching(valid: np.ndarray):
    row_count, column_count = valid.shape
    assigned = np.full((column_count,), -1, dtype=np.int32)

    def augment(row, seen):
        for column in np.flatnonzero(valid[row]):
            if seen[column]:
                continue
            seen[column] = True
            if assigned[column] < 0 or augment(int(assigned[column]), seen):
                assigned[column] = row
                return True
        return False

    for row in range(row_count):
        if not augment(row, np.zeros((column_count,), dtype=bool)):
            return None
    rows = np.full((row_count,), -1, dtype=np.int32)
    for column, row in enumerate(assigned):
        if row >= 0:
            rows[row] = column
    return rows


def diagram_bottleneck_distance(
    source: PersistenceDiagram,
    target: PersistenceDiagram,
    /,
) -> DiagramDistanceResult:
    """Compute exact represented L-infinity bottleneck threshold by matching."""
    costs, base_valid = _augmented_cost(
        source,
        target,
        ground_order=np.inf,
        power=1.0,
    )
    if costs.shape[0] == 0:
        return DiagramDistanceResult(
            jnp.asarray(0.0),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.asarray(True),
            method="diagram-bottleneck",
            source_diagram_id=source.diagram_id,
            target_diagram_id=target.diagram_id,
        )
    candidates = np.unique(costs[np.isfinite(costs) & base_valid])
    assignment = None
    threshold = np.nan
    for candidate in candidates:
        resolved = _perfect_matching(base_valid & (costs <= candidate))
        if resolved is not None:
            assignment = resolved
            threshold = float(candidate)
            break
    valid = assignment is not None
    return DiagramDistanceResult(
        jnp.asarray(threshold),
        jnp.asarray(
            np.full((costs.shape[0],), -1, dtype=np.int32)
            if assignment is None
            else assignment
        ),
        jnp.asarray(valid),
        method="diagram-bottleneck",
        source_diagram_id=source.diagram_id,
        target_diagram_id=target.diagram_id,
    )


def diagram_sliced_wasserstein_distance(
    source: PersistenceDiagram,
    target: PersistenceDiagram,
    /,
    *,
    degree: int,
    num_directions: int = 64,
    order: float = 2.0,
) -> DiagramDistanceResult:
    """Compute a deterministic projected finite-bar Wasserstein estimator."""
    directions = int(num_directions)
    order_ = float(order)
    if directions <= 0 or order_ <= 0.0:
        raise ValueError("Sliced diagram directions and order must be positive.")

    def points(diagram):
        selected = np.asarray(diagram.degrees) == int(degree)
        if np.any(selected & ~np.asarray(diagram.has_finite_death)):
            raise ValueError("Sliced diagram distance requires finite bars.")
        selected &= np.asarray(diagram.has_finite_death)
        return jnp.stack(
            (
                jnp.asarray(np.asarray(diagram.birth_values)[selected]),
                jnp.asarray(np.asarray(diagram.death_values)[selected]),
            ),
            axis=-1,
        )

    left = points(source)
    right = points(target)
    if left.shape[0] + right.shape[0] == 0:
        value = jnp.asarray(0.0)
    else:
        angles = (jnp.arange(directions) + 0.5) * jnp.pi / directions
        vectors = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        diagonal_vector = jnp.asarray([0.5, 0.5])
        left_diagonal = (right @ diagonal_vector)[:, None] * jnp.ones((1, 2))
        right_diagonal = (left @ diagonal_vector)[:, None] * jnp.ones((1, 2))
        augmented_left = jnp.concatenate((left, left_diagonal), axis=0)
        augmented_right = jnp.concatenate((right, right_diagonal), axis=0)
        left_projection = jnp.sort(augmented_left @ vectors.T, axis=0)
        right_projection = jnp.sort(augmented_right @ vectors.T, axis=0)
        per_direction = jnp.mean(
            jnp.abs(left_projection - right_projection) ** order_,
            axis=0,
        ) ** (1.0 / order_)
        value = jnp.mean(per_direction)
    return DiagramDistanceResult(
        value,
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.asarray(True),
        method=f"diagram-sliced-wasserstein-{directions}",
        source_diagram_id=source.diagram_id,
        target_diagram_id=target.diagram_id,
    )


__all__ = [
    "DiagramDistanceResult",
    "diagram_bottleneck_distance",
    "diagram_wasserstein_distance",
    "diagram_sliced_wasserstein_distance",
]
