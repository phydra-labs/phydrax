#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._frozendict import frozendict
from ..._geometry_predicates import PredicateMode, orient2d
from ..._strict import StrictModule
from ...optim._residual_graph import (
    ParameterBlock,
    PreparedResidualGraph,
    ResidualBlock,
    ResidualGraphProblem,
    prepare_residual_graph,
)
from ...typing import checked
from ._schema import DesignState, ParameterBinding, ParameterId, ParameterSchema
from ._sketch import FixedPoint, PointDistance, Sketch, _finite_norm
from ._sketch_certificate import RootDOFEvidence


class DesignParameterization(StrictModule):
    """Scaled reduced coordinates over selected leaves of a geometry design state.

    Coordinates are dimensionless displacements from ``reference``. Each physical
    parameter is scaled by its schema ``physical_scale``; leaves not selected by
    ``parameter_ids`` remain fixed at their reference values.
    """

    reference: DesignState
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    scales: Array
    lower_bounds: Array
    upper_bounds: Array

    @checked
    def __init__(
        self,
        reference: DesignState,
        parameter_ids: Sequence[ParameterId] | None = None,
        /,
    ) -> None:
        identifiers = (
            tuple(spec.parameter_id for spec in reference.schema.specs if spec.trainable)
            if parameter_ids is None
            else tuple(parameter_ids)
        )
        if not identifiers:
            raise ValueError("A design parameterization requires active parameters.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Active design parameter IDs must be unique.")

        bindings: list[ParameterBinding] = []
        offsets = [0]
        scale_parts = []
        lower_parts = []
        upper_parts = []
        for parameter_id in identifiers:
            if not isinstance(parameter_id, ParameterId):
                raise TypeError("parameter_ids must contain ParameterId values.")
            index = reference.schema.index(parameter_id)
            spec = reference.schema.specs[index]
            if not spec.trainable:
                raise ValueError(f"Parameter {parameter_id} is not trainable.")
            size = reference.values[index].size
            if size == 0:
                raise ValueError(f"Parameter {parameter_id} cannot be empty.")
            bindings.append(ParameterBinding(parameter_id, index))
            offsets.append(offsets[-1] + size)
            scale_parts.append(jnp.full((size,), spec.physical_scale, dtype=jnp.float64))
            lower, upper = spec.bounds
            reference_values = jnp.ravel(reference.values[index]).astype("float64")
            lower_parts.append(
                jnp.full((size,), -jnp.inf, dtype=jnp.float64)
                if lower is None
                else (jnp.full((size,), lower, dtype=jnp.float64) - reference_values)
                / spec.physical_scale
            )
            upper_parts.append(
                jnp.full((size,), jnp.inf, dtype=jnp.float64)
                if upper is None
                else (jnp.full((size,), upper, dtype=jnp.float64) - reference_values)
                / spec.physical_scale
            )

        self.reference = reference
        self.bindings = tuple(bindings)
        self.offsets = tuple(offsets)
        self.scales = jnp.concatenate(tuple(scale_parts))
        self.lower_bounds = jnp.concatenate(tuple(lower_parts))
        self.upper_bounds = jnp.concatenate(tuple(upper_parts))

    @property
    def dimension(self) -> int:
        return self.offsets[-1]

    @property
    def parameter_ids(self) -> tuple[ParameterId, ...]:
        return tuple(binding.parameter_id for binding in self.bindings)

    @property
    def initial_coordinates(self) -> Array:
        return jnp.zeros((self.dimension,), dtype=self.scales.dtype)

    def reduce(self, state: DesignState, /) -> Array:
        """Map a full design state to dimensionless reduced coordinates."""
        self._validate_state(state)
        parts = []
        for binding, start, stop in zip(
            self.bindings,
            self.offsets[:-1],
            self.offsets[1:],
            strict=True,
        ):
            delta = jnp.ravel(state.values[binding.index]).astype("float64") - jnp.ravel(
                self.reference.values[binding.index]
            ).astype("float64")
            parts.append(delta / self.scales[start:stop])
        return jnp.concatenate(tuple(parts))

    def expand(self, coordinates: ArrayLike, /) -> DesignState:
        """Map dimensionless reduced coordinates to a full design state."""
        vector = jnp.asarray(coordinates, dtype=self.scales.dtype)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Reduced coordinates must have shape {(self.dimension,)}, got {vector.shape}."
            )
        values = list(self.reference.values)
        for binding, start, stop in zip(
            self.bindings,
            self.offsets[:-1],
            self.offsets[1:],
            strict=True,
        ):
            reference = self.reference.values[binding.index]
            physical = jnp.ravel(reference).astype("float64") + (
                self.scales[start:stop] * vector[start:stop]
            )
            values[binding.index] = physical.reshape(reference.shape).astype(
                reference.dtype
            )
        return DesignState(self.reference.schema, values)

    @checked
    def _validate_state(self, state: DesignState, /) -> None:
        if state.schema != self.reference.schema:
            raise ValueError("Design state and parameterization schemas do not match.")


class DesignBindingGraph(StrictModule):
    """Stable named reads from a design state into an analysis model."""

    schema: ParameterSchema = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        schema: ParameterSchema,
        bindings: Mapping[str, ParameterId | ParameterBinding],
        /,
    ) -> None:
        names = tuple(bindings)
        if not names:
            raise ValueError("A design binding graph requires at least one binding.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Design binding names must be non-empty strings.")
        compiled = []
        for name in names:
            binding = bindings[name]
            if isinstance(binding, ParameterId):
                index = schema.index(binding)
                compiled.append(ParameterBinding(binding, index))
            elif isinstance(binding, ParameterBinding):
                index = schema.index(binding.parameter_id)
                if binding.index != index:
                    raise ValueError(f"Binding {name!r} has an invalid schema index.")
                compiled.append(binding)
            else:
                raise TypeError(
                    "Design binding values must be ParameterId or ParameterBinding."
                )
        self.schema = schema
        self.names = names
        self.bindings = tuple(compiled)

    @checked
    def read(self, state: DesignState, /) -> frozendict[str, Array]:
        """Read every named binding without copying the underlying arrays."""
        if state.schema != self.schema:
            raise ValueError("Design state and binding graph schemas do not match.")
        return frozendict(
            {
                name: binding.read(state)
                for name, binding in zip(self.names, self.bindings, strict=True)
            }
        )


class DesignEvaluation(StrictModule):
    """One reduced-design evaluation with explicit validity evidence."""

    coordinates: Array
    state: DesignState
    bound_values: frozendict[str, Array]
    objective: Array
    constraints: Array
    valid: Array

    def __init__(
        self,
        *,
        coordinates: ArrayLike,
        state: DesignState,
        bound_values: Mapping[str, ArrayLike],
        objective: ArrayLike,
        constraints: ArrayLike,
        valid: ArrayLike,
    ) -> None:
        objective_ = jnp.asarray(objective, dtype=jnp.float64)
        constraints_ = jnp.asarray(constraints, dtype=jnp.float64)
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        if objective_.ndim != 0:
            raise ValueError("A design objective must be scalar.")
        if constraints_.ndim != 1:
            raise ValueError("Design constraints must be a one-dimensional residual.")
        if valid_.ndim != 0:
            raise ValueError("Design validity must be scalar.")
        self.coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
        self.state = state
        self.bound_values = frozendict(
            {name: jnp.asarray(value) for name, value in bound_values.items()}
        )
        self.objective = objective_
        self.constraints = constraints_
        self.valid = valid_


class ReducedDesignProblem(StrictModule):
    """Reduced coordinates and named bindings for an existing objective engine."""

    parameterization: DesignParameterization
    binding_graph: DesignBindingGraph
    objective_fn: Callable[[DesignState, frozendict[str, Array]], ArrayLike] = eqx.field(
        static=True
    )
    constraint_fns: tuple[
        Callable[[DesignState, frozendict[str, Array]], ArrayLike], ...
    ] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parameterization: DesignParameterization,
        binding_graph: DesignBindingGraph,
        objective: Callable[[DesignState, frozendict[str, Array]], ArrayLike],
        /,
        *,
        constraints: Sequence[
            Callable[[DesignState, frozendict[str, Array]], ArrayLike]
        ] = (),
    ) -> None:
        if binding_graph.schema != parameterization.reference.schema:
            raise ValueError("Parameterization and binding graph schemas do not match.")
        if not callable(objective):
            raise TypeError("objective must be callable.")
        constraint_tuple = tuple(constraints)
        if any(not callable(constraint) for constraint in constraint_tuple):
            raise TypeError("Every design constraint must be callable.")
        self.parameterization = parameterization
        self.binding_graph = binding_graph
        self.objective_fn = objective
        self.constraint_fns = constraint_tuple

    def evaluate(self, coordinates: ArrayLike, /) -> DesignEvaluation:
        """Evaluate objective and residuals without choosing an optimizer."""
        state = self.parameterization.expand(coordinates)
        bound_values = self.binding_graph.read(state)
        objective = jnp.asarray(self.objective_fn(state, bound_values), dtype=jnp.float64)
        if objective.ndim != 0:
            raise ValueError("The reduced design objective must return a scalar.")
        if self.constraint_fns:
            residuals = tuple(
                jnp.ravel(jnp.asarray(constraint(state, bound_values), dtype=jnp.float64))
                for constraint in self.constraint_fns
            )
            constraints = jnp.concatenate(residuals)
        else:
            constraints = jnp.zeros((0,), dtype=objective.dtype)
        valid = jnp.isfinite(objective) & jnp.all(jnp.isfinite(constraints))
        return DesignEvaluation(
            coordinates=coordinates,
            state=state,
            bound_values=bound_values,
            objective=objective,
            constraints=constraints,
            valid=valid,
        )

    def objective(self, coordinates: ArrayLike, /) -> Array:
        """Return the scalar objective for use by the existing optimizer stack."""
        return self.evaluate(coordinates).objective


@dataclass(frozen=True)
class _SketchDistanceTriangleWitness:
    """Exact represented-distance discriminant and represented-layout branch sign."""

    point: int
    anchors: tuple[int, int]
    constraint_indices: tuple[int, int, int]
    discriminant: Fraction
    orientation_sign: int


@dataclass(frozen=True)
class _SketchDistanceRigidityWitness:
    """Conditional triangulated rigidity modulo pose, not root/existence evidence."""

    seed_points: tuple[int, int]
    seed_constraint_index: int
    triangles: tuple[_SketchDistanceTriangleWitness, ...]


def _sketch_distance_rigidity_witness(
    indices: tuple[int, ...],
    edges: tuple[tuple[int, int, int, Array, Array], ...],
    reference: np.ndarray,
    /,
) -> _SketchDistanceRigidityWitness | None:
    """Prove a non-tangent construction using exact represented input values.

    The selected equations imply rigidity up to Euclidean pose and the recorded
    discrete orientations, conditional on their simultaneous satisfaction.
    Neither root existence nor satisfaction by the floating reference is claimed.
    """
    pairs: dict[tuple[int, int], list[tuple[int, Fraction]]] = {}
    for index, first, second, target, _ in edges:
        pair = (min(first, second), max(first, second))
        pairs.setdefault(pair, []).append(
            (index, Fraction.from_float(float(np.asarray(target))))
        )
    for seed_index, first, second, _, _ in edges:
        seed = (min(first, second), max(first, second))
        admitted = set(seed)
        triangles: list[_SketchDistanceTriangleWitness] = []
        while len(admitted) < len(indices):
            step = None
            for point in indices:
                if point in admitted:
                    continue
                for a, b in combinations(sorted(admitted), 2):
                    first_pair = (min(a, point), max(a, point))
                    second_pair = (min(b, point), max(b, point))
                    shared_pair = (a, b)
                    if any(
                        pair not in pairs
                        for pair in (first_pair, second_pair, shared_pair)
                    ):
                        continue
                    orientation = orient2d(
                        reference[a],
                        reference[b],
                        reference[point],
                        mode=PredicateMode.EXACT,
                    )
                    if not bool(np.asarray(orientation.certain)):
                        continue
                    sign = int(np.asarray(orientation.signs))
                    if sign not in (-1, 1):
                        continue
                    for first_edge, second_edge, shared_edge in product(
                        pairs[first_pair], pairs[second_pair], pairs[shared_pair]
                    ):
                        first_square = first_edge[1] ** 2
                        second_square = second_edge[1] ** 2
                        shared_square = shared_edge[1] ** 2
                        discriminant = (
                            4 * first_square * second_square
                            - (first_square + second_square - shared_square) ** 2
                        )
                        if discriminant > 0:
                            step = _SketchDistanceTriangleWitness(
                                point,
                                (a, b),
                                (first_edge[0], second_edge[0], shared_edge[0]),
                                discriminant,
                                sign,
                            )
                            break
                    if step is not None:
                        break
                if step is not None:
                    break
            if step is None:
                break
            admitted.add(step.point)
            triangles.append(step)
        if len(admitted) == len(indices):
            return _SketchDistanceRigidityWitness(seed, seed_index, tuple(triangles))
    return None


@dataclass(frozen=True)
class _SketchRigidCluster:
    """One distance-connected reference component; rank is numerical, not proof."""

    point_indices: tuple[int, ...]
    constraint_indices: tuple[int, ...]
    fixed_point_constraint_indices: tuple[int, ...]
    local_points: Array
    pose_seed: Array
    branch_witness: tuple[int, int, int] | None
    branch_sign: int | None
    root_evidence: RootDOFEvidence
    exact_distance_witness: _SketchDistanceRigidityWitness | None
    locally_rigid_indicated: bool

    def points_at_pose(self, pose: ArrayLike, /) -> Array:
        """Apply a declared orientation-preserving pose to the reference seed."""
        pose_ = jnp.asarray(pose, dtype=self.local_points.dtype).reshape((3,))
        cosine, sine = jnp.cos(pose_[2]), jnp.sin(pose_[2])
        rotation = jnp.stack((jnp.stack((cosine, -sine)), jnp.stack((sine, cosine))))
        return self.local_points @ rotation.T + pose_[:2]


@dataclass(frozen=True)
class _SketchRigidClusterPreparation:
    """Host preparation; every original constraint remains in the numerical problem."""

    reference_points: Array
    seed_points: Array
    dimensions: DesignState
    graph: PreparedResidualGraph | None
    clusters: tuple[_SketchRigidCluster, ...]
    retained_constraint_indices: tuple[int, ...]


def _prepare_sketch_rigid_clusters(
    sketch: Sketch,
    reference_points: ArrayLike,
    /,
    *,
    dimensions: DesignState | None = None,
    tolerance: float = 1e-9,
) -> _SketchRigidClusterPreparation:
    """Prepare distance-connected pose seeds, without eliminating any equations.

    Only positive, noncollapsed point-distance factors form the incidence graph.
    Other constraints and arc incidences stay in the original residual. Local
    rank is a pointwise numerical indication, not a rigidity or uniqueness
    certificate. For one fixed-point constraint on a locally rigid component,
    the seed translates its reference layout to that target without rotating it.
    Floating, multiply anchored, and non-rigid components are left unchanged.
    This explicit seed choice neither fixes a gauge nor selects a global branch.
    """
    if not isinstance(sketch, Sketch):
        raise TypeError("sketch must be a Sketch.")
    points = jnp.asarray(reference_points, dtype=sketch.points.dtype)
    host = np.asarray(points)
    if points.shape != sketch.points.shape or not np.all(np.isfinite(host)):
        raise ValueError("reference_points must be a finite layout matching the sketch.")
    state = sketch.dimensions if dimensions is None else dimensions
    if not isinstance(state, DesignState) or state.schema != sketch.dimension_schema:
        raise ValueError("dimensions must match the sketch dimension schema.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ <= 0:
        raise ValueError("tolerance must be finite and positive.")

    bindings = {item.constraint_index: item for item in sketch.dimension_bindings}
    edges: list[tuple[int, int, int, Array, Array]] = []
    for index, constraint in enumerate(sketch.constraints):
        if not isinstance(constraint, PointDistance):
            continue
        first, second = constraint.first_point, constraint.second_point
        binding = bindings.get(index)
        target = (
            constraint.distance if binding is None else state.values[binding.state_index]
        )
        target_host = float(np.asarray(target))
        separation = np.linalg.norm(host[first] - host[second])
        if (
            first == second
            or not isfinite(target_host)
            or target_host <= 0
            or not isfinite(float(separation))
            or separation == 0
        ):
            continue
        edges.append((index, first, second, target, constraint.weight))

    retained = tuple(range(len(sketch.constraints)))
    if not edges:
        return _SketchRigidClusterPreparation(points, points, state, None, (), retained)

    vertices = tuple(
        sorted({point for _, first, second, _, _ in edges for point in (first, second)})
    )
    parameters = tuple(
        ParameterBlock(
            lambda layout, point=point: layout[point],
            lambda layout, value, point=point: layout.at[point].set(value),
            block_id=f"point.{point}",
        )
        for point in vertices
    )

    def factor(target: Array, weight: Array) -> Callable:
        def evaluate(values: tuple[Array, ...], args: object) -> Array:
            del args
            return jnp.atleast_1d(
                jnp.sqrt(weight) * (_finite_norm(values[1] - values[0]) - target)
            )

        return evaluate

    factors = tuple(
        ResidualBlock(
            factor(target, weight),
            (f"point.{first}", f"point.{second}"),
            block_id=f"constraint.{index}",
        )
        for index, first, second, target, weight in edges
    )
    graph = prepare_residual_graph(
        ResidualGraphProblem(parameters, factors, problem_id=sketch._problem_id()), points
    )
    adjacency = np.asarray(graph.adjacency)
    remaining = set(range(len(vertices)))
    components: list[tuple[int, ...]] = []
    while remaining:
        pending = [min(remaining)]
        component: set[int] = set()
        while pending:
            point = pending.pop()
            if point in component:
                continue
            component.add(point)
            incident = np.flatnonzero(adjacency[:, point])
            for factor_index in incident:
                pending.extend(
                    int(value) for value in np.flatnonzero(adjacency[factor_index])
                )
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))

    clusters: list[_SketchRigidCluster] = []
    seed_points = points
    for component_indices in components:
        indices = tuple(vertices[index] for index in component_indices)
        positions = {point: index for index, point in enumerate(indices)}
        selected = tuple(
            edge for edge in edges if edge[1] in positions and edge[2] in positions
        )
        reference = points[jnp.asarray(indices)]

        def component_residual(vector: Array) -> Array:
            layout = vector.reshape((-1, 2))
            return jnp.stack(
                tuple(
                    jnp.sqrt(weight)
                    * (
                        _finite_norm(layout[positions[second]] - layout[positions[first]])
                        - target
                    )
                    for _, first, second, target, weight in selected
                )
            )

        vector = reference.reshape((-1,))
        evidence = RootDOFEvidence(
            sketch._problem_id(),
            vector,
            component_residual(vector),
            jax.jacfwd(component_residual)(vector),
            residual_tolerance=tolerance_,
        )
        witness = None
        sign = None
        for triple in combinations(indices, 3):
            result = orient2d(
                *(host[index] for index in triple), mode=PredicateMode.EXACT
            )
            if bool(np.asarray(result.certain)) and int(np.asarray(result.signs)) != 0:
                witness = triple
                sign = int(np.asarray(result.signs))
                break
        fixed = tuple(
            index
            for index, constraint in enumerate(sketch.constraints)
            if isinstance(constraint, FixedPoint) and constraint.point in positions
        )
        rigid = evidence.root_converged and evidence.jacobian_rank == 2 * len(indices) - 3
        pose_origin = reference[0]
        if rigid and len(fixed) == 1 and (len(indices) == 2 or witness is not None):
            anchor = sketch.constraints[fixed[0]]
            assert isinstance(anchor, FixedPoint)
            displacement = anchor.target - points[anchor.point]
            translated = reference + displacement
            if np.all(np.isfinite(np.asarray(translated))):
                seed_points = seed_points.at[jnp.asarray(indices)].set(translated)
                pose_origin = translated[0]
        clusters.append(
            _SketchRigidCluster(
                indices,
                tuple(edge[0] for edge in selected),
                fixed,
                reference - reference[0],
                jnp.concatenate((pose_origin, jnp.zeros((1,), dtype=reference.dtype))),
                witness,
                sign,
                evidence,
                _sketch_distance_rigidity_witness(indices, selected, host),
                rigid,
            )
        )
    return _SketchRigidClusterPreparation(
        points, seed_points, state, graph, tuple(clusters), retained
    )


__all__ = [
    "DesignBindingGraph",
    "DesignEvaluation",
    "DesignParameterization",
    "ReducedDesignProblem",
]
