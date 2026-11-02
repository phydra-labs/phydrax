#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import inspect
from math import isfinite
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint, iter_array_bytes
from ..._geometry_predicates import PredicateMode, orient2d
from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...typing import PRNGKey
from .._contracts import CompiledGeometry, GeometrySource, GeometryTolerance
from ..analytic._operations import BlendCSG, SharpCSG
from ._constraints import AbstractDesignConstraint, DesignConstraintSystem
from ._schema import DesignState, ParameterId, ParameterSchema
from ._sketch import Sketch, SketchPreparationEvidence, SketchSolution
from ._sketch_certificate import (
    LineCircleConstruction,
    RootDOFEvidence,
    TwoCircleConstruction,
)
from ._sketch_dimensions import SketchDimensionSensitivity


class CSGContinuationPolicy(StrictModule):
    """Finite positive nonincreasing blend-width epochs and terminal tolerance."""

    widths: tuple[float, ...] = eqx.field(static=True)
    terminal_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        widths: Sequence[float],
        /,
        *,
        terminal_tolerance: float = 1e-8,
    ) -> None:
        widths_ = tuple(float(value) for value in widths)
        if not widths_ or any(not isfinite(value) or value <= 0.0 for value in widths_):
            raise ValueError("CSG continuation widths must be finite and positive.")
        if any(right > left for left, right in zip(widths_, widths_[1:])):
            raise ValueError("CSG continuation widths must be nonincreasing.")
        tolerance = float(terminal_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("terminal_tolerance must be finite and nonnegative.")
        self.widths = widths_
        self.terminal_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {"kind": "csg-continuation", "widths": widths_, "terminal": tolerance}
        )


class PreparedCSGContinuation(StrictModule):
    """One fixed smooth topology, terminal sharp topology, and exact state maps."""

    policy: CSGContinuationPolicy
    smooth_geometry: CompiledGeometry
    sharp_geometry: CompiledGeometry
    smooth_constraints: tuple[AbstractDesignConstraint, ...]
    sharp_constraints: tuple[AbstractDesignConstraint, ...]
    width_parameter_ids: tuple[ParameterId, ...] = eqx.field(static=True)
    shared_parameter_ids: tuple[ParameterId, ...] = eqx.field(static=True)
    smooth_shared_indices: tuple[int, ...] = eqx.field(static=True)
    sharp_shared_indices: tuple[int, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class CSGContinuationResult(StrictModule):
    """Per-epoch optimization evidence and independent sharp acceptance."""

    state: DesignState
    terminal_state: DesignState
    epoch_results: tuple[Any, ...]
    terminal_residual: Array
    terminal_residual_norm: Array
    sharp_valid: Array
    accepted: Array
    widths: Array
    preparation_id: str = eqx.field(static=True)


class _SmoothBuild:
    def __init__(self) -> None:
        self.width_ids: list[ParameterId] = []

    def convert(self, source: GeometrySource, path: str = "root") -> GeometrySource:
        if isinstance(source, SharpCSG):
            feature_id = f"csg-continuation-{path}"
            width_id = ParameterId(feature_id, "width")
            self.width_ids.append(width_id)
            children = tuple(
                self.convert(child, f"{path}-{index}")
                for index, child in enumerate(source.children)
            )
            return BlendCSG(
                children,
                1.0,
                operation=source.operation,
                feature_id=feature_id,
            )
        if isinstance(source, BlendCSG):
            raise TypeError(
                "Continuation input must be sharp; nested BlendCSG is unsupported."
            )
        if not isinstance(source, GeometrySource):
            raise TypeError("CSG children must be GeometrySource values.")
        return source


def _freeze_widths(
    geometry: CompiledGeometry, width_ids: tuple[ParameterId, ...], /
) -> CompiledGeometry:
    width_set = frozenset(width_ids)
    specs = tuple(
        replace(spec, trainable=False) if spec.parameter_id in width_set else spec
        for spec in geometry.schema.specs
    )
    schema = ParameterSchema(specs)
    state = DesignState(schema, geometry.state.values)
    return CompiledGeometry(geometry.kernel, state, tolerance=geometry.tolerance)


def prepare_csg_continuation(
    source: SharpCSG,
    constraints: Sequence[AbstractDesignConstraint],
    policy: CSGContinuationPolicy,
    /,
    *,
    tolerance: GeometryTolerance = GeometryTolerance(),
) -> PreparedCSGContinuation:
    """Prepare a supported recursive sharp intersection/difference continuation."""
    if not isinstance(source, SharpCSG):
        raise TypeError("source must be a SharpCSG intersection or difference.")
    if not isinstance(policy, CSGContinuationPolicy):
        raise TypeError("policy must be a CSGContinuationPolicy.")
    constraints_ = tuple(constraints)
    if not constraints_ or any(
        not isinstance(value, AbstractDesignConstraint) for value in constraints_
    ):
        raise TypeError("constraints must contain design constraint objects.")
    builder = _SmoothBuild()
    smooth_source = builder.convert(source)
    width_ids = tuple(builder.width_ids)
    if not width_ids:
        raise ValueError("No supported sharp CSG node was found.")
    smooth = _freeze_widths(smooth_source.compile(tolerance=tolerance), width_ids)
    sharp = source.compile(tolerance=tolerance)
    # Construction validates capabilities and every parameter reference on both epochs.
    DesignConstraintSystem(smooth, constraints_)
    DesignConstraintSystem(sharp, constraints_)
    smooth_ids = smooth.schema.parameter_ids
    sharp_ids = sharp.schema.parameter_ids
    shared = tuple(
        parameter_id for parameter_id in sharp_ids if parameter_id in smooth_ids
    )
    if set(shared) != set(sharp_ids):
        missing = set(sharp_ids) - set(shared)
        raise ValueError(
            f"Smooth continuation omits sharp parameters: {sorted(map(str, missing))}."
        )
    smooth_indices = tuple(smooth.schema.index(parameter_id) for parameter_id in shared)
    sharp_indices = tuple(sharp.schema.index(parameter_id) for parameter_id in shared)
    for smooth_index, sharp_index in zip(smooth_indices, sharp_indices, strict=True):
        left, right = smooth.schema.specs[smooth_index], sharp.schema.specs[sharp_index]
        if (left.shape, left.dtype, left.role) != (right.shape, right.dtype, right.role):
            raise ValueError("Shared CSG parameter schemas are not exactly transferable.")
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-csg-continuation",
            "policy": policy.policy_id,
            "width_ids": [str(value) for value in width_ids],
            "shared": [str(value) for value in shared],
        }
    )
    return PreparedCSGContinuation(
        policy,
        smooth,
        sharp,
        constraints_,
        constraints_,
        width_ids,
        shared,
        smooth_indices,
        sharp_indices,
        identifier,
    )


def _with_width(
    prepared: PreparedCSGContinuation, state: DesignState, width: float, /
) -> DesignState:
    updates = {
        parameter_id: jnp.asarray(
            width, dtype=state.values[state.schema.index(parameter_id)].dtype
        )
        for parameter_id in prepared.width_parameter_ids
    }
    return state.updated(updates)


def _terminal_state(
    prepared: PreparedCSGContinuation, smooth_state: DesignState, /
) -> DesignState:
    values = list(prepared.sharp_geometry.state.values)
    for smooth_index, sharp_index in zip(
        prepared.smooth_shared_indices,
        prepared.sharp_shared_indices,
        strict=True,
    ):
        values[sharp_index] = smooth_state.values[smooth_index]
    return DesignState(prepared.sharp_geometry.schema, values)


def solve_csg_continuation(
    prepared: PreparedCSGContinuation,
    /,
    *,
    initial_state: DesignState | None = None,
    search: DifferentialEvolutionSearch | None = None,
    key: PRNGKey | None = None,
    bounds: Mapping[ParameterId, tuple[ArrayLike, ArrayLike]] | None = None,
    solve_options: Mapping[str, Any] | None = None,
) -> CSGContinuationResult:
    """Run fixed-width epochs and audit terminal acceptance on sharp geometry."""
    if not isinstance(prepared, PreparedCSGContinuation):
        raise TypeError("prepared must be a PreparedCSGContinuation.")
    state = prepared.smooth_geometry.state if initial_state is None else initial_state
    if (
        not isinstance(state, DesignState)
        or state.schema != prepared.smooth_geometry.schema
    ):
        raise ValueError("initial_state must match the prepared smooth schema.")
    if search is not None and not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be DifferentialEvolutionSearch or None.")
    if search is not None and key is None:
        raise ValueError("Global continuation search requires an explicit key.")
    options = {} if solve_options is None else dict(solve_options)
    epoch_results: list[Any] = []
    system = DesignConstraintSystem(prepared.smooth_geometry, prepared.smooth_constraints)
    for epoch, width in enumerate(prepared.policy.widths):
        state = _with_width(prepared, state, width)
        if search is None:
            result = system.solve(initial_state=state, **options)
        else:
            # A non-None search was validated above to come with an explicit key.
            search_key = cast(PRNGKey, key)
            result = system.search(
                search,
                key=jr.fold_in(search_key, epoch),
                bounds=bounds,
                initial_state=state,
            )
        epoch_results.append(result)
        state = result.state
    terminal_state = _terminal_state(prepared, state)
    sharp_system = DesignConstraintSystem(
        prepared.sharp_geometry, prepared.sharp_constraints
    )
    residual = sharp_system.residual(terminal_state)
    residual_norm = jnp.linalg.norm(residual)
    validity = prepared.sharp_geometry.validity(terminal_state)
    accepted = (
        validity.accepted
        & jnp.all(jnp.isfinite(residual))
        & (residual_norm <= prepared.policy.terminal_tolerance)
    )
    return CSGContinuationResult(
        state,
        terminal_state,
        tuple(epoch_results),
        residual,
        residual_norm,
        validity.accepted,
        accepted,
        jnp.asarray(prepared.policy.widths),
        prepared.preparation_id,
    )


class SketchBranchWitness(StrictModule):
    """An ordered triple of point indices whose orientation is tracked."""

    points: tuple[int, int, int] = eqx.field(static=True)

    def __init__(self, points: Sequence[int], /) -> None:
        indices = tuple(points)
        if len(indices) != 3 or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("A branch witness requires three nonnegative point indices.")
        if len(set(indices)) != 3:
            raise ValueError("A branch witness requires three distinct points.")
        self.points = indices


def _sketch_witness_signs(
    points: ArrayLike, witnesses: tuple[SketchBranchWitness, ...], /
) -> tuple[int, ...] | None:
    layout = np.asarray(points)
    if not np.all(np.isfinite(layout)):
        return None
    signs: list[int] = []
    for witness in witnesses:
        a, b, c = witness.points
        result = orient2d(layout[a], layout[b], layout[c], mode=PredicateMode.EXACT)
        if not bool(np.asarray(result.certain)):
            return None
        sign = int(np.asarray(result.signs))
        if sign == 0:
            return None
        signs.append(sign)
    return tuple(signs)


def _same_array_bytes(first: np.ndarray, second: np.ndarray, /) -> bool:
    """Compare C-order represented bytes exactly, with bounded temporary storage."""
    left, right = iter_array_bytes(first), iter_array_bytes(second)
    a = b = memoryview(b"")
    while True:
        if not a:
            a = next(left, None)
        if not b:
            b = next(right, None)
        if a is None or b is None:
            return a is None and b is None
        width = min(len(a), len(b))
        if a[:width] != b[:width]:
            return False
        a, b = a[width:], b[width:]


def _same_sketch_tree(first: Any, second: Any, /) -> bool:
    """Compare tree metadata and represented dynamic leaves without tolerance."""
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    if first_tree != second_tree or len(first_leaves) != len(second_leaves):
        return False
    for left, right in zip(first_leaves, second_leaves, strict=True):
        if isinstance(left, (jax.Array, np.ndarray, np.generic)) or isinstance(
            right, (jax.Array, np.ndarray, np.generic)
        ):
            a, b = np.asarray(left), np.asarray(right)
            if a.dtype != b.dtype or a.shape != b.shape or not _same_array_bytes(a, b):
                return False
        elif type(left) is not type(right) or left != right:
            return False
    return True


class _SketchSolveCheckpoint(StrictModule):
    """One authoritative solve input, independent of its accepted output."""

    coordinates: Array
    dimensions: DesignState
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        dimensions: DesignState,
        max_iterations: int,
        tolerance: float,
        damping: float,
        /,
    ) -> None:
        if type(dimensions) is not DesignState:
            raise TypeError("Checkpoint dimensions must be an exact DesignState.")
        DesignState(dimensions.schema, dimensions.values)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("Checkpoint max_iterations must be a positive integer.")
        if (
            not isfinite(tolerance)
            or tolerance <= 0
            or not isfinite(damping)
            or damping <= 0
        ):
            raise ValueError(
                "Checkpoint tolerance and damping must be finite and positive."
            )
        coordinates_ = jnp.asarray(coordinates, dtype=jnp.float64)
        if coordinates_.ndim != 1 or not np.all(np.isfinite(np.asarray(coordinates_))):
            raise ValueError("Checkpoint coordinates must be a finite vector.")
        self.coordinates = coordinates_
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping


class SketchContinuationReference(StrictModule):
    """A solved reference layout and its explicitly declared orientation witnesses.

    Witness signs only distinguish the specified represented point orientations;
    they do not prove global branch uniqueness or exclude an intervening event.
    No gauge or least-motion objective is added to the numerical problem.
    """

    sketch: Sketch
    solution: SketchSolution
    checkpoint: _SketchSolveCheckpoint
    witnesses: tuple[SketchBranchWitness, ...] = eqx.field(static=True)
    witness_signs: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        sketch: Sketch,
        solution: SketchSolution,
        witnesses: Sequence[SketchBranchWitness],
        /,
        *,
        checkpoint: _SketchSolveCheckpoint | None = None,
    ) -> None:
        if not isinstance(sketch, Sketch) or not isinstance(solution, SketchSolution):
            raise TypeError("Expected a Sketch and a SketchSolution.")
        if solution.root_evidence.problem_id != sketch._problem_id():
            raise ValueError("Reference solution belongs to a different sketch.")
        if solution.dimensions.schema != sketch.dimension_schema:
            raise ValueError("Reference dimensions do not match the sketch.")
        if checkpoint is not None:
            if type(checkpoint) is not _SketchSolveCheckpoint:
                raise TypeError("Expected an exact sketch solve checkpoint.")
            checked_checkpoint = _SketchSolveCheckpoint(
                checkpoint.coordinates,
                checkpoint.dimensions,
                checkpoint.max_iterations,
                checkpoint.tolerance,
                checkpoint.damping,
            )
            if not _same_sketch_tree(checkpoint, checked_checkpoint):
                raise ValueError("Checkpoint representation is not canonical.")
            if checkpoint.dimensions.schema != sketch.dimension_schema or (
                checkpoint.coordinates.shape
                != (sketch.points.size + sketch.circle_radii.size,)
            ):
                raise ValueError(
                    "Checkpoint dimensions or coordinate shape do not match the sketch."
                )
            if (
                not _same_sketch_tree(
                    checkpoint.coordinates, solution.preparation.reference_coordinates
                )
                or not _same_sketch_tree(checkpoint.dimensions, solution.dimensions)
                or checkpoint.max_iterations != solution.preparation.max_iterations
                or checkpoint.damping != solution.preparation.damping
                or checkpoint.tolerance != solution.root_evidence.residual_tolerance
            ):
                raise ValueError(
                    "Checkpoint does not match the solution preparation inputs."
                )
        solution.preparation.verify(
            sketch,
            solution.dimensions,
            tolerance=solution.root_evidence.residual_tolerance,
        )
        if solution.points.shape != sketch.points.shape or (
            solution.circle_radii.shape != sketch.circle_radii.shape
        ):
            raise ValueError("Reference layout does not match the sketch.")
        if (
            not bool(np.asarray(solution.converged))
            or not solution.root_evidence.root_converged
        ):
            raise ValueError("Reference must be a converged numerical root.")
        radii = np.asarray(solution.circle_radii)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError("Reference radii must be finite and positive.")
        coordinates = jnp.concatenate(
            (solution.points.reshape((-1,)), solution.circle_radii)
        )
        point_size = solution.points.size

        def residual(vector: Array) -> Array:
            points = vector[:point_size].reshape(sketch.points.shape)
            return sketch.residual(
                points, vector[point_size:], dimensions=solution.dimensions
            )

        values = residual(coordinates)
        evidence = RootDOFEvidence(
            sketch._problem_id(),
            coordinates,
            values,
            jax.jacfwd(residual)(coordinates),
            residual_tolerance=solution.root_evidence.residual_tolerance,
        )
        recorded = solution.root_evidence
        if not evidence.root_converged:
            raise ValueError("Reference layout is not a converged numerical root.")
        if (
            evidence.evidence_id != recorded.evidence_id
            or evidence.jacobian_rank != recorded.jacobian_rank
            or evidence.regular_root_indicated != recorded.regular_root_indicated
        ):
            raise ValueError(
                "Reference root evidence does not match the supplied sketch and layout."
            )
        if not np.array_equal(np.asarray(solution.residual), np.asarray(values)):
            raise ValueError(
                "Reference residual does not match the supplied sketch and layout."
            )
        witnesses_ = tuple(witnesses)
        if any(not isinstance(item, SketchBranchWitness) for item in witnesses_):
            raise TypeError("witnesses must contain SketchBranchWitness objects.")
        if any(max(item.points) >= sketch.points.shape[0] for item in witnesses_):
            raise ValueError("A branch witness references an absent point.")
        signs = _sketch_witness_signs(solution.points, witnesses_)
        if signs is None:
            raise ValueError(
                "Reference witnesses must have resolved nonzero orientations."
            )
        self.sketch = sketch
        self.solution = solution
        self.checkpoint = (
            checkpoint
            if checkpoint is not None
            else _SketchSolveCheckpoint(
                solution.preparation.reference_coordinates,
                solution.dimensions,
                solution.preparation.max_iterations,
                solution.root_evidence.residual_tolerance,
                solution.preparation.damping,
            )
        )
        self.witnesses = witnesses_
        self.witness_signs = signs


_SKETCH_ARCHIVE_CONSTRAINT_NAMES = (
    "Coincident",
    "FixedPoint",
    "PointDistance",
    "Horizontal",
    "Vertical",
    "Parallel",
    "Perpendicular",
    "EqualLength",
    "LineAngle",
    "Midpoint",
    "PointOnLine",
    "Radius",
    "TangentLineCircle",
    "TangentCircles",
)


class SketchContinuationReferenceArchiveRecipe(NamedTuple):
    """Authoritative sketch inputs and accepted layout, replayed on restoration."""

    points: Any
    lines: Any
    circle_centers: Any
    circle_radii: Any
    arc_endpoints: Any
    arc_centers: Any
    arc_counterclockwise: Any
    constraints: tuple[Any, ...]
    feature_id: str
    constraint_ids: tuple[str, ...]
    constraint_priorities: tuple[int, ...]
    dimensions: tuple[Any, ...]
    accepted_points: Any
    accepted_radii: Any
    preparation_reference: Any
    preparation_seed: Any
    preparation_pinned_indices: tuple[int, ...]
    preparation_pinned_values: Any
    preparation_constructions: tuple[Any, ...]
    preparation_rigid_clusters: tuple[Any, ...]
    preparation_rigid_reduction_points: tuple[tuple[int, ...], ...]
    preparation_max_iterations: int
    preparation_damping: float
    preparation_reduced_root_accepted: bool
    residual_tolerance: float
    evidence_id: str
    iterations: int
    status: int
    witnesses: tuple[tuple[int, int, int], ...]
    witness_signs: tuple[int, ...]
    checkpoint_coordinates: Any
    checkpoint_dimensions: tuple[Any, ...]
    checkpoint_max_iterations: int
    checkpoint_tolerance: float
    checkpoint_damping: float

    @classmethod
    def from_reference(
        cls, reference: SketchContinuationReference, /
    ) -> SketchContinuationReferenceArchiveRecipe:
        from . import _sketch as sketch_module

        if type(reference) is not SketchContinuationReference:
            raise TypeError("Expected an exact SketchContinuationReference.")
        sketch = reference.sketch
        checked = SketchContinuationReference(
            sketch,
            reference.solution,
            reference.witnesses,
            checkpoint=reference.checkpoint,
        )
        if checked.witness_signs != reference.witness_signs:
            raise ValueError("Reference witness evidence does not match the layout.")
        constraints = []
        for constraint in sketch.constraints:
            name = type(constraint).__name__
            if name not in _SKETCH_ARCHIVE_CONSTRAINT_NAMES or type(
                constraint
            ) is not getattr(sketch_module, name):
                raise TypeError(
                    f"No sketch archive recipe for constraint {type(constraint)!r}."
                )
            fields = {}
            for field in inspect.signature(type(constraint).__init__).parameters:
                if field == "self":
                    continue
                value = getattr(constraint, field)
                if isinstance(value, ParameterId):
                    value = (value.feature_id, value.name)
                elif isinstance(value, (jax.Array, np.ndarray)):
                    value = np.asarray(value)
                fields[field] = value
            constraints.append((name, fields))
        solution = reference.solution
        return cls(
            np.asarray(sketch.points),
            np.asarray(sketch.lines),
            np.asarray(sketch.circle_centers),
            np.asarray(sketch.circle_radii),
            np.asarray(sketch.arc_endpoints),
            np.asarray(sketch.arc_centers),
            np.asarray(sketch.arc_counterclockwise),
            tuple(constraints),
            sketch.feature_id,
            sketch.constraint_ids,
            sketch.constraint_priorities,
            tuple(np.asarray(value) for value in solution.dimensions.values),
            np.asarray(solution.points),
            np.asarray(solution.circle_radii),
            np.asarray(solution.preparation.reference_coordinates),
            np.asarray(solution.preparation.seed_coordinates),
            solution.preparation.pinned_indices,
            np.asarray(solution.preparation.pinned_values),
            solution.preparation.constructions,
            solution.preparation.rigid_clusters,
            solution.preparation.rigid_reduction_points,
            solution.preparation.max_iterations,
            solution.preparation.damping,
            solution.preparation.reduced_root_accepted,
            solution.root_evidence.residual_tolerance,
            solution.root_evidence.evidence_id,
            int(np.asarray(solution.iterations)),
            int(np.asarray(solution.status)),
            tuple(witness.points for witness in reference.witnesses),
            reference.witness_signs,
            np.asarray(reference.checkpoint.coordinates),
            tuple(np.asarray(value) for value in reference.checkpoint.dimensions.values),
            reference.checkpoint.max_iterations,
            reference.checkpoint.tolerance,
            reference.checkpoint.damping,
        )

    def restore(self, /) -> SketchContinuationReference:
        from . import _sketch as sketch_module
        from ._reduced import (
            _SketchDistanceRigidityWitness,
            _SketchDistanceTriangleWitness,
        )

        constraints = []
        for name, stored_fields in self.constraints:
            if name not in _SKETCH_ARCHIVE_CONSTRAINT_NAMES:
                raise ValueError(f"Unknown archived sketch constraint {name!r}.")
            fields = dict(stored_fields)
            if fields.get("parameter_id") is not None:
                fields["parameter_id"] = ParameterId(*fields["parameter_id"])
            constraints.append(getattr(sketch_module, name)(**fields))
        sketch = Sketch(
            self.points,
            lines=self.lines,
            circle_centers=self.circle_centers,
            circle_radii=self.circle_radii,
            arc_endpoints=self.arc_endpoints,
            arc_centers=self.arc_centers,
            arc_counterclockwise=self.arc_counterclockwise,
            constraints=constraints,
            feature_id=self.feature_id,
            constraint_ids=self.constraint_ids,
            constraint_priorities=self.constraint_priorities,
        )
        dimensions = DesignState(sketch.dimension_schema, self.dimensions)
        points = jnp.asarray(self.accepted_points, dtype=jnp.float64)
        radii = jnp.asarray(self.accepted_radii, dtype=jnp.float64)
        coordinates = jnp.concatenate((points.reshape((-1,)), radii))
        point_size = points.size

        def residual(vector: Array) -> Array:
            return sketch.residual(
                vector[:point_size].reshape(sketch.points.shape),
                vector[point_size:],
                dimensions=dimensions,
            )

        values = residual(coordinates)
        evidence = RootDOFEvidence(
            sketch._problem_id(),
            coordinates,
            values,
            jax.jacfwd(residual)(coordinates),
            residual_tolerance=self.residual_tolerance,
        )
        if evidence.evidence_id != self.evidence_id or not evidence.root_converged:
            raise ValueError("Archived sketch root evidence failed replay.")
        if any(
            type(witness) not in (TwoCircleConstruction, LineCircleConstruction)
            for _, witness in self.preparation_constructions
        ):
            raise ValueError("Unknown archived sketch construction type.")
        if any(
            type(witness) is not _SketchDistanceRigidityWitness
            or any(
                type(triangle) is not _SketchDistanceTriangleWitness
                for triangle in witness.triangles
            )
            for witness in self.preparation_rigid_clusters
        ):
            raise ValueError("Unknown archived rigid sketch witness type.")
        preparation = SketchPreparationEvidence(
            sketch._problem_id(),
            self.preparation_reference,
            self.preparation_seed,
            self.preparation_pinned_indices,
            self.preparation_pinned_values,
            self.preparation_constructions,
            self.preparation_rigid_clusters,
            self.preparation_rigid_reduction_points,
            self.preparation_max_iterations,
            self.preparation_damping,
            self.preparation_reduced_root_accepted,
        )
        solution = SketchSolution(
            points=points,
            circle_radii=radii,
            residual=values,
            residual_norm=jnp.linalg.norm(values),
            converged=True,
            iterations=self.iterations,
            status=self.status,
            root_evidence=evidence,
            preparation=preparation,
            dimensions=dimensions,
        )
        checkpoint = _SketchSolveCheckpoint(
            self.checkpoint_coordinates,
            DesignState(sketch.dimension_schema, self.checkpoint_dimensions),
            self.checkpoint_max_iterations,
            self.checkpoint_tolerance,
            self.checkpoint_damping,
        )
        replayed = sketch.solve(
            initial_coordinates=checkpoint.coordinates,
            dimensions=checkpoint.dimensions,
            max_iterations=checkpoint.max_iterations,
            tolerance=checkpoint.tolerance,
            damping=checkpoint.damping,
        )
        if not _same_sketch_tree(solution, replayed):
            raise ValueError("Archived sketch output does not match checkpoint replay.")
        reference = SketchContinuationReference(
            sketch,
            solution,
            tuple(SketchBranchWitness(value) for value in self.witnesses),
            checkpoint=checkpoint,
        )
        if reference.witness_signs != self.witness_signs:
            raise ValueError("Archived sketch branch evidence failed replay.")
        return reference


class SketchContinuationResult(StrictModule):
    """Attempted root and accepted reference, kept separate for rollback."""

    reference: SketchContinuationReference
    attempt: SketchSolution
    prior_reference: SketchContinuationReference
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def _validate_transition(self, sketch: Sketch, /) -> None:
        """Replay accepted endpoints before any transition derivative."""
        if not self.accepted or self.reason != "accepted":
            raise ValueError("Continuation sensitivity requires an accepted transition.")
        if not _same_sketch_tree(
            sketch, self.prior_reference.sketch
        ) or not _same_sketch_tree(sketch, self.reference.sketch):
            raise ValueError(
                "Continuation reference belongs to a different sketch source."
            )
        prior = SketchContinuationReference(
            sketch,
            self.prior_reference.solution,
            self.prior_reference.witnesses,
            checkpoint=self.prior_reference.checkpoint,
        )
        prior_coordinates = jnp.concatenate(
            (prior.solution.points.reshape((-1,)), prior.solution.circle_radii)
        )
        if not _same_sketch_tree(
            self.reference.checkpoint.coordinates, prior_coordinates
        ):
            raise ValueError(
                "Continuation checkpoint does not match the prior reference layout."
            )
        current = SketchContinuationReference(
            sketch,
            self.reference.solution,
            self.reference.witnesses,
            checkpoint=self.reference.checkpoint,
        )
        attempted = SketchContinuationReference(
            sketch, self.attempt, prior.witnesses, checkpoint=self.reference.checkpoint
        )
        prior_points = tuple(item.points for item in prior.witnesses)
        current_points = tuple(item.points for item in current.witnesses)
        if (
            prior_points != current_points
            or prior.witness_signs != self.prior_reference.witness_signs
            or current.witness_signs != self.reference.witness_signs
            or prior.witness_signs != current.witness_signs
            or attempted.witness_signs != current.witness_signs
        ):
            raise ValueError(
                "Continuation witness evidence does not match the transition."
            )
        old_evidence = prior.solution.root_evidence
        new_evidence = attempted.solution.root_evidence
        if (
            old_evidence.jacobian_rank != new_evidence.jacobian_rank
            or old_evidence.regular_root_indicated != new_evidence.regular_root_indicated
        ):
            raise ValueError(
                "Continuation sensitivity requires unchanged rank and regularity."
            )
        current_solution = current.solution
        if (
            current_solution.root_evidence.evidence_id != new_evidence.evidence_id
            or current_solution.dimensions.schema != self.attempt.dimensions.schema
            or any(
                not np.array_equal(np.asarray(first), np.asarray(second))
                for first, second in zip(
                    current_solution.dimensions.values,
                    self.attempt.dimensions.values,
                    strict=True,
                )
            )
            or not np.array_equal(
                np.asarray(current_solution.points), np.asarray(self.attempt.points)
            )
            or not np.array_equal(
                np.asarray(current_solution.circle_radii),
                np.asarray(self.attempt.circle_radii),
            )
        ):
            raise ValueError("Accepted reference does not match the attempted root.")
        checkpoint = self.reference.checkpoint
        replayed = sketch.solve(
            initial_coordinates=checkpoint.coordinates,
            dimensions=checkpoint.dimensions,
            max_iterations=checkpoint.max_iterations,
            tolerance=checkpoint.tolerance,
            damping=checkpoint.damping,
        )
        if not _same_sketch_tree(current_solution, replayed):
            raise ValueError("Accepted transition does not match checkpoint replay.")

    def dimension_sensitivity(
        self, sketch: Sketch, parameter_id: ParameterId, /
    ) -> SketchDimensionSensitivity:
        """Pointwise sensitivity after a replayed endpoint transition, not a path proof."""
        self._validate_transition(sketch)
        return sketch.dimension_sensitivity(self.attempt, parameter_id)

    def dimension_jvp(
        self, sketch: Sketch, tangents: Mapping[ParameterId, ArrayLike], /
    ) -> Array:
        """Apply the accepted fixed-branch dimension-to-coordinate derivative."""
        self._validate_transition(sketch)
        result = jnp.zeros(
            (self.attempt.points.size + self.attempt.circle_radii.size,),
            dtype=self.attempt.points.dtype,
        )
        for parameter_id, tangent in tangents.items():
            value = jnp.asarray(tangent, dtype=result.dtype)
            if value.shape != ():
                raise ValueError("Sketch dimension tangents must be scalar.")
            derivative = sketch.dimension_sensitivity(self.attempt, parameter_id)
            result = result + value * derivative.coordinates
        return result

    def dimension_vjp(
        self,
        sketch: Sketch,
        cotangent: ArrayLike,
        parameter_ids: Sequence[ParameterId],
        /,
    ) -> dict[ParameterId, Array]:
        """Apply the transpose fixed-branch derivative to a coordinate cotangent."""
        self._validate_transition(sketch)
        vector = jnp.asarray(cotangent, dtype=self.attempt.points.dtype)
        size = self.attempt.points.size + self.attempt.circle_radii.size
        if vector.shape != (size,):
            raise ValueError(f"Coordinate cotangent must have shape {(size,)}.")
        identifiers = tuple(parameter_ids)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Dimension parameter IDs must be unique.")
        return {
            parameter_id: jnp.vdot(
                sketch.dimension_sensitivity(self.attempt, parameter_id).coordinates,
                vector,
            )
            for parameter_id in identifiers
        }


def continue_sketch(
    sketch: Sketch,
    reference: SketchContinuationReference,
    dimensions: DesignState,
    /,
    *,
    solve_options: Mapping[str, Any] | None = None,
) -> SketchContinuationResult:
    """Solve an edit from a reference, retaining it unchanged on rejection.

    Only endpoint witness signs and numerical rank/regularity are compared.
    Acceptance is not a proof that the path between layouts was event-free.
    This host-side operation can be called independently from any saved reference.
    """
    if not isinstance(sketch, Sketch) or not isinstance(
        reference, SketchContinuationReference
    ):
        raise TypeError("Expected a Sketch and SketchContinuationReference.")
    if not _same_sketch_tree(sketch, reference.sketch):
        raise ValueError("Reference solution belongs to a different sketch source.")
    if reference.solution.root_evidence.problem_id != sketch._problem_id():
        raise ValueError("Reference solution belongs to a different sketch.")
    checked = SketchContinuationReference(
        sketch, reference.solution, reference.witnesses, checkpoint=reference.checkpoint
    )
    if checked.witness_signs != reference.witness_signs:
        raise ValueError("Reference witness evidence does not match the layout.")
    if (
        not isinstance(dimensions, DesignState)
        or dimensions.schema != sketch.dimension_schema
    ):
        raise ValueError("dimensions must match the sketch dimension schema.")
    options = {} if solve_options is None else dict(solve_options)
    if "initial_coordinates" in options or "dimensions" in options:
        raise ValueError("Continuation controls initial_coordinates and dimensions.")
    initial = jnp.concatenate(
        (reference.solution.points.reshape((-1,)), reference.solution.circle_radii)
    )
    attempt = sketch.solve(initial_coordinates=initial, dimensions=dimensions, **options)
    evidence = attempt.root_evidence
    previous = reference.solution.root_evidence
    if not bool(np.asarray(attempt.converged)) or not evidence.root_converged:
        reason = "not_converged"
    elif (
        evidence.jacobian_rank != previous.jacobian_rank
        or evidence.regular_root_indicated != previous.regular_root_indicated
    ):
        reason = "rank_changed"
    else:
        signs = _sketch_witness_signs(attempt.points, reference.witnesses)
        if signs is None:
            reason = "witness_unresolved"
        elif signs != reference.witness_signs:
            reason = "witness_changed"
        else:
            reason = "accepted"
    next_reference = reference
    if reason == "accepted":
        try:
            next_reference = SketchContinuationReference(
                sketch,
                attempt,
                reference.witnesses,
                checkpoint=_SketchSolveCheckpoint(
                    initial,
                    dimensions,
                    options.get("max_iterations", 32),
                    options.get("tolerance", 1e-9),
                    options.get("damping", 1e-8),
                ),
            )
        except ValueError:
            reason = "invalid_attempt"
    accepted = reason == "accepted"
    return SketchContinuationResult(next_reference, attempt, reference, accepted, reason)


__all__ = [
    "CSGContinuationPolicy",
    "CSGContinuationResult",
    "PreparedCSGContinuation",
    "SketchBranchWitness",
    "SketchContinuationReference",
    "SketchContinuationResult",
    "continue_sketch",
    "prepare_csg_continuation",
    "solve_csg_continuation",
]
