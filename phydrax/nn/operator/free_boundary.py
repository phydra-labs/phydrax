#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FreeSurfaceGeometryState, JAXPLICStageReconstruction
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ..layers import warp_jacobian
from .data import FunctionSamples, OperatorBatch, OperatorPrediction
from .task import OperatorTask


FreeBoundaryRepresentation = Literal["reference_map", "level_set", "phase_fraction"]


class FreeBoundaryOperatorSpec(StrictModule, NonTrainableState):
    """Named geometry/state contract layered over an ordinary ``OperatorTask``."""

    representation: FreeBoundaryRepresentation
    geometry_field: str
    state_fields: tuple[str, ...]
    query_name: str
    topology_changes: bool

    def __init__(
        self,
        representation: FreeBoundaryRepresentation,
        geometry_field: str,
        state_fields: Sequence[str],
        query_name: str,
        /,
        *,
        topology_changes: bool = False,
    ):
        if representation not in ("reference_map", "level_set", "phase_fraction"):
            raise ValueError("Unknown free-boundary representation.")
        geometry = str(geometry_field)
        states = tuple(str(name) for name in state_fields)
        query = str(query_name)
        if not geometry or not query or not states or any(not name for name in states):
            raise ValueError("Free-boundary field and query names must be non-empty.")
        if len(set(states)) != len(states) or geometry in states:
            raise ValueError("Geometry and state field names must be distinct.")
        if representation == "reference_map" and topology_changes:
            raise ValueError(
                "A reference-map representation cannot claim topology change."
            )
        self.representation = representation
        self.geometry_field = geometry
        self.state_fields = states
        self.query_name = query
        self.topology_changes = bool(topology_changes)

    def validate_task(self, task: OperatorTask, /) -> None:
        if not isinstance(task, OperatorTask):
            raise TypeError("task must be an OperatorTask.")
        fields = task.field_by_name
        required = (self.geometry_field,) + self.state_fields
        missing = tuple(name for name in required if name not in fields)
        if missing:
            raise KeyError(f"Free-boundary task is missing fields {missing}.")
        if self.query_name not in task.query_by_name:
            raise KeyError(f"Free-boundary task is missing query {self.query_name!r}.")
        for name in required:
            field = fields[name]
            if not field.is_target or field.query_name != self.query_name:
                raise ValueError(
                    f"Free-boundary field {name!r} must target query {self.query_name!r}."
                )
        geometry = fields[self.geometry_field]
        coordinate_dimension = task.query_by_name[self.query_name].coordinate_dimension
        if self.representation == "reference_map":
            if (
                geometry.representation != "vector"
                or geometry.channel_count != coordinate_dimension
            ):
                raise ValueError(
                    "Reference-map geometry must be a vector with one channel per coordinate."
                )
        elif geometry.channels != "scalar":
            raise ValueError("Level-set and phase-fraction geometry must be scalar.")

    def validate_prediction(self, prediction: OperatorPrediction, /) -> None:
        if not isinstance(prediction, OperatorPrediction):
            raise TypeError("prediction must be an OperatorPrediction.")
        required = (self.geometry_field,) + self.state_fields
        missing = tuple(name for name in required if name not in prediction.fields)
        if missing:
            raise KeyError(f"Free-boundary prediction is missing fields {missing}.")
        for name in required:
            if prediction.field(name).query_name != self.query_name:
                raise ValueError(
                    f"Free-boundary prediction field {name!r} uses the wrong query."
                )


class ReferenceMapEvidence(StrictModule):
    jacobian: Array
    determinant: Array
    minimum_singular_value: Array
    orientation_preserving: Array
    nonsingular: Array


class ReferenceMapConstraintLoss(StrictModule):
    nonfolding: Array
    geometric_conservation: Array
    successful: Array

    @property
    def total(self) -> Array:
        return self.nonfolding + self.geometric_conservation


class CorrectedOperatorStep(StrictModule):
    values: Array
    residual_before: Array
    residual_after: Array
    conservation_error: Array
    accepted: Array


class CorrectedOperatorRollout(StrictModule):
    predictions: Array
    corrected: Array
    residual_before: Array
    residual_after: Array
    conservation_error: Array
    accepted: Array


def reference_map_jacobian(
    map_values: ArrayLike,
    reference: FunctionSamples,
    /,
    *,
    boundary: Sequence[Literal["periodic", "reflect", "clamp", "constant"]] | None = None,
) -> Array:
    """Differentiate a tensor-grid physical map with respect to reference axes."""

    if not isinstance(reference, FunctionSamples) or not reference.axes:
        raise TypeError("Reference-map Jacobians require tensor-grid FunctionSamples.")
    values = jnp.asarray(map_values)
    dimension = len(reference.axes)
    if values.shape[-1:] != (dimension,):
        raise ValueError("Map values must end in one component per reference axis.")
    sample_shape = tuple(axis.size for axis in reference.axes)
    if tuple(int(size) for size in values.shape[-dimension - 1 : -1]) != sample_shape:
        raise ValueError("Map values do not match the reference tensor-grid shape.")
    nodes = tuple(axis.nodes for axis in reference.axes)
    lattice = jnp.stack(jnp.meshgrid(*nodes, indexing="ij"), axis=-1)
    case_shape = values.shape[: -dimension - 1]
    lattice = jnp.broadcast_to(lattice, case_shape + sample_shape + (dimension,))
    displacement = values - lattice
    modes = (
        tuple("periodic" if axis.periodic else "clamp" for axis in reference.axes)
        if boundary is None
        else tuple(boundary)
    )
    return warp_jacobian(displacement, boundary=modes, axis_nodes=nodes)


def reference_map_evidence(
    map_values: ArrayLike,
    reference: FunctionSamples,
    /,
    *,
    determinant_floor: float = 1.0e-8,
    singular_value_floor: float = 1.0e-8,
) -> ReferenceMapEvidence:
    """Return orientation and local-invertibility evidence for a reference map."""

    determinant_floor_ = _positive_float(determinant_floor, "determinant_floor")
    singular_floor = _positive_float(singular_value_floor, "singular_value_floor")
    jacobian = reference_map_jacobian(map_values, reference)
    determinant = jnp.linalg.det(jacobian)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    minimum = singular_values[..., -1]
    return ReferenceMapEvidence(
        jacobian=jacobian,
        determinant=determinant,
        minimum_singular_value=minimum,
        orientation_preserving=jnp.all(determinant > determinant_floor_),
        nonsingular=jnp.all(minimum > singular_floor),
    )


def pullback_scalar_gradient(
    reference_gradient: ArrayLike,
    jacobian: ArrayLike,
    /,
    *,
    singular_tolerance: float = 1.0e-12,
    maximum_condition: float = 1.0e12,
) -> Array:
    """Transform a scalar gradient by solving ``J.T grad_x = grad_reference``."""

    matrix = jnp.asarray(jacobian)
    gradient = jnp.asarray(reference_gradient)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("jacobian must end in square matrix axes.")
    dimension = int(matrix.shape[-1])
    plan = SmallLinearSolvePlan(
        dimension,
        singular_tolerance=singular_tolerance,
        maximum_condition=maximum_condition,
    )
    result = solve_small_linear(plan, jnp.swapaxes(matrix, -1, -2), gradient)
    return eqx.error_if(
        result.value,
        jnp.any(~result.successful),
        "Reference-map gradient pullback is singular or ill-conditioned.",
    )


def reference_map_constraint_loss(
    current_map: ArrayLike,
    next_map: ArrayLike,
    reference: FunctionSamples,
    step_size: float,
    /,
    *,
    determinant_floor: float = 1.0e-8,
) -> ReferenceMapConstraintLoss:
    """Penalize map folding and the discrete geometric conservation law."""

    dt = _positive_float(step_size, "step_size")
    floor = _positive_float(determinant_floor, "determinant_floor")
    current_jacobian = reference_map_jacobian(current_map, reference)
    next_jacobian = reference_map_jacobian(next_map, reference)
    current_det = jnp.linalg.det(current_jacobian)
    next_det = jnp.linalg.det(next_jacobian)
    midpoint = 0.5 * (current_jacobian + next_jacobian)
    rate = (next_jacobian - current_jacobian) / dt
    dimension = int(midpoint.shape[-1])
    solve = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        midpoint,
        rate,
    )
    midpoint_det = jnp.linalg.det(midpoint)
    predicted_det_rate = midpoint_det * jnp.trace(solve.value, axis1=-2, axis2=-1)
    observed_det_rate = (next_det - current_det) / dt
    gcl_residual = observed_det_rate - predicted_det_rate
    nonfolding = jnp.mean(
        jax_relu(floor - current_det) ** 2 + jax_relu(floor - next_det) ** 2
    )
    geometric = jnp.mean(gcl_residual**2)
    successful = (
        jnp.all(solve.successful) & jnp.all(current_det > 0.0) & jnp.all(next_det > 0.0)
    )
    return ReferenceMapConstraintLoss(
        nonfolding=nonfolding,
        geometric_conservation=geometric,
        successful=successful,
    )


def operator_batch_from_vof(
    reconstruction: JAXPLICStageReconstruction,
    cell_coordinates: ArrayLike,
    cell_measures: ArrayLike,
    fields: Mapping[str, ArrayLike],
    /,
    *,
    query_name: str = "cells",
) -> OperatorBatch:
    """Adapt certified VOF/PLIC state to measured operator source/query branches."""

    if not isinstance(reconstruction, JAXPLICStageReconstruction):
        raise TypeError("reconstruction must be JAXPLICStageReconstruction.")
    coordinates = _point_coordinates(cell_coordinates, "cell_coordinates")
    measures = _point_weights(cell_measures, coordinates.shape[0], "cell_measures")
    inputs = {
        str(name): FunctionSamples(
            values=_point_values(value, coordinates.shape[0], str(name)),
            coordinates=coordinates,
            quadrature_weights=measures,
            support_id=reconstruction.geometry_id,
        )
        for name, value in fields.items()
    }
    inputs["volume_fraction"] = FunctionSamples(
        values=reconstruction.volume_fraction,
        coordinates=coordinates,
        quadrature_weights=measures,
        support_id=reconstruction.geometry_id,
    )
    interface_values = jnp.concatenate(
        (
            reconstruction.normals,
            reconstruction.interface_measures[:, None],
            reconstruction.interface_evidence[:, None],
        ),
        axis=-1,
    )
    inputs["interface"] = FunctionSamples(
        values=interface_values,
        coordinates=reconstruction.interface_centers,
        quadrature_weights=reconstruction.interface_measures,
        mask=reconstruction.interface_active,
        support_id=f"{reconstruction.geometry_id}:plic-interface",
    )
    queries = {
        str(query_name): FunctionSamples(
            values=None,
            coordinates=coordinates,
            quadrature_weights=measures,
            support_id=reconstruction.geometry_id,
        )
    }
    return OperatorBatch(inputs=inputs, queries=queries)


def operator_batch_from_sph_free_surface(
    geometry: FreeSurfaceGeometryState,
    particle_coordinates: ArrayLike,
    particle_volumes: ArrayLike,
    fields: Mapping[str, ArrayLike],
    /,
    *,
    query_name: str = "particles",
) -> OperatorBatch:
    """Adapt particle and reconstructed free-surface state to operator branches."""

    if not isinstance(geometry, FreeSurfaceGeometryState):
        raise TypeError("geometry must be FreeSurfaceGeometryState.")
    coordinates = _point_coordinates(particle_coordinates, "particle_coordinates")
    volumes = _point_weights(particle_volumes, coordinates.shape[0], "particle_volumes")
    support = "sph-particles"
    inputs = {
        str(name): FunctionSamples(
            values=_point_values(value, coordinates.shape[0], str(name)),
            coordinates=coordinates,
            quadrature_weights=volumes,
            support_id=support,
        )
        for name, value in fields.items()
    }
    surface_values = jnp.concatenate(
        (
            geometry.normal,
            geometry.curvature[:, None],
            geometry.signed_distance[:, None],
            geometry.confidence[:, None],
        ),
        axis=-1,
    )
    inputs["free_surface"] = FunctionSamples(
        values=surface_values,
        coordinates=geometry.surface_point,
        quadrature_weights=volumes * geometry.kernel_volume_fraction,
        mask=geometry.successful,
        support_id="sph-free-surface",
    )
    queries = {
        str(query_name): FunctionSamples(
            values=None,
            coordinates=coordinates,
            quadrature_weights=volumes,
            support_id=support,
        )
    }
    return OperatorBatch(inputs=inputs, queries=queries)


def solver_corrected_operator_rollout(
    model: Callable,
    initial_batch: OperatorBatch,
    steps: int,
    correct: Callable[[Array, OperatorBatch, int], CorrectedOperatorStep],
    advance: Callable[[OperatorBatch, Array, int], OperatorBatch],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> CorrectedOperatorRollout:
    """Roll an operator only through accepted residual-controlled corrections."""

    count = int(steps)
    if count <= 0:
        raise ValueError("steps must be positive.")
    if not isinstance(initial_batch, OperatorBatch):
        raise TypeError("initial_batch must be an OperatorBatch.")
    keys = jr.split(key, count)
    batch = initial_batch
    predictions = []
    corrected_values = []
    before = []
    after = []
    conservation = []
    accepted = []
    for index in range(count):
        prediction = jnp.asarray(model(batch, key=keys[index]))
        correction = correct(prediction, batch, index)
        if not isinstance(correction, CorrectedOperatorStep):
            raise TypeError("correct must return CorrectedOperatorStep.")
        checked = eqx.error_if(
            correction.values,
            ~jnp.all(correction.accepted),
            "Solver-corrected operator step was rejected.",
        )
        checked = eqx.error_if(
            checked,
            jnp.any(correction.residual_after > correction.residual_before),
            "Solver correction increased the declared residual.",
        )
        predictions.append(prediction)
        corrected_values.append(checked)
        before.append(correction.residual_before)
        after.append(correction.residual_after)
        conservation.append(correction.conservation_error)
        accepted.append(correction.accepted)
        if index + 1 < count:
            batch = advance(batch, checked, index)
            if not isinstance(batch, OperatorBatch):
                raise TypeError("advance must return an OperatorBatch.")
    return CorrectedOperatorRollout(
        predictions=jnp.stack(predictions),
        corrected=jnp.stack(corrected_values),
        residual_before=jnp.stack(before),
        residual_after=jnp.stack(after),
        conservation_error=jnp.stack(conservation),
        accepted=jnp.stack(accepted),
    )


def jax_relu(value: ArrayLike, /) -> Array:
    return jnp.maximum(jnp.asarray(value), 0.0)


def _point_coordinates(value: ArrayLike, name: str, /) -> Array:
    coordinates = jnp.asarray(value, dtype=float)
    if coordinates.ndim != 2 or min(coordinates.shape) <= 0:
        raise ValueError(f"{name} must have non-empty shape (point, coordinate).")
    if not bool(jnp.all(jnp.isfinite(coordinates))):
        raise ValueError(f"{name} must be finite.")
    return coordinates


def _point_weights(value: ArrayLike, count: int, name: str, /) -> Array:
    weights = jnp.asarray(value, dtype=float)
    if (
        weights.shape != (count,)
        or not bool(jnp.all(jnp.isfinite(weights)))
        or bool(jnp.any(weights < 0.0))
    ):
        raise ValueError(f"{name} must contain one finite nonnegative value per point.")
    return weights


def _point_values(value: ArrayLike, count: int, name: str, /) -> Array:
    values = jnp.asarray(value)
    if values.ndim < 1 or values.shape[0] != count:
        raise ValueError(f"Operator source {name!r} must start with the point count.")
    return values


def _positive_float(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


__all__ = [
    "CorrectedOperatorRollout",
    "CorrectedOperatorStep",
    "FreeBoundaryOperatorSpec",
    "FreeBoundaryRepresentation",
    "ReferenceMapConstraintLoss",
    "ReferenceMapEvidence",
    "operator_batch_from_sph_free_surface",
    "operator_batch_from_vof",
    "pullback_scalar_gradient",
    "reference_map_constraint_loss",
    "reference_map_evidence",
    "reference_map_jacobian",
    "solver_corrected_operator_rollout",
]
