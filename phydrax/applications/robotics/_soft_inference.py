#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Calibration, fixed-mode co-design, and sampling MPC for reduced soft robots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...control._sampling_mpc import (
    plan_sampling_mpc,
    SamplingMPCPlan,
    SamplingMPCRealizationBinding,
    SamplingMPCRealizations,
    SamplingMPCResult,
    SamplingMPCSelectedReplay,
)
from ...optim import (
    MinimizationProblem,
    NonlinearConstraint,
    ParameterBlock,
    ResidualBlock,
    ResidualGraphProblem,
    solve_residual_graph,
    SQP,
    StateDesignConstraint,
    StateDesignProblem,
)


CalibrationSplit: TypeAlias = Literal["train", "validation", "held_out"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _softplus_inverse(value: Array, /) -> Array:
    return value + jnp.log(-jnp.expm1(-value))


def _tree_finite(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves))
    )


class PositiveParameterMap(StrictModule, NonTrainableState):
    """Smooth latent coordinates for one positive scalar or array parameter."""

    minimum: float = eqx.field(static=True)
    maximum: float | None = eqx.field(static=True)
    name: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        minimum: float = 0.0,
        maximum: float | None = None,
    ):
        lower = float(minimum)
        upper = None if maximum is None else float(maximum)
        if not isfinite(lower) or lower < 0.0:
            raise ValueError(
                "PositiveParameterMap minimum must be finite and non-negative."
            )
        if upper is not None and (not isfinite(upper) or upper <= lower):
            raise ValueError("PositiveParameterMap maximum must exceed minimum.")
        identifier = _identifier(name, "PositiveParameterMap name")
        self.minimum = lower
        self.maximum = upper
        self.name = identifier
        self.map_id = canonical_fingerprint(
            {
                "kind": "positive-parameter-map",
                "name": identifier,
                "minimum": lower,
                "maximum": upper,
            }
        )

    def to_physical(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        if self.maximum is None:
            return self.minimum + jax.nn.softplus(value)
        return self.minimum + (self.maximum - self.minimum) * jax.nn.sigmoid(value)

    def to_latent(self, physical: ArrayLike, /) -> Array:
        value = jnp.asarray(physical)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        if self.maximum is None:
            shifted = eqx.error_if(
                value - self.minimum,
                jnp.any(~jnp.isfinite(value) | (value <= self.minimum)),
                f"Physical parameter {self.name} must exceed its minimum.",
            )
            return _softplus_inverse(shifted)
        unit = (value - self.minimum) / (self.maximum - self.minimum)
        unit = eqx.error_if(
            unit,
            jnp.any(~jnp.isfinite(unit) | (unit <= 0.0) | (unit >= 1.0)),
            f"Physical parameter {self.name} must lie strictly inside its bounds.",
        )
        return jnp.log(unit) - jnp.log1p(-unit)


class BoundedParameterMap(StrictModule, NonTrainableState):
    """Smooth latent coordinates for a finite open physical interval."""

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(self, name: str, lower: float, upper: float, /):
        lower_ = float(lower)
        upper_ = float(upper)
        if not isfinite(lower_) or not isfinite(upper_) or upper_ <= lower_:
            raise ValueError("BoundedParameterMap requires finite increasing bounds.")
        identifier = _identifier(name, "BoundedParameterMap name")
        self.lower = lower_
        self.upper = upper_
        self.name = identifier
        self.map_id = canonical_fingerprint(
            {
                "kind": "bounded-parameter-map",
                "name": identifier,
                "lower": lower_,
                "upper": upper_,
            }
        )

    def to_physical(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(value)

    def to_latent(self, physical: ArrayLike, /) -> Array:
        value = jnp.asarray(physical)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        unit = (value - self.lower) / (self.upper - self.lower)
        unit = eqx.error_if(
            unit,
            jnp.any(~jnp.isfinite(unit) | (unit <= 0.0) | (unit >= 1.0)),
            f"Physical parameter {self.name} must lie strictly inside its bounds.",
        )
        return jnp.log(unit) - jnp.log1p(-unit)


class SPDParameterMap(StrictModule, NonTrainableState):
    """Packed Cholesky coordinates for one symmetric positive-definite matrix."""

    dimension: int = eqx.field(static=True)
    diagonal_floor: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        dimension: int,
        /,
        *,
        diagonal_floor: float = 1e-10,
    ):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("SPDParameterMap dimension must be a positive integer.")
        floor = float(diagonal_floor)
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("SPDParameterMap diagonal_floor must be positive.")
        identifier = _identifier(name, "SPDParameterMap name")
        self.dimension = dimension
        self.diagonal_floor = floor
        self.name = identifier
        self.map_id = canonical_fingerprint(
            {
                "kind": "spd-parameter-map",
                "name": identifier,
                "dimension": dimension,
                "diagonal_floor": floor,
            }
        )

    @property
    def coordinate_size(self) -> int:
        return self.dimension * (self.dimension + 1) // 2

    def to_physical(self, latent: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(latent)
        if coordinates.shape != (self.coordinate_size,):
            raise ValueError(
                f"SPD latent coordinates must have shape {(self.coordinate_size,)}."
            )
        if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
            coordinates = coordinates.astype(float)
        rows, columns = np.tril_indices(self.dimension)
        diagonal = np.flatnonzero(rows == columns)
        factor_coordinates = coordinates.at[diagonal].set(
            self.diagonal_floor + jax.nn.softplus(coordinates[diagonal])
        )
        factor = (
            jnp.zeros((self.dimension, self.dimension), dtype=coordinates.dtype)
            .at[rows, columns]
            .set(factor_coordinates)
        )
        return factor @ factor.T

    def to_latent(self, physical: ArrayLike, /) -> Array:
        matrix = jnp.asarray(physical)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"SPD physical matrix must have shape {(self.dimension, self.dimension)}."
            )
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = matrix.astype(float)
        valid = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.allclose(matrix, matrix.T)
            & jnp.all(jnp.linalg.eigvalsh(matrix) > 0.0)
        )
        matrix = eqx.error_if(
            matrix,
            ~valid,
            f"Physical parameter {self.name} must be finite, symmetric, and SPD.",
        )
        factor = jnp.linalg.cholesky(matrix)
        rows, columns = np.tril_indices(self.dimension)
        packed = factor[rows, columns]
        diagonal = np.flatnonzero(rows == columns)
        shifted = eqx.error_if(
            packed[diagonal] - self.diagonal_floor,
            jnp.any(packed[diagonal] <= self.diagonal_floor),
            f"Physical parameter {self.name} violates its Cholesky diagonal floor.",
        )
        return packed.at[diagonal].set(_softplus_inverse(shifted))


PhysicalParameterMap: TypeAlias = (
    PositiveParameterMap | BoundedParameterMap | SPDParameterMap
)


class ReducedRodParameterization(StrictModule, NonTrainableState):
    """Named positive, bounded, and SPD coordinates for reduced-rod physics."""

    maps: tuple[PhysicalParameterMap, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        maps: Sequence[PhysicalParameterMap],
        /,
        *,
        parameterization_id: str | None = None,
    ):
        maps_ = tuple(maps)
        if not maps_ or any(
            not isinstance(
                value,
                (PositiveParameterMap, BoundedParameterMap, SPDParameterMap),
            )
            for value in maps_
        ):
            raise TypeError("maps must contain supported physical parameter maps.")
        names = tuple(value.name for value in maps_)
        if len(set(names)) != len(names):
            raise ValueError("Physical parameter map names must be unique.")
        identity = (
            canonical_fingerprint(
                {
                    "kind": "reduced-rod-parameterization",
                    "maps": [value.map_id for value in maps_],
                }
            )
            if parameterization_id is None
            else _identifier(parameterization_id, "parameterization_id")
        )
        self.maps = maps_
        self.names = names
        self.parameterization_id = identity

    def _mapping(self, values: Mapping[str, Any], name: str, /) -> Mapping[str, Any]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping keyed by parameter name.")
        if set(values) != set(self.names):
            raise ValueError(f"{name} keys must exactly match {self.names}.")
        return values

    def to_physical(self, latent: Mapping[str, Any], /) -> dict[str, Array]:
        values = self._mapping(latent, "latent parameters")
        return {
            parameter_map.name: parameter_map.to_physical(values[parameter_map.name])
            for parameter_map in self.maps
        }

    def to_latent(self, physical: Mapping[str, Any], /) -> dict[str, Array]:
        values = self._mapping(physical, "physical parameters")
        return {
            parameter_map.name: parameter_map.to_latent(values[parameter_map.name])
            for parameter_map in self.maps
        }


class CalibrationExperiment(StrictModule, NonTrainableState):
    """One whitened reduced-rod experiment assigned to exactly one split."""

    weight: Any
    residual: Callable[[Mapping[str, Array], Any], PyTree[Array]] = eqx.field(static=True)
    route_valid: Callable[[Mapping[str, Array], Any], ArrayLike] | None = eqx.field(
        static=True
    )
    split: CalibrationSplit = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[Mapping[str, Array], Any], PyTree[Array]],
        /,
        *,
        split: CalibrationSplit,
        experiment_id: str,
        route_id: str,
        weight: Any = None,
        route_valid: Callable[[Mapping[str, Array], Any], ArrayLike] | None = None,
    ):
        if not callable(residual):
            raise TypeError("CalibrationExperiment residual must be callable.")
        if route_valid is not None and not callable(route_valid):
            raise TypeError("CalibrationExperiment route_valid must be callable or None.")
        if split not in ("train", "validation", "held_out"):
            raise ValueError("CalibrationExperiment split is invalid.")
        self.weight = weight
        self.residual = residual
        self.route_valid = route_valid
        self.split = split
        self.experiment_id = _identifier(experiment_id, "experiment_id")
        self.route_id = _identifier(route_id, "route_id")


class CalibrationAcceptance(StrictModule, NonTrainableState):
    """Declared disjoint-split and identifiability acceptance thresholds."""

    maximum_training_rmse: float = eqx.field(static=True)
    maximum_validation_rmse: float = eqx.field(static=True)
    maximum_held_out_rmse: float = eqx.field(static=True)
    maximum_held_out_absolute: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    require_validation: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_training_rmse: float = np.inf,
        maximum_validation_rmse: float = np.inf,
        maximum_held_out_rmse: float,
        maximum_held_out_absolute: float = np.inf,
        maximum_condition_number: float = np.inf,
        require_validation: bool = False,
    ):
        values = (
            maximum_training_rmse,
            maximum_validation_rmse,
            maximum_held_out_rmse,
            maximum_held_out_absolute,
            maximum_condition_number,
        )
        converted = tuple(float(value) for value in values)
        if any(np.isnan(value) or value < 0.0 for value in converted):
            raise ValueError("Calibration acceptance thresholds must be non-negative.")
        (
            self.maximum_training_rmse,
            self.maximum_validation_rmse,
            self.maximum_held_out_rmse,
            self.maximum_held_out_absolute,
            self.maximum_condition_number,
        ) = converted
        self.require_validation = bool(require_validation)


class CalibrationSplitEvidence(StrictModule, NonTrainableState):
    """Whitened residual and route evidence for one immutable data split."""

    residuals: Array
    rmse: Array
    maximum_absolute: Array
    finite: Array
    route_valid: Array
    accepted: Array
    residual_count: Array
    split: CalibrationSplit = eqx.field(static=True)
    experiment_ids: tuple[str, ...] = eqx.field(static=True)
    route_ids: tuple[str, ...] = eqx.field(static=True)


class CalibrationIdentifiabilityEvidence(StrictModule, NonTrainableState):
    """SVD, null-space, and covariance evidence from training data only."""

    residual_jacobian: Array
    singular_values: Array
    right_singular_vectors: Array
    null_projection: Array
    numerical_rank: Array
    rank_threshold: Array
    condition_number: Array
    fisher_information: Array
    latent_covariance: Array
    physical_covariance: Array
    physical_correlation: Array
    finite: Array
    full_rank: Array
    accepted: Array
    parameter_count: int = eqx.field(static=True)
    residual_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ReducedRodCalibrationProblem(StrictModule, NonTrainableState):
    """Reduced-rod physical fit lowered to the native residual-graph solver."""

    parameterization: ReducedRodParameterization
    source_physical_parameters: Mapping[str, Array]
    source_realization: Any
    experiments: tuple[CalibrationExperiment, ...]
    residual_blocks: tuple[ResidualBlock, ...]
    graph: ResidualGraphProblem
    acceptance: CalibrationAcceptance
    realize: Callable[[Mapping[str, Array], Any], Any] = eqx.field(static=True)
    admissible: Callable[[Mapping[str, Array], Any, Any], ArrayLike] | None = eqx.field(
        static=True
    )
    relative_rank_tolerance: float = eqx.field(static=True)
    absolute_rank_tolerance: float = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    rod_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterization: ReducedRodParameterization,
        source_physical_parameters: Mapping[str, Any],
        source_realization: Any,
        experiments: Sequence[CalibrationExperiment],
        /,
        *,
        acceptance: CalibrationAcceptance,
        realize: Callable[[Mapping[str, Array], Any], Any],
        source_realization_id: str,
        rod_id: str,
        reduction_id: str,
        actuator_id: str,
        plant_id: str,
        admissible: Callable[[Mapping[str, Array], Any, Any], ArrayLike] | None = None,
        relative_rank_tolerance: float = 1e-8,
        absolute_rank_tolerance: float = 0.0,
        problem_id: str,
    ):
        if not isinstance(parameterization, ReducedRodParameterization):
            raise TypeError("parameterization must be ReducedRodParameterization.")
        if not isinstance(acceptance, CalibrationAcceptance):
            raise TypeError("acceptance must be CalibrationAcceptance.")
        if not callable(realize):
            raise TypeError("realize must be callable.")
        if admissible is not None and not callable(admissible):
            raise TypeError("admissible must be callable or None.")
        experiments_ = tuple(experiments)
        if not experiments_ or any(
            not isinstance(value, CalibrationExperiment) for value in experiments_
        ):
            raise TypeError("experiments must contain CalibrationExperiment values.")
        experiment_ids = tuple(value.experiment_id for value in experiments_)
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Calibration experiment IDs must be globally unique.")
        if not any(value.split == "train" for value in experiments_):
            raise ValueError("Calibration requires at least one training experiment.")
        if not any(value.split == "held_out" for value in experiments_):
            raise ValueError("Calibration requires at least one held-out experiment.")
        if acceptance.require_validation and not any(
            value.split == "validation" for value in experiments_
        ):
            raise ValueError("Calibration policy requires a validation experiment.")
        relative = float(relative_rank_tolerance)
        absolute = float(absolute_rank_tolerance)
        if not isfinite(relative) or relative < 0.0:
            raise ValueError("relative_rank_tolerance must be finite and non-negative.")
        if not isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_rank_tolerance must be finite and non-negative.")
        source_physical = {
            name: jnp.asarray(source_physical_parameters[name])
            for name in parameterization.names
        }
        parameterization.to_latent(source_physical)
        parameter_block = ParameterBlock(
            lambda values: values,
            lambda values, replacement: replacement,
            block_id="reduced-rod-physical-latent",
        )
        residual_blocks = []
        for experiment in experiments_:
            residual_blocks.append(
                ResidualBlock(
                    lambda values, args, experiment=experiment: experiment.residual(
                        parameterization.to_physical(values[0]), args
                    ),
                    (parameter_block.block_id,),
                    weight=experiment.weight,
                    block_id=f"calibration:{experiment.experiment_id}",
                )
            )
        graph = ResidualGraphProblem(
            (parameter_block,),
            tuple(
                block
                for block, experiment in zip(residual_blocks, experiments_, strict=True)
                if experiment.split == "train"
            ),
            problem_id=f"{problem_id}:training-residual-graph",
        )
        self.parameterization = parameterization
        self.source_physical_parameters = source_physical
        self.source_realization = source_realization
        self.experiments = experiments_
        self.residual_blocks = tuple(residual_blocks)
        self.graph = graph
        self.acceptance = acceptance
        self.realize = realize
        self.admissible = admissible
        self.relative_rank_tolerance = relative
        self.absolute_rank_tolerance = absolute
        self.source_realization_id = _identifier(
            source_realization_id, "source_realization_id"
        )
        self.rod_id = _identifier(rod_id, "rod_id")
        self.reduction_id = _identifier(reduction_id, "reduction_id")
        self.actuator_id = _identifier(actuator_id, "actuator_id")
        self.plant_id = _identifier(plant_id, "plant_id")
        self.problem_id = _identifier(problem_id, "problem_id")

    def training_residual(self, latent: PyTree[Any], args: Any = None, /):
        return self.graph.residual(latent, args)

    def split_evidence(
        self,
        split: CalibrationSplit,
        latent: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> CalibrationSplitEvidence:
        if split not in ("train", "validation", "held_out"):
            raise ValueError("Unknown calibration split.")
        physical = self.parameterization.to_physical(latent)
        selected = tuple(
            (experiment, block)
            for experiment, block in zip(
                self.experiments, self.residual_blocks, strict=True
            )
            if experiment.split == split
        )
        flattened = tuple(
            ravel_pytree(block.weighted_residual((latent,), args))[0]
            for _, block in selected
        )
        residuals = (
            jnp.concatenate(flattened)
            if flattened
            else jnp.empty((0,), dtype=jax.tree.leaves(latent)[0].dtype)
        )
        route_values = tuple(
            jnp.asarray(
                True
                if experiment.route_valid is None
                else experiment.route_valid(physical, args),
                dtype=bool,
            ).reshape(())
            for experiment, _ in selected
        )
        route_valid = (
            jnp.all(jnp.stack(route_values)) if route_values else jnp.asarray(True)
        )
        finite = jnp.all(jnp.isfinite(residuals))
        rmse = (
            jnp.sqrt(jnp.mean(jnp.square(residuals)))
            if residuals.size
            else jnp.asarray(0.0, dtype=residuals.dtype)
        )
        maximum = (
            jnp.max(jnp.abs(residuals))
            if residuals.size
            else jnp.asarray(0.0, dtype=residuals.dtype)
        )
        if split == "train":
            threshold = self.acceptance.maximum_training_rmse
            accepted = finite & route_valid & (rmse <= threshold)
        elif split == "validation":
            threshold = self.acceptance.maximum_validation_rmse
            accepted = finite & route_valid & (rmse <= threshold)
            if not selected and self.acceptance.require_validation:
                accepted = jnp.asarray(False)
        else:
            accepted = (
                finite
                & route_valid
                & (rmse <= self.acceptance.maximum_held_out_rmse)
                & (maximum <= self.acceptance.maximum_held_out_absolute)
            )
        return CalibrationSplitEvidence(
            residuals,
            rmse,
            maximum,
            finite,
            route_valid,
            accepted,
            jnp.asarray(residuals.size, dtype=jnp.int32),
            split,
            tuple(experiment.experiment_id for experiment, _ in selected),
            tuple(experiment.route_id for experiment, _ in selected),
        )

    def identifiability(
        self,
        latent: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> CalibrationIdentifiabilityEvidence:
        flat_latent, unravel = ravel_pytree(latent)
        if flat_latent.size < 1:
            raise ValueError("Calibration requires at least one latent coordinate.")

        def residual_vector(coordinates):
            return ravel_pytree(self.training_residual(unravel(coordinates), args))[0]

        residuals = residual_vector(flat_latent)
        jacobian = jax.jacrev(residual_vector)(flat_latent)
        _, singular_values, right_vectors = jnp.linalg.svd(jacobian, full_matrices=True)
        largest = (
            singular_values[0]
            if singular_values.size
            else jnp.asarray(0.0, dtype=jacobian.dtype)
        )
        threshold = jnp.maximum(
            self.absolute_rank_tolerance,
            self.relative_rank_tolerance * largest,
        )
        rank = jnp.sum(singular_values > threshold, dtype=jnp.int32)
        parameter_count = int(flat_latent.size)
        full_rank = rank == parameter_count
        null_mask = jnp.arange(parameter_count, dtype=jnp.int32) >= rank
        null_projection = (
            right_vectors.T * null_mask.astype(jacobian.dtype)
        ) @ right_vectors
        padded_singular_values = jnp.pad(
            singular_values,
            (0, max(parameter_count - singular_values.size, 0)),
        )
        smallest = padded_singular_values[parameter_count - 1]
        condition = jnp.where(
            full_rank & (smallest > 0.0),
            largest / smallest,
            jnp.inf,
        )
        fisher = jacobian.T @ jacobian
        latent_covariance_candidate = jnp.linalg.pinv(
            fisher, rtol=self.relative_rank_tolerance
        )
        latent_covariance = jnp.where(
            full_rank,
            latent_covariance_candidate,
            jnp.full_like(latent_covariance_candidate, jnp.nan),
        )

        def physical_vector(coordinates):
            return ravel_pytree(self.parameterization.to_physical(unravel(coordinates)))[
                0
            ]

        physical_jacobian = jax.jacrev(physical_vector)(flat_latent)
        physical_covariance_candidate = (
            physical_jacobian @ latent_covariance_candidate @ physical_jacobian.T
        )
        physical_covariance = jnp.where(
            full_rank,
            physical_covariance_candidate,
            jnp.full_like(physical_covariance_candidate, jnp.nan),
        )
        scales = jnp.sqrt(jnp.maximum(jnp.diag(physical_covariance_candidate), 0.0))
        denominator = scales[:, None] * scales[None, :]
        correlation_candidate = jnp.where(
            denominator > 0.0,
            physical_covariance_candidate / denominator,
            0.0,
        )
        physical_correlation = jnp.where(
            full_rank,
            correlation_candidate,
            jnp.full_like(correlation_candidate, jnp.nan),
        )
        finite = (
            jnp.all(jnp.isfinite(residuals))
            & jnp.all(jnp.isfinite(jacobian))
            & jnp.all(jnp.isfinite(singular_values))
        )
        accepted = (
            finite & full_rank & (condition <= self.acceptance.maximum_condition_number)
        )
        return CalibrationIdentifiabilityEvidence(
            jacobian,
            singular_values,
            right_vectors,
            null_projection,
            rank,
            threshold,
            condition,
            fisher,
            latent_covariance,
            physical_covariance,
            physical_correlation,
            finite,
            full_rank,
            accepted,
            parameter_count,
            int(residuals.size),
            self.problem_id,
            canonical_fingerprint(
                {
                    "kind": "reduced-rod-identifiability",
                    "problem": self.problem_id,
                    "parameterization": self.parameterization.parameterization_id,
                }
            ),
        )


class ReducedRodCalibrationStatus(IntEnum):
    """Acceptance outcome after optimization and independent qualification."""

    SUCCESS = 0
    OPTIMIZATION_FAILED = 1
    NONPHYSICAL = 2
    RANK_DEFICIENT = 3
    TRAINING_FAILED = 4
    VALIDATION_FAILED = 5
    HELD_OUT_FAILED = 6
    MODE_FAILED = 7


class ReducedRodCalibrationResult(StrictModule, NonTrainableState):
    """Candidate fit and atomic accepted/source realization selection."""

    optimization: Any
    candidate_latent: PyTree[Array]
    candidate_physical_parameters: Mapping[str, Array]
    accepted_physical_parameters: Mapping[str, Array]
    candidate_realization: Any
    accepted_realization: Any
    identifiability: CalibrationIdentifiabilityEvidence
    training: CalibrationSplitEvidence
    validation: CalibrationSplitEvidence
    held_out: CalibrationSplitEvidence
    physical: Array
    admissible: Array
    accepted: Array
    status: Array
    candidate_realization_id: str = eqx.field(static=True)
    accepted_realization_id: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    rod_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted & (self.status == int(ReducedRodCalibrationStatus.SUCCESS))


def calibrate_reduced_rod(
    problem: ReducedRodCalibrationProblem,
    initial_latent: Mapping[str, Any],
    /,
    *,
    args: Any = None,
    termination: Any = None,
    route_policy: Any = None,
    initial_damping: float = 1e-3,
) -> ReducedRodCalibrationResult:
    """Solve the training graph, then gate acceptance on disjoint evidence."""

    if not isinstance(problem, ReducedRodCalibrationProblem):
        raise TypeError("problem must be ReducedRodCalibrationProblem.")
    optimization = solve_residual_graph(
        problem.graph,
        initial_latent,
        termination=termination,
        route_policy=route_policy,
        args=args,
        initial_damping=initial_damping,
    )
    candidate_latent = optimization.parameters
    candidate_physical = problem.parameterization.to_physical(candidate_latent)
    candidate_realization = problem.realize(
        candidate_physical, problem.source_realization
    )
    identifiability = problem.identifiability(candidate_latent, args)
    training = problem.split_evidence("train", candidate_latent, args)
    validation = problem.split_evidence("validation", candidate_latent, args)
    held_out = problem.split_evidence("held_out", candidate_latent, args)
    physical = _tree_finite(candidate_physical)
    admissible = (
        jnp.asarray(True)
        if problem.admissible is None
        else jnp.asarray(
            problem.admissible(candidate_physical, candidate_realization, args),
            dtype=bool,
        ).reshape(())
    )
    accepted = (
        optimization.successful
        & physical
        & identifiability.accepted
        & training.accepted
        & validation.accepted
        & held_out.accepted
        & admissible
    )
    status = jnp.where(
        ~optimization.successful,
        int(ReducedRodCalibrationStatus.OPTIMIZATION_FAILED),
        jnp.where(
            ~physical,
            int(ReducedRodCalibrationStatus.NONPHYSICAL),
            jnp.where(
                ~identifiability.accepted,
                int(ReducedRodCalibrationStatus.RANK_DEFICIENT),
                jnp.where(
                    ~training.accepted,
                    int(ReducedRodCalibrationStatus.TRAINING_FAILED),
                    jnp.where(
                        ~validation.accepted,
                        int(ReducedRodCalibrationStatus.VALIDATION_FAILED),
                        jnp.where(
                            ~held_out.accepted,
                            int(ReducedRodCalibrationStatus.HELD_OUT_FAILED),
                            jnp.where(
                                ~admissible,
                                int(ReducedRodCalibrationStatus.MODE_FAILED),
                                int(ReducedRodCalibrationStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    accepted_python = bool(accepted)
    accepted_physical = (
        candidate_physical if accepted_python else problem.source_physical_parameters
    )
    accepted_realization = (
        candidate_realization if accepted_python else problem.source_realization
    )
    candidate_id = canonical_fingerprint(
        {
            "kind": "reduced-rod-calibration-candidate",
            "problem": problem.problem_id,
            "physical": array_tree_fingerprint(candidate_physical),
            "rod": problem.rod_id,
            "reduction": problem.reduction_id,
            "actuator": problem.actuator_id,
            "plant": problem.plant_id,
        }
    )
    accepted_id = candidate_id if accepted_python else problem.source_realization_id
    result_id = canonical_fingerprint(
        {
            "kind": "reduced-rod-calibration-result",
            "problem": problem.problem_id,
            "candidate": candidate_id,
            "accepted": accepted_python,
            "accepted_realization": accepted_id,
        }
    )
    return ReducedRodCalibrationResult(
        optimization,
        candidate_latent,
        candidate_physical,
        accepted_physical,
        candidate_realization,
        accepted_realization,
        identifiability,
        training,
        validation,
        held_out,
        physical,
        admissible,
        accepted,
        status,
        candidate_id,
        accepted_id,
        problem.source_realization_id,
        problem.rod_id,
        problem.reduction_id,
        problem.actuator_id,
        problem.plant_id,
        problem.problem_id,
        result_id,
    )


class FixedModeDerivativeEvidence(StrictModule, NonTrainableState):
    """Evidence that a derivative stayed inside one smooth realized mode."""

    material_margin: Array
    kinematic_margin: Array
    actuator_margin: Array
    contact_margin: Array
    active_set_margin: Array
    condition_number: Array
    jvp_residual: Array
    vjp_residual: Array
    finite: Array
    primal_accepted: Array
    route_fixed: Array
    accepted: Array
    morphology_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    fixed_mode_id: str = eqx.field(static=True)
    primal_result_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        material_margin: ArrayLike,
        kinematic_margin: ArrayLike,
        actuator_margin: ArrayLike,
        contact_margin: ArrayLike,
        active_set_margin: ArrayLike,
        condition_number: ArrayLike,
        jvp_residual: ArrayLike,
        vjp_residual: ArrayLike,
        finite: ArrayLike,
        primal_accepted: ArrayLike,
        route_fixed: ArrayLike,
        morphology_id: str,
        actuator_id: str,
        control_id: str,
        fixed_mode_id: str,
        primal_result_id: str,
        maximum_condition_number: float = np.inf,
        maximum_derivative_residual: float = np.inf,
    ):
        margins = tuple(
            jnp.asarray(value).reshape(())
            for value in (
                material_margin,
                kinematic_margin,
                actuator_margin,
                contact_margin,
                active_set_margin,
            )
        )
        condition = jnp.asarray(condition_number).reshape(())
        jvp = jnp.asarray(jvp_residual).reshape(())
        vjp = jnp.asarray(vjp_residual).reshape(())
        finite_ = jnp.asarray(finite, dtype=bool).reshape(()) & jnp.all(
            jnp.stack(
                tuple(jnp.isfinite(value) for value in margins)
                + (jnp.isfinite(condition), jnp.isfinite(jvp), jnp.isfinite(vjp))
            )
        )
        primal = jnp.asarray(primal_accepted, dtype=bool).reshape(())
        fixed = jnp.asarray(route_fixed, dtype=bool).reshape(())
        maximum_condition = float(maximum_condition_number)
        maximum_residual = float(maximum_derivative_residual)
        if np.isnan(maximum_condition) or maximum_condition < 0.0:
            raise ValueError("maximum_condition_number must be non-negative.")
        if np.isnan(maximum_residual) or maximum_residual < 0.0:
            raise ValueError("maximum_derivative_residual must be non-negative.")
        accepted = (
            finite_
            & primal
            & fixed
            & jnp.all(jnp.stack(tuple(value > 0.0 for value in margins)))
            & (condition >= 0.0)
            & (jvp >= 0.0)
            & (vjp >= 0.0)
            & (condition <= maximum_condition)
            & (jvp <= maximum_residual)
            & (vjp <= maximum_residual)
        )
        morphology = _identifier(morphology_id, "morphology_id")
        actuator = _identifier(actuator_id, "actuator_id")
        control = _identifier(control_id, "control_id")
        mode = _identifier(fixed_mode_id, "fixed_mode_id")
        primal_id = _identifier(primal_result_id, "primal_result_id")
        domain_id = canonical_fingerprint(
            {
                "kind": "soft-robot-fixed-mode-derivative-domain",
                "morphology": morphology,
                "actuator": actuator,
                "control": control,
                "mode": mode,
                "primal": primal_id,
            }
        )
        (
            self.material_margin,
            self.kinematic_margin,
            self.actuator_margin,
            self.contact_margin,
            self.active_set_margin,
        ) = margins
        self.condition_number = condition
        self.jvp_residual = jvp
        self.vjp_residual = vjp
        self.finite = finite_
        self.primal_accepted = primal
        self.route_fixed = fixed
        self.accepted = accepted
        self.morphology_id = morphology
        self.actuator_id = actuator
        self.control_id = control
        self.fixed_mode_id = mode
        self.primal_result_id = primal_id
        self.domain_id = domain_id


class SoftCoDesignConstraint(StrictModule, NonTrainableState):
    """Bound constraint evaluated on state and physical co-design values."""

    function: Callable[[PyTree[Any], Mapping[str, Array], Any], PyTree[Array]] = (
        eqx.field(static=True)
    )
    lower: Any
    upper: Any
    depends_on_state: bool = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[PyTree[Any], Mapping[str, Array], Any], PyTree[Array]],
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        depends_on_state: bool = True,
        constraint_id: str,
    ):
        if not callable(function):
            raise TypeError("SoftCoDesignConstraint function must be callable.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.depends_on_state = bool(depends_on_state)
        self.constraint_id = _identifier(constraint_id, "constraint_id")


class CoDesignHeldOutScenario(StrictModule, NonTrainableState):
    """One disjoint scenario that can reject but never train a co-design."""

    qualifies: Callable[[PyTree[Any], Mapping[str, Array], Any, Any], ArrayLike] = (
        eqx.field(static=True)
    )
    scenario_id: str = eqx.field(static=True)

    def __init__(
        self,
        qualifies: Callable[[PyTree[Any], Mapping[str, Array], Any, Any], ArrayLike],
        /,
        *,
        scenario_id: str,
    ):
        if not callable(qualifies):
            raise TypeError("CoDesignHeldOutScenario qualifies must be callable.")
        self.qualifies = qualifies
        self.scenario_id = _identifier(scenario_id, "scenario_id")


class SoftRobotCoDesignProblem(StrictModule, NonTrainableState):
    """Fixed-mode morphology/actuator/controller design over StateDesignProblem."""

    parameterization: ReducedRodParameterization
    source_design: Mapping[str, Array]
    source_physical_design: Mapping[str, Array]
    source_realization: Any
    state_design: StateDesignProblem
    held_out_scenarios: tuple[CoDesignHeldOutScenario, ...]
    realize: Callable[[Mapping[str, Array], Any], Any] = eqx.field(static=True)
    derivative_evidence: Callable[
        [PyTree[Any], Mapping[str, Array], Any, str, Any],
        FixedModeDerivativeEvidence,
    ] = eqx.field(static=True)
    realization_admissible: (
        Callable[[PyTree[Any], Mapping[str, Array], Any, Any], ArrayLike] | None
    ) = eqx.field(static=True)
    morphology_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    fixed_mode_id: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    co_design_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterization: ReducedRodParameterization,
        source_design: Mapping[str, Any],
        source_realization: Any,
        state_residual: Callable[[PyTree[Any], Mapping[str, Array], Any], PyTree[Array]],
        objective: Callable[[PyTree[Any], Mapping[str, Array], Any], Any],
        /,
        *,
        realize: Callable[[Mapping[str, Array], Any], Any],
        derivative_evidence: Callable[
            [PyTree[Any], Mapping[str, Array], Any, str, Any],
            FixedModeDerivativeEvidence,
        ],
        held_out_scenarios: Sequence[CoDesignHeldOutScenario],
        morphology_id: str,
        actuator_id: str,
        control_id: str,
        fixed_mode_id: str,
        source_realization_id: str,
        constraints: Sequence[SoftCoDesignConstraint] = (),
        state_solver: Any = None,
        acceptance_policy: Any = None,
        state_admissibility: Callable[[PyTree[Any], Mapping[str, Array], Any], ArrayLike]
        | None = None,
        realization_matches: Callable[[PyTree[Any], Mapping[str, Array], Any], ArrayLike]
        | None = None,
        realization_admissible: Callable[
            [PyTree[Any], Mapping[str, Array], Any, Any], ArrayLike
        ]
        | None = None,
        design_bounds: Bounds | None = None,
        has_aux: bool = False,
        problem_id: str,
    ):
        if not isinstance(parameterization, ReducedRodParameterization):
            raise TypeError("parameterization must be ReducedRodParameterization.")
        for value, name in (
            (state_residual, "state_residual"),
            (objective, "objective"),
            (realize, "realize"),
            (derivative_evidence, "derivative_evidence"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable.")
        if state_admissibility is not None and not callable(state_admissibility):
            raise TypeError("state_admissibility must be callable or None.")
        if realization_matches is not None and not callable(realization_matches):
            raise TypeError("realization_matches must be callable or None.")
        if realization_admissible is not None and not callable(realization_admissible):
            raise TypeError("realization_admissible must be callable or None.")
        constraints_ = tuple(constraints)
        if any(not isinstance(value, SoftCoDesignConstraint) for value in constraints_):
            raise TypeError("constraints must contain SoftCoDesignConstraint values.")
        held_out = tuple(held_out_scenarios)
        if not held_out or any(
            not isinstance(value, CoDesignHeldOutScenario) for value in held_out
        ):
            raise TypeError(
                "held_out_scenarios must contain at least one held-out scenario."
            )
        scenario_ids = tuple(value.scenario_id for value in held_out)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("Held-out co-design scenario IDs must be unique.")
        source_latent = {
            name: jnp.asarray(value) for name, value in source_design.items()
        }
        source_physical = parameterization.to_physical(source_latent)

        def physical(values):
            return parameterization.to_physical(values)

        state_design_constraints = tuple(
            StateDesignConstraint(
                lambda state, design, args, constraint=constraint: constraint.function(
                    state, physical(design), args
                ),
                lower=constraint.lower,
                upper=constraint.upper,
                constraint_id=constraint.constraint_id,
                depends_on_state=constraint.depends_on_state,
            )
            for constraint in constraints_
        )
        state_design = StateDesignProblem(
            lambda state, design, args: state_residual(state, physical(design), args),
            lambda state, design, args: objective(state, physical(design), args),
            state_solver=state_solver,
            acceptance_policy=acceptance_policy,
            state_admissibility=(
                None
                if state_admissibility is None
                else lambda state, design, args: state_admissibility(
                    state, physical(design), args
                )
            ),
            state_realization=(
                None
                if realization_matches is None
                else lambda state, design, args: realization_matches(
                    state, physical(design), args
                )
            ),
            design_bounds=design_bounds,
            constraints=state_design_constraints,
            has_aux=has_aux,
            problem_id=f"{problem_id}:state-design",
        )
        morphology = _identifier(morphology_id, "morphology_id")
        actuator = _identifier(actuator_id, "actuator_id")
        control = _identifier(control_id, "control_id")
        mode = _identifier(fixed_mode_id, "fixed_mode_id")
        source_id = _identifier(source_realization_id, "source_realization_id")
        identifier = _identifier(problem_id, "problem_id")
        self.parameterization = parameterization
        self.source_design = source_latent
        self.source_physical_design = source_physical
        self.source_realization = source_realization
        self.state_design = state_design
        self.held_out_scenarios = held_out
        self.realize = realize
        self.derivative_evidence = derivative_evidence
        self.realization_admissible = realization_admissible
        self.morphology_id = morphology
        self.actuator_id = actuator
        self.control_id = control
        self.fixed_mode_id = mode
        self.source_realization_id = source_id
        self.problem_id = identifier
        self.co_design_id = canonical_fingerprint(
            {
                "kind": "soft-robot-fixed-mode-co-design",
                "problem": identifier,
                "parameterization": parameterization.parameterization_id,
                "morphology": morphology,
                "actuator": actuator,
                "control": control,
                "fixed_mode": mode,
                "source_realization": source_id,
                "held_out": list(scenario_ids),
            }
        )

    def physical_design(self, design: Mapping[str, Any], /) -> Mapping[str, Array]:
        return self.parameterization.to_physical(design)

    def as_state_design_problem(self) -> StateDesignProblem:
        return self.state_design

    def compile_sqp(
        self,
        initial_state: PyTree[Any],
        initial_design: Mapping[str, Any],
        /,
        *,
        sample_args: Any = None,
        method: SQP | None = None,
    ) -> SoftRobotCoDesignSQPCompilation:
        residual = self.state_design.residual(initial_state, initial_design, sample_args)
        zeros = jax.tree.map(jnp.zeros_like, residual)
        state_lower = jax.tree.map(
            lambda value: jnp.full_like(value, -jnp.inf), initial_state
        )
        state_upper = jax.tree.map(
            lambda value: jnp.full_like(value, jnp.inf), initial_state
        )
        if self.state_design.design_bounds is None:
            design_lower = jax.tree.map(
                lambda value: jnp.full_like(value, -jnp.inf), initial_design
            )
            design_upper = jax.tree.map(
                lambda value: jnp.full_like(value, jnp.inf), initial_design
            )
        else:
            design_lower, design_upper = self.state_design.design_bounds.materialize(
                initial_design
            )

        def objective(values, args):
            value, auxiliary = self.state_design.value(values[0], values[1], args)
            return (value, auxiliary) if self.state_design.has_aux else value

        constraints = [
            NonlinearConstraint(
                lambda values, args: self.state_design.residual(
                    values[0], values[1], args
                ),
                lower=zeros,
                upper=zeros,
                constraint_id=f"{self.problem_id}:state-equation",
            )
        ]
        constraints.extend(
            NonlinearConstraint(
                lambda values, args, constraint=constraint: constraint.value(
                    values[0], values[1], args
                ),
                lower=constraint.lower,
                upper=constraint.upper,
                constraint_id=constraint.constraint_id,
            )
            for constraint in self.state_design.constraints
        )
        minimization = MinimizationProblem(
            objective,
            has_aux=self.state_design.has_aux,
            bounds=Bounds(
                (state_lower, design_lower),
                (state_upper, design_upper),
            ),
            constraints=tuple(constraints),
            problem_id=f"{self.problem_id}:fixed-mode-all-at-once-sqp",
        )
        return SoftRobotCoDesignSQPCompilation(
            self,
            minimization,
            SQP() if method is None else method,
            canonical_fingerprint(
                {
                    "kind": "soft-robot-co-design-sqp-compilation",
                    "co_design": self.co_design_id,
                    "minimization": minimization.problem_id,
                }
            ),
        )

    def accept_result(
        self,
        optimization: Any,
        /,
        *,
        args: Any = None,
    ) -> SoftRobotCoDesignResult:
        if not hasattr(optimization, "state") or not hasattr(optimization, "design"):
            if not hasattr(optimization, "parameters"):
                raise TypeError(
                    "optimization must expose state/design or all-at-once parameters."
                )
            state, design = optimization.parameters
        else:
            state, design = optimization.state, optimization.design
        physical = self.physical_design(design)
        candidate_realization = self.realize(physical, self.source_realization)
        primal_result_id = canonical_fingerprint(
            {
                "kind": "soft-robot-co-design-primal",
                "co_design": self.co_design_id,
                "state": array_tree_fingerprint(state),
                "physical": array_tree_fingerprint(physical),
            }
        )
        derivative = self.derivative_evidence(
            state,
            physical,
            candidate_realization,
            primal_result_id,
            args,
        )
        if not isinstance(derivative, FixedModeDerivativeEvidence):
            raise TypeError(
                "derivative_evidence must return FixedModeDerivativeEvidence."
            )
        if (
            derivative.morphology_id != self.morphology_id
            or derivative.actuator_id != self.actuator_id
            or derivative.control_id != self.control_id
            or derivative.fixed_mode_id != self.fixed_mode_id
            or derivative.primal_result_id != primal_result_id
        ):
            raise ValueError("Derivative evidence identifiers do not match co-design.")
        held_out = jnp.stack(
            tuple(
                jnp.asarray(
                    scenario.qualifies(state, physical, candidate_realization, args),
                    dtype=bool,
                ).reshape(())
                for scenario in self.held_out_scenarios
            )
        )
        admissible = (
            jnp.asarray(True)
            if self.realization_admissible is None
            else jnp.asarray(
                self.realization_admissible(state, physical, candidate_realization, args),
                dtype=bool,
            ).reshape(())
        )
        optimization_successful = jnp.asarray(
            optimization.successful, dtype=bool
        ).reshape(())
        accepted = (
            optimization_successful
            & _tree_finite(physical)
            & derivative.accepted
            & jnp.all(held_out)
            & admissible
        )
        accepted_python = bool(accepted)
        accepted_design = physical if accepted_python else self.source_physical_design
        accepted_realization = (
            candidate_realization if accepted_python else self.source_realization
        )
        candidate_id = canonical_fingerprint(
            {
                "kind": "soft-robot-co-design-candidate",
                "co_design": self.co_design_id,
                "physical": array_tree_fingerprint(physical),
            }
        )
        accepted_id = candidate_id if accepted_python else self.source_realization_id
        return SoftRobotCoDesignResult(
            optimization,
            state,
            design,
            physical,
            accepted_design,
            candidate_realization,
            accepted_realization,
            derivative,
            held_out,
            admissible,
            accepted,
            candidate_id,
            accepted_id,
            tuple(value.scenario_id for value in self.held_out_scenarios),
            self.morphology_id,
            self.actuator_id,
            self.control_id,
            self.fixed_mode_id,
            self.co_design_id,
            canonical_fingerprint(
                {
                    "kind": "soft-robot-co-design-result",
                    "co_design": self.co_design_id,
                    "candidate": candidate_id,
                    "accepted": accepted_python,
                }
            ),
        )


class SoftRobotCoDesignSQPCompilation(StrictModule, NonTrainableState):
    """All-at-once fixed-mode co-design lowered to the existing SQP method."""

    problem: SoftRobotCoDesignProblem
    minimization: MinimizationProblem
    method: SQP
    compilation_id: str = eqx.field(static=True)


class SoftRobotCoDesignResult(StrictModule, NonTrainableState):
    """Candidate co-design and atomic accepted/source realization selection."""

    optimization: Any
    state: PyTree[Array]
    latent_design: Mapping[str, Array]
    candidate_design: Mapping[str, Array]
    accepted_design: Mapping[str, Array]
    candidate_realization: Any
    accepted_realization: Any
    derivative_evidence: FixedModeDerivativeEvidence
    held_out_accepted: Array
    admissible: Array
    accepted: Array
    candidate_realization_id: str = eqx.field(static=True)
    accepted_realization_id: str = eqx.field(static=True)
    held_out_scenario_ids: tuple[str, ...] = eqx.field(static=True)
    morphology_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    fixed_mode_id: str = eqx.field(static=True)
    co_design_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class SoftPlantMPCResult(StrictModule, NonTrainableState):
    """Sampling result plus the selected accepted replay's derivative domain."""

    sampling: SamplingMPCResult
    derivative_evidence: FixedModeDerivativeEvidence
    accepted: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def selected_accepted_replay(self) -> SamplingMPCSelectedReplay:
        return self.sampling.replay

    @property
    def successful(self) -> Array:
        return self.accepted


class SoftPlantMPCPlan(StrictModule, NonTrainableState):
    """Realization-aware sampling MPC bound to one declared soft-robot mode."""

    sampling: SamplingMPCPlan
    derivative_evidence: Callable[
        [SamplingMPCSelectedReplay, SamplingMPCResult], FixedModeDerivativeEvidence
    ] = eqx.field(static=True)
    morphology_id: str = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    fixed_mode_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def solve(
        self,
        state: Any,
        key: Array,
        /,
        *,
        warm_start: ArrayLike = False,
    ) -> SoftPlantMPCResult:
        sampling = self.sampling.solve(state, key, warm_start=warm_start)
        derivative = self.derivative_evidence(sampling.replay, sampling)
        if not isinstance(derivative, FixedModeDerivativeEvidence):
            raise TypeError(
                "Soft plant derivative callback must return FixedModeDerivativeEvidence."
            )
        if (
            derivative.morphology_id != self.morphology_id
            or derivative.actuator_id != self.actuator_id
            or derivative.control_id != self.control_id
            or derivative.fixed_mode_id != self.fixed_mode_id
            or derivative.primal_result_id != sampling.result_id
        ):
            raise ValueError("Soft plant derivative evidence identifiers are stale.")
        accepted = sampling.successful & sampling.replay.accepted & derivative.accepted
        return SoftPlantMPCResult(
            sampling,
            derivative,
            accepted,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "soft-plant-mpc-result", "plan": self.plan_id}
            ),
        )


def build_soft_plant_mpc(
    problem: Any,
    parameterization: Any,
    realizations: SamplingMPCRealizations,
    /,
    *,
    realization_binding: SamplingMPCRealizationBinding | None,
    realization_binding_id: str,
    derivative_evidence: Callable[
        [SamplingMPCSelectedReplay, SamplingMPCResult], FixedModeDerivativeEvidence
    ],
    morphology_id: str,
    actuator_id: str,
    control_id: str,
    fixed_mode_id: str,
    **sampling_options: Any,
) -> SoftPlantMPCPlan:
    """Build sampling MPC without differentiating its sorting or model resampling."""

    if not isinstance(realizations, SamplingMPCRealizations):
        raise TypeError("realizations must be SamplingMPCRealizations.")
    if not callable(derivative_evidence):
        raise TypeError("derivative_evidence must be callable.")
    sampling = plan_sampling_mpc(
        problem,
        parameterization,
        realizations=realizations,
        realization_binding=realization_binding,
        realization_binding_id=realization_binding_id,
        **sampling_options,
    )
    morphology = _identifier(morphology_id, "morphology_id")
    actuator = _identifier(actuator_id, "actuator_id")
    control = _identifier(control_id, "control_id")
    mode = _identifier(fixed_mode_id, "fixed_mode_id")
    plan_id = canonical_fingerprint(
        {
            "kind": "soft-plant-realization-aware-mpc",
            "sampling": sampling.plan_id,
            "realizations": realizations.batch_id,
            "morphology": morphology,
            "actuator": actuator,
            "control": control,
            "fixed_mode": mode,
        }
    )
    return SoftPlantMPCPlan(
        sampling,
        derivative_evidence,
        morphology,
        actuator,
        control,
        mode,
        plan_id,
    )


__all__ = [
    "BoundedParameterMap",
    "CalibrationAcceptance",
    "CalibrationExperiment",
    "CalibrationIdentifiabilityEvidence",
    "CalibrationSplit",
    "CalibrationSplitEvidence",
    "CoDesignHeldOutScenario",
    "FixedModeDerivativeEvidence",
    "PositiveParameterMap",
    "ReducedRodCalibrationProblem",
    "ReducedRodCalibrationResult",
    "ReducedRodCalibrationStatus",
    "ReducedRodParameterization",
    "SPDParameterMap",
    "SoftCoDesignConstraint",
    "SoftPlantMPCPlan",
    "SoftPlantMPCResult",
    "SoftRobotCoDesignProblem",
    "SoftRobotCoDesignResult",
    "SoftRobotCoDesignSQPCompilation",
    "build_soft_plant_mpc",
    "calibrate_reduced_rod",
]
