#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array

import phydrax.ein as ein
from phydrax._frozendict import frozendict
from phydrax._model import AbstractArrayModel, FrozenModel, register_artifact_value
from phydrax._numerics import solve_weighted_least_squares
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.nn._keys import EvalKey, fold_in_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel
from phydrax.nn.operator.topology import gather_operator_graph_entities

from ._deeponet import (
    AbstractBasisTrunk,
    AbstractBranchEncoder,
    DeepONet,
)


FunctionProjectionRankPolicy = Literal["error", "regularized"]

FUNCTION_PROJECTION_SUCCESS = 0
FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT = 1
FUNCTION_PROJECTION_RANK_DEFICIENT = 2
FUNCTION_PROJECTION_NONFINITE = 3


def _raise_or_error(value: Array, predicate: Array, message: str, /) -> Array:
    if not isinstance(predicate, jax_core.Tracer) and bool(predicate):
        raise eqx.EquinoxRuntimeError(message)
    return eqx.error_if(value, predicate, message)


FUNCTION_PROJECTION_INVALID_MEASURE = 4
FUNCTION_PROJECTION_REGULARIZED = 5


class FunctionProjectionPolicy(StrictModule, NonTrainableState):
    """Numerical and measure semantics for one learned-frame projection."""

    ridge: float = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)
    min_samples: int | None = eqx.field(static=True)
    scale_frame: bool = eqx.field(static=True)
    require_physical_quadrature: bool = eqx.field(static=True)
    rank_policy: FunctionProjectionRankPolicy = eqx.field(static=True)
    channel_metric: Array | None
    channel_factor: Array | None

    def __init__(
        self,
        *,
        ridge: float = 0.0,
        rcond: float | None = None,
        min_samples: int | None = None,
        scale_frame: bool = True,
        require_physical_quadrature: bool = False,
        rank_policy: FunctionProjectionRankPolicy = "error",
        channel_metric: Array | None = None,
    ):
        ridge_ = float(ridge)
        if not math.isfinite(ridge_) or ridge_ < 0.0:
            raise ValueError("ridge must be finite and nonnegative.")
        rcond_ = None if rcond is None else float(rcond)
        if rcond_ is not None and (not math.isfinite(rcond_) or rcond_ < 0.0):
            raise ValueError("rcond must be finite and nonnegative or None.")
        minimum = None if min_samples is None else int(min_samples)
        if minimum is not None and minimum <= 0:
            raise ValueError("min_samples must be positive or None.")
        if rank_policy not in ("error", "regularized"):
            raise ValueError("rank_policy must be 'error' or 'regularized'.")
        if rank_policy == "regularized" and ridge_ <= 0.0:
            raise ValueError("rank_policy='regularized' requires positive ridge.")

        metric: Array | None
        factor: Array | None
        if channel_metric is None:
            metric = None
            factor = None
        else:
            host_metric = np.asarray(channel_metric)
            if host_metric.ndim != 2 or host_metric.shape[0] != host_metric.shape[1]:
                raise ValueError("channel_metric must be a square matrix.")
            if host_metric.shape[0] <= 0:
                raise ValueError("channel_metric must be nonempty.")
            if not np.issubdtype(host_metric.dtype, np.number):
                raise TypeError("channel_metric must be numeric.")
            dtype = np.result_type(host_metric.dtype, np.float64)
            host_metric = np.asarray(host_metric, dtype=dtype)
            if not np.all(np.isfinite(host_metric)):
                raise ValueError("channel_metric must be finite.")
            if not np.allclose(
                host_metric,
                np.conjugate(host_metric.T),
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError("channel_metric must be Hermitian.")
            eigenvalues = np.linalg.eigvalsh(host_metric)
            scale = max(float(np.linalg.norm(host_metric, ord=2)), 1.0)
            tolerance = np.finfo(eigenvalues.dtype).eps * host_metric.shape[0] * scale
            if np.any(eigenvalues <= tolerance):
                raise ValueError("channel_metric must be positive definite.")
            metric = jnp.asarray(host_metric)
            factor = jnp.asarray(np.linalg.cholesky(host_metric))

        self.ridge = ridge_
        self.rcond = rcond_
        self.min_samples = minimum
        self.scale_frame = bool(scale_frame)
        self.require_physical_quadrature = bool(require_physical_quadrature)
        self.rank_policy = rank_policy
        self.channel_metric = metric
        self.channel_factor = factor


class FunctionProjectionReport(StrictModule):
    """Encoded coefficients with explicit projection and identification evidence."""

    coefficients: Array
    residual_energy: Array
    target_energy: Array
    relative_residual: Array
    singular_values: Array
    sample_count: Array
    weight_sum: Array
    rank: Array
    condition_number: Array
    normal_equation_error: Array
    solved: Array
    identified: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficients: Array,
        residual_energy: Array,
        target_energy: Array,
        relative_residual: Array,
        singular_values: Array,
        sample_count: Array,
        weight_sum: Array,
        rank: Array,
        condition_number: Array,
        normal_equation_error: Array,
        solved: Array,
        identified: Array,
        valid: Array,
        status: Array,
        case_shape: tuple[int, ...],
        frame_id: str,
        method: str,
    ):
        shape = tuple(int(size) for size in case_shape)
        coefficient_array = jnp.asarray(coefficients)
        if coefficient_array.shape[: len(shape)] != shape:
            raise ValueError("Projection coefficients do not start with case_shape.")
        for name, value in (
            ("residual_energy", residual_energy),
            ("target_energy", target_energy),
            ("relative_residual", relative_residual),
            ("sample_count", sample_count),
            ("weight_sum", weight_sum),
            ("rank", rank),
            ("condition_number", condition_number),
            ("normal_equation_error", normal_equation_error),
            ("solved", solved),
            ("identified", identified),
            ("valid", valid),
            ("status", status),
        ):
            if jnp.asarray(value).shape != shape:
                raise ValueError(f"Projection {name} must have shape {shape}.")
        singular_array = jnp.asarray(singular_values)
        if singular_array.shape[: len(shape)] != shape:
            raise ValueError("Projection singular_values do not start with case_shape.")
        if not frame_id:
            raise ValueError("Projection frame_id must be nonempty.")
        if not method:
            raise ValueError("Projection method must be nonempty.")

        self.coefficients = coefficient_array
        self.residual_energy = jnp.asarray(residual_energy)
        self.target_energy = jnp.asarray(target_energy)
        self.relative_residual = jnp.asarray(relative_residual)
        self.singular_values = singular_array
        self.sample_count = jnp.asarray(sample_count, dtype=jnp.int32)
        self.weight_sum = jnp.asarray(weight_sum)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition_number = jnp.asarray(condition_number)
        self.normal_equation_error = jnp.asarray(normal_equation_error)
        self.solved = jnp.asarray(solved, dtype=bool)
        self.identified = jnp.asarray(identified, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.case_shape = shape
        self.frame_id = str(frame_id)
        self.method = str(method)

    def require_coefficients(self) -> Array:
        """Return coefficients or raise at runtime if any case is invalid."""
        return _raise_or_error(
            self.coefficients,
            jnp.any(~self.valid),
            "Function projection has invalid coefficients; inspect its report.",
        )


def _frame_coordinates(
    query: FunctionSamples,
    coord_dim: int,
    case_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array]:
    coordinates = query.coordinates_array(case_shape=case_shape)
    if int(coordinates.shape[-1]) != coord_dim:
        raise ValueError(
            "Function frame coordinate dimension does not match its model; got "
            f"{coordinates.shape[-1]} and {coord_dim}."
        )
    requested = query.mask_array(case_shape=case_shape)
    finite = jnp.all(jnp.isfinite(coordinates), axis=-1)
    usable = requested & finite
    safe_coordinates = jnp.where(usable[..., None], coordinates, 0.0)
    return safe_coordinates, usable


def _evaluate_pointwise_model(
    model: AbstractArrayModel,
    coordinates: Array,
    usable: Array,
    output_size: int,
    /,
    *,
    key: EvalKey,
) -> Array:
    flat_coordinates = coordinates.reshape((-1, int(coordinates.shape[-1])))
    evaluated = jax.vmap(lambda point: model(point, key=key))(flat_coordinates)
    values = jnp.asarray(evaluated).reshape(coordinates.shape[:-1] + (output_size,))
    return jnp.where(usable[..., None], values, jnp.zeros((), dtype=values.dtype))


def _canonical_sample_values(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    /,
) -> Array:
    if samples.values is None:
        raise ValueError("Function projection requires observed sample values.")
    values = jnp.asarray(samples.values)
    base_shape = case_shape + samples.sample_shape
    if tuple(int(size) for size in values.shape) == base_shape:
        if channels != 1:
            raise ValueError(
                "Vector-valued frame samples require an explicit channel axis."
            )
        return values[..., None]
    expected = base_shape + (channels,)
    if tuple(int(size) for size in values.shape) != expected:
        raise ValueError(
            f"Function sample values must have shape {base_shape} or {expected}; "
            f"got {values.shape}."
        )
    return values


def _sample_axes(
    case_shape: tuple[int, ...], sample_shape: tuple[int, ...]
) -> tuple[int, ...]:
    start = len(case_shape)
    return tuple(range(start, start + len(sample_shape)))


def _case_any(value: Array, axes: tuple[int, ...]) -> Array:
    return jnp.any(value, axis=axes)


def _case_sum(value: Array, axes: tuple[int, ...]) -> Array:
    return jnp.sum(value, axis=axes)


class AbstractFunctionFrameEvaluator(StrictModule):
    """Explicit coordinate, topology, or prepared-manifold frame evaluator."""

    @abstractmethod
    def evaluate(
        self,
        query: FunctionSamples,
        case_shape: tuple[int, ...],
        channels: int,
        rank: int,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        raise NotImplementedError


class TopologyFunctionFrameEvaluator(AbstractFunctionFrameEvaluator):
    """Evaluate a native graph/cochain entity model in canonical topology order."""

    model: AbstractArrayModel
    feature_name: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        feature_name: str,
        evaluator_id: str,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be AbstractArrayModel.")
        if not feature_name or not evaluator_id:
            raise ValueError("Topology evaluator names must be nonempty.")
        self.model = model
        self.feature_name = str(feature_name)
        self.evaluator_id = str(evaluator_id)

    def evaluate(
        self,
        query: FunctionSamples,
        case_shape: tuple[int, ...],
        channels: int,
        rank: int,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        topology = query.topology
        if topology is None:
            raise ValueError(
                "Topology frame evaluator requires FunctionSamples.topology."
            )
        if topology.entity == "node":
            payload = topology.graph.nodes
        elif topology.entity == "edge":
            payload = topology.graph.edges
        else:
            payload = topology.graph.globals
        if not isinstance(payload, Mapping) or self.feature_name not in payload:
            raise KeyError(
                f"Topology entity payload lacks feature {self.feature_name!r}."
            )
        entity_features = jnp.asarray(payload[self.feature_name])
        flat = entity_features.reshape((entity_features.shape[0], -1))
        entity_values = jax.vmap(lambda value: self.model(value, key=key))(flat)
        if entity_values.shape[-1:] != (channels * rank,):
            raise ValueError(
                "Topology frame model output must equal channels times frame rank."
            )
        gathered = gather_operator_graph_entities(
            query,
            entity_values,
            case_shape=case_shape,
        )
        return gathered.reshape(case_shape + query.sample_shape + (channels, rank))


class PreparedManifoldFunctionFrameEvaluator(AbstractFunctionFrameEvaluator):
    """Bind caller-prepared atlas/tangent/measure evaluation evidence."""

    evaluator: Callable[..., Array]
    evidence_id: str = eqx.field(static=True)

    def __init__(self, evaluator: Callable[..., Array], /, *, evidence_id: str):
        if not callable(evaluator) or not evidence_id:
            raise ValueError(
                "Prepared manifold evaluator requires callable and evidence id."
            )
        self.evaluator = evaluator
        self.evidence_id = str(evidence_id)

    def evaluate(
        self,
        query: FunctionSamples,
        case_shape: tuple[int, ...],
        channels: int,
        rank: int,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        values = jnp.asarray(
            self.evaluator(
                query,
                case_shape=case_shape,
                key=key,
            )
        )
        expected = case_shape + query.sample_shape + (channels, rank)
        if values.shape != expected:
            raise ValueError(
                f"Prepared manifold evaluator must return {expected}; got {values.shape}."
            )
        return values


class LearnedFunctionFrame(AbstractBasisTrunk):
    """Trainable coordinate-evaluated finite frame with weighted projection."""

    basis_model: AbstractArrayModel | None
    offset_model: AbstractArrayModel | None
    rank: int = eqx.field(static=True)
    evaluator: AbstractFunctionFrameEvaluator | None
    latent_size: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        basis_model: AbstractArrayModel | None = None,
        rank: int,
        coord_dim: int,
        out_size: int | Literal["scalar"] = "scalar",
        offset_model: AbstractArrayModel | None = None,
        evaluator: AbstractFunctionFrameEvaluator | None = None,
        frame_id: str,
    ):
        if basis_model is not None and not isinstance(
            basis_model,
            AbstractArrayModel,
        ):
            raise TypeError("basis_model must be an AbstractArrayModel or None.")
        if evaluator is not None and not isinstance(
            evaluator,
            AbstractFunctionFrameEvaluator,
        ):
            raise TypeError("evaluator must be AbstractFunctionFrameEvaluator or None.")
        if offset_model is not None and not isinstance(offset_model, AbstractArrayModel):
            raise TypeError("offset_model must be an AbstractArrayModel or None.")
        rank_ = int(rank)
        coord_dim_ = int(coord_dim)
        channels = _get_size(out_size)
        if rank_ <= 0 or coord_dim_ <= 0:
            raise ValueError("rank and coord_dim must be positive.")
        if basis_model is None and evaluator is None:
            raise ValueError(
                "A coordinate basis_model or explicit evaluator is required."
            )
        if basis_model is not None:
            if _get_size(basis_model.in_size) != coord_dim_:
                raise ValueError("basis_model.in_size must match coord_dim.")
            if _get_size(basis_model.out_size) != rank_ * channels:
                raise ValueError("basis_model.out_size must equal rank*out_size.")
        if offset_model is not None:
            if _get_size(offset_model.in_size) != coord_dim_:
                raise ValueError("offset_model.in_size must match coord_dim.")
            if _get_size(offset_model.out_size) != channels:
                raise ValueError("offset_model.out_size must match out_size.")
        identifier = str(frame_id)
        if not identifier:
            raise ValueError("frame_id must be non-empty.")

        self.basis_model = basis_model
        self.offset_model = offset_model
        self.rank = rank_
        self.latent_size = rank_
        self.coord_dim = coord_dim_
        self.evaluator = evaluator
        self.out_size = out_size
        self.frame_id = identifier

    @property
    def requires_fixed_query(self) -> bool:
        return False

    @property
    def channels(self) -> int:
        return _get_size(self.out_size)

    def evaluate(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        key: EvalKey = None,
    ) -> Array:
        cases = tuple(int(size) for size in case_shape)
        if self.evaluator is not None:
            values = self.evaluator.evaluate(
                query,
                cases,
                self.channels,
                self.rank,
                key=key,
            )
            mask = query.mask_array(case_shape=cases)
            return jnp.where(mask[..., None, None], values, 0.0)
        assert self.basis_model is not None
        coordinates, usable = _frame_coordinates(query, self.coord_dim, cases)
        flat_basis = _evaluate_pointwise_model(
            self.basis_model,
            coordinates,
            usable,
            self.channels * self.rank,
            key=fold_in_eval_key(key, 0),
        )
        return flat_basis.reshape(cases + query.sample_shape + (self.channels, self.rank))

    def evaluate_offset(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        key: EvalKey = None,
    ) -> Array:
        cases = tuple(int(size) for size in case_shape)
        offset_model = self.offset_model
        if offset_model is None:
            return jnp.zeros(cases + query.sample_shape + (self.channels,), dtype=float)
        coordinates, usable = _frame_coordinates(query, self.coord_dim, cases)
        return _evaluate_pointwise_model(
            offset_model,
            coordinates,
            usable,
            self.channels,
            key=fold_in_eval_key(key, 1),
        )

    def decode(
        self,
        coefficients: Array,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        key: EvalKey = None,
    ) -> Array:
        cases = tuple(int(size) for size in case_shape)
        coefficient_array = jnp.asarray(coefficients)
        expected = cases + (self.rank,)
        if tuple(int(size) for size in coefficient_array.shape) != expected:
            raise ValueError(
                f"Function coefficients must have shape {expected}; "
                f"got {coefficient_array.shape}."
            )
        basis = self.evaluate(query, case_shape=cases, key=key)
        offset = self.evaluate_offset(query, case_shape=cases, key=key)
        coefficient_shape = cases + (1,) * len(query.sample_shape) + (1, self.rank)
        output = offset + jnp.sum(
            basis * coefficient_array.reshape(coefficient_shape),
            axis=-1,
        )
        mask = query.mask_array(case_shape=cases)
        output = output * mask[..., None]
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def project(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        policy: FunctionProjectionPolicy | None = None,
        key: EvalKey = None,
    ) -> FunctionProjectionReport:
        cases = tuple(int(size) for size in case_shape)
        resolved_policy = FunctionProjectionPolicy() if policy is None else policy
        if not isinstance(resolved_policy, FunctionProjectionPolicy):
            raise TypeError("policy must be a FunctionProjectionPolicy or None.")
        if (
            resolved_policy.channel_metric is not None
            and resolved_policy.channel_metric.shape != (self.channels, self.channels)
        ):
            raise ValueError("channel_metric size must match the frame output channels.")

        target = _canonical_sample_values(samples, cases, self.channels)
        coordinates = samples.coordinates_array(case_shape=cases)
        requested = samples.mask_array(case_shape=cases)
        quadrature = samples.quadrature(case_shape=cases)
        coordinate_finite = jnp.all(jnp.isfinite(coordinates), axis=-1)
        quadrature_finite = jnp.isfinite(quadrature)
        valid_measure = quadrature_finite & (quadrature >= 0.0) & coordinate_finite
        positive_measure = valid_measure & (quadrature > 0.0)
        measure_active = requested & positive_measure
        invalid_measure_sites = requested & ~valid_measure

        basis = self.evaluate(samples, case_shape=cases, key=key)
        offset = self.evaluate_offset(samples, case_shape=cases, key=key)
        target_finite = jnp.all(jnp.isfinite(target), axis=-1)
        frame_finite = jnp.all(jnp.isfinite(basis), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(offset), axis=-1
        )
        finite_site = target_finite & frame_finite
        active = measure_active & finite_site
        axes = _sample_axes(cases, samples.sample_shape)
        invalid_measure = _case_any(invalid_measure_sites, axes)
        if (
            resolved_policy.require_physical_quadrature
            and not samples.has_physical_quadrature
        ):
            invalid_measure = jnp.ones(cases, dtype=bool)
        nonfinite = _case_any(measure_active & ~finite_site, axes)

        safe_target = jnp.where(active[..., None], target, 0.0)
        safe_basis = jnp.where(active[..., None, None], basis, 0.0)
        safe_offset = jnp.where(active[..., None], offset, 0.0)
        centered_target = safe_target - safe_offset

        factor = resolved_policy.channel_factor
        if factor is None:
            metric_basis = safe_basis
            metric_target = centered_target
        else:
            metric_basis = ein.contract("...cr,cd->...dr", safe_basis, factor)
            metric_target = ein.contract("...c,cd->...d", centered_target, factor)

        physical_weights = jnp.where(active, quadrature, 0.0)
        weight_sum = _case_sum(physical_weights, axes)
        normalization = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        normalized_weights = physical_weights / normalization.reshape(
            cases + (1,) * len(samples.sample_shape)
        )
        equation_weights = jnp.broadcast_to(
            normalized_weights[..., None] / float(self.channels),
            cases + samples.sample_shape + (self.channels,),
        )
        equation_count = prod(samples.sample_shape) * self.channels
        design = metric_basis.reshape(cases + (equation_count, self.rank))
        response = metric_target.reshape(cases + (equation_count,))
        weights = equation_weights.reshape(cases + (equation_count,))
        case_count = prod(cases) if cases else 1
        flat_design = design.reshape((case_count, equation_count, self.rank))
        flat_response = response.reshape((case_count, equation_count))
        flat_weights = weights.reshape((case_count, equation_count))

        def solve_case(matrix, values, equation_weight):
            return solve_weighted_least_squares(
                matrix,
                values,
                weights=equation_weight,
                center=False,
                scale=resolved_policy.scale_frame,
                ridge=resolved_policy.ridge,
                rcond=resolved_policy.rcond,
                min_samples=1,
                max_features=self.rank,
            )

        linear_result = jax.vmap(solve_case)(
            flat_design,
            flat_response,
            flat_weights,
        )
        coefficients = linear_result.raw_coefficients.reshape(cases + (self.rank,))
        singular_values = linear_result.singular_values.reshape(
            cases + (linear_result.singular_values.shape[-1],)
        )
        rank = linear_result.rank.reshape(cases)
        condition_number = linear_result.condition_number.reshape(cases)
        normal_equation_error = linear_result.normal_equation_error.reshape(cases)
        solved = linear_result.valid.reshape(cases)
        sample_count = _case_sum(active.astype(jnp.int32), axes).astype(jnp.int32)
        minimum_sites = (
            1 if resolved_policy.min_samples is None else resolved_policy.min_samples
        )
        enough_support = (sample_count >= minimum_sites) & (
            sample_count * self.channels >= self.rank
        )
        identified = enough_support & (rank == self.rank) & ~invalid_measure & ~nonfinite
        rank_deficient = enough_support & (rank < self.rank)
        regularized = (
            rank_deficient
            & (resolved_policy.rank_policy == "regularized")
            & ~invalid_measure
            & ~nonfinite
        )
        valid = (
            solved
            & ~invalid_measure
            & ~nonfinite
            & enough_support
            & (identified | regularized)
        )
        status = jnp.where(
            invalid_measure,
            FUNCTION_PROJECTION_INVALID_MEASURE,
            jnp.where(
                nonfinite,
                FUNCTION_PROJECTION_NONFINITE,
                jnp.where(
                    ~enough_support,
                    FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT,
                    jnp.where(
                        rank_deficient,
                        jnp.where(
                            regularized,
                            FUNCTION_PROJECTION_REGULARIZED,
                            FUNCTION_PROJECTION_RANK_DEFICIENT,
                        ),
                        FUNCTION_PROJECTION_SUCCESS,
                    ),
                ),
            ),
        ).astype(jnp.int32)

        if resolved_policy.rank_policy == "error":
            coefficients = _raise_or_error(
                coefficients,
                jnp.any(rank_deficient & ~invalid_measure & ~nonfinite),
                "Observed function samples do not identify the learned frame.",
            )

        coefficient_shape = cases + (1,) * len(samples.sample_shape) + (1, self.rank)
        prediction = safe_offset + jnp.sum(
            safe_basis * coefficients.reshape(coefficient_shape),
            axis=-1,
        )
        residual = safe_target - prediction
        if factor is not None:
            residual = ein.contract("...c,cd->...d", residual, factor)
        residual_density = jnp.sum(jnp.abs(residual) ** 2, axis=-1)
        target_density = jnp.sum(jnp.abs(metric_target) ** 2, axis=-1)
        residual_energy = _case_sum(
            normalized_weights * residual_density,
            axes,
        )
        target_energy = _case_sum(
            normalized_weights * target_density,
            axes,
        )
        positive_target = target_energy > 0.0
        relative_residual = jnp.where(
            positive_target,
            jnp.sqrt(residual_energy / jnp.where(positive_target, target_energy, 1.0)),
            jnp.where(residual_energy == 0.0, 0.0, jnp.inf),
        )

        return FunctionProjectionReport(
            coefficients=coefficients,
            residual_energy=residual_energy,
            target_energy=target_energy,
            relative_residual=relative_residual,
            singular_values=singular_values,
            sample_count=sample_count,
            weight_sum=weight_sum,
            rank=rank,
            condition_number=condition_number,
            normal_equation_error=normal_equation_error,
            solved=solved,
            identified=identified,
            valid=valid,
            status=status,
            case_shape=cases,
            frame_id=self.frame_id,
            method="weighted-scaled-ridge-svd",
        )

    def frozen(self) -> "LearnedFunctionFrame":
        basis_model = (
            None
            if self.basis_model is None
            else self.basis_model
            if isinstance(self.basis_model, FrozenModel)
            else FrozenModel(self.basis_model)
        )
        offset_model = self.offset_model
        if offset_model is not None and not isinstance(offset_model, FrozenModel):
            offset_model = FrozenModel(offset_model)
        evaluator = self.evaluator
        if isinstance(evaluator, TopologyFunctionFrameEvaluator) and not isinstance(
            evaluator.model,
            FrozenModel,
        ):
            evaluator = TopologyFunctionFrameEvaluator(
                FrozenModel(evaluator.model),
                feature_name=evaluator.feature_name,
                evaluator_id=evaluator.evaluator_id,
            )
        return LearnedFunctionFrame(
            basis_model=basis_model,
            offset_model=offset_model,
            rank=self.rank,
            coord_dim=self.coord_dim,
            out_size=self.out_size,
            evaluator=evaluator,
            frame_id=self.frame_id,
        )


class ProjectionBranchEncoder(AbstractBranchEncoder):
    """DeepONet branch that projects one sampled function into frame coordinates."""

    frame: LearnedFunctionFrame
    policy: FunctionProjectionPolicy
    coefficient_map: AbstractArrayModel | None
    latent_size: int = eqx.field(static=True)

    def __init__(
        self,
        frame: LearnedFunctionFrame,
        /,
        *,
        policy: FunctionProjectionPolicy | None = None,
        coefficient_map: AbstractArrayModel | None = None,
        latent_size: int | None = None,
    ):
        if not isinstance(frame, LearnedFunctionFrame):
            raise TypeError("frame must be a LearnedFunctionFrame.")
        resolved_policy = FunctionProjectionPolicy() if policy is None else policy
        if not isinstance(resolved_policy, FunctionProjectionPolicy):
            raise TypeError("policy must be a FunctionProjectionPolicy or None.")
        if coefficient_map is not None and not isinstance(
            coefficient_map, AbstractArrayModel
        ):
            raise TypeError("coefficient_map must be an AbstractArrayModel or None.")
        output_rank = frame.rank if latent_size is None else int(latent_size)
        if output_rank <= 0:
            raise ValueError("latent_size must be positive.")
        if coefficient_map is None:
            if output_rank != frame.rank:
                raise ValueError(
                    "Without a coefficient_map, latent_size must equal the frame rank."
                )
        else:
            if _get_size(coefficient_map.in_size) != frame.rank:
                raise ValueError(
                    "coefficient_map.in_size must equal the source frame rank."
                )
            if _get_size(coefficient_map.out_size) != output_rank:
                raise ValueError("coefficient_map.out_size must equal latent_size.")
        self.frame = frame
        self.policy = resolved_policy
        self.coefficient_map = coefficient_map
        self.latent_size = output_rank

    def project(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...],
        key: EvalKey = None,
    ) -> FunctionProjectionReport:
        return self.frame.project(
            samples,
            case_shape=case_shape,
            policy=self.policy,
            key=key,
        )

    def map_coefficients(
        self,
        coefficients: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        coefficient_map = self.coefficient_map
        if coefficient_map is None:
            return coefficients
        values = jnp.asarray(coefficients)
        case_shape = values.shape[:-1]
        flat = values.reshape((-1, self.frame.rank))
        mapped = jax.vmap(
            lambda value: coefficient_map(
                value,
                key=fold_in_eval_key(key, 2),
            )
        )(flat)
        return jnp.asarray(mapped).reshape(case_shape + (self.latent_size,))

    def __call__(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_ndim: int,
        key: EvalKey = None,
    ) -> Array:
        if samples.values is None:
            raise ValueError("Projection branch requires source function values.")
        values = jnp.asarray(samples.values)
        case_shape = tuple(int(size) for size in values.shape[: int(case_ndim)])
        report = self.project(samples, case_shape=case_shape, key=key)
        return self.map_coefficients(report.require_coefficients(), key=key)


class FunctionFrameSource(StrictModule):
    """One explicitly named source frame and its coefficient map."""

    name: str = eqx.field(static=True)
    frame: LearnedFunctionFrame
    projection_policy: FunctionProjectionPolicy
    coefficient_map: AbstractArrayModel | None

    def __init__(
        self,
        name: str,
        frame: LearnedFunctionFrame,
        /,
        *,
        projection_policy: FunctionProjectionPolicy | None = None,
        coefficient_map: AbstractArrayModel | None = None,
    ):
        if not name:
            raise ValueError("FunctionFrameSource name must be nonempty.")
        if not isinstance(frame, LearnedFunctionFrame):
            raise TypeError("frame must be LearnedFunctionFrame.")
        policy = (
            FunctionProjectionPolicy() if projection_policy is None else projection_policy
        )
        if not isinstance(policy, FunctionProjectionPolicy):
            raise TypeError("projection_policy must be FunctionProjectionPolicy.")
        if coefficient_map is not None and not isinstance(
            coefficient_map,
            AbstractArrayModel,
        ):
            raise TypeError("coefficient_map must be AbstractArrayModel or None.")
        self.name = str(name)
        self.frame = frame
        self.projection_policy = policy
        self.coefficient_map = coefficient_map


class FunctionFrameEncoding(StrictModule):
    """Reusable ordered multi-source coefficients and projection evidence."""

    coefficients: frozendict[str, Array]
    reports: frozendict[str, FunctionProjectionReport]
    frame_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    fused_coefficients: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    fusion: Literal["sum", "product", "concat"] = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Mapping[str, Array],
        reports: Mapping[str, FunctionProjectionReport],
        fused_coefficients: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        frame_ids: Sequence[tuple[str, str]],
        fusion: Literal["sum", "product", "concat"],
    ):
        coefficient_values = frozendict(
            (str(name), jnp.asarray(value)) for name, value in coefficients.items()
        )
        report_values = frozendict(reports)
        if tuple(coefficient_values) != tuple(report_values):
            raise ValueError("Encoding coefficient and report source order must agree.")
        shape = tuple(int(size) for size in case_shape)
        if any(value.shape[:-1] != shape for value in coefficient_values.values()):
            raise ValueError(
                "Every source coefficient must have the encoding case shape."
            )
        fused = jnp.asarray(fused_coefficients)
        if fused.shape[:-1] != shape:
            raise ValueError("Fused coefficients must have the encoding case shape.")
        identities = tuple((str(name), str(frame_id)) for name, frame_id in frame_ids)
        if tuple(name for name, _ in identities) != tuple(coefficient_values):
            raise ValueError("Frame identities must follow source order.")
        for name, frame_id in identities:
            if report_values[name].frame_id != frame_id:
                raise ValueError(
                    f"Projection report for {name!r} has a stale frame identity."
                )
        self.coefficients = coefficient_values
        self.reports = report_values
        self.frame_ids = identities
        self.fused_coefficients = fused
        self.case_shape = shape
        self.fusion = fusion


class FunctionFrameReconstructor(AbstractEncodedOperatorModel):
    """Project ordered named sources and decode their fused target coefficients."""

    operator_architecture: ClassVar[str] = "FunctionFrameReconstructor"

    operator: DeepONet
    sources: tuple[FunctionFrameSource, ...]
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        sources: Sequence[FunctionFrameSource] | Mapping[str, FunctionFrameSource],
        target_frame: LearnedFunctionFrame,
        fusion: Literal["sum", "product", "concat"] = "sum",
        branch_mixer: AbstractArrayModel | None = None,
    ):
        source_values = (
            tuple(sources.values()) if isinstance(sources, Mapping) else tuple(sources)
        )
        if not source_values or any(
            not isinstance(source, FunctionFrameSource) for source in source_values
        ):
            raise TypeError("sources must contain FunctionFrameSource values.")
        names = tuple(source.name for source in source_values)
        if len(set(names)) != len(names):
            raise ValueError("Function-frame source names must be unique.")
        if not isinstance(target_frame, LearnedFunctionFrame):
            raise TypeError("target_frame must be LearnedFunctionFrame.")
        branches = {
            source.name: ProjectionBranchEncoder(
                source.frame,
                policy=source.projection_policy,
                coefficient_map=source.coefficient_map,
                latent_size=target_frame.rank,
            )
            for source in source_values
        }
        self.operator = DeepONet(
            branch=branches,
            trunk=target_frame,
            coord_dim=target_frame.coord_dim,
            latent_size=target_frame.rank,
            out_size=target_frame.out_size,
            in_size=source_values[0].frame.out_size,
            fusion=fusion,
            branch_mixer=branch_mixer,
            source_key=None,
            use_bias=False,
        )
        self.sources = source_values
        self.in_size = source_values[0].frame.out_size
        self.out_size = target_frame.out_size
        self.coord_dim = target_frame.coord_dim
        self.latent_size = target_frame.rank

    @property
    def target_frame(self) -> LearnedFunctionFrame:
        trunk = self.operator.trunk
        if not isinstance(trunk, LearnedFunctionFrame):
            raise TypeError("Function-frame operator has an invalid target frame.")
        return trunk

    def _fuse(self, values: tuple[Array, ...], /, *, key: EvalKey) -> Array:
        if self.operator.fusion == "sum":
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result / jnp.sqrt(float(len(values)))
        if self.operator.fusion == "product":
            result = values[0]
            for value in values[1:]:
                result = result * value
            return result
        mixer = self.operator.branch_mixer
        assert mixer is not None
        concatenated = jnp.concatenate(values, axis=-1)
        flat = concatenated.reshape((-1, concatenated.shape[-1]))
        mixed = jax.vmap(lambda value: mixer(value, key=key))(flat)
        return mixed.reshape(concatenated.shape[:-1] + (self.latent_size,))

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> FunctionFrameEncoding:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("FunctionFrameReconstructor requires OperatorBatch.")
        if tuple(batch.inputs) != tuple(source.name for source in self.sources):
            raise ValueError("Operator inputs must exactly match ordered frame sources.")
        coefficients: dict[str, Array] = {}
        reports: dict[str, FunctionProjectionReport] = {}
        mapped = []
        for index, source in enumerate(self.sources):
            branch = self.operator.branches[source.name]
            if not isinstance(branch, ProjectionBranchEncoder):
                raise TypeError("Invalid projection branch.")
            report = branch.project(
                batch.input(source.name),
                case_shape=batch.case_shape,
                key=fold_in_eval_key(key, 2 * index),
            )
            value = branch.map_coefficients(
                report.require_coefficients(),
                key=fold_in_eval_key(key, 2 * index + 1),
            )
            coefficients[source.name] = value
            reports[source.name] = report
            mapped.append(value)
        fused = self._fuse(tuple(mapped), key=fold_in_eval_key(key, 2 * len(mapped)))
        return FunctionFrameEncoding(
            coefficients,
            reports,
            fused,
            case_shape=batch.case_shape,
            frame_ids=tuple(
                (source.name, source.frame.frame_id) for source in self.sources
            ),
            fusion=self.operator.fusion,
        )

    def decode_query(
        self,
        state: FunctionFrameEncoding,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(state, FunctionFrameEncoding):
            raise TypeError("state must be FunctionFrameEncoding.")
        expected = tuple((source.name, source.frame.frame_id) for source in self.sources)
        if state.frame_ids != expected or state.fusion != self.operator.fusion:
            raise ValueError("Encoded state frame epoch or fusion identity is stale.")
        return self.target_frame.decode(
            state.fused_coefficients,
            query,
            case_shape=state.case_shape,
            key=key,
        )

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        return self.__call_operator_batch__(batch, key=key)

    def frozen(self) -> "FunctionFrameReconstructor":
        frozen_sources = tuple(
            FunctionFrameSource(
                source.name,
                source.frame.frozen(),
                projection_policy=source.projection_policy,
                coefficient_map=(
                    None
                    if source.coefficient_map is None
                    else FrozenModel(source.coefficient_map)
                ),
            )
            for source in self.sources
        )
        mixer = self.operator.branch_mixer
        return FunctionFrameReconstructor(
            sources=frozen_sources,
            target_frame=self.target_frame.frozen(),
            fusion=self.operator.fusion,
            branch_mixer=None if mixer is None else FrozenModel(mixer),
        )


for _artifact_id, _artifact_value in (
    ("phydrax.operator.function_frame:FunctionFrameEncoding@1", FunctionFrameEncoding),
    (
        "phydrax.operator.function_frame:FunctionProjectionPolicy@1",
        FunctionProjectionPolicy,
    ),
    (
        "phydrax.operator.function_frame:FunctionProjectionReport@1",
        FunctionProjectionReport,
    ),
    (
        "phydrax.operator.function_frame:LearnedFunctionFrame@1",
        LearnedFunctionFrame,
    ),
    (
        "phydrax.operator.function_frame:ProjectionBranchEncoder@1",
        ProjectionBranchEncoder,
    ),
    (
        "phydrax.operator.function_frame:FunctionFrameSource@1",
        FunctionFrameSource,
    ),
    (
        "phydrax.operator.function_frame:TopologyFunctionFrameEvaluator@1",
        TopologyFunctionFrameEvaluator,
    ),
):
    register_artifact_value(_artifact_id, _artifact_value)

del _artifact_id, _artifact_value


__all__ = [
    "AbstractFunctionFrameEvaluator",
    "FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT",
    "FUNCTION_PROJECTION_INVALID_MEASURE",
    "FUNCTION_PROJECTION_NONFINITE",
    "FUNCTION_PROJECTION_RANK_DEFICIENT",
    "FUNCTION_PROJECTION_REGULARIZED",
    "FUNCTION_PROJECTION_SUCCESS",
    "FunctionFrameEncoding",
    "FunctionFrameReconstructor",
    "FunctionFrameSource",
    "PreparedManifoldFunctionFrameEvaluator",
    "TopologyFunctionFrameEvaluator",
    "FunctionProjectionPolicy",
    "FunctionProjectionReport",
    "FunctionProjectionRankPolicy",
    "LearnedFunctionFrame",
    "ProjectionBranchEncoder",
]
