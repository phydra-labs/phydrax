#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    AbstractSparseLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DiagonalLinearOperator,
    OperatorProperties,
    RankPolicy,
)
from ..linalg.eigen import SelfAdjointSpectrumPolicy
from ..sparse import EdgeRelation, SparseLinearMap


class ExactMoments(StrictModule):
    """A vector of moments requested exactly on a regular interior branch."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        values_ = _moment_values(values)
        self.values = values_


class IntervalMoments(StrictModule):
    """Hard componentwise moment intervals for a general calibration route."""

    lower: Array
    upper: Array
    values: Array

    def __init__(self, lower: ArrayLike, upper: ArrayLike, /):
        lower_ = _moment_values(lower)
        upper_ = _moment_values(upper).astype(lower_.dtype)
        if upper_.shape != lower_.shape:
            raise ValueError("Moment interval endpoints must have identical shapes.")
        invalid = jnp.any(
            ~jnp.isfinite(lower_) | ~jnp.isfinite(upper_) | (lower_ > upper_)
        )
        if isinstance(invalid, jax_core.Tracer):
            lower_ = eqx.error_if(
                lower_, invalid, "Moment intervals must be finite with lower <= upper."
            )
        elif bool(invalid):
            raise ValueError("Moment intervals must be finite with lower <= upper.")
        self.lower = lower_
        self.upper = upper_
        self.values = 0.5 * (lower_ + upper_)


class QuadraticMoments(StrictModule):
    """Moment targets reconciled by one PSD covariance operator."""

    values: Array
    covariance: AbstractLinearOperator

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        covariance: ArrayLike | AbstractLinearOperator | None = None,
    ):
        values_ = _moment_values(values)
        count = int(values_.shape[0])
        if covariance is None:
            operator = DiagonalLinearOperator(
                jnp.ones_like(values_),
                properties=OperatorProperties(
                    diagonal=True,
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={
                        "diagonal": "construction",
                        "self_adjoint": "construction",
                        "positive_definite": "construction",
                    },
                ),
            )
        elif isinstance(covariance, AbstractLinearOperator):
            operator = covariance
        else:
            matrix = np.asarray(covariance, dtype=np.dtype(values_.dtype))
            if matrix.shape != (count, count):
                raise ValueError(
                    f"covariance must have shape ({count}, {count}); got {matrix.shape}."
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("Moment covariance must be finite.")
            symmetric = 0.5 * (matrix + matrix.T)
            tolerance = (
                64.0
                * np.finfo(symmetric.dtype).eps
                * max(float(np.max(np.abs(symmetric))), 1.0)
            )
            if np.min(np.linalg.eigvalsh(symmetric)) < -tolerance:
                raise ValueError("Moment covariance must be positive semidefinite.")
            operator = DenseLinearOperator(
                jnp.asarray(symmetric),
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_semidefinite=True,
                    evidence={
                        "self_adjoint": "verified",
                        "positive_semidefinite": "verified",
                    },
                ),
            )
        if (
            not isinstance(operator.source, ArraySpace)
            or not operator.source.compatible(operator.target)
            or operator.source.shape != (count,)
            or operator.batch_shape
        ):
            raise ValueError(
                "Moment covariance must be an unbatched endomorphism on the target."
            )
        if operator.source.dtype != np.dtype(values_.dtype):
            raise TypeError("Moment covariance dtype must match target moments.")
        evidence = operator.properties.evidence_for("positive_semidefinite")
        if (
            not operator.properties.self_adjoint
            or not operator.properties.positive_semidefinite
            or evidence not in ("construction", "transformed", "verified")
        ):
            raise ValueError(
                "Moment covariance requires constructive or verified PSD evidence."
            )
        self.values = values_
        self.covariance = operator


MomentTarget: TypeAlias = ExactMoments | IntervalMoments | QuadraticMoments


class GroupMassConstraints(StrictModule):
    """Exact or interval masses applied through one prepared sparse group map."""

    group_map: AbstractSparseLinearOperator
    target: ExactMoments | IntervalMoments
    group_count: int = eqx.field(static=True)

    def __init__(
        self,
        group_map: AbstractSparseLinearOperator,
        target: ExactMoments | IntervalMoments,
        /,
    ):
        if not isinstance(group_map, AbstractSparseLinearOperator):
            raise TypeError("group_map must be an AbstractSparseLinearOperator.")
        if not isinstance(target, (ExactMoments, IntervalMoments)):
            raise TypeError("Group mass target must be exact or interval moments.")
        if (
            not isinstance(group_map.source, ArraySpace)
            or not isinstance(group_map.target, ArraySpace)
            or len(group_map.source.shape) != 1
            or len(group_map.target.shape) != 1
        ):
            raise ValueError("Group maps must act between one-dimensional ArraySpaces.")
        groups = int(group_map.target.shape[0])
        if target.values.shape != (groups,):
            raise ValueError(f"Group target must have shape ({groups},).")
        self.group_map = group_map
        self.target = target
        self.group_count = groups


def stratified_group_constraints(
    labels: ArrayLike,
    target: ExactMoments | IntervalMoments,
    /,
    *,
    active: ArrayLike | None = None,
) -> GroupMassConstraints:
    """Compile one disjoint active partition into the group-mass contract."""

    labels_ = np.asarray(labels)
    if labels_.ndim != 1 or not np.issubdtype(labels_.dtype, np.integer):
        raise TypeError("labels must be one integer vector.")
    active_ = (
        np.ones(labels_.shape, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool)
    )
    if active_.shape != labels_.shape:
        raise ValueError("active must match labels.")
    groups = int(target.values.shape[0])
    if np.any(labels_[active_] < 0) or np.any(labels_[active_] >= groups):
        raise ValueError("Every active source point must belong to one target group.")
    sources = np.flatnonzero(active_).astype(np.int32)
    destinations = labels_[active_].astype(np.int32)
    relation = EdgeRelation(
        sources,
        destinations,
        source_size=int(labels_.shape[0]),
        target_size=groups,
    )
    group_map = SparseLinearMap(
        relation,
        jnp.ones((sources.size,), dtype=target.values.dtype),
    )
    return GroupMassConstraints(group_map, target)


class EqualWeightSubset(StrictModule):
    """Select exactly ``cardinality`` atoms with weights ``z_i / cardinality``."""

    cardinality: int = eqx.field(static=True)

    def __init__(self, cardinality: int, /):
        if isinstance(cardinality, bool) or int(cardinality) <= 0:
            raise ValueError("cardinality must be a positive integer.")
        self.cardinality = int(cardinality)


class BoundaryFacePolicy(StrictModule):
    """Bounded exact-target face discovery and forced-zero tolerance."""

    maximum_linear_programs: int = eqx.field(static=True)
    zero_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_linear_programs: int = 10_000,
        zero_tolerance: float = 1e-10,
    ):
        programs = int(maximum_linear_programs)
        tolerance = float(zero_tolerance)
        if programs <= 0 or not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Boundary face capacity/tolerance is invalid.")
        self.maximum_linear_programs = programs
        self.zero_tolerance = tolerance


class MomentCalibrationExecutionPolicy(StrictModule):
    """Explicit calibration route; unsupported structures never switch silently."""

    solver: object | None

    route: str = eqx.field(static=True)

    def __init__(self, route: str = "dual-relative-entropy", /, *, solver=None):
        if route not in (
            "dual-relative-entropy",
            "canonical-conic",
            "mixed-integer",
        ):
            raise ValueError("Unknown moment calibration execution route.")
        self.route = route

        self.solver = solver


class MomentCalibrationPolicy(StrictModule):
    """Affine-rank, regularity, and bounded geometry policy for calibration."""

    rank: RankPolicy
    spectrum: SelfAdjointSpectrumPolicy
    affine_absolute_tolerance: float = eqx.field(static=True)
    affine_relative_tolerance: float = eqx.field(static=True)
    regularity_relative_tolerance: float = eqx.field(static=True)
    maximum_moments: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        rank: RankPolicy | None = None,
        spectrum: SelfAdjointSpectrumPolicy | None = None,
        affine_absolute_tolerance: float = 1e-10,
        affine_relative_tolerance: float = 1e-8,
        regularity_relative_tolerance: float = 1e-10,
        maximum_moments: int = 512,
    ):
        rank_ = RankPolicy() if rank is None else rank
        spectrum_ = SelfAdjointSpectrumPolicy() if spectrum is None else spectrum
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be a RankPolicy or None.")
        if not isinstance(spectrum_, SelfAdjointSpectrumPolicy):
            raise TypeError("spectrum must be a SelfAdjointSpectrumPolicy or None.")
        tolerances = (
            float(affine_absolute_tolerance),
            float(affine_relative_tolerance),
            float(regularity_relative_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Calibration tolerances must be finite and non-negative.")
        maximum = int(maximum_moments)
        if maximum < 1:
            raise ValueError("maximum_moments must be positive.")
        self.rank = rank_
        self.spectrum = spectrum_
        (
            self.affine_absolute_tolerance,
            self.affine_relative_tolerance,
            self.regularity_relative_tolerance,
        ) = tolerances
        self.maximum_moments = maximum


class MomentCalibrationProblem(StrictModule):
    """A finite prior, linear moment map, and exact or soft target moments."""

    moment_map: AbstractLinearOperator
    target: MomentTarget
    prior_log_weights: Array
    mask: Array
    source_points: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    group_constraints: GroupMassConstraints | None
    subset: EqualWeightSubset | None
    boundary: BoundaryFacePolicy | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        features: ArrayLike | AbstractLinearOperator,
        target: MomentTarget,
        /,
        *,
        prior_log_weights: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        group_constraints: GroupMassConstraints | None = None,
        subset: EqualWeightSubset | None = None,
        boundary: BoundaryFacePolicy | None = None,
        problem_id: str | None = None,
    ):
        moment_map = _moment_operator(features)
        if not isinstance(target, (ExactMoments, IntervalMoments, QuadraticMoments)):
            raise TypeError(
                "target must be ExactMoments, IntervalMoments, or QuadraticMoments."
            )
        source_points = int(moment_map.source.shape[0])
        moment_count = int(moment_map.target.shape[0])
        if target.values.shape != (moment_count,):
            raise ValueError(
                f"Target moments must have shape ({moment_count},); "
                f"got {target.values.shape}."
            )
        dtype = moment_map.source.dtype
        if isinstance(target, ExactMoments):
            target_ = ExactMoments(target.values.astype(dtype))
        elif isinstance(target, IntervalMoments):
            target_ = IntervalMoments(
                target.lower.astype(dtype), target.upper.astype(dtype)
            )
        else:
            if target.covariance.source.dtype != dtype:
                raise TypeError(
                    "Quadratic moment covariance dtype must match the moment map."
                )
            target_ = QuadraticMoments(
                target.values.astype(dtype),
                covariance=target.covariance,
            )
        if prior_log_weights is None:
            prior = jnp.zeros((source_points,), dtype=dtype)
        else:
            prior = jnp.asarray(prior_log_weights, dtype=dtype)
            if prior.shape != (source_points,):
                raise ValueError(f"prior_log_weights must have shape ({source_points},).")
        if mask is None:
            mask_ = jnp.ones((source_points,), dtype=bool)
        else:
            mask_ = jnp.asarray(mask, dtype=bool)
            if mask_.shape != (source_points,):
                raise ValueError(f"mask must have shape ({source_points},).")
        if group_constraints is not None:
            if not isinstance(group_constraints, GroupMassConstraints):
                raise TypeError("group_constraints must be GroupMassConstraints or None.")
            if group_constraints.group_map.source.shape != (source_points,):
                raise ValueError("Group map source size must match calibration support.")
        if subset is not None:
            if not isinstance(subset, EqualWeightSubset):
                raise TypeError("subset must be EqualWeightSubset or None.")
            if subset.cardinality > int(np.sum(np.asarray(mask_))):
                raise ValueError(
                    "Equal-weight subset cardinality exceeds active support size."
                )
        if boundary is not None and not isinstance(boundary, BoundaryFacePolicy):
            raise TypeError("boundary must be BoundaryFacePolicy or None.")
        self.moment_map = moment_map
        self.target = target_
        self.prior_log_weights = prior
        self.mask = mask_
        self.source_points = source_points
        self.moment_count = moment_count
        self.group_constraints = group_constraints
        self.subset = subset
        self.boundary = boundary
        self.problem_id = (
            canonical_fingerprint(
                {
                    "kind": "moment-calibration",
                    "operator": moment_map.operator_id,
                    "source_points": source_points,
                    "moment_count": moment_count,
                    "target": type(target_).__name__,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty.")


def _moment_values(values: ArrayLike, /) -> Array:
    values_ = jnp.asarray(values)
    if values_.ndim != 1 or values_.shape[0] == 0:
        raise ValueError("Moment values must be a non-empty one-dimensional array.")
    if jnp.issubdtype(values_.dtype, jnp.complexfloating):
        raise TypeError("Moment values must be real.")
    if not jnp.issubdtype(values_.dtype, jnp.inexact):
        values_ = values_.astype(float)
    invalid = ~jnp.all(jnp.isfinite(values_))
    if isinstance(invalid, jax_core.Tracer):
        values_ = eqx.error_if(
            values_,
            invalid,
            "Moment target values must be finite.",
        )
    elif bool(invalid):
        raise ValueError("Moment target values must be finite.")
    return values_


def _moment_operator(
    features: ArrayLike | AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    if isinstance(features, AbstractLinearOperator):
        operator = features
    else:
        values = jnp.asarray(features)
        if values.ndim != 2 or any(size == 0 for size in values.shape):
            raise ValueError(
                "Dense features must have shape (source_points, moment_count)."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Moment features must be real.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        invalid = ~jnp.all(jnp.isfinite(values))
        if isinstance(invalid, jax_core.Tracer):
            values = eqx.error_if(
                values,
                invalid,
                "Dense moment features must be finite.",
            )
        elif bool(invalid):
            raise ValueError("Dense moment features must be finite.")
        operator = DenseLinearOperator(jnp.swapaxes(values, 0, 1))
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("Moment maps must act between ArraySpace values.")
    if len(operator.source.shape) != 1 or len(operator.target.shape) != 1:
        raise ValueError("Moment maps must act between one-dimensional arrays.")
    if operator.batch_shape:
        raise ValueError("Batched moment maps are not supported.")
    if not operator.capabilities.transpose:
        raise ValueError("Moment maps require a transpose action.")
    if operator.source.dtype != operator.target.dtype:
        raise TypeError("Moment-map source and target dtypes must match.")
    if not np.issubdtype(operator.source.dtype, np.floating):
        raise TypeError("Moment maps must have a real floating-point dtype.")
    return operator


__all__ = [
    "BoundaryFacePolicy",
    "EqualWeightSubset",
    "ExactMoments",
    "GroupMassConstraints",
    "IntervalMoments",
    "MomentCalibrationExecutionPolicy",
    "MomentCalibrationPolicy",
    "MomentCalibrationProblem",
    "MomentTarget",
    "QuadraticMoments",
    "stratified_group_constraints",
]
