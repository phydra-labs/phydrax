#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import tree_allfinite
from ..linalg import AbstractLinearOperator
from ..ml._numerics import project_simplex
from ..nn.parameters._transforms import (
    AbstractParameterTransform,
    IntervalTransform,
    PositiveDefiniteTransform,
    PositiveSemidefiniteTransform,
    PositiveTransform,
    SimplexTransform,
)
from ..nonlinear._vi import ProjectionDerivativePolicy
from ..optim._programming._cones import AbstractConvexCone, SecondOrderCone
from ..optim._programming._psd_cone import PositiveSemidefiniteCone
from ..optim._stochastic import (
    certify_chance_constraint,
    ChanceCertificatePolicy,
    ChanceConstraint,
    ChanceConstraintCertificate,
    SampleBatch,
)


FeasibilityScope: TypeAlias = Literal["local", "global"]
FeasibleSetTopology: TypeAlias = Literal["open", "closed"]
FeasibilityKind: TypeAlias = Literal[
    "projection", "parameterization", "kkt", "complementarity", "chance"
]


class FeasibilityEvidence(StrictModule):
    """Truthful membership, optimality, and derivative evidence for one result."""

    violation: Array
    lower_violation: Array
    upper_violation: Array
    cone_violation: Array
    projection_distance: Array
    stationarity_residual: Array
    dual_feasibility_residual: Array
    complementarity_residual: Array
    minimum_slack: Array
    branch_margin: Array
    active_mask: Array
    derivative_status: Array
    finite: Array
    feasible: Array
    optimality_certified: Array
    derivative_certified: Array
    certified: Array
    native_evidence: Any
    scope: FeasibilityScope = eqx.field(static=True)
    topology: FeasibleSetTopology = eqx.field(static=True)
    kind: FeasibilityKind = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


class AbstractFeasibilityMap(StrictModule):
    """Array-level projection or parameterization with explicit set semantics."""

    scope: AbstractAttribute[FeasibilityScope]
    topology: AbstractAttribute[FeasibleSetTopology]
    kind: AbstractAttribute[FeasibilityKind]

    @abc.abstractmethod
    def apply(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        raise NotImplementedError


def _real_array(value: Any, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real floating-point array.")
    return array


def _maximum(value: Array, /) -> Array:
    return jnp.max(value, initial=jnp.asarray(0.0, dtype=value.dtype))


def _certificate(
    *,
    violation: Array,
    lower_violation: Array,
    upper_violation: Array,
    cone_violation: Array,
    projection_distance: Array,
    minimum_slack: Array,
    branch_margin: Array,
    active_mask: Array,
    derivative_status: Array,
    finite: Array,
    feasible: Array,
    derivative_certified: Array,
    scope: FeasibilityScope,
    topology: FeasibleSetTopology,
    kind: FeasibilityKind,
    derivative_id: str,
    certificate_id: str,
    native_evidence: Any = None,
    stationarity_residual: Array | None = None,
    dual_feasibility_residual: Array | None = None,
    complementarity_residual: Array | None = None,
    optimality_certified: Array | None = None,
) -> FeasibilityEvidence:
    dtype = violation.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    optimality = (
        jnp.asarray(True) if optimality_certified is None else optimality_certified
    )
    return FeasibilityEvidence(
        violation=violation,
        lower_violation=lower_violation,
        upper_violation=upper_violation,
        cone_violation=cone_violation,
        projection_distance=projection_distance,
        stationarity_residual=(
            zero if stationarity_residual is None else stationarity_residual
        ),
        dual_feasibility_residual=(
            zero if dual_feasibility_residual is None else dual_feasibility_residual
        ),
        complementarity_residual=(
            zero if complementarity_residual is None else complementarity_residual
        ),
        minimum_slack=minimum_slack,
        branch_margin=branch_margin,
        active_mask=active_mask,
        derivative_status=derivative_status,
        finite=finite,
        feasible=feasible,
        optimality_certified=optimality,
        derivative_certified=derivative_certified,
        certified=finite & feasible & optimality & derivative_certified,
        native_evidence=native_evidence,
        scope=scope,
        topology=topology,
        kind=kind,
        derivative_id=derivative_id,
        certificate_id=certificate_id,
    )


class BoxProjection(AbstractFeasibilityMap):
    """Exact represented Euclidean projection onto a closed real box."""

    lower: Any
    upper: Any
    tolerance: float = eqx.field(static=True)
    derivative_policy: ProjectionDerivativePolicy
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="closed")
    kind: FeasibilityKind = eqx.field(static=True, default="projection")

    def __init__(
        self,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        /,
        *,
        tolerance: float = 1e-7,
        derivative_policy: ProjectionDerivativePolicy | None = None,
    ):
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.lower = lower
        self.upper = upper
        self.tolerance = tolerance_
        self.derivative_policy = (
            ProjectionDerivativePolicy()
            if derivative_policy is None
            else derivative_policy
        )
        if not isinstance(self.derivative_policy, ProjectionDerivativePolicy):
            raise TypeError(
                "derivative_policy must be ProjectionDerivativePolicy or None."
            )

    def _bounds(self, value: Array, /) -> tuple[Array, Array]:
        lower = jnp.broadcast_to(jnp.asarray(self.lower, dtype=value.dtype), value.shape)
        upper = jnp.broadcast_to(jnp.asarray(self.upper, dtype=value.dtype), value.shape)
        invalid = jnp.any(jnp.isnan(lower) | jnp.isnan(upper) | (lower > upper))
        lower = eqx.error_if(lower, invalid, "Box bounds must be ordered and non-NaN.")
        return lower, upper

    def apply(self, value: Any, /) -> Array:
        source = _real_array(value, "box projection input")
        lower, upper = self._bounds(source)
        return jnp.minimum(jnp.maximum(source, lower), upper)

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = _real_array(source, "box projection input")
        result_ = _real_array(result, "box projection result")
        if result_.shape != source_.shape:
            raise ValueError("Box projection input and result shapes must match.")
        lower, upper = self._bounds(result_)
        lower_violation = _maximum(jnp.maximum(lower - result_, 0.0))
        upper_violation = _maximum(jnp.maximum(result_ - upper, 0.0))
        violation = jnp.maximum(lower_violation, upper_violation)
        finite_lower = jnp.isfinite(lower)
        finite_upper = jnp.isfinite(upper)
        lower_distance = jnp.where(finite_lower, jnp.abs(source_ - lower), jnp.inf)
        upper_distance = jnp.where(finite_upper, jnp.abs(source_ - upper), jnp.inf)
        branch_margin = jnp.min(
            jnp.minimum(lower_distance, upper_distance), initial=jnp.inf
        )
        derivative_status = self.derivative_policy.derivative_status(branch_margin)
        active = (finite_lower & (result_ <= lower + self.tolerance)) | (
            finite_upper & (result_ >= upper - self.tolerance)
        )
        slack = jnp.minimum(
            jnp.where(finite_lower, result_ - lower, jnp.inf),
            jnp.where(finite_upper, upper - result_, jnp.inf),
        )
        minimum_slack = jnp.min(slack, initial=jnp.inf)
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        expected = jnp.minimum(jnp.maximum(source_, lower), upper)
        stationarity = jnp.linalg.norm(result_ - expected)
        return _certificate(
            violation=violation,
            lower_violation=lower_violation,
            upper_violation=upper_violation,
            cone_violation=jnp.asarray(0.0, dtype=result_.dtype),
            projection_distance=jnp.linalg.norm(result_ - source_),
            stationarity_residual=stationarity,
            optimality_certified=stationarity <= self.tolerance,
            minimum_slack=minimum_slack,
            branch_margin=branch_margin,
            active_mask=active,
            derivative_status=derivative_status,
            finite=finite,
            feasible=violation <= self.tolerance,
            derivative_certified=derivative_status != 2,
            scope="global",
            topology="closed",
            kind="projection",
            derivative_id=self.derivative_policy.mode,
            certificate_id="closed-box-euclidean-projection",
        )

    def project(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        result = self.apply(value)
        return result, self.certify(value, result)


class SimplexProjection(AbstractFeasibilityMap):
    """Exact represented Euclidean projection onto a closed unit simplex."""

    tolerance: float = eqx.field(static=True)
    derivative_policy: ProjectionDerivativePolicy
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="closed")
    kind: FeasibilityKind = eqx.field(static=True, default="projection")

    def __init__(
        self,
        *,
        tolerance: float = 1e-7,
        derivative_policy: ProjectionDerivativePolicy | None = None,
    ):
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.tolerance = tolerance_
        self.derivative_policy = (
            ProjectionDerivativePolicy()
            if derivative_policy is None
            else derivative_policy
        )
        if not isinstance(self.derivative_policy, ProjectionDerivativePolicy):
            raise TypeError(
                "derivative_policy must be ProjectionDerivativePolicy or None."
            )

    def apply(self, value: Any, /) -> Array:
        source = _real_array(value, "simplex projection input")
        if source.ndim < 1 or int(source.shape[-1]) < 1:
            raise ValueError("Simplex projection requires a nonempty trailing axis.")
        return project_simplex(source)

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = _real_array(source, "simplex projection input")
        result_ = _real_array(result, "simplex projection result")
        if source_.shape != result_.shape:
            raise ValueError("Simplex projection input and result shapes must match.")
        nonnegative_violation = _maximum(jnp.maximum(-result_, 0.0))
        mass_violation = jnp.max(jnp.abs(jnp.sum(result_, axis=-1) - 1.0), initial=0.0)
        violation = jnp.maximum(nonnegative_violation, mass_violation)
        positive = result_ > 0.0
        active_count = jnp.maximum(jnp.sum(positive, axis=-1), 1)
        threshold = (
            jnp.sum(jnp.where(positive, source_ - result_, 0.0), axis=-1) / active_count
        )
        branch_margin = jnp.min(jnp.abs(source_ - threshold[..., None]), initial=jnp.inf)
        derivative_status = self.derivative_policy.derivative_status(branch_margin)
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        expected = project_simplex(source_)
        stationarity = jnp.linalg.norm(result_ - expected)
        return _certificate(
            violation=violation,
            lower_violation=nonnegative_violation,
            upper_violation=mass_violation,
            cone_violation=violation,
            projection_distance=jnp.linalg.norm(result_ - source_),
            stationarity_residual=stationarity,
            optimality_certified=stationarity <= self.tolerance,
            minimum_slack=jnp.min(result_, initial=jnp.inf),
            branch_margin=branch_margin,
            active_mask=result_ <= self.tolerance,
            derivative_status=derivative_status,
            finite=finite,
            feasible=violation <= self.tolerance,
            derivative_certified=derivative_status != 2,
            scope="global",
            topology="closed",
            kind="projection",
            derivative_id=self.derivative_policy.mode,
            certificate_id="closed-simplex-euclidean-projection",
        )

    def project(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        result = self.apply(value)
        return result, self.certify(value, result)


class ConeProjection(AbstractFeasibilityMap):
    """Exact represented Euclidean projection for one native convex cone."""

    cone: AbstractConvexCone
    tolerance: float = eqx.field(static=True)
    derivative_policy: ProjectionDerivativePolicy
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="closed")
    kind: FeasibilityKind = eqx.field(static=True, default="projection")

    def __init__(
        self,
        cone: AbstractConvexCone,
        /,
        *,
        tolerance: float = 1e-7,
        derivative_policy: ProjectionDerivativePolicy | None = None,
    ):
        tolerance_ = float(tolerance)
        if not isinstance(cone, AbstractConvexCone):
            raise TypeError("cone must be an AbstractConvexCone.")
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.cone = cone
        self.tolerance = tolerance_
        self.derivative_policy = (
            ProjectionDerivativePolicy()
            if derivative_policy is None
            else derivative_policy
        )
        if not isinstance(self.derivative_policy, ProjectionDerivativePolicy):
            raise TypeError(
                "derivative_policy must be ProjectionDerivativePolicy or None."
            )

    def apply(self, value: Any, /) -> Array:
        return self.cone.project(value)

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = _real_array(source, "cone projection input")
        result_ = _real_array(result, "cone projection result")
        if source_.shape != result_.shape:
            raise ValueError("Cone projection input and result shapes must match.")
        violation = self.cone.residual(result_)
        branch_margin = self.cone.projection_smoothness_margin(source_)
        derivative_status = self.derivative_policy.derivative_status(branch_margin)
        interior_margin = self.cone.interior_margin(result_)
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        expected = self.cone.project(source_)
        stationarity = jnp.linalg.norm(result_ - expected)
        return _certificate(
            violation=jnp.max(violation),
            lower_violation=jnp.asarray(0.0, dtype=result_.dtype),
            upper_violation=jnp.asarray(0.0, dtype=result_.dtype),
            cone_violation=jnp.max(violation),
            projection_distance=jnp.linalg.norm(result_ - source_),
            stationarity_residual=stationarity,
            optimality_certified=stationarity <= self.tolerance,
            minimum_slack=jnp.min(interior_margin),
            branch_margin=jnp.min(branch_margin),
            active_mask=interior_margin <= self.tolerance,
            derivative_status=jnp.max(derivative_status),
            finite=finite,
            feasible=jnp.all(violation <= self.tolerance),
            derivative_certified=jnp.all(derivative_status != 2),
            scope="global",
            topology="closed",
            kind="projection",
            derivative_id=self.derivative_policy.mode,
            certificate_id=f"closed-cone-projection/{self.cone.cone_id}",
        )

    def project(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        result = self.apply(value)
        return result, self.certify(value, result)


class SecondOrderConeProjection(AbstractFeasibilityMap):
    """Exact represented Euclidean projection onto a Lorentz cone."""

    projection: ConeProjection
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="closed")
    kind: FeasibilityKind = eqx.field(static=True, default="projection")

    def __init__(
        self,
        dimension: int,
        /,
        *,
        tolerance: float = 1e-7,
        derivative_policy: ProjectionDerivativePolicy | None = None,
    ):
        self.projection = ConeProjection(
            SecondOrderCone(dimension),
            tolerance=tolerance,
            derivative_policy=derivative_policy,
        )

    def apply(self, value: Any, /) -> Array:
        return self.projection.apply(value)

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        return self.projection.certify(source, result)

    def project(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        return self.projection.project(value)


class PositiveSemidefiniteProjection(AbstractFeasibilityMap):
    """Exact represented Frobenius projection of real symmetric matrices to PSD."""

    cone: PositiveSemidefiniteCone
    coordinate_projection: ConeProjection
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="closed")
    kind: FeasibilityKind = eqx.field(static=True, default="projection")

    def __init__(
        self,
        matrix_size: int,
        /,
        *,
        tolerance: float = 1e-7,
        derivative_policy: ProjectionDerivativePolicy | None = None,
    ):
        cone = PositiveSemidefiniteCone(matrix_size)
        self.cone = cone
        self.coordinate_projection = ConeProjection(
            cone,
            tolerance=tolerance,
            derivative_policy=derivative_policy,
        )

    def _matrix(self, value: Any, name: str, /) -> Array:
        matrix = _real_array(value, name)
        expected = (self.cone.matrix_size, self.cone.matrix_size)
        if matrix.ndim < 2 or matrix.shape[-2:] != expected:
            raise ValueError(f"{name} must end in shape {expected}.")
        return matrix

    def apply(self, value: Any, /) -> Array:
        matrix = self._matrix(value, "PSD projection input")
        symmetric = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
        return self.cone.unpack(self.cone.project(self.cone.pack(symmetric)))

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = self._matrix(source, "PSD projection input")
        result_ = self._matrix(result, "PSD projection result")
        if source_.shape != result_.shape:
            raise ValueError("PSD projection input and result shapes must match.")
        symmetric_source = 0.5 * (source_ + jnp.swapaxes(source_, -1, -2))
        symmetric_result = 0.5 * (result_ + jnp.swapaxes(result_, -1, -2))
        coordinate = self.coordinate_projection.certify(
            self.cone.pack(symmetric_source), self.cone.pack(symmetric_result)
        )
        asymmetry = jnp.linalg.norm(result_ - symmetric_result)
        violation = jnp.maximum(coordinate.violation, asymmetry)
        expected = self.apply(source_)
        stationarity = jnp.linalg.norm(result_ - expected)
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        tolerance = self.coordinate_projection.tolerance
        return _certificate(
            violation=violation,
            lower_violation=jnp.asarray(0.0, dtype=result_.dtype),
            upper_violation=asymmetry,
            cone_violation=jnp.maximum(coordinate.cone_violation, asymmetry),
            projection_distance=jnp.linalg.norm(result_ - source_),
            stationarity_residual=stationarity,
            minimum_slack=coordinate.minimum_slack,
            branch_margin=coordinate.branch_margin,
            active_mask=coordinate.active_mask,
            derivative_status=coordinate.derivative_status,
            finite=finite,
            feasible=violation <= tolerance,
            optimality_certified=stationarity <= tolerance,
            derivative_certified=coordinate.derivative_certified,
            scope="global",
            topology="closed",
            kind="projection",
            derivative_id=coordinate.derivative_id,
            certificate_id="closed-positive-semidefinite-frobenius-projection",
        )

    def project(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        result = self.apply(value)
        return result, self.certify(value, result)


ParameterizationKind: TypeAlias = Literal[
    "positive", "interval", "simplex", "positive-semidefinite", "positive-definite"
]


class FeasibleParameterization(AbstractFeasibilityMap):
    """Map free coordinates into their transform's declared feasible set."""

    transform: AbstractParameterTransform
    parameterization: ParameterizationKind = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    topology: FeasibleSetTopology = eqx.field(static=True)
    kind: FeasibilityKind = eqx.field(static=True, default="parameterization")

    def __init__(
        self,
        transform: AbstractParameterTransform,
        parameterization: ParameterizationKind,
        /,
        *,
        tolerance: float = 1e-7,
    ):
        tolerance_ = float(tolerance)
        if not isinstance(transform, AbstractParameterTransform):
            raise TypeError("transform must be an AbstractParameterTransform.")
        expected = {
            "positive": PositiveTransform,
            "interval": IntervalTransform,
            "simplex": SimplexTransform,
            "positive-semidefinite": PositiveSemidefiniteTransform,
            "positive-definite": PositiveDefiniteTransform,
        }
        if parameterization not in expected:
            raise ValueError("Unknown feasible parameterization kind.")
        if not isinstance(transform, expected[parameterization]):
            raise TypeError(f"{parameterization} has an incompatible transform.")
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.transform = transform
        self.parameterization = parameterization
        self.topology = (
            "closed" if parameterization == "positive-semidefinite" else "open"
        )
        self.tolerance = tolerance_

    def apply(self, value: Any, /) -> Array:
        return self.transform(value)

    def _margin(self, result: Array, /) -> tuple[Array, Array]:
        if self.parameterization == "positive":
            minimum = jnp.asarray(self.transform.minimum, dtype=result.dtype)
            return jnp.min(result - minimum), jnp.max(jnp.maximum(minimum - result, 0.0))
        if self.parameterization == "interval":
            lower = jnp.asarray(self.transform.lower, dtype=result.dtype)
            upper = jnp.asarray(self.transform.upper, dtype=result.dtype)
            margin = jnp.min(jnp.minimum(result - lower, upper - result))
            violation = jnp.max(
                jnp.maximum(jnp.maximum(lower - result, result - upper), 0.0)
            )
            return margin, violation
        if self.parameterization == "simplex":
            margin = jnp.min(result)
            violation = jnp.maximum(
                jnp.max(jnp.maximum(-result, 0.0)),
                jnp.max(jnp.abs(jnp.sum(result, axis=-1) - 1.0)),
            )
            return margin, violation
        if result.ndim < 2 or result.shape[-2] != result.shape[-1]:
            raise ValueError("Matrix parameterizations require square trailing axes.")
        matrix_size = int(result.shape[-1])
        cone = PositiveSemidefiniteCone(matrix_size)
        symmetric = 0.5 * (result + jnp.swapaxes(result, -1, -2))
        asymmetry = jnp.linalg.norm(result - symmetric)
        margin = jnp.min(cone.interior_margin(cone.pack(symmetric)))
        violation = jnp.maximum(jnp.max(cone.residual(cone.pack(symmetric))), asymmetry)
        return margin, violation

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = _real_array(source, "parameterization coordinates")
        result_ = _real_array(result, "parameterized value")
        margin, violation = self._margin(result_)
        expected = _real_array(self.apply(source_), "parameterized expected value")
        if expected.shape != result_.shape:
            raise ValueError("Parameterized result has the wrong shape.")
        map_residual = jnp.linalg.norm(result_ - expected)
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        closed = self.parameterization == "positive-semidefinite"
        feasible = ((margin >= -self.tolerance) if closed else (margin > 0.0)) & (
            violation <= self.tolerance
        )
        return _certificate(
            violation=violation,
            lower_violation=violation,
            upper_violation=jnp.asarray(0.0, dtype=result_.dtype),
            cone_violation=violation,
            stationarity_residual=map_residual,
            optimality_certified=map_residual <= self.tolerance,
            projection_distance=jnp.asarray(jnp.nan, dtype=result_.dtype),
            minimum_slack=margin,
            branch_margin=margin,
            active_mask=margin <= self.tolerance,
            derivative_status=jnp.asarray(0, dtype=jnp.int32),
            finite=finite,
            feasible=feasible,
            derivative_certified=finite,
            scope="global",
            topology=self.topology,
            kind="parameterization",
            derivative_id="smooth-transform",
            certificate_id=f"{self.topology}-{self.parameterization}-parameterization",
        )

    def parameterize(self, value: Any, /) -> tuple[Array, FeasibilityEvidence]:
        result = self.apply(value)
        return result, self.certify(value, result)


class StrictSecondOrderConeParameterization(AbstractFeasibilityMap):
    """Smooth global map into the strict interior of a Lorentz cone."""

    cone: SecondOrderCone
    minimum_margin: float = eqx.field(static=True)
    scope: FeasibilityScope = eqx.field(static=True, default="global")
    topology: FeasibleSetTopology = eqx.field(static=True, default="open")
    kind: FeasibilityKind = eqx.field(static=True, default="parameterization")

    def __init__(self, dimension: int, /, *, minimum_margin: float = 1e-6):
        margin = float(minimum_margin)
        if not isfinite(margin) or margin <= 0.0:
            raise ValueError("minimum_margin must be finite and positive.")
        self.cone = SecondOrderCone(dimension)
        self.minimum_margin = margin

    def apply(self, value: Any, /) -> Array:
        source = _real_array(value, "SOC parameterization coordinates")
        if source.shape[-1] != self.cone.dimension:
            raise ValueError("SOC parameter coordinates have the wrong trailing size.")
        vector = source[..., 1:]
        margin = PositiveTransform(self.minimum_margin)(source[..., :1])
        scalar = jnp.sqrt(
            jnp.sum(jnp.square(vector), axis=-1, keepdims=True) + jnp.square(margin)
        )
        return jnp.concatenate((scalar, vector), axis=-1)

    def certify(self, source: Any, result: Any, /) -> FeasibilityEvidence:
        source_ = _real_array(source, "SOC parameterization coordinates")
        result_ = _real_array(result, "SOC parameterization result")
        margin = self.cone.interior_margin(result_)
        minimum_margin = jnp.min(margin)
        violation = jnp.max(self.cone.residual(result_))
        finite = jnp.all(jnp.isfinite(source_)) & jnp.all(jnp.isfinite(result_))
        expected = self.apply(source_)
        map_residual = jnp.linalg.norm(result_ - expected)
        return _certificate(
            violation=violation,
            lower_violation=jnp.asarray(0.0, dtype=result_.dtype),
            upper_violation=jnp.asarray(0.0, dtype=result_.dtype),
            cone_violation=violation,
            stationarity_residual=map_residual,
            optimality_certified=map_residual == 0.0,
            projection_distance=jnp.asarray(jnp.nan, dtype=result_.dtype),
            minimum_slack=minimum_margin,
            branch_margin=minimum_margin,
            active_mask=margin <= 0.0,
            derivative_status=jnp.asarray(0, dtype=jnp.int32),
            finite=finite,
            feasible=minimum_margin > 0.0,
            derivative_certified=finite & (minimum_margin > 0.0),
            scope="global",
            topology="open",
            kind="parameterization",
            derivative_id="smooth-interior-map",
            certificate_id="open-second-order-cone-parameterization",
        )


class AbstractCoefficientPropertyProvider(StrictModule):
    """Typed coefficient-property lowering through a linear representation."""

    representation: Any
    operator: AbstractLinearOperator
    provider_id: AbstractAttribute[str]

    def __init__(self, representation: Any, operator: AbstractLinearOperator, /):
        from ._linear_representation import AbstractLinearRepresentation

        if not isinstance(representation, AbstractLinearRepresentation):
            raise TypeError("representation must implement AbstractLinearRepresentation.")
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not operator.source.compatible(representation.coefficient_space):
            raise ValueError("Property operator source must match coefficient space.")
        self.representation = representation
        self.operator = operator

    def value(self, fields: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(self.representation.extract(fields))


class MonotonicityProvider(AbstractCoefficientPropertyProvider):
    direction: Literal["increasing", "decreasing"] = eqx.field(static=True)

    def __init__(
        self,
        representation: Any,
        derivative_operator: AbstractLinearOperator,
        /,
        *,
        direction: Literal["increasing", "decreasing"] = "increasing",
    ):
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'.")
        super().__init__(representation, derivative_operator)
        self.direction = direction

    @property
    def provider_id(self) -> str:
        return f"coefficient-monotonicity/{self.direction}/{self.operator.operator_id}"

    def margin(self, fields: PyTree[Any], /) -> PyTree[Array]:
        sign = 1.0 if self.direction == "increasing" else -1.0
        return jax.tree.map(lambda leaf: sign * leaf, self.value(fields))


class ConvexityProvider(AbstractCoefficientPropertyProvider):
    curvature: Literal["convex", "concave"] = eqx.field(static=True)

    def __init__(
        self,
        representation: Any,
        second_derivative_operator: AbstractLinearOperator,
        /,
        *,
        curvature: Literal["convex", "concave"] = "convex",
    ):
        if curvature not in ("convex", "concave"):
            raise ValueError("curvature must be 'convex' or 'concave'.")
        super().__init__(representation, second_derivative_operator)
        self.curvature = curvature

    @property
    def provider_id(self) -> str:
        return f"coefficient-convexity/{self.curvature}/{self.operator.operator_id}"

    def margin(self, fields: PyTree[Any], /) -> PyTree[Array]:
        sign = 1.0 if self.curvature == "convex" else -1.0
        return jax.tree.map(lambda leaf: sign * leaf, self.value(fields))


class PositivityProvider(AbstractCoefficientPropertyProvider):
    strict: bool = eqx.field(static=True)

    def __init__(
        self,
        representation: Any,
        evaluation_operator: AbstractLinearOperator,
        /,
        *,
        strict: bool = False,
    ):
        super().__init__(representation, evaluation_operator)
        self.strict = bool(strict)

    @property
    def provider_id(self) -> str:
        topology = "strict" if self.strict else "closed"
        return f"coefficient-positivity/{topology}/{self.operator.operator_id}"

    def margin(self, fields: PyTree[Any], /) -> PyTree[Array]:
        return self.value(fields)


class NormalizationProvider(AbstractCoefficientPropertyProvider):
    target: Any

    def __init__(
        self,
        representation: Any,
        mass_operator: AbstractLinearOperator,
        /,
        *,
        target: Any = 1.0,
    ):
        super().__init__(representation, mass_operator)
        target_ = mass_operator.target.validate(target)
        if not tree_allfinite(target_):
            raise ValueError("Normalization target must be finite.")
        self.target = target_

    @property
    def provider_id(self) -> str:
        return f"coefficient-normalization/{self.operator.operator_id}"

    def residual(self, fields: PyTree[Any], /) -> PyTree[Array]:
        return jax.tree.map(
            lambda left, right: left - right, self.value(fields), self.target
        )


class ChanceFeasibility(StrictModule):
    """Unsmoothed certificate hook for a caller-supplied frozen scenario batch."""

    constraint: ChanceConstraint
    policy: ChanceCertificatePolicy

    def __init__(self, constraint: ChanceConstraint, policy: ChanceCertificatePolicy, /):
        if not isinstance(constraint, ChanceConstraint):
            raise TypeError("constraint must be a ChanceConstraint.")
        if not isinstance(policy, ChanceCertificatePolicy):
            raise TypeError("policy must be a ChanceCertificatePolicy.")
        self.constraint = constraint
        self.policy = policy

    def certify(
        self, parameters: PyTree[Any], batch: SampleBatch, /, *, args: Any = None
    ) -> ChanceConstraintCertificate:
        return certify_chance_constraint(
            self.constraint, parameters, batch, policy=self.policy, args=args
        )


__all__ = [
    "AbstractCoefficientPropertyProvider",
    "AbstractFeasibilityMap",
    "BoxProjection",
    "ChanceFeasibility",
    "ConeProjection",
    "ConvexityProvider",
    "FeasibilityEvidence",
    "FeasibleParameterization",
    "MonotonicityProvider",
    "NormalizationProvider",
    "PositiveSemidefiniteProjection",
    "PositivityProvider",
    "SecondOrderConeProjection",
    "SimplexProjection",
    "StrictSecondOrderConeParameterization",
]
