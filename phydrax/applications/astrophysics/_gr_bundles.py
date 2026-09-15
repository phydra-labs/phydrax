#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._identity import callable_payload
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._metric import LorentzianMetric


def gr_chart_identity(metric: LorentzianMetric, /) -> str:
    """Return the canonical coordinate-chart identity used by GR products."""
    if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
        raise TypeError("GR chart identity requires a four-dimensional LorentzianMetric.")
    return canonical_fingerprint(
        {
            "kind": "gr-coordinate-chart",
            "name": metric.chart.name,
            "coordinates": metric.chart.coordinates,
        }
    )


def _identified_callable_payload(
    value: Callable,
    /,
    *,
    owner: str,
    semantic_id: str | None,
    numeric_id: str | None,
) -> dict[str, object]:
    if (semantic_id is None) != (numeric_id is None):
        raise ValueError(f"{owner} semantic_id and numeric_id must be supplied together.")
    if semantic_id is None:
        return callable_payload(value)
    if (
        not isinstance(semantic_id, str)
        or not semantic_id
        or not isinstance(numeric_id, str)
        or not numeric_id
    ):
        raise ValueError(f"{owner} semantic and numeric IDs must be non-empty strings.")
    return {
        "semantic_content_id": semantic_id,
        "numeric_content_id": numeric_id,
    }


def gr_metric_identity(
    metric: LorentzianMetric,
    /,
    *,
    semantic_id: str | None = None,
    numeric_id: str | None = None,
) -> str:
    """Content-address a metric map; opaque callables require explicit identities."""
    if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
        raise TypeError(
            "GR metric identity requires a four-dimensional LorentzianMetric."
        )
    matrix_payload = _identified_callable_payload(
        metric.matrix_function,
        owner="Metric",
        semantic_id=semantic_id,
        numeric_id=numeric_id,
    )
    return canonical_fingerprint(
        {
            "kind": "gr-lorentzian-metric",
            "chart_id": gr_chart_identity(metric),
            "convention": metric.convention,
            "matrix_function": matrix_payload,
        }
    )


class AbstractGRConstantOfMotion(StrictModule):
    """Scalar phase-space quantity whose drift is retained as ray evidence."""

    name: str = eqx.field(static=True)
    constant_id: str = eqx.field(static=True)

    @abstractmethod
    def __call__(
        self,
        metric: LorentzianMetric,
        coordinates: Array,
        tangent: Array,
        /,
    ) -> Array:
        raise NotImplementedError


class GRCallableConstantOfMotion(AbstractGRConstantOfMotion):
    """Named differentiable phase-space scalar supplied by an application."""

    evaluator: Callable[[LorentzianMetric, Array, Array], Array]
    name: str = eqx.field(static=True)
    constant_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[[LorentzianMetric, Array, Array], Array],
        /,
        *,
        name: str,
        evaluator_semantic_id: str | None = None,
        evaluator_numeric_id: str | None = None,
    ):
        if not callable(evaluator):
            raise TypeError("Constant-of-motion evaluator must be callable.")
        if not isinstance(name, str) or not name:
            raise ValueError("Constant-of-motion name must be non-empty.")
        self.evaluator = evaluator
        self.name = name
        self.constant_id = canonical_fingerprint(
            {
                "kind": "gr-callable-constant-of-motion",
                "name": name,
                "evaluator": _identified_callable_payload(
                    evaluator,
                    owner="Constant evaluator",
                    semantic_id=evaluator_semantic_id,
                    numeric_id=evaluator_numeric_id,
                ),
            }
        )

    def __call__(
        self,
        metric: LorentzianMetric,
        coordinates: Array,
        tangent: Array,
        /,
    ) -> Array:
        value = jnp.asarray(self.evaluator(metric, coordinates, tangent))
        if value.shape != ():
            raise ValueError("Constant-of-motion evaluator must return a scalar.")
        return value


class GRCoordinateMomentumConstant(AbstractGRConstantOfMotion):
    """Covariant momentum along a declared coordinate Killing direction."""

    coordinate_index: int = eqx.field(static=True)
    sign: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    constant_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_index: int,
        /,
        *,
        name: str,
        sign: float = 1.0,
    ):
        index = int(coordinate_index)
        if index < 0 or index >= 4:
            raise ValueError("coordinate_index must lie in [0, 4).")
        if not isinstance(name, str) or not name:
            raise ValueError("Constant-of-motion name must be non-empty.")
        self.coordinate_index = index
        self.sign = float(sign)
        self.name = name
        self.constant_id = canonical_fingerprint(
            {
                "kind": "gr-coordinate-momentum-constant",
                "name": name,
                "coordinate_index": index,
                "sign": self.sign,
            }
        )

    def __call__(
        self,
        metric: LorentzianMetric,
        coordinates: Array,
        tangent: Array,
        /,
    ) -> Array:
        covector = ein.contract("ij,j->i", metric(coordinates), tangent)
        return self.sign * covector[self.coordinate_index]


class GRJacobiEvidence(StrictModule, NonTrainableState):
    """Screen-projected Jacobi map and fixed-history caustic evidence."""

    jacobi_map: Array
    determinant: Array
    caustic: Array
    valid: Array

    def __init__(
        self,
        jacobi_map: ArrayLike,
        determinant: ArrayLike,
        caustic: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        mapping = jnp.asarray(jacobi_map)
        determinant_ = jnp.asarray(determinant, dtype=mapping.dtype)
        caustic_ = jnp.asarray(caustic, dtype=bool)
        valid_ = jnp.asarray(valid, dtype=bool)
        if mapping.shape[-2:] != (2, 2):
            raise ValueError("Jacobi maps must end in shape (2, 2).")
        if any(
            value.shape != mapping.shape[:-2]
            for value in (determinant_, caustic_, valid_)
        ):
            raise ValueError("Jacobi evidence must share the ray/history shape.")
        self.jacobi_map = mapping
        self.determinant = determinant_
        self.caustic = caustic_
        self.valid = valid_


class GRRayBundleEvidence(StrictModule, NonTrainableState):
    """Constants-of-motion, parallel transport, and geodesic-deviation evidence."""

    constant_values: Array
    constant_relative_drift: Array
    constant_valid: Array
    transport_residual: Array
    transport_valid: Array
    jacobi: GRJacobiEvidence | None
    valid: Array
    constant_names: tuple[str, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        constant_values: ArrayLike,
        constant_relative_drift: ArrayLike,
        constant_valid: ArrayLike,
        transport_residual: ArrayLike,
        transport_valid: ArrayLike,
        jacobi: GRJacobiEvidence | None,
        valid: ArrayLike,
        /,
        *,
        constant_names: Sequence[str],
        bundle_id: str,
    ):
        values = jnp.asarray(constant_values)
        drift = jnp.asarray(constant_relative_drift, dtype=values.dtype)
        constants_valid = jnp.asarray(constant_valid, dtype=bool)
        transport_error = jnp.asarray(transport_residual, dtype=values.dtype)
        transport_valid_ = jnp.asarray(transport_valid, dtype=bool)
        valid_ = jnp.asarray(valid, dtype=bool)
        names = tuple(str(name) for name in constant_names)
        if values.ndim != 3 or values.shape[-1] != len(names):
            raise ValueError(
                "constant_values must have shape (num_rays, num_history, num_constants)."
            )
        if drift.shape != (values.shape[0], values.shape[2]):
            raise ValueError(
                "constant_relative_drift must have shape (num_rays, num_constants)."
            )
        if constants_valid.shape != drift.shape:
            raise ValueError("constant_valid must match constant_relative_drift.")
        if (
            transport_error.shape != values.shape[:2]
            or transport_valid_.shape != values.shape[:2]
        ):
            raise ValueError("Transport evidence must match the ray/history axes.")
        if valid_.shape != (values.shape[0],):
            raise ValueError("Bundle validity must have shape (num_rays,).")
        if jacobi is not None:
            if not isinstance(jacobi, GRJacobiEvidence):
                raise TypeError("jacobi must be GRJacobiEvidence or None.")
            if jacobi.valid.shape != values.shape[:2]:
                raise ValueError("Jacobi evidence must match the ray/history axes.")
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("Constant-of-motion names must be unique and non-empty.")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise ValueError("bundle_id must be non-empty.")
        self.constant_values = values
        self.constant_relative_drift = drift
        self.constant_valid = constants_valid
        self.transport_residual = transport_error
        self.transport_valid = transport_valid_
        self.jacobi = jacobi
        self.valid = valid_
        self.constant_names = names
        self.bundle_id = bundle_id


def _metric_at_active(
    metric: LorentzianMetric,
    coordinates: Array,
    active: Array,
    /,
) -> Array:
    identity = jnp.eye(4, dtype=coordinates.dtype)
    return jax.lax.cond(
        active,
        lambda point: metric(point),
        lambda _: identity,
        coordinates,
    )


def _constant_history(
    quantity: AbstractGRConstantOfMotion,
    metric: LorentzianMetric,
    points: Array,
    velocities: Array,
    mask: Array,
    /,
) -> Array:
    def one(point: Array, velocity: Array, enabled: Array) -> Array:
        return jax.lax.cond(
            enabled,
            lambda operands: quantity(metric, operands[0], operands[1]),
            lambda operands: jnp.zeros((), dtype=operands[0].dtype),
            (point, velocity),
        )

    return jax.vmap(jax.vmap(one))(points, velocities, mask)


def build_gr_ray_bundle_evidence(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    tangents: ArrayLike,
    active: ArrayLike,
    transported_screen_basis: ArrayLike | None,
    jacobi_variations: ArrayLike | None,
    constants: Sequence[AbstractGRConstantOfMotion] = (),
    /,
    *,
    metric_id: str,
    tolerance: float,
    bundle_id: str | None = None,
) -> GRRayBundleEvidence:
    """Evaluate fixed-history ray-bundle conservation and caustic diagnostics."""

    if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
        raise TypeError(
            "Ray bundle evidence requires a four-dimensional LorentzianMetric."
        )
    points = jnp.asarray(coordinates)
    if not isinstance(metric_id, str) or not metric_id:
        raise ValueError("metric_id must be a non-empty string.")
    velocities = jnp.asarray(tangents, dtype=points.dtype)
    mask = jnp.asarray(active, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 4 or velocities.shape != points.shape:
        raise ValueError("Ray histories must have shape (num_rays, num_history, 4).")
    if mask.shape != points.shape[:2]:
        raise ValueError("active must match ray/history axes.")
    tolerance_ = float(tolerance)
    if not tolerance_ > 0.0:
        raise ValueError("tolerance must be positive.")
    quantities = tuple(constants)
    if any(not isinstance(value, AbstractGRConstantOfMotion) for value in quantities):
        raise TypeError("constants must contain AbstractGRConstantOfMotion values.")
    names = tuple(value.name for value in quantities)
    if len(set(names)) != len(names):
        raise ValueError("Constants of motion must have unique names.")

    metrics = jax.vmap(
        jax.vmap(lambda point, enabled: _metric_at_active(metric, point, enabled))
    )(points, mask)
    if quantities:
        columns = tuple(
            _constant_history(quantity, metric, points, velocities, mask)
            for quantity in quantities
        )
        constant_values = jnp.stack(columns, axis=-1)
        first_index = jnp.argmax(mask.astype(jnp.int32), axis=1)
        reference = jnp.take_along_axis(
            constant_values,
            first_index[:, None, None],
            axis=1,
        )[:, 0, :]
        difference = jnp.where(
            mask[..., None],
            jnp.abs(constant_values - reference[:, None, :]),
            0.0,
        )
        scale = jnp.maximum(1.0, jnp.abs(reference))
        drift = jnp.max(difference, axis=1) / scale
        constant_valid = jnp.isfinite(drift) & (drift <= tolerance_)
    else:
        constant_values = jnp.empty(points.shape[:2] + (0,), dtype=points.dtype)
        drift = jnp.empty((points.shape[0], 0), dtype=points.dtype)
        constant_valid = jnp.empty((points.shape[0], 0), dtype=bool)

    spatial_sign = float(1 if metric.convention == "mostly_plus" else -1)
    if transported_screen_basis is None:
        transport_residual = jnp.zeros(points.shape[:2], dtype=points.dtype)
        transport_valid = mask
        basis = None
    else:
        basis = jnp.asarray(transported_screen_basis, dtype=points.dtype)
        if basis.shape != points.shape[:2] + (2, 4):
            raise ValueError(
                "transported_screen_basis must have shape (rays, history, 2, 4)."
            )
        gram = ein.contract("rhai,rhij,rhbj->rhab", basis, metrics, basis)
        tangent_pairing = ein.contract("rhai,rhij,rhj->rha", basis, metrics, velocities)
        target = spatial_sign * jnp.eye(2, dtype=points.dtype)
        transport_residual = jnp.maximum(
            jnp.max(jnp.abs(gram - target), axis=(-2, -1)),
            jnp.max(jnp.abs(tangent_pairing), axis=-1),
        )
        transport_residual = jnp.where(mask, transport_residual, 0.0)
        transport_valid = (
            mask & jnp.isfinite(transport_residual) & (transport_residual <= tolerance_)
        )

    if jacobi_variations is None:
        jacobi = None
        jacobi_ray_valid = jnp.ones((points.shape[0],), dtype=bool)
    else:
        variations = jnp.asarray(jacobi_variations, dtype=points.dtype)
        if variations.shape != points.shape[:2] + (2, 8):
            raise ValueError("jacobi_variations must have shape (rays, history, 2, 8).")
        if basis is None:
            raise ValueError("Jacobi evidence requires transported_screen_basis.")
        deviation = variations[..., :4]
        mapping = ein.contract("rhai,rhij,rhbj->rhab", basis, metrics, deviation)
        determinant = (
            mapping[..., 0, 0] * mapping[..., 1, 1]
            - mapping[..., 0, 1] * mapping[..., 1, 0]
        )
        previous = jnp.concatenate((determinant[:, :1], determinant[:, :-1]), axis=1)
        previous_active = jnp.concatenate(
            (jnp.zeros_like(mask[:, :1]), mask[:, :-1]), axis=1
        )
        scale = jnp.max(jnp.abs(mapping), axis=(-2, -1))
        sign_crossing = (determinant * previous) < 0.0
        resolved_singular = (
            (jnp.arange(points.shape[1])[None, :] > 1)
            & (scale > jnp.sqrt(tolerance_))
            & (jnp.abs(determinant) <= tolerance_ * jnp.maximum(1.0, scale**2))
        )
        caustic = mask & previous_active & (sign_crossing | resolved_singular)
        jacobi_valid = mask & jnp.all(jnp.isfinite(mapping), axis=(-2, -1))
        jacobi = GRJacobiEvidence(mapping, determinant, caustic, jacobi_valid)
        jacobi_ray_valid = jnp.all(jnp.where(mask, jacobi_valid, True), axis=1)

    has_active = jnp.any(mask, axis=1)
    constants_ray_valid = jnp.all(constant_valid, axis=-1)
    transport_ray_valid = jnp.all(jnp.where(mask, transport_valid, True), axis=1)
    valid = has_active & constants_ray_valid & transport_ray_valid & jacobi_ray_valid
    resolved_id = (
        canonical_fingerprint(
            {
                "kind": "gr-ray-bundle-evidence",
                "metric_id": metric_id,
                "constants": tuple(value.constant_id for value in quantities),
                "shape": points.shape,
                "tolerance": tolerance_,
            }
        )
        if bundle_id is None
        else str(bundle_id)
    )
    return GRRayBundleEvidence(
        constant_values,
        drift,
        constant_valid,
        transport_residual,
        transport_valid,
        jacobi,
        valid,
        constant_names=names,
        bundle_id=resolved_id,
    )


__all__ = [
    "gr_chart_identity",
    "gr_metric_identity",
    "AbstractGRConstantOfMotion",
    "GRCallableConstantOfMotion",
    "GRCoordinateMomentumConstant",
    "GRJacobiEvidence",
    "GRRayBundleEvidence",
    "build_gr_ray_bundle_evidence",
]
