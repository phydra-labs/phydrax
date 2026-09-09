#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host preparation of physically labelled, fixed-support observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    DiscreteFieldSpace,
    FieldTransfer,
    TensorDofLayout,
    TransferProperties,
)
from ...linalg import ArraySpace, FunctionLinearOperator
from ...stochastic import ObservationSequence
from ._quantities import GeophysicalQuantity
from ._time import GeophysicalTimeSpec, TemporalSupport


class GeophysicalObservationOperator(StrictModule, NonTrainableState):
    """One native field transfer with exact quantity, time and support semantics.

    The transfer acts on values in ``quantity.unit``; observed values are converted
    to that same unit during preparation. No conservative-remapping claim is
    inferred from a point interpolation. Interval measurements require an
    interval-aware model and are deliberately rejected by this instantaneous API.
    """

    transfer: FieldTransfer
    quantity: GeophysicalQuantity
    time: GeophysicalTimeSpec
    temporal: TemporalSupport
    kind: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        quantity: GeophysicalQuantity,
        /,
        *,
        time: GeophysicalTimeSpec,
        temporal: TemporalSupport | None = None,
        kind: str = "station",
    ):
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a native FieldTransfer.")
        if not isinstance(quantity, GeophysicalQuantity) or not isinstance(
            time, GeophysicalTimeSpec
        ):
            raise TypeError("quantity and time must be geophysical descriptors.")
        temporal_ = TemporalSupport() if temporal is None else temporal
        if not isinstance(temporal_, TemporalSupport):
            raise TypeError("temporal must be TemporalSupport.")
        if (
            temporal_.kind != "instantaneous"
            or temporal_.bounds is not None
            or temporal_.position != "point"
        ):
            raise ValueError(
                "This operator supports instantaneous point-time observations only."
            )
        if kind not in ("station", "grid", "profile"):
            raise ValueError("kind must be station, grid, or profile.")
        if not isinstance(transfer.source.vector_space, ArraySpace) or not isinstance(
            transfer.target.vector_space, ArraySpace
        ):
            raise TypeError("Observation transfers require native ArraySpace fields.")
        self.transfer = transfer
        self.quantity = quantity
        self.time = time
        self.temporal = temporal_
        self.kind = kind
        self.operator_id = canonical_fingerprint(
            {
                "kind": "geophysical-observation",
                "transfer": transfer.transfer_id,
                "operator_content": array_tree_fingerprint(transfer.primal_operator),
                "quantity": quantity.quantity_id,
                "time": time.time_id,
                "temporal": temporal_.support_id,
                "sampling": kind,
            }
        )

    def __call__(self, values: ArrayLike, /) -> Array:
        return self.transfer.primal_operator.mv(values)


def prepare_tensor_observation_operator(
    source: DiscreteFieldSpace,
    coordinates: Mapping[str, ArrayLike],
    points: Mapping[str, ArrayLike],
    quantity: GeophysicalQuantity,
    /,
    *,
    source_support_id: str,
    time: GeophysicalTimeSpec,
    kind: str = "station",
    periods: Mapping[str, float] | None = None,
) -> GeophysicalObservationOperator:
    """Prepare sparse multilinear station, flattened-grid or profile sampling.

    Axis names/order must match the source tensor layout exactly. Each target
    axis is a vector of the same length, in the source coordinate units. A grid
    is supplied as flattened meshgrid coordinates; a profile as point columns.
    A pressure/height coordinate is an ordinary, explicitly named tensor axis:
    it must be fixed, monotone increasing and supplied in matching units. Hybrid
    pressure columns must first be reconstructed on fixed pressure coordinates.
    Periodic axes contain one period without a duplicated endpoint. Other axes
    reject extrapolation. This is not spherical distance interpolation or a
    localized filter; longitude periodicity alone does not imply either claim.
    """
    if not isinstance(source, DiscreteFieldSpace) or not isinstance(
        source.layout, TensorDofLayout
    ):
        raise TypeError("source must have a native TensorDofLayout.")
    if source.support_id != source_support_id:
        raise ValueError("Source support identity does not match the coordinate support.")
    if source.representation != "point_value" or source.layout.component_shape:
        raise ValueError("Multilinear sampling requires scalar point-value fields.")
    if not isinstance(source.vector_space, ArraySpace):
        raise TypeError("source must use an ArraySpace.")
    axes = source.layout.axis_names
    if set(coordinates) != set(axes) or set(points) != set(axes):
        raise ValueError(
            "Coordinate and point axes must exactly match the source layout."
        )
    periodic = {} if periods is None else dict(periods)
    if set(periodic) - set(axes):
        raise ValueError("Periodic axes must belong to the source layout.")
    nodes = tuple(np.asarray(coordinates[name], dtype=float) for name in axes)
    targets = tuple(np.asarray(points[name], dtype=float) for name in axes)
    count = targets[0].size
    if count == 0 or any(
        value.shape != (count,) or not np.all(np.isfinite(value)) for value in targets
    ):
        raise ValueError(
            "Each target coordinate must be a finite, nonempty point vector."
        )
    brackets = []
    fractions = []
    for name, size, node, target in zip(
        axes, source.layout.axis_shape, nodes, targets, strict=True
    ):
        if (
            node.shape != (size,)
            or not np.all(np.isfinite(node))
            or np.any(np.diff(node) <= 0)
        ):
            raise ValueError(
                "Source coordinates must be finite, strictly increasing axis vectors."
            )
        if name in periodic:
            period = float(periodic[name])
            if not np.isfinite(period) or period <= 0 or node[-1] - node[0] >= period:
                raise ValueError(
                    "Periodic coordinates must span less than one positive period."
                )
            extended = np.concatenate((node, [node[0] + period]))
            wrapped = np.mod(target - node[0], period) + node[0]
            lower = np.searchsorted(extended, wrapped, side="right") - 1
            upper = (lower + 1) % size
            fraction = (wrapped - extended[lower]) / (
                extended[lower + 1] - extended[lower]
            )
        elif size == 1:
            if np.any(target != node[0]):
                raise ValueError("Singleton axes only support their exact coordinate.")
            lower = upper = np.zeros(count, dtype=np.int32)
            fraction = np.zeros(count)
        else:
            if np.any(target < node[0]) or np.any(target > node[-1]):
                raise ValueError(
                    "Observation points cannot extrapolate beyond the source support."
                )
            lower = np.clip(np.searchsorted(node, target, side="right") - 1, 0, size - 2)
            upper = lower + 1
            fraction = (target - node[lower]) / (node[upper] - node[lower])
        brackets.append((lower, upper))
        fractions.append(fraction)
    routes = []
    weights = []
    for corner in product((0, 1), repeat=len(axes)):
        index = tuple(
            bracket[side] for bracket, side in zip(brackets, corner, strict=True)
        )
        routes.append(np.ravel_multi_index(index, source.layout.axis_shape))
        weight = np.ones(count)
        for fraction, side in zip(fractions, corner, strict=True):
            weight *= fraction if side else 1.0 - fraction
        weights.append(weight)
    route = jnp.asarray(np.stack(routes, axis=1), dtype=jnp.int32)
    weight = jnp.asarray(np.stack(weights, axis=1), dtype=source.vector_space.dtype)
    identity = canonical_fingerprint(
        {
            "kind": "tensor-geophysical-sampling",
            "source": source.field_space_id,
            "coordinates": array_tree_fingerprint(nodes),
            "points": array_tree_fingerprint(targets),
            "periods": periodic,
            "sampling": kind,
        }
    )
    target_space = ArraySpace(
        (count,), dtype=source.vector_space.dtype, space_id=identity
    )
    target = DiscreteFieldSpace(
        quantity.name,
        identity,
        TensorDofLayout(("observation",), (count,)),
        target_space,
        representation="point_value",
    )

    def action(values):
        return jnp.sum(values.reshape(-1)[route] * weight, axis=-1)

    def transpose(values):
        flat = jnp.zeros((source.vector_space.size,), dtype=source.vector_space.dtype)
        return (
            flat.at[route.reshape(-1)]
            .add((values[:, None] * weight).reshape(-1))
            .reshape(source.vector_space.shape)
        )

    operator = FunctionLinearOperator(
        action,
        source=source.vector_space,
        target=target_space,
        transpose_action=transpose,
        operator_id=identity,
    )
    transfer = FieldTransfer(
        source,
        target,
        operator,
        properties=TransferProperties(
            constant_preserving=True, positivity_preserving=True, exact_on=("constants",)
        ),
    )
    return GeophysicalObservationOperator(transfer, quantity, time=time, kind=kind)


class GeophysicalObservationPolicy(StrictModule, NonTrainableState):
    """Explicit missing/QC/error policy, never silent nonfinite rejection.

    ``qc_accept`` supplied to preparation is always intersected with the declared
    availability mask. Nonfinite *available and QC-accepted* values raise unless
    ``nonfinite='mask'``. Bounds are in the operator's units. Representativeness
    standard deviations are supplied separately, in the observation input unit,
    and add independent variance, not a spatial-localization weight.
    """

    nonfinite: str = eqx.field(static=True)
    bounds: tuple[float, float] | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, nonfinite: str = "raise", bounds: tuple[float, float] | None = None
    ):
        if nonfinite not in ("raise", "mask"):
            raise ValueError("nonfinite must be raise or mask.")
        bounds_ = None if bounds is None else tuple(float(value) for value in bounds)
        if bounds_ is not None and (
            len(bounds_) != 2
            or not np.all(np.isfinite(bounds_))
            or bounds_[0] > bounds_[1]
        ):
            raise ValueError("bounds must be an ordered finite pair.")
        self.nonfinite = nonfinite
        self.bounds = bounds_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "geophysical-observation-policy",
                "nonfinite": nonfinite,
                "bounds": bounds_,
            }
        )


class PreparedGeophysicalObservations(StrictModule, NonTrainableState):
    operator: GeophysicalObservationOperator
    sequence: ObservationSequence
    error_variance: Array
    qc_rejected: Array
    unavailable: Array
    policy: GeophysicalObservationPolicy
    preparation_id: str = eqx.field(static=True)


def prepare_geophysical_observations(
    operator: GeophysicalObservationOperator,
    times: Sequence[str] | ArrayLike,
    values: ArrayLike,
    error_std: ArrayLike,
    /,
    *,
    quantity: GeophysicalQuantity,
    time: GeophysicalTimeSpec,
    temporal: TemporalSupport | None = None,
    target_support_id: str,
    availability: ArrayLike | None = None,
    qc_accept: ArrayLike | None = None,
    representativeness_std: ArrayLike = 0.0,
    policy: GeophysicalObservationPolicy | None = None,
    case_id: str = "geophysical-case",
) -> PreparedGeophysicalObservations:
    """Prepare one physical case with shape ``(time, *target_field_shape)``.

    Scalar errors or arrays exactly matching ``values`` are accepted. Missing
    errors may be nonfinite only for masked observations. The native sequence
    receives finite placeholders AND its mask; placeholders are never evidence.
    Time is checked by calendar/epoch/unit identity, with no nearest-time snap.
    """
    if not isinstance(operator, GeophysicalObservationOperator):
        raise TypeError("operator must be a GeophysicalObservationOperator.")
    if (
        not isinstance(quantity, GeophysicalQuantity)
        or quantity.compatibility_id != operator.quantity.compatibility_id
    ):
        raise ValueError(
            "Observation quantity is physically incompatible with the operator."
        )
    if not isinstance(time, GeophysicalTimeSpec) or time.time_id != operator.time.time_id:
        raise ValueError(
            "Observation calendar, epoch and time unit must match the operator."
        )
    temporal_ = TemporalSupport() if temporal is None else temporal
    if temporal_.support_id != operator.temporal.support_id:
        raise ValueError("Observation temporal support does not match the operator.")
    if target_support_id != operator.transfer.target.support_id:
        raise ValueError(
            "Observation target support does not match the prepared operator."
        )
    policy_ = GeophysicalObservationPolicy() if policy is None else policy
    if not isinstance(policy_, GeophysicalObservationPolicy):
        raise TypeError("policy must be GeophysicalObservationPolicy.")
    raw_times = np.asarray(times)
    if raw_times.ndim != 1:
        raise ValueError("Observation times must be a one-dimensional vector.")
    time_values = (
        time.encode(tuple(str(value) for value in raw_times))
        if raw_times.dtype.kind in ("U", "S", "O")
        else np.asarray(raw_times, dtype=float)
    )
    if (
        time_values.ndim != 1
        or not time_values.size
        or not np.all(np.isfinite(time_values))
        or np.any(np.diff(time_values) <= 0)
    ):
        raise ValueError(
            "Observation times must be a finite, strictly increasing vector."
        )
    raw = np.asarray(values, dtype=float)
    shape = (len(time_values),) + operator.transfer.target.vector_space.shape
    if raw.shape != shape:
        raise ValueError(f"Observation values must have exact shape {shape}.")

    def mask_array(value, name):
        if value is None:
            return np.ones(shape, dtype=bool)
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind != "b":
            raise ValueError(
                f"{name} must be a boolean array with exact observation shape."
            )
        return array

    available = mask_array(availability, "availability")
    accepted = mask_array(qc_accept, "qc_accept")
    finite = np.isfinite(raw)
    if policy_.nonfinite == "raise" and np.any(available & accepted & ~finite):
        raise ValueError("Available, QC-accepted observations must be finite.")
    factor = float(quantity.si_factor / operator.quantity.si_factor)
    converted = raw * factor
    accepted = accepted & finite
    if policy_.bounds is not None:
        accepted &= (converted >= policy_.bounds[0]) & (converted <= policy_.bounds[1])
    mask = available & accepted

    def errors(value, name, positive):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.broadcast_to(array, shape)
        elif array.shape != shape:
            raise ValueError(f"{name} must be scalar or have exact observation shape.")
        invalid = ~np.isfinite(array) | (array <= 0 if positive else array < 0)
        if np.any(mask & invalid):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'} on accepted observations."
            )
        return np.where(mask, array * factor, 0.0)

    measurement = errors(error_std, "error_std", True)
    representation = errors(representativeness_std, "representativeness_std", False)
    variance = np.where(mask, measurement**2 + representation**2, 1.0)
    if np.any(mask & (~np.isfinite(variance) | (variance <= 0))):
        raise ValueError(
            "Accepted observation variance must be finite and strictly positive."
        )
    clean = np.where(mask, converted, 0.0)
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-geophysical-observations",
            "operator": operator.operator_id,
            "policy": policy_.policy_id,
            "case": case_id,
            "input_quantity": quantity.quantity_id,
            "data": array_tree_fingerprint(
                (time_values, clean, variance, mask, available, accepted)
            ),
        }
    )
    target_rank = len(operator.transfer.target.vector_space.shape)
    sequence = ObservationSequence(
        time_values,
        clean,
        observation_axes=tuple(f"observation_{index}" for index in range(target_rank)),
        observation_mask=mask,
        case_ids=(case_id,),
        sequence_id=preparation_id,
        sensor_id=operator.operator_id,
        discretization_id=operator.transfer.target.field_space_id,
    )
    return PreparedGeophysicalObservations(
        operator=operator,
        sequence=sequence,
        error_variance=jnp.asarray(variance),
        qc_rejected=jnp.asarray(available & ~accepted),
        unavailable=jnp.asarray(~available),
        policy=policy_,
        preparation_id=preparation_id,
    )


__all__ = [
    "GeophysicalObservationOperator",
    "GeophysicalObservationPolicy",
    "PreparedGeophysicalObservations",
    "prepare_geophysical_observations",
    "prepare_tensor_observation_operator",
]
