#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from phydrax.domain import AbstractGeometry, AbstractScalarDomain, Domain, DomainFunction

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import inverse_distance_stencil
from .._strict import StrictModule
from ._observation import PointObservationAction


CardinalInterpolation: TypeAlias = Literal["idw", "compact"]
CardinalExtensionScope: TypeAlias = Literal["global_idw", "local_compact"]


def _domain_factor(domain: Domain, label: str, /) -> object:
    return domain.factor(label)


def _anchor_coordinates(
    domain: Domain,
    action: PointObservationAction,
    /,
) -> tuple[Array, ...]:
    count = action.observation_count
    coordinates: list[Array] = []
    for label in domain.labels:
        if label not in action.batch:
            raise KeyError(f"Point observation batch is missing domain label {label!r}.")
        field = action.batch[label]
        if not isinstance(field, cx.Field):
            raise TypeError(
                "Cardinal point observations require array-valued coordax fields."
            )
        value = jnp.asarray(field.data, dtype=float)
        factor = _domain_factor(domain, label)
        if isinstance(factor, AbstractGeometry):
            dimension = int(factor.spatial_dim)
            if value.ndim == 1 and int(value.shape[0]) == dimension:
                value = jnp.broadcast_to(value, (count, dimension))
            if value.ndim != 2 or int(value.shape[1]) != dimension:
                raise ValueError(
                    f"Geometry observations for {label!r} require shape "
                    f"(N, {dimension}), got {value.shape}."
                )
        elif isinstance(factor, AbstractScalarDomain):
            if value.ndim == 0:
                value = jnp.broadcast_to(value, (count,))
            elif value.ndim == 2 and int(value.shape[1]) == 1:
                value = value[:, 0]
            if value.ndim != 1:
                raise ValueError(
                    f"Scalar observations for {label!r} require shape (N,), "
                    f"got {value.shape}."
                )
        else:
            raise TypeError(
                f"Cardinal corrections do not support factor "
                f"{type(factor).__name__} for label {label!r}."
            )
        if int(value.shape[0]) != count:
            raise ValueError(
                f"Observation coordinate {label!r} has {value.shape[0]} rows; "
                f"expected {count}."
            )
        coordinates.append(value)
    return tuple(coordinates)


def _validate_distinct_anchors(anchors: tuple[Array, ...], count: int, /) -> None:
    coordinates = jnp.concatenate(
        tuple(jnp.asarray(anchor).reshape((count, -1)) for anchor in anchors),
        axis=1,
    )
    coincident = jnp.all(
        coordinates[:, None, :] == coordinates[None, :, :],
        axis=-1,
    )
    coincident = coincident & ~jnp.eye(count, dtype=bool)
    if bool(jnp.any(coincident)):
        raise ValueError(
            "Cardinal correction anchors must be pairwise distinct; coincident "
            "observations do not admit independent exact cardinal actions."
        )


def _positive_vector(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 0:
        array = jnp.broadcast_to(array, (count,))
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape {(count,)}, got {array.shape}.")
    if bool(jnp.any(~jnp.isfinite(array))) or bool(jnp.any(array <= 0.0)):
        raise ValueError(f"{name} values must be finite and positive.")
    return array


def _nonnegative_vector(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 0:
        array = jnp.broadcast_to(array, (count,))
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape {(count,)}, got {array.shape}.")
    if bool(jnp.any(~jnp.isfinite(array))) or bool(jnp.any(array < 0.0)):
        raise ValueError(f"{name} values must be finite and non-negative.")
    return array


class CardinalCorrectionEvidence(StrictModule):
    """Exactness and extension scope of a finite cardinal correction family."""

    provider_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    restriction_scope: str = eqx.field(static=True)
    extension_scope: CardinalExtensionScope = eqx.field(static=True)
    local_support: bool = eqx.field(static=True)
    preserves_multiplier_zeros: bool = eqx.field(static=True)
    uses_envelopes: bool = eqx.field(static=True)
    interpolation_exact_off_support: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider_id: str,
        action_id: str,
        field: str,
        observation_count: int,
        extension_scope: CardinalExtensionScope,
        preserves_multiplier_zeros: bool,
        uses_envelopes: bool,
    ):
        self.provider_id = str(provider_id)
        self.action_id = str(action_id)
        self.field = str(field)
        self.observation_count = int(observation_count)
        self.restriction_scope = "exact_finite_observations"
        self.extension_scope = extension_scope
        self.local_support = extension_scope == "local_compact"
        self.preserves_multiplier_zeros = bool(preserves_multiplier_zeros)
        self.uses_envelopes = bool(uses_envelopes)
        # IDW and compact weights define an extension; neither turns the finite
        # observations into a continuum equality.
        self.interpolation_exact_off_support = False


class _CardinalBasisEvaluator(StrictModule):
    labels: tuple[str, ...] = eqx.field(static=True)
    geometry: tuple[bool, ...] = eqx.field(static=True)
    anchors: tuple[Array, ...]
    lengthscales: tuple[Array, ...]
    interpolation: CardinalInterpolation = eqx.field(static=True)
    exponent: Array
    snap_tolerance_squared: Array
    support_radius: Array
    valid: Array
    source_index: Array
    envelope_enabled: tuple[bool, ...] = eqx.field(static=True)
    envelope_scale: Array
    preservation_weight: DomainFunction | None
    preservation_positions: tuple[int, ...] = eqx.field(static=True)
    multiplier_at_anchors: Array

    def _distance_squared(self, args: tuple[Any, ...], /) -> Array:
        if len(args) != len(self.labels):
            raise ValueError(
                f"Cardinal basis expected {len(self.labels)} coordinates, got {len(args)}."
            )
        count = int(self.valid.shape[0])
        tuple_positions = tuple(
            i for i, value in enumerate(args) if isinstance(value, tuple)
        )
        if not tuple_positions:
            distance = jnp.zeros((count,), dtype=float)
            for anchor, scale, geometry, query in zip(
                self.anchors,
                self.lengthscales,
                self.geometry,
                args,
                strict=True,
            ):
                query_array = jnp.asarray(query, dtype=float)
                if geometry:
                    difference = (anchor - query_array[None, :]) / scale[:, None]
                    distance = distance + jnp.sum(difference * difference, axis=1)
                else:
                    difference = (anchor - query_array) / scale
                    distance = distance + difference * difference
            return distance

        axis_positions: dict[tuple[int, int], int] = {}
        axis_count = 0
        for position in tuple_positions:
            coordinates = args[position]
            for coordinate in range(len(coordinates)):
                axis_positions[(position, coordinate)] = axis_count
                axis_count += 1
        distance = jnp.zeros((1,) * axis_count + (count,), dtype=float)
        for position, (anchor, scale, geometry, query) in enumerate(
            zip(
                self.anchors,
                self.lengthscales,
                self.geometry,
                args,
                strict=True,
            )
        ):
            if isinstance(query, tuple):
                if not geometry or anchor.ndim != 2:
                    raise TypeError(
                        f"Coordinate-separable evaluation requires geometry label "
                        f"{self.labels[position]!r}."
                    )
                if len(query) != int(anchor.shape[1]):
                    raise ValueError(
                        f"Coordinate-separable label {self.labels[position]!r} expects "
                        f"{anchor.shape[1]} axes, got {len(query)}."
                    )
                for coordinate, values in enumerate(query):
                    vector = jnp.asarray(values, dtype=float).reshape((-1,))
                    shape = [1] * axis_count + [count]
                    shape[axis_positions[(position, coordinate)]] = int(vector.shape[0])
                    query_axis = vector.reshape(tuple(shape[:-1]) + (1,))
                    anchor_axis = anchor[:, coordinate].reshape(
                        (1,) * axis_count + (count,)
                    )
                    scale_axis = scale.reshape((1,) * axis_count + (count,))
                    difference = (anchor_axis - query_axis) / scale_axis
                    distance = distance + difference * difference
                continue
            query_array = jnp.asarray(query, dtype=float)
            if geometry:
                difference = (anchor - query_array[None, :]) / scale[:, None]
                addition = jnp.sum(difference * difference, axis=1)
            else:
                difference = (anchor - query_array) / scale
                addition = difference * difference
            distance = distance + addition.reshape((1,) * axis_count + (count,))
        return distance

    def _weights(self, distance: Array, /) -> Array:
        count = int(self.valid.shape[0])
        candidate_indices = jnp.broadcast_to(
            jnp.arange(count, dtype=jnp.int32),
            distance.shape,
        )
        candidate_shape = (1,) * (distance.ndim - 1) + (count,)
        valid = jnp.broadcast_to(self.valid.reshape(candidate_shape), distance.shape)
        snap = jnp.broadcast_to(
            self.snap_tolerance_squared.reshape(candidate_shape),
            distance.shape,
        )
        if self.interpolation == "idw":
            exponent = jnp.broadcast_to(
                self.exponent.reshape(candidate_shape), distance.shape
            )
            return inverse_distance_stencil(
                candidate_indices,
                distance,
                source_size=count,
                valid=valid,
                power=exponent,
                regularization=1e-12,
                snap_tolerance_squared=snap,
                snap_policy="first",
            ).weights

        radius = self.support_radius.reshape(candidate_shape)
        normalized_distance = jnp.sqrt(jnp.maximum(distance, 0.0)) / radius
        one_minus = jnp.maximum(1.0 - normalized_distance, 0.0)
        raw = jnp.where(
            valid & (normalized_distance < 1.0),
            (one_minus**4) * (4.0 * normalized_distance + 1.0),
            0.0,
        )
        total = jnp.sum(raw, axis=-1, keepdims=True)
        compact = raw / jnp.where(total > 0.0, total, 1.0)
        masked_distance = jnp.where(valid, distance, jnp.inf)
        nearest = jnp.argmin(masked_distance, axis=-1)
        nearest_distance = jnp.take_along_axis(
            masked_distance, nearest[..., None], axis=-1
        )[..., 0]
        nearest_snap = jnp.take_along_axis(snap, nearest[..., None], axis=-1)[..., 0]
        should_snap = (nearest_distance == 0.0) | (nearest_distance < nearest_snap)
        snap_weights = jax.nn.one_hot(nearest, count, dtype=distance.dtype) * valid
        return jnp.where(should_snap[..., None], snap_weights, compact)

    def _envelope(self, distance: Array, /) -> Array:
        if not any(self.envelope_enabled):
            return jnp.ones_like(distance)
        count = int(self.valid.shape[0])
        candidate_shape = (1,) * (distance.ndim - 1) + (count,)
        source = self.source_index.reshape(candidate_shape)
        factors: list[Array] = []
        for index, enabled in enumerate(self.envelope_enabled):
            if not enabled:
                factors.append(jnp.ones(distance.shape[:-1], dtype=float))
                continue
            mask = source == index
            nearest = jnp.min(jnp.where(mask, distance, jnp.inf), axis=-1)
            scale = self.envelope_scale[index]
            factors.append(jnp.exp(-(nearest / (scale * scale + 1e-12))))
        by_source = jnp.stack(factors, axis=-1)
        return jnp.take_along_axis(by_source, source, axis=-1)

    def _query_multiplier(
        self, args: tuple[Any, ...], /, *, key=None, **kwargs: Any
    ) -> Array:
        if self.preservation_weight is None:
            return jnp.asarray(1.0, dtype=float)
        selected = tuple(args[position] for position in self.preservation_positions)
        return jnp.asarray(
            self.preservation_weight.func(*selected, key=key, **kwargs),
            dtype=float,
        )

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        distance = self._distance_squared(args)
        weights = self._weights(distance) * self._envelope(distance)
        multiplier = self._query_multiplier(args, key=key, **kwargs)
        while multiplier.ndim < weights.ndim - 1:
            multiplier = jnp.expand_dims(multiplier, axis=-1)
        scaled = weights / self.multiplier_at_anchors.reshape(
            (1,) * (weights.ndim - 1) + self.multiplier_at_anchors.shape
        )
        return multiplier[..., None] * scaled


class _CardinalLinearCombination(StrictModule):
    basis: DomainFunction
    coefficients: Array
    components: tuple[int, ...] | None = eqx.field(static=True)
    output_width: int | None = eqx.field(static=True)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        basis_values = jnp.asarray(self.basis.func(*args, key=key, **kwargs), dtype=float)
        coefficients = jnp.asarray(self.coefficients)
        event_shape = tuple(int(size) for size in coefficients.shape[1:])
        flat = coefficients.reshape((int(coefficients.shape[0]), -1))
        combined = contract("...i,ij->...j", basis_values, flat)
        combined = combined.reshape(basis_values.shape[:-1] + event_shape)
        if self.components is None:
            return combined
        width = self.output_width
        if width is None:
            raise ValueError("Component corrections require a declared output width.")
        if combined.ndim == basis_values.ndim - 1:
            combined = combined[..., None]
        out = jnp.zeros(combined.shape[:-1] + (width,), dtype=combined.dtype)
        return out.at[..., jnp.asarray(self.components, dtype=jnp.int32)].set(combined)


class CardinalCorrectionAction(StrictModule):
    """Candidate/final correction action consumed by the joint affine projector."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    basis: DomainFunction
    components: tuple[int, ...] | None = eqx.field(static=True)
    output_width: int | None = eqx.field(static=True)
    evidence: CardinalCorrectionEvidence

    def __init__(
        self,
        *,
        field: str,
        basis: DomainFunction,
        components: tuple[int, ...] | None,
        output_width: int | None,
        evidence: CardinalCorrectionEvidence,
    ):
        field_ = str(field)
        if field_ != evidence.field:
            raise ValueError("Cardinal action field and evidence field disagree.")
        width = None if output_width is None else int(output_width)
        if width is not None and width <= 0:
            raise ValueError("output_width must be positive when provided.")
        if components is not None:
            if width is None:
                raise ValueError(
                    "Component cardinal corrections require a declared output_width."
                )
            if any(component >= width for component in components):
                raise ValueError(
                    f"Cardinal components {components!r} exceed output width {width}."
                )
        self.field_names = (field_,)
        self.basis = basis
        self.components = components
        self.output_width = width
        self.evidence = evidence

    def lift(self, product_residual: Any, /) -> tuple[DomainFunction, ...]:
        residual = product_residual
        if isinstance(product_residual, tuple):
            if len(product_residual) != 1:
                raise ValueError(
                    "A cardinal action expects one residual block; joint response "
                    "assembly must combine multiple blocks before calling lift."
                )
            residual = product_residual[0]
        coefficients = jnp.asarray(residual)
        count = self.evidence.observation_count
        if coefficients.ndim == 0 or int(coefficients.shape[0]) != count:
            raise ValueError(
                f"Cardinal residual must have leading observation size {count}, "
                f"got {coefficients.shape}."
            )
        if self.components is not None and (
            coefficients.ndim < 2 or int(coefficients.shape[-1]) != len(self.components)
        ):
            raise ValueError(
                "Component cardinal residuals must have a trailing axis with "
                f"{len(self.components)} entries, got {coefficients.shape}."
            )
        correction = DomainFunction(
            domain=self.basis.domain,
            deps=self.basis.deps,
            func=_CardinalLinearCombination(
                self.basis,
                coefficients,
                self.components,
                self.output_width,
            ),
            metadata={
                "cardinal_correction": True,
                "provider_id": self.evidence.provider_id,
                "exact_scope": self.evidence.restriction_scope,
            },
        )
        return (correction,)

    __call__ = lift


class CardinalCorrectionPlan(StrictModule):
    """Immutable finite-observation cardinal correction provider plan."""

    action: PointObservationAction
    domain: Domain
    interpolation: CardinalInterpolation = eqx.field(static=True)
    anchors: tuple[Array, ...]
    lengthscales: tuple[Array, ...]
    exponent: Array
    snap_tolerance_squared: Array
    support_radius: Array
    valid: Array
    source_index: Array
    envelope_enabled: tuple[bool, ...] = eqx.field(static=True)
    envelope_scale: Array
    preservation_weight: DomainFunction | None
    provider_id: str = eqx.field(static=True)
    evidence: CardinalCorrectionEvidence
    _preservation_positions: tuple[int, ...] = eqx.field(static=True)
    _multiplier_at_anchors: Array

    def __init__(
        self,
        action: PointObservationAction,
        domain: Domain,
        /,
        *,
        interpolation: CardinalInterpolation = "idw",
        exponent: ArrayLike = 2.0,
        snap_tolerance_squared: ArrayLike = 1e-12,
        lengthscales: Mapping[str, ArrayLike] | None = None,
        support_radius: ArrayLike = 1.0,
        valid: ArrayLike | None = None,
        source_index: ArrayLike | None = None,
        envelope_enabled: Sequence[bool] = (),
        envelope_scale: ArrayLike = 1.0,
        preservation_weight: DomainFunction | None = None,
    ):
        if not isinstance(action, PointObservationAction):
            raise TypeError("CardinalCorrectionPlan requires PointObservationAction.")
        if not isinstance(domain, Domain):
            raise TypeError("CardinalCorrectionPlan requires a Domain.")
        if interpolation not in ("idw", "compact"):
            raise ValueError("interpolation must be 'idw' or 'compact'.")
        count = action.observation_count
        anchors = _anchor_coordinates(domain, action)
        _validate_distinct_anchors(anchors, count)
        lengthscale_values = {} if lengthscales is None else dict(lengthscales)
        unknown = tuple(
            label for label in lengthscale_values if label not in domain.labels
        )
        if unknown:
            raise KeyError(f"Unknown cardinal lengthscale labels {unknown!r}.")
        lengthscale_tuple = tuple(
            _positive_vector(
                f"lengthscales[{label!r}]",
                lengthscale_values.get(label, 1.0),
                count,
            )
            for label in domain.labels
        )
        exponent_ = _positive_vector("exponent", exponent, count)
        snap_ = _nonnegative_vector(
            "snap_tolerance_squared", snap_tolerance_squared, count
        )
        radius_ = _positive_vector("support_radius", support_radius, count)
        valid_ = (
            jnp.ones((count,), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (count,) or not bool(jnp.all(valid_)):
            raise ValueError(
                "Every declared observation must be valid; mask invalid padded rows "
                "out of the PointObservationAction before exact preparation."
            )
        source_ = (
            jnp.zeros((count,), dtype=jnp.int32)
            if source_index is None
            else jnp.asarray(source_index, dtype=jnp.int32)
        )
        if source_.shape != (count,) or bool(jnp.any(source_ < 0)):
            raise ValueError("source_index must have shape (N,) and be non-negative.")
        source_count = int(jnp.max(source_)) + 1
        enabled = (
            tuple(False for _ in range(source_count))
            if not envelope_enabled
            else tuple(bool(value) for value in envelope_enabled)
        )
        if len(enabled) != source_count:
            raise ValueError(
                f"envelope_enabled must contain {source_count} source entries."
            )
        envelope_scale_ = _positive_vector("envelope_scale", envelope_scale, source_count)
        if preservation_weight is not None:
            if not isinstance(preservation_weight, DomainFunction):
                raise TypeError("preservation_weight must be a DomainFunction.")
            if not preservation_weight.domain.same_support(domain):
                raise ValueError("preservation_weight must share the correction domain.")
            evaluated = preservation_weight(action.batch)
            if not isinstance(evaluated, cx.Field):
                raise TypeError("preservation_weight batch evaluation must return Field.")
            multiplier = jnp.asarray(evaluated.data, dtype=float)
            if multiplier.ndim > 1 and multiplier.shape[-1:] == (1,):
                multiplier = multiplier[..., 0]
            if multiplier.ndim == 0:
                multiplier = jnp.broadcast_to(multiplier, (count,))
            if multiplier.shape != (count,):
                raise ValueError(
                    "preservation_weight must be scalar-valued at every observation."
                )
            if bool(jnp.any(~jnp.isfinite(multiplier))) or bool(
                jnp.any(multiplier == 0.0)
            ):
                raise ValueError(
                    "Observations cannot lie where the preservation multiplier vanishes."
                )
            preservation_positions = tuple(
                domain.labels.index(label) for label in preservation_weight.deps
            )
        else:
            multiplier = jnp.ones((count,), dtype=float)
            preservation_positions = ()
        provider_id = canonical_fingerprint(
            {
                "kind": "cardinal-correction-plan-v1",
                "action": action.action_id,
                "interpolation": interpolation,
                "anchors": array_tree_fingerprint(anchors),
                "lengthscales": array_tree_fingerprint(lengthscale_tuple),
                "exponent": array_tree_fingerprint(exponent_),
                "snap": array_tree_fingerprint(snap_),
                "radius": array_tree_fingerprint(radius_),
                "source": array_tree_fingerprint(source_),
                "envelopes": enabled,
                "envelope_scale": array_tree_fingerprint(envelope_scale_),
                "preservation": None
                if preservation_weight is None
                else repr(preservation_weight.deps),
            }
        )
        scope: CardinalExtensionScope = (
            "global_idw" if interpolation == "idw" else "local_compact"
        )
        evidence = CardinalCorrectionEvidence(
            provider_id=provider_id,
            action_id=action.action_id,
            field=action.field,
            observation_count=count,
            extension_scope=scope,
            preserves_multiplier_zeros=preservation_weight is not None,
            uses_envelopes=any(enabled),
        )
        self.action = action
        self.domain = domain
        self.interpolation = interpolation
        self.anchors = anchors
        self.lengthscales = lengthscale_tuple
        self.exponent = exponent_
        self.snap_tolerance_squared = snap_
        self.support_radius = radius_
        self.valid = valid_
        self.source_index = source_
        self.envelope_enabled = enabled
        self.envelope_scale = envelope_scale_
        self.preservation_weight = preservation_weight
        self.provider_id = provider_id
        self.evidence = evidence
        self._preservation_positions = preservation_positions
        self._multiplier_at_anchors = multiplier

    def candidate_action(
        self,
        *,
        output_width: int | None = None,
    ) -> CardinalCorrectionAction:
        geometry = tuple(
            isinstance(_domain_factor(self.domain, label), AbstractGeometry)
            for label in self.domain.labels
        )
        evaluator = _CardinalBasisEvaluator(
            self.domain.labels,
            geometry,
            self.anchors,
            self.lengthscales,
            self.interpolation,
            self.exponent,
            self.snap_tolerance_squared,
            self.support_radius,
            self.valid,
            self.source_index,
            self.envelope_enabled,
            self.envelope_scale,
            self.preservation_weight,
            self._preservation_positions,
            self._multiplier_at_anchors,
        )
        basis = DomainFunction(
            domain=self.domain,
            deps=self.domain.labels,
            func=evaluator,
            metadata={
                "cardinal_basis": True,
                "provider_id": self.provider_id,
                "exact_scope": "finite_observations",
                "extension_scope": self.evidence.extension_scope,
            },
        )
        return CardinalCorrectionAction(
            field=self.action.field,
            basis=basis,
            components=self.action.components,
            output_width=output_width,
            evidence=self.evidence,
        )


class IDWCardinalCorrectionProvider(StrictModule):
    """Global inverse-distance cardinal correction for finite observations."""

    plan: CardinalCorrectionPlan

    def __init__(
        self,
        action: PointObservationAction,
        domain: Domain,
        /,
        **kwargs: Any,
    ):
        self.plan = CardinalCorrectionPlan(
            action,
            domain,
            interpolation="idw",
            **kwargs,
        )

    @property
    def provider_id(self) -> str:
        return self.plan.provider_id

    @property
    def evidence(self) -> CardinalCorrectionEvidence:
        return self.plan.evidence

    def candidate_action(
        self,
        *,
        output_width: int | None = None,
    ) -> CardinalCorrectionAction:
        return self.plan.candidate_action(output_width=output_width)


class CompactCardinalCorrectionProvider(StrictModule):
    """Local Wendland-C2 cardinal correction with explicit finite support."""

    plan: CardinalCorrectionPlan

    def __init__(
        self,
        action: PointObservationAction,
        domain: Domain,
        /,
        **kwargs: Any,
    ):
        self.plan = CardinalCorrectionPlan(
            action,
            domain,
            interpolation="compact",
            **kwargs,
        )

    @property
    def provider_id(self) -> str:
        return self.plan.provider_id

    @property
    def evidence(self) -> CardinalCorrectionEvidence:
        return self.plan.evidence

    def candidate_action(
        self,
        *,
        output_width: int | None = None,
    ) -> CardinalCorrectionAction:
        return self.plan.candidate_action(output_width=output_width)


__all__ = [
    "CardinalCorrectionAction",
    "CardinalCorrectionEvidence",
    "CardinalCorrectionPlan",
    "CardinalExtensionScope",
    "CardinalInterpolation",
    "CompactCardinalCorrectionProvider",
    "IDWCardinalCorrectionProvider",
]
