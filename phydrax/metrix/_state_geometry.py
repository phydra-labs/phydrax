#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.sparse.linalg as jsparse
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _same_shape(value: Array, reference: Array, name: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must preserve state shape {reference.shape}; got {value.shape}."
        )


def _norm(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.linalg.norm(array.reshape((-1,)))


def _pair(covector: ArrayLike, vector: ArrayLike, /) -> Array:
    covector_array = jnp.asarray(covector)
    vector_array = jnp.asarray(vector)
    if covector_array.shape != vector_array.shape:
        raise ValueError(
            "Algebraic duality requires covector and vector coordinates "
            f"with identical shapes; got {covector_array.shape} and "
            f"{vector_array.shape}."
        )
    return jnp.sum(covector_array * vector_array)


def _relative_residual(left: ArrayLike, right: ArrayLike, /) -> Array:
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "Residual operands must have identical shapes; "
            f"got {left_array.shape} and {right_array.shape}."
        )
    scale = jnp.maximum(1.0, jnp.maximum(_norm(left_array), _norm(right_array)))
    return _norm(left_array - right_array) / scale


class StateChartEvidence(StrictModule):
    """Machine-checkable inverse, differential, and duality chart evidence."""

    source_membership: Array
    target_membership: Array
    finite: Array
    inverse_roundtrip_residual: Array
    forward_inverse_differential_residual: Array
    inverse_forward_differential_residual: Array
    vjp_duality_residual: Array
    cut_locus_margin: Array
    scale: Array
    valid: Array


class StateTransportEvidence(StrictModule):
    """Machine-checkable tangent transport and cotangent pullback evidence."""

    source_membership: Array
    target_membership: Array
    finite: Array
    identity_residual: Array
    roundtrip_residual: Array
    duality_residual: Array
    isometry_residual: Array
    isometry_claimed: Array
    scale: Array
    valid: Array


class AbstractStateGeometry(StrictModule):
    """Four-space retraction geometry for one array-valued state.

    Point storage, local perturbations, physical tangents, local covectors, and
    physical covectors are distinct roles and need not have equal shapes.
    Differential operations name their exact domain and codomain. VJPs are
    algebraic transposes and never silently apply a Riesz map.
    """

    geometry_id: AbstractAttribute[str]
    retraction_method: AbstractAttribute[str]
    trivial: AbstractAttribute[bool]
    supports_exact_inverse: AbstractAttribute[bool]
    supports_exact_differential: AbstractAttribute[bool]
    supports_transport: AbstractAttribute[bool]
    supports_isometric_transport: AbstractAttribute[bool]
    supports_commutator_free: AbstractAttribute[bool]

    @abstractmethod
    def contains(self, state: ArrayLike, /) -> Array:
        """Return one scalar boolean indicating membership in the point space."""
        raise NotImplementedError

    @abstractmethod
    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        """Project a point-storage ambient vector to a physical tangent."""
        raise NotImplementedError

    @abstractmethod
    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        """Map a local perturbation at ``state`` to point storage."""
        raise NotImplementedError

    @abstractmethod
    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        """Return local perturbation coordinates for a nearby point."""
        raise NotImplementedError

    @abstractmethod
    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        """Push a local velocity to a physical tangent at the retracted point."""
        raise NotImplementedError

    @abstractmethod
    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Push a physical tangent through the inverse chart to local velocity."""
        raise NotImplementedError

    @abstractmethod
    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        """Algebraically transpose the retraction differential to a local covector."""
        raise NotImplementedError

    @abstractmethod
    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Transport a physical tangent from ``state`` to ``point``."""
        raise NotImplementedError

    @abstractmethod
    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        """Pull a physical cotangent at ``point`` back to ``state``."""
        raise NotImplementedError

    @abstractmethod
    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        """Return a non-negative scalar margin for the supported inverse chart."""
        raise NotImplementedError

    def local_retraction(self, state: ArrayLike, /) -> LocalRetraction:
        """Bind this geometry's exact retraction operations to one base point."""
        return LocalRetraction(self, state)

    def interpolate(
        self,
        left: ArrayLike,
        right: ArrayLike,
        weight: ArrayLike,
        /,
    ) -> Array:
        local = self.inverse_retract(left, right)
        scaled = jnp.asarray(weight) * jnp.asarray(local)
        return self.retract(left, scaled)

    def chart_evidence(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        require_inverse: bool = True,
        require_differential: bool = True,
    ) -> StateChartEvidence:
        """Audit a chart inverse and its two differential directions."""
        source = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        local_direction = jnp.asarray(local_velocity)
        target_covector = jnp.asarray(cotangent)
        target = jnp.asarray(self.retract(source, local))
        source_membership = jnp.asarray(self.contains(source), dtype=bool)
        target_membership = jnp.asarray(self.contains(target), dtype=bool)
        dtype = jnp.result_type(source.dtype, local.dtype, jnp.float32)
        unavailable = jnp.asarray(jnp.nan, dtype=dtype)
        inverse_residual = unavailable
        forward_inverse_residual = unavailable
        inverse_forward_residual = unavailable
        vjp_residual = unavailable
        finite_terms = [
            jnp.all(jnp.isfinite(source)),
            jnp.all(jnp.isfinite(local)),
            jnp.all(jnp.isfinite(local_direction)),
            jnp.all(jnp.isfinite(target_covector)),
            jnp.all(jnp.isfinite(target)),
        ]
        if require_inverse and self.supports_exact_inverse:
            recovered_local = jnp.asarray(self.inverse_retract(source, target))
            inverse_residual = _relative_residual(recovered_local, local)
            finite_terms.append(jnp.all(jnp.isfinite(recovered_local)))
        if require_differential and self.supports_exact_differential:
            pushed = jnp.asarray(self.retraction_jvp(source, local, local_direction))
            recovered_direction = jnp.asarray(
                self.retraction_inverse_jvp(source, target, pushed)
            )
            reconstructed_tangent = jnp.asarray(
                self.retraction_jvp(source, local, recovered_direction)
            )
            local_covector = jnp.asarray(
                self.retraction_vjp(source, local, target_covector)
            )
            forward_inverse_residual = _relative_residual(
                recovered_direction,
                local_direction,
            )
            inverse_forward_residual = _relative_residual(
                reconstructed_tangent,
                pushed,
            )
            target_pairing = _pair(target_covector, pushed)
            local_pairing = _pair(local_covector, local_direction)
            pairing_scale = jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(target_pairing), jnp.abs(local_pairing)),
            )
            vjp_residual = jnp.abs(target_pairing - local_pairing) / pairing_scale
            finite_terms.extend(
                (
                    jnp.all(jnp.isfinite(pushed)),
                    jnp.all(jnp.isfinite(recovered_direction)),
                    jnp.all(jnp.isfinite(reconstructed_tangent)),
                    jnp.all(jnp.isfinite(local_covector)),
                )
            )
        margin = jnp.asarray(self.cut_locus_margin(source, target), dtype=dtype)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.maximum(_norm(source), _norm(target)),
                jnp.maximum(_norm(local), _norm(local_direction)),
            ),
        )
        finite = jnp.all(jnp.stack(tuple(finite_terms))) & jnp.isfinite(scale)
        tolerance = jnp.sqrt(jnp.finfo(dtype).eps) * max(
            8,
            int((source.size + local.size + local_direction.size) ** 0.5),
        )
        inverse_valid = jnp.asarray(not require_inverse) | (
            jnp.asarray(self.supports_exact_inverse)
            & (inverse_residual <= tolerance)
            & (margin >= 0.0)
        )
        differential_valid = jnp.asarray(not require_differential) | (
            jnp.asarray(self.supports_exact_differential)
            & (forward_inverse_residual <= tolerance)
            & (inverse_forward_residual <= tolerance)
            & (vjp_residual <= tolerance)
        )
        valid = (
            source_membership
            & target_membership
            & finite
            & jnp.asarray(inverse_valid)
            & jnp.asarray(differential_valid)
        )
        return StateChartEvidence(
            source_membership,
            target_membership,
            finite,
            inverse_residual,
            forward_inverse_residual,
            inverse_forward_residual,
            vjp_residual,
            margin,
            scale,
            valid,
        )

    def transport_evidence(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        require_transport: bool = True,
        require_isometry: bool = False,
    ) -> StateTransportEvidence:
        """Audit transport identity, inverse, algebraic duality, and isometry."""
        source = jnp.asarray(state)
        target = jnp.asarray(point)
        source_tangent = jnp.asarray(tangent)
        target_covector = jnp.asarray(cotangent)
        source_membership = jnp.asarray(self.contains(source), dtype=bool)
        target_membership = jnp.asarray(self.contains(target), dtype=bool)
        dtype = jnp.result_type(
            source.dtype,
            source_tangent.dtype,
            target_covector.dtype,
            jnp.float32,
        )
        unavailable = jnp.asarray(jnp.nan, dtype=dtype)
        identity_residual = unavailable
        roundtrip_residual = unavailable
        duality_residual = unavailable
        isometry_residual = unavailable
        finite_terms = [
            jnp.all(jnp.isfinite(source)),
            jnp.all(jnp.isfinite(target)),
            jnp.all(jnp.isfinite(source_tangent)),
            jnp.all(jnp.isfinite(target_covector)),
        ]
        if (
            require_transport or require_isometry or self.supports_isometric_transport
        ) and self.supports_transport:
            identity = jnp.asarray(self.transport_tangent(source, source, source_tangent))
            transported = jnp.asarray(
                self.transport_tangent(source, target, source_tangent)
            )
            recovered = jnp.asarray(self.transport_tangent(target, source, transported))
            pulled = jnp.asarray(
                self.transport_cotangent_pullback(
                    source,
                    target,
                    target_covector,
                )
            )
            identity_residual = _relative_residual(identity, source_tangent)
            roundtrip_residual = _relative_residual(recovered, source_tangent)
            target_pairing = _pair(target_covector, transported)
            source_pairing = _pair(pulled, source_tangent)
            pairing_scale = jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(target_pairing), jnp.abs(source_pairing)),
            )
            duality_residual = jnp.abs(target_pairing - source_pairing) / pairing_scale
            norm_scale = jnp.maximum(
                1.0,
                jnp.maximum(_norm(source_tangent), _norm(transported)),
            )
            isometry_residual = (
                jnp.abs(_norm(transported) - _norm(source_tangent)) / norm_scale
            )
            finite_terms.extend(
                (
                    jnp.all(jnp.isfinite(identity)),
                    jnp.all(jnp.isfinite(transported)),
                    jnp.all(jnp.isfinite(recovered)),
                    jnp.all(jnp.isfinite(pulled)),
                )
            )
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.maximum(_norm(source), _norm(target)),
                jnp.maximum(_norm(source_tangent), _norm(target_covector)),
            ),
        )
        finite = jnp.all(jnp.stack(tuple(finite_terms))) & jnp.isfinite(scale)
        tolerance = jnp.sqrt(jnp.finfo(dtype).eps) * max(
            8,
            int((source.size + source_tangent.size + target_covector.size) ** 0.5),
        )
        transport_valid = jnp.asarray(not require_transport) | (
            jnp.asarray(self.supports_transport)
            & (identity_residual <= tolerance)
            & (roundtrip_residual <= tolerance)
            & (duality_residual <= tolerance)
        )
        isometry_required = require_isometry or self.supports_isometric_transport
        isometry_valid = jnp.asarray(not isometry_required) | (
            jnp.asarray(self.supports_isometric_transport)
            & (isometry_residual <= tolerance)
        )
        valid = (
            source_membership
            & target_membership
            & finite
            & jnp.asarray(transport_valid)
            & jnp.asarray(isometry_valid)
        )
        return StateTransportEvidence(
            source_membership,
            target_membership,
            finite,
            identity_residual,
            roundtrip_residual,
            duality_residual,
            isometry_residual,
            jnp.asarray(self.supports_isometric_transport),
            scale,
            valid,
        )


class LocalRetraction(StrictModule):
    """One validated base point with exact chart differential operations."""

    geometry: AbstractStateGeometry
    base_point: Array
    retraction_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(self, geometry: AbstractStateGeometry, base_point: ArrayLike, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("LocalRetraction geometry must be an AbstractStateGeometry.")
        base = jnp.asarray(base_point)
        membership = jnp.asarray(geometry.contains(base), dtype=bool)
        if membership.shape != ():
            raise ValueError("State geometry contains() must return a scalar boolean.")
        base = eqx.error_if(
            base,
            ~membership,
            "LocalRetraction base point is outside the state geometry.",
        )
        self.geometry = geometry
        self.base_point = base
        self.retraction_id = f"{geometry.geometry_id}:local-retraction"
        self.resolved_method = geometry.retraction_method

    def evaluate(self, local_tangent: ArrayLike, /) -> Array:
        return jnp.asarray(self.geometry.retract(self.base_point, local_tangent))

    def __call__(self, local_tangent: ArrayLike, /) -> Array:
        return self.evaluate(local_tangent)

    def jvp(
        self,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        if not self.geometry.supports_exact_differential:
            raise ValueError(
                "Local retraction JVP requires exact differential capability."
            )
        return jnp.asarray(
            self.geometry.retraction_jvp(
                self.base_point,
                local_tangent,
                local_velocity,
            )
        )

    def inverse_jvp(
        self,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        if not self.geometry.supports_exact_differential:
            raise ValueError(
                "Local inverse-retraction JVP requires exact differential capability."
            )
        return jnp.asarray(
            self.geometry.retraction_inverse_jvp(
                self.base_point,
                point,
                tangent,
            )
        )

    def vjp(
        self,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        if not self.geometry.supports_exact_differential:
            raise ValueError(
                "Local retraction VJP requires exact differential capability."
            )
        return jnp.asarray(
            self.geometry.retraction_vjp(
                self.base_point,
                local_tangent,
                cotangent,
            )
        )

    def chart_evidence(
        self,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        require_inverse: bool = True,
        require_differential: bool = True,
    ) -> StateChartEvidence:
        return self.geometry.chart_evidence(
            self.base_point,
            local_tangent,
            local_velocity,
            cotangent,
            require_inverse=require_inverse,
            require_differential=require_differential,
        )


class EuclideanStateGeometry(AbstractStateGeometry):
    """Explicit equal-space geometry for unconstrained array-valued states."""

    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_id: str = "state-geometry:euclidean",
    ):
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.retraction_method = "addition"
        self.trivial = True
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        return jnp.all(jnp.isfinite(jnp.asarray(state)))

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        vector_array = jnp.asarray(vector)
        _same_shape(vector_array, state_array, "Euclidean tangent")
        return vector_array

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        _same_shape(local, state_array, "Euclidean local tangent")
        return state_array + local

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Euclidean retraction point")
        return point_array - state_array

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        velocity = jnp.asarray(local_velocity)
        _same_shape(local, state_array, "Euclidean local tangent")
        _same_shape(velocity, local, "Euclidean local velocity")
        return velocity

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Euclidean retraction point")
        _same_shape(vector, state_array, "Euclidean tangent")
        return vector

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        covector = jnp.asarray(cotangent)
        _same_shape(local, state_array, "Euclidean local tangent")
        _same_shape(covector, state_array, "Euclidean cotangent")
        return covector

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Euclidean transport point")
        _same_shape(vector, state_array, "Euclidean tangent")
        return vector

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        covector = jnp.asarray(cotangent)
        _same_shape(point_array, state_array, "Euclidean transport point")
        _same_shape(covector, state_array, "Euclidean cotangent")
        return covector

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        _same_shape(jnp.asarray(point), state_array, "Euclidean chart point")
        return jnp.asarray(1.0, dtype=state_array.dtype)


class EmbeddedStateGeometry(AbstractStateGeometry):
    """Adapter for an embedded point space with explicit exact operations."""

    membership: Callable[[Array], Array]
    tangent_projection: Callable[[Array, Array], Array]
    retraction: Callable[[Array, Array], Array]
    inverse_retraction: Callable[[Array, Array], Array] | None
    retraction_jvp_action: Callable[[Array, Array, Array], Array] | None
    retraction_inverse_jvp_action: Callable[[Array, Array, Array], Array] | None
    retraction_vjp_action: Callable[[Array, Array, Array], Array] | None
    tangent_transport_action: Callable[[Array, Array, Array], Array] | None
    cotangent_transport_pullback_action: Callable[[Array, Array, Array], Array] | None
    cut_locus_margin_action: Callable[[Array, Array], Array] | None
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        membership: Callable[[Array], Array],
        tangent_projection: Callable[[Array, Array], Array],
        retraction: Callable[[Array, Array], Array],
        geometry_id: str,
        retraction_method: str,
        inverse_retraction: Callable[[Array, Array], Array] | None = None,
        retraction_jvp_action: (Callable[[Array, Array, Array], Array] | None) = None,
        retraction_inverse_jvp_action: (
            Callable[[Array, Array, Array], Array] | None
        ) = None,
        retraction_vjp_action: (Callable[[Array, Array, Array], Array] | None) = None,
        tangent_transport_action: (Callable[[Array, Array, Array], Array] | None) = None,
        cotangent_transport_pullback_action: (
            Callable[[Array, Array, Array], Array] | None
        ) = None,
        cut_locus_margin_action: Callable[[Array, Array], Array] | None = None,
        isometric_transport: bool = False,
        supports_commutator_free: bool = False,
    ):
        for function, name in (
            (membership, "membership"),
            (tangent_projection, "tangent_projection"),
            (retraction, "retraction"),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be callable.")
        optional_actions = (
            (inverse_retraction, "inverse_retraction"),
            (retraction_jvp_action, "retraction_jvp_action"),
            (
                retraction_inverse_jvp_action,
                "retraction_inverse_jvp_action",
            ),
            (retraction_vjp_action, "retraction_vjp_action"),
            (tangent_transport_action, "tangent_transport_action"),
            (
                cotangent_transport_pullback_action,
                "cotangent_transport_pullback_action",
            ),
            (cut_locus_margin_action, "cut_locus_margin_action"),
        )
        for function, name in optional_actions:
            if function is not None and not callable(function):
                raise TypeError(f"{name} must be callable or None.")
        exact_differential = all(
            action is not None
            for action in (
                retraction_jvp_action,
                retraction_inverse_jvp_action,
                retraction_vjp_action,
            )
        )
        transport = (
            tangent_transport_action is not None
            and cotangent_transport_pullback_action is not None
        )
        if bool(isometric_transport) and not transport:
            raise ValueError("Isometric transport requires both transport actions.")
        self.membership = membership
        self.tangent_projection = tangent_projection
        self.retraction = retraction
        self.inverse_retraction = inverse_retraction
        self.retraction_jvp_action = retraction_jvp_action
        self.retraction_inverse_jvp_action = retraction_inverse_jvp_action
        self.retraction_vjp_action = retraction_vjp_action
        self.tangent_transport_action = tangent_transport_action
        self.cotangent_transport_pullback_action = cotangent_transport_pullback_action
        self.cut_locus_margin_action = cut_locus_margin_action
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.retraction_method = _identifier(
            retraction_method,
            "retraction_method",
        )
        self.trivial = False
        self.supports_exact_inverse = inverse_retraction is not None
        self.supports_exact_differential = exact_differential
        self.supports_transport = transport
        self.supports_isometric_transport = bool(isometric_transport)
        self.supports_commutator_free = bool(supports_commutator_free)

    def contains(self, state: ArrayLike, /) -> Array:
        membership = jnp.asarray(self.membership(jnp.asarray(state)), dtype=bool)
        if membership.shape != ():
            raise ValueError("Embedded membership must return a scalar boolean.")
        return membership

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        return jnp.asarray(
            self.tangent_projection(jnp.asarray(state), jnp.asarray(vector))
        )

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point = jnp.asarray(self.retraction(state_array, jnp.asarray(local_tangent)))
        _same_shape(point, state_array, "Embedded retraction")
        return point

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Embedded retraction point")
        if self.inverse_retraction is None:
            raise ValueError(
                "Embedded inverse_retract requires an explicit "
                "inverse_retraction callable."
            )
        return jnp.asarray(self.inverse_retraction(state_array, point_array))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        if self.retraction_jvp_action is None:
            raise ValueError(
                "Embedded retraction_jvp requires an explicit "
                "retraction_jvp_action callable."
            )
        return jnp.asarray(
            self.retraction_jvp_action(
                jnp.asarray(state),
                jnp.asarray(local_tangent),
                jnp.asarray(local_velocity),
            )
        )

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        if self.retraction_inverse_jvp_action is None:
            raise ValueError(
                "Embedded retraction_inverse_jvp requires an explicit "
                "retraction_inverse_jvp_action callable."
            )
        return jnp.asarray(
            self.retraction_inverse_jvp_action(
                jnp.asarray(state),
                jnp.asarray(point),
                jnp.asarray(tangent),
            )
        )

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        if self.retraction_vjp_action is None:
            raise ValueError(
                "Embedded retraction_vjp requires an explicit "
                "retraction_vjp_action callable."
            )
        return jnp.asarray(
            self.retraction_vjp_action(
                jnp.asarray(state),
                jnp.asarray(local_tangent),
                jnp.asarray(cotangent),
            )
        )

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        if self.tangent_transport_action is None:
            raise ValueError(
                "Embedded transport_tangent requires an explicit "
                "tangent_transport_action callable."
            )
        return jnp.asarray(
            self.tangent_transport_action(
                jnp.asarray(state),
                jnp.asarray(point),
                jnp.asarray(tangent),
            )
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        if self.cotangent_transport_pullback_action is None:
            raise ValueError(
                "Embedded transport_cotangent_pullback requires an explicit "
                "cotangent_transport_pullback_action callable."
            )
        return jnp.asarray(
            self.cotangent_transport_pullback_action(
                jnp.asarray(state),
                jnp.asarray(point),
                jnp.asarray(cotangent),
            )
        )

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        if self.cut_locus_margin_action is None:
            return jnp.asarray(1.0, dtype=jnp.asarray(state).dtype)
        margin = jnp.asarray(
            self.cut_locus_margin_action(jnp.asarray(state), jnp.asarray(point))
        )
        if margin.shape != ():
            raise ValueError("cut_locus_margin_action must return one scalar.")
        return margin


def _role_shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{name} must contain positive dimensions.")
    return shape


class PointwiseStateGeometry(AbstractStateGeometry):
    """Apply one possibly unequal-space geometry over leading batch axes."""

    geometry: AbstractStateGeometry
    point_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    tangent_shape: tuple[int, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        point_shape: Sequence[int],
        /,
        *,
        local_shape: Sequence[int] | None = None,
        tangent_shape: Sequence[int] | None = None,
        geometry_id: str | None = None,
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("Pointwise geometry must wrap an AbstractStateGeometry.")
        point = _role_shape(point_shape, "point_shape")
        local = point if local_shape is None else _role_shape(local_shape, "local_shape")
        tangent = (
            point
            if tangent_shape is None
            else _role_shape(tangent_shape, "tangent_shape")
        )
        self.geometry = geometry
        self.point_shape = point
        self.local_shape = local
        self.tangent_shape = tangent
        point_signature = "x".join(map(str, point))
        role_signature = (
            point_signature
            if local == point and tangent == point
            else (
                f"point={point_signature}:"
                f"local={'x'.join(map(str, local))}:"
                f"tangent={'x'.join(map(str, tangent))}"
            )
        )
        self.geometry_id = (
            f"{geometry.geometry_id}:pointwise:{role_signature}"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = f"pointwise:{geometry.retraction_method}"
        self.trivial = geometry.trivial
        self.supports_exact_inverse = geometry.supports_exact_inverse
        self.supports_exact_differential = geometry.supports_exact_differential
        self.supports_transport = geometry.supports_transport
        self.supports_isometric_transport = geometry.supports_isometric_transport
        self.supports_commutator_free = geometry.supports_commutator_free

    def _validate(
        self,
        value: ArrayLike,
        shape: tuple[int, ...],
        name: str,
        /,
    ) -> Array:
        array = jnp.asarray(value)
        if array.ndim < len(shape) or array.shape[-len(shape) :] != shape:
            raise ValueError(
                f"{name} must have trailing role shape {shape}; got {array.shape}."
            )
        return array

    def _map(
        self,
        function: Callable[..., Array],
        values: tuple[Array, ...],
        shapes: tuple[tuple[int, ...], ...],
        /,
    ) -> Array:
        leading = values[0].shape[: -len(shapes[0])]
        for value, shape in zip(values[1:], shapes[1:], strict=True):
            if value.shape[: -len(shape)] != leading:
                raise ValueError("Pointwise role arrays must share leading axes.")
        if not leading:
            return jnp.asarray(function(*values))
        flattened = tuple(
            value.reshape((-1,) + shape)
            for value, shape in zip(values, shapes, strict=True)
        )
        mapped = jax.vmap(function)(*flattened)
        return jnp.asarray(mapped).reshape(leading + mapped.shape[1:])

    def _mapped(
        self,
        function: Callable[..., Array],
        values: tuple[Array, ...],
        shapes: tuple[tuple[int, ...], ...],
        result_shape: tuple[int, ...],
        name: str,
        /,
    ) -> Array:
        mapped = self._map(function, values, shapes)
        result = self._validate(mapped, result_shape, name)
        leading = values[0].shape[: -len(shapes[0])]
        if result.shape[: -len(result_shape)] != leading:
            raise ValueError(f"{name} changed pointwise leading axes.")
        return result

    def contains(self, state: ArrayLike, /) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        membership = self._map(
            self.geometry.contains,
            (state_array,),
            (self.point_shape,),
        )
        return jnp.all(membership)

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        ambient = self._validate(
            vector,
            self.point_shape,
            "Pointwise ambient vector",
        )
        return self._mapped(
            self.geometry.project_tangent,
            (state_array, ambient),
            (self.point_shape, self.point_shape),
            self.tangent_shape,
            "Pointwise tangent projection",
        )

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        local = self._validate(
            local_tangent,
            self.local_shape,
            "Pointwise local tangent",
        )
        return self._mapped(
            self.geometry.retract,
            (state_array, local),
            (self.point_shape, self.local_shape),
            self.point_shape,
            "Pointwise retraction",
        )

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        point_array = self._validate(
            point,
            self.point_shape,
            "Pointwise retraction point",
        )
        return self._mapped(
            self.geometry.inverse_retract,
            (state_array, point_array),
            (self.point_shape, self.point_shape),
            self.local_shape,
            "Pointwise inverse retraction",
        )

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        local = self._validate(
            local_tangent,
            self.local_shape,
            "Pointwise local tangent",
        )
        velocity = self._validate(
            local_velocity,
            self.local_shape,
            "Pointwise local velocity",
        )
        return self._mapped(
            self.geometry.retraction_jvp,
            (state_array, local, velocity),
            (self.point_shape, self.local_shape, self.local_shape),
            self.tangent_shape,
            "Pointwise retraction JVP",
        )

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        point_array = self._validate(
            point,
            self.point_shape,
            "Pointwise retraction point",
        )
        vector = self._validate(
            tangent,
            self.tangent_shape,
            "Pointwise tangent",
        )
        return self._mapped(
            self.geometry.retraction_inverse_jvp,
            (state_array, point_array, vector),
            (self.point_shape, self.point_shape, self.tangent_shape),
            self.local_shape,
            "Pointwise inverse-retraction JVP",
        )

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, self.point_shape, "Pointwise state")
        local = self._validate(
            local_tangent,
            self.local_shape,
            "Pointwise local tangent",
        )
        covector = self._validate(
            cotangent,
            self.tangent_shape,
            "Pointwise cotangent",
        )
        return self._mapped(
            self.geometry.retraction_vjp,
            (state_array, local, covector),
            (self.point_shape, self.local_shape, self.tangent_shape),
            self.local_shape,
            "Pointwise retraction VJP",
        )

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        source = self._validate(state, self.point_shape, "Pointwise state")
        target = self._validate(point, self.point_shape, "Pointwise point")
        vector = self._validate(
            tangent,
            self.tangent_shape,
            "Pointwise tangent",
        )
        return self._mapped(
            self.geometry.transport_tangent,
            (source, target, vector),
            (self.point_shape, self.point_shape, self.tangent_shape),
            self.tangent_shape,
            "Pointwise tangent transport",
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        source = self._validate(state, self.point_shape, "Pointwise state")
        target = self._validate(point, self.point_shape, "Pointwise point")
        covector = self._validate(
            cotangent,
            self.tangent_shape,
            "Pointwise cotangent",
        )
        return self._mapped(
            self.geometry.transport_cotangent_pullback,
            (source, target, covector),
            (self.point_shape, self.point_shape, self.tangent_shape),
            self.tangent_shape,
            "Pointwise cotangent pullback",
        )

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        source = self._validate(state, self.point_shape, "Pointwise state")
        target = self._validate(point, self.point_shape, "Pointwise point")
        margins = self._map(
            self.geometry.cut_locus_margin,
            (source, target),
            (self.point_shape, self.point_shape),
        )
        return jnp.min(margins)


MatrixRetraction: TypeAlias = Literal["exponential", "cayley"]


def _dimension(value: int, /) -> int:
    dimension = int(value)
    if dimension < 2:
        raise ValueError("Matrix state geometry dimension must be at least two.")
    return dimension


def _matrix_shape(value: ArrayLike, dimension: int, name: str, /) -> Array:
    array = jnp.asarray(value)
    expected = (dimension, dimension)
    if array.shape[-2:] != expected:
        raise ValueError(
            f"{name} must have trailing matrix shape {expected}; got {array.shape}."
        )
    return array


def _transpose(value: Array, /) -> Array:
    return jnp.swapaxes(value, -1, -2)


def _symmetric(value: Array, /) -> Array:
    return 0.5 * (value + _transpose(value))


def _skew(value: Array, /) -> Array:
    return 0.5 * (value - _transpose(value))


def _matrix_map(function: Callable[[Array], Array], value: Array, /) -> Array:
    if value.ndim == 2:
        return function(value)
    leading = value.shape[:-2]
    flat = value.reshape((-1,) + value.shape[-2:])
    mapped = jax.vmap(function)(flat)
    return mapped.reshape(leading + mapped.shape[1:])


def _matrix_exponential(value: Array, /) -> Array:
    return _matrix_map(jsp.linalg.expm, value)


def _symmetric_matrix_logarithm_primal(value: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(value))
    safe = jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny)
    return _symmetric(
        (eigenvectors * jnp.expand_dims(jnp.log(safe), axis=-2))
        @ _transpose(eigenvectors)
    )


@jax.custom_jvp
def _symmetric_matrix_logarithm(value: Array, /) -> Array:
    return _symmetric_matrix_logarithm_primal(value)


@_symmetric_matrix_logarithm.defjvp
def _symmetric_matrix_logarithm_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    symmetric_value = _symmetric(value)
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric_value)
    safe = jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny)
    left = jnp.expand_dims(safe, axis=-1)
    right = jnp.expand_dims(safe, axis=-2)
    difference = left - right
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(left), jnp.abs(right)),
        jnp.finfo(value.dtype).tiny,
    )
    close = jnp.abs(difference) <= jnp.sqrt(jnp.finfo(value.dtype).eps) * scale
    safe_difference = jnp.where(close, 1.0, difference)
    relative_difference = jnp.where(close, 0.0, difference / right)
    divided_difference = jnp.log1p(relative_difference) / safe_difference
    coefficient = jnp.where(
        close,
        2.0 / (left + right),
        divided_difference,
    )
    transformed = _transpose(eigenvectors) @ _symmetric(tangent) @ eigenvectors
    derivative = eigenvectors @ (coefficient * transformed) @ _transpose(eigenvectors)
    return _symmetric_matrix_logarithm_primal(value), _symmetric(derivative)


def _symmetric_matrix_square_root_primal(value: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(value))
    roots = jnp.sqrt(jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny))
    return _symmetric(
        (eigenvectors * jnp.expand_dims(roots, axis=-2)) @ _transpose(eigenvectors)
    )


@jax.custom_jvp
def _symmetric_matrix_square_root(value: Array, /) -> Array:
    return _symmetric_matrix_square_root_primal(value)


@_symmetric_matrix_square_root.defjvp
def _symmetric_matrix_square_root_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(value))
    roots = jnp.sqrt(jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny))
    left = jnp.expand_dims(roots, axis=-1)
    right = jnp.expand_dims(roots, axis=-2)
    coefficient = 1.0 / (left + right)
    transformed = _transpose(eigenvectors) @ _symmetric(tangent) @ eigenvectors
    derivative = eigenvectors @ (coefficient * transformed) @ _transpose(eigenvectors)
    return _symmetric_matrix_square_root_primal(value), _symmetric(derivative)


def _principal_local_so_logarithm(value: Array, /) -> Array:
    identity = jnp.eye(value.shape[-1], dtype=value.dtype)
    sum_with_identity = jax.lax.stop_gradient(value) + identity
    singular_values = jnp.linalg.svd(sum_with_identity, compute_uv=False)
    cut_locus_tolerance = 32.0 * value.shape[-1] * jnp.finfo(value.dtype).eps
    value = eqx.error_if(
        value,
        jnp.any(
            (~jnp.all(jnp.isfinite(singular_values), axis=-1))
            | (singular_values[..., -1] <= cut_locus_tolerance)
        ),
        "SO exponential inverse_retract requires a principal local rotation "
        "away from the rotation-by-pi cut locus.",
    )
    cayley = _skew(jnp.linalg.solve(value + identity, value - identity))
    for _ in range(2):
        square_root = _symmetric_matrix_square_root(
            _symmetric(identity - cayley @ cayley)
        )
        cayley = _skew(jnp.linalg.solve(identity + square_root, cayley))
    square = cayley @ cayley
    term = cayley
    series = cayley
    for denominator in range(3, 64, 2):
        term = term @ square
        series = series + term / denominator
    return _skew(8.0 * series)


class SpecialOrthogonalStateGeometry(AbstractStateGeometry):
    """Explicit equal-space matrix geometry for SO(n) points and tangents."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: MatrixRetraction = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        retraction: MatrixRetraction = "exponential",
        tolerance: float = 1e-6,
        geometry_id: str | None = None,
    ):
        n = _dimension(dimension)
        if retraction not in ("exponential", "cayley"):
            raise ValueError("SO(n) retraction must be 'exponential' or 'cayley'.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.dimension = n
        self.tolerance = float(tolerance)
        self.geometry_id = (
            f"state-geometry:so:{n}:{retraction}"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = retraction
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        orthogonality = jnp.max(
            jnp.abs(_transpose(matrix) @ matrix - identity),
            axis=(-2, -1),
        )
        determinant = jnp.linalg.det(matrix)
        finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        return jnp.all(finite & (orthogonality <= self.tolerance) & (determinant > 0.0))

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        ambient = _matrix_shape(vector, self.dimension, "SO(n) tangent")
        _same_shape(ambient, matrix, "SO(n) tangent")
        return matrix @ _skew(_transpose(matrix) @ ambient)

    def _left_trivialize_tangent(
        self,
        state: Array,
        tangent: ArrayLike,
        /,
    ) -> Array:
        projected = self.project_tangent(state, tangent)
        return _skew(_transpose(state) @ projected)

    def _increment(self, local_tangent: Array, /) -> Array:
        algebra = _skew(local_tangent)
        if self.retraction_method == "exponential":
            return _matrix_exponential(algebra)
        identity = jnp.eye(self.dimension, dtype=algebra.dtype)
        return jnp.linalg.solve(
            identity - 0.5 * algebra,
            identity + 0.5 * algebra,
        )

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SO(n) local tangent",
        )
        _same_shape(local, matrix, "SO(n) local tangent")
        return matrix @ self._increment(local)

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) retraction point")
        _same_shape(target, matrix, "SO(n) retraction point")
        relative = _transpose(matrix) @ target
        if self.retraction_method == "cayley":
            identity = jnp.eye(self.dimension, dtype=relative.dtype)
            cayley = jnp.linalg.solve(relative + identity, relative - identity)
            return 2.0 * _skew(cayley)
        return _principal_local_so_logarithm(relative)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SO(n) local tangent",
        )
        velocity = _matrix_shape(
            local_velocity,
            self.dimension,
            "SO(n) local velocity",
        )
        _same_shape(local, matrix, "SO(n) local tangent")
        _same_shape(velocity, local, "SO(n) local velocity")
        point, tangent = jax.jvp(
            lambda value: self.retract(matrix, value),
            (local,),
            (_skew(velocity),),
        )
        return self.project_tangent(point, tangent)

    def _one_retraction_inverse_jvp(
        self,
        matrix: Array,
        local: Array,
        tangent: Array,
        /,
    ) -> Array:
        point = self.retract(matrix, local)
        vector = self.project_tangent(point, tangent)
        body_velocity = self._left_trivialize_tangent(point, vector)
        if self.retraction_method == "cayley":
            identity = jnp.eye(self.dimension, dtype=matrix.dtype)
            right_factor = self._increment(local) + identity
            relative_velocity = _transpose(matrix) @ vector
            left_factor = 2.0 * (identity - 0.5 * local) @ relative_velocity
            velocity = jnp.linalg.solve(
                _transpose(right_factor),
                _transpose(left_factor),
            )
            return _skew(_transpose(velocity))

        def differential(local_velocity):
            ambient_velocity = self.retraction_jvp(
                matrix,
                local,
                _skew(local_velocity),
            )
            return self._left_trivialize_tangent(point, ambient_velocity) + _symmetric(
                local_velocity
            )

        tolerance = 1e-10 if matrix.dtype == jnp.dtype(jnp.float64) else 1e-5
        restart = 8
        krylov_cycles = max(
            4,
            2 * (self.dimension * self.dimension + restart - 1) // restart,
        )
        right_hand_side_norm = jnp.linalg.norm(body_velocity)
        scale = jnp.maximum(
            right_hand_side_norm,
            jnp.finfo(matrix.dtype).tiny,
        )
        normalized_body_velocity = body_velocity / scale
        normalized_velocity, _ = jsparse.gmres(
            differential,
            normalized_body_velocity,
            x0=normalized_body_velocity,
            tol=tolerance,
            atol=0.0,
            restart=restart,
            maxiter=krylov_cycles,
        )
        velocity = scale * normalized_velocity
        residual = differential(velocity) - body_velocity
        residual_norm = jnp.linalg.norm(residual)
        relative_residual = residual_norm / scale
        failed = jnp.where(
            right_hand_side_norm == 0.0,
            residual_norm != 0.0,
            relative_residual > 2.0 * tolerance,
        )
        velocity = eqx.error_if(
            velocity,
            failed,
            "SO exponential inverse-retraction JVP matrix-free solve did not converge.",
        )
        return _skew(velocity)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) retraction point")
        vector = _matrix_shape(tangent, self.dimension, "SO(n) tangent")
        _same_shape(target, matrix, "SO(n) retraction point")
        _same_shape(vector, matrix, "SO(n) tangent")
        local = self.inverse_retract(matrix, target)
        if matrix.ndim == 2:
            return self._one_retraction_inverse_jvp(matrix, local, vector)
        leading = matrix.shape[:-2]
        flat_shape = (-1, self.dimension, self.dimension)
        velocities = jax.vmap(self._one_retraction_inverse_jvp)(
            matrix.reshape(flat_shape),
            local.reshape(flat_shape),
            vector.reshape(flat_shape),
        )
        return velocities.reshape(leading + (self.dimension, self.dimension))

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SO(n) local tangent",
        )
        covector = _matrix_shape(cotangent, self.dimension, "SO(n) cotangent")
        _same_shape(local, matrix, "SO(n) local tangent")
        _same_shape(covector, matrix, "SO(n) cotangent")
        _, transpose = jax.vjp(lambda value: self.retract(matrix, value), local)
        return _skew(transpose(covector)[0])

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) transport point")
        vector = _matrix_shape(tangent, self.dimension, "SO(n) tangent")
        _same_shape(target, matrix, "SO(n) transport point")
        _same_shape(vector, matrix, "SO(n) tangent")
        local = self._left_trivialize_tangent(matrix, vector)
        return target @ local

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) transport point")
        covector = _matrix_shape(cotangent, self.dimension, "SO(n) cotangent")
        _same_shape(target, matrix, "SO(n) transport point")
        _same_shape(covector, matrix, "SO(n) cotangent")
        seed = jnp.zeros_like(matrix)
        _, transpose = jax.vjp(
            lambda vector: self.transport_tangent(matrix, target, vector),
            seed,
        )
        return transpose(covector)[0]

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) point")
        _same_shape(target, matrix, "SO(n) point")
        relative = _transpose(matrix) @ target
        identity = jnp.eye(self.dimension, dtype=relative.dtype)
        if self.retraction_method == "exponential":
            cayley = jnp.linalg.solve(relative + identity, relative - identity)
            radius = jnp.linalg.norm(cayley, ord=2, axis=(-2, -1))
            return jnp.min(0.5 - radius)
        singular_values = jnp.linalg.svd(
            relative + identity,
            compute_uv=False,
        )
        return jnp.min(singular_values[..., -1])


class SymmetricPositiveDefiniteStateGeometry(AbstractStateGeometry):
    """Explicit equal-space congruence/exponential geometry for SPD(n)."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        tolerance: float = 1e-8,
        geometry_id: str | None = None,
    ):
        n = _dimension(dimension)
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.dimension = n
        self.tolerance = float(tolerance)
        self.geometry_id = (
            f"state-geometry:spd:{n}:congruence-exponential"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = "congruence-exponential"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = False
        self.supports_commutator_free = False

    def _congruence_factor(self, state: Array, /) -> Array:
        return jnp.linalg.cholesky(_symmetric(state))

    def _inverse_congruence(
        self,
        factor: Array,
        value: Array,
        /,
    ) -> Array:
        left_solved = jnp.linalg.solve(factor, value)
        return _transpose(jnp.linalg.solve(factor, _transpose(left_solved)))

    def contains(self, state: ArrayLike, /) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        symmetry_error = jnp.max(
            jnp.abs(matrix - _transpose(matrix)),
            axis=(-2, -1),
        )
        minimum = jnp.min(jnp.linalg.eigvalsh(_symmetric(matrix)), axis=-1)
        finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        return jnp.all(
            finite & (symmetry_error <= self.tolerance) & (minimum > self.tolerance)
        )

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        ambient = _matrix_shape(vector, self.dimension, "SPD(n) tangent")
        _same_shape(ambient, matrix, "SPD(n) tangent")
        return _symmetric(ambient)

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        _same_shape(local, matrix, "SPD(n) local tangent")
        factor = self._congruence_factor(matrix)
        return _symmetric(
            factor @ _matrix_exponential(_symmetric(local)) @ _transpose(factor)
        )

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) retraction point")
        _same_shape(target, matrix, "SPD(n) retraction point")
        factor = self._congruence_factor(matrix)
        relative = _symmetric(self._inverse_congruence(factor, target))
        return _symmetric_matrix_logarithm(relative)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        velocity = _matrix_shape(
            local_velocity,
            self.dimension,
            "SPD(n) local velocity",
        )
        _same_shape(local, matrix, "SPD(n) local tangent")
        _same_shape(velocity, local, "SPD(n) local velocity")
        _, tangent = jax.jvp(
            lambda value: self.retract(matrix, value),
            (local,),
            (_symmetric(velocity),),
        )
        return _symmetric(tangent)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) retraction point")
        vector = _matrix_shape(tangent, self.dimension, "SPD(n) tangent")
        _same_shape(target, matrix, "SPD(n) retraction point")
        _same_shape(vector, matrix, "SPD(n) tangent")
        vector = self.project_tangent(target, vector)
        _, velocity = jax.jvp(
            lambda value: self.inverse_retract(matrix, value),
            (target,),
            (vector,),
        )
        return _symmetric(velocity)

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        covector = _matrix_shape(cotangent, self.dimension, "SPD(n) cotangent")
        _same_shape(local, matrix, "SPD(n) local tangent")
        _same_shape(covector, matrix, "SPD(n) cotangent")
        _, transpose = jax.vjp(lambda value: self.retract(matrix, value), local)
        return _symmetric(transpose(covector)[0])

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        source = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) transport point")
        vector = _matrix_shape(tangent, self.dimension, "SPD(n) tangent")
        _same_shape(target, source, "SPD(n) transport point")
        _same_shape(vector, source, "SPD(n) tangent")
        source_factor = self._congruence_factor(source)
        target_factor = self._congruence_factor(target)
        coordinates = _symmetric(
            self._inverse_congruence(
                source_factor,
                self.project_tangent(source, vector),
            )
        )
        return _symmetric(target_factor @ coordinates @ _transpose(target_factor))

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        source = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) transport point")
        covector = _matrix_shape(cotangent, self.dimension, "SPD(n) cotangent")
        _same_shape(target, source, "SPD(n) transport point")
        _same_shape(covector, source, "SPD(n) cotangent")
        seed = jnp.zeros_like(source)
        _, transpose = jax.vjp(
            lambda vector: self.transport_tangent(source, target, vector),
            seed,
        )
        return _symmetric(transpose(covector)[0])

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        source = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) point")
        _same_shape(target, source, "SPD(n) point")
        factor = self._congruence_factor(source)
        relative = _symmetric(self._inverse_congruence(factor, target))
        return jnp.min(jnp.linalg.eigvalsh(relative))


__all__ = [
    "AbstractStateGeometry",
    "EmbeddedStateGeometry",
    "EuclideanStateGeometry",
    "LocalRetraction",
    "PointwiseStateGeometry",
    "SpecialOrthogonalStateGeometry",
    "StateChartEvidence",
    "StateTransportEvidence",
    "SymmetricPositiveDefiniteStateGeometry",
]
