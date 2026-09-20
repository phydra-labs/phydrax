#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Concrete regulated two-dimensional twisted N=(2,2) SYM declarations."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...metrix import AbstractStateGeometry, EuclideanStateGeometry
from ._actions import (
    prepare_twisted_sym,
    PreparedTwistedSYMAction,
    TwistedSYMConfiguration,
    TwistedSYMPlan,
)


class TwistedN2SYMPlan(StrictModule):
    """Regulated finite 2D N=(2,2) theory and real RHMC coordinate bounds."""

    bosonic_plan: TwistedSYMPlan = eqx.field(static=True)
    temporal_axis: int = eqx.field(static=True)
    fermion_boundary_phase: complex = eqx.field(static=True)
    fermion_mass: float = eqx.field(static=True)
    scalar_mass: float = eqx.field(static=True)
    u1_mass: float = eqx.field(static=True)
    coordinate_bound: float = eqx.field(static=True)
    derivative_norm_bound: float = eqx.field(static=True)
    normal_spectral_lower: float = eqx.field(static=True)
    normal_spectral_upper: float = eqx.field(static=True)
    maximum_fermion_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_shape: Sequence[int],
        /,
        *,
        matrix_rank: int,
        coupling: float,
        fermion_mass: float,
        coordinate_bound: float,
        temporal_axis: int = 0,
        fermion_boundary_phase: complex = -1.0,
        scalar_mass: float = 0.0,
        u1_mass: float = 0.0,
        lattice_spacing: float = 1.0,
        maximum_field_elements: int = 10_000_000,
        maximum_fermion_elements: int = 10_000_000,
    ):
        shape = tuple(lattice_shape)
        if len(shape) != 2:
            raise ValueError("TwistedN2SYMPlan requires exactly two lattice axes.")
        rank = int(matrix_rank)
        temporal = int(temporal_axis)
        boundary = complex(fermion_boundary_phase)
        mass = float(fermion_mass)
        scalar = float(scalar_mass)
        u1 = float(u1_mass)
        bound = float(coordinate_bound)
        maximum_fermions = int(maximum_fermion_elements)
        if temporal not in (0, 1):
            raise ValueError("temporal_axis must select one of the two lattice axes.")
        if not np.isfinite(boundary) or not np.isclose(abs(boundary), 1.0):
            raise ValueError("fermion_boundary_phase must be finite and unit magnitude.")
        if not all(np.isfinite(value) for value in (mass, scalar, u1, bound)):
            raise ValueError(
                "Twisted N=2 regulator and coordinate values must be finite."
            )
        if mass <= 0.0 or scalar < 0.0 or u1 < 0.0 or bound <= 0.0:
            raise ValueError(
                "Fermion/bound regulators must be positive and bosonic masses nonnegative."
            )
        bosonic = TwistedSYMPlan(
            shape,
            matrix_rank=rank,
            coupling=coupling,
            lattice_spacing=lattice_spacing,
            maximum_field_elements=maximum_field_elements,
        )
        required = prod(shape) * 4 * rank * rank
        if maximum_fermions < 1 or required > maximum_fermions:
            raise ValueError(
                f"Twisted Kähler–Dirac fields require {required} complex elements; capacity is {maximum_fermions}."
            )
        link_frobenius_bound = sqrt(2.0) * rank * bound
        derivative_bound = 16.0 * link_frobenius_bound / bosonic.lattice_spacing
        lower = mass**2
        upper = derivative_bound**2 + mass**2
        self.bosonic_plan = bosonic
        self.temporal_axis = temporal
        self.fermion_boundary_phase = boundary
        self.fermion_mass = mass
        self.scalar_mass = scalar
        self.u1_mass = u1
        self.coordinate_bound = bound
        self.derivative_norm_bound = derivative_bound
        self.normal_spectral_lower = lower
        self.normal_spectral_upper = upper
        self.maximum_fermion_elements = maximum_fermions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "regulated-two-dimensional-twisted-n2-sym-plan",
                "bosonic_plan": bosonic.plan_id,
                "temporal_axis": temporal,
                "fermion_boundary_phase": (boundary.real, boundary.imag),
                "fermion_mass": mass,
                "scalar_mass": scalar,
                "u1_mass": u1,
                "coordinate_bound": bound,
                "derivative_norm_bound": derivative_bound,
                "normal_spectral_interval": (lower, upper),
                "maximum_fermion_elements": maximum_fermions,
            }
        )

    def prepare_bosonic(self) -> PreparedTwistedSYMAction:
        return prepare_twisted_sym(self.bosonic_plan)


class TwistedSYMCoordinateLayout(StrictModule):
    """Invertible real storage for independent forward/reverse complex links."""

    prepared: PreparedTwistedSYMAction
    coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, prepared: PreparedTwistedSYMAction, /):
        if not isinstance(prepared, PreparedTwistedSYMAction):
            raise TypeError("prepared must be PreparedTwistedSYMAction.")
        shape = prepared.plan.lattice_shape + (
            2,
            len(prepared.plan.lattice_shape),
            prepared.plan.matrix_rank,
            prepared.plan.matrix_rank,
            2,
        )
        self.prepared = prepared
        self.coordinate_shape = shape
        self.layout_id = canonical_fingerprint(
            {
                "kind": "twisted-sym-real-coordinate-layout",
                "prepared": prepared.prepared_id,
                "coordinate_shape": shape,
                "branch_order": ("forward", "reverse"),
                "complex_order": ("real", "imaginary"),
            }
        )

    def unpack(self, coordinates: ArrayLike, /) -> TwistedSYMConfiguration:
        values = jnp.asarray(coordinates)
        if values.shape != self.coordinate_shape:
            raise ValueError(
                f"Twisted-SYM coordinates must have shape {self.coordinate_shape}."
            )
        if jnp.iscomplexobj(values) or not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError("Twisted-SYM RHMC coordinates must be real inexact arrays.")
        complex_values = values[..., 0] + 1.0j * values[..., 1]
        branch_axis = len(self.prepared.plan.lattice_shape)
        forward = jnp.take(complex_values, 0, axis=branch_axis)
        reverse = jnp.take(complex_values, 1, axis=branch_axis)
        return self.prepared.configuration(forward, reverse)

    def pack(self, configuration: TwistedSYMConfiguration, /) -> Array:
        if not isinstance(configuration, TwistedSYMConfiguration):
            raise TypeError("configuration must be TwistedSYMConfiguration.")
        self.prepared._validate(configuration)
        branch_axis = len(self.prepared.plan.lattice_shape)
        complex_values = jnp.stack(
            (configuration.links.values, configuration.reverse_links.values),
            axis=branch_axis,
        )
        coordinates = jnp.stack(
            (jnp.real(complex_values), jnp.imag(complex_values)), axis=-1
        )
        if coordinates.shape != self.coordinate_shape:
            raise RuntimeError("Packed Twisted-SYM coordinates violate their layout.")
        return coordinates

    def evidence(self, coordinates: ArrayLike, /) -> TwistedSYMCoordinateEvidence:
        values = jnp.asarray(coordinates)
        configuration = self.unpack(values)
        recovered = self.pack(configuration)
        residual = jnp.linalg.norm(recovered - values) / jnp.maximum(
            1.0, jnp.linalg.norm(values)
        )
        return TwistedSYMCoordinateEvidence(
            roundtrip_residual=residual,
            finite=jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(recovered)),
            layout_id=self.layout_id,
            coordinate_fingerprint=array_tree_fingerprint(np.asarray(values)),
            claim="real-coordinate-layout-preserves-independent-complexified-links",
        )


class TwistedSYMCoordinateEvidence(StrictModule):
    roundtrip_residual: Array
    finite: Array
    layout_id: str = eqx.field(static=True)
    coordinate_fingerprint: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class BoundedEuclideanStateGeometry(AbstractStateGeometry):
    """Euclidean addition with a hard finite coordinate-domain membership gate."""

    euclidean: EuclideanStateGeometry
    coordinate_bound: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, coordinate_bound: float, /, *, geometry_id: str):
        bound = float(coordinate_bound)
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("coordinate_bound must be positive and finite.")
        euclidean = EuclideanStateGeometry(geometry_id=geometry_id)
        self.euclidean = euclidean
        self.coordinate_bound = bound
        self.geometry_id = euclidean.geometry_id
        self.retraction_method = euclidean.retraction_method
        self.trivial = euclidean.trivial
        self.supports_exact_inverse = euclidean.supports_exact_inverse
        self.supports_exact_differential = euclidean.supports_exact_differential
        self.supports_transport = euclidean.supports_transport
        self.supports_isometric_transport = euclidean.supports_isometric_transport
        self.supports_commutator_free = euclidean.supports_commutator_free

    def contains(self, state: ArrayLike, /) -> Array:
        values = jnp.asarray(state)
        return jnp.all(jnp.isfinite(values)) & (
            jnp.max(jnp.abs(values)) <= self.coordinate_bound
        )

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        return self.euclidean.project_tangent(state, vector)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.euclidean.retract(state, local_tangent)

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        return self.euclidean.inverse_retract(state, point)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        return self.euclidean.retraction_jvp(state, local_tangent, local_velocity)

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        return self.euclidean.retraction_inverse_jvp(state, point, tangent)

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.euclidean.retraction_vjp(state, local_tangent, cotangent)

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        return self.euclidean.transport_tangent(state, point, tangent)

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.euclidean.transport_cotangent_pullback(state, point, cotangent)

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        return self.euclidean.cut_locus_margin(state, point)


__all__ = [
    "BoundedEuclideanStateGeometry",
    "TwistedN2SYMPlan",
    "TwistedSYMCoordinateEvidence",
    "TwistedSYMCoordinateLayout",
]
