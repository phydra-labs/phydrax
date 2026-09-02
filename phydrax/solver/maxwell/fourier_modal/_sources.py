#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ....discretization.spectral import (
    LatticeHarmonicDiscretization,
    PreparedBrillouinZone,
)
from ._boundary_cascade import BoundaryRelation, compose_boundary_relations
from ._factorization import _dense_solve
from ._layer import PreparedLayerOperator
from ._scattering import HomogeneousPortModes, MaxwellPortScatteringOperator


class FourierModalExcitation(StrictModule):
    """Port inputs and named surface-current channels with trailing RHS axes."""

    left_incident: Array
    right_incident: Array
    electric_currents: tuple[Array, ...]
    magnetic_currents: tuple[Array, ...]
    channel_weights: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        left_incident: ArrayLike,
        right_incident: ArrayLike,
        /,
        *,
        source_ids: Sequence[str] = (),
        electric_currents: Sequence[ArrayLike] = (),
        magnetic_currents: Sequence[ArrayLike] = (),
        channel_weights: ArrayLike | None = None,
    ):
        left = jnp.asarray(left_incident)
        right = jnp.asarray(right_incident, dtype=left.dtype)
        if left.ndim < 2 or right.shape != left.shape:
            raise ValueError(
                "Port amplitudes must have equal shape (port_size, rhs_count)."
            )
        identifiers = tuple(str(value) for value in source_ids)
        electric = tuple(
            jnp.asarray(value, dtype=left.dtype) for value in electric_currents
        )
        magnetic = tuple(
            jnp.asarray(value, dtype=left.dtype) for value in magnetic_currents
        )
        if len(identifiers) != len(electric) or len(identifiers) != len(magnetic):
            raise ValueError(
                "Each source ID requires electric and magnetic current arrays."
            )
        harmonic_count = left.shape[0] // 2
        required_shape = (3, harmonic_count, left.shape[1])
        if any(value.shape != required_shape for value in electric + magnetic):
            raise ValueError(f"Surface currents must have shape {required_shape}.")
        weights = (
            jnp.ones((left.shape[1],), dtype=left.real.dtype)
            if channel_weights is None
            else jnp.asarray(channel_weights, dtype=left.real.dtype)
        )
        if weights.shape != (left.shape[1],):
            raise ValueError("channel_weights must contain one value per RHS channel.")
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
            "channel_weights must be finite and non-negative.",
        )
        self.left_incident = left
        self.right_incident = right
        self.source_ids = identifiers
        self.electric_currents = electric
        self.magnetic_currents = magnetic
        self.channel_weights = weights

    @property
    def rhs_count(self) -> int:
        return int(self.left_incident.shape[1])


def plane_wave_excitation(
    scattering: MaxwellPortScatteringOperator,
    harmonic_mode_id: str,
    polarization: str,
    /,
    *,
    side: str = "left",
    amplitude: ArrayLike = 1.0,
) -> FourierModalExcitation:
    modes = scattering.left_modes if side == "left" else scattering.right_modes
    if not isinstance(modes, HomogeneousPortModes):
        raise TypeError("plane_wave_excitation is homogeneous-port only.")
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'.")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    mode_ids = (
        scattering.left_modes.mode_ids
        if side == "left"
        else scattering.right_modes.mode_ids
    )
    identifier = f"{harmonic_mode_id}:{polarization}"
    if identifier not in mode_ids:
        raise KeyError(f"Unknown port mode {identifier!r}.")
    index = mode_ids.index(identifier)
    dtype = scattering.s11.matrix.dtype
    left = jnp.zeros((scattering.block_size, 1), dtype=dtype)
    right = jnp.zeros_like(left)
    if side == "left":
        left = left.at[index, 0].set(jnp.asarray(amplitude, dtype=dtype))
    else:
        right = right.at[index, 0].set(jnp.asarray(amplitude, dtype=dtype))
    return FourierModalExcitation(left, right)


def port_mode_excitation(
    scattering: MaxwellPortScatteringOperator,
    mode_id: str,
    /,
    *,
    side: str = "left",
    amplitude: ArrayLike = 1.0,
) -> FourierModalExcitation:
    """Excite one stable incoming mode of a homogeneous or periodic port."""

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    modes = scattering.left_modes if side == "left" else scattering.right_modes
    identifier = str(mode_id)
    if identifier not in modes.mode_ids:
        raise KeyError(f"Unknown port mode {identifier!r}.")
    index = modes.mode_ids.index(identifier)
    dtype = scattering.s11.matrix.dtype
    left = jnp.zeros((scattering.block_size, 1), dtype=dtype)
    right = jnp.zeros_like(left)
    if side == "left":
        left = left.at[index, 0].set(jnp.asarray(amplitude, dtype=dtype))
    else:
        right = right.at[index, 0].set(jnp.asarray(amplitude, dtype=dtype))
    return FourierModalExcitation(left, right)


def point_source_coefficients(
    lattice: LatticeHarmonicDiscretization,
    bloch_wavevector: ArrayLike,
    position: ArrayLike,
    /,
) -> Array:
    point = jnp.asarray(position, dtype=lattice.primitive_vectors.dtype)
    if point.shape != (2,):
        raise ValueError("position must have shape (2,).")
    wavevectors = lattice.in_plane_wavevectors(bloch_wavevector)
    return jnp.exp(-1j * contract("hd,d->h", wavevectors, point)) / lattice.cell_measure


def gaussian_source_coefficients(
    lattice: LatticeHarmonicDiscretization,
    bloch_wavevector: ArrayLike,
    position: ArrayLike,
    width: ArrayLike,
    /,
) -> Array:
    widths = jnp.asarray(width, dtype=lattice.primitive_vectors.dtype)
    if widths.ndim == 0:
        widths = jnp.broadcast_to(widths, (2,))
    if widths.shape != (2,):
        raise ValueError("width must be scalar or have shape (2,).")
    widths = eqx.error_if(
        widths,
        jnp.any(widths <= 0.0),
        "Gaussian source widths must be positive.",
    )
    point = point_source_coefficients(lattice, bloch_wavevector, position)
    wavevectors = lattice.in_plane_wavevectors(bloch_wavevector)
    envelope = jnp.exp(
        -0.5
        * ((wavevectors[:, 0] * widths[0]) ** 2 + (wavevectors[:, 1] * widths[1]) ** 2)
    )
    return point * envelope


class AffineBoundaryRelation(StrictModule):
    relation: BoundaryRelation
    electric_source: Array
    magnetic_source: Array


def homogeneous_affine_relation(
    relation: BoundaryRelation,
    rhs_count: int,
    /,
) -> AffineBoundaryRelation:
    zero = jnp.zeros(
        (relation.tangential_size, int(rhs_count)),
        dtype=relation.a.dtype,
    )
    return AffineBoundaryRelation(relation, zero, zero)


def source_plane_affine_relation(
    layer: PreparedLayerOperator,
    electric_current: Array,
    magnetic_current: Array,
    /,
) -> AffineBoundaryRelation:
    """Build the affine tangential-field jump for electric and magnetic sheets."""
    count = layer.harmonic_count
    electric = jnp.asarray(electric_current, dtype=layer.matrix.dtype)
    magnetic = jnp.asarray(magnetic_current, dtype=layer.matrix.dtype)
    if (
        electric.ndim != 3
        or electric.shape[:2] != (3, count)
        or magnetic.shape != electric.shape
    ):
        raise ValueError(
            "Surface currents must have shape (3, harmonic_count, rhs_count)."
        )
    jx, jy, jz = electric[0], electric[1], electric[2]
    mx, my, mz = magnetic[0], magnetic[1], magnetic[2]
    epsilon = layer.permittivity
    mu = layer.permeability
    epsilon_z_current = _dense_solve(epsilon[2, 2], jz)
    mu_z_current = _dense_solve(mu[2, 2], mz)
    delta_hx = jy - epsilon[1, 2] @ epsilon_z_current
    delta_hy = -jx + epsilon[0, 2] @ epsilon_z_current
    delta_ex = -my - mu[1, 2] @ mu_z_current
    delta_ey = mx + mu[0, 2] @ mu_z_current
    electric_jump = jnp.concatenate((delta_ex, delta_ey), axis=0)
    magnetic_jump = jnp.concatenate((-delta_hx, -delta_hy), axis=0)
    size = 2 * count
    identity = jnp.eye(size, dtype=layer.matrix.dtype)
    zero = jnp.zeros_like(identity)
    from ._boundary_cascade import BoundaryRelationDiagnostics

    relation = BoundaryRelation(
        identity,
        zero,
        zero,
        identity,
        BoundaryRelationDiagnostics(
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(True),
            jnp.asarray(True),
        ),
    )
    return AffineBoundaryRelation(relation, electric_jump, magnetic_jump)


def compose_affine_boundary_relations(
    left: AffineBoundaryRelation,
    right: AffineBoundaryRelation,
    /,
) -> AffineBoundaryRelation:
    relation = compose_boundary_relations(left.relation, right.relation)
    size = left.relation.tangential_size
    identity = jnp.eye(size, dtype=left.relation.a.dtype)
    system = identity - left.relation.b @ right.relation.c
    source_rhs = left.electric_source + left.relation.b @ right.magnetic_source
    middle_source = _dense_solve(system, source_rhs)
    electric_source = right.relation.a @ middle_source + right.electric_source
    magnetic_source = left.magnetic_source + left.relation.d @ (
        right.relation.c @ middle_source + right.magnetic_source
    )
    return AffineBoundaryRelation(relation, electric_source, magnetic_source)


def emitted_port_amplitudes(
    affine: AffineBoundaryRelation,
    left_modes: HomogeneousPortModes,
    right_modes: HomogeneousPortModes,
    /,
) -> tuple[Array, Array]:
    """Solve source-only outgoing amplitudes with no incident port field."""
    relation = affine.relation
    wl = left_modes.electric_matrix
    vl = left_modes.magnetic_matrix
    wr = right_modes.electric_matrix
    vr = right_modes.magnetic_matrix
    system = jnp.block(
        [
            [wr - relation.b @ vr, -relation.a @ wl],
            [-relation.d @ vr, -vl - relation.c @ wl],
        ]
    )
    right_hand_side = jnp.concatenate(
        (affine.electric_source, affine.magnetic_source), axis=0
    )
    outgoing = _dense_solve(system, right_hand_side)
    size = relation.tangential_size
    return outgoing[:size], outgoing[size:]


def integrate_brillouin_fields(
    fields: ArrayLike,
    rule: PreparedBrillouinZone,
    /,
) -> Array:
    values = jnp.asarray(fields)
    if values.shape[: len(rule.plan.grid_shape)] != rule.plan.grid_shape:
        raise ValueError("fields must begin with the Brillouin grid shape.")
    weights = rule.weights.reshape(
        rule.plan.grid_shape + (1,) * (values.ndim - len(rule.plan.grid_shape))
    )
    return jnp.sum(values * weights, axis=tuple(range(len(rule.plan.grid_shape))))


def integrate_brillouin_power(
    power: ArrayLike,
    rule: PreparedBrillouinZone,
    /,
) -> Array:
    values = jnp.asarray(power)
    if values.shape[: len(rule.plan.grid_shape)] != rule.plan.grid_shape:
        raise ValueError("power must begin with the Brillouin grid shape.")
    weights = rule.weights.reshape(
        rule.plan.grid_shape + (1,) * (values.ndim - len(rule.plan.grid_shape))
    )
    return jnp.sum(values * weights, axis=tuple(range(len(rule.plan.grid_shape))))


__all__ = [
    "AffineBoundaryRelation",
    "FourierModalExcitation",
    "compose_affine_boundary_relations",
    "emitted_port_amplitudes",
    "gaussian_source_coefficients",
    "homogeneous_affine_relation",
    "integrate_brillouin_fields",
    "integrate_brillouin_power",
    "plane_wave_excitation",
    "port_mode_excitation",
    "point_source_coefficients",
    "source_plane_affine_relation",
]
