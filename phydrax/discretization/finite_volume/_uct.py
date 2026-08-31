#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class UCTElectromotiveResult(StrictModule):
    components: tuple[Array, ...]
    one_dimensional_consistency_defect: Array
    maximum_dissipation: Array


class AbstractUCTElectromotivePlan(StrictModule, NonTrainableState):
    electromotive_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def electromotive(
        self,
        full_state: Array,
        face_fluxes: tuple[Array, ...],
        signal_speeds: tuple[Array, ...],
        dimension: int,
        /,
    ) -> UCTElectromotiveResult:
        raise NotImplementedError


class FluxCTElectromotivePlan(AbstractUCTElectromotivePlan):
    """Arithmetic flux-CT reference used as the monotone fallback."""

    def __init__(self):
        self.electromotive_id = canonical_fingerprint({"kind": "flux-ct-emf"})

    def electromotive(
        self,
        full_state: Array,
        face_fluxes: tuple[Array, ...],
        signal_speeds: tuple[Array, ...],
        dimension: int,
        /,
    ) -> UCTElectromotiveResult:
        del full_state, signal_speeds
        dimension_ = int(dimension)
        if dimension_ == 1:
            components = ()
        elif dimension_ == 2:
            flux_x, flux_y = face_fluxes
            components = (
                0.25
                * (
                    -flux_x[..., 6]
                    - jnp.roll(flux_x[..., 6], -1, axis=1)
                    + flux_y[..., 5]
                    + jnp.roll(flux_y[..., 5], -1, axis=0)
                ),
            )
        elif dimension_ == 3:
            flux_x, flux_y, flux_z = face_fluxes
            components = (
                0.25
                * (
                    -flux_y[..., 7]
                    - jnp.roll(flux_y[..., 7], -1, axis=2)
                    + flux_z[..., 6]
                    + jnp.roll(flux_z[..., 6], -1, axis=1)
                ),
                0.25
                * (
                    flux_x[..., 7]
                    + jnp.roll(flux_x[..., 7], -1, axis=2)
                    - flux_z[..., 5]
                    - jnp.roll(flux_z[..., 5], -1, axis=0)
                ),
                0.25
                * (
                    -flux_x[..., 6]
                    - jnp.roll(flux_x[..., 6], -1, axis=1)
                    + flux_y[..., 5]
                    + jnp.roll(flux_y[..., 5], -1, axis=0)
                ),
            )
        else:
            raise ValueError("UCT dimension must be one, two, or three.")
        dtype = face_fluxes[0].dtype
        return UCTElectromotiveResult(
            components=components,
            one_dimensional_consistency_defect=jnp.asarray(0.0, dtype=dtype),
            maximum_dissipation=jnp.asarray(0.0, dtype=dtype),
        )


class HLLUCTElectromotivePlan(AbstractUCTElectromotivePlan):
    """Local-Lax-Friedrichs HLL-UCT edge electromotive construction."""

    reference: FluxCTElectromotivePlan
    dissipation_scale: float = eqx.field(static=True)

    def __init__(self, *, dissipation_scale: float = 1.0):
        scale = float(dissipation_scale)
        if not 0.0 <= scale <= 2.0:
            raise ValueError("HLL-UCT dissipation scale must be between zero and two.")
        self.reference = FluxCTElectromotivePlan()
        self.dissipation_scale = scale
        self.electromotive_id = canonical_fingerprint(
            {"kind": "hll-uct-emf", "dissipation_scale": scale}
        )

    @staticmethod
    def _edge_speed(speed: Array, transverse_axis: int, /) -> Array:
        return jnp.maximum(speed, jnp.roll(speed, -1, axis=transverse_axis))

    def electromotive(
        self,
        full_state: Array,
        face_fluxes: tuple[Array, ...],
        signal_speeds: tuple[Array, ...],
        dimension: int,
        /,
    ) -> UCTElectromotiveResult:
        dimension_ = int(dimension)
        reference = self.reference.electromotive(
            full_state, face_fluxes, signal_speeds, dimension_
        )
        if dimension_ == 1:
            return reference
        bx, by, bz = (full_state[..., 5 + axis] for axis in range(3))
        scale = jnp.asarray(self.dissipation_scale, dtype=full_state.dtype)
        if dimension_ == 2:
            alpha_x = self._edge_speed(signal_speeds[0], 1)
            alpha_y = self._edge_speed(signal_speeds[1], 0)
            jump_by = jnp.roll(by, -1, axis=0) - by
            jump_bx = jnp.roll(bx, -1, axis=1) - bx
            ez = reference.components[0] + 0.25 * scale * (
                alpha_x * jump_by - alpha_y * jump_bx
            )
            components = (ez,)
            maximum = jnp.maximum(jnp.max(alpha_x), jnp.max(alpha_y))
        elif dimension_ == 3:
            alpha_x_y = self._edge_speed(signal_speeds[0], 1)
            alpha_x_z = self._edge_speed(signal_speeds[0], 2)
            alpha_y_x = self._edge_speed(signal_speeds[1], 0)
            alpha_y_z = self._edge_speed(signal_speeds[1], 2)
            alpha_z_x = self._edge_speed(signal_speeds[2], 0)
            alpha_z_y = self._edge_speed(signal_speeds[2], 1)
            ex = reference.components[0] + 0.25 * scale * (
                alpha_y_z * (jnp.roll(bz, -1, axis=1) - bz)
                - alpha_z_y * (jnp.roll(by, -1, axis=2) - by)
            )
            ey = reference.components[1] + 0.25 * scale * (
                alpha_z_x * (jnp.roll(bx, -1, axis=2) - bx)
                - alpha_x_z * (jnp.roll(bz, -1, axis=0) - bz)
            )
            ez = reference.components[2] + 0.25 * scale * (
                alpha_x_y * (jnp.roll(by, -1, axis=0) - by)
                - alpha_y_x * (jnp.roll(bx, -1, axis=1) - bx)
            )
            components = ex, ey, ez
            maximum = jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(value)
                        for value in (
                            alpha_x_y,
                            alpha_x_z,
                            alpha_y_x,
                            alpha_y_z,
                            alpha_z_x,
                            alpha_z_y,
                        )
                    )
                )
            )
        else:
            raise ValueError("UCT dimension must be one, two, or three.")
        defect = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(value - baseline))
                    for value, baseline in zip(
                        components, reference.components, strict=True
                    )
                )
            ),
            initial=0.0,
        )
        return UCTElectromotiveResult(
            components=components,
            one_dimensional_consistency_defect=defect,
            maximum_dissipation=maximum,
        )


__all__ = [
    "AbstractUCTElectromotivePlan",
    "FluxCTElectromotivePlan",
    "HLLUCTElectromotivePlan",
    "UCTElectromotiveResult",
]
