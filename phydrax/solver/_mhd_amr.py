#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class MagneticAMRTransferDiagnostics(StrictModule):
    divergence_before: Array
    divergence_after: Array
    divergence_change: Array
    flux_defect: Array


class DivergenceFreeMagneticTransferPlan(StrictModule, NonTrainableState):
    """Factor-two Cartesian magnetic-flux restriction and prolongation."""

    coarse: StructuredCochainBridge
    fine: StructuredCochainBridge
    refinement_ratio: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: StructuredCochainBridge,
        fine: StructuredCochainBridge,
        /,
        *,
        refinement_ratio: int = 2,
    ):
        ratio = int(refinement_ratio)
        if (
            coarse.dimension != fine.dimension
            or coarse.dimension not in (2, 3)
            or ratio <= 1
            or any(
                fine_count != ratio * coarse_count
                for fine_count, coarse_count in zip(
                    fine.grid.shape, coarse.grid.shape, strict=True
                )
            )
        ):
            raise ValueError("Magnetic AMR transfer requires uniformly nested grids.")
        self.coarse = coarse
        self.fine = fine
        self.refinement_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "magnetic-amr-transfer",
                "coarse": coarse.bridge_id,
                "fine": fine.bridge_id,
                "refinement_ratio": ratio,
            }
        )

    def _restrict_component(self, value: Array, axis: int, /) -> Array:
        ratio = self.refinement_ratio
        result = value
        for current_axis in reversed(range(value.ndim)):
            coarse_count = self.coarse.grid.shape[current_axis]
            shape = (
                result.shape[:current_axis]
                + (coarse_count, ratio)
                + result.shape[current_axis + 1 :]
            )
            result = result.reshape(shape)
            reduction_axis = current_axis + 1
            result = (
                result.take(0, axis=reduction_axis)
                if current_axis == axis
                else jnp.mean(result, axis=reduction_axis)
            )
        return result

    def restrict(
        self, fine_flux: ArrayLike, /
    ) -> tuple[Array, MagneticAMRTransferDiagnostics]:
        fine = jnp.asarray(fine_flux)
        components = self.fine.unpack_normal_flux(fine)
        restricted_components = tuple(
            self._restrict_component(component, axis)
            for axis, component in enumerate(components)
        )
        coarse_flux = self.coarse.pack_normal_flux(restricted_components)
        before = self.fine.exterior_derivative(self.fine.dimension - 1, fine)
        after = self.coarse.exterior_derivative(self.coarse.dimension - 1, coarse_flux)
        diagnostics = MagneticAMRTransferDiagnostics(
            divergence_before=before,
            divergence_after=after,
            divergence_change=jnp.max(jnp.abs(after), initial=0.0)
            - jnp.max(jnp.abs(before), initial=0.0),
            flux_defect=jnp.asarray(0.0, dtype=fine.dtype),
        )
        return coarse_flux, diagnostics

    def prolong(
        self, coarse_flux: ArrayLike, /
    ) -> tuple[Array, MagneticAMRTransferDiagnostics]:
        coarse = jnp.asarray(coarse_flux)
        components = self.coarse.unpack_normal_flux(coarse)
        ratio = self.refinement_ratio
        prolonged = tuple(
            jnp.repeat(
                jnp.repeat(
                    component,
                    ratio,
                    axis=0,
                ),
                ratio,
                axis=1,
            )
            if component.ndim == 2
            else jnp.repeat(
                jnp.repeat(jnp.repeat(component, ratio, axis=0), ratio, axis=1),
                ratio,
                axis=2,
            )
            for component in components
        )
        fine_flux = self.fine.pack_normal_flux(prolonged)
        before = self.coarse.exterior_derivative(self.coarse.dimension - 1, coarse)
        after = self.fine.exterior_derivative(self.fine.dimension - 1, fine_flux)
        diagnostics = MagneticAMRTransferDiagnostics(
            divergence_before=before,
            divergence_after=after,
            divergence_change=jnp.max(jnp.abs(after), initial=0.0)
            - jnp.max(jnp.abs(before), initial=0.0),
            flux_defect=jnp.asarray(0.0, dtype=coarse.dtype),
        )
        return fine_flux, diagnostics


class ElectromotiveForceRegister(StrictModule):
    coarse_integral: Array
    fine_integral_restricted: Array
    mismatch: Array
    register_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_integral: ArrayLike,
        fine_integral_restricted: ArrayLike,
        /,
        *,
        register_id: str,
    ):
        coarse = jnp.asarray(coarse_integral)
        fine = jnp.asarray(fine_integral_restricted, dtype=coarse.dtype)
        if coarse.shape != fine.shape or not register_id:
            raise ValueError("EMF register arrays or identity are invalid.")
        self.coarse_integral = coarse
        self.fine_integral_restricted = fine
        self.mismatch = fine - coarse
        self.register_id = register_id


class ConstrainedMHDAMRSynchronizationPlan(StrictModule, NonTrainableState):
    coarse_bridge: StructuredCochainBridge
    plan_id: str = eqx.field(static=True)

    def __init__(self, coarse_bridge: StructuredCochainBridge, /):
        if coarse_bridge.dimension not in (2, 3):
            raise ValueError("MHD AMR synchronization requires 2D or 3D cochains.")
        self.coarse_bridge = coarse_bridge
        self.plan_id = canonical_fingerprint(
            {"kind": "mhd-amr-reflux-curl", "bridge": coarse_bridge.bridge_id}
        )

    def reflux_curl(
        self,
        coarse_magnetic_flux: ArrayLike,
        register: ElectromotiveForceRegister,
        /,
    ) -> tuple[Array, MagneticAMRTransferDiagnostics]:
        magnetic = jnp.asarray(coarse_magnetic_flux)
        degree = self.coarse_bridge.dimension - 2
        correction = -self.coarse_bridge.exterior_derivative(degree, register.mismatch)
        updated = magnetic + correction
        before = self.coarse_bridge.exterior_derivative(degree + 1, magnetic)
        after = self.coarse_bridge.exterior_derivative(degree + 1, updated)
        diagnostics = MagneticAMRTransferDiagnostics(
            divergence_before=before,
            divergence_after=after,
            divergence_change=jnp.max(jnp.abs(after - before), initial=0.0),
            flux_defect=jnp.sum(correction),
        )
        return updated, diagnostics


__all__ = [
    "ConstrainedMHDAMRSynchronizationPlan",
    "DivergenceFreeMagneticTransferPlan",
    "ElectromotiveForceRegister",
    "MagneticAMRTransferDiagnostics",
]
