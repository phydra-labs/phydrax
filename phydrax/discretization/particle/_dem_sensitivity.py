#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMDiagnostics


class DEMSensitivityMode(StrEnum):
    SHARP_BRANCHWISE = "sharp_branchwise"
    SMOOTH_SURROGATE = "smooth_surrogate"
    HYBRID_EVENT_AWARE = "hybrid_event_aware"


class DEMTrainableMaterialParameters(StrictModule):
    """Unconstrained continuous coordinates for one static DEM material schema."""

    log_young_modulus: Array
    poisson_coordinate: Array
    restitution_coordinate: Array
    friction_coordinate: Array
    material_count: int = eqx.field(static=True)
    parameter_schema_id: str = eqx.field(static=True)

    @classmethod
    def from_materials(cls, materials: Any, /) -> DEMTrainableMaterialParameters:
        young = jnp.asarray(materials.young_modulus)
        poisson = jnp.asarray(materials.poisson_ratio)
        restitution = jnp.asarray(materials.restitution)
        friction = jnp.asarray(materials.friction)
        poisson_unit = jnp.clip((poisson + 1.0) / 1.5, 1.0e-12, 1.0 - 1.0e-12)
        restitution_unit = jnp.clip(restitution, 1.0e-12, 1.0 - 1.0e-12)
        friction_safe = jnp.maximum(friction, 1.0e-12)
        return cls(
            jnp.log(young),
            jnp.log(poisson_unit) - jnp.log1p(-poisson_unit),
            jnp.log(restitution_unit) - jnp.log1p(-restitution_unit),
            jnp.log(jnp.expm1(friction_safe)),
            int(materials.material_count),
            canonical_fingerprint(
                {
                    "kind": "dem-trainable-material-parameters",
                    "material_id": materials.material_id,
                    "material_count": int(materials.material_count),
                    "shapes": {
                        "young": list(young.shape),
                        "poisson": list(poisson.shape),
                        "restitution": list(restitution.shape),
                        "friction": list(friction.shape),
                    },
                }
            ),
        )

    def apply(self, materials: Any, /):
        if int(materials.material_count) != self.material_count:
            raise ValueError("Trainable parameters do not match material count.")
        young = jnp.exp(self.log_young_modulus)
        poisson = -1.0 + 1.5 * jax.nn.sigmoid(self.poisson_coordinate)
        restitution_coordinate = 0.5 * (
            self.restitution_coordinate + self.restitution_coordinate.T
        )
        friction_coordinate = 0.5 * (
            self.friction_coordinate + self.friction_coordinate.T
        )
        restitution = jax.nn.sigmoid(restitution_coordinate)
        friction = jax.nn.softplus(friction_coordinate)
        return eqx.tree_at(
            lambda table: (
                table.young_modulus,
                table.poisson_ratio,
                table.restitution,
                table.friction,
            ),
            materials,
            (young, poisson, restitution, friction),
        )


class DEMSensitivityPolicy(StrictModule, NonTrainableState):
    mode: DEMSensitivityMode = eqx.field(static=True)
    activation_margin: float = eqx.field(static=True)
    no_tension_margin: float = eqx.field(static=True)
    friction_margin: float = eqx.field(static=True)
    frame_margin: float = eqx.field(static=True)
    acceptance_margin: float = eqx.field(static=True)
    neighborhood_margin: float = eqx.field(static=True)
    perturbation_scale: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: DEMSensitivityMode = DEMSensitivityMode.SHARP_BRANCHWISE,
        activation_margin: float = 1.0e-8,
        no_tension_margin: float = 1.0e-8,
        friction_margin: float = 1.0e-8,
        frame_margin: float = 1.0e-8,
        acceptance_margin: float = 1.0e-8,
        neighborhood_margin: float = 1.0e-8,
        perturbation_scale: float = 1.0e-8,
    ):
        if not isinstance(mode, DEMSensitivityMode):
            raise TypeError("mode must be a DEMSensitivityMode.")
        values = tuple(
            float(value)
            for value in (
                activation_margin,
                no_tension_margin,
                friction_margin,
                frame_margin,
                acceptance_margin,
                neighborhood_margin,
                perturbation_scale,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Sensitivity margins must be finite and positive.")
        (
            self.activation_margin,
            self.no_tension_margin,
            self.friction_margin,
            self.frame_margin,
            self.acceptance_margin,
            self.neighborhood_margin,
            self.perturbation_scale,
        ) = values
        self.mode = mode
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dem-sensitivity-policy",
                "mode": mode.value,
                "margins": list(values),
            }
        )


class DEMLocalValidityCertificate(StrictModule, NonTrainableState):
    activation_margin: Array
    no_tension_margin: Array
    friction_margin: Array
    frame_margin: Array
    acceptance_margin: Array
    neighborhood_margin: Array
    forward_successful: Array
    locally_valid: Array
    policy_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


class DEMSensitivityResult(StrictModule):
    primal: Any
    sensitivity: Any
    certificate: DEMLocalValidityCertificate
    usable: Array
    mode: DEMSensitivityMode = eqx.field(static=True)


def dem_local_validity_certificate(
    diagnostics: DEMDiagnostics,
    policy: DEMSensitivityPolicy,
    /,
) -> DEMLocalValidityCertificate:
    if not isinstance(diagnostics, DEMDiagnostics):
        raise TypeError("diagnostics must be DEMDiagnostics.")
    if not isinstance(policy, DEMSensitivityPolicy):
        raise TypeError("policy must be a DEMSensitivityPolicy.")
    margins = (
        diagnostics.minimum_gap_margin,
        diagnostics.minimum_no_tension_margin,
        diagnostics.minimum_friction_switch_margin,
        diagnostics.minimum_frame_transport_margin,
        diagnostics.acceptance_margin,
        diagnostics.neighborhood_certificate_margin,
    )
    thresholds = (
        policy.activation_margin,
        policy.no_tension_margin,
        policy.friction_margin,
        policy.frame_margin,
        policy.acceptance_margin,
        policy.neighborhood_margin,
    )
    valid = diagnostics.successful
    for margin, threshold in zip(margins, thresholds, strict=True):
        valid = valid & (margin >= threshold)
    return DEMLocalValidityCertificate(
        *margins,
        diagnostics.successful,
        valid,
        policy.policy_id,
        canonical_fingerprint(
            {
                "kind": "dem-local-validity-certificate",
                "policy": policy.policy_id,
                "schema": "dem-local-validity:v1",
            }
        ),
    )


def _invalid_sensitivity(tree: PyTree[Any], /):
    return jax.tree.map(
        lambda leaf: jnp.full_like(leaf, jnp.nan) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def sharp_branchwise_jvp(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    direction: PyTree[Any],
    diagnostics: DEMDiagnostics,
    policy: DEMSensitivityPolicy,
    /,
) -> DEMSensitivityResult:
    if policy.mode is not DEMSensitivityMode.SHARP_BRANCHWISE:
        raise ValueError("sharp_branchwise_jvp requires sharp_branchwise policy.")
    primal, tangent = jax.jvp(function, (parameters,), (direction,))
    certificate = dem_local_validity_certificate(diagnostics, policy)
    sensitivity = jax.lax.cond(
        certificate.locally_valid,
        lambda value: value,
        _invalid_sensitivity,
        tangent,
    )
    return DEMSensitivityResult(
        primal,
        sensitivity,
        certificate,
        certificate.locally_valid,
        policy.mode,
    )


def sharp_branchwise_vjp(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    cotangent: PyTree[Any],
    diagnostics: DEMDiagnostics,
    policy: DEMSensitivityPolicy,
    /,
) -> DEMSensitivityResult:
    if policy.mode is not DEMSensitivityMode.SHARP_BRANCHWISE:
        raise ValueError("sharp_branchwise_vjp requires sharp_branchwise policy.")
    primal, pullback = jax.vjp(function, parameters)
    sensitivity = pullback(cotangent)[0]
    certificate = dem_local_validity_certificate(diagnostics, policy)
    sensitivity = jax.lax.cond(
        certificate.locally_valid,
        lambda value: value,
        _invalid_sensitivity,
        sensitivity,
    )
    return DEMSensitivityResult(
        primal,
        sensitivity,
        certificate,
        certificate.locally_valid,
        policy.mode,
    )


__all__ = [
    "DEMLocalValidityCertificate",
    "DEMSensitivityMode",
    "DEMSensitivityPolicy",
    "DEMSensitivityResult",
    "DEMTrainableMaterialParameters",
    "dem_local_validity_certificate",
    "sharp_branchwise_jvp",
    "sharp_branchwise_vjp",
]
