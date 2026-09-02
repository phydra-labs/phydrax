#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._closure import DifferentiationContract
from ._products import (
    CosmologyProductProvenance,
    MatterPowerDescriptor,
    MatterPowerTable,
)


class OneLoopSPTEvidence(StrictModule):
    p22: Array
    p13: Array
    relative_correction: Array
    infrared_cancellation_ratio: Array
    finite: Array
    positive_output: Array
    within_validity: Array
    successful: Array


class OneLoopSPTResult(StrictModule):
    power: MatterPowerTable
    evidence: OneLoopSPTEvidence
    successful: Array


class OneLoopEdSSPTPlan(StrictModule, NonTrainableState):
    """Fixed-quadrature EdS-kernel one-loop matter power P11+P22+P13."""

    output_wavenumbers: Array
    radial_nodes: Array
    radial_weights: Array
    angular_nodes: Array
    angular_weights: Array
    maximum_relative_correction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        output_wavenumbers: ArrayLike,
        /,
        *,
        radial_order: int = 96,
        angular_order: int = 64,
        radial_ratio_domain: tuple[float, float] = (1.0e-3, 1.0e3),
        maximum_relative_correction: float = 0.5,
    ):
        k = np.asarray(output_wavenumbers, dtype=float).reshape((-1,))
        r_order = int(radial_order)
        x_order = int(angular_order)
        r_min, r_max = (float(value) for value in radial_ratio_domain)
        correction = float(maximum_relative_correction)
        if (
            k.size < 2
            or np.any(~np.isfinite(k))
            or np.any(k <= 0.0)
            or np.any(np.diff(k) <= 0.0)
            or r_order < 16
            or x_order < 16
            or not np.isfinite(r_min)
            or not np.isfinite(r_max)
            or r_min <= 0.0
            or r_max <= r_min
            or not np.isfinite(correction)
            or correction <= 0.0
        ):
            raise ValueError("One-loop SPT quadrature policy is invalid.")
        radial_rule = gauss_legendre_data(r_order)
        log_min = np.log(r_min)
        log_max = np.log(r_max)
        log_nodes = 0.5 * (log_max - log_min) * radial_rule.nodes + 0.5 * (
            log_max + log_min
        )
        radial_nodes = np.exp(log_nodes)
        radial_weights = 0.5 * (log_max - log_min) * radial_rule.weights * radial_nodes
        angular_rule = gauss_legendre_data(x_order)
        self.output_wavenumbers = jnp.asarray(k)
        self.radial_nodes = jnp.asarray(radial_nodes)
        self.radial_weights = jnp.asarray(radial_weights)
        self.angular_nodes = jnp.asarray(angular_rule.nodes)
        self.angular_weights = jnp.asarray(angular_rule.weights)
        self.maximum_relative_correction = correction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "one-loop-eds-spt",
                "output_wavenumbers": k.tolist(),
                "radial_order": r_order,
                "angular_order": x_order,
                "radial_ratio_domain": [r_min, r_max],
                "maximum_relative_correction": correction,
            }
        )

    def evaluate(
        self,
        background: FLRWBackground,
        linear_power: MatterPowerTable,
        scale_factor: ArrayLike,
        /,
    ) -> OneLoopSPTResult:
        if (
            linear_power.descriptor.stage != "linear"
            or not linear_power.descriptor.is_auto
        ):
            raise ValueError("One-loop SPT requires linear auto-power.")
        if linear_power.descriptor.spatial_dimension != 3:
            raise ValueError("One-loop SPT requires three-dimensional power.")
        scale = background.require_flat(jnp.asarray(scale_factor))
        scale = background.realization.require_compatible(linear_power.realization, scale)
        k = self.output_wavenumbers.astype(linear_power.power_values.dtype)
        r = self.radial_nodes.astype(k.dtype)
        x = self.angular_nodes.astype(k.dtype)
        q = k[:, None] * r[None, :]
        s = jnp.sqrt(
            jnp.maximum(
                1.0 + r[None, :, None] ** 2 - 2.0 * r[None, :, None] * x[None, None, :],
                jnp.finfo(k.dtype).tiny,
            )
        )
        second_q = k[:, None, None] * s
        p_k = linear_power.evaluate(k, scale)
        p_q = linear_power.evaluate(q, scale)
        p_second = linear_power.evaluate(second_q, scale)
        numerator = (
            3.0 * r[None, :, None]
            + 7.0 * x[None, None, :]
            - 10.0 * r[None, :, None] * x[None, None, :] ** 2
        ) ** 2
        kernel_22 = numerator / s**4
        angular = contract(
            "x,krx->kr",
            self.angular_weights.astype(k.dtype),
            p_second * kernel_22,
        )
        prefactor_22 = k**3 / (98.0 * (2.0 * jnp.pi) ** 2)
        p22 = prefactor_22 * contract(
            "r,kr,kr->k",
            self.radial_weights.astype(k.dtype),
            p_q,
            angular,
        )
        log_ratio = jnp.log(jnp.abs((1.0 + r) / (1.0 - r)))
        kernel_13 = (
            12.0 / r**2
            - 158.0
            + 100.0 * r**2
            - 42.0 * r**4
            + 3.0 * (r**2 - 1.0) ** 3 * (7.0 * r**2 + 2.0) * log_ratio / r**3
        )
        radial_13 = contract(
            "r,kr,r->k",
            self.radial_weights.astype(k.dtype),
            p_q,
            kernel_13,
        )
        p13 = k**3 * p_k * radial_13 / (252.0 * (2.0 * jnp.pi) ** 2)
        correction = p22 + p13
        output = p_k + correction
        relative = jnp.abs(correction) / jnp.maximum(p_k, jnp.finfo(p_k.dtype).tiny)
        cancellation = jnp.abs(correction[0]) / jnp.maximum(
            jnp.abs(p22[0]) + jnp.abs(p13[0]),
            jnp.finfo(p_k.dtype).tiny,
        )
        finite = (
            jnp.all(jnp.isfinite(p22))
            & jnp.all(jnp.isfinite(p13))
            & jnp.all(jnp.isfinite(output))
        )
        positive = jnp.all(output >= 0.0)
        validity = jnp.all(relative <= self.maximum_relative_correction)
        successful = finite & positive & validity
        output = eqx.error_if(
            output,
            ~successful,
            "One-loop SPT exceeded its finite, positive, or relative-correction domain.",
        )
        provenance = CosmologyProductProvenance(
            producer="phydrax.applications.cosmology.OneLoopEdSSPTPlan",
            producer_version="native",
            model_form_id=linear_power.provenance.model_form_id,
            request_id=linear_power.provenance.request_id,
            numerical_policy_id=self.plan_id,
            physics_policy_id="flat-massless-neutrino-eds-kernel-one-loop-spt",
            scale_id=linear_power.scale.scale_id,
            source_kind="native",
            differentiation=linear_power.provenance.differentiation.meet(
                DifferentiationContract(
                    upstream_physical_parameters=True,
                    stored_values=True,
                    query_coordinates=True,
                    local_parameters=False,
                )
            ),
            parent_product_ids=(linear_power.provenance.provenance_id,),
        )
        descriptor = MatterPowerDescriptor(
            linear_power.descriptor.left_field,
            linear_power.descriptor.right_field,
            gauge=linear_power.descriptor.gauge,
            normalization=linear_power.descriptor.normalization,
            stage="nonlinear",
            shot_noise=linear_power.descriptor.shot_noise,
            spatial_dimension=3,
        )
        power = MatterPowerTable(
            jnp.asarray([scale, scale + jnp.finfo(scale.dtype).eps]),
            k,
            jnp.stack((output, output)),
            descriptor,
            linear_power.scale,
            provenance,
            linear_power.realization,
        )
        evidence = OneLoopSPTEvidence(
            p22,
            p13,
            relative,
            cancellation,
            finite,
            positive,
            validity,
            successful,
        )
        return OneLoopSPTResult(power, evidence, successful)


__all__ = ["OneLoopEdSSPTPlan", "OneLoopSPTEvidence", "OneLoopSPTResult"]
