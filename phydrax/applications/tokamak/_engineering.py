#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit tokamak-core to fusion-source to activation coupling."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nuclear import (
    ActivationStepResult,
    NuclideInventory,
    PreparedActivationNetwork,
    ThermalFusionReactionPlan,
    ThermalFusionReactionResult,
)
from ._core_transport import (
    PreparedTokamakCoreTransport,
    TokamakCoreState,
    TokamakCoreTransportStepResult,
    TokamakEdgeFlux,
    TokamakTransportCoefficients,
    TokamakTransportSources,
)


class PreparedNeutronResponse(StrictModule, NonTrainableState):
    """Fixed linear map from core neutron rate to one region's group flux."""

    scalar_flux_per_source_m2: Array
    source_id: str = eqx.field(static=True)
    response_id: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)

    def apply(self, source_rate_s: ArrayLike, /) -> Array:
        source = jnp.asarray(source_rate_s, dtype=self.scalar_flux_per_source_m2.dtype)
        if source.shape != (self.scalar_flux_per_source_m2.shape[1],):
            raise ValueError("Neutron source must match the response core-cell axis.")
        return contract("gc,c->g", self.scalar_flux_per_source_m2, source)


@dataclass(frozen=True, slots=True)
class NeutronResponsePlan:
    scalar_flux_per_source_m2: np.ndarray
    source_id: str
    differentiable: bool = False
    response_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.array(self.scalar_flux_per_source_m2, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("Neutron response must have shape (group, core_cell).")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Neutron response entries must be finite and nonnegative.")
        source = str(self.source_id).strip()
        if not source or source != self.source_id:
            raise ValueError("source_id must be non-empty canonical text.")
        if not isinstance(self.differentiable, bool):
            raise TypeError("differentiable must be boolean.")
        values.setflags(write=False)
        object.__setattr__(self, "scalar_flux_per_source_m2", values)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "response_id",
            canonical_fingerprint(
                {
                    "kind": "neutron-response-plan",
                    "values": array_tree_fingerprint(values),
                    "source": source,
                    "differentiable": self.differentiable,
                }
            ),
        )

    def prepare(self) -> PreparedNeutronResponse:
        values = jnp.asarray(self.scalar_flux_per_source_m2)
        if not self.differentiable:
            values = jax.lax.stop_gradient(values)
        return PreparedNeutronResponse(
            values,
            self.source_id,
            self.response_id,
            "native" if self.differentiable else "constant-artifact",
        )


class FusionActivationStepResult(StrictModule):
    transport: TokamakCoreTransportStepResult
    fusion: ThermalFusionReactionResult
    core_neutron_source_rate_s: Array
    activation_scalar_flux_m2_s: Array
    activation: ActivationStepResult
    charged_product_heating_w_m3: Array
    finite: Array
    sensitivity_valid: Array
    successful: Array


class PreparedFusionActivationScenario(StrictModule, NonTrainableState):
    transport: PreparedTokamakCoreTransport
    fusion: ThermalFusionReactionPlan
    activation: PreparedActivationNetwork
    neutron_response: PreparedNeutronResponse
    reactant_a_fraction: float = eqx.field(static=True)
    reactant_b_fraction: float = eqx.field(static=True)
    neutron_product_index: int = eqx.field(static=True)
    scenario_id: str = eqx.field(static=True)

    def step(
        self,
        core_state: TokamakCoreState,
        inventory: NuclideInventory,
        dt_s: ArrayLike,
        transport_coefficients: TokamakTransportCoefficients,
        transport_sources: TokamakTransportSources,
        edge_flux: TokamakEdgeFlux | None = None,
        /,
    ) -> FusionActivationStepResult:
        transport = self.transport.step(
            core_state,
            dt_s,
            transport_coefficients,
            transport_sources,
            edge_flux,
        )
        state = transport.accepted_state
        density_a = self.reactant_a_fraction * state.electron_density_m3
        density_b = self.reactant_b_fraction * state.electron_density_m3
        fusion = self.fusion.evaluate(
            density_a,
            density_b,
            state.ion_thermal_energy_j,
        )
        neutron_rate_density = fusion.products.product_rate_density_m3_s[
            :, self.neutron_product_index
        ]
        neutron_rate = neutron_rate_density * self.transport.geometry.cell_volume_m3
        scalar_flux = self.neutron_response.apply(neutron_rate)
        activation_dt = jnp.where(
            transport.successful & fusion.successful,
            jnp.asarray(dt_s),
            -jnp.ones_like(jnp.asarray(dt_s)),
        )
        activation = self.activation.step(inventory, scalar_flux, activation_dt)
        charged_index = 1 - self.neutron_product_index
        charged_heating = fusion.products.product_power_density_w_m3[:, charged_index]
        finite = (
            transport.finite
            & fusion.finite
            & activation.finite
            & jnp.all(jnp.isfinite(scalar_flux))
            & jnp.all(jnp.isfinite(charged_heating))
        )
        successful = (
            transport.successful & fusion.successful & activation.successful & finite
        )
        sensitivity_valid = successful & (
            self.neutron_response.differentiation == "native"
        )
        return FusionActivationStepResult(
            transport,
            fusion,
            neutron_rate,
            scalar_flux,
            activation,
            charged_heating,
            finite,
            sensitivity_valid,
            successful,
        )


@dataclass(frozen=True, slots=True)
class FusionActivationScenarioPlan:
    transport: PreparedTokamakCoreTransport
    fusion: ThermalFusionReactionPlan
    activation: PreparedActivationNetwork
    neutron_response: NeutronResponsePlan
    reactant_a_fraction: float
    reactant_b_fraction: float
    neutron_product_index: int
    scenario_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transport, PreparedTokamakCoreTransport):
            raise TypeError("transport must be PreparedTokamakCoreTransport.")
        if not isinstance(self.fusion, ThermalFusionReactionPlan):
            raise TypeError("fusion must be ThermalFusionReactionPlan.")
        if not isinstance(self.activation, PreparedActivationNetwork):
            raise TypeError("activation must be PreparedActivationNetwork.")
        if not isinstance(self.neutron_response, NeutronResponsePlan):
            raise TypeError("neutron_response must be NeutronResponsePlan.")
        fractions = (float(self.reactant_a_fraction), float(self.reactant_b_fraction))
        if (
            any(not np.isfinite(value) or value < 0.0 for value in fractions)
            or sum(fractions) > 1.0
        ):
            raise ValueError(
                "Fusion reactant fractions must be nonnegative and sum to at most one."
            )
        product = int(self.neutron_product_index)
        if product not in (0, 1):
            raise ValueError("neutron_product_index must be zero or one.")
        if self.neutron_response.scalar_flux_per_source_m2.shape != (
            self.activation.microscopic_cross_sections_m2.shape[1],
            self.transport.cell_count,
        ):
            raise ValueError(
                "Neutron response must connect core cells to activation groups."
            )
        if (
            self.fusion.product_species_ids[product]
            == self.fusion.product_species_ids[1 - product]
        ):
            raise ValueError(
                "Fusion neutron and charged products must be distinguishable."
            )
        object.__setattr__(self, "reactant_a_fraction", fractions[0])
        object.__setattr__(self, "reactant_b_fraction", fractions[1])
        object.__setattr__(self, "neutron_product_index", product)
        object.__setattr__(
            self,
            "scenario_id",
            canonical_fingerprint(
                {
                    "kind": "fusion-activation-scenario",
                    "transport": self.transport.plan_id,
                    "fusion": self.fusion.plan_id,
                    "activation": self.activation.network_id,
                    "response": self.neutron_response.response_id,
                    "reactant_fractions": list(fractions),
                    "neutron_product_index": product,
                    "coupling": "staggered-transport-fusion-response-activation",
                }
            ),
        )

    def prepare(self) -> PreparedFusionActivationScenario:
        return PreparedFusionActivationScenario(
            self.transport,
            self.fusion,
            self.activation,
            self.neutron_response.prepare(),
            self.reactant_a_fraction,
            self.reactant_b_fraction,
            self.neutron_product_index,
            self.scenario_id,
        )


__all__ = [
    "FusionActivationScenarioPlan",
    "FusionActivationStepResult",
    "NeutronResponsePlan",
    "PreparedFusionActivationScenario",
    "PreparedNeutronResponse",
]
