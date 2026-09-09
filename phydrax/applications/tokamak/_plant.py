#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional discrete-plant adapter for native tokamak core transport."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...dynamics import (
    ArrayDiscreteSystemPlant,
    DiscreteSystem,
    DiscreteTransitionResult,
    ExecutableSignature,
    InputLayout,
    NumericRevision,
    PlantParameters,
    SemanticProvenance,
    StateLayout,
)
from ._core_transport import (
    PreparedTokamakCoreTransport,
    TokamakCoreState,
    TokamakEdgeFlux,
    TokamakTransportCoefficients,
    TokamakTransportSources,
)


@dataclass(frozen=True, slots=True)
class PreparedTokamakCorePlant:
    plant: ArrayDiscreteSystemPlant
    parameters: PlantParameters
    plan_id: str


@dataclass(frozen=True, slots=True)
class TokamakCorePlantPlan:
    transport: PreparedTokamakCoreTransport
    coefficients: TokamakTransportCoefficients
    reset_state: TokamakCoreState
    command_lower: np.ndarray
    command_upper: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transport, PreparedTokamakCoreTransport):
            raise TypeError("transport must be PreparedTokamakCoreTransport.")
        if not isinstance(self.coefficients, TokamakTransportCoefficients):
            raise TypeError("coefficients must be TokamakTransportCoefficients.")
        if not isinstance(self.reset_state, TokamakCoreState):
            raise TypeError("reset_state must be TokamakCoreState.")
        count = self.transport.cell_count
        if self.reset_state.electron_density_m3.shape != (count,):
            raise ValueError("reset_state must match the prepared transport geometry.")
        command_size = 3 * count + 3
        lower = np.array(self.command_lower, dtype=np.float64, copy=True)
        upper = np.array(self.command_upper, dtype=np.float64, copy=True)
        if lower.shape != (command_size,) or upper.shape != lower.shape:
            raise ValueError("Tokamak plant command bounds have the wrong shape.")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("Tokamak plant command bounds must be finite and ordered.")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "command_lower", lower)
        object.__setattr__(self, "command_upper", upper)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-core-plant-plan",
                    "transport": self.transport.plan_id,
                    "coefficients": self.coefficients.model_id,
                    "reset": array_tree_fingerprint(
                        (
                            self.reset_state.electron_density_m3,
                            self.reset_state.electron_thermal_energy_j,
                            self.reset_state.ion_thermal_energy_j,
                        )
                    ),
                    "command_lower": array_tree_fingerprint(lower),
                    "command_upper": array_tree_fingerprint(upper),
                }
            ),
        )

    def prepare(self) -> PreparedTokamakCorePlant:
        count = self.transport.cell_count
        reset = jnp.stack(
            (
                self.reset_state.electron_density_m3,
                self.reset_state.electron_thermal_energy_j,
                self.reset_state.ion_thermal_energy_j,
            )
        )
        lower = jnp.asarray(self.command_lower, dtype=reset.dtype)
        upper = jnp.asarray(self.command_upper, dtype=reset.dtype)
        transport = self.transport
        coefficients = self.coefficients

        def transition(context, state, commands, args):
            del args
            command_valid = jnp.all((commands >= lower) & (commands <= upper))
            sources = TokamakTransportSources(
                commands[:count],
                commands[count : 2 * count],
                commands[2 * count : 3 * count],
            )
            edge = TokamakEdgeFlux(commands[-3], commands[-2], commands[-1])
            source_state = TokamakCoreState(state[0], state[1], state[2], context.source)
            result = transport.step(
                source_state,
                context.duration,
                coefficients,
                sources,
                edge,
            )
            candidate = jnp.stack(
                (
                    result.candidate_state.electron_density_m3,
                    result.candidate_state.electron_thermal_energy_j,
                    result.candidate_state.ion_thermal_energy_j,
                )
            )
            successful = command_valid & result.successful
            accepted = jnp.where(successful, candidate, state)
            status = jnp.where(
                successful,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.where(
                    command_valid,
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(2, dtype=jnp.int32),
                ),
            )
            return DiscreteTransitionResult(candidate, accepted, successful, status)

        system = DiscreteSystem(
            transition,
            state_layout=StateLayout((3, count), axes=("quantity", "rho_cell")),
            input_layout=InputLayout(
                (3 * count + 3,), axes=("actuator",), roles="control"
            ),
            system_id=self.plan_id,
        )
        semantic = SemanticProvenance(
            {
                "kind": "tokamak-core-plant",
                "equations": (
                    "electron-particle-balance",
                    "electron-thermal-energy-balance",
                    "ion-thermal-energy-balance",
                ),
                "control_order": (
                    "particle-source-by-cell",
                    "electron-heating-by-cell",
                    "ion-heating-by-cell",
                    "edge-particle-rate",
                    "edge-electron-power",
                    "edge-ion-power",
                ),
                "transport_plan": self.transport.plan_id,
                "plant_plan": self.plan_id,
            },
            resource_ids={"geometry": self.transport.geometry.geometry_id},
        )
        numeric = NumericRevision(
            semantic,
            {
                "conductances": self.coefficients.stacked(),
                "command_lower": lower,
                "command_upper": upper,
            },
        )
        signature = ExecutableSignature(
            shapes={
                "state": reset.shape,
                "control": (3 * count + 3,),
            },
            dtypes={"state": reset.dtype, "control": reset.dtype},
            topology_ids={"flux_surface_geometry": self.transport.geometry.geometry_id},
            capacities={"radial_cells": count},
            algorithm_facts={"step": "implicit-tridiagonal-backward-euler"},
            backend_facts={"runtime": "jax"},
        )
        plant = ArrayDiscreteSystemPlant(
            system,
            lambda key: reset,
            reset_fallback=reset,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            execution_signature=signature,
            control_dtype=reset.dtype,
        )
        parameters = PlantParameters(
            (), plant.parameter_schema.schema_id, plant.numeric_revision
        )
        return PreparedTokamakCorePlant(plant, parameters, self.plan_id)


__all__ = ["PreparedTokamakCorePlant", "TokamakCorePlantPlan"]
