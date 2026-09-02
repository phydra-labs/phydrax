#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_ledger import (
    ConservationStageLedger,
)
from ...discretization.fem._reference import FiniteElementSpec


class ConservativeSubcellEvidence(StrictModule, NonTrainableState):
    subcell_volumes: Array
    constant_defect: Array
    conservation_defect: Array
    positive_volumes: Array
    evidence_id: str = eqx.field(static=True)


class ConservativeSubcellPlan(StrictModule, NonTrainableState):
    dg_to_subcell: Array
    subcell_to_dg: Array
    subcell_volumes: Array
    evidence: ConservativeSubcellEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(self, element: FiniteElementSpec, volume_rule: Any, /):
        from ...integration._rules import reference_rule_data

        if not isinstance(element, FiniteElementSpec):
            raise TypeError("element must be FiniteElementSpec.")
        data = reference_rule_data(volume_rule)
        if data.cell != element.cell_kind:
            raise ValueError("Subcell quadrature and element cell kinds differ.")
        basis = np.asarray(element.tabulate(data.points)[0])
        points = np.asarray(data.points)
        nodes = np.asarray(element.reference_nodes)
        nearest = np.argmin(
            np.max(np.abs(points[:, None, :] - nodes[None, :, :]), axis=-1),
            axis=1,
        )
        projection = np.zeros((nodes.shape[0], nodes.shape[0]), dtype=float)
        volumes = np.zeros((nodes.shape[0],), dtype=float)
        for point, subcell in enumerate(nearest):
            weight = float(np.asarray(data.weights)[point])
            projection[subcell] += weight * basis[point]
            volumes[subcell] += weight
        if np.any(volumes <= 0.0) or np.linalg.matrix_rank(projection) != nodes.shape[0]:
            raise ValueError("Positive unisolvent subcells could not be prepared.")
        reconstruction = np.linalg.solve(projection, np.eye(nodes.shape[0]))
        constant = np.ones((nodes.shape[0],))
        constant_defect = float(np.max(np.abs(projection @ constant - volumes)))
        conservation_defect = float(
            np.max(np.abs(np.sum(projection, axis=0) - np.sum(projection, axis=0)))
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "conservative-subcell-evidence",
                "element": element.element_id,
                "projection": array_tree_fingerprint(projection),
                "volumes": array_tree_fingerprint(volumes),
                "constant_defect": constant_defect,
            }
        )
        self.dg_to_subcell = jnp.asarray(projection)
        self.subcell_to_dg = jnp.asarray(reconstruction)
        self.subcell_volumes = jnp.asarray(volumes)
        self.evidence = ConservativeSubcellEvidence(
            self.subcell_volumes,
            jnp.asarray(constant_defect),
            jnp.asarray(conservation_defect),
            jnp.asarray(True),
            evidence_id,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-subcell-plan",
                "element": element.element_id,
                "evidence": evidence_id,
            }
        )

    def contents(self, nodal_state: ArrayLike, /) -> Array:
        return ein.contract(
            "si,...iv->...sv",
            self.dg_to_subcell,
            jnp.asarray(nodal_state),
            backend="jax",
        )

    def averages(self, nodal_state: ArrayLike, /) -> Array:
        return self.contents(nodal_state) / self.subcell_volumes[:, None]

    def reconstruct(self, subcell_contents: ArrayLike, /) -> Array:
        return ein.contract(
            "is,...sv->...iv",
            self.subcell_to_dg,
            jnp.asarray(subcell_contents),
            backend="jax",
        )


class RobustnessSensorState(StrictModule):
    strength: Array
    troubled: Array
    hysteresis_counter: Array


class RobustnessSensorPlan(StrictModule, NonTrainableState):
    activation: float = eqx.field(static=True)
    release: float = eqx.field(static=True)
    hysteresis_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        activation: float = 0.2,
        release: float = 0.1,
        hysteresis_steps: int = 2,
    ):
        activation_ = float(activation)
        release_ = float(release)
        steps = int(hysteresis_steps)
        if (
            not math.isfinite(activation_)
            or not math.isfinite(release_)
            or activation_ <= release_
            or release_ < 0.0
            or steps < 0
        ):
            raise ValueError("Robustness sensor thresholds are invalid.")
        self.activation = activation_
        self.release = release_
        self.hysteresis_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "robustness-sensor-plan",
                "activation": activation_,
                "release": release_,
                "hysteresis_steps": steps,
            }
        )

    def initial_state(self, cell_count: int, dtype=float, /) -> RobustnessSensorState:
        return RobustnessSensorState(
            jnp.zeros((cell_count,), dtype=dtype),
            jnp.zeros((cell_count,), dtype=bool),
            jnp.zeros((cell_count,), dtype=jnp.int32),
        )

    def evaluate(
        self,
        cell_state: ArrayLike,
        state: RobustnessSensorState,
        /,
        *,
        entropy_residual: ArrayLike | None = None,
    ) -> RobustnessSensorState:
        values = jnp.asarray(cell_state)
        mean = jnp.mean(values, axis=1, keepdims=True)
        modal_indicator = jnp.max(jnp.abs(values - mean), axis=(1, 2)) / jnp.maximum(
            jnp.max(jnp.abs(values), axis=(1, 2)), 1.0e-14
        )
        indicator = modal_indicator
        if entropy_residual is not None:
            entropy = jnp.asarray(entropy_residual)
            if entropy.shape != indicator.shape:
                raise ValueError("Entropy sensor residual shape is incompatible.")
            indicator = jnp.maximum(indicator, jnp.abs(entropy))
        active = indicator >= self.activation
        release = indicator <= self.release
        counter = jnp.where(
            active,
            self.hysteresis_steps,
            jnp.where(
                release,
                jnp.maximum(state.hysteresis_counter - 1, 0),
                state.hysteresis_counter,
            ),
        )
        troubled = active | (counter > 0)
        strength = jnp.clip(
            (indicator - self.release) / (self.activation - self.release), 0.0, 1.0
        )
        return RobustnessSensorState(strength, troubled, counter)


class EntropyViscosityPlan(StrictModule, NonTrainableState):
    coefficient: float = eqx.field(static=True)
    maximum_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, coefficient: float = 0.5, maximum_fraction: float = 0.5, /):
        coefficient_ = float(coefficient)
        maximum_ = float(maximum_fraction)
        if (
            not math.isfinite(coefficient_)
            or coefficient_ < 0.0
            or not math.isfinite(maximum_)
            or not 0.0 <= maximum_ <= 1.0
        ):
            raise ValueError("Entropy-viscosity controls are invalid.")
        self.coefficient = coefficient_
        self.maximum_fraction = maximum_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "entropy-viscosity-plan",
                "coefficient": coefficient_,
                "maximum_fraction": maximum_,
            }
        )

    def viscosity(
        self,
        sensor_strength: ArrayLike,
        cell_length: ArrayLike,
        wave_speed: ArrayLike,
        /,
    ) -> Array:
        return jnp.minimum(
            self.coefficient
            * jnp.asarray(sensor_strength)
            * jnp.asarray(cell_length)
            * jnp.asarray(wave_speed),
            self.maximum_fraction * jnp.asarray(cell_length) * jnp.asarray(wave_speed),
        )


class ConservationCorrectionResult(StrictModule):
    selected_ledger: ConservationStageLedger
    stage_content_rate: Array
    correction_level: Array
    successful: Array
    decision_id: str = eqx.field(static=True)


class ConservationCorrectionLadderPlan(StrictModule, NonTrainableState):
    differentiability_policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, /, *, differentiability_policy_id: str = "branchwise"):
        policy = str(differentiability_policy_id)
        if policy not in ("branchwise", "unsupported"):
            raise ValueError("Hard correction ladders are branchwise or unsupported.")
        self.differentiability_policy_id = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservation-correction-ladder",
                "differentiability": policy,
            }
        )

    def apply(
        self,
        high_order: ConservationStageLedger,
        low_order: ConservationStageLedger,
        sensor: RobustnessSensorState,
        /,
        *,
        cell_content: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        admissible=None,
    ) -> ConservationCorrectionResult:
        if not isinstance(high_order, ConservationStageLedger) or not isinstance(
            low_order, ConservationStageLedger
        ):
            raise TypeError("Correction ladder requires conservation stage ledgers.")
        if (
            high_order.geometry_layout_id != low_order.geometry_layout_id
            or high_order.topology_epoch_id != low_order.topology_epoch_id
            or tuple(block.route_id for block in high_order.blocks)
            != tuple(block.route_id for block in low_order.blocks)
            or sensor.strength.shape != (high_order.cell_count,)
        ):
            raise ValueError(
                "Correction ladder ledger identities or sensor shape differ."
            )
        selected_blocks = []
        factors = []
        for high, low in zip(high_order.blocks, low_order.blocks, strict=True):
            owner_strength = sensor.strength[high.owner_cells]
            neighbour_index = jnp.maximum(high.neighbour_cells, 0)
            neighbour_strength = jnp.where(
                high.neighbour_cells >= 0,
                sensor.strength[neighbour_index],
                owner_strength,
            )
            high_fraction = 1.0 - jnp.maximum(owner_strength, neighbour_strength)
            shape = high_fraction.shape + (1,) * len(high.component_shape)
            rate = (
                high_fraction.reshape(shape) * high.flux_rate
                + (1.0 - high_fraction.reshape(shape)) * low.flux_rate
            )
            selected_blocks.append(high.with_flux_rate(rate))
            factors.append(high_fraction)
        cell_factor = 1.0 - sensor.strength.reshape(
            sensor.strength.shape + (1,) * len(high_order.component_shape)
        )
        source = (
            cell_factor * high_order.source_rate
            + (1.0 - cell_factor) * low_order.source_rate
        )
        levels = jnp.where(
            sensor.strength >= 1.0,
            2,
            jnp.where(sensor.troubled, 1, 0),
        ).astype(jnp.int32)
        selected = high_order.with_selected_rates(
            tuple(selected_blocks),
            source,
            blend_factors=tuple(factors),
            troubled_cell_mask=sensor.troubled,
            correction_level=levels,
            accepted=True,
            differentiability_policy_id=self.differentiability_policy_id,
        )
        rate = selected.scatter_content_rate()
        successful = jnp.asarray(True)
        if cell_content is not None or step_size is not None or admissible is not None:
            if cell_content is None or step_size is None or not callable(admissible):
                raise ValueError(
                    "Content, step size, and admissibility must be supplied together."
                )
            candidate = jnp.asarray(cell_content) + jnp.asarray(step_size) * rate
            successful = jnp.all(admissible(candidate)) & jnp.all(jnp.isfinite(candidate))
        return ConservationCorrectionResult(
            selected,
            rate,
            levels,
            successful,
            canonical_fingerprint(
                {
                    "kind": "conservation-correction-decision",
                    "plan": self.plan_id,
                    "high_order": high_order.ledger_id,
                    "low_order": low_order.ledger_id,
                }
            ),
        )


__all__ = [
    "ConservationCorrectionLadderPlan",
    "ConservationCorrectionResult",
    "ConservativeSubcellEvidence",
    "ConservativeSubcellPlan",
    "EntropyViscosityPlan",
    "RobustnessSensorPlan",
    "RobustnessSensorState",
]
