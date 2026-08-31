#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._dem_contact_state import DEMCohesionHistory, DEMContactEvaluationContext


class DEMCohesionComponentHistory(StrictModule):
    active: Array
    bridge_volume: Array
    previous_gap: Array
    birth_step: Array


class DEMCohesionResponse(StrictModule):
    force_magnitude: Array
    friction_load: Array
    stiffness: Array
    elastic_energy: Array
    dissipated_work: Array
    active: Array
    born: Array
    ruptured: Array
    birth_margin: Array
    rupture_margin: Array
    bridge_volume_source: Array
    bridge_volume_release: Array
    bridge_volume_residual: Array
    next_history: DEMCohesionHistory
    successful: Array


class AbstractDEMCohesionPlan(StrictModule, NonTrainableState):
    cohesion_law_id: AbstractAttribute[str]
    maximum_interaction_range: AbstractAttribute[float]

    @abc.abstractmethod
    def initialize_history(self, capacity: int, dtype: Any, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        batch: Any,
        normal: Any,
        history: Any,
        context: DEMContactEvaluationContext,
        materials: Any,
        /,
    ) -> tuple[DEMCohesionResponse, Any]:
        raise NotImplementedError


class DMTContactCohesionPlan(AbstractDEMCohesionPlan):
    surface_energy: Array
    cutoff: Array
    maximum_interaction_range: float = eqx.field(static=True)
    cohesion_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_energy: ArrayLike,
        cutoff: ArrayLike,
        /,
        *,
        cohesion_law_id: str | None = None,
    ):
        energy = np.asarray(surface_energy)
        cutoff_ = np.asarray(cutoff)
        _validate_pair_parameter("surface_energy", energy, positive=True)
        _validate_pair_parameter("cutoff", cutoff_, positive=True)
        _require_matching_schema(energy, cutoff_)
        generated = canonical_fingerprint(
            {
                "kind": "dmt-contact-cohesion",
                "surface_energy": array_tree_fingerprint(energy),
                "cutoff": array_tree_fingerprint(cutoff_),
            }
        )
        identifier = generated if cohesion_law_id is None else str(cohesion_law_id)
        if not identifier:
            raise ValueError("cohesion_law_id must be nonempty.")
        self.surface_energy = jnp.asarray(energy)
        self.cutoff = jnp.asarray(cutoff_)
        self.maximum_interaction_range = float(np.max(cutoff_))
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return _empty_component_history(capacity, dtype)

    def evaluate(self, batch, normal, history, context, materials, /):
        del normal
        energy = _pair_value(
            self.surface_energy,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        cutoff = _pair_value(
            self.cutoff,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        pull_off = 4.0 * jnp.pi * batch.effective_radius * energy
        positive_gap = jnp.maximum(batch.gap, 0.0)
        shape = jnp.where(
            batch.gap <= 0.0,
            1.0,
            jnp.maximum(1.0 - positive_gap / cutoff, 0.0),
        )
        active = (
            batch.valid
            & (batch.gap < cutoff)
            & (context.left_inverse_mass + context.right_inverse_mass > 0.0)
        )
        force = jnp.where(active, -pull_off * shape, 0.0)
        potential = jnp.where(
            batch.gap <= 0.0,
            pull_off * batch.gap - 0.5 * pull_off * cutoff,
            -0.5 * pull_off * cutoff * shape**2,
        )
        stiffness = jnp.where(active & (batch.gap > 0.0), pull_off / cutoff, 0.0)
        next_history = DEMCohesionComponentHistory(
            active,
            jnp.zeros_like(batch.gap),
            jnp.where(context.current_valid, batch.gap, 0.0),
            history.birth_step,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(stiffness))
        )
        response = DEMCohesionResponse(
            force,
            jnp.abs(force),
            stiffness,
            jnp.where(active, potential, 0.0),
            jnp.zeros_like(force),
            active,
            active & ~history.active,
            history.active & ~active,
            jnp.abs(batch.gap),
            jnp.abs(cutoff - batch.gap),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            DEMCohesionHistory(()),
            finite,
        )
        return response, next_history


class LinearCapillaryBridgePlan(AbstractDEMCohesionPlan):
    """Conservative finite-range bridge with linear force-separation energy."""

    surface_tension: Array
    contact_angle: Array
    bridge_volume: Array
    rupture_distance: Array
    maximum_interaction_range: float = eqx.field(static=True)
    cohesion_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_tension: ArrayLike,
        contact_angle: ArrayLike,
        bridge_volume: ArrayLike,
        rupture_distance: ArrayLike,
        /,
        *,
        cohesion_law_id: str | None = None,
    ):
        tension = np.asarray(surface_tension)
        angle = np.asarray(contact_angle)
        volume = np.asarray(bridge_volume)
        rupture = np.asarray(rupture_distance)
        _validate_pair_parameter("surface_tension", tension, positive=True)
        _validate_pair_parameter("contact_angle", angle, nonnegative=True)
        _validate_pair_parameter("bridge_volume", volume, positive=True)
        _validate_pair_parameter("rupture_distance", rupture, positive=True)
        _require_matching_schema(tension, angle, volume, rupture)
        if np.any(angle > np.pi):
            raise ValueError("contact_angle must lie in [0, pi].")
        generated = canonical_fingerprint(
            {
                "kind": "linear-capillary-bridge",
                "surface_tension": array_tree_fingerprint(tension),
                "contact_angle": array_tree_fingerprint(angle),
                "bridge_volume": array_tree_fingerprint(volume),
                "rupture_distance": array_tree_fingerprint(rupture),
            }
        )
        identifier = generated if cohesion_law_id is None else str(cohesion_law_id)
        if not identifier:
            raise ValueError("cohesion_law_id must be nonempty.")
        self.surface_tension = jnp.asarray(tension)
        self.contact_angle = jnp.asarray(angle)
        self.bridge_volume = jnp.asarray(volume)
        self.rupture_distance = jnp.asarray(rupture)
        self.maximum_interaction_range = float(np.max(rupture))
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return _empty_component_history(capacity, dtype)

    def evaluate(self, batch, normal, history, context, materials, /):
        del normal
        tension = _pair_value(
            self.surface_tension,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        angle = _pair_value(
            self.contact_angle,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        volume = _pair_value(
            self.bridge_volume,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        rupture = _pair_value(
            self.rupture_distance,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        born = batch.valid & (batch.gap <= 0.0) & ~history.active & (volume > 0.0)
        retained = history.active & batch.valid & (batch.gap < rupture)
        active = born | retained
        ruptured = history.active & ~active
        bridge_volume = jnp.where(born, volume, history.bridge_volume)
        bridge_volume = jnp.where(active, bridge_volume, 0.0)
        peak = (
            2.0
            * jnp.pi
            * batch.effective_radius
            * tension
            * jnp.maximum(jnp.cos(angle), 0.0)
        )
        positive_gap = jnp.maximum(batch.gap, 0.0)
        shape = jnp.maximum(1.0 - positive_gap / rupture, 0.0)
        force = jnp.where(active, -peak * shape, 0.0)
        potential = jnp.where(
            batch.gap <= 0.0,
            peak * batch.gap - 0.5 * peak * rupture,
            -0.5 * peak * rupture * shape**2,
        )
        birth_step = jnp.where(born, context.step_index, history.birth_step)
        birth_step = jnp.where(active, birth_step, -1)
        next_history = DEMCohesionComponentHistory(
            active,
            bridge_volume,
            jnp.where(context.current_valid, batch.gap, 0.0),
            birth_step.astype(jnp.int32),
        )
        bridge_source = jnp.where(born, volume, 0.0)
        bridge_release = jnp.where(ruptured, history.bridge_volume, 0.0)
        bridge_residual = (
            bridge_volume - history.bridge_volume - bridge_source + bridge_release
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(bridge_volume >= 0.0)
            & jnp.all(
                jnp.abs(bridge_residual)
                <= 64.0
                * jnp.finfo(bridge_residual.dtype).eps
                * jnp.maximum(jnp.abs(volume), 1.0)
            )
        )
        response = DEMCohesionResponse(
            force,
            jnp.abs(force),
            jnp.where(active & (batch.gap > 0.0), peak / rupture, 0.0),
            jnp.where(active, potential, 0.0),
            jnp.zeros_like(force),
            active,
            born,
            ruptured,
            jnp.abs(batch.gap),
            jnp.abs(rupture - batch.gap),
            bridge_source,
            bridge_release,
            bridge_residual,
            DEMCohesionHistory(()),
            finite,
        )
        return response, next_history


class NearContactLubricationPlan(AbstractDEMCohesionPlan):
    dynamic_viscosity: Array
    cutoff: Array
    minimum_gap: Array
    maximum_interaction_range: float = eqx.field(static=True)
    cohesion_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_viscosity: ArrayLike,
        cutoff: ArrayLike,
        minimum_gap: ArrayLike,
        /,
        *,
        cohesion_law_id: str | None = None,
    ):
        viscosity = np.asarray(dynamic_viscosity)
        cutoff_ = np.asarray(cutoff)
        minimum = np.asarray(minimum_gap)
        _validate_pair_parameter("dynamic_viscosity", viscosity, positive=True)
        _validate_pair_parameter("cutoff", cutoff_, positive=True)
        _validate_pair_parameter("minimum_gap", minimum, positive=True)
        _require_matching_schema(viscosity, cutoff_, minimum)
        if np.any(minimum >= cutoff_):
            raise ValueError("minimum_gap must be smaller than cutoff.")
        generated = canonical_fingerprint(
            {
                "kind": "near-contact-lubrication",
                "dynamic_viscosity": array_tree_fingerprint(viscosity),
                "cutoff": array_tree_fingerprint(cutoff_),
                "minimum_gap": array_tree_fingerprint(minimum),
            }
        )
        identifier = generated if cohesion_law_id is None else str(cohesion_law_id)
        if not identifier:
            raise ValueError("cohesion_law_id must be nonempty.")
        self.dynamic_viscosity = jnp.asarray(viscosity)
        self.cutoff = jnp.asarray(cutoff_)
        self.minimum_gap = jnp.asarray(minimum)
        self.maximum_interaction_range = float(np.max(cutoff_))
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return _empty_component_history(capacity, dtype)

    def evaluate(self, batch, normal, history, context, materials, /):
        del normal
        viscosity = _pair_value(
            self.dynamic_viscosity,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        cutoff = _pair_value(
            self.cutoff,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        minimum = _pair_value(
            self.minimum_gap,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        active = batch.valid & (batch.gap > 0.0) & (batch.gap < cutoff)
        denominator = jnp.maximum(batch.gap, minimum)
        coefficient = 6.0 * jnp.pi * viscosity * batch.effective_radius**2 / denominator
        force = jnp.where(active, -coefficient * batch.normal_velocity, 0.0)
        dissipated = jnp.where(
            active,
            coefficient * batch.normal_velocity**2 * context.step_size,
            0.0,
        )
        next_history = DEMCohesionComponentHistory(
            active,
            jnp.zeros_like(batch.gap),
            jnp.where(context.current_valid, batch.gap, 0.0),
            history.birth_step,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(dissipated))
            & jnp.all(dissipated >= 0.0)
        )
        response = DEMCohesionResponse(
            force,
            jnp.zeros_like(force),
            jnp.where(active, coefficient / denominator, 0.0),
            jnp.zeros_like(force),
            dissipated,
            active,
            active & ~history.active,
            history.active & ~active,
            jnp.abs(batch.gap),
            jnp.abs(cutoff - batch.gap),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            DEMCohesionHistory(()),
            finite,
        )
        return response, next_history


class CompositeDEMCohesionPlan(AbstractDEMCohesionPlan):
    components: tuple[AbstractDEMCohesionPlan, ...]
    maximum_interaction_range: float = eqx.field(static=True)
    cohesion_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[AbstractDEMCohesionPlan],
        /,
        *,
        cohesion_law_id: str | None = None,
    ):
        values = tuple(components)
        if not values or any(
            not isinstance(value, AbstractDEMCohesionPlan) for value in values
        ):
            raise TypeError("components must contain cohesion plans.")
        generated = canonical_fingerprint(
            {
                "kind": "composite-dem-cohesion",
                "components": [value.cohesion_law_id for value in values],
            }
        )
        identifier = generated if cohesion_law_id is None else str(cohesion_law_id)
        if not identifier:
            raise ValueError("cohesion_law_id must be nonempty.")
        self.components = values
        self.maximum_interaction_range = max(
            value.maximum_interaction_range for value in values
        )
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return DEMCohesionHistory(
            tuple(value.initialize_history(capacity, dtype) for value in self.components)
        )

    def evaluate(self, batch, normal, history, context, materials, /):
        if not isinstance(history, DEMCohesionHistory) or len(history.components) != len(
            self.components
        ):
            raise ValueError("Cohesion history does not match composite plan.")
        scalar = jnp.zeros_like(batch.gap)
        force = scalar
        friction = scalar
        stiffness = scalar
        energy = scalar
        dissipated = scalar
        active = jnp.zeros_like(batch.valid)
        born = jnp.zeros_like(batch.valid)
        ruptured = jnp.zeros_like(batch.valid)
        birth_margin = jnp.full_like(batch.gap, jnp.inf)
        rupture_margin = jnp.full_like(batch.gap, jnp.inf)
        bridge_source = scalar
        bridge_release = scalar
        bridge_residual = scalar
        successful = jnp.asarray(True)
        next_components = []
        for plan, component_history in zip(
            self.components, history.components, strict=True
        ):
            response, next_history = plan.evaluate(
                batch, normal, component_history, context, materials
            )
            force = force + response.force_magnitude
            friction = friction + response.friction_load
            stiffness = stiffness + response.stiffness
            energy = energy + response.elastic_energy
            dissipated = dissipated + response.dissipated_work
            active = active | response.active
            born = born | response.born
            ruptured = ruptured | response.ruptured
            birth_margin = jnp.minimum(birth_margin, response.birth_margin)
            rupture_margin = jnp.minimum(rupture_margin, response.rupture_margin)
            bridge_source = bridge_source + response.bridge_volume_source
            bridge_release = bridge_release + response.bridge_volume_release
            bridge_residual = bridge_residual + response.bridge_volume_residual
            successful = successful & response.successful
            next_components.append(next_history)
        combined_history = DEMCohesionHistory(tuple(next_components))
        return DEMCohesionResponse(
            force,
            friction,
            stiffness,
            energy,
            dissipated,
            active,
            born,
            ruptured,
            birth_margin,
            rupture_margin,
            bridge_source,
            bridge_release,
            bridge_residual,
            combined_history,
            successful,
        ), combined_history


def evaluate_dem_cohesion(
    plan: AbstractDEMCohesionPlan,
    batch: Any,
    normal: Any,
    history: DEMCohesionHistory,
    context: DEMContactEvaluationContext,
    materials: Any,
    /,
) -> DEMCohesionResponse:
    if isinstance(plan, CompositeDEMCohesionPlan):
        response, _ = plan.evaluate(batch, normal, history, context, materials)
        return response
    if not isinstance(history, DEMCohesionHistory) or len(history.components) != 1:
        raise ValueError("Cohesion history does not match prepared plan.")
    response, next_component = plan.evaluate(
        batch,
        normal,
        history.components[0],
        context,
        materials,
    )
    next_history = DEMCohesionHistory((next_component,))
    return eqx.tree_at(lambda value: value.next_history, response, next_history)


def zero_cohesion_response(
    shape: tuple[int, ...], dtype: Any, history: DEMCohesionHistory, /
) -> DEMCohesionResponse:
    scalar = jnp.zeros(shape, dtype=dtype)
    mask = jnp.zeros(shape, dtype=bool)
    return DEMCohesionResponse(
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        mask,
        mask,
        mask,
        jnp.full(shape, jnp.inf, dtype=dtype),
        jnp.full(shape, jnp.inf, dtype=dtype),
        scalar,
        scalar,
        scalar,
        history,
        jnp.asarray(True),
    )


def _empty_component_history(capacity: int, dtype: Any):
    count = int(capacity)
    if count < 0:
        raise ValueError("Cohesion history capacity must be nonnegative.")
    scalar = jnp.zeros((count,), dtype=dtype)
    mask = jnp.zeros((count,), dtype=bool)
    return DEMCohesionComponentHistory(
        mask,
        scalar,
        scalar,
        -jnp.ones((count,), dtype=jnp.int32),
    )


def _pair_value(parameter, left, right, material_count: int):
    if parameter.ndim == 0:
        return jnp.broadcast_to(parameter, left.shape)
    if parameter.shape != (material_count, material_count):
        raise ValueError("Cohesion pair table does not match material count.")
    return parameter[left, right]


def _validate_pair_parameter(
    name: str,
    value: np.ndarray,
    *,
    positive: bool = False,
    nonnegative: bool = False,
):
    if value.ndim not in (0, 2):
        raise ValueError(f"{name} must be scalar or a square pair table.")
    if value.ndim == 2 and (
        value.shape[0] != value.shape[1] or not np.array_equal(value, value.T)
    ):
        raise ValueError(f"{name} pair table must be square and symmetric.")
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(value <= 0.0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(value < 0.0):
        raise ValueError(f"{name} must be nonnegative.")


def _require_matching_schema(*values: np.ndarray):
    first = values[0]
    if any(
        value.ndim != first.ndim or value.shape != first.shape for value in values[1:]
    ):
        raise ValueError("Cohesion parameter schemas must match.")


__all__ = [
    "AbstractDEMCohesionPlan",
    "CompositeDEMCohesionPlan",
    "DEMCohesionComponentHistory",
    "DEMCohesionResponse",
    "DMTContactCohesionPlan",
    "LinearCapillaryBridgePlan",
    "NearContactLubricationPlan",
    "evaluate_dem_cohesion",
]
