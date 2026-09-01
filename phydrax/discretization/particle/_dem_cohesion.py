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
    bridge_surface_area: Array
    model_validity_margin: Array
    fit_extrapolation_margin: Array
    next_history: DEMCohesionHistory
    successful: Array


class AbstractDEMCohesionPlan(StrictModule, NonTrainableState):
    cohesion_law_id: AbstractAttribute[str]
    maximum_interaction_range: AbstractAttribute[float | None]

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

    @abc.abstractmethod
    def interaction_range_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def interaction_extents_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> np.ndarray:
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

    def interaction_range_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> float:
        del radii, material_ids, material_count
        return self.maximum_interaction_range

    def interaction_extents_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> np.ndarray:
        del material_ids, material_count
        return _static_interaction_extents(radii, self.maximum_interaction_range)

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
            jnp.zeros_like(force),
            jnp.full_like(force, jnp.inf),
            jnp.full_like(force, jnp.inf),
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

    def interaction_range_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> float:
        del radii, material_ids, material_count
        return self.maximum_interaction_range

    def interaction_extents_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> np.ndarray:
        del material_ids, material_count
        return _static_interaction_extents(radii, self.maximum_interaction_range)

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
            jnp.zeros_like(force),
            jnp.full_like(force, jnp.inf),
            jnp.full_like(force, jnp.inf),
            DEMCohesionHistory(()),
            finite,
        )
        return response, next_history


class BagheriCapillaryBridgePlan(AbstractDEMCohesionPlan):
    """Finite-volume capillary bridge fit with an analytic branch potential."""

    surface_tension: Array
    contact_angle: Array
    bridge_volume: Array
    conserve_liquid: bool = eqx.field(static=True)
    maximum_interaction_range: float | None = eqx.field(static=True)
    cohesion_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_tension: ArrayLike,
        contact_angle: ArrayLike,
        bridge_volume: ArrayLike,
        /,
        *,
        conserve_liquid: bool = False,
        cohesion_law_id: str | None = None,
    ):
        tension = np.asarray(surface_tension)
        angle = np.asarray(contact_angle)
        volume = np.asarray(bridge_volume)
        _validate_pair_parameter("surface_tension", tension, positive=True)
        _validate_pair_parameter("contact_angle", angle, nonnegative=True)
        _validate_pair_parameter("bridge_volume", volume, positive=True)
        _require_matching_schema(tension, angle, volume)
        maximum_angle = np.deg2rad(50.0)
        if np.any(angle > maximum_angle):
            raise ValueError("contact_angle must lie in the fitted interval [0, 50 deg].")
        conserved = bool(conserve_liquid)
        generated = canonical_fingerprint(
            {
                "kind": "bagheri-capillary-bridge",
                "surface_tension": array_tree_fingerprint(tension),
                "contact_angle": array_tree_fingerprint(angle),
                "bridge_volume": array_tree_fingerprint(volume),
                "conserve_liquid": conserved,
            }
        )
        identifier = generated if cohesion_law_id is None else str(cohesion_law_id)
        if not identifier:
            raise ValueError("cohesion_law_id must be nonempty.")
        self.surface_tension = jnp.asarray(tension)
        self.contact_angle = jnp.asarray(angle)
        self.bridge_volume = jnp.asarray(volume)
        self.conserve_liquid = conserved
        self.maximum_interaction_range = None
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return _empty_component_history(capacity, dtype)

    def pair_bridge_volume(
        self,
        left_material: Array,
        right_material: Array,
        material_count: int,
        /,
    ) -> Array:
        return _pair_value(
            self.bridge_volume,
            left_material,
            right_material,
            material_count,
        )

    def interaction_extents_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> np.ndarray:
        radius = np.asarray(radii, dtype=float)
        material = np.asarray(material_ids)
        count = int(material_count)
        if (
            radius.ndim != 1
            or material.shape != radius.shape
            or np.any(~np.isfinite(radius))
            or np.any(radius <= 0.0)
            or not np.issubdtype(material.dtype, np.integer)
            or count <= 0
            or np.any((material < 0) | (material >= count))
        ):
            raise ValueError("Radius-dependent capillary envelopes are invalid.")
        volume = _host_pair_table(self.bridge_volume, count)
        angle = _host_pair_table(self.contact_angle, count)
        present = np.unique(material)
        minimum_radius = np.full((count,), np.inf)
        for material_id in present:
            minimum_radius[material_id] = np.min(radius[material == material_id])
        additional = np.zeros_like(radius)
        for index, (body_radius, body_material) in enumerate(
            zip(radius, material, strict=True)
        ):
            reach = 0.0
            for partner_material in present:
                partner_radius = minimum_radius[partner_material]
                characteristic = (
                    2.0 * body_radius * partner_radius / (body_radius + partner_radius)
                )
                pair_volume = volume[body_material, partner_material]
                dimensionless = pair_volume / characteristic**3
                critical = (
                    characteristic
                    * (1.0 + 0.5 * angle[body_material, partner_material])
                    * (dimensionless ** (1.0 / 3.0) + 0.1 * dimensionless ** (2.0 / 3.0))
                )
                reach = max(reach, float(critical))
            additional[index] = 0.5 * reach
        return radius + additional

    def interaction_range_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> float:
        radius = np.asarray(radii, dtype=float)
        extent = self.interaction_extents_for_radii(radius, material_ids, material_count)
        return 2.0 * float(np.max(extent - radius))

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
        requested_volume = _pair_value(
            self.bridge_volume,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        allocated_volume = (
            history.bridge_volume if self.conserve_liquid else requested_volume
        )
        prospective_volume = jnp.where(
            history.active, history.bridge_volume, allocated_volume
        )
        characteristic_radius = 2.0 * batch.effective_radius
        safe_radius = jnp.where(characteristic_radius > 0.0, characteristic_radius, 1.0)
        safe_volume = jnp.where(
            prospective_volume > 0.0,
            prospective_volume,
            1.0e-6 * safe_radius**3,
        )
        raw_dimensionless_volume = safe_volume / safe_radius**3
        dimensionless_volume = jnp.clip(raw_dimensionless_volume, 1.0e-6, 1.0e-1)
        maximum_angle = jnp.asarray(np.deg2rad(50.0), dtype=batch.gap.dtype)
        safe_angle = jnp.clip(angle, 0.0, maximum_angle)
        critical_distance = (
            safe_radius
            * (1.0 + 0.5 * safe_angle)
            * (
                dimensionless_volume ** (1.0 / 3.0)
                + 0.1 * dimensionless_volume ** (2.0 / 3.0)
            )
        )
        safe_critical = jnp.where(critical_distance > 0.0, critical_distance, 1.0)
        normalized_separation = jnp.maximum(batch.gap, 0.0) / safe_critical
        fit_volume_valid = (raw_dimensionless_volume >= 1.0e-6) & (
            raw_dimensionless_volume <= 1.0e-1
        )
        fit_angle_valid = (angle >= 0.0) & (angle <= maximum_angle)
        domain_valid = (
            (prospective_volume > 0.0)
            & (characteristic_radius > 0.0)
            & fit_volume_valid
            & fit_angle_valid
        )
        prospective_birth = (
            batch.valid & (batch.gap <= 0.0) & ~history.active & (allocated_volume > 0.0)
        )
        born = prospective_birth & domain_valid
        retained = (
            history.active & batch.valid & domain_valid & (batch.gap < critical_distance)
        )
        active = born | retained
        ruptured = history.active & ~active
        bridge_volume = jnp.where(born, allocated_volume, history.bridge_volume)
        bridge_volume = jnp.where(active, bridge_volume, 0.0)
        separation = jnp.clip(normalized_separation, 0.0, 1.0)

        volume_log = jnp.log(dimensionless_volume)
        a_separation = (
            -0.3319 * dimensionless_volume**0.4974 + 0.6717 * dimensionless_volume**0.1995
        )
        b_separation = 13.84 * dimensionless_volume ** (
            -0.3909
        ) - 12.11 * dimensionless_volume ** (-0.3945)
        a_contact = 0.4158 * dimensionless_volume**0.2835 + 0.6474
        b_contact = -0.2087 * dimensionless_volume**0.3113 + 2.267
        contact_scale = 1.0 - a_contact * jnp.sin(safe_angle) ** b_contact
        a_angle = -0.007815 * volume_log**2 - 0.2105 * volume_log - 1.426
        b_angle = (
            -1.78 * dimensionless_volume** 0.8351
            + 0.6669 * dimensionless_volume ** (-0.01391)
        )
        c_angle = a_angle * safe_angle**3 + b_angle * safe_angle + 1.0
        q = c_angle * b_separation
        denominator = 1.0 + a_separation * q * separation + q * separation**2
        numerator = 1.0 + a_separation * separation
        shape = numerator / denominator
        contact_force = (
            2.0
            * jnp.pi
            * tension
            * safe_radius
            * (1.0 - 0.3823 * dimensionless_volume**0.2586)
            * contact_scale
        )
        force = jnp.where(active, -contact_force * shape, 0.0)
        shape_derivative = (
            a_separation * denominator
            - numerator * (a_separation * q + 2.0 * q * separation)
        ) / denominator**2
        stiffness = jnp.where(
            active & (batch.gap > 0.0),
            -contact_force * shape_derivative / safe_critical,
            0.0,
        )

        discriminant = 4.0 * q - (a_separation * q) ** 2
        safe_discriminant = jnp.maximum(discriminant, jnp.finfo(batch.gap.dtype).tiny)
        root = jnp.sqrt(safe_discriminant)

        def primitive(value):
            polynomial = 1.0 + a_separation * q * value + q * value**2
            return a_separation / (2.0 * q) * jnp.log(polynomial) + 2.0 * (
                1.0 - 0.5 * a_separation**2
            ) / root * jnp.arctan((2.0 * q * value + a_separation * q) / root)

        contact_potential = (
            -contact_force
            * safe_critical
            * (primitive(jnp.ones_like(separation)) - primitive(separation))
        )
        zero_gap_potential = (
            -contact_force
            * safe_critical
            * (
                primitive(jnp.ones_like(separation))
                - primitive(jnp.zeros_like(separation))
            )
        )
        potential = jnp.where(
            batch.gap <= 0.0,
            zero_gap_potential + contact_force * batch.gap,
            contact_potential,
        )
        surface_area = bagheri_capillary_bridge_surface_area(
            safe_radius,
            batch.gap,
            safe_volume,
            safe_angle,
        )
        surface_area = jnp.where(active, surface_area, 0.0)

        birth_step = jnp.where(born, context.step_index, history.birth_step)
        birth_step = jnp.where(active, birth_step, -1)
        next_history = DEMCohesionComponentHistory(
            active,
            bridge_volume,
            jnp.where(context.current_valid, batch.gap, 0.0),
            birth_step.astype(jnp.int32),
        )
        previous_bridge_volume = jnp.where(history.active, history.bridge_volume, 0.0)
        bridge_source = jnp.where(born, allocated_volume, 0.0)
        bridge_release = jnp.where(ruptured, previous_bridge_volume, 0.0)
        bridge_residual = (
            bridge_volume - previous_bridge_volume - bridge_source + bridge_release
        )
        volume_margin = jnp.minimum(
            jnp.log(raw_dimensionless_volume / 1.0e-6),
            jnp.log(1.0e-1 / raw_dimensionless_volume),
        )
        angle_margin = jnp.minimum(
            safe_angle / maximum_angle,
            (maximum_angle - safe_angle) / maximum_angle,
        )
        model_margin = jnp.minimum(volume_margin, angle_margin)
        relevant = prospective_birth | history.active
        model_margin = jnp.where(relevant, model_margin, jnp.inf)
        extrapolation_margin = jnp.where(active, 0.9 - separation, jnp.inf)
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(stiffness))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(surface_area))
            & jnp.all(discriminant > 0.0)
            & jnp.all(
                jnp.abs(bridge_residual)
                <= 64.0
                * jnp.finfo(bridge_residual.dtype).eps
                * jnp.maximum(jnp.abs(prospective_volume), 1.0)
            )
        )
        successful = finite & jnp.all(~relevant | domain_valid)
        response = DEMCohesionResponse(
            force,
            jnp.abs(force),
            stiffness,
            jnp.where(active, potential, 0.0),
            jnp.zeros_like(force),
            active,
            born,
            ruptured,
            jnp.abs(batch.gap),
            jnp.abs(critical_distance - batch.gap),
            bridge_source,
            bridge_release,
            bridge_residual,
            surface_area,
            model_margin,
            extrapolation_margin,
            DEMCohesionHistory(()),
            successful,
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

    def interaction_range_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> float:
        del radii, material_ids, material_count
        return self.maximum_interaction_range

    def interaction_extents_for_radii(
        self, radii, material_ids, material_count: int, /
    ) -> np.ndarray:
        del material_ids, material_count
        return _static_interaction_extents(radii, self.maximum_interaction_range)

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
            jnp.zeros_like(force),
            jnp.full_like(force, jnp.inf),
            jnp.full_like(force, jnp.inf),
            DEMCohesionHistory(()),
            finite,
        )
        return response, next_history


class CompositeDEMCohesionPlan(AbstractDEMCohesionPlan):
    components: tuple[AbstractDEMCohesionPlan, ...]
    maximum_interaction_range: float | None = eqx.field(static=True)
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
        static_ranges = tuple(
            value.maximum_interaction_range
            for value in values
            if value.maximum_interaction_range is not None
        )
        self.maximum_interaction_range = (
            None if len(static_ranges) != len(values) else max(static_ranges)
        )
        self.cohesion_law_id = identifier

    def initialize_history(self, capacity: int, dtype: Any, /):
        return DEMCohesionHistory(
            tuple(value.initialize_history(capacity, dtype) for value in self.components)
        )

    def interaction_range_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> float:
        return max(
            component.interaction_range_for_radii(radii, material_ids, material_count)
            for component in self.components
        )

    def interaction_extents_for_radii(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        material_count: int,
        /,
    ) -> np.ndarray:
        extents = tuple(
            component.interaction_extents_for_radii(radii, material_ids, material_count)
            for component in self.components
        )
        return np.maximum.reduce(extents)

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
        bridge_area = scalar
        model_margin = jnp.full_like(batch.gap, jnp.inf)
        extrapolation_margin = jnp.full_like(batch.gap, jnp.inf)
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
            bridge_area = bridge_area + response.bridge_surface_area
            model_margin = jnp.minimum(model_margin, response.model_validity_margin)
            extrapolation_margin = jnp.minimum(
                extrapolation_margin, response.fit_extrapolation_margin
            )
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
            bridge_area,
            model_margin,
            extrapolation_margin,
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
        scalar,
        jnp.full(shape, jnp.inf, dtype=dtype),
        jnp.full(shape, jnp.inf, dtype=dtype),
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


def _static_interaction_extents(radii: ArrayLike, maximum_range: float, /) -> np.ndarray:
    radius = np.asarray(radii, dtype=float)
    if radius.ndim != 1 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Interaction-envelope radii must be finite and positive.")
    return radius + 0.5 * float(maximum_range)


def _pair_value(parameter, left, right, material_count: int):
    if parameter.ndim == 0:
        return jnp.broadcast_to(parameter, left.shape)
    if parameter.shape != (material_count, material_count):
        raise ValueError("Cohesion pair table does not match material count.")
    return parameter[left, right]


def _host_pair_table(parameter: ArrayLike, material_count: int, /) -> np.ndarray:
    value = np.asarray(parameter, dtype=float)
    count = int(material_count)
    if count <= 0:
        raise ValueError("material_count must be positive.")
    if value.ndim == 0:
        return np.full((count, count), float(value))
    if value.shape != (count, count):
        raise ValueError("Cohesion pair table does not match material count.")
    return value


def bagheri_capillary_bridge_surface_area(
    characteristic_radius: ArrayLike,
    gap: ArrayLike,
    bridge_volume: ArrayLike,
    contact_angle: ArrayLike,
    /,
) -> Array:
    """Evaluate the fitted exposed liquid area for one or more bridge states."""

    radius = jnp.asarray(characteristic_radius)
    separation_gap = jnp.asarray(gap, dtype=radius.dtype)
    volume = jnp.asarray(bridge_volume, dtype=radius.dtype)
    angle = jnp.asarray(contact_angle, dtype=radius.dtype)
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    safe_volume = jnp.where(volume > 0.0, volume, 1.0e-6 * safe_radius**3)
    dimensionless_volume = jnp.clip(safe_volume / safe_radius**3, 1.0e-6, 1.0e-1)
    maximum_angle = jnp.asarray(np.deg2rad(50.0), dtype=radius.dtype)
    theta = jnp.clip(angle, 0.0, maximum_angle)
    log_volume = jnp.log(dimensionless_volume)
    critical = (
        safe_radius
        * (1.0 + 0.5 * theta)
        * (
            dimensionless_volume ** (1.0 / 3.0)
            + 0.1 * dimensionless_volume ** (2.0 / 3.0)
        )
    )
    normalized_separation = jnp.clip(
        jnp.maximum(separation_gap, 0.0) / critical, 0.0, 1.0
    )

    zero_angle_contact = jnp.exp(0.000734 * log_volume**2 + 0.771 * log_volume + 2.107)
    contact_a = (
        0.5615 * dimensionless_volume ** (-0.00836)
        + 0.7813 * dimensionless_volume**0.2046
    )
    contact_b = (
        -0.09168 * dimensionless_volume ** (-0.02997)
        - 0.4159 * dimensionless_volume**0.1933
    )
    contact_area = zero_angle_contact / (1.0 + contact_a * theta + contact_b * theta**2)

    separation_a = 0.00539 * dimensionless_volume ** (
        -0.2332
    ) + 0.00029 * dimensionless_volume ** (-0.0971)
    separation_b = 0.000057 * dimensionless_volume ** (
        -0.5419
    ) + 0.01318 * dimensionless_volume ** (-0.2780)
    separation_c = (
        -0.003518 * dimensionless_volume** 0.0237
        + 0.00123 * dimensionless_volume ** (-0.2918)
    )
    separation_d = -0.003147 * dimensionless_volume ** (
        -0.0244
    ) + 0.00475 * dimensionless_volume ** (-0.2453)
    angle_e_a = (
        0.0000298 * log_volume**4
        + 0.00111 * log_volume**3
        + 0.01421 * log_volume**2
        + 0.05821 * log_volume
        + 0.2012
    )
    angle_e_b = (
        0.0000877 * log_volume**4
        + 0.00305 * log_volume**3
        + 0.03849 * log_volume**2
        + 0.2521 * log_volume
        + 0.243
    )
    angle_f_a = (
        -0.9185 * dimensionless_volume**0.0612 - 11.46 * dimensionless_volume**0.8370
    )
    angle_f_b = 3.078 * dimensionless_volume**0.09386 + 12.77 * dimensionless_volume**0.68
    angle_g_a = (
        -0.9588 * dimensionless_volume**0.0607 - 7.343 * dimensionless_volume**0.818
    )
    angle_g_b = (
        3.119 * dimensionless_volume**0.07801 + 7.643 * dimensionless_volume**0.5406
    )
    angle_e = (angle_e_a * theta**2 + angle_e_a * theta - angle_e_b) / (theta - angle_e_b)
    angle_f = angle_f_a * theta**2 + angle_f_b * theta + 1.0
    angle_g = angle_g_a * theta**2 + angle_g_b * theta + 1.0
    scaled = normalized_separation
    numerator = (
        separation_a * scaled**3
        - angle_e * angle_f * (1.0 + theta) * separation_b * scaled**2
        - separation_c * scaled
        - 0.01
    )
    denominator = (
        separation_a * scaled**4
        - angle_f * (1.0 + theta) * separation_d * scaled**2
        - angle_e * angle_g * separation_a * scaled
        - 0.01
    )
    return safe_radius**2 * contact_area * numerator / denominator


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
    "BagheriCapillaryBridgePlan",
    "CompositeDEMCohesionPlan",
    "DEMCohesionComponentHistory",
    "DEMCohesionResponse",
    "DMTContactCohesionPlan",
    "LinearCapillaryBridgePlan",
    "NearContactLubricationPlan",
    "bagheri_capillary_bridge_surface_area",
    "evaluate_dem_cohesion",
]
