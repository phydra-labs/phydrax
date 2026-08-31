#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._dem_cohesion import (
    AbstractDEMCohesionPlan,
    evaluate_dem_cohesion,
    zero_cohesion_response,
)
from ._dem_contact_state import (
    DEMCohesionHistory,
    DEMContactEvaluationContext,
    DEMContactHistory,
    DEMNormalHistory,
    DEMRotationalHistory,
    DEMTangentialHistory,
)
from ._rigid_sphere import sphere_lever_torque


class DEMContactBatch(StrictModule):
    """Contact geometry and kinematics aligned to one fixed-capacity route batch."""

    normal: Array
    gap: Array
    overlap: Array
    effective_radius: Array
    left_arm: Array
    right_arm: Array
    normal_velocity: Array
    tangential_velocity: Array
    left_angular_velocity: Array
    right_angular_velocity: Array
    valid: Array


class DEMNormalResponse(StrictModule):
    force_magnitude: Array
    friction_load: Array
    stiffness: Array
    damping: Array
    no_tension_margin: Array
    elastic_energy: Array
    viscous_endpoint_loss: Array
    plastic_dissipated_work: Array
    next_maximum_overlap: Array
    next_plastic_overlap: Array
    next_previous_overlap: Array
    active: Array
    successful: Array


class DEMTangentialResponse(StrictModule):
    force: Array
    displacement: Array
    elastic_energy: Array
    constitutive_loss_estimate: Array
    sticking: Array
    sliding: Array
    friction_defect: Array
    switch_margin: Array
    successful: Array


class DEMRotationalResponse(StrictModule):
    left_torque: Array
    right_torque: Array
    rolling_torque_left: Array
    rolling_torque_right: Array
    torsional_torque_left: Array
    torsional_torque_right: Array
    elastic_energy: Array
    dissipated_work: Array
    rolling_dissipated_work: Array
    torsional_dissipated_work: Array
    rolling_yielded: Array
    torsional_yielded: Array
    rolling_margin: Array
    torsional_margin: Array
    next_history: DEMRotationalHistory
    active: Array
    successful: Array


class DEMContactResponse(StrictModule):
    pair_force: Array
    left_torque: Array
    right_torque: Array
    next_history: DEMContactHistory
    normal_force: Array
    tangential_force: Array
    cohesion_force: Array
    active: Array
    sticking: Array
    sliding: Array
    cohesion_births: Array
    cohesion_ruptures: Array
    bridge_volume_source: Array
    bridge_volume_release: Array
    bridge_volume_residual: Array
    rolling_yielded: Array
    torsional_yielded: Array
    elastic_energy: Array
    normal_viscous_endpoint_loss: Array
    rotational_torque_left: Array
    rotational_torque_right: Array
    normal_plastic_dissipated_work: Array
    rotational_dissipated_work: Array
    cohesion_dissipated_work: Array
    tangential_constitutive_loss_estimate: Array
    friction_defect: Array
    switch_margin: Array
    activation_margin: Array
    cohesion_birth_margin: Array
    cohesion_rupture_margin: Array
    no_tension_margin: Array
    frame_transport_margin: Array
    rolling_yield_margin: Array
    torsional_yield_margin: Array
    maximum_overlap_fraction: Array
    successful: Array


class AbstractDEMNormalContactPlan(StrictModule, NonTrainableState):
    normal_law_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        batch: DEMContactBatch,
        previous_history: DEMContactHistory,
        left_inverse_mass: Array,
        right_inverse_mass: Array,
        left_radius: Array,
        right_radius: Array,
        left_material: Array,
        right_material: Array,
        materials: Any,
        step_size: Array,
        /,
    ) -> DEMNormalResponse:
        raise NotImplementedError


def _unilateral_restitution_from_ratio(ratio: Array, /) -> Array:
    ratio_ = jnp.clip(ratio, 0.0, 0.999999)
    root = jnp.sqrt(jnp.maximum(1.0 - ratio_**2, 1.0e-12))
    angle = jnp.arctan2(-2.0 * ratio_ * root, 1.0 - 2.0 * ratio_**2)
    angle = jnp.where(angle <= 0.0, angle + jnp.pi, angle)
    velocity = jnp.cos(angle) - ratio_ / root * jnp.sin(angle)
    return -jnp.exp(-ratio_ * angle / root) * velocity


@jax.custom_jvp
def _unilateral_damping_ratio(restitution: Array, /) -> Array:
    target = jnp.clip(restitution, 1.0e-8, 1.0)

    def iteration(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        value = _unilateral_restitution_from_ratio(midpoint)
        lower = jnp.where(value > target, midpoint, lower)
        upper = jnp.where(value > target, upper, midpoint)
        return lower, upper

    lower, upper = jax.lax.fori_loop(
        0,
        48,
        iteration,
        (jnp.zeros_like(target), jnp.full_like(target, 0.999)),
    )
    return 0.5 * (lower + upper)


@_unilateral_damping_ratio.defjvp
def _unilateral_damping_ratio_jvp(primals, tangents):
    (restitution,) = primals
    (restitution_tangent,) = tangents
    ratio = _unilateral_damping_ratio(restitution)
    _, slope = jax.jvp(
        _unilateral_restitution_from_ratio,
        (ratio,),
        (jnp.ones_like(ratio),),
    )
    return ratio, restitution_tangent / slope


class LinearSpringDashpotNormalPlan(AbstractDEMNormalContactPlan):
    stiffness: Array
    normal_law_id: str = eqx.field(static=True)

    def __init__(self, stiffness: ArrayLike, /, *, normal_law_id: str | None = None):
        values = np.asarray(stiffness)
        if values.ndim not in (0, 2):
            raise ValueError("Linear normal stiffness must be scalar or a square table.")
        if values.ndim == 2 and values.shape[0] != values.shape[1]:
            raise ValueError("Linear normal stiffness table must be square.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Linear normal stiffness must be finite and positive.")
        if values.ndim == 2 and not np.array_equal(values, values.T):
            raise ValueError("Linear normal stiffness table must be symmetric.")
        generated = canonical_fingerprint(
            {
                "kind": "linear-spring-dashpot-normal",
                "stiffness": array_tree_fingerprint(values),
            }
        )
        identifier = generated if normal_law_id is None else str(normal_law_id)
        if not identifier:
            raise ValueError("normal_law_id must be nonempty.")
        self.stiffness = jnp.asarray(values)
        self.normal_law_id = identifier

    def evaluate(
        self,
        batch,
        previous_history,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        del left_radius, right_radius, previous_history
        stiffness = _pair_parameter(
            self.stiffness, left_material, right_material, materials.material_count
        ).astype(batch.overlap.dtype)
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        damping_ratio = _unilateral_damping_ratio(
            materials.restitution.astype(batch.overlap.dtype)
        )[left_material, right_material]
        damping = 2.0 * damping_ratio * jnp.sqrt(effective_mass * stiffness)
        active = (
            batch.valid
            & (batch.overlap > 0.0)
            & (left_inverse_mass + right_inverse_mass > 0.0)
        )
        elastic = stiffness * batch.overlap
        trial = elastic - damping * batch.normal_velocity
        force = jnp.where(active, jnp.maximum(trial, 0.0), 0.0)
        energy = jnp.where(active, 0.5 * stiffness * batch.overlap**2, 0.0)
        dissipated = jnp.where(
            active,
            damping * batch.normal_velocity**2 * jnp.asarray(step_size),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(dissipated))
        )
        return DEMNormalResponse(
            force,
            force,
            stiffness,
            damping,
            jnp.abs(trial),
            energy,
            jnp.maximum(dissipated, 0.0),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.where(active, batch.overlap, 0.0),
            active,
            finite,
        )


class HertzNormalContactPlan(AbstractDEMNormalContactPlan):
    normal_law_id: str = eqx.field(static=True)

    def __init__(self, *, normal_law_id: str | None = None):
        generated = canonical_fingerprint({"kind": "hertz-normal-tsuji-damping"})
        identifier = generated if normal_law_id is None else str(normal_law_id)
        if not identifier:
            raise ValueError("normal_law_id must be nonempty.")
        self.normal_law_id = identifier

    def evaluate(
        self,
        batch,
        previous_history,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        del previous_history
        del left_radius, right_radius
        effective_radius = batch.effective_radius
        effective_young = materials.effective_young_modulus(
            left_material, right_material
        ).astype(batch.overlap.dtype)
        root = jnp.sqrt(jnp.maximum(effective_radius * batch.overlap, 0.0))
        elastic = (4.0 / 3.0) * effective_young * root * batch.overlap
        tangent_stiffness = 2.0 * effective_young * root
        restitution = materials.pair_restitution(left_material, right_material).astype(
            batch.overlap.dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = (
            -2.0
            * jnp.sqrt(5.0 / 6.0)
            * beta
            * jnp.sqrt(effective_mass * tangent_stiffness)
        )
        active = (
            batch.valid
            & (batch.overlap > 0.0)
            & (left_inverse_mass + right_inverse_mass > 0.0)
        )
        trial = elastic - damping * batch.normal_velocity
        force = jnp.where(active, jnp.maximum(trial, 0.0), 0.0)
        energy = jnp.where(
            active,
            (8.0 / 15.0)
            * effective_young
            * jnp.sqrt(jnp.maximum(effective_radius, 0.0))
            * batch.overlap**2.5,
            0.0,
        )
        dissipated = jnp.where(
            active,
            damping * batch.normal_velocity**2 * jnp.asarray(step_size),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(dissipated))
        )
        return DEMNormalResponse(
            force,
            force,
            tangent_stiffness,
            damping,
            jnp.abs(trial),
            energy,
            jnp.maximum(dissipated, 0.0),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.where(active, batch.overlap, 0.0),
            active,
            finite,
        )


class ThorntonLinearPlasticNormalPlan(AbstractDEMNormalContactPlan):
    """Bilinear elasto-plastic loading with elastic unloading history."""

    loading_stiffness: Array
    unloading_stiffness: Array
    hardening_stiffness: Array
    yield_overlap: Array
    normal_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        loading_stiffness: ArrayLike,
        unloading_stiffness: ArrayLike,
        hardening_stiffness: ArrayLike,
        yield_overlap: ArrayLike,
        /,
        *,
        normal_law_id: str | None = None,
    ):
        arrays = tuple(
            np.asarray(value)
            for value in (
                loading_stiffness,
                unloading_stiffness,
                hardening_stiffness,
                yield_overlap,
            )
        )
        shape = arrays[0].shape
        if arrays[0].ndim not in (0, 2) or any(
            value.shape != shape for value in arrays[1:]
        ):
            raise ValueError(
                "Plastic-law parameters must share a scalar or pair-table schema."
            )
        if arrays[0].ndim == 2 and (
            arrays[0].shape[0] != arrays[0].shape[1]
            or any(not np.array_equal(value, value.T) for value in arrays)
        ):
            raise ValueError("Plastic-law pair tables must be square and symmetric.")
        loading, unloading, hardening, yield_ = arrays
        if (
            np.any(~np.isfinite(loading))
            or np.any(~np.isfinite(unloading))
            or np.any(~np.isfinite(hardening))
            or np.any(~np.isfinite(yield_))
            or np.any(loading <= 0.0)
            or np.any(unloading <= loading)
            or np.any(hardening < 0.0)
            or np.any(hardening > loading)
            or np.any(yield_ <= 0.0)
        ):
            raise ValueError(
                "Plastic stiffnesses require unloading > loading > 0, "
                "0 <= hardening <= loading, and positive yield overlap."
            )
        generated = canonical_fingerprint(
            {
                "kind": "thornton-linear-plastic-normal",
                "loading": array_tree_fingerprint(loading),
                "unloading": array_tree_fingerprint(unloading),
                "hardening": array_tree_fingerprint(hardening),
                "yield_overlap": array_tree_fingerprint(yield_),
            }
        )
        identifier = generated if normal_law_id is None else str(normal_law_id)
        if not identifier:
            raise ValueError("normal_law_id must be nonempty.")
        self.loading_stiffness = jnp.asarray(loading)
        self.unloading_stiffness = jnp.asarray(unloading)
        self.hardening_stiffness = jnp.asarray(hardening)
        self.yield_overlap = jnp.asarray(yield_)
        self.normal_law_id = identifier

    def evaluate(
        self,
        batch,
        previous_history,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        del left_radius, right_radius
        dtype = batch.overlap.dtype
        count = materials.material_count
        loading_stiffness = _pair_parameter(
            self.loading_stiffness, left_material, right_material, count
        ).astype(dtype)
        unloading_stiffness = _pair_parameter(
            self.unloading_stiffness, left_material, right_material, count
        ).astype(dtype)
        hardening = _pair_parameter(
            self.hardening_stiffness, left_material, right_material, count
        ).astype(dtype)
        yield_overlap = _pair_parameter(
            self.yield_overlap, left_material, right_material, count
        ).astype(dtype)
        overlap = batch.overlap
        previous_maximum = previous_history.normal.maximum_overlap
        previous_plastic = previous_history.normal.plastic_overlap
        previous_overlap = previous_history.normal.previous_overlap
        maximum = jnp.maximum(previous_maximum, overlap)
        yield_force = loading_stiffness * yield_overlap
        maximum_force = jnp.where(
            maximum <= yield_overlap,
            loading_stiffness * maximum,
            yield_force + hardening * (maximum - yield_overlap),
        )
        plastic_overlap = jnp.where(
            maximum <= yield_overlap,
            0.0,
            maximum - maximum_force / unloading_stiffness,
        )
        loading_force = jnp.where(
            overlap <= yield_overlap,
            loading_stiffness * overlap,
            yield_force + hardening * (overlap - yield_overlap),
        )
        unloading_force = unloading_stiffness * jnp.maximum(
            overlap - plastic_overlap, 0.0
        )
        loading_branch = overlap >= previous_maximum
        elastic_force = jnp.where(
            loading_branch,
            loading_force,
            jnp.minimum(unloading_force, loading_force),
        )
        current_stiffness = jnp.where(
            loading_branch,
            jnp.where(overlap <= yield_overlap, loading_stiffness, hardening),
            unloading_stiffness,
        )
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        restitution = materials.pair_restitution(left_material, right_material).astype(
            dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = (
            -2.0 * beta * jnp.sqrt(effective_mass * jnp.maximum(current_stiffness, 0.0))
        )
        active = (
            batch.valid & (overlap > 0.0) & (left_inverse_mass + right_inverse_mass > 0.0)
        )
        trial = elastic_force - damping * batch.normal_velocity
        force = jnp.where(active, jnp.maximum(trial, 0.0), 0.0)
        energy = jnp.where(
            active,
            0.5
            * jnp.where(
                maximum <= yield_overlap,
                loading_stiffness * overlap**2,
                unloading_stiffness * (overlap - plastic_overlap) ** 2,
            ),
            0.0,
        )
        previous_force = jnp.where(
            previous_maximum <= yield_overlap,
            loading_stiffness * previous_overlap,
            unloading_stiffness * jnp.maximum(previous_overlap - previous_plastic, 0.0),
        )
        previous_energy = 0.5 * jnp.where(
            previous_maximum <= yield_overlap,
            loading_stiffness * previous_overlap**2,
            unloading_stiffness * (previous_overlap - previous_plastic) ** 2,
        )
        compression_work = (
            0.5 * (previous_force + elastic_force) * (overlap - previous_overlap)
        )
        plastic_loss = jnp.where(
            active,
            jnp.maximum(compression_work - (energy - previous_energy), 0.0),
            0.0,
        )
        viscous_loss = jnp.where(
            active,
            damping * batch.normal_velocity**2 * jnp.asarray(step_size),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(plastic_loss))
        )
        return DEMNormalResponse(
            force,
            force,
            current_stiffness,
            damping,
            jnp.abs(trial),
            energy,
            jnp.maximum(viscous_loss, 0.0),
            plastic_loss,
            jnp.where(active, maximum, 0.0),
            jnp.where(active, plastic_overlap, 0.0),
            jnp.where(active, overlap, 0.0),
            active,
            finite,
        )


class AbstractDEMTangentialContactPlan(StrictModule, NonTrainableState):
    tangential_law_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        batch: DEMContactBatch,
        normal: DEMNormalResponse,
        transported_displacement: Array,
        left_inverse_mass: Array,
        right_inverse_mass: Array,
        left_radius: Array,
        right_radius: Array,
        left_material: Array,
        right_material: Array,
        materials: Any,
        step_size: Array,
        /,
    ) -> DEMTangentialResponse:
        raise NotImplementedError


class CundallStrackTangentialPlan(AbstractDEMTangentialContactPlan):
    stiffness: Array
    tangential_law_id: str = eqx.field(static=True)

    def __init__(self, stiffness: ArrayLike, /, *, tangential_law_id: str | None = None):
        values = np.asarray(stiffness)
        if values.ndim not in (0, 2):
            raise ValueError("Tangential stiffness must be scalar or a square table.")
        if values.ndim == 2 and values.shape[0] != values.shape[1]:
            raise ValueError("Tangential stiffness table must be square.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Tangential stiffness must be finite and positive.")
        if values.ndim == 2 and not np.array_equal(values, values.T):
            raise ValueError("Tangential stiffness table must be symmetric.")
        generated = canonical_fingerprint(
            {
                "kind": "cundall-strack-tangential",
                "stiffness": array_tree_fingerprint(values),
            }
        )
        identifier = generated if tangential_law_id is None else str(tangential_law_id)
        if not identifier:
            raise ValueError("tangential_law_id must be nonempty.")
        self.stiffness = jnp.asarray(values)
        self.tangential_law_id = identifier

    def evaluate(
        self,
        batch,
        normal,
        transported_displacement,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        del left_radius, right_radius
        stiffness = _pair_parameter(
            self.stiffness, left_material, right_material, materials.material_count
        ).astype(batch.overlap.dtype)
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        restitution = materials.pair_restitution(left_material, right_material).astype(
            batch.overlap.dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = -2.0 * beta * jnp.sqrt(effective_mass * stiffness)
        return _tangential_response(
            batch,
            normal,
            transported_displacement,
            stiffness,
            damping,
            materials.pair_friction(left_material, right_material),
            step_size,
        )


class MindlinTangentialContactPlan(AbstractDEMTangentialContactPlan):
    tangential_law_id: str = eqx.field(static=True)

    def __init__(self, *, tangential_law_id: str | None = None):
        generated = canonical_fingerprint({"kind": "mindlin-tangential-tsuji-damping"})
        identifier = generated if tangential_law_id is None else str(tangential_law_id)
        if not identifier:
            raise ValueError("tangential_law_id must be nonempty.")
        self.tangential_law_id = identifier

    def evaluate(
        self,
        batch,
        normal,
        transported_displacement,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        del left_radius, right_radius
        effective_radius = batch.effective_radius
        shear = materials.effective_shear_modulus(left_material, right_material).astype(
            batch.overlap.dtype
        )
        stiffness = (
            8.0 * shear * jnp.sqrt(jnp.maximum(effective_radius * batch.overlap, 0.0))
        )
        restitution = materials.pair_restitution(left_material, right_material).astype(
            batch.overlap.dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = -2.0 * jnp.sqrt(5.0 / 6.0) * beta * jnp.sqrt(effective_mass * stiffness)
        return _tangential_response(
            batch,
            normal,
            transported_displacement,
            stiffness,
            damping,
            materials.pair_friction(left_material, right_material),
            step_size,
        )


class AbstractDEMRotationalContactPlan(StrictModule, NonTrainableState):
    rotational_law_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        batch: DEMContactBatch,
        normal: DEMNormalResponse,
        history: DEMRotationalHistory,
        context: DEMContactEvaluationContext,
        materials: Any,
        ambient_dimension: int,
        /,
    ) -> DEMRotationalResponse:
        raise NotImplementedError


class ConstantRollingResistancePlan(AbstractDEMRotationalContactPlan):
    rotational_law_id: str = eqx.field(static=True)

    def __init__(self, *, rotational_law_id: str | None = None):
        generated = canonical_fingerprint({"kind": "constant-rolling-resistance"})
        identifier = generated if rotational_law_id is None else str(rotational_law_id)
        if not identifier:
            raise ValueError("rotational_law_id must be nonempty.")
        self.rotational_law_id = identifier

    def evaluate(
        self,
        batch,
        normal,
        history,
        context,
        materials,
        ambient_dimension,
        /,
    ):
        coefficient = materials.pair_rolling_friction(
            context.left_material, context.right_material
        ).astype(batch.overlap.dtype)
        effective_radius = batch.effective_radius
        if ambient_dimension == 2:
            relative = batch.left_angular_velocity - batch.right_angular_velocity
        elif ambient_dimension == 3:
            relative_raw = batch.left_angular_velocity - batch.right_angular_velocity
            normal_component = jnp.sum(relative_raw * batch.normal, axis=-1)
            relative = relative_raw - normal_component[:, None] * batch.normal
        else:
            raise ValueError("Rolling resistance requires dimension 2 or 3.")
        rate = _safe_vector_norm(relative)
        safe_rate = jnp.where(rate > 0.0, rate, 1.0)
        limit = coefficient * effective_radius * normal.friction_load
        active = normal.active & (rate > 0.0) & (limit > 0.0)
        left_torque = -limit[:, None] * relative / safe_rate[:, None]
        left_torque = jnp.where(active[:, None], left_torque, 0.0)
        right_torque = -left_torque
        loss = jnp.where(
            active,
            -jnp.sum(left_torque * relative, axis=-1) * context.step_size,
            0.0,
        )
        angular = jnp.zeros_like(history.rolling_displacement)
        next_history = DEMRotationalHistory(
            angular,
            angular,
            jnp.where(normal.active[:, None], batch.normal, 0.0),
            active,
            jnp.zeros_like(active),
        )
        successful = (
            jnp.all(jnp.isfinite(left_torque))
            & jnp.all(jnp.isfinite(loss))
            & jnp.all(loss >= 0.0)
        )
        zero_torque = jnp.zeros_like(left_torque)
        scalar = jnp.zeros_like(batch.gap)
        return DEMRotationalResponse(
            left_torque,
            right_torque,
            left_torque,
            right_torque,
            zero_torque,
            zero_torque,
            scalar,
            loss,
            loss,
            scalar,
            active,
            jnp.zeros_like(active),
            jnp.where(active, 0.0, limit),
            jnp.full_like(batch.gap, jnp.inf),
            next_history,
            active,
            successful,
        )


class DEMContactModelPlan(StrictModule, NonTrainableState):
    normal: AbstractDEMNormalContactPlan
    cohesion: AbstractDEMCohesionPlan | None
    tangential: AbstractDEMTangentialContactPlan | None
    rotational: AbstractDEMRotationalContactPlan | None
    contact_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal: AbstractDEMNormalContactPlan,
        /,
        *,
        cohesion: AbstractDEMCohesionPlan | None = None,
        tangential: AbstractDEMTangentialContactPlan | None = None,
        rotational: AbstractDEMRotationalContactPlan | None = None,
        contact_model_id: str | None = None,
    ):
        if not isinstance(normal, AbstractDEMNormalContactPlan):
            raise TypeError("normal must be an AbstractDEMNormalContactPlan.")
        if cohesion is not None and not isinstance(cohesion, AbstractDEMCohesionPlan):
            raise TypeError("cohesion must be an AbstractDEMCohesionPlan or None.")
        if tangential is not None and not isinstance(
            tangential, AbstractDEMTangentialContactPlan
        ):
            raise TypeError(
                "tangential must be an AbstractDEMTangentialContactPlan or None."
            )
        if isinstance(tangential, MindlinTangentialContactPlan) and not isinstance(
            normal, HertzNormalContactPlan
        ):
            raise TypeError("Mindlin tangential contact requires Hertz normal contact.")
        if rotational is not None and not isinstance(
            rotational, AbstractDEMRotationalContactPlan
        ):
            raise TypeError(
                "rotational must be an AbstractDEMRotationalContactPlan or None."
            )
        generated = canonical_fingerprint(
            {
                "kind": "dem-contact-model",
                "normal": normal.normal_law_id,
                "cohesion": (None if cohesion is None else cohesion.cohesion_law_id),
                "tangential": None
                if tangential is None
                else tangential.tangential_law_id,
                "rotational": (
                    None if rotational is None else rotational.rotational_law_id
                ),
            }
        )
        identifier = generated if contact_model_id is None else str(contact_model_id)
        if not identifier:
            raise ValueError("contact_model_id must be nonempty.")
        self.normal = normal
        self.cohesion = cohesion
        self.tangential = tangential
        self.rotational = rotational
        self.contact_model_id = identifier

    @property
    def interaction_range(self) -> float:
        from ._dem_smooth import SmoothPenaltyNormalPlan

        normal_range = (
            self.normal.maximum_range
            if isinstance(self.normal, SmoothPenaltyNormalPlan)
            else 0.0
        )
        cohesion_range = (
            0.0 if self.cohesion is None else self.cohesion.maximum_interaction_range
        )
        return max(normal_range, cohesion_range)

    def prepare(
        self, materials: Any, ambient_dimension: int, /
    ) -> PreparedDEMContactModel:
        return PreparedDEMContactModel(self, materials, ambient_dimension)


class PreparedDEMContactModel(StrictModule, NonTrainableState):
    plan: DEMContactModelPlan

    @property
    def interaction_range(self) -> float:
        return self.plan.interaction_range

    materials: Any
    ambient_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: DEMContactModelPlan, materials: Any, ambient_dimension: int, /
    ):
        if not isinstance(plan, DEMContactModelPlan):
            raise TypeError("plan must be a DEMContactModelPlan.")
        dimension = int(ambient_dimension)
        if dimension not in (2, 3):
            raise ValueError("DEM contact model requires ambient dimension 2 or 3.")
        if not isinstance(materials.material_count, int) or materials.material_count <= 0:
            raise TypeError("materials must provide a positive material_count.")
        self.plan = plan
        self.materials = materials
        self.ambient_dimension = dimension
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dem-contact-model",
                "plan": plan.contact_model_id,
                "materials": materials.material_id,
                "ambient_dimension": dimension,
            }
        )

    def empty_history(self, capacity: int, dtype: Any, /) -> DEMContactHistory:
        history = DEMContactHistory.empty(
            capacity,
            self.ambient_dimension,
            dtype,
        )
        cohesion = (
            DEMCohesionHistory(())
            if self.plan.cohesion is None
            else self.plan.cohesion.initialize_history(capacity, dtype)
        )
        if not isinstance(cohesion, DEMCohesionHistory):
            cohesion = DEMCohesionHistory((cohesion,))
        return DEMContactHistory(
            history.pair_keys,
            history.valid,
            history.active,
            history.normal,
            cohesion,
            history.tangential,
            history.rotational,
        )

    def evaluate(
        self,
        batch: DEMContactBatch,
        previous_history: DEMContactHistory,
        context: DEMContactEvaluationContext,
        /,
        *,
        frame_tolerance: float = 1.0e-10,
    ) -> DEMContactResponse:
        if not isinstance(context, DEMContactEvaluationContext):
            raise TypeError("context must be a DEMContactEvaluationContext.")
        current_keys = context.current_keys
        current_valid = context.current_valid
        continued = context.continued
        left_inverse_mass = context.left_inverse_mass
        right_inverse_mass = context.right_inverse_mass
        left_radius = context.left_radius
        right_radius = context.right_radius
        left_material = context.left_material
        right_material = context.right_material
        step_size = context.step_size
        _validate_batch_inputs(
            batch,
            previous_history,
            current_keys,
            current_valid,
            continued,
            left_inverse_mass,
            right_inverse_mass,
            left_radius,
            right_radius,
            left_material,
            right_material,
        )
        if batch.gap.shape[0] == 0:
            vector = jnp.zeros((0, self.ambient_dimension), dtype=batch.normal.dtype)
            angular = jnp.zeros(
                (0, 1 if self.ambient_dimension == 2 else 3),
                dtype=batch.normal.dtype,
            )
            scalar = jnp.zeros((0,), dtype=batch.normal.dtype)
            mask = jnp.zeros((0,), dtype=bool)
            return DEMContactResponse(
                pair_force=vector,
                left_torque=angular,
                right_torque=angular,
                next_history=previous_history.with_routes(current_keys, current_valid),
                normal_force=vector,
                tangential_force=vector,
                cohesion_force=vector,
                active=mask,
                sticking=mask,
                sliding=mask,
                cohesion_births=mask,
                cohesion_ruptures=mask,
                bridge_volume_source=scalar,
                bridge_volume_release=scalar,
                bridge_volume_residual=scalar,
                rolling_yielded=mask,
                torsional_yielded=mask,
                elastic_energy=scalar,
                normal_viscous_endpoint_loss=scalar,
                rotational_torque_left=angular,
                rotational_torque_right=angular,
                normal_plastic_dissipated_work=scalar,
                rotational_dissipated_work=scalar,
                cohesion_dissipated_work=scalar,
                tangential_constitutive_loss_estimate=scalar,
                friction_defect=scalar,
                switch_margin=jnp.full((0,), jnp.inf, dtype=batch.normal.dtype),
                activation_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                cohesion_birth_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                cohesion_rupture_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                rolling_yield_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                torsional_yield_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                no_tension_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                frame_transport_margin=jnp.asarray(jnp.inf, dtype=batch.normal.dtype),
                maximum_overlap_fraction=jnp.zeros((), dtype=batch.normal.dtype),
                successful=self.materials.admissible(),
            )
        transported, transport_success, frame_margin = _transport_history(
            previous_history.tangential.displacement,
            previous_history.tangential.previous_normal,
            batch.normal,
            continued & previous_history.active,
            self.ambient_dimension,
            float(frame_tolerance),
        )
        normal = self.plan.normal.evaluate(
            batch,
            previous_history,
            left_inverse_mass,
            right_inverse_mass,
            left_radius,
            right_radius,
            left_material,
            right_material,
            self.materials,
            step_size,
        )
        if self.plan.cohesion is None:
            cohesion = zero_cohesion_response(
                batch.gap.shape,
                batch.gap.dtype,
                previous_history.cohesion,
            )
        else:
            cohesion = evaluate_dem_cohesion(
                self.plan.cohesion,
                batch,
                normal,
                previous_history.cohesion,
                context,
                self.materials,
            )
        friction_normal = eqx.tree_at(
            lambda response: response.friction_load,
            normal,
            normal.friction_load + cohesion.friction_load,
        )
        if self.plan.tangential is None:
            tangential = _zero_tangential_response(batch, normal.active)
        else:
            tangential = self.plan.tangential.evaluate(
                batch,
                friction_normal,
                transported,
                left_inverse_mass,
                right_inverse_mass,
                left_radius,
                right_radius,
                left_material,
                right_material,
                self.materials,
                step_size,
            )
        if self.plan.rotational is None:
            rotational = _zero_rotational_response(
                batch,
                self.ambient_dimension,
                previous_history.rotational,
            )
        else:
            rotational = self.plan.rotational.evaluate(
                batch,
                friction_normal,
                previous_history.rotational,
                context,
                self.materials,
                self.ambient_dimension,
            )
        normal_force = normal.force_magnitude[:, None] * batch.normal
        cohesion_force = cohesion.force_magnitude[:, None] * batch.normal
        pair_force = normal_force + cohesion_force + tangential.force
        left_torque = (
            sphere_lever_torque(batch.left_arm, tangential.force, self.ambient_dimension)
            + rotational.left_torque
        )
        right_torque = (
            sphere_lever_torque(
                batch.right_arm, -tangential.force, self.ambient_dimension
            )
            + rotational.right_torque
        )
        active = normal.active | cohesion.active
        next_history = DEMContactHistory(
            jnp.where(current_valid, current_keys, -1).astype(jnp.int64),
            current_valid,
            active,
            DEMNormalHistory(
                normal.next_maximum_overlap,
                normal.next_plastic_overlap,
                normal.next_previous_overlap,
            ),
            cohesion.next_history,
            DEMTangentialHistory(
                tangential.sliding,
                jnp.where(normal.active[:, None], batch.normal, 0.0),
                jnp.where(normal.active[:, None], tangential.displacement, 0.0),
            ),
            rotational.next_history,
        )
        radius_scale = jnp.maximum(jnp.minimum(left_radius, right_radius), 1.0e-30)
        maximum_overlap = jnp.max(
            jnp.where(normal.active, batch.overlap / radius_scale, 0.0)
        )
        activation_margin = jnp.min(jnp.where(batch.valid, jnp.abs(batch.gap), jnp.inf))
        cohesion_birth_margin = jnp.min(
            jnp.where(batch.valid, cohesion.birth_margin, jnp.inf)
        )
        cohesion_rupture_margin = jnp.min(
            jnp.where(batch.valid, cohesion.rupture_margin, jnp.inf)
        )
        no_tension_margin = jnp.min(
            jnp.where(batch.valid, normal.no_tension_margin, jnp.inf)
        )
        successful = (
            normal.successful
            & cohesion.successful
            & tangential.successful
            & rotational.successful
            & transport_success
            & self.materials.admissible()
            & jnp.all(jnp.isfinite(pair_force))
            & jnp.all(jnp.isfinite(left_torque))
            & jnp.all(jnp.isfinite(right_torque))
        )
        return DEMContactResponse(
            pair_force=jnp.where(active[:, None], pair_force, 0.0),
            left_torque=jnp.where(active[:, None], left_torque, 0.0),
            right_torque=jnp.where(active[:, None], right_torque, 0.0),
            next_history=next_history,
            normal_force=jnp.where(normal.active[:, None], normal_force, 0.0),
            tangential_force=jnp.where(normal.active[:, None], tangential.force, 0.0),
            cohesion_force=jnp.where(cohesion.active[:, None], cohesion_force, 0.0),
            active=active,
            sticking=tangential.sticking,
            sliding=tangential.sliding,
            cohesion_births=cohesion.born,
            cohesion_ruptures=cohesion.ruptured,
            bridge_volume_source=cohesion.bridge_volume_source,
            bridge_volume_release=cohesion.bridge_volume_release,
            bridge_volume_residual=cohesion.bridge_volume_residual,
            rolling_yielded=rotational.rolling_yielded,
            torsional_yielded=rotational.torsional_yielded,
            elastic_energy=(
                normal.elastic_energy
                + cohesion.elastic_energy
                + tangential.elastic_energy
                + rotational.elastic_energy
            ),
            normal_viscous_endpoint_loss=normal.viscous_endpoint_loss,
            rotational_torque_left=rotational.left_torque,
            rotational_torque_right=rotational.right_torque,
            normal_plastic_dissipated_work=normal.plastic_dissipated_work,
            rotational_dissipated_work=rotational.dissipated_work,
            cohesion_dissipated_work=cohesion.dissipated_work,
            tangential_constitutive_loss_estimate=(tangential.constitutive_loss_estimate),
            friction_defect=tangential.friction_defect,
            switch_margin=tangential.switch_margin,
            activation_margin=activation_margin,
            cohesion_birth_margin=cohesion_birth_margin,
            cohesion_rupture_margin=cohesion_rupture_margin,
            no_tension_margin=no_tension_margin,
            rolling_yield_margin=jnp.min(rotational.rolling_margin),
            torsional_yield_margin=jnp.min(rotational.torsional_margin),
            frame_transport_margin=frame_margin,
            maximum_overlap_fraction=maximum_overlap,
            successful=successful,
        )


def _pair_parameter(
    parameter: Array, left: Array, right: Array, material_count: int, /
) -> Array:
    if parameter.ndim == 0:
        return jnp.broadcast_to(parameter, left.shape)
    if parameter.shape != (material_count, material_count):
        raise ValueError("Contact pair table does not match material count.")
    return parameter[left, right]


def _effective_mass(left_inverse: Array, right_inverse: Array, /) -> Array:
    inverse = left_inverse + right_inverse
    return jnp.where(inverse > 0.0, 1.0 / inverse, 1.0)


def _safe_vector_norm(vector: Array, /) -> Array:
    squared = jnp.sum(vector * vector, axis=-1)
    root = jnp.sqrt(jnp.maximum(squared, jnp.finfo(vector.dtype).tiny))
    return jnp.where(squared > 0.0, root, 0.0)


def _transport_history(
    displacement: Array,
    old_normal: Array,
    new_normal: Array,
    continued: Array,
    dimension: int,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    if dimension == 2:
        cosine = jnp.sum(old_normal * new_normal, axis=-1)
        sine = old_normal[:, 0] * new_normal[:, 1] - old_normal[:, 1] * new_normal[:, 0]
        rotated = jnp.stack(
            (
                cosine * displacement[:, 0] - sine * displacement[:, 1],
                sine * displacement[:, 0] + cosine * displacement[:, 1],
            ),
            axis=-1,
        )
        bad = (
            continued
            & (cosine <= -1.0 + tolerance)
            & (_safe_vector_norm(displacement) > tolerance)
        )
    elif dimension == 3:
        axis = jnp.cross(old_normal, new_normal)
        cosine = jnp.sum(old_normal * new_normal, axis=-1)
        denominator = jnp.where(1.0 + cosine > tolerance, 1.0 + cosine, 1.0)
        first = jnp.cross(axis, displacement)
        rotated = displacement + first + jnp.cross(axis, first) / denominator[:, None]
        bad = (
            continued
            & (1.0 + cosine <= tolerance)
            & (_safe_vector_norm(displacement) > tolerance)
        )
    else:
        raise ValueError("Contact-frame transport requires dimension 2 or 3.")
    margin = jnp.min(jnp.where(continued, 1.0 + cosine, jnp.inf))
    return (
        jnp.where(continued[:, None], rotated, 0.0),
        ~jnp.any(bad),
        margin,
    )


def _tangential_response(
    batch: DEMContactBatch,
    normal: DEMNormalResponse,
    transported: Array,
    stiffness: Array,
    damping: Array,
    friction: Array,
    step_size: Array,
    /,
) -> DEMTangentialResponse:
    active = normal.active
    displacement = transported + jnp.asarray(step_size) * batch.tangential_velocity
    trial = (
        -stiffness[:, None] * displacement - damping[:, None] * batch.tangential_velocity
    )
    trial_norm = _safe_vector_norm(trial)
    limit = friction.astype(trial_norm.dtype) * normal.friction_load
    sliding = active & (trial_norm > limit)
    sticking = active & ~sliding
    safe_norm = jnp.where(trial_norm > 0.0, trial_norm, 1.0)
    capped = limit[:, None] * trial / safe_norm[:, None]
    force = jnp.where(sliding[:, None], capped, trial)
    corrected = (
        -(force + damping[:, None] * batch.tangential_velocity)
        / jnp.where(stiffness > 0.0, stiffness, 1.0)[:, None]
    )
    displacement = jnp.where(sliding[:, None], corrected, displacement)
    force = jnp.where(active[:, None], force, 0.0)
    displacement = jnp.where(active[:, None], displacement, 0.0)
    energy = jnp.where(
        active, 0.5 * stiffness * jnp.sum(displacement * displacement, axis=-1), 0.0
    )
    old_energy = jnp.where(
        active, 0.5 * stiffness * jnp.sum(transported * transported, axis=-1), 0.0
    )
    mechanical_loss = -jnp.sum(force * batch.tangential_velocity, axis=-1) * jnp.asarray(
        step_size
    ) - (energy - old_energy)
    viscous_loss = (
        damping
        * jnp.sum(batch.tangential_velocity * batch.tangential_velocity, axis=-1)
        * jnp.asarray(step_size)
    )
    dissipated = jnp.where(
        active, jnp.maximum(jnp.maximum(mechanical_loss, viscous_loss), 0.0), 0.0
    )
    force_norm = _safe_vector_norm(force)
    defect = jnp.where(active, jnp.maximum(force_norm - limit, 0.0), 0.0)
    margin = jnp.where(active, jnp.abs(trial_norm - limit), jnp.inf)
    successful = (
        jnp.all(jnp.isfinite(force))
        & jnp.all(jnp.isfinite(displacement))
        & jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(dissipated))
    )
    return DEMTangentialResponse(
        force,
        displacement,
        energy,
        dissipated,
        sticking,
        sliding,
        defect,
        margin,
        successful,
    )


def _zero_rotational_response(
    batch: DEMContactBatch,
    ambient_dimension: int,
    history: DEMRotationalHistory,
    /,
) -> DEMRotationalResponse:
    angular_dimension = 1 if ambient_dimension == 2 else 3
    torque = jnp.zeros(
        (batch.overlap.shape[0], angular_dimension),
        dtype=batch.overlap.dtype,
    )
    scalar = jnp.zeros_like(batch.overlap)
    mask = jnp.zeros_like(batch.valid)
    next_history = DEMRotationalHistory(
        torque,
        torque,
        jnp.zeros_like(batch.normal),
        mask,
        mask,
    )
    del history
    return DEMRotationalResponse(
        left_torque=torque,
        right_torque=torque,
        rolling_torque_left=torque,
        rolling_torque_right=torque,
        torsional_torque_left=torque,
        torsional_torque_right=torque,
        elastic_energy=scalar,
        dissipated_work=scalar,
        rolling_dissipated_work=scalar,
        torsional_dissipated_work=scalar,
        rolling_yielded=mask,
        torsional_yielded=mask,
        rolling_margin=jnp.full_like(batch.gap, jnp.inf),
        torsional_margin=jnp.full_like(batch.gap, jnp.inf),
        next_history=next_history,
        active=mask,
        successful=jnp.asarray(True),
    )


def _zero_tangential_response(
    batch: DEMContactBatch, active: Array, /
) -> DEMTangentialResponse:
    zeros_vector = jnp.zeros_like(batch.tangential_velocity)
    zeros = jnp.zeros_like(batch.overlap)
    return DEMTangentialResponse(
        zeros_vector,
        zeros_vector,
        zeros,
        zeros,
        active,
        jnp.zeros_like(batch.valid),
        zeros,
        jnp.full_like(batch.overlap, jnp.inf),
        jnp.asarray(True),
    )


def _validate_batch_inputs(
    batch,
    history,
    keys,
    valid,
    continued,
    left_inverse_mass,
    right_inverse_mass,
    left_radius,
    right_radius,
    left_material,
    right_material,
    /,
) -> None:
    if not isinstance(batch, DEMContactBatch):
        raise TypeError("batch must be a DEMContactBatch.")
    if not isinstance(history, DEMContactHistory):
        raise TypeError("previous_history must be a DEMContactHistory.")
    capacity = batch.gap.shape[0]
    dimension = batch.normal.shape[1]
    if batch.normal.shape != (capacity, dimension) or dimension not in (2, 3):
        raise ValueError("DEM contact batch normal shape is invalid.")
    vector_shape = (capacity, dimension)
    angular_shape = (capacity, 1 if dimension == 2 else 3)
    if (
        batch.overlap.shape != (capacity,)
        or batch.effective_radius.shape != (capacity,)
        or batch.left_arm.shape != vector_shape
        or batch.right_arm.shape != vector_shape
        or batch.tangential_velocity.shape != vector_shape
        or batch.normal_velocity.shape != (capacity,)
        or batch.left_angular_velocity.shape != angular_shape
        or batch.right_angular_velocity.shape != angular_shape
        or batch.valid.shape != (capacity,)
    ):
        raise ValueError("DEM contact batch fields have inconsistent shapes.")
    if (
        history.pair_keys.shape != (capacity,)
        or history.valid.shape != (capacity,)
        or history.active.shape != (capacity,)
        or history.normal.maximum_overlap.shape != (capacity,)
        or history.normal.plastic_overlap.shape != (capacity,)
        or history.normal.previous_overlap.shape != (capacity,)
        or history.tangential.sliding.shape != (capacity,)
        or history.tangential.previous_normal.shape != vector_shape
        or history.tangential.displacement.shape != vector_shape
        or history.rotational.rolling_displacement.shape != angular_shape
        or history.rotational.torsional_displacement.shape != angular_shape
        or history.rotational.previous_normal.shape != vector_shape
        or history.rotational.rolling_yielded.shape != (capacity,)
        or history.rotational.torsional_yielded.shape != (capacity,)
    ):
        raise ValueError("DEM contact history does not match contact capacity.")
    if any(
        not eqx.is_array(leaf) or leaf.ndim == 0 or leaf.shape[0] != capacity
        for leaf in jax.tree.leaves(history.cohesion)
    ):
        raise ValueError("DEM cohesion history does not match contact capacity.")
    scalar_arrays = (
        keys,
        valid,
        continued,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
    )
    if any(jnp.asarray(value).shape != (capacity,) for value in scalar_arrays):
        raise ValueError("DEM contact endpoint arrays must have contact-capacity shape.")


__all__ = [
    "AbstractDEMNormalContactPlan",
    "AbstractDEMRotationalContactPlan",
    "AbstractDEMTangentialContactPlan",
    "ConstantRollingResistancePlan",
    "CundallStrackTangentialPlan",
    "DEMContactBatch",
    "DEMContactHistory",
    "DEMContactModelPlan",
    "DEMContactResponse",
    "DEMNormalResponse",
    "DEMRotationalResponse",
    "DEMTangentialResponse",
    "HertzNormalContactPlan",
    "LinearSpringDashpotNormalPlan",
    "MindlinTangentialContactPlan",
    "PreparedDEMContactModel",
    "ThorntonLinearPlasticNormalPlan",
]
