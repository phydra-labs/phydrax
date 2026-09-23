#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from .._exponential_family import (
    FiniteSupportNaturalSolvePlan,
    solve_finite_support_mean,
)
from ..discretization._kinetic_entropy import solve_kinetic_entropy_root
from ..discretization.discrete_velocity import (
    CompressibleKineticPopulationState,
    PositiveCompressibleKineticPlan,
)
from ._iree import IREEExportPolicy, IREEExportResult, save_iree


def save_compressible_kinetic_iree(
    plan: PositiveCompressibleKineticPlan,
    path: str | Path,
    state_template: CompressibleKineticPopulationState,
    relaxation_rate,
    /,
    *,
    policy: IREEExportPolicy | None = None,
    validate: bool = True,
) -> IREEExportResult:
    """Export one fixed-shape local kinetic collision with status outputs."""
    if not isinstance(plan, PositiveCompressibleKineticPlan):
        raise TypeError("plan must be a PositiveCompressibleKineticPlan.")
    if not isinstance(state_template, CompressibleKineticPopulationState):
        raise TypeError("state_template must be CompressibleKineticPopulationState.")
    if state_template.layout.layout_id != plan.layout.layout_id:
        raise ValueError("state_template layout does not match the kinetic plan.")
    if plan.rule.velocities.dtype != state_template.populations[0].dtype:
        raise ValueError(
            "IREE export requires rule coefficients and population storage to "
            "share one dtype."
        )
    selected_policy = (
        IREEExportPolicy(target_backend="vmvx", runtime_driver="local-task")
        if policy is None
        else policy
    )
    if not isinstance(selected_policy, IREEExportPolicy):
        raise TypeError("policy must be IREEExportPolicy or None.")
    if selected_policy.target_backend == "vmvx" and state_template.populations[
        0
    ].dtype != jnp.dtype("float32"):
        raise ValueError("Portable VMVX kinetic export requires float32 state.")
    rate = jnp.asarray(relaxation_rate, dtype=state_template.populations[0].dtype)
    inputs = (
        *state_template.populations,
        state_template.equilibrium_dual,
        state_template.stabilizer,
        rate,
    )
    population_count = len(state_template.populations)
    base_solve = plan.family.solve_plan
    portable_solve = FiniteSupportNaturalSolvePlan(
        maximum_steps=base_solve.maximum_steps,
        residual_tolerance=base_solve.residual_tolerance,
        minimum_probability=base_solve.minimum_probability,
        line_search_factors=base_solve.line_search_factors,
        portable=True,
    )

    def collision(*arrays):
        populations = tuple(arrays[:population_count])
        dual, stabilizer, local_rate = arrays[population_count:]
        spatial_shape = populations[0].shape[:-1]
        frame = jnp.zeros(
            spatial_shape + (plan.rule.dimension,),
            dtype=populations[0].dtype,
        )
        frame_scale = jnp.ones(spatial_shape, dtype=populations[0].dtype)
        state = CompressibleKineticPopulationState(
            populations,
            dual,
            stabilizer,
            frame,
            frame_scale,
            plan.layout,
        )
        old = plan.moments(state)
        target_features = plan._target_features(old.velocity, old.temperature)
        equilibrium_solve = solve_finite_support_mean(
            plan.family,
            target_features,
            initial=dual,
            plan=portable_solve,
        )
        particle_equilibrium = old.density[..., None] * equilibrium_solve.probabilities
        equilibrium = [particle_equilibrium]
        if len(plan.layout.fields) == 2:
            equilibrium.append(
                plan.internal_heat_capacity
                * old.temperature[..., None]
                * particle_equilibrium
            )
        if plan.collision_kind == "entropic":
            root = solve_kinetic_entropy_root(
                plan.entropy_root,
                populations[0],
                particle_equilibrium - populations[0],
                base_measure=plan.rule.base_probabilities,
                initial=stabilizer,
            )
            factor = 0.5 * local_rate * root.evidence.alpha
            next_stabilizer = root.evidence.alpha
            root_successful = root.evidence.successful
        else:
            factor = local_rate
            next_stabilizer = jnp.ones_like(stabilizer)
            root_successful = jnp.ones_like(stabilizer, dtype=jnp.bool_)
        candidates = tuple(
            value + factor[..., None] * (target - value)
            for value, target in zip(populations, equilibrium, strict=True)
        )
        candidate_state = CompressibleKineticPopulationState(
            candidates,
            equilibrium_solve.conversion.natural.values,
            next_stabilizer,
            frame,
            frame_scale,
            plan.layout,
        )
        new = plan.moments(candidate_state)
        mass_defect = new.density - old.density
        energy_defect = new.total_energy - old.total_energy
        minimum = jnp.min(
            jnp.stack(tuple(jnp.min(value, axis=-1) for value in candidates)),
            axis=0,
        )
        tolerance = jnp.maximum(
            512.0 * jnp.finfo(old.density.dtype).eps,
            8.0 * plan.family.solve_plan.residual_tolerance,
        )
        successful = (
            equilibrium_solve.evidence.successful
            & root_successful
            & old.admissible
            & new.admissible
            & jnp.isfinite(local_rate)
            & (local_rate > 0.0)
            & (local_rate < 2.0)
            & (minimum > 0.0)
            & (jnp.abs(mass_defect) <= tolerance * jnp.maximum(old.density, 1.0))
            & (
                jnp.abs(energy_defect)
                <= tolerance * jnp.maximum(jnp.abs(old.total_energy), 1.0)
            )
        )
        accepted = tuple(
            jnp.where(successful[..., None], candidate, original)
            for candidate, original in zip(candidates, populations, strict=True)
        )
        return (
            *accepted,
            jnp.where(
                successful[..., None],
                equilibrium_solve.conversion.natural.values,
                dual,
            ),
            jnp.where(successful, next_stabilizer, stabilizer),
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )

    input_names = tuple(field.name for field in plan.layout.fields) + (
        "equilibrium_dual",
        "stabilizer",
        "relaxation_rate",
    )
    output_names = tuple(f"accepted_{field.name}" for field in plan.layout.fields) + (
        "accepted_equilibrium_dual",
        "accepted_stabilizer",
        "successful",
        "status",
    )
    return save_iree(
        collision,
        path,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        policy=selected_policy,
        validate=validate,
    )


__all__ = ["save_compressible_kinetic_iree"]
