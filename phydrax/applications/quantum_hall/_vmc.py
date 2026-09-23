#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Landau-level-mixing variational Monte Carlo on the monopole sphere."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._sampling import MetropolisHastings, SingleElectronSphereProposal
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nn.quantum import MonopoleAttentionAmplitude
from ...operators.quantum import (
    evaluate_local_operator,
    LogAmplitude,
    MonopoleSphereCoulombHamiltonian,
    uniform_sphere_electron_walkers,
)
from ...solver import VariationalMonteCarloProblem
from ._sphere import HaldaneSpherePlan
from ._trial_states import LaughlinSphereAmplitude


class LandauLevelMixingVMCPlan(StrictModule, NonTrainableState):
    sphere: HaldaneSpherePlan
    landau_level_mixing: float = eqx.field(static=True)
    chain_count: int = eqx.field(static=True)
    proposal_angle: float = eqx.field(static=True)
    hidden_dimension: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sphere: HaldaneSpherePlan,
        landau_level_mixing: float,
        /,
        *,
        chain_count: int = 64,
        proposal_angle: float = 0.25,
        hidden_dimension: int = 32,
        layer_count: int = 2,
        determinant_count: int = 4,
    ):
        if not isinstance(sphere, HaldaneSpherePlan):
            raise TypeError("sphere must be HaldaneSpherePlan.")
        if sphere.statistics != "fermion":
            raise ValueError("Landau-level-mixing VMC currently supports fermions only.")
        mixing = float(landau_level_mixing)
        chains = int(chain_count)
        angle = float(proposal_angle)
        hidden = int(hidden_dimension)
        layers = int(layer_count)
        determinants = int(determinant_count)
        if (
            not isfinite(mixing)
            or mixing <= 0.0
            or chains < 1
            or not isfinite(angle)
            or angle <= 0.0
            or hidden < 2
            or layers < 1
            or determinants < 1
        ):
            raise ValueError("Landau-level-mixing VMC controls are invalid.")
        self.sphere = sphere
        self.landau_level_mixing = mixing
        self.chain_count = chains
        self.proposal_angle = angle
        self.hidden_dimension = hidden
        self.layer_count = layers
        self.determinant_count = determinants
        self.plan_id = canonical_fingerprint(
            {
                "kind": "landau-level-mixing-vmc-plan",
                "sphere": sphere.plan_id,
                "landau_level_mixing": mixing,
                "chain_count": chains,
                "proposal_angle": angle,
                "hidden_dimension": hidden,
                "layer_count": layers,
                "determinant_count": determinants,
            }
        )


class PreparedLandauLevelMixingVMC(StrictModule):
    plan: LandauLevelMixingVMCPlan = eqx.field(static=True)
    model: MonopoleAttentionAmplitude
    operator: MonopoleSphereCoulombHamiltonian
    problem: VariationalMonteCarloProblem
    prepared_id: str = eqx.field(static=True)


def prepare_landau_level_mixing_vmc(
    plan: LandauLevelMixingVMCPlan,
    key: Key[Array, ""],
    /,
) -> PreparedLandauLevelMixingVMC:
    if not isinstance(plan, LandauLevelMixingVMCPlan):
        raise TypeError("plan must be LandauLevelMixingVMCPlan.")
    model_key, walker_key = jr.split(key)
    model = MonopoleAttentionAmplitude(
        plan.sphere.particle_count,
        plan.sphere.twice_monopole_flux,
        model_key,
        hidden_dimension=plan.hidden_dimension,
        layer_count=plan.layer_count,
        determinant_count=plan.determinant_count,
    )
    operator = MonopoleSphereCoulombHamiltonian(
        plan.sphere.particle_count,
        plan.sphere.twice_monopole_flux,
        kinetic_strength=1.0,
        interaction_strength=plan.landau_level_mixing,
    )
    proposal = SingleElectronSphereProposal(
        plan.sphere.particle_count,
        plan.proposal_angle,
    )
    walkers = uniform_sphere_electron_walkers(
        walker_key,
        plan.chain_count,
        plan.sphere.particle_count,
    )
    problem = VariationalMonteCarloProblem(
        model,
        operator,
        MetropolisHastings(proposal),
        walkers,
        complex_parameter_mode="nonholomorphic",
        problem_id=plan.plan_id,
    )
    return PreparedLandauLevelMixingVMC(
        plan,
        model,
        operator,
        problem,
        canonical_fingerprint(
            {
                "kind": "prepared-landau-level-mixing-vmc",
                "plan": plan.plan_id,
                "model": model.network_id,
                "operator": operator.operator_id,
                "problem": problem.problem_id,
            }
        ),
    )


class QuantumHallVMCObservables(StrictModule, NonTrainableState):
    energy_mean: Array
    energy_standard_error: Array
    laughlin_overlap: Array
    pair_angle_edges: Array
    pair_angle_probabilities: Array
    valid_sample_count: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def _complex_ratio(target: LogAmplitude, source: LogAmplitude, /) -> Array:
    return (
        jnp.exp(target.log_abs - source.log_abs) * target.phase * jnp.conj(source.phase)
    )


def evaluate_quantum_hall_vmc_observables(
    prepared: PreparedLandauLevelMixingVMC,
    model: MonopoleAttentionAmplitude,
    samples: ArrayLike,
    /,
    *,
    pair_angle_bins: int = 32,
    laughlin_exponent: int = 3,
) -> QuantumHallVMCObservables:
    if not isinstance(prepared, PreparedLandauLevelMixingVMC):
        raise TypeError("prepared must be PreparedLandauLevelMixingVMC.")
    if not isinstance(model, MonopoleAttentionAmplitude):
        raise TypeError("model must be MonopoleAttentionAmplitude.")
    values = jnp.asarray(samples)
    if (
        values.ndim < 3
        or tuple(values.shape[-2:]) != prepared.operator.configuration_shape
    ):
        raise ValueError("samples must end in the sphere configuration shape.")
    flat = values.reshape((-1,) + prepared.operator.configuration_shape)
    coordinate_valid = (
        jnp.all(jnp.isfinite(flat), axis=(-2, -1))
        & jnp.all(flat[..., 0] >= 0.0, axis=-1)
        & jnp.all(flat[..., 0] <= jnp.pi, axis=-1)
    )
    local = evaluate_local_operator(model, prepared.operator, flat)
    valid = local.successful & coordinate_valid
    safe_energy = jnp.where(valid, local.value, 0.0)
    count = jnp.sum(valid, dtype=jnp.int32)
    denominator = jnp.maximum(count, 1)
    energy_mean = jnp.sum(safe_energy) / denominator
    centered = jnp.where(valid, local.value - energy_mean, 0.0)
    energy_variance = jnp.sum(jnp.abs(centered) ** 2) / jnp.maximum(count - 1, 1)
    energy_error = jnp.sqrt(energy_variance / denominator)
    laughlin = LaughlinSphereAmplitude(
        prepared.plan.sphere.particle_count,
        laughlin_exponent,
    )
    if laughlin.twice_monopole_flux != prepared.plan.sphere.twice_monopole_flux:
        raise ValueError(
            "Laughlin overlap exponent does not match the prepared sphere flux."
        )
    source_amplitudes = jax.vmap(model)(flat)
    target_amplitudes = jax.vmap(laughlin)(flat)
    ratio = jax.vmap(_complex_ratio)(target_amplitudes, source_amplitudes)
    ratio_valid = (
        source_amplitudes.nonzero
        & source_amplitudes.valid
        & target_amplitudes.valid
        & coordinate_valid
        & jnp.isfinite(ratio)
    )
    raw_ratio_count = jnp.sum(ratio_valid)
    ratio_denominator = jnp.maximum(raw_ratio_count, 1)
    safe_ratio = jnp.where(ratio_valid, ratio, 0.0j)
    ratio_mean = jnp.sum(safe_ratio) / ratio_denominator
    ratio_norm = (
        jnp.sum(jnp.where(ratio_valid, jnp.abs(ratio) ** 2, 0.0)) / ratio_denominator
    )
    overlap = jnp.abs(ratio_mean) / jnp.sqrt(
        jnp.maximum(ratio_norm, jnp.finfo(ratio_norm.dtype).tiny)
    )
    safe_flat = jnp.where(coordinate_valid[:, None, None], flat, 0.0)
    theta, phi = safe_flat[..., 0], safe_flat[..., 1]
    sine = jnp.sin(theta)
    cartesian = jnp.stack(
        (sine * jnp.cos(phi), sine * jnp.sin(phi), jnp.cos(theta)), axis=-1
    )
    cosine = contract("sia,sja->sij", cartesian, cartesian, backend="jax")
    pair_mask = np.triu(
        np.ones((prepared.plan.sphere.particle_count,) * 2, dtype=np.bool_),
        k=1,
    )
    pair_angles = jnp.arccos(jnp.clip(cosine[:, pair_mask], -1.0, 1.0)).reshape((-1,))
    pair_count = pair_mask.sum()
    bins = int(pair_angle_bins)
    if bins < 2:
        raise ValueError("pair_angle_bins must be at least two.")
    edges = jnp.linspace(0.0, jnp.pi, bins + 1)
    indices = jnp.clip(
        jnp.searchsorted(edges, pair_angles, side="right") - 1, 0, bins - 1
    )
    pair_weights = jnp.repeat(coordinate_valid.astype(jnp.float64), pair_count)
    histogram = jnp.zeros((bins,), dtype=jnp.float64).at[indices].add(pair_weights)
    probabilities = histogram / jnp.maximum(jnp.sum(histogram), 1.0)
    successful = (
        (count > 1)
        & (raw_ratio_count > 0)
        & (jnp.sum(pair_weights) > 0)
        & jnp.isfinite(energy_mean)
        & jnp.isfinite(energy_error)
        & jnp.isfinite(overlap)
        & jnp.all(jnp.isfinite(probabilities))
    )
    result_id = canonical_fingerprint(
        {
            "kind": "quantum-hall-vmc-observables",
            "prepared": prepared.prepared_id,
            "model": model.network_id,
            "samples": array_tree_fingerprint(np.asarray(values)),
            "pair_angle_bins": bins,
            "laughlin_exponent": int(laughlin_exponent),
        }
    )
    return QuantumHallVMCObservables(
        energy_mean,
        energy_error,
        overlap,
        edges,
        probabilities,
        count,
        successful,
        result_id,
    )


__all__ = [
    "LandauLevelMixingVMCPlan",
    "PreparedLandauLevelMixingVMC",
    "QuantumHallVMCObservables",
    "evaluate_quantum_hall_vmc_observables",
    "prepare_landau_level_mixing_vmc",
]
