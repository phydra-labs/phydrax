#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.nn.operator.representations import (
    CliffordGradeFeatures,
    CliffordGradeRepresentation,
)

from .clifford_models import (
    DifferentialCliffordOperatorBlock,
    PeriodicCliffordLaplacian,
)


@dataclass(frozen=True, slots=True)
class CliffordBenchmarkScenario:
    name: str
    representation: CliffordGradeRepresentation
    inputs: Array
    targets: Array
    grid_shape: tuple[int, ...]
    diagnostics: Mapping[str, Array | str]
    scenario_id: str


@dataclass(frozen=True, slots=True)
class CliffordDecisionSmokeReport:
    passed: bool
    scenario_names: tuple[str, ...]
    baseline_relative_errors: tuple[float, ...]
    candidate_relative_errors: tuple[float, ...]
    candidate_ids: tuple[str, ...]
    report_id: str


def _scenario(
    name: str,
    representation: CliffordGradeRepresentation,
    inputs: Array,
    targets: Array,
    grid_shape: tuple[int, ...],
    diagnostics: Mapping[str, Array | str],
    /,
) -> CliffordBenchmarkScenario:
    if inputs.shape != targets.shape or inputs.shape[-1] != representation.packed_size:
        raise ValueError("Clifford benchmark input and target schemas do not match.")
    if inputs.shape[-len(grid_shape) - 1 : -1] != grid_shape:
        raise ValueError("Clifford benchmark grid shape does not match field arrays.")
    scenario_id = canonical_fingerprint(
        {
            "kind": "clifford-operator-benchmark-scenario-v1",
            "name": name,
            "representation": representation.representation_id,
            "grid_shape": list(grid_shape),
            "data": array_tree_fingerprint((inputs, targets)),
            "diagnostic_keys": sorted(diagnostics),
        }
    )
    return CliffordBenchmarkScenario(
        name,
        representation,
        inputs,
        targets,
        grid_shape,
        diagnostics,
        scenario_id,
    )


def multigrade_incompressible_scenario(
    resolution: int = 8,
    /,
    *,
    viscosity: float = 1e-2,
    time_step: float = 0.1,
) -> CliffordBenchmarkScenario:
    """Return a periodic velocity-vector/vorticity-bivector diffusion step."""
    count = int(resolution)
    if count < 4:
        raise ValueError("Incompressible Clifford scenario requires resolution >= 4.")
    coordinate = 2.0 * jnp.pi * jnp.arange(count) / count
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    velocity = jnp.stack(
        (jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)),
        axis=-1,
    )
    vorticity = 2.0 * jnp.sin(x) * jnp.sin(y)
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1))
    representation = CliffordGradeRepresentation(algebra, (1, 1, 1))
    leading = (1, count, count)
    inputs = representation.join(
        CliffordGradeFeatures(
            (
                jnp.zeros(leading + (1, 1)),
                velocity[None, ..., None, :],
                vorticity[None, ..., None, None],
            )
        )
    )
    decay = jnp.exp(-2.0 * float(viscosity) * float(time_step))
    targets = decay * inputs
    divergence = jnp.cos(x) * jnp.cos(y) - jnp.cos(x) * jnp.cos(y)
    return _scenario(
        "clifford_incompressible_velocity_vorticity_2d",
        representation,
        inputs,
        targets,
        (count, count),
        {
            "maximum_divergence": jnp.max(jnp.abs(divergence)),
            "viscosity": jnp.asarray(viscosity),
            "time_step": jnp.asarray(time_step),
        },
    )


def entropy_euler_scenario(resolution: int = 8, /) -> CliffordBenchmarkScenario:
    """Return a smooth admissible Euler state and one periodic transport shift."""
    count = int(resolution)
    if count < 4:
        raise ValueError("Entropy Euler Clifford scenario requires resolution >= 4.")
    coordinate = 2.0 * jnp.pi * jnp.arange(count) / count
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    density = 1.0 + 0.08 * jnp.sin(x)
    velocity_x = 0.2 + 0.03 * jnp.cos(y)
    velocity_y = 0.04 * jnp.sin(x)
    pressure = 1.0 + 0.05 * jnp.cos(x + y)
    primitive = jnp.stack(
        (density, velocity_x, velocity_y, pressure),
        axis=-1,
    )
    system = phx.equations.EulerSystem(2)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    state = system.primitive_to_conserved(primitive)
    target_state = jnp.roll(state, 1, axis=0)
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1))
    representation = CliffordGradeRepresentation(algebra, (2, 1, 0))

    def pack(conserved: Array) -> Array:
        return representation.join(
            CliffordGradeFeatures(
                (
                    jnp.stack((conserved[..., 0], conserved[..., -1]), axis=-1)[
                        ..., :, None
                    ],
                    conserved[..., 1:3][..., None, :],
                    jnp.zeros(conserved.shape[:-1] + (0, 1)),
                )
            )
        )

    inputs = pack(state)[None, ...]
    targets = pack(target_state)[None, ...]
    volumes = jnp.full((count, count), 1.0 / (count * count))
    relative_entropy = phx.discretization.integrated_finite_volume_relative_entropy(
        pair,
        target_state,
        state,
        volumes,
    )
    return _scenario(
        "clifford_entropy_euler_2d",
        representation,
        inputs,
        targets,
        (count, count),
        {
            "pair_id": pair.pair_id,
            "admissible": jnp.all(pair.admissible(state))
            & jnp.all(pair.admissible(target_state)),
            "integrated_relative_entropy": relative_entropy,
            "total_entropy": jnp.sum(volumes * pair.entropy(state)),
        },
    )


def multigrade_maxwell_scenario(
    resolution: int = 8,
    /,
    *,
    phase_step: float = 0.2,
) -> CliffordBenchmarkScenario:
    """Return an analytic electric-vector/magnetic-bivector plane-wave step."""
    count = int(resolution)
    if count < 4:
        raise ValueError("Maxwell Clifford scenario requires resolution >= 4.")
    coordinate = 2.0 * jnp.pi * jnp.arange(count) / count
    phase = coordinate[:, None, None]
    wave = jnp.broadcast_to(jnp.sin(phase), (count, count, count))
    target_wave = jnp.broadcast_to(
        jnp.sin(phase - float(phase_step)),
        (count, count, count),
    )
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1, 1))
    representation = CliffordGradeRepresentation(algebra, (0, 1, 1, 0))

    def pack(amplitude: Array) -> Array:
        zeros = jnp.zeros_like(amplitude)
        electric = jnp.stack((zeros, amplitude, zeros), axis=-1)
        magnetic = jnp.stack((amplitude, zeros, zeros), axis=-1)
        leading = amplitude.shape
        return representation.join(
            CliffordGradeFeatures(
                (
                    jnp.zeros(leading + (0, 1)),
                    electric[..., None, :],
                    magnetic[..., None, :],
                    jnp.zeros(leading + (0, 1)),
                )
            )
        )

    inputs = pack(wave)[None, ...]
    targets = pack(target_wave)[None, ...]
    return _scenario(
        "clifford_maxwell_plane_wave_3d",
        representation,
        inputs,
        targets,
        (count, count, count),
        {
            "phase_step": jnp.asarray(phase_step),
            "electric_divergence": jnp.asarray(0.0),
            "magnetic_divergence": jnp.asarray(0.0),
        },
    )


def clifford_benchmark_scenarios(
    resolution: int = 8,
    /,
) -> tuple[CliffordBenchmarkScenario, ...]:
    return (
        multigrade_incompressible_scenario(resolution),
        entropy_euler_scenario(resolution),
        multigrade_maxwell_scenario(resolution),
    )


def run_clifford_decision_smoke(
    resolution: int = 8,
    /,
    *,
    key: Array = jr.key(0),
) -> CliffordDecisionSmokeReport:
    """Exercise all candidates; this smoke is not a promotion-eligible training run."""
    scenarios = clifford_benchmark_scenarios(resolution)
    keys = jr.split(key, len(scenarios))
    baseline_errors = []
    candidate_errors = []
    candidate_ids = []
    finite = True
    for scenario, scenario_key in zip(scenarios, keys):
        context = PeriodicCliffordLaplacian(scenario.grid_shape)
        candidate = DifferentialCliffordOperatorBlock(
            scenario.representation,
            context,
            latent_channels=2,
            residual_scale=0.01,
            key=scenario_key,
        )
        output = candidate(scenario.inputs)
        target_norm = jnp.maximum(jnp.linalg.norm(scenario.targets), 1e-12)
        baseline_error = jnp.linalg.norm(scenario.inputs - scenario.targets) / target_norm
        candidate_error = jnp.linalg.norm(output - scenario.targets) / target_norm
        baseline_errors.append(float(baseline_error))
        candidate_errors.append(float(candidate_error))
        candidate_ids.append(candidate.candidate_id)
        finite = finite and bool(jnp.all(jnp.isfinite(output)))
    report_id = canonical_fingerprint(
        {
            "kind": "clifford-decision-smoke-v1",
            "scenarios": [scenario.scenario_id for scenario in scenarios],
            "baseline_errors": baseline_errors,
            "candidate_errors": candidate_errors,
            "candidate_ids": candidate_ids,
            "profile": "smoke-not-promotion-eligible",
        }
    )
    return CliffordDecisionSmokeReport(
        passed=finite,
        scenario_names=tuple(scenario.name for scenario in scenarios),
        baseline_relative_errors=tuple(baseline_errors),
        candidate_relative_errors=tuple(candidate_errors),
        candidate_ids=tuple(candidate_ids),
        report_id=report_id,
    )


__all__ = [
    "clifford_benchmark_scenarios",
    "CliffordBenchmarkScenario",
    "CliffordDecisionSmokeReport",
    "entropy_euler_scenario",
    "multigrade_incompressible_scenario",
    "multigrade_maxwell_scenario",
    "run_clifford_decision_smoke",
]
