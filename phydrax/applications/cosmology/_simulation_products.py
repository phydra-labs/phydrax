#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed dark-matter analysis products and exact restart payload contracts."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...discretization.particle import ParticleDiscretization
from ...solver._production_runtime import (
    ArtifactCheckpointStore,
    CheckpointCommitReceipt,
    DurableCheckpointStore,
    ProductionCaseManifest,
)
from ...solver._runtime_lifecycle import (
    read_runtime_checkpoint,
    RuntimeCheckpointEncodingPlan,
    RuntimeCheckpointEnvelope,
    write_runtime_checkpoint,
)
from ._mixed_matter import (
    SharedPeriodicGravityResult,
    WaveParticleCosmologyResult,
    WaveParticleGasCosmologyResult,
)
from ._particle_mesh import (
    CosmologicalParticleMeshPlan,
    CosmologicalParticleMeshResult,
)
from ._sidm import CosmologicalSIDMPlan, CosmologicalSIDMResult
from ._wave_dark_matter import (
    PreparedPeriodicWaveDarkMatter,
    WaveDarkMatterPoissonResult,
    WaveDarkMatterResult,
    WaveDarkMatterState,
)


_HEX_DIGITS = frozenset("0123456789abcdef")
_UNSPECIFIED_PARENT = object()
_STORE_VERIFIED_PARENT = object()


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return normalized


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(normalized))


def _fingerprint_id(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    normalized = _identifier(value, name)
    if len(normalized) != 64 or any(
        character not in _HEX_DIGITS for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase 256-bit content identifier.")
    return normalized


def _scalar(value: ArrayLike, name: str, /, *, inexact: bool = True) -> Array:
    result = jax.lax.stop_gradient(jnp.asarray(value))
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if inexact and not jnp.issubdtype(result.dtype, jnp.inexact):
        raise TypeError(f"{name} must have an inexact dtype.")
    return result


def _artifact_matches_status(
    artifact: ScientificArtifactEnvelope,
    successful: Array,
    /,
) -> None:
    if not isinstance(artifact, ScientificArtifactEnvelope):
        raise TypeError("artifact must be ScientificArtifactEnvelope.")
    succeeded = bool(np.asarray(successful))
    if succeeded != (artifact.status == "complete"):
        raise ValueError("Product evidence and artifact completion status disagree.")


def _evidence_check_value(
    status: SimulationProductStatusEvidence,
    name: str,
    /,
) -> bool:
    if name not in status.check_names:
        raise ValueError(f"Product evidence is missing required check {name!r}.")
    return bool(np.asarray(status.checks[status.check_names.index(name)]))


def _require_evidence_check(
    status: SimulationProductStatusEvidence,
    name: str,
    actual: bool,
    /,
) -> None:
    recorded = _evidence_check_value(status, name)
    if recorded != actual:
        raise ValueError(f"Product evidence check {name!r} contradicts its payload.")


def _lineage_valid(
    stable_ids: np.ndarray,
    incarnations: np.ndarray,
    lineage: np.ndarray,
    /,
) -> bool:
    identifiers = tuple(stable_ids.tolist())
    parents = tuple(lineage[:, 0].tolist())
    epochs = np.asarray(lineage[:, 1])
    if (
        np.any(incarnations < 0)
        or np.any(epochs < -1)
        or np.any((np.asarray(parents) == -1) != (epochs == -1))
    ):
        return False
    parent_by_id = dict(zip(identifiers, parents, strict=True))
    known = set(identifiers)
    if any(parent != -1 and parent not in known for parent in parents):
        return False
    for identifier in identifiers:
        visited: set[int] = set()
        current = identifier
        while current != -1:
            if current in visited:
                return False
            visited.add(current)
            current = parent_by_id[current]
    return True


def _tree_schema_id(tree: Any, role: str, /) -> str:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    if any(not eqx.is_array(leaf) for _, leaf in path_leaves):
        raise TypeError(f"{role} must be an array-only PyTree.")
    root_type = f"{type(tree).__module__}.{type(tree).__qualname__}"
    return canonical_fingerprint(
        {
            "kind": "dark-matter-checkpoint-tree-schema",
            "role": role,
            "root_type": root_type,
            "treedef": str(treedef),
            "leaves": [
                {
                    "path": jax.tree_util.keystr(path) or "<root>",
                    "shape": list(leaf.shape),
                    "dtype": np.dtype(leaf.dtype).str,
                }
                for path, leaf in path_leaves
            ],
        }
    )


class SimulationProductStatusEvidence(StrictModule, NonTrainableState):
    """Named physical/status checks retained by every native analysis product."""

    checks: Array
    check_names: tuple[str, ...] = eqx.field(static=True)
    successful: Array
    status_code: Array
    failure_reason: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        check_names: Sequence[str],
        checks: ArrayLike,
        /,
        *,
        failure_reason: str = "none",
    ):
        if not isinstance(check_names, Sequence) or isinstance(check_names, str):
            raise TypeError("check_names must be a sequence of identifiers.")
        names = tuple(
            _identifier(value, "product evidence check") for value in check_names
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("Product evidence check names must be nonempty and unique.")
        values = jax.lax.stop_gradient(
            jnp.asarray(checks, dtype=jnp.bool_).reshape((-1,))
        )
        if values.size != len(names):
            raise ValueError("Product evidence checks and names must align.")
        successful = jnp.all(values)
        succeeded = bool(np.asarray(successful))
        reason = _identifier(failure_reason, "failure_reason")
        if succeeded != (reason == "none"):
            raise ValueError(
                "Successful evidence requires exactly failure_reason='none'."
            )
        failed = ~values
        status = jnp.where(
            jnp.any(failed),
            jnp.argmax(failed).astype(jnp.int32) + 1,
            jnp.asarray(0, dtype=jnp.int32),
        )
        self.checks = values
        self.check_names = names
        self.successful = successful
        self.status_code = status
        self.failure_reason = reason
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "simulation-product-status-evidence",
                "check_names": list(names),
                "checks": array_tree_fingerprint(np.asarray(values)),
                "failure_reason": reason,
            }
        )


def _wave_state_content_id(psi: ArrayLike, scale_factor: ArrayLike, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "wave-snapshot-producer-state",
            "arrays": array_tree_fingerprint((np.asarray(psi), np.asarray(scale_factor))),
        }
    )


def _wave_result_content_id(result: WaveDarkMatterResult, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "wave-dark-matter-result",
            "prepared": result.prepared_id,
            "state": _wave_state_content_id(result.state.psi, result.state.scale_factor),
            "diagnostics": array_tree_fingerprint(result.diagnostics),
            "successful": bool(np.asarray(result.successful)),
        }
    )


def _gravity_result_content_id(result: SharedPeriodicGravityResult, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "shared-periodic-gravity-result",
            "plan": result.plan_id,
            "arrays": array_tree_fingerprint(result),
            "successful": bool(np.asarray(result.successful)),
        }
    )


class WaveSnapshotEvidence(StrictModule, NonTrainableState):
    norm: Array
    maximum_kinetic_phase: Array
    producer_prepared_id: str = eqx.field(static=True)
    maximum_potential_phase: Array
    de_broglie_nyquist_fraction: Array
    poisson_relative_residual: Array
    potential_zero_mode_absolute: Array
    status: SimulationProductStatusEvidence
    producer_result_id: str = eqx.field(static=True)
    producer_state_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, result: WaveDarkMatterResult, /):
        if not isinstance(result, WaveDarkMatterResult):
            raise TypeError("result must be WaveDarkMatterResult.")
        diagnostics = result.diagnostics
        values = (
            _scalar(diagnostics.norm[-1], "norm"),
            _scalar(diagnostics.maximum_kinetic_phase[-1], "maximum_kinetic_phase"),
            _scalar(
                diagnostics.maximum_potential_phase[-1],
                "maximum_potential_phase",
            ),
            _scalar(
                diagnostics.de_broglie_nyquist_fraction[-1],
                "de_broglie_nyquist_fraction",
            ),
            _scalar(
                diagnostics.poisson_relative_residual[-1],
                "poisson_relative_residual",
            ),
            _scalar(
                diagnostics.potential_zero_mode_absolute[-1],
                "potential_zero_mode_absolute",
            ),
        )
        accepted = bool(np.asarray(result.successful))
        final_accepted = bool(np.asarray(diagnostics.accepted[-1]))
        status = SimulationProductStatusEvidence(
            (
                "accepted",
                "finite",
                "phase-resolved",
                "de-broglie-resolved",
                "poisson-closed",
                "norm-conserved",
            ),
            jnp.asarray(
                (
                    accepted,
                    diagnostics.finite[-1],
                    diagnostics.phase_resolved[-1],
                    diagnostics.de_broglie_resolved[-1],
                    diagnostics.poisson_closed[-1],
                    final_accepted,
                ),
                dtype=jnp.bool_,
            ),
            failure_reason=(
                "none" if accepted else "wave-production-result-unsuccessful"
            ),
        )
        producer_result = _wave_result_content_id(result)
        producer_state = _wave_state_content_id(
            result.state.psi, result.state.scale_factor
        )
        (
            self.norm,
            self.maximum_kinetic_phase,
            self.maximum_potential_phase,
            self.de_broglie_nyquist_fraction,
            self.poisson_relative_residual,
            self.potential_zero_mode_absolute,
        ) = values
        self.status = status
        self.producer_prepared_id = result.prepared_id
        self.producer_result_id = producer_result
        self.producer_state_id = producer_state
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "wave-snapshot-evidence",
                "status": status.evidence_id,
                "producer_prepared": result.prepared_id,
                "producer_result": producer_result,
                "producer_state": producer_state,
                "values": array_tree_fingerprint(
                    tuple(np.asarray(value) for value in values)
                ),
            }
        )


class ParticleSnapshotEvidence(StrictModule, NonTrainableState):
    status: SimulationProductStatusEvidence
    producer_result_id: str = eqx.field(static=True)
    producer_state_id: str = eqx.field(static=True)
    producer_support_id: str = eqx.field(static=True)
    interaction_id: str = eqx.field(static=True)
    producer_scale_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: (
            CosmologicalParticleMeshResult
            | CosmologicalSIDMResult
            | WaveParticleCosmologyResult
            | WaveParticleGasCosmologyResult
        ),
        particles: ParticleDiscretization,
        /,
        *,
        producer_plan: CosmologicalParticleMeshPlan | CosmologicalSIDMPlan | None = None,
    ):
        if not isinstance(
            result,
            (
                CosmologicalParticleMeshResult,
                CosmologicalSIDMResult,
                WaveParticleCosmologyResult,
                WaveParticleGasCosmologyResult,
            ),
        ):
            raise TypeError("result must be a native cosmological particle result.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if isinstance(
            result, (WaveParticleCosmologyResult, WaveParticleGasCosmologyResult)
        ):
            if result.particle_support_id != particles.prepared_id:
                raise ValueError("Mixed particle result changed support identity.")
            interaction = result.particle_plan_id
            producer_scale_id = result.scale_id
            state = result.state.particles
        else:
            if not isinstance(
                producer_plan, (CosmologicalParticleMeshPlan, CosmologicalSIDMPlan)
            ):
                raise TypeError(
                    "Standalone particle results require their typed producer plan."
                )
            if isinstance(producer_plan, CosmologicalSIDMPlan):
                plan_particles = producer_plan.particle_mesh.kinematics.particles
                producer_scale_id = producer_plan.particle_mesh.kinematics.scale.scale_id
            else:
                plan_particles = producer_plan.kinematics.particles
                producer_scale_id = producer_plan.kinematics.scale.scale_id
            if plan_particles.prepared_id != particles.prepared_id:
                raise ValueError("Particle producer plan changed support.")
            interaction = producer_plan.plan_id
            state = result.state
        expected_shape = (particles.capacity, particles.ambient_dimension)
        support_matched = (
            state.positions.shape == expected_shape
            and state.canonical_momenta.shape == expected_shape
        )
        active = np.asarray(particles.active_mask, dtype=np.bool_)
        finite = (
            bool(np.all(np.isfinite(np.asarray(state.positions)[active])))
            and bool(np.all(np.isfinite(np.asarray(state.canonical_momenta)[active])))
            and bool(np.isfinite(np.asarray(state.scale_factor)))
        )
        stable_ids_unique = (
            len(set(np.asarray(particles.particle_ids).tolist())) == particles.capacity
        )
        accepted = bool(np.asarray(result.successful))
        status = SimulationProductStatusEvidence(
            (
                "accepted",
                "finite",
                "stable-ids-unique",
                "inactive-preserved",
                "support-matched",
                "interaction-valid",
            ),
            jnp.asarray(
                (
                    accepted,
                    finite,
                    stable_ids_unique,
                    accepted,
                    support_matched,
                    accepted,
                ),
                dtype=jnp.bool_,
            ),
            failure_reason=(
                "none" if accepted else "cosmological-particle-result-unsuccessful"
            ),
        )
        producer_state = canonical_fingerprint(
            {
                "kind": "cosmological-particle-state",
                "arrays": array_tree_fingerprint(state),
            }
        )
        producer_result = canonical_fingerprint(
            {
                "kind": "cosmological-particle-result",
                "state": producer_state,
                "arrays": array_tree_fingerprint(result),
                "support": particles.prepared_id,
                "interaction": interaction,
                "scale": producer_scale_id,
                "successful": accepted,
            }
        )
        self.status = status
        self.producer_result_id = producer_result
        self.producer_state_id = producer_state
        self.producer_support_id = particles.prepared_id
        self.interaction_id = interaction
        self.producer_scale_id = producer_scale_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "particle-snapshot-evidence",
                "status": status.evidence_id,
                "producer_result": producer_result,
                "producer_state": producer_state,
                "producer_support": particles.prepared_id,
                "interaction": interaction,
                "producer_scale": producer_scale_id,
            }
        )


class GasSnapshotEvidence(StrictModule, NonTrainableState):
    minimum_density: Array
    minimum_internal_energy: Array
    status: SimulationProductStatusEvidence
    producer_result_id: str = eqx.field(static=True)
    producer_state_id: str = eqx.field(static=True)
    producer_prepared_id: str = eqx.field(static=True)
    producer_identity: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: WaveParticleGasCosmologyResult,
        /,
    ):
        if not isinstance(result, WaveParticleGasCosmologyResult):
            raise TypeError("result must be WaveParticleGasCosmologyResult.")
        gas = result.state.gas
        fields = jnp.asarray(gas.cell_average)
        density = fields[..., 0]
        momentum = fields[..., 1:-1]
        total_energy = fields[..., -1]
        safe_density = jnp.where(density > 0.0, density, 1.0)
        internal_energy = total_energy - jnp.sum(momentum**2, axis=-1) / (
            2.0 * safe_density
        )
        minimum_density = _scalar(jnp.min(density), "minimum_density")
        minimum_internal = _scalar(jnp.min(internal_energy), "minimum_internal_energy")
        accepted = bool(np.asarray(result.successful))
        finite = bool(np.all(np.isfinite(np.asarray(fields))))
        density_positive = bool(np.all(np.asarray(density) > 0.0))
        thermodynamic_valid = bool(np.all(np.asarray(internal_energy) > 0.0))
        status = SimulationProductStatusEvidence(
            (
                "accepted",
                "finite",
                "density-positive",
                "thermodynamic-valid",
                "source-closed",
            ),
            jnp.asarray(
                (
                    accepted,
                    finite,
                    density_positive,
                    thermodynamic_valid,
                    accepted,
                ),
                dtype=jnp.bool_,
            ),
            failure_reason=(
                "none" if accepted else "cosmological-gas-result-unsuccessful"
            ),
        )
        prepared_id = result.prepared_id
        producer_identity = (
            result.assembler_id,
            result.gas_plan_id,
            result.gravity_plan_id,
            result.prepared_id,
            result.scale_id,
            "accepted-end-scale-factor",
        )
        producer_state = canonical_fingerprint(
            {
                "kind": "cosmological-gas-state",
                "arrays": array_tree_fingerprint((gas.cell_average, gas.scale_factor)),
            }
        )
        producer = canonical_fingerprint(
            {
                "kind": "cosmological-gas-result",
                "prepared": prepared_id,
                "state": producer_state,
                "arrays": array_tree_fingerprint(result),
                "successful": accepted,
            }
        )
        self.minimum_density = minimum_density
        self.minimum_internal_energy = minimum_internal
        self.status = status
        self.producer_result_id = producer
        self.producer_prepared_id = prepared_id
        self.producer_state_id = producer_state
        self.producer_identity = producer_identity
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "gas-snapshot-evidence",
                "status": status.evidence_id,
                "producer_result": producer,
                "producer_prepared": prepared_id,
                "producer_state": producer_state,
                "producer_identity": list(producer_identity),
                "values": array_tree_fingerprint(
                    (np.asarray(minimum_density), np.asarray(minimum_internal))
                ),
            }
        )


class CommonGravitySnapshotEvidence(StrictModule, NonTrainableState):
    poisson_relative_residual: Array
    potential_zero_mode_absolute: Array
    source_balance_defect: Array
    net_force_defect: Array
    status: SimulationProductStatusEvidence
    producer_result_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, result: SharedPeriodicGravityResult, /):
        if not isinstance(result, SharedPeriodicGravityResult):
            raise TypeError("result must be SharedPeriodicGravityResult.")
        values = (
            _scalar(result.poisson_relative_residual, "poisson_relative_residual"),
            _scalar(result.gauge_defect, "potential_zero_mode_absolute"),
            _scalar(
                result.assembly.particle_mass_balance_defect,
                "source_balance_defect",
            ),
            _scalar(
                jnp.sqrt(jnp.sum(result.total_force**2)),
                "net_force_defect",
            ),
        )
        accepted = bool(np.asarray(result.successful))
        status = SimulationProductStatusEvidence(
            (
                "accepted",
                "finite",
                "poisson-closed",
                "zero-mode-removed",
                "source-closed",
                "force-closed",
            ),
            jnp.asarray(
                (
                    result.successful,
                    result.finite,
                    result.successful,
                    result.successful,
                    result.assembly.successful,
                    result.successful,
                ),
                dtype=jnp.bool_,
            ),
            failure_reason=("none" if accepted else "shared-gravity-result-unsuccessful"),
        )
        producer = _gravity_result_content_id(result)
        (
            self.poisson_relative_residual,
            self.potential_zero_mode_absolute,
            self.source_balance_defect,
            self.net_force_defect,
        ) = values
        self.status = status
        self.producer_result_id = producer
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "common-gravity-snapshot-evidence",
                "status": status.evidence_id,
                "producer_result": producer,
                "values": array_tree_fingerprint(
                    tuple(np.asarray(value) for value in values)
                ),
            }
        )


class WaveSimulationSnapshot(StrictModule, NonTrainableState):
    psi: Array
    scale_factor: Array
    density: Array | None
    potential: Array | None
    evidence: WaveSnapshotEvidence
    artifact: ScientificArtifactEnvelope
    grid_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_time_level: str = eqx.field(static=True)
    producer_result_id: str = eqx.field(static=True)
    poisson_result_id: str | None = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        psi: ArrayLike,
        scale_factor: ArrayLike,
        evidence: WaveSnapshotEvidence,
        artifact: ScientificArtifactEnvelope,
        /,
        *,
        prepared: PreparedPeriodicWaveDarkMatter,
        grid_id: str,
        physics_id: str,
        solver_id: str,
        scale_id: str,
        coordinate_time_level: str,
        density: ArrayLike | None = None,
        potential: ArrayLike | None = None,
        poisson_result: WaveDarkMatterPoissonResult | None = None,
    ):
        value = jax.lax.stop_gradient(jnp.asarray(psi))
        scale = _scalar(scale_factor, "scale_factor")
        density_ = (
            None if density is None else jax.lax.stop_gradient(jnp.asarray(density))
        )
        potential_ = (
            None if potential is None else jax.lax.stop_gradient(jnp.asarray(potential))
        )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Wave snapshot psi must have a complex dtype.")
        if density_ is not None and density_.shape != value.shape:
            raise ValueError("Materialized wave density must match psi shape.")
        if potential_ is not None and potential_.shape != value.shape:
            raise ValueError("Materialized wave potential must match psi shape.")
        if not isinstance(evidence, WaveSnapshotEvidence):
            raise TypeError("evidence must be WaveSnapshotEvidence.")
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        if prepared.prepared_id != evidence.producer_prepared_id:
            raise ValueError("Wave snapshot substituted its prepared solver identity.")
        state_id = _wave_state_content_id(value, scale)
        if state_id != evidence.producer_state_id:
            raise ValueError(
                "Wave snapshot payload does not match its solver-produced result."
            )
        if density_ is not None or potential_ is not None:
            if not isinstance(poisson_result, WaveDarkMatterPoissonResult):
                raise ValueError(
                    "Materialized wave fields require their typed Poisson result."
                )
            authoritative_poisson = prepared.poisson(WaveDarkMatterState(value, scale))
            if (
                density_ is None
                or potential_ is None
                or not np.array_equal(
                    np.asarray(density_), np.asarray(poisson_result.density)
                )
                or not np.array_equal(
                    np.asarray(potential_), np.asarray(poisson_result.potential)
                )
                or not np.array_equal(
                    np.asarray(density_), np.asarray(authoritative_poisson.density)
                )
                or not np.array_equal(
                    np.asarray(potential_),
                    np.asarray(authoritative_poisson.potential),
                )
                or not bool(np.asarray(poisson_result.successful))
                or not bool(np.asarray(authoritative_poisson.successful))
            ):
                raise ValueError("Materialized wave fields changed their Poisson result.")
            poisson_result_id = canonical_fingerprint(
                {
                    "kind": "wave-dark-matter-poisson-result",
                    "arrays": array_tree_fingerprint(poisson_result),
                    "prepared": prepared.prepared_id,
                    "source_state": state_id,
                }
            )
        elif poisson_result is not None:
            raise ValueError(
                "A Poisson result requires intentionally materialized fields."
            )
        else:
            poisson_result_id = None
        finite = (
            bool(np.all(np.isfinite(np.asarray(value))))
            and bool(np.isfinite(np.asarray(scale)))
            and float(np.asarray(scale)) > 0.0
            and (density_ is None or bool(np.all(np.isfinite(np.asarray(density_)))))
            and (potential_ is None or bool(np.all(np.isfinite(np.asarray(potential_)))))
            and all(
                bool(np.isfinite(np.asarray(item)))
                for item in (
                    evidence.norm,
                    evidence.maximum_kinetic_phase,
                    evidence.maximum_potential_phase,
                    evidence.de_broglie_nyquist_fraction,
                    evidence.poisson_relative_residual,
                    evidence.potential_zero_mode_absolute,
                )
            )
        )
        _require_evidence_check(evidence.status, "finite", finite)
        _artifact_matches_status(artifact, evidence.status.successful)
        identities = tuple(
            _identifier(item, name)
            for item, name in (
                (grid_id, "grid_id"),
                (physics_id, "physics_id"),
                (solver_id, "solver_id"),
                (scale_id, "scale_id"),
                (coordinate_time_level, "coordinate_time_level"),
            )
        )
        expected_identities = (
            prepared.discretization.prepared_id,
            prepared.plan.plan_id,
            prepared.prepared_id,
            prepared.scale_id,
            "accepted-end-scale-factor",
        )
        if identities != expected_identities:
            raise ValueError(
                "Wave snapshot grid, physics, solver, scale, or time identity changed."
            )
        arrays = (value, scale, density_, potential_)
        self.psi = value
        self.scale_factor = scale
        self.density = density_
        self.potential = potential_
        self.evidence = evidence
        self.artifact = artifact
        (
            self.grid_id,
            self.physics_id,
            self.solver_id,
            self.scale_id,
            self.coordinate_time_level,
        ) = identities
        self.producer_result_id = evidence.producer_result_id
        self.poisson_result_id = poisson_result_id
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "wave-simulation-snapshot",
                "identities": list(identities),
                "artifact": artifact.artifact_id,
                "evidence": evidence.evidence_id,
                "producer_result": evidence.producer_result_id,
                "poisson_result": poisson_result_id,
                "arrays": array_tree_fingerprint(arrays),
            }
        )


class ParticleSimulationSnapshot(StrictModule, NonTrainableState):
    stable_ids: Array
    positions: Array
    canonical_momenta: Array
    active_mask: Array
    macro_masses: Array
    packet_weights: Array
    incarnations: Array
    lineage_ids: Array
    scale_factor: Array
    evidence: ParticleSnapshotEvidence
    artifact: ScientificArtifactEnvelope
    support_id: str = eqx.field(static=True)
    interaction_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_time_level: str = eqx.field(static=True)
    inactive_reference_id: str | None = eqx.field(static=True)
    producer_result_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        stable_ids: ArrayLike,
        positions: ArrayLike,
        canonical_momenta: ArrayLike,
        active_mask: ArrayLike,
        macro_masses: ArrayLike,
        packet_weights: ArrayLike,
        incarnations: ArrayLike,
        lineage_ids: ArrayLike,
        scale_factor: ArrayLike,
        evidence: ParticleSnapshotEvidence,
        artifact: ScientificArtifactEnvelope,
        producer_result: (
            CosmologicalParticleMeshResult
            | CosmologicalSIDMResult
            | WaveParticleCosmologyResult
            | WaveParticleGasCosmologyResult
        ),
        particle_support: ParticleDiscretization,
        /,
        *,
        support_id: str,
        interaction_id: str,
        physics_id: str,
        scale_id: str,
        coordinate_time_level: str,
        inactive_reference_positions: ArrayLike | None = None,
        inactive_reference_momenta: ArrayLike | None = None,
        inactive_reference_masses: ArrayLike | None = None,
        inactive_reference_weights: ArrayLike | None = None,
    ):
        ids = jax.lax.stop_gradient(jnp.asarray(stable_ids))
        position = jax.lax.stop_gradient(jnp.asarray(positions))
        momentum = jax.lax.stop_gradient(
            jnp.asarray(canonical_momenta, dtype=position.dtype)
        )
        active = jax.lax.stop_gradient(jnp.asarray(active_mask, dtype=jnp.bool_))
        mass = jax.lax.stop_gradient(jnp.asarray(macro_masses, dtype=position.dtype))
        weights = jax.lax.stop_gradient(jnp.asarray(packet_weights, dtype=position.dtype))
        incarnation = jax.lax.stop_gradient(jnp.asarray(incarnations))
        lineage = jax.lax.stop_gradient(jnp.asarray(lineage_ids))
        scale = _scalar(scale_factor, "scale_factor")
        count = ids.size
        if (
            ids.ndim != 1
            or not jnp.issubdtype(ids.dtype, jnp.integer)
            or position.ndim != 2
            or momentum.shape != position.shape
            or position.shape[0] != count
            or active.shape != (count,)
            or mass.shape != (count,)
            or weights.shape != (count,)
            or incarnation.shape != (count,)
            or not jnp.issubdtype(incarnation.dtype, jnp.integer)
            or lineage.shape != (count, 2)
            or not jnp.issubdtype(lineage.dtype, jnp.integer)
        ):
            raise ValueError("Particle snapshot capacity arrays are inconsistent.")
        if not isinstance(evidence, ParticleSnapshotEvidence):
            raise TypeError("evidence must be ParticleSnapshotEvidence.")
        if not isinstance(
            producer_result,
            (
                CosmologicalParticleMeshResult,
                CosmologicalSIDMResult,
                WaveParticleCosmologyResult,
                WaveParticleGasCosmologyResult,
            ),
        ):
            raise TypeError("producer_result must be a native particle result.")
        if not isinstance(particle_support, ParticleDiscretization):
            raise TypeError("particle_support must be ParticleDiscretization.")
        typed_state = (
            producer_result.state.particles
            if isinstance(
                producer_result,
                (WaveParticleCosmologyResult, WaveParticleGasCosmologyResult),
            )
            else producer_result.state
        )
        producer_state = canonical_fingerprint(
            {
                "kind": "cosmological-particle-state",
                "arrays": array_tree_fingerprint(typed_state),
            }
        )
        if (
            producer_state != evidence.producer_state_id
            or particle_support.prepared_id != evidence.producer_support_id
            or not np.array_equal(
                np.asarray(ids), np.asarray(particle_support.particle_ids)
            )
            or not np.array_equal(
                np.asarray(active), np.asarray(particle_support.active_mask)
            )
            or not np.array_equal(
                np.asarray(mass), np.asarray(particle_support.plan.masses)
            )
            or not np.array_equal(np.asarray(position), np.asarray(typed_state.positions))
            or not np.array_equal(
                np.asarray(momentum), np.asarray(typed_state.canonical_momenta)
            )
            or not np.array_equal(np.asarray(scale), np.asarray(typed_state.scale_factor))
            or not np.array_equal(np.asarray(weights), np.ones((count,)))
            or not np.array_equal(np.asarray(incarnation), np.zeros((count,)))
            or not np.array_equal(np.asarray(lineage), -np.ones((count, 2)))
        ):
            raise ValueError("Particle snapshot changed its typed result or support.")
        ids_host = np.asarray(ids)
        active_host = np.asarray(active, dtype=np.bool_)
        incarnation_host = np.asarray(incarnation)
        lineage_host = np.asarray(lineage)
        stable_ids_unique = len(set(ids_host.tolist())) == count
        inactive = ~active_host
        references = (
            inactive_reference_positions,
            inactive_reference_momenta,
            inactive_reference_masses,
            inactive_reference_weights,
        )
        if np.any(inactive) and any(value is None for value in references):
            raise ValueError(
                "Inactive particle slots require an explicit preservation reference."
            )
        if any(value is not None for value in references):
            if any(value is None for value in references):
                raise ValueError(
                    "Inactive particle references must be supplied together."
                )
            reference_arrays = tuple(np.asarray(value) for value in references)
            actual_arrays = tuple(
                np.asarray(value) for value in (position, momentum, mass, weights)
            )
            if any(
                reference.shape != actual.shape
                for reference, actual in zip(reference_arrays, actual_arrays, strict=True)
            ):
                raise ValueError("Inactive particle reference shapes changed.")
            inactive_preserved = all(
                np.array_equal(actual[inactive], reference[inactive], equal_nan=True)
                for actual, reference in zip(actual_arrays, reference_arrays, strict=True)
            )
            inactive_reference_id = canonical_fingerprint(
                {
                    "kind": "particle-inactive-slot-reference",
                    "arrays": array_tree_fingerprint(reference_arrays),
                    "mask": array_tree_fingerprint(active),
                }
            )
        else:
            inactive_preserved = True
            inactive_reference_id = None
        finite = (
            bool(np.isfinite(np.asarray(scale)))
            and float(np.asarray(scale)) > 0.0
            and bool(np.all(np.isfinite(np.asarray(position)[active_host])))
            and bool(np.all(np.isfinite(np.asarray(momentum)[active_host])))
            and bool(np.all(np.isfinite(np.asarray(mass)[active_host])))
            and bool(np.all(np.isfinite(np.asarray(weights)[active_host])))
        )
        if not _lineage_valid(ids_host, incarnation_host, lineage_host):
            raise ValueError(
                "Particle snapshot incarnation or lineage invariants are invalid."
            )
        _require_evidence_check(evidence.status, "finite", finite)
        _require_evidence_check(evidence.status, "stable-ids-unique", stable_ids_unique)
        _require_evidence_check(evidence.status, "inactive-preserved", inactive_preserved)
        _artifact_matches_status(artifact, evidence.status.successful)
        identities = tuple(
            _identifier(item, name)
            for item, name in (
                (support_id, "support_id"),
                (interaction_id, "interaction_id"),
                (physics_id, "physics_id"),
                (scale_id, "scale_id"),
                (coordinate_time_level, "coordinate_time_level"),
            )
        )
        if (
            identities[0] != evidence.producer_support_id
            or identities[1] != evidence.interaction_id
            or identities[2] != evidence.producer_result_id
            or identities[4] != "accepted-end-scale-factor"
            or identities[3] != evidence.producer_scale_id
        ):
            raise ValueError(
                "Particle snapshot support, interaction, physics, or time identity changed."
            )
        self.stable_ids = ids
        self.positions = position
        self.canonical_momenta = momentum
        self.active_mask = active
        self.macro_masses = mass
        self.packet_weights = weights
        self.incarnations = incarnation
        self.lineage_ids = lineage
        self.scale_factor = scale
        self.evidence = evidence
        self.artifact = artifact
        (
            self.support_id,
            self.interaction_id,
            self.physics_id,
            self.scale_id,
            self.coordinate_time_level,
        ) = identities
        self.inactive_reference_id = inactive_reference_id
        self.producer_result_id = evidence.producer_result_id
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "particle-simulation-snapshot",
                "identities": list(identities),
                "artifact": artifact.artifact_id,
                "evidence": evidence.evidence_id,
                "producer_result": evidence.producer_result_id,
                "arrays": array_tree_fingerprint(
                    (
                        ids,
                        position,
                        momentum,
                        active,
                        mass,
                        weights,
                        incarnation,
                        lineage,
                        scale,
                    )
                ),
                "inactive_reference": inactive_reference_id,
            }
        )


class GasSimulationSnapshot(StrictModule, NonTrainableState):
    conserved_fields: Array
    scale_factor: Array
    evidence: GasSnapshotEvidence
    artifact: ScientificArtifactEnvelope
    component_names: tuple[str, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_time_level: str = eqx.field(static=True)
    producer_result_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        conserved_fields: ArrayLike,
        scale_factor: ArrayLike,
        evidence: GasSnapshotEvidence,
        artifact: ScientificArtifactEnvelope,
        /,
        *,
        component_names: Sequence[str],
        geometry_id: str,
        eos_id: str,
        source_id: str,
        physics_id: str,
        scale_id: str,
        coordinate_time_level: str,
    ):
        fields = jax.lax.stop_gradient(jnp.asarray(conserved_fields))
        scale = _scalar(scale_factor, "scale_factor")
        if not isinstance(component_names, Sequence) or isinstance(component_names, str):
            raise TypeError("component_names must be a sequence of identifiers.")
        components = tuple(
            _identifier(value, "gas component name") for value in component_names
        )
        if not components or len(set(components)) != len(components):
            raise ValueError("Gas component names must be nonempty and unique.")
        if fields.ndim < 1 or fields.shape[-1] != len(components):
            raise ValueError("Gas conserved fields must end in the declared components.")
        if not isinstance(evidence, GasSnapshotEvidence):
            raise TypeError("evidence must be GasSnapshotEvidence.")
        density_names = {
            "density",
            "mass",
            "comoving-density",
            "comoving-mass-density",
        }
        density_indices = tuple(
            index for index, name in enumerate(components) if name in density_names
        )
        if len(density_indices) != 1:
            raise ValueError(
                "Gas snapshot must declare exactly one conserved density component."
            )
        minimum_density = jnp.min(fields[..., density_indices[0]])
        density_positive = bool(np.asarray(minimum_density > 0.0))
        finite = (
            bool(np.all(np.isfinite(np.asarray(fields))))
            and bool(np.isfinite(np.asarray(scale)))
            and float(np.asarray(scale)) > 0.0
            and bool(np.isfinite(np.asarray(evidence.minimum_density)))
            and bool(np.isfinite(np.asarray(evidence.minimum_internal_energy)))
        )
        if not np.array_equal(
            np.asarray(minimum_density), np.asarray(evidence.minimum_density)
        ):
            raise ValueError("Gas minimum-density evidence contradicts conserved fields.")
        _require_evidence_check(evidence.status, "finite", finite)
        _require_evidence_check(evidence.status, "density-positive", density_positive)
        _artifact_matches_status(artifact, evidence.status.successful)
        identities = tuple(
            _identifier(item, name)
            for item, name in (
                (geometry_id, "geometry_id"),
                (eos_id, "eos_id"),
                (source_id, "source_id"),
                (physics_id, "physics_id"),
                (scale_id, "scale_id"),
                (coordinate_time_level, "coordinate_time_level"),
            )
        )
        producer_state = canonical_fingerprint(
            {
                "kind": "cosmological-gas-state",
                "arrays": array_tree_fingerprint((fields, scale)),
            }
        )
        if (
            producer_state != evidence.producer_state_id
            or identities != evidence.producer_identity
        ):
            raise ValueError("Gas payload changed its typed producer result.")
        self.conserved_fields = fields
        self.scale_factor = scale
        self.evidence = evidence
        self.artifact = artifact
        self.component_names = components
        self.producer_result_id = evidence.producer_result_id
        (
            self.geometry_id,
            self.eos_id,
            self.source_id,
            self.physics_id,
            self.scale_id,
            self.coordinate_time_level,
        ) = identities
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "gas-simulation-snapshot",
                "components": list(components),
                "identities": list(identities),
                "artifact": artifact.artifact_id,
                "evidence": evidence.evidence_id,
                "producer_result": evidence.producer_result_id,
                "arrays": array_tree_fingerprint((fields, scale)),
            }
        )


class CommonGravitySimulationSnapshot(StrictModule, NonTrainableState):
    total_comoving_density: Array
    potential: Array
    cell_acceleration: Array
    scale_factor: Array
    evidence: CommonGravitySnapshotEvidence
    artifact: ScientificArtifactEnvelope
    source_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    producer_result_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    potential_time_level: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_comoving_density: ArrayLike,
        potential: ArrayLike,
        cell_acceleration: ArrayLike,
        scale_factor: ArrayLike,
        evidence: CommonGravitySnapshotEvidence,
        artifact: ScientificArtifactEnvelope,
        /,
        *,
        source_id: str,
        operator_id: str,
        geometry_id: str,
        physics_id: str,
        scale_id: str,
        potential_time_level: str,
        producer_result: SharedPeriodicGravityResult,
    ):
        density = jax.lax.stop_gradient(jnp.asarray(total_comoving_density))
        potential_ = jax.lax.stop_gradient(jnp.asarray(potential, dtype=density.dtype))
        acceleration = jax.lax.stop_gradient(
            jnp.asarray(cell_acceleration, dtype=density.dtype)
        )
        scale = _scalar(scale_factor, "scale_factor")
        if potential_.shape != density.shape or acceleration.shape[:-1] != density.shape:
            raise ValueError("Common-gravity field shapes are inconsistent.")
        if not isinstance(evidence, CommonGravitySnapshotEvidence):
            raise TypeError("evidence must be CommonGravitySnapshotEvidence.")
        if not isinstance(producer_result, SharedPeriodicGravityResult):
            raise TypeError("producer_result must be SharedPeriodicGravityResult.")
        producer_id = _gravity_result_content_id(producer_result)
        if (
            evidence.producer_result_id != producer_id
            or not np.array_equal(
                np.asarray(density),
                np.asarray(producer_result.assembly.total_density),
            )
            or not np.array_equal(
                np.asarray(potential_), np.asarray(producer_result.potential)
            )
            or not np.array_equal(
                np.asarray(acceleration),
                np.asarray(producer_result.cell_acceleration),
            )
            or not np.array_equal(
                np.asarray(scale),
                np.asarray(producer_result.assembly.scale_factor),
            )
        ):
            raise ValueError("Common-gravity payload changed its typed producer result.")
        finite = (
            bool(np.all(np.isfinite(np.asarray(density))))
            and bool(np.all(np.isfinite(np.asarray(potential_))))
            and bool(np.all(np.isfinite(np.asarray(acceleration))))
            and bool(np.isfinite(np.asarray(scale)))
            and float(np.asarray(scale)) > 0.0
            and all(
                bool(np.isfinite(np.asarray(item)))
                for item in (
                    evidence.poisson_relative_residual,
                    evidence.potential_zero_mode_absolute,
                    evidence.source_balance_defect,
                    evidence.net_force_defect,
                )
            )
        )
        _require_evidence_check(evidence.status, "finite", finite)
        _artifact_matches_status(artifact, evidence.status.successful)
        identities = tuple(
            _identifier(item, name)
            for item, name in (
                (source_id, "source_id"),
                (operator_id, "operator_id"),
                (geometry_id, "geometry_id"),
                (physics_id, "physics_id"),
                (scale_id, "scale_id"),
                (potential_time_level, "potential_time_level"),
            )
        )
        expected_identities = (
            producer_result.assembly.assembler_id,
            producer_result.plan_id,
            producer_result.assembly.assembler_id,
            producer_result.plan_id,
            producer_result.assembly.scale_id,
            "accepted-end-scale-factor",
        )
        if identities != expected_identities:
            raise ValueError(
                "Common-gravity source, operator, geometry, physics, scale, or time-level identity changed."
            )
        self.total_comoving_density = density
        self.potential = potential_
        self.cell_acceleration = acceleration
        self.scale_factor = scale
        self.evidence = evidence
        self.artifact = artifact
        (
            self.source_id,
            self.operator_id,
            self.geometry_id,
            self.physics_id,
            self.scale_id,
            self.potential_time_level,
        ) = identities
        self.producer_result_id = producer_id
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "common-gravity-simulation-snapshot",
                "identities": list(identities),
                "artifact": artifact.artifact_id,
                "evidence": evidence.evidence_id,
                "producer_result": producer_id,
                "arrays": array_tree_fingerprint(
                    (density, potential_, acceleration, scale)
                ),
            }
        )


class CosmologyOutputBundle(StrictModule, NonTrainableState):
    scale_factor: Array
    wave: WaveSimulationSnapshot | None
    particles: ParticleSimulationSnapshot | None
    gas: GasSimulationSnapshot | None
    common_gravity: CommonGravitySimulationSnapshot | None
    artifact: ScientificArtifactEnvelope
    status: SimulationProductStatusEvidence
    child_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    shared_gravity_result_id: str | None = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    parent_checkpoint_id: str | None = eqx.field(static=True)
    output_cursor: int = eqx.field(static=True)
    cosmology_id: str = eqx.field(static=True)
    coordinate_time_level: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    producer_result_id: str | None = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factor: ArrayLike,
        artifact: ScientificArtifactEnvelope,
        /,
        *,
        execution_id: str,
        parent_checkpoint_id: str | None,
        output_cursor: int,
        cosmology_id: str,
        scale_id: str,
        wave: WaveSimulationSnapshot | None = None,
        particles: ParticleSimulationSnapshot | None = None,
        gas: GasSimulationSnapshot | None = None,
        common_gravity: CommonGravitySimulationSnapshot | None = None,
        producer_result: (
            WaveParticleCosmologyResult | WaveParticleGasCosmologyResult | None
        ) = None,
    ):
        scale = _scalar(scale_factor, "scale_factor")
        children = tuple(
            value for value in (wave, particles, gas, common_gravity) if value is not None
        )
        if not children:
            raise ValueError(
                "Cosmology output bundle requires at least one typed snapshot."
            )
        expected_types = (
            (wave, WaveSimulationSnapshot),
            (particles, ParticleSimulationSnapshot),
            (gas, GasSimulationSnapshot),
            (common_gravity, CommonGravitySimulationSnapshot),
        )
        if any(
            value is not None and not isinstance(value, kind)
            for value, kind in expected_types
        ):
            raise TypeError("Cosmology output bundle contains an invalid snapshot type.")
        for child in children:
            if child.scale_id != scale_id or not np.array_equal(
                np.asarray(child.scale_factor), np.asarray(scale)
            ):
                raise ValueError(
                    "Cosmology bundle children must share one exact scale level."
                )
        time_levels = tuple(
            value
            for value in (
                None if wave is None else wave.coordinate_time_level,
                None if particles is None else particles.coordinate_time_level,
                None if gas is None else gas.coordinate_time_level,
                (None if common_gravity is None else common_gravity.potential_time_level),
            )
            if value is not None
        )
        if set(time_levels) != {"accepted-end-scale-factor"}:
            raise ValueError(
                "Cosmology bundle children must use the accepted-end time level."
            )
        coordinate_time_level = "accepted-end-scale-factor"
        if len(children) > 1 or common_gravity is not None:
            if not isinstance(
                producer_result,
                (WaveParticleCosmologyResult, WaveParticleGasCosmologyResult),
            ):
                raise ValueError(
                    "Coupled bundles require their typed mixed-cosmology result."
                )
            if wave is not None:
                raise ValueError(
                    "Typed mixed-wave snapshot evidence is not yet available."
                )
            mixed_wave_id = _wave_state_content_id(
                producer_result.state.wave.psi,
                producer_result.state.wave.scale_factor,
            )
            mixed_particle_id = canonical_fingerprint(
                {
                    "kind": "cosmological-particle-state",
                    "arrays": array_tree_fingerprint(producer_result.state.particles),
                }
            )
            gas_mismatch = gas is not None and (
                not isinstance(producer_result, WaveParticleGasCosmologyResult)
                or gas.evidence.producer_result_id
                != GasSnapshotEvidence(producer_result).producer_result_id
            )
            gravity_mismatch = (
                common_gravity is not None
                and common_gravity.producer_result_id
                != _gravity_result_content_id(producer_result.gravity)
            )
            if (
                (
                    wave is not None
                    and (
                        wave.evidence.producer_state_id != mixed_wave_id
                        or wave.evidence.producer_prepared_id
                        != producer_result.wave_prepared_id
                    )
                )
                or (
                    particles is not None
                    and (
                        particles.evidence.producer_state_id != mixed_particle_id
                        or particles.evidence.interaction_id
                        != producer_result.particle_plan_id
                        or particles.evidence.producer_scale_id
                        != producer_result.scale_id
                    )
                )
                or gas_mismatch
                or gravity_mismatch
            ):
                raise ValueError(
                    "Bundle children do not belong to one mixed producer result."
                )
            producer_result_id = canonical_fingerprint(
                {
                    "kind": "mixed-cosmology-output-result",
                    "prepared": producer_result.prepared_id,
                    "arrays": array_tree_fingerprint(producer_result),
                }
            )
        elif producer_result is not None:
            raise ValueError("Unused mixed producer result was supplied.")
        else:
            producer_result_id = None
        child_artifacts = tuple(child.artifact.artifact_id for child in children)
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if not set(child_artifacts).issubset(set(artifact.parent_artifact_ids)):
            raise ValueError(
                "Bundle artifact must retain every child artifact as a parent."
            )
        child_success = tuple(child.evidence.status.successful for child in children)
        all_children = all(bool(np.asarray(value)) for value in child_success)
        status = SimulationProductStatusEvidence(
            ("children-successful", "artifact-complete"),
            jnp.asarray((all_children, artifact.status == "complete"), dtype=jnp.bool_),
            failure_reason=(
                "none"
                if artifact.status == "complete" and all_children
                else artifact.failure_reason
            ),
        )
        _artifact_matches_status(artifact, status.successful)
        cursor = int(output_cursor)
        if cursor < 0:
            raise ValueError("output_cursor must be nonnegative.")
        execution = _identifier(execution_id, "execution_id")
        cosmology = _identifier(cosmology_id, "cosmology_id")
        scale_identity = _identifier(scale_id, "scale_id")
        parent = _fingerprint_id(parent_checkpoint_id, "parent_checkpoint_id")
        gravity_id = None if common_gravity is None else common_gravity.producer_result_id
        self.scale_factor = scale
        self.wave = wave
        self.particles = particles
        self.gas = gas
        self.common_gravity = common_gravity
        self.artifact = artifact
        self.status = status
        self.child_artifact_ids = child_artifacts
        self.shared_gravity_result_id = gravity_id
        self.execution_id = execution
        self.parent_checkpoint_id = parent
        self.output_cursor = cursor
        self.cosmology_id = cosmology
        self.scale_id = scale_identity
        self.coordinate_time_level = coordinate_time_level
        self.producer_result_id = producer_result_id
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "cosmology-output-bundle",
                "scale_factor": array_tree_fingerprint(np.asarray(scale)),
                "children": [child.snapshot_id for child in children],
                "child_artifacts": list(child_artifacts),
                "gravity": gravity_id,
                "execution": execution,
                "parent_checkpoint": parent,
                "output_cursor": cursor,
                "cosmology": cosmology,
                "scale": scale_identity,
                "coordinate_time_level": coordinate_time_level,
                "producer_result": producer_result_id,
                "artifact": artifact.artifact_id,
                "status": status.evidence_id,
            }
        )


_ANALYSIS_PRODUCT_TYPES = (
    WaveSimulationSnapshot,
    ParticleSimulationSnapshot,
    GasSimulationSnapshot,
    CommonGravitySimulationSnapshot,
    CosmologyOutputBundle,
)


def _contains_analysis_product(
    value: Any,
    /,
    *,
    seen: set[int] | None = None,
) -> bool:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return False
    visited.add(identity)
    if isinstance(value, _ANALYSIS_PRODUCT_TYPES):
        return True
    if isinstance(value, dict):
        return any(
            _contains_analysis_product(item, seen=visited) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_analysis_product(item, seen=visited) for item in value)
    if is_dataclass(value) and not eqx.is_array(value):
        return any(
            _contains_analysis_product(
                object.__getattribute__(value, field.name), seen=visited
            )
            for field in fields(value)
        )
    return False


class DarkMatterRestartSnapshot(StrictModule, NonTrainableState):
    """Accepted restart state; this is deliberately not an analysis product."""

    component_state: Any
    stable_ids: Array
    active_mask: Array
    incarnations: Array
    lineage_ids: Array
    prng_root: Array
    event_epoch: Array
    time: Array
    accepted_step: Array
    schedule_cursor: Array
    output_cursor: Array
    accepted_evidence: Any
    parent_checkpoint_id: str | None = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_state: Any,
        /,
        *,
        stable_ids: ArrayLike = (),
        active_mask: ArrayLike = (),
        incarnations: ArrayLike = (),
        lineage_ids: ArrayLike = np.empty((0, 2), dtype=np.int64),
        prng_root: ArrayLike = (),
        event_epoch: ArrayLike = 0,
        time: ArrayLike,
        accepted_step: ArrayLike,
        schedule_cursor: ArrayLike,
        output_cursor: ArrayLike,
        accepted_evidence: Any = (),
        parent_checkpoint_id: str | None = None,
    ):
        if _contains_analysis_product(component_state):
            raise TypeError(
                "Analysis products cannot be promoted to restart component state."
            )
        if not isinstance(component_state, StrictModule):
            raise TypeError(
                "Restart component_state must be an explicit native StrictModule state."
            )
        leaves, treedef = jax.tree.flatten(component_state)
        if not leaves or any(not eqx.is_array(leaf) for leaf in leaves):
            raise TypeError(
                "Restart component_state must be a nonempty array-only PyTree."
            )
        state = jax.tree.unflatten(
            treedef,
            tuple(jax.lax.stop_gradient(jnp.asarray(leaf)) for leaf in leaves),
        )
        ids_input = np.asarray(stable_ids)
        incarnation_input = np.asarray(incarnations)
        ids = jax.lax.stop_gradient(
            jnp.zeros((0,), dtype=jnp.int64)
            if ids_input.size == 0
            else jnp.asarray(stable_ids)
        )
        active = jax.lax.stop_gradient(jnp.asarray(active_mask, dtype=jnp.bool_))
        incarnation = jax.lax.stop_gradient(
            jnp.zeros((0,), dtype=jnp.int64)
            if incarnation_input.size == 0
            else jnp.asarray(incarnations)
        )
        lineage = jax.lax.stop_gradient(jnp.asarray(lineage_ids))
        if ids.ndim != 1 or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError("Restart stable_ids must be an integer vector.")
        count = ids.size
        if (
            active.shape != (count,)
            or incarnation.shape != (count,)
            or not jnp.issubdtype(incarnation.dtype, jnp.integer)
            or lineage.shape != (count, 2)
            or not jnp.issubdtype(lineage.dtype, jnp.integer)
            or len(set(np.asarray(ids, dtype=np.int64).tolist())) != count
            or not _lineage_valid(
                np.asarray(ids),
                np.asarray(incarnation),
                np.asarray(lineage),
            )
        ):
            raise ValueError(
                "Restart stable identity, mask, incarnation, or lineage changed."
            )
        root_input = jnp.asarray(prng_root)
        root = (
            jnp.zeros((0,), dtype=jnp.uint32)
            if root_input.size == 0
            else jnp.asarray(jr.key_data(prng_root), dtype=jnp.uint32)
        )
        if root.shape not in ((0,), (2,)):
            raise ValueError("Restart PRNG root must be absent or one canonical JAX key.")
        epoch = jnp.asarray(event_epoch, dtype=jnp.int64)
        step = jnp.asarray(accepted_step, dtype=jnp.int64)
        schedule = jnp.asarray(schedule_cursor, dtype=jnp.int64)
        output = jnp.asarray(output_cursor, dtype=jnp.int64)
        time_ = _scalar(time, "time")
        if any(value.shape != () for value in (epoch, step, schedule, output)) or any(
            int(np.asarray(value)) < 0 for value in (epoch, step, schedule, output)
        ):
            raise ValueError("Restart epochs and cursors must be nonnegative scalars.")
        evidence_leaves, evidence_tree = jax.tree.flatten(accepted_evidence)
        if any(not eqx.is_array(leaf) for leaf in evidence_leaves):
            raise TypeError("accepted_evidence must be an array-only PyTree.")
        evidence = jax.tree.unflatten(
            evidence_tree,
            tuple(jax.lax.stop_gradient(jnp.asarray(leaf)) for leaf in evidence_leaves),
        )
        parent = _fingerprint_id(parent_checkpoint_id, "parent_checkpoint_id")
        self.component_state = state
        self.stable_ids = ids
        self.active_mask = active
        self.incarnations = incarnation
        self.lineage_ids = lineage
        self.prng_root = root
        self.event_epoch = epoch
        self.time = time_
        self.accepted_step = step
        self.schedule_cursor = schedule
        self.output_cursor = output
        self.accepted_evidence = evidence
        self.parent_checkpoint_id = parent
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "dark-matter-restart-snapshot",
                "arrays": array_tree_fingerprint(
                    (
                        state,
                        ids,
                        active,
                        incarnation,
                        lineage,
                        root,
                        epoch,
                        time_,
                        step,
                        schedule,
                        output,
                        evidence,
                    )
                ),
                "parent_checkpoint": parent,
            }
        )


class DarkMatterCheckpointRecoveryEvidence(StrictModule, NonTrainableState):
    status: SimulationProductStatusEvidence
    source_checkpoint_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    recovery_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_checkpoint_id: str,
        contract_id: str,
        /,
        *,
        compatibility_verified: ArrayLike,
        content_verified: ArrayLike,
        stable_ids_verified: ArrayLike,
        cursor_verified: ArrayLike,
        event_epoch_verified: ArrayLike,
        parent_chain_verified: ArrayLike,
        failure_reason: str = "none",
    ):
        checkpoint = _fingerprint_id(source_checkpoint_id, "source_checkpoint_id")
        if checkpoint is None:
            raise ValueError("source_checkpoint_id must be present.")
        contract = _identifier(contract_id, "contract_id")
        status = SimulationProductStatusEvidence(
            (
                "compatibility-verified",
                "content-verified",
                "stable-ids-verified",
                "cursor-verified",
                "event-epoch-verified",
                "parent-chain-verified",
            ),
            jnp.asarray(
                (
                    compatibility_verified,
                    content_verified,
                    stable_ids_verified,
                    cursor_verified,
                    event_epoch_verified,
                    parent_chain_verified,
                ),
                dtype=jnp.bool_,
            ),
            failure_reason=failure_reason,
        )
        self.status = status
        self.source_checkpoint_id = checkpoint
        self.contract_id = contract
        self.recovery_id = canonical_fingerprint(
            {
                "kind": "dark-matter-checkpoint-recovery-evidence",
                "source_checkpoint": checkpoint,
                "contract": contract,
                "status": status.evidence_id,
            }
        )


class DarkMatterCheckpointPayload(StrictModule, NonTrainableState):
    snapshot: DarkMatterRestartSnapshot
    envelope: RuntimeCheckpointEnvelope
    recovery: DarkMatterCheckpointRecoveryEvidence
    successful: Array
    contract_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshot: DarkMatterRestartSnapshot,
        envelope: RuntimeCheckpointEnvelope,
        recovery: DarkMatterCheckpointRecoveryEvidence,
        contract_id: str,
        /,
    ):
        if not isinstance(snapshot, DarkMatterRestartSnapshot):
            raise TypeError("snapshot must be DarkMatterRestartSnapshot.")
        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        if not isinstance(recovery, DarkMatterCheckpointRecoveryEvidence):
            raise TypeError("recovery must be DarkMatterCheckpointRecoveryEvidence.")
        contract = _identifier(contract_id, "contract_id")
        if (
            recovery.contract_id != contract
            or recovery.source_checkpoint_id != envelope.checkpoint_id
        ):
            raise ValueError("Checkpoint recovery evidence does not bind this payload.")
        self.snapshot = snapshot
        self.envelope = envelope
        self.recovery = recovery
        self.successful = recovery.status.successful
        self.contract_id = contract
        self.payload_id = canonical_fingerprint(
            {
                "kind": "dark-matter-checkpoint-payload",
                "snapshot": snapshot.snapshot_id,
                "checkpoint": envelope.checkpoint_id,
                "recovery": recovery.recovery_id,
                "contract": contract,
            }
        )


def _parent_arrays(parent_checkpoint_id: str | None, /) -> tuple[Array, Array]:
    parent = _fingerprint_id(parent_checkpoint_id, "parent_checkpoint_id")
    if parent is None:
        return jnp.asarray(False), jnp.zeros((32,), dtype=jnp.uint8)
    return jnp.asarray(True), jnp.asarray(
        np.frombuffer(bytes.fromhex(parent), dtype=np.uint8)
    )


def _parent_identifier(present: ArrayLike, encoded: ArrayLike, /) -> str | None:
    present_ = bool(np.asarray(present))
    bytes_ = np.asarray(encoded, dtype=np.uint8)
    if bytes_.shape != (32,):
        raise ValueError("Encoded parent checkpoint identity changed shape.")
    return bytes_.tobytes().hex() if present_ else None


def _controller_state(snapshot: DarkMatterRestartSnapshot, /) -> tuple[Any, ...]:
    parent_present, parent_bytes = _parent_arrays(snapshot.parent_checkpoint_id)
    return (
        snapshot.output_cursor,
        snapshot.event_epoch,
        snapshot.stable_ids,
        snapshot.active_mask,
        snapshot.incarnations,
        snapshot.lineage_ids,
        parent_present,
        parent_bytes,
        snapshot.accepted_evidence,
    )


class DarkMatterCheckpointContract(StrictModule, NonTrainableState):
    """Profile compatibility binding over the native runtime checkpoint encoding."""

    encoding_plan: RuntimeCheckpointEncodingPlan
    profile_name: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    support_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    interaction_ids: tuple[str, ...] = eqx.field(static=True)
    artifact_ids: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    controller_schema_id: str = eqx.field(static=True)
    stable_id_layout_id: str = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        profile_name: str,
        physics_id: str,
        support_ids: Sequence[str],
        source_ids: Sequence[str],
        interaction_ids: Sequence[str],
        artifact_ids: Sequence[str],
        topology_id: str,
        method_id: str,
        precision_id: str,
        restart_template: DarkMatterRestartSnapshot,
        topology_epoch_id: str,
        scale_id: str,
        partition_id: str | None = "serial",
        encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
    ):
        encoding = (
            RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
        )
        if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
            raise TypeError(
                "encoding_plan must be RuntimeCheckpointEncodingPlan or None."
            )
        if not isinstance(restart_template, DarkMatterRestartSnapshot):
            raise TypeError("restart_template must be DarkMatterRestartSnapshot.")
        profile = _identifier(profile_name, "profile_name")
        physics = _identifier(physics_id, "physics_id")
        supports = _identifiers(support_ids, "support_id")
        sources = _identifiers(source_ids, "source_id")
        interactions = _identifiers(interaction_ids, "interaction_id")
        artifacts = _identifiers(artifact_ids, "artifact_id")
        topology = _identifier(topology_id, "topology_id")
        method = _identifier(method_id, "method_id")
        precision = _identifier(precision_id, "precision_id")
        epoch = _identifier(topology_epoch_id, "topology_epoch_id")
        scale = _identifier(scale_id, "scale_id")
        partition = (
            "serial"
            if partition_id is None
            else _identifier(partition_id, "partition_id")
        )
        state_schema = _tree_schema_id(
            restart_template.component_state, "component-state"
        )
        controller_schema = _tree_schema_id(
            _controller_state(restart_template), "controller-state"
        )
        inexact_dtypes = tuple(
            np.dtype(leaf.dtype)
            for leaf in jax.tree.leaves(restart_template.component_state)
            if np.issubdtype(np.dtype(leaf.dtype), np.inexact)
        )
        if not inexact_dtypes:
            raise TypeError("Restart component state requires an inexact array leaf.")
        state_dtype = np.result_type(*inexact_dtypes).name
        stable_layout = canonical_fingerprint(
            {
                "kind": "dark-matter-stable-id-layout",
                "stable_ids": array_tree_fingerprint(restart_template.stable_ids),
                "capacity": restart_template.stable_ids.size,
            }
        )
        record = {
            "kind": "dark-matter-checkpoint-contract",
            "profile": profile,
            "physics": physics,
            "supports": list(supports),
            "sources": list(sources),
            "interactions": list(interactions),
            "artifacts": list(artifacts),
            "topology": topology,
            "partition": partition,
            "method": method,
            "precision": precision,
            "topology_epoch": epoch,
            "scale": scale,
            "encoding": encoding.encoding_id,
            "state_schema": state_schema,
            "controller_schema": controller_schema,
            "stable_id_layout": stable_layout,
            "state_dtype": state_dtype,
        }
        contract = canonical_fingerprint(record)
        self.encoding_plan = encoding
        self.profile_name = profile
        self.physics_id = physics
        self.support_ids = supports
        self.source_ids = sources
        self.interaction_ids = interactions
        self.artifact_ids = artifacts
        self.topology_id = topology
        self.partition_id = partition
        self.method_id = method
        self.precision_id = precision
        self.topology_epoch_id = epoch
        self.scale_id = scale
        self.state_schema_id = state_schema
        self.controller_schema_id = controller_schema
        self.stable_id_layout_id = stable_layout
        self.state_dtype = state_dtype
        self.runtime_id = canonical_fingerprint(
            {"kind": "dark-matter-checkpoint-runtime", "contract": contract}
        )
        self.contract_id = contract

    @property
    def runtime_partition_id(self) -> str | None:
        return None if self.partition_id == "serial" else self.partition_id

    def case_manifest(self, /) -> ProductionCaseManifest:
        return ProductionCaseManifest(
            problem_id=self.physics_id,
            method_id=self.method_id,
            precision_id=self.precision_id,
            topology_id=self.topology_id,
            geometry_layout_id=self.topology_epoch_id,
            dtype=self.state_dtype,
        )

    def _require_snapshot_schema(self, snapshot: DarkMatterRestartSnapshot, /) -> None:
        if not isinstance(snapshot, DarkMatterRestartSnapshot):
            raise TypeError("snapshot must be DarkMatterRestartSnapshot.")
        stable_layout = canonical_fingerprint(
            {
                "kind": "dark-matter-stable-id-layout",
                "stable_ids": array_tree_fingerprint(snapshot.stable_ids),
                "capacity": snapshot.stable_ids.size,
            }
        )
        if (
            _tree_schema_id(snapshot.component_state, "component-state")
            != self.state_schema_id
            or _tree_schema_id(_controller_state(snapshot), "controller-state")
            != self.controller_schema_id
            or stable_layout != self.stable_id_layout_id
        ):
            raise ValueError(
                "Restart state type, tree, controller schema, or stable-ID layout changed."
            )

    def _build_payload(
        self,
        snapshot: DarkMatterRestartSnapshot,
        /,
        *,
        parent_chain_verified: bool,
    ) -> DarkMatterCheckpointPayload:
        self._require_snapshot_schema(snapshot)
        envelope = RuntimeCheckpointEnvelope(
            snapshot.component_state,
            time=snapshot.time,
            step_index=snapshot.accepted_step,
            schedule_cursor=snapshot.schedule_cursor,
            mesh_id=self.topology_id,
            method_id=self.method_id,
            precision_id=self.precision_id,
            topology_epoch_id=self.topology_epoch_id,
            controller_state=_controller_state(snapshot),
            rng_state=snapshot.prng_root,
            partition_id=self.runtime_partition_id,
            runtime_id=self.runtime_id,
            encoding_plan=self.encoding_plan,
        )
        verified = bool(parent_chain_verified)
        recovery = DarkMatterCheckpointRecoveryEvidence(
            envelope.checkpoint_id,
            self.contract_id,
            compatibility_verified=True,
            content_verified=True,
            stable_ids_verified=True,
            cursor_verified=True,
            event_epoch_verified=True,
            parent_chain_verified=verified,
            failure_reason="none" if verified else "parent-chain-unverified",
        )
        return DarkMatterCheckpointPayload(snapshot, envelope, recovery, self.contract_id)

    def payload(
        self, snapshot: DarkMatterRestartSnapshot, /
    ) -> DarkMatterCheckpointPayload:
        """Encode content exactly; non-root ancestry remains unverified until commit."""

        return self._build_payload(
            snapshot,
            parent_chain_verified=snapshot.parent_checkpoint_id is None,
        )

    def _snapshot_from_envelope(
        self,
        envelope: RuntimeCheckpointEnvelope,
        template: DarkMatterRestartSnapshot,
        /,
        *,
        expected_parent_checkpoint_id: str | None | object,
    ) -> DarkMatterRestartSnapshot:
        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        self._require_snapshot_schema(template)
        if (
            envelope.mesh_id != self.topology_id
            or envelope.method_id != self.method_id
            or envelope.precision_id != self.precision_id
            or envelope.topology_epoch_id != self.topology_epoch_id
            or envelope.partition_id != self.runtime_partition_id
            or envelope.runtime_id != self.runtime_id
            or envelope.encoding_plan.encoding_id != self.encoding_plan.encoding_id
            or _tree_schema_id(envelope.state, "component-state") != self.state_schema_id
            or _tree_schema_id(envelope.controller_state, "controller-state")
            != self.controller_schema_id
        ):
            raise ValueError(
                "Runtime checkpoint changed physics, support, state, or encoding."
            )
        controller = envelope.controller_state
        if not isinstance(controller, tuple) or len(controller) != 9:
            raise ValueError("Dark-matter checkpoint controller payload changed.")
        (
            output_cursor,
            event_epoch,
            stable_ids,
            active_mask,
            incarnations,
            lineage_ids,
            parent_present,
            parent_bytes,
            accepted_evidence,
        ) = controller
        stable_layout = canonical_fingerprint(
            {
                "kind": "dark-matter-stable-id-layout",
                "stable_ids": array_tree_fingerprint(stable_ids),
                "capacity": np.asarray(stable_ids).size,
            }
        )
        if stable_layout != self.stable_id_layout_id:
            raise ValueError("Checkpoint stable particle IDs changed support.")
        parent = _parent_identifier(parent_present, parent_bytes)
        if expected_parent_checkpoint_id is _UNSPECIFIED_PARENT:
            if parent is not None:
                raise ValueError(
                    "Non-root checkpoint recovery requires an explicit expected parent."
                )
        elif expected_parent_checkpoint_id is not _STORE_VERIFIED_PARENT:
            expected = _fingerprint_id(
                expected_parent_checkpoint_id, "expected_parent_checkpoint_id"
            )
            if parent != expected:
                raise ValueError("Checkpoint parent chain changed.")
        snapshot = DarkMatterRestartSnapshot(
            envelope.state,
            stable_ids=stable_ids,
            active_mask=active_mask,
            incarnations=incarnations,
            lineage_ids=lineage_ids,
            prng_root=envelope.rng_state,
            event_epoch=event_epoch,
            time=envelope.time,
            accepted_step=envelope.step_index,
            schedule_cursor=envelope.schedule_cursor,
            output_cursor=output_cursor,
            accepted_evidence=accepted_evidence,
            parent_checkpoint_id=parent,
        )
        self._require_snapshot_schema(snapshot)
        return snapshot

    def restore_envelope(
        self,
        envelope: RuntimeCheckpointEnvelope,
        template: DarkMatterRestartSnapshot,
        /,
        *,
        expected_parent_checkpoint_id: str | None | object = _UNSPECIFIED_PARENT,
    ) -> DarkMatterCheckpointPayload:
        snapshot = self._snapshot_from_envelope(
            envelope,
            template,
            expected_parent_checkpoint_id=expected_parent_checkpoint_id,
        )
        rebuilt = self._build_payload(snapshot, parent_chain_verified=True)
        if rebuilt.envelope.checkpoint_id != envelope.checkpoint_id:
            raise ValueError("Restored checkpoint did not round-trip exactly.")
        return rebuilt

    def _verified_payload(
        self, payload: DarkMatterCheckpointPayload, /
    ) -> DarkMatterCheckpointPayload:
        if not isinstance(payload, DarkMatterCheckpointPayload):
            raise TypeError("payload must be DarkMatterCheckpointPayload.")
        if payload.contract_id != self.contract_id:
            raise ValueError("Checkpoint payload belongs to another contract.")
        restored = self._snapshot_from_envelope(
            payload.envelope,
            payload.snapshot,
            expected_parent_checkpoint_id=payload.snapshot.parent_checkpoint_id,
        )
        expected = self.payload(restored)
        if (
            restored.snapshot_id != payload.snapshot.snapshot_id
            or expected.envelope.checkpoint_id != payload.envelope.checkpoint_id
            or expected.envelope.content_digest != payload.envelope.content_digest
            or expected.recovery.recovery_id != payload.recovery.recovery_id
            or expected.payload_id != payload.payload_id
            or bool(np.asarray(expected.successful))
            != bool(np.asarray(payload.successful))
        ):
            raise ValueError(
                "Checkpoint snapshot, envelope, recovery, or payload identity was spliced."
            )
        return expected

    def write(
        self,
        path: str | os.PathLike[str],
        payload: DarkMatterCheckpointPayload,
        /,
    ) -> Path:
        verified = self._verified_payload(payload)
        if not bool(np.asarray(verified.successful)):
            raise ValueError(
                "Standalone checkpoint writes require verified root ancestry."
            )
        return write_runtime_checkpoint(path, verified.envelope)

    def read(
        self,
        path: str | os.PathLike[str],
        template: DarkMatterRestartSnapshot,
        /,
        *,
        expected_parent_checkpoint_id: str | None | object = _UNSPECIFIED_PARENT,
    ) -> DarkMatterCheckpointPayload:
        self._require_snapshot_schema(template)
        envelope = read_runtime_checkpoint(
            path,
            state_template=template.component_state,
            mesh_id=self.topology_id,
            method_id=self.method_id,
            precision_id=self.precision_id,
            topology_epoch_id=self.topology_epoch_id,
            controller_template=_controller_state(template),
            rng_template=template.prng_root,
            partition_id=self.runtime_partition_id,
            runtime_id=self.runtime_id,
            encoding_plan=self.encoding_plan,
        )
        return self.restore_envelope(
            envelope,
            template,
            expected_parent_checkpoint_id=expected_parent_checkpoint_id,
        )

    def require_payload(self, payload: DarkMatterCheckpointPayload, /) -> None:
        self._verified_payload(payload)

    def require_store(
        self,
        store: DurableCheckpointStore | ArtifactCheckpointStore,
        /,
    ) -> None:
        if not isinstance(store, (DurableCheckpointStore, ArtifactCheckpointStore)):
            raise TypeError("store must be a native production checkpoint store.")
        if self.partition_id != "serial":
            raise ValueError(
                "Non-serial checkpoints require the distributed shard store."
            )
        manifest = store.manifest
        if (
            manifest.problem_id != self.physics_id
            or manifest.method_id != self.method_id
            or manifest.precision_id != self.precision_id
            or manifest.topology_id != self.topology_id
            or manifest.geometry_layout_id != self.topology_epoch_id
            or manifest.dtype != self.state_dtype
            or store.encoding_plan.encoding_id != self.encoding_plan.encoding_id
        ):
            raise ValueError("Checkpoint store does not exactly bind this contract.")

    def commit(
        self,
        store: DurableCheckpointStore | ArtifactCheckpointStore,
        payload: DarkMatterCheckpointPayload,
        /,
    ) -> CheckpointCommitReceipt:
        self.require_store(store)
        verified = self._verified_payload(payload)
        generation = store.generation_for_commit(verified.envelope)
        if generation == 0:
            if verified.snapshot.parent_checkpoint_id is not None:
                raise ValueError("Initial checkpoint cannot declare a parent checkpoint.")
        else:
            latest = store.latest(
                verified.snapshot.component_state,
                controller_template=_controller_state(verified.snapshot),
                rng_template=verified.snapshot.prng_root,
                runtime_id=self.runtime_id,
            )
            if latest.checkpoint_id == verified.envelope.checkpoint_id:
                return store.receipt_for(verified.envelope)
            if latest.checkpoint_id != verified.snapshot.parent_checkpoint_id:
                raise ValueError(
                    "Checkpoint parent is not the latest complete checkpoint."
                )
        return store.commit(generation, verified.envelope)

    def restore_latest(
        self,
        store: DurableCheckpointStore | ArtifactCheckpointStore,
        template: DarkMatterRestartSnapshot,
        /,
    ) -> DarkMatterCheckpointPayload:
        self.require_store(store)
        envelope = store.latest(
            template.component_state,
            controller_template=_controller_state(template),
            rng_template=template.prng_root,
            runtime_id=self.runtime_id,
        )
        return self.restore_envelope(
            envelope,
            template,
            expected_parent_checkpoint_id=_STORE_VERIFIED_PARENT,
        )


__all__ = [
    "CommonGravitySimulationSnapshot",
    "CommonGravitySnapshotEvidence",
    "CosmologyOutputBundle",
    "DarkMatterCheckpointContract",
    "DarkMatterCheckpointPayload",
    "DarkMatterCheckpointRecoveryEvidence",
    "DarkMatterRestartSnapshot",
    "GasSimulationSnapshot",
    "GasSnapshotEvidence",
    "ParticleSimulationSnapshot",
    "ParticleSnapshotEvidence",
    "SimulationProductStatusEvidence",
    "WaveSimulationSnapshot",
    "WaveSnapshotEvidence",
]
