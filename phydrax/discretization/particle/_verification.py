#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import read_array_archive, write_array_archive
from ..._fingerprint import canonical_fingerprint
from ..._publication import publish_bytes
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._qualification import (
    ParticleBenchmarkIdentity,
    ParticleMethodMaturity,
    ParticleQualificationResult,
)


class ParticleBenchmarkRecord(StrictModule, NonTrainableState):
    identity: ParticleBenchmarkIdentity
    qualification: ParticleQualificationResult
    metrics: tuple[tuple[str, float], ...] = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        identity: ParticleBenchmarkIdentity,
        qualification: ParticleQualificationResult,
        metrics: Sequence[tuple[str, float]],
        /,
    ):
        values = tuple((str(name), float(value)) for name, value in metrics)
        self.identity = identity
        self.qualification = qualification
        self.metrics = values
        self.record_id = canonical_fingerprint(
            {
                "kind": "particle-benchmark-record",
                "identity": identity.benchmark_id,
                "qualification": qualification.result_id,
                "metrics": list(values),
            }
        )


class ParticleBenchmarkRegistry(StrictModule, NonTrainableState):
    records: tuple[ParticleBenchmarkRecord, ...]
    registry_id: str = eqx.field(static=True)

    def __init__(self, records: Sequence[ParticleBenchmarkRecord], /):
        values = tuple(records)
        identifiers = tuple(record.record_id for record in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Particle benchmark records must be unique.")
        self.records = values
        self.registry_id = canonical_fingerprint(
            {"kind": "particle-benchmark-registry", "records": list(identifiers)}
        )


class ParticleRefinementReport(StrictModule):
    resolutions: Array
    errors: Array
    observed_orders: Array
    extrapolated_error: Array
    monotone: Array


def particle_refinement_report(
    resolutions: ArrayLike, errors: ArrayLike, /
) -> ParticleRefinementReport:
    resolution = jnp.asarray(resolutions)
    error = jnp.asarray(errors)
    if resolution.ndim != 1 or error.shape != resolution.shape or resolution.size < 2:
        raise ValueError("Refinement data must be matching vectors of length >= 2.")
    order = jnp.log(error[:-1] / error[1:]) / jnp.log(resolution[1:] / resolution[:-1])
    extrapolated = error[-1] / jnp.maximum(
        resolution[-1] ** jnp.maximum(order[-1], 0.0), 1.0
    )
    return ParticleRefinementReport(
        resolution,
        error,
        order,
        extrapolated,
        jnp.all(error[1:] <= error[:-1]),
    )


class ParticleQualificationArtifact(StrictModule, NonTrainableState):
    registry: ParticleBenchmarkRegistry
    method_id: str = eqx.field(static=True)
    code_version: str = eqx.field(static=True)
    package_fingerprint: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        registry: ParticleBenchmarkRegistry,
        method_id: str,
        code_version: str,
        package_fingerprint: str,
        /,
    ):
        values = tuple(
            str(value) for value in (method_id, code_version, package_fingerprint)
        )
        if any(not value for value in values):
            raise ValueError("Qualification artifact identity fields must be non-empty.")
        self.registry = registry
        self.method_id, self.code_version, self.package_fingerprint = values
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "particle-qualification-artifact",
                "registry": registry.registry_id,
                "method": values[0],
                "code_version": values[1],
                "package_fingerprint": values[2],
            }
        )


def write_particle_qualification_artifact(
    path: str | Path, artifact: ParticleQualificationArtifact, /
) -> None:
    destination = Path(path)
    payload = {
        "artifact_id": artifact.artifact_id,
        "method_id": artifact.method_id,
        "code_version": artifact.code_version,
        "package_fingerprint": artifact.package_fingerprint,
        "registry_id": artifact.registry.registry_id,
        "records": [
            {
                "record_id": record.record_id,
                "benchmark_id": record.identity.benchmark_id,
                "name": record.identity.name,
                "maturity": record.qualification.maturity.value,
                "execution_successful": bool(record.qualification.execution_successful),
                "numerical_constraints_satisfied": bool(
                    record.qualification.numerical_constraints_satisfied
                ),
                "production_gate_satisfied": bool(
                    record.qualification.production_gate_satisfied
                ),
                "metrics": dict(record.metrics),
            }
            for record in artifact.registry.records
        ],
    }
    content = (json.dumps(payload, allow_nan=False, indent=2) + "\n").encode("utf-8")
    publish_bytes(
        destination,
        content,
        maximum_bytes=64 * 1024 * 1024,
        mode="atomic_replace",
    )


class ParticleReplayPacket(StrictModule, NonTrainableState):
    state: Array
    time: Array
    step_index: Array
    last_successful_state: Array
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    failure_status: str = eqx.field(static=True)
    packet_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: ArrayLike,
        time: ArrayLike,
        step_index: ArrayLike,
        last_successful_state: ArrayLike,
        /,
        *,
        problem_id: str,
        method_id: str,
        failure_status: str,
    ):
        self.state = jnp.asarray(state)
        self.time = jnp.asarray(time)
        self.step_index = jnp.asarray(step_index, dtype=jnp.int32)
        self.last_successful_state = jnp.asarray(last_successful_state)
        self.problem_id = str(problem_id)
        self.method_id = str(method_id)
        self.failure_status = str(failure_status)
        self.packet_id = canonical_fingerprint(
            {
                "kind": "particle-replay-packet",
                "problem": self.problem_id,
                "method": self.method_id,
                "failure": self.failure_status,
                "state_shape": list(self.state.shape),
                "state_dtype": str(self.state.dtype),
            }
        )


def write_particle_replay(path: str | Path, packet: ParticleReplayPacket, /) -> None:
    write_array_archive(
        path,
        manifest={
            "kind": "particle-replay",
            "problem_id": packet.problem_id,
            "method_id": packet.method_id,
            "failure_status": packet.failure_status,
            "packet_id": packet.packet_id,
        },
        arrays={
            "state": np.asarray(packet.state),
            "time": np.asarray(packet.time),
            "step_index": np.asarray(packet.step_index),
            "last_successful_state": np.asarray(packet.last_successful_state),
        },
    )


def read_particle_replay(path: str | Path, /) -> ParticleReplayPacket:
    metadata, arrays = read_array_archive(path)
    if metadata.get("kind") != "particle-replay":
        raise ValueError("Archive is not a particle replay packet.")
    packet = ParticleReplayPacket(
        arrays["state"],
        arrays["time"],
        arrays["step_index"],
        arrays["last_successful_state"],
        problem_id=metadata["problem_id"],
        method_id=metadata["method_id"],
        failure_status=metadata["failure_status"],
    )
    if packet.packet_id != metadata["packet_id"]:
        raise ValueError("Particle replay packet identity mismatch.")
    return packet


def replay_particle_failure(
    packet: ParticleReplayPacket,
    step_function: Callable[[Array, Array, Any], Array],
    args: Any = None,
    /,
) -> Array:
    return jnp.asarray(step_function(packet.time, packet.last_successful_state, args))


class ParticleSupportMatrixEntry(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    maturity: ParticleMethodMaturity = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ParticleSupportMatrix(StrictModule, NonTrainableState):
    entries: tuple[ParticleSupportMatrixEntry, ...]
    matrix_id: str = eqx.field(static=True)

    def __init__(self, entries: Sequence[ParticleSupportMatrixEntry], /):
        values = tuple(entries)
        self.entries = values
        self.matrix_id = canonical_fingerprint(
            {
                "kind": "particle-support-matrix",
                "entries": [
                    {
                        "method": value.method,
                        "dimension": value.dimension,
                        "backend": value.backend,
                        "precision": value.precision,
                        "maturity": value.maturity.value,
                        "evidence": value.evidence_id,
                    }
                    for value in values
                ],
            }
        )


__all__ = [
    "ParticleBenchmarkRecord",
    "ParticleBenchmarkRegistry",
    "ParticleQualificationArtifact",
    "ParticleRefinementReport",
    "ParticleReplayPacket",
    "ParticleSupportMatrix",
    "ParticleSupportMatrixEntry",
    "particle_refinement_report",
    "read_particle_replay",
    "replay_particle_failure",
    "write_particle_qualification_artifact",
    "write_particle_replay",
]
