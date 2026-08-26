#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


_SPECTRAL_ARTIFACT_SCHEMA = 1


class SpectralStateArtifact(StrictModule):
    """Portable modal seed or exact fixed-step one-step checkpoint."""

    state: Array
    time: Array
    step: Array
    step_size: Array | None
    discretization_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    restartable: bool = eqx.field(static=True)
    extra: Mapping[str, Any] = eqx.field(static=True)

    def __init__(
        self,
        state: ArrayLike,
        time: ArrayLike,
        step: int | ArrayLike,
        /,
        *,
        discretization_id: str,
        compilation_id: str,
        method_id: str,
        source_hash: str,
        step_size: ArrayLike | None = None,
        restartable: bool = False,
        extra: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
    ):
        state_ = jnp.asarray(state)
        time_ = jnp.asarray(time, dtype=float)
        step_ = jnp.asarray(step, dtype=jnp.int64)
        step_size_ = None if step_size is None else jnp.asarray(step_size, dtype=float)
        if not jnp.issubdtype(state_.dtype, jnp.complexfloating):
            raise TypeError("Spectral artifact state must be full complex coefficients.")
        if state_.size < 1 or not bool(jnp.all(jnp.isfinite(state_))):
            raise ValueError("Spectral artifact state must be finite and nonempty.")
        if time_.shape != () or not bool(jnp.isfinite(time_)):
            raise ValueError("Artifact time must be one finite scalar.")
        if step_.shape != () or int(step_) < 0:
            raise ValueError("Artifact step must be one nonnegative integer.")
        if step_size_ is not None and (
            step_size_.shape != ()
            or not bool(jnp.isfinite(step_size_) & (step_size_ > 0.0))
        ):
            raise ValueError("step_size must be one finite positive scalar or None.")
        if bool(restartable) != (step_size_ is not None):
            raise ValueError(
                "restartable fixed-step artifacts require step_size, and seeds omit it."
            )
        identifiers = tuple(
            str(value)
            for value in (
                discretization_id,
                compilation_id,
                method_id,
                source_hash,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Artifact provenance identifiers must be non-empty.")
        extra_ = {} if extra is None else dict(extra)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "spectral-state-artifact-v1",
                    "state": array_tree_fingerprint(np.asarray(state_)),
                    "time": float(time_),
                    "step": int(step_),
                    "step_size": None if step_size_ is None else float(step_size_),
                    "discretization": identifiers[0],
                    "compilation": identifiers[1],
                    "method": identifiers[2],
                    "source_hash": identifiers[3],
                    "restartable": bool(restartable),
                    "extra": extra_,
                }
            )
            if artifact_id is None
            else str(artifact_id)
        )
        if not identifier:
            raise ValueError("artifact_id must be non-empty.")
        self.state = state_
        self.time = time_
        self.step = step_
        self.step_size = step_size_
        (
            self.discretization_id,
            self.compilation_id,
            self.method_id,
            self.source_hash,
        ) = identifiers
        self.artifact_id = identifier
        self.restartable = bool(restartable)
        self.extra = extra_


def write_spectral_state_artifact(
    path: str | Path,
    artifact: SpectralStateArtifact,
    /,
) -> Path:
    """Atomically write one checksum-validated spectral state artifact."""
    if not isinstance(artifact, SpectralStateArtifact):
        raise TypeError("artifact must be a SpectralStateArtifact.")
    manifest = {
        "schema_version": _SPECTRAL_ARTIFACT_SCHEMA,
        "kind": "spectral-fixed-step-checkpoint"
        if artifact.restartable
        else "spectral-seed",
        "time": float(artifact.time),
        "step": int(artifact.step),
        "step_size": None if artifact.step_size is None else float(artifact.step_size),
        "state_shape": list(artifact.state.shape),
        "state_dtype": np.dtype(artifact.state.dtype).str,
        "discretization_id": artifact.discretization_id,
        "compilation_id": artifact.compilation_id,
        "method_id": artifact.method_id,
        "source_hash": artifact.source_hash,
        "artifact_id": artifact.artifact_id,
        "restartable": artifact.restartable,
        "extra": dict(artifact.extra),
    }
    return write_array_archive(
        path,
        manifest=manifest,
        arrays={"state": np.asarray(artifact.state)},
    )


def read_spectral_state_artifact(
    path: str | Path,
    /,
    *,
    expected_discretization_id: str | None = None,
    expected_compilation_id: str | None = None,
) -> SpectralStateArtifact:
    """Read and compatibility-check one spectral seed or fixed-step checkpoint."""
    manifest, arrays = read_array_archive(path)
    if manifest.get("schema_version") != _SPECTRAL_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported spectral artifact schema version.")
    if set(arrays) != {"state"}:
        raise ValueError("Spectral artifact must contain exactly one state array.")
    discretization_id = str(manifest["discretization_id"])
    compilation_id = str(manifest["compilation_id"])
    if expected_discretization_id is not None and discretization_id != str(
        expected_discretization_id
    ):
        raise ValueError("Spectral artifact discretization identity does not match.")
    if expected_compilation_id is not None and compilation_id != str(
        expected_compilation_id
    ):
        raise ValueError("Spectral artifact compilation identity does not match.")
    restartable = bool(manifest["restartable"])
    return SpectralStateArtifact(
        arrays["state"],
        manifest["time"],
        manifest["step"],
        discretization_id=discretization_id,
        compilation_id=compilation_id,
        method_id=str(manifest["method_id"]),
        source_hash=str(manifest["source_hash"]),
        step_size=manifest["step_size"],
        restartable=restartable,
        extra=dict(manifest["extra"]),
        artifact_id=str(manifest["artifact_id"]),
    )


__all__ = [
    "SpectralStateArtifact",
    "read_spectral_state_artifact",
    "write_spectral_state_artifact",
]
