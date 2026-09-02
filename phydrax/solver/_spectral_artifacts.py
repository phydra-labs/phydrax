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
from ..linalg import AbstractRealCoordinateMap, RealCoordinateEvidence


class SpectralStateArtifact(StrictModule):
    """Portable modal seed or exact fixed-step one-step checkpoint."""

    state: Array
    time: Array
    step: Array
    step_size: Array | None
    coordinate_evidence: RealCoordinateEvidence | None
    discretization_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    restartable: bool = eqx.field(static=True)
    extra: Mapping[str, Any] = eqx.field(static=True)
    full_state_bytes: int = eqx.field(static=True)
    stored_state_bytes: int = eqx.field(static=True)
    fixed_coordinate_count: int | None = eqx.field(static=True)
    conjugate_pair_count: int | None = eqx.field(static=True)

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
        coordinate_evidence: RealCoordinateEvidence | None = None,
        stored_state_bytes: int | None = None,
        fixed_coordinate_count: int | None = None,
        conjugate_pair_count: int | None = None,
    ):
        state_ = jnp.asarray(state)
        raw_time = jnp.asarray(time)
        raw_step = jnp.asarray(step)
        raw_step_size = None if step_size is None else jnp.asarray(step_size)
        if jnp.iscomplexobj(raw_time) or (
            raw_step_size is not None and jnp.iscomplexobj(raw_step_size)
        ):
            raise TypeError("Spectral artifact time and step_size must be real.")
        if raw_step.shape != () or not jnp.issubdtype(raw_step.dtype, jnp.signedinteger):
            raise TypeError("Artifact step must be one signed integer scalar.")
        time_ = raw_time.astype(jnp.result_type(raw_time.dtype, jnp.float32))
        step_ = raw_step
        step_size_ = (
            None
            if raw_step_size is None
            else raw_step_size.astype(jnp.result_type(raw_step_size.dtype, jnp.float32))
        )
        if not jnp.issubdtype(state_.dtype, jnp.complexfloating):
            raise TypeError("Spectral artifact state must be full complex coefficients.")
        if state_.size < 1 or not bool(jnp.all(jnp.isfinite(state_))):
            raise ValueError("Spectral artifact state must be finite and nonempty.")
        if time_.shape != () or not bool(jnp.isfinite(time_)):
            raise ValueError("Artifact time must be one finite scalar.")
        if int(step_) < 0:
            raise ValueError("Artifact step must be nonnegative.")
        if step_size_ is not None and (
            step_size_.shape != ()
            or not bool(jnp.isfinite(step_size_) & (step_size_ > 0.0))
        ):
            raise ValueError("step_size must be one finite positive scalar or None.")
        if not isinstance(restartable, (bool, np.bool_)):
            raise TypeError("restartable must be Boolean.")
        if bool(restartable) != (step_size_ is not None):
            raise ValueError(
                "restartable fixed-step artifacts require step_size, and seeds omit it."
            )
        if coordinate_evidence is not None and not isinstance(
            coordinate_evidence, RealCoordinateEvidence
        ):
            raise TypeError("coordinate_evidence must be RealCoordinateEvidence or None.")
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
        full_bytes = int(state_.nbytes)
        stored_bytes = (
            full_bytes if stored_state_bytes is None else int(stored_state_bytes)
        )
        fixed = None if fixed_coordinate_count is None else int(fixed_coordinate_count)
        pairs = None if conjugate_pair_count is None else int(conjugate_pair_count)
        if stored_bytes <= 0 or stored_bytes > full_bytes:
            raise ValueError("stored_state_bytes must lie in (0, full_state_bytes].")
        if (fixed is not None and fixed < 0) or (pairs is not None and pairs < 0):
            raise ValueError("Hermitian coordinate counts must be nonnegative.")
        extra_ = {} if extra is None else dict(extra)
        computed_id = canonical_fingerprint(
            {
                "kind": "spectral-state-artifact",
                "state": array_tree_fingerprint(state_),
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
        if artifact_id is not None and str(artifact_id) != computed_id:
            raise ValueError("artifact_id does not match spectral artifact contents.")
        self.state = state_
        self.time = time_
        self.step = step_
        self.step_size = step_size_
        self.coordinate_evidence = coordinate_evidence
        (
            self.discretization_id,
            self.compilation_id,
            self.method_id,
            self.source_hash,
        ) = identifiers
        self.artifact_id = computed_id
        self.restartable = bool(restartable)
        self.extra = extra_
        self.full_state_bytes = full_bytes
        self.stored_state_bytes = stored_bytes
        self.fixed_coordinate_count = fixed
        self.conjugate_pair_count = pairs


def _evidence_manifest(evidence: RealCoordinateEvidence, /) -> dict[str, Any]:
    return {
        "domain_kind": evidence.domain_kind,
        "source_space_id": evidence.source_space_id,
        "coordinate_space_id": evidence.coordinate_space_id,
        "source_dtype": evidence.source_dtype,
        "coordinate_dtype": evidence.coordinate_dtype,
        "source_shape": list(evidence.source_shape),
        "coordinate_shape": list(evidence.coordinate_shape),
        "norm_relation": evidence.norm_relation,
        "projection_kind": evidence.projection_kind,
        "map_id": evidence.map_id,
        "evidence_id": evidence.evidence_id,
    }


def _hermitian_counts(
    state_coordinates: AbstractRealCoordinateMap | None,
    /,
) -> tuple[int | None, int | None]:
    from ..discretization.spectral import HermitianSpectralCoordinates

    if not isinstance(state_coordinates, HermitianSpectralCoordinates):
        return None, None
    return (
        int(state_coordinates.fixed_indices.size),
        int(state_coordinates.representative_indices.size),
    )


def write_spectral_state_artifact(
    path: str | Path,
    artifact: SpectralStateArtifact,
    /,
    *,
    state_coordinates: AbstractRealCoordinateMap | None = None,
) -> Path:
    """Atomically write full or declared minimal-real spectral coordinates."""
    if not isinstance(artifact, SpectralStateArtifact):
        raise TypeError("artifact must be a SpectralStateArtifact.")
    if state_coordinates is not None and not isinstance(
        state_coordinates, AbstractRealCoordinateMap
    ):
        raise TypeError("state_coordinates must be AbstractRealCoordinateMap or None.")
    if state_coordinates is None:
        stored = artifact.state
        evidence_manifest = None
    else:
        state_coordinates.validate_state(artifact.state)
        defect = state_coordinates.defect(artifact.state)
        if (
            not bool(jnp.isfinite(defect))
            or float(defect) > 128.0 * np.finfo(artifact.state.real.dtype).eps
        ):
            raise ValueError("Artifact state violates its declared coordinate domain.")
        stored = state_coordinates.to_real_coordinates(artifact.state)
        if jnp.iscomplexobj(stored):
            raise TypeError("Artifact execution coordinates must be real.")
        evidence_manifest = _evidence_manifest(state_coordinates.evidence)
    fixed, pairs = _hermitian_counts(state_coordinates)
    manifest = {
        "kind": "spectral-fixed-step-checkpoint"
        if artifact.restartable
        else "spectral-seed",
        "time": float(artifact.time),
        "step": int(artifact.step),
        "step_size": None if artifact.step_size is None else float(artifact.step_size),
        "public_state_shape": list(artifact.state.shape),
        "public_state_dtype": np.dtype(artifact.state.dtype).str,
        "stored_state_shape": list(stored.shape),
        "stored_state_dtype": np.dtype(stored.dtype).str,
        "full_state_bytes": int(artifact.state.nbytes),
        "stored_state_bytes": int(stored.nbytes),
        "fixed_coordinate_count": fixed,
        "conjugate_pair_count": pairs,
        "real_coordinate_evidence": evidence_manifest,
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
        arrays={"state_coordinates": np.asarray(stored)},
    )


def read_spectral_state_artifact(
    path: str | Path,
    /,
    *,
    state_coordinates: AbstractRealCoordinateMap | None = None,
    expected_discretization_id: str | None = None,
    expected_compilation_id: str | None = None,
) -> SpectralStateArtifact:
    """Read and compatibility-check one spectral seed or checkpoint."""
    manifest, arrays = read_array_archive(path)
    if set(arrays) != {"state_coordinates"}:
        raise ValueError("Spectral artifact must contain exactly one coordinate array.")
    stored = arrays["state_coordinates"]
    if list(stored.shape) != manifest.get("stored_state_shape"):
        raise ValueError("Stored spectral coordinates do not match their manifest shape.")
    if np.dtype(stored.dtype).str != manifest.get("stored_state_dtype"):
        raise ValueError("Stored spectral coordinates do not match their manifest dtype.")
    stored_evidence = manifest.get("real_coordinate_evidence")
    if stored_evidence is None:
        if state_coordinates is not None:
            raise ValueError("Full-complex artifact does not declare real coordinates.")
        state = stored
        evidence = None
    else:
        if not isinstance(state_coordinates, AbstractRealCoordinateMap):
            raise ValueError(
                "Coordinate-resident artifact requires the exact declared state map."
            )
        expected_evidence = _evidence_manifest(state_coordinates.evidence)
        if stored_evidence != expected_evidence:
            raise ValueError("Spectral artifact coordinate evidence does not match.")
        state = np.asarray(state_coordinates.from_real_coordinates(stored))
        evidence = state_coordinates.evidence
    if list(state.shape) != manifest.get("public_state_shape"):
        raise ValueError("Public spectral state does not match its manifest shape.")
    if np.dtype(state.dtype).str != manifest.get("public_state_dtype"):
        raise ValueError("Public spectral state does not match its manifest dtype.")
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
    restartable_value = manifest["restartable"]
    if not isinstance(restartable_value, bool):
        raise TypeError("Spectral artifact restartable manifest value must be Boolean.")
    restartable = restartable_value
    expected_kind = "spectral-fixed-step-checkpoint" if restartable else "spectral-seed"
    if manifest.get("kind") != expected_kind:
        raise ValueError("Spectral artifact kind and restartability disagree.")
    stored_id = str(manifest["artifact_id"])
    artifact = SpectralStateArtifact(
        state,
        manifest["time"],
        manifest["step"],
        discretization_id=discretization_id,
        compilation_id=compilation_id,
        method_id=str(manifest["method_id"]),
        source_hash=str(manifest["source_hash"]),
        step_size=manifest["step_size"],
        restartable=restartable,
        extra=dict(manifest["extra"]),
        coordinate_evidence=evidence,
        stored_state_bytes=int(manifest["stored_state_bytes"]),
        fixed_coordinate_count=manifest["fixed_coordinate_count"],
        conjugate_pair_count=manifest["conjugate_pair_count"],
    )
    if artifact.artifact_id != stored_id:
        raise ValueError("Spectral artifact content fingerprint does not match.")
    if artifact.full_state_bytes != int(manifest["full_state_bytes"]):
        raise ValueError("Spectral artifact full-state byte count does not match.")
    return artifact


__all__ = [
    "SpectralStateArtifact",
    "read_spectral_state_artifact",
    "write_spectral_state_artifact",
]
