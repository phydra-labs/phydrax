#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._array_archive import (
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    pack_array_tree,
    unpack_array_tree,
)
from .._execution_resources import ResourceRequest
from .._fingerprint import canonical_fingerprint, canonical_json
from .._host_io import descriptor_relative_path, open_directory_descriptor
from .._iteration import (
    bind_iteration_scope,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationSession,
    IterationSessionState,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..lifecycle._archive import decode_logical_arrays, encode_logical_arrays
from ..lifecycle._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    CheckpointResourcePolicy,
    ChunkRecord,
    RepositoryConflictError,
    RepositoryCorruptionError,
)
from ..lifecycle._migration import MigrationReport
from ..lifecycle._repository import ObjectNotFoundError
from ..lifecycle._resolved_run import ResolvedRunSpec
from ..logging import emit
from ._fixed_step import (
    _canonical_structured_state,
    _state_dtype,
    AbstractFixedStepMethod,
    RetriedFixedStepResult,
    retry_fixed_step,
    RobustRetryPolicy,
)
from ._runtime_lifecycle import (
    AcceptedStepTriggerGraph,
    AcceptedStepTriggerGraphState,
    ByteBoundedAsyncPublisher,
    ExactTimeSchedule,
    read_runtime_checkpoint,
    restore_runtime_checkpoint_arrays,
    RuntimeCheckpointEncodingPlan,
    RuntimeCheckpointEnvelope,
    RuntimeRestartRelation,
    StreamingMomentPlan,
    StreamingMomentState,
    verify_runtime_checkpoint_envelope,
    write_runtime_checkpoint,
)


RunStatus = Literal["ready", "running", "completed", "failed", "canceled"]
ProductionTriggerAction = Literal["checkpoint", "publish", "stop"]


def _finite_array_tree(state: Any, /) -> Array:
    leaves = jax.tree.leaves(state)
    if not leaves:
        raise ValueError("Production state must contain array leaves.")
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _canonical_auxiliary_tree(tree: Any, role: str, /) -> Any:
    leaves, treedef = jax.tree.flatten(tree)
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError(f"Production {role} must be an array-only PyTree.")
    return jax.tree.unflatten(treedef, tuple(jnp.asarray(leaf) for leaf in leaves))


_JSON_FILE_LIMIT = 65_536
_REGULAR_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _entry_exists(directory_descriptor: int, name: str, /) -> bool:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_regular_file_at(
    directory_descriptor: int, name: str, maximum_bytes: int, /
) -> bytes:
    descriptor = os.open(name, _REGULAR_FILE_OPEN_FLAGS, dir_fd=directory_descriptor)
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise ValueError("Checkpoint metadata must be a regular file.")
        if information.st_size > maximum_bytes:
            raise ValueError("Checkpoint metadata exceeds its byte limit.")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValueError("Checkpoint metadata exceeds its byte limit.")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_file_identity_at(
    directory_descriptor: int,
    name: str,
    maximum_bytes: int,
    /,
) -> tuple[int, str]:
    descriptor = os.open(name, _REGULAR_FILE_OPEN_FLAGS, dir_fd=directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Checkpoint payload must be a regular file.")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError("Checkpoint payload size is outside its durable bound.")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, min(1_048_576, maximum_bytes - size + 1)):
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("Checkpoint payload exceeds its durable byte bound.")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            size != before.st_size
            or size != after.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            raise ValueError("Checkpoint payload changed during integrity verification.")
        return size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_json_at(
    directory_descriptor: int, name: str, maximum_bytes: int, /
) -> Mapping[str, Any]:
    payload = _read_regular_file_at(directory_descriptor, name, maximum_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Checkpoint metadata is not valid JSON.") from error
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint metadata must be a JSON object.")
    return value


def _write_json_atomic_at(
    directory_descriptor: int,
    name: str,
    payload: Mapping[str, Any],
    /,
) -> None:
    encoded = (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > _JSON_FILE_LIMIT:
        raise ValueError("Checkpoint metadata exceeds its byte limit.")
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass


def _validate_json_nesting(payload: str, maximum: int, /) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in payload:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                raise ValueError("Repository JSON exceeds its nesting limit.")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Repository JSON nesting is invalid.")


class ProductionCaseManifest(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        problem_id: str,
        method_id: str,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ):
        values = tuple(
            str(value)
            for value in (
                problem_id,
                method_id,
                precision_id,
                topology_id,
                geometry_layout_id,
                dtype,
            )
        )
        if any(not value for value in values):
            raise ValueError("Production case manifest identities are required.")
        backend = jax.default_backend()
        (
            self.problem_id,
            self.method_id,
            self.precision_id,
            self.topology_id,
            self.geometry_layout_id,
            self.dtype,
        ) = values
        self.backend = backend
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "production-case-manifest",
                "problem": self.problem_id,
                "method": self.method_id,
                "precision": self.precision_id,
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "backend": backend,
                "dtype": self.dtype,
            }
        )


class CheckpointGenerationPolicy(StrictModule, NonTrainableState):
    retention: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, retention: int = 3, /):
        retention_ = int(retention)
        if retention_ <= 0:
            raise ValueError("Checkpoint retention must be positive.")
        self.retention = retention_
        self.policy_id = canonical_fingerprint(
            {"kind": "checkpoint-generation-policy", "retention": retention_}
        )


@dataclass(frozen=True, slots=True)
class CheckpointCommitReceipt:
    """Content-bound evidence issued only after a durable checkpoint publication."""

    store_id: str
    checkpoint_id: str
    content_digest: str
    runtime_id: str
    generation: int
    accepted_step: int
    commit_id: str
    commit_locator: str
    durable_size_bytes: int
    durable_sha256: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        digests = (
            self.store_id,
            self.checkpoint_id,
            self.content_digest,
            self.runtime_id,
            self.commit_id,
            self.durable_sha256,
        )
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("Checkpoint receipt content identities are invalid.")
        if (
            type(self.generation) is not int
            or self.generation < 0
            or type(self.accepted_step) is not int
            or self.accepted_step < 0
        ):
            raise ValueError("Checkpoint receipt generation or accepted step is invalid.")
        if type(self.durable_size_bytes) is not int or self.durable_size_bytes <= 0:
            raise ValueError("Checkpoint receipt durable byte size is invalid.")
        if (
            type(self.commit_locator) is not str
            or not self.commit_locator
            or self.commit_locator != self.commit_locator.strip()
        ):
            raise ValueError("Checkpoint receipt commit locator is invalid.")
        object.__setattr__(
            self,
            "receipt_id",
            canonical_fingerprint(
                {
                    "kind": "checkpoint-commit-receipt",
                    "store": self.store_id,
                    "checkpoint": self.checkpoint_id,
                    "content": self.content_digest,
                    "runtime": self.runtime_id,
                    "generation": self.generation,
                    "accepted_step": self.accepted_step,
                    "commit": self.commit_id,
                    "locator": self.commit_locator,
                    "durable_size_bytes": self.durable_size_bytes,
                    "durable_sha256": self.durable_sha256,
                }
            ),
        )


class DurableCheckpointStore:
    """Descriptor-confined, crash-consistent checkpoint generations."""

    _POINTER_KEYS = frozenset(
        {
            "generation",
            "accepted_step",
            "checkpoint",
            "checkpoint_id",
            "content_digest",
            "manifest_id",
            "runtime_id",
            "encoding_id",
            "store_id",
            "archive_size_bytes",
            "archive_sha256",
            "commit_id",
        }
    )

    def __init__(
        self,
        root: str | Path,
        manifest: ProductionCaseManifest,
        policy: CheckpointGenerationPolicy,
        /,
        *,
        encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
    ):
        self._root_descriptor = -1
        if not isinstance(manifest, ProductionCaseManifest) or not isinstance(
            policy, CheckpointGenerationPolicy
        ):
            raise TypeError("Checkpoint store requires manifest and policy.")
        encoding = (
            RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
        )
        if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
            raise TypeError(
                "encoding_plan must be RuntimeCheckpointEncodingPlan or None."
            )
        self.root = Path(root)
        root_descriptor = open_directory_descriptor(self.root, create=True)
        information = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(information.st_mode)
            or information.st_uid != os.geteuid()
            or information.st_mode & 0o022
        ):
            os.close(root_descriptor)
            raise PermissionError(
                "Checkpoint root must be an owner-controlled non-writable directory."
            )
        self._root_descriptor = root_descriptor
        self.manifest = manifest
        self.policy = policy
        self.encoding_plan = encoding
        self.store_id = canonical_fingerprint(
            {
                "kind": "durable-checkpoint-store",
                "manifest": manifest.manifest_id,
                "policy": policy.policy_id,
                "encoding": encoding.encoding_id,
            }
        )

    def close(self) -> None:
        descriptor = self._root_descriptor
        if descriptor >= 0:
            self._root_descriptor = -1
            os.close(descriptor)

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _generation_name(generation: int) -> str:
        return f"generation-{generation:08d}.phx"

    def _read_pointer(self) -> dict[str, Any]:
        if not _entry_exists(self._root_descriptor, "committed.json"):
            raise FileNotFoundError("No committed checkpoint generation exists.")
        payload = dict(
            _read_json_at(self._root_descriptor, "committed.json", _JSON_FILE_LIMIT)
        )
        if set(payload) != self._POINTER_KEYS:
            raise ValueError("Committed checkpoint pointer schema is corrupt.")
        generation = payload["generation"]
        accepted_step = payload["accepted_step"]
        if (
            type(generation) is not int
            or generation < 0
            or type(accepted_step) is not int
            or accepted_step < 0
        ):
            raise ValueError(
                "Committed checkpoint generation or accepted step is corrupt."
            )
        if payload["checkpoint"] != self._generation_name(generation):
            raise ValueError("Committed checkpoint pointer path is stale or unsafe.")
        identities = (
            "checkpoint_id",
            "content_digest",
            "manifest_id",
            "runtime_id",
            "encoding_id",
            "store_id",
            "archive_sha256",
            "commit_id",
        )
        if any(
            type(payload[name]) is not str
            or len(payload[name]) != 64
            or any(character not in "0123456789abcdef" for character in payload[name])
            for name in identities
        ):
            raise ValueError("Committed checkpoint pointer identity is corrupt.")
        if (
            type(payload["archive_size_bytes"]) is not int
            or payload["archive_size_bytes"] <= 0
            or payload["archive_size_bytes"]
            > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_container_bytes
        ):
            raise ValueError("Committed checkpoint archive size is corrupt.")
        commit_id = payload.pop("commit_id")
        expected_commit_id = canonical_fingerprint(
            {"kind": "durable-checkpoint-commit", **payload}
        )
        payload["commit_id"] = commit_id
        if commit_id != expected_commit_id:
            raise ValueError("Committed checkpoint pointer identity is corrupt.")
        return payload

    def _receipt(self, pointer: Mapping[str, Any], /) -> CheckpointCommitReceipt:
        return CheckpointCommitReceipt(
            self.store_id,
            pointer["checkpoint_id"],
            pointer["content_digest"],
            pointer["runtime_id"],
            pointer["generation"],
            pointer["accepted_step"],
            pointer["commit_id"],
            pointer["checkpoint"],
            pointer["archive_size_bytes"],
            pointer["archive_sha256"],
        )

    def verify_commit(
        self, receipt: CheckpointCommitReceipt, /
    ) -> CheckpointCommitReceipt:
        """Verify a receipt against the currently published durable pointer."""

        if not isinstance(receipt, CheckpointCommitReceipt):
            raise TypeError("receipt must be CheckpointCommitReceipt.")
        pointer = self._read_pointer()
        expected = self._receipt(pointer)
        if receipt != expected:
            raise ValueError("Checkpoint receipt does not match durable store state.")
        archive_size, archive_sha256 = _regular_file_identity_at(
            self._root_descriptor,
            pointer["checkpoint"],
            DEFAULT_ARRAY_ARCHIVE_LIMITS.max_container_bytes,
        )
        if (
            archive_size != receipt.durable_size_bytes
            or archive_sha256 != receipt.durable_sha256
        ):
            raise ValueError("Durable checkpoint archive integrity changed.")
        return expected

    def receipt_for(
        self, envelope: RuntimeCheckpointEnvelope, /
    ) -> CheckpointCommitReceipt:
        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        receipt = self.verify_commit(self._receipt(self._read_pointer()))
        if (
            receipt.checkpoint_id != envelope.checkpoint_id
            or receipt.content_digest != envelope.content_digest
            or receipt.runtime_id != envelope.runtime_id
            or receipt.accepted_step != int(np.asarray(envelope.step_index))
        ):
            raise ValueError("Durable checkpoint does not match the runtime envelope.")
        return receipt

    def generation_for_commit(self, envelope: RuntimeCheckpointEnvelope, /) -> int:
        """Select the store sequence without conflating it with accepted steps."""

        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        if not _entry_exists(self._root_descriptor, "committed.json"):
            return 0
        current = self.verify_commit(self._receipt(self._read_pointer()))
        if (
            current.checkpoint_id == envelope.checkpoint_id
            and current.content_digest == envelope.content_digest
            and current.runtime_id == envelope.runtime_id
            and current.accepted_step == int(np.asarray(envelope.step_index))
        ):
            return current.generation
        return current.generation + 1

    def commit(
        self,
        generation: int,
        envelope: RuntimeCheckpointEnvelope,
        /,
    ) -> CheckpointCommitReceipt:
        if isinstance(generation, bool):
            raise TypeError("Checkpoint generation must be an integer.")
        generation_ = int(generation)
        if generation_ < 0 or not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise ValueError("Checkpoint generation or envelope is invalid.")
        verify_runtime_checkpoint_envelope(envelope)
        if (
            envelope.mesh_id != self.manifest.topology_id
            or envelope.method_id != self.manifest.method_id
            or envelope.precision_id != self.manifest.precision_id
            or envelope.topology_epoch_id != self.manifest.geometry_layout_id
            or envelope.encoding_plan.encoding_id != self.encoding_plan.encoding_id
        ):
            raise ValueError("Checkpoint envelope does not belong to this store.")
        if _entry_exists(self._root_descriptor, "committed.json"):
            current = self._read_pointer()
            if generation_ == current["generation"] and (
                envelope.checkpoint_id == current["checkpoint_id"]
                and envelope.content_digest == current["content_digest"]
                and envelope.runtime_id == current["runtime_id"]
            ):
                return self.verify_commit(self._receipt(current))
            if generation_ <= current["generation"]:
                raise ValueError("Checkpoint generations must increase monotonically.")
        target_name = self._generation_name(generation_)
        if _entry_exists(self._root_descriptor, target_name):
            raise FileExistsError(
                "Checkpoint generation already exists and is immutable."
            )
        temporary_name = f".{target_name}.{secrets.token_hex(16)}.archive"
        try:
            write_runtime_checkpoint(
                descriptor_relative_path(self._root_descriptor, temporary_name),
                envelope,
            )
            archive_size, archive_sha256 = _regular_file_identity_at(
                self._root_descriptor,
                temporary_name,
                DEFAULT_ARRAY_ARCHIVE_LIMITS.max_container_bytes,
            )
            os.replace(
                temporary_name,
                target_name,
                src_dir_fd=self._root_descriptor,
                dst_dir_fd=self._root_descriptor,
            )
            os.fsync(self._root_descriptor)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=self._root_descriptor)
            except FileNotFoundError:
                pass
        pointer = {
            "generation": generation_,
            "accepted_step": int(np.asarray(envelope.step_index)),
            "checkpoint": target_name,
            "checkpoint_id": envelope.checkpoint_id,
            "content_digest": envelope.content_digest,
            "manifest_id": self.manifest.manifest_id,
            "runtime_id": envelope.runtime_id,
            "encoding_id": self.encoding_plan.encoding_id,
            "store_id": self.store_id,
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
        }
        pointer["commit_id"] = canonical_fingerprint(
            {"kind": "durable-checkpoint-commit", **pointer}
        )
        _write_json_atomic_at(self._root_descriptor, "committed.json", pointer)
        generations = sorted(
            name
            for name in os.listdir(self._root_descriptor)
            if len(name) == len("generation-00000000.phx")
            and name.startswith("generation-")
            and name.endswith(".phx")
            and name[11:19].isdigit()
        )
        for obsolete in generations[: -self.policy.retention]:
            os.unlink(obsolete, dir_fd=self._root_descriptor)
        if len(generations) > self.policy.retention:
            os.fsync(self._root_descriptor)
        return self.verify_commit(self._receipt(pointer))

    def latest(
        self,
        state_template: Any,
        /,
        *,
        controller_template: Any = (),
        observer_templates: Sequence[Any] = (),
        rng_template: Any = (),
        runtime_id: str | None = None,
    ) -> RuntimeCheckpointEnvelope:
        pointer = self._read_pointer()
        if (
            pointer["store_id"] != self.store_id
            or pointer["manifest_id"] != self.manifest.manifest_id
        ):
            raise ValueError("Committed checkpoint belongs to another store.")
        if pointer["encoding_id"] != self.encoding_plan.encoding_id:
            raise ValueError("Committed checkpoint encoding identity changed.")
        if runtime_id is not None and pointer["runtime_id"] != str(runtime_id):
            raise ValueError("Committed checkpoint belongs to another prepared runtime.")
        envelope = read_runtime_checkpoint(
            descriptor_relative_path(self._root_descriptor, pointer["checkpoint"]),
            state_template=state_template,
            mesh_id=self.manifest.topology_id,
            method_id=self.manifest.method_id,
            precision_id=self.manifest.precision_id,
            topology_epoch_id=self.manifest.geometry_layout_id,
            controller_template=controller_template,
            observer_templates=observer_templates,
            rng_template=rng_template,
            runtime_id=pointer["runtime_id"],
            encoding_plan=self.encoding_plan,
        )
        if (
            envelope.checkpoint_id != pointer["checkpoint_id"]
            or envelope.content_digest != pointer["content_digest"]
            or int(np.asarray(envelope.step_index)) != pointer["accepted_step"]
        ):
            raise ValueError("Committed checkpoint pointer checksum is stale.")
        self.verify_commit(self._receipt(pointer))
        return envelope

    def commit_terminal(self, payload: Mapping[str, Any], /) -> None:
        _write_json_atomic_at(self._root_descriptor, "terminal.json", payload)


@dataclass(frozen=True, slots=True)
class _RepositoryOutboxEvent:
    event_id: str
    cursor: int
    state: Any
    delivered: bool


class ArtifactCheckpointStore:
    """Repository-backed logical checkpoints with a transactional output outbox."""

    transactional_outbox = True

    def __init__(
        self,
        repository: ArtifactRepository,
        manifest: ProductionCaseManifest,
        policy: CheckpointGenerationPolicy,
        resolved_run_spec: ResolvedRunSpec,
        /,
        *,
        writer_id: str,
        resource_request: ResourceRequest,
        artifact_id: str | None = None,
        encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
    ):
        if not isinstance(manifest, ProductionCaseManifest) or not isinstance(
            policy, CheckpointGenerationPolicy
        ):
            raise TypeError("Artifact checkpoint store requires manifest and policy.")
        if not isinstance(resolved_run_spec, ResolvedRunSpec):
            raise TypeError("resolved_run_spec must be ResolvedRunSpec.")
        if not isinstance(resource_request, ResourceRequest):
            raise TypeError("resource_request must be ResourceRequest.")
        if resource_request.resource_id != resolved_run_spec.resource_policy_id:
            raise ValueError(
                "Resolved resource policy does not match the checkpoint request."
            )
        required_methods = (
            "begin",
            "write_chunk",
            "commit",
            "get_manifest",
            "read_chunk",
        )
        if any(
            not callable(getattr(repository, name, None)) for name in required_methods
        ):
            raise TypeError("repository does not implement ArtifactRepository.")
        if not hasattr(repository, "support_tuple"):
            raise TypeError("repository does not expose its exact support tuple.")
        provider_id = str(getattr(repository, "provider_id", ""))
        if provider_id != resolved_run_spec.repository_id:
            raise ValueError("Resolved repository identity does not match the provider.")
        support_tuple_id = str(getattr(repository.support_tuple, "support_tuple_id", ""))
        deployment_tuple_ids = tuple(
            sorted(
                dependency.support_tuple_id
                for dependency in resolved_run_spec.deployment_dependencies
            )
        )
        if not support_tuple_id or support_tuple_id not in deployment_tuple_ids:
            raise ValueError(
                "Repository support tuple was not admitted by deployment dependencies."
            )
        if resolved_run_spec.checkpoint_policy_id != policy.policy_id:
            raise ValueError(
                "Resolved checkpoint policy does not match the store policy."
            )
        maximum = int(getattr(repository, "maximum_chunk_bytes", 0))
        maximum_manifest = int(getattr(repository, "maximum_metadata_bytes", 0))
        if maximum <= 0 or maximum_manifest <= 0:
            raise ValueError(
                "Repository chunk and metadata byte limits must be positive."
            )
        checkpoint_resources = CheckpointResourcePolicy.from_resource_request(
            resource_request,
            maximum_manifest_bytes=maximum_manifest,
            maximum_chunk_bytes=maximum,
        )
        encoding = (
            RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
        )
        if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
            raise TypeError(
                "encoding_plan must be RuntimeCheckpointEncodingPlan or None."
            )
        writer = str(writer_id)
        if not writer:
            raise ValueError("Repository checkpoint writer_id must be nonempty.")
        artifact = (
            canonical_fingerprint(
                {
                    "kind": "production-checkpoint-artifact",
                    "case": manifest.manifest_id,
                    "resolved_run": resolved_run_spec.spec_id,
                }
            )
            if artifact_id is None
            else str(artifact_id)
        )
        if not artifact:
            raise ValueError("Repository checkpoint artifact_id must be nonempty.")
        self.repository = repository
        self.manifest = manifest
        self.policy = policy
        self.resolved_run_spec = resolved_run_spec
        self.writer_id = writer
        self.artifact_id = artifact
        self.encoding_plan = encoding
        self.resource_request = resource_request
        self.checkpoint_resources = checkpoint_resources
        self.repository_support_tuple_id = support_tuple_id
        self.deployment_support_tuple_ids = deployment_tuple_ids
        self.store_id = canonical_fingerprint(
            {
                "kind": "artifact-checkpoint-store",
                "artifact": artifact,
                "provider": provider_id,
                "repository_support_tuple": support_tuple_id,
                "deployment_support_tuples": deployment_tuple_ids,
                "resolved_run": resolved_run_spec.spec_id,
                "policy": policy.policy_id,
                "encoding": encoding.encoding_id,
                "checkpoint_resources": checkpoint_resources.policy_id,
            }
        )
        self._maximum_manifest_bytes = maximum_manifest
        self._maximum_chunk_bytes = maximum
        self._runtime_id: str | None = None
        self._restart_relation: RuntimeRestartRelation | None = None
        self._migration_report: MigrationReport | None = None
        self._events: tuple[_RepositoryOutboxEvent, ...] = ()
        self._last_envelope: RuntimeCheckpointEnvelope | None = None
        self._last_repository_manifest: ArtifactManifest | None = None
        self._last_generation = -1
        self._last_receipt: CheckpointCommitReceipt | None = None
        self._terminal_payload: Mapping[str, Any] | None = None
        self.last_replay_classification: str | None = None

    def bind_runtime(
        self,
        runtime_id: str,
        relation: RuntimeRestartRelation,
        /,
        *,
        migration_report: MigrationReport | None = None,
    ) -> None:
        """Bind the one prepared runtime and its explicit restart admission."""

        runtime = str(runtime_id)
        if not runtime or not isinstance(relation, RuntimeRestartRelation):
            raise TypeError("Repository runtime binding is invalid.")
        if relation.target_topology_id != self.manifest.topology_id:
            raise ValueError(
                "Restart relation target does not match the prepared topology."
            )
        admitted = set(self.deployment_support_tuple_ids)
        if any(value not in admitted for value in relation.support_tuple_ids):
            raise ValueError("Restart relation cites an unadmitted support tuple.")
        if migration_report is not None:
            if not isinstance(migration_report, MigrationReport):
                raise TypeError("migration_report must be MigrationReport or None.")
            if (
                migration_report.output_digest
                != self.resolved_run_spec.prepared_configuration_id
            ):
                raise ValueError(
                    "Configuration migration output does not match the resolved run."
                )
        bound_relation = self._restart_relation
        if self._runtime_id is not None and (
            self._runtime_id != runtime
            or bound_relation is None
            or bound_relation.relation_id != relation.relation_id
        ):
            raise ValueError(
                "Artifact checkpoint store is already bound to another runtime."
            )
        self._runtime_id = runtime
        self._restart_relation = relation
        self._migration_report = migration_report

    @staticmethod
    def _checkpoint_manifest(envelope: RuntimeCheckpointEnvelope, /) -> dict[str, Any]:
        return {
            "kind": "runtime-checkpoint",
            "checkpoint_id": envelope.checkpoint_id,
            "runtime_id": envelope.runtime_id,
            "content_digest": envelope.content_digest,
            "encoding_id": envelope.encoding_plan.encoding_id,
            "mesh_id": envelope.mesh_id,
            "method_id": envelope.method_id,
            "precision_id": envelope.precision_id,
            "topology_epoch_id": envelope.topology_epoch_id,
            "partition_id": envelope.partition_id,
            **envelope.tree_specs_record(),
        }

    def _write_payload(
        self,
        transaction: Any,
        logical_name: str,
        payload: bytes,
        /,
    ) -> tuple[ChunkRecord, ...]:
        offsets = tuple(range(0, len(payload), self._maximum_chunk_bytes))
        if not offsets:
            offsets = (0,)
        return tuple(
            self.repository.write_chunk(
                transaction,
                logical_name,
                index,
                offset,
                payload[offset : offset + self._maximum_chunk_bytes],
            )
            for index, offset in enumerate(offsets)
        )

    def _validate_payloads(self, payloads: Mapping[str, bytes], /) -> None:
        resources = self.checkpoint_resources
        if len(payloads) > resources.maximum_logical_payloads:
            raise ValueError(
                "Checkpoint exceeds the logical-payload count resource limit."
            )
        total = 0
        chunk_count = 0
        outbox_bytes = 0
        for logical_name, payload in payloads.items():
            size = len(payload)
            if size > resources.maximum_logical_payload_bytes:
                raise ValueError(
                    "Checkpoint logical payload exceeds its resource byte limit."
                )
            if (
                logical_name == "runtime" or logical_name.endswith("-manifest")
            ) and size > resources.maximum_manifest_bytes:
                raise ValueError(
                    "Checkpoint logical manifest exceeds its resource byte limit."
                )
            if total > resources.maximum_total_plaintext_bytes - size:
                raise ValueError(
                    "Checkpoint exceeds the aggregate plaintext resource limit."
                )
            total += size
            chunk_count += max(
                1, (size + self._maximum_chunk_bytes - 1) // self._maximum_chunk_bytes
            )
            if logical_name.startswith("outbox-"):
                outbox_bytes += size
        if chunk_count > resources.maximum_chunks:
            raise ValueError("Checkpoint exceeds the aggregate chunk-count limit.")
        if outbox_bytes > resources.maximum_outbox_bytes:
            raise ValueError("Checkpoint outbox exceeds its resource byte limit.")

    def _snapshot_payloads(
        self,
        envelope: RuntimeCheckpointEnvelope,
        generation: int,
        /,
    ) -> tuple[dict[str, bytes], dict[str, Any]]:
        state_plaintext = sum(
            value.size * np.dtype(value.dtype).itemsize
            for value in envelope.archive_arrays.values()
        )
        outbox_plaintext = sum(
            leaf.size * np.dtype(leaf.dtype).itemsize
            for event in self._events
            for leaf in jax.tree.leaves(event.state)
        )
        if (
            state_plaintext > self.checkpoint_resources.maximum_total_plaintext_bytes
            or outbox_plaintext > self.checkpoint_resources.maximum_outbox_bytes
            or state_plaintext
            > self.checkpoint_resources.maximum_total_plaintext_bytes - outbox_plaintext
        ):
            raise ValueError("Checkpoint array payloads exceed staging resources.")
        state_collection = encode_logical_arrays(
            envelope.archive_arrays, logical_prefix="state"
        )
        payloads = {
            "state-manifest": state_collection.manifest,
            **dict(state_collection.payloads),
        }
        outbox_arrays: dict[str, Any] = {}
        outbox_records = []
        for event in self._events:
            specification = pack_array_tree(
                f"outbox/{event.cursor:016d}",
                event.state,
                outbox_arrays,
            )
            outbox_records.append(
                {
                    "event_id": event.event_id,
                    "cursor": event.cursor,
                    "delivered": event.delivered,
                    "state": specification,
                }
            )
        outbox_collection_id = None
        if outbox_arrays:
            outbox_collection = encode_logical_arrays(
                outbox_arrays, logical_prefix="outbox"
            )
            payloads["outbox-manifest"] = outbox_collection.manifest
            payloads.update(outbox_collection.payloads)
            outbox_collection_id = outbox_collection.collection_id
        relation = self._restart_relation
        if self._runtime_id is None or relation is None:
            raise RuntimeError("Artifact checkpoint store is not bound to a runtime.")
        runtime_record = {
            **self._checkpoint_manifest(envelope),
            "generation": generation,
            "accepted_step": int(np.asarray(envelope.step_index)),
            "state_collection_id": state_collection.collection_id,
            "outbox_collection_id": outbox_collection_id,
            "outbox": outbox_records,
            "terminal": self._terminal_payload,
            "resolved_run_spec": self.resolved_run_spec.to_record(),
            "repository_support_tuple_id": self.repository_support_tuple_id,
            "deployment_support_tuple_ids": list(self.deployment_support_tuple_ids),
            "restart_relation": {
                "source_topology_id": relation.source_topology_id,
                "target_topology_id": relation.target_topology_id,
                "classification": relation.classification,
                "tolerance": relation.tolerance,
                "support_tuple_ids": list(relation.support_tuple_ids),
                "relation_id": relation.relation_id,
            },
            "migration_report": None
            if self._migration_report is None
            else self._migration_report.to_record(),
            "resource_policy_id": self.resolved_run_spec.resource_policy_id,
            "checkpoint_resource_policy_id": self.checkpoint_resources.policy_id,
        }
        payloads["runtime"] = canonical_json(runtime_record).encode("utf-8")
        self._validate_payloads(payloads)
        return payloads, runtime_record

    def _write_snapshot(
        self,
        envelope: RuntimeCheckpointEnvelope,
        /,
        *,
        phase: str,
        generation: int | None = None,
    ) -> ArtifactManifest:
        generation_ = self._last_generation if generation is None else int(generation)
        if generation_ < 0:
            raise ValueError("Repository checkpoint generation is unavailable.")
        payloads, runtime_record = self._snapshot_payloads(envelope, generation_)
        attempt_id = canonical_fingerprint(
            {
                "kind": "production-repository-attempt",
                "checkpoint": envelope.checkpoint_id,
                "generation": generation_,
                "phase": phase,
                "outbox": tuple(
                    (event.event_id, event.cursor, event.delivered)
                    for event in self._events
                ),
                "terminal": None
                if self._terminal_payload is None
                else self._terminal_payload.get("terminal_id"),
            }
        )
        relation = self._restart_relation
        if relation is None:
            raise RuntimeError("Artifact checkpoint store is not bound to a runtime.")
        metadata = {
            "kind": "production-checkpoint",
            "phase": phase,
            "checkpoint_id": envelope.checkpoint_id,
            "content_digest": envelope.content_digest,
            "generation": str(generation_),
            "accepted_step": str(int(np.asarray(envelope.step_index))),
            "runtime_id": envelope.runtime_id,
            "resolved_run_spec_id": self.resolved_run_spec.spec_id,
            "resource_policy_id": self.resolved_run_spec.resource_policy_id,
            "checkpoint_resource_policy_id": self.checkpoint_resources.policy_id,
            "relation_id": relation.relation_id,
            "state_collection_id": runtime_record["state_collection_id"],
        }
        try:
            transaction = self.repository.begin(
                self.artifact_id,
                self.writer_id,
                attempt_id=attempt_id,
            )
            chunks = tuple(
                chunk
                for logical_name, payload in sorted(payloads.items())
                for chunk in self._write_payload(transaction, logical_name, payload)
            )
            committed = self.repository.commit(
                transaction,
                chunks,
                metadata=metadata,
            )
        except RepositoryConflictError:
            committed = self.repository.get_manifest(self.artifact_id)
            committed_metadata = dict(committed.metadata)
            if any(
                committed_metadata.get(name) != value for name, value in metadata.items()
            ):
                raise
        self.checkpoint_resources.validate_manifest(committed)
        self._last_envelope = envelope
        self._last_repository_manifest = committed
        self._last_generation = generation_
        self._last_receipt = self._receipt_from_manifest(committed)
        return committed

    def _read_payloads(self, manifest: ArtifactManifest, /) -> dict[str, bytes]:
        self.checkpoint_resources.validate_manifest(manifest)
        grouped: dict[str, list[ChunkRecord]] = {}
        for chunk in manifest.chunks:
            grouped.setdefault(chunk.logical_name, []).append(chunk)
        payloads: dict[str, bytes] = {}
        for logical_name, chunks in grouped.items():
            ordered = tuple(sorted(chunks, key=lambda value: value.index))
            payload = b"".join(
                self.repository.read_chunk(
                    manifest,
                    chunk,
                    maximum_plaintext_bytes=min(
                        self._maximum_chunk_bytes,
                        self.checkpoint_resources.maximum_logical_payload_bytes,
                    ),
                )
                for chunk in ordered
            )
            expected_size = sum(chunk.plaintext_size for chunk in ordered)
            if len(payload) != expected_size:
                raise RepositoryCorruptionError(
                    "Repository logical payload size differs from its manifest."
                )
            payloads[logical_name] = payload
        return payloads

    def _json_object(self, payload: bytes, role: str, /) -> Mapping[str, Any]:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Repository {role} is not valid JSON.") from error
        _validate_json_nesting(text, self.checkpoint_resources.maximum_json_nesting)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"Repository {role} is not valid JSON.") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"Repository {role} must be a JSON object.")
        return value

    def _configuration_compatible(
        self,
        source: ResolvedRunSpec,
        /,
    ) -> None:
        target = self.resolved_run_spec
        if source.spec_id == target.spec_id:
            return
        source_record = source.to_record()
        target_record = target.to_record()
        for record in (source_record, target_record):
            record.pop("spec_id")
            record.pop("prepared_configuration_id")
        if source_record != target_record:
            raise ValueError(
                "Resolved run changed outside an explicit configuration migration."
            )
        report = self._migration_report
        if report is None or (
            report.input_digest != source.prepared_configuration_id
            or report.output_digest != target.prepared_configuration_id
        ):
            raise ValueError(
                "Configuration changed without an explicit migration lineage."
            )

    def _receipt_from_manifest(
        self, manifest: ArtifactManifest, /
    ) -> CheckpointCommitReceipt:
        self.checkpoint_resources.validate_manifest(manifest)
        metadata = dict(manifest.metadata)
        required = (
            "checkpoint_id",
            "content_digest",
            "generation",
            "accepted_step",
            "runtime_id",
        )
        if any(type(metadata.get(name)) is not str for name in required):
            raise RepositoryCorruptionError(
                "Repository checkpoint commit receipt metadata is invalid."
            )
        generation_text = metadata["generation"]
        accepted_step_text = metadata["accepted_step"]
        if (
            not generation_text.isdigit()
            or not accepted_step_text.isdigit()
            or str(int(generation_text)) != generation_text
            or str(int(accepted_step_text)) != accepted_step_text
        ):
            raise RepositoryCorruptionError(
                "Repository checkpoint generation or accepted step is invalid."
            )
        if (
            metadata.get("resource_policy_id")
            != self.resolved_run_spec.resource_policy_id
            or metadata.get("checkpoint_resource_policy_id")
            != self.checkpoint_resources.policy_id
        ):
            raise RepositoryCorruptionError(
                "Repository checkpoint resource policy identity changed."
            )
        durable_payload = canonical_json(manifest.to_record()).encode("utf-8")
        if len(durable_payload) > self.checkpoint_resources.maximum_manifest_bytes:
            raise RepositoryCorruptionError(
                "Repository checkpoint manifest exceeds its durable byte bound."
            )
        return CheckpointCommitReceipt(
            self.store_id,
            metadata["checkpoint_id"],
            metadata["content_digest"],
            metadata["runtime_id"],
            int(generation_text),
            int(accepted_step_text),
            manifest.manifest_id,
            manifest.artifact_id,
            len(durable_payload),
            hashlib.sha256(durable_payload).hexdigest(),
        )

    def verify_commit(
        self, receipt: CheckpointCommitReceipt, /
    ) -> CheckpointCommitReceipt:
        """Verify a receipt against the repository's readable commit target."""

        if not isinstance(receipt, CheckpointCommitReceipt):
            raise TypeError("receipt must be CheckpointCommitReceipt.")
        committed = self.repository.get_manifest(self.artifact_id)
        self.checkpoint_resources.validate_manifest(committed)
        expected = self._receipt_from_manifest(committed)
        if receipt != expected:
            raise ValueError("Checkpoint receipt does not match repository state.")
        return expected

    def receipt_for(
        self, envelope: RuntimeCheckpointEnvelope, /
    ) -> CheckpointCommitReceipt:
        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        receipt = self.verify_commit(
            self._receipt_from_manifest(self.repository.get_manifest(self.artifact_id))
        )
        if (
            receipt.checkpoint_id != envelope.checkpoint_id
            or receipt.content_digest != envelope.content_digest
            or receipt.runtime_id != envelope.runtime_id
            or receipt.accepted_step != int(np.asarray(envelope.step_index))
        ):
            raise ValueError("Repository commit does not match the runtime envelope.")
        return receipt

    def generation_for_commit(self, envelope: RuntimeCheckpointEnvelope, /) -> int:
        """Select the repository sequence without conflating accepted steps."""

        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        current = self._last_receipt
        if current is None:
            try:
                manifest = self.repository.get_manifest(self.artifact_id)
            except ObjectNotFoundError:
                return 0
            current = self._receipt_from_manifest(manifest)
            self._last_repository_manifest = manifest
            self._last_generation = current.generation
            self._last_receipt = current
        if (
            current.checkpoint_id == envelope.checkpoint_id
            and current.content_digest == envelope.content_digest
            and current.runtime_id == envelope.runtime_id
            and current.accepted_step == int(np.asarray(envelope.step_index))
        ):
            return current.generation
        return current.generation + 1

    def commit(
        self,
        generation: int,
        envelope: RuntimeCheckpointEnvelope,
        /,
    ) -> CheckpointCommitReceipt:
        if isinstance(generation, bool):
            raise TypeError("Checkpoint generation must be an integer.")
        generation_ = int(generation)
        if generation_ < 0 or not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise ValueError("Checkpoint generation or envelope is invalid.")
        verify_runtime_checkpoint_envelope(envelope)
        if (
            envelope.mesh_id != self.manifest.topology_id
            or envelope.method_id != self.manifest.method_id
            or envelope.precision_id != self.manifest.precision_id
            or envelope.topology_epoch_id != self.manifest.geometry_layout_id
            or envelope.encoding_plan.encoding_id != self.encoding_plan.encoding_id
            or envelope.runtime_id != self._runtime_id
        ):
            raise ValueError("Checkpoint envelope does not belong to this store.")
        if self._last_receipt is None:
            self.generation_for_commit(envelope)
        current = self._last_receipt
        if current is not None:
            if (
                envelope.checkpoint_id == current.checkpoint_id
                and envelope.content_digest == current.content_digest
                and envelope.runtime_id == current.runtime_id
                and int(np.asarray(envelope.step_index)) == current.accepted_step
                and generation_ == current.generation
            ):
                return self.verify_commit(current)
            if generation_ <= current.generation:
                raise ValueError("Checkpoint generations must increase monotonically.")
        committed = self._write_snapshot(
            envelope, phase="checkpoint", generation=generation_
        )
        receipt = self._receipt_from_manifest(committed)
        self._last_receipt = receipt
        return self.verify_commit(receipt)

    def stage_output(
        self,
        event_id: str,
        cursor: int,
        state: Any,
        /,
    ) -> None:
        """Stage one immutable ordered output; duplicate event IDs are idempotent."""

        identifier = str(event_id)
        cursor_ = int(cursor)
        if not identifier or cursor_ < 0:
            raise ValueError("Repository output event identity or cursor is invalid.")
        for event in self._events:
            if event.event_id == identifier:
                if event.cursor != cursor_:
                    raise ValueError("Duplicate output event changed its cursor.")
                return
            if event.cursor == cursor_:
                raise ValueError("Output cursor is already bound to another event.")
        expected_cursor = len(self._events)
        if cursor_ != expected_cursor:
            raise ValueError("Repository output cursors must be contiguous and ordered.")
        if len(self._events) >= self.checkpoint_resources.maximum_outbox_records:
            raise ValueError("Repository checkpoint outbox record limit exceeded.")
        canonical = _canonical_structured_state(state)
        candidate_bytes = sum(
            leaf.size * np.dtype(leaf.dtype).itemsize
            for event in self._events
            for leaf in jax.tree.leaves(event.state)
        ) + sum(
            leaf.size * np.dtype(leaf.dtype).itemsize
            for leaf in jax.tree.leaves(canonical)
        )
        if candidate_bytes > self.checkpoint_resources.maximum_outbox_bytes:
            raise ValueError("Repository checkpoint outbox byte limit exceeded.")
        snapshot = jax.tree.map(
            lambda leaf: np.asarray(jax.device_get(leaf)).copy(),
            canonical,
        )
        self._events = self._events + (
            _RepositoryOutboxEvent(identifier, cursor_, snapshot, False),
        )

    def dispatch_outbox(
        self,
        publisher: ByteBoundedAsyncPublisher | None,
        /,
    ) -> None:
        """Deliver committed outbox entries in cursor order and durably acknowledge."""

        pending = tuple(event for event in self._events if not event.delivered)
        if not pending:
            return
        if publisher is None:
            raise ValueError("A publisher is required to dispatch repository outputs.")
        if self._last_envelope is None:
            raise RuntimeError("Outbox entries cannot dispatch before checkpoint commit.")
        for event in pending:
            publisher.publish(event.event_id, event.state)
        publisher.drain()
        delivered_ids = {event.event_id for event in pending}
        self._events = tuple(
            _RepositoryOutboxEvent(
                event.event_id,
                event.cursor,
                event.state,
                event.delivered or event.event_id in delivered_ids,
            )
            for event in self._events
        )
        self._write_snapshot(self._last_envelope, phase="outbox-ack")

    def latest(
        self,
        state_template: Any,
        /,
        *,
        controller_template: Any = (),
        observer_templates: Sequence[Any] = (),
        rng_template: Any = (),
        runtime_id: str | None = None,
    ) -> RuntimeCheckpointEnvelope:
        if self._runtime_id is None or self._restart_relation is None:
            raise RuntimeError("Artifact checkpoint store is not bound to a runtime.")
        if runtime_id is not None and str(runtime_id) != self._runtime_id:
            raise ValueError(
                "Requested runtime identity does not match the store binding."
            )
        repository_manifest = self.repository.get_manifest(self.artifact_id)
        metadata = dict(repository_manifest.metadata)
        committed_receipt = self._receipt_from_manifest(repository_manifest)
        if metadata.get("kind") != "production-checkpoint":
            raise ValueError("Repository artifact is not a production checkpoint.")
        payloads = self._read_payloads(repository_manifest)
        unknown_payloads = {
            name
            for name in payloads
            if name not in {"runtime", "state-manifest", "outbox-manifest"}
            and not name.startswith(("state-", "outbox-"))
        }
        if unknown_payloads:
            raise ValueError("Repository checkpoint contains unknown logical chunks.")
        if "runtime" not in payloads or "state-manifest" not in payloads:
            raise ValueError("Repository checkpoint is missing required logical chunks.")
        runtime_record = self._json_object(payloads["runtime"], "runtime manifest")
        source_spec_record = runtime_record.get("resolved_run_spec")
        if not isinstance(source_spec_record, Mapping):
            raise ValueError("Repository checkpoint has no resolved run specification.")
        source_spec = ResolvedRunSpec.from_record(source_spec_record)
        self._configuration_compatible(source_spec)
        if (
            runtime_record.get("repository_support_tuple_id")
            != self.repository_support_tuple_id
            or tuple(runtime_record.get("deployment_support_tuple_ids", ()))
            != self.deployment_support_tuple_ids
            or runtime_record.get("resource_policy_id")
            != self.resolved_run_spec.resource_policy_id
            or runtime_record.get("checkpoint_resource_policy_id")
            != self.checkpoint_resources.policy_id
        ):
            raise ValueError("Repository deployment or resource-policy identity changed.")
        runtime_generation = runtime_record.get("generation")
        runtime_accepted_step = runtime_record.get("accepted_step")
        if (
            type(runtime_generation) is not int
            or runtime_generation < 0
            or type(runtime_accepted_step) is not int
            or runtime_accepted_step < 0
            or committed_receipt.checkpoint_id != runtime_record.get("checkpoint_id")
            or committed_receipt.content_digest != runtime_record.get("content_digest")
            or committed_receipt.runtime_id != runtime_record.get("runtime_id")
            or committed_receipt.generation != runtime_generation
            or committed_receipt.accepted_step != runtime_accepted_step
        ):
            raise ValueError(
                "Repository commit receipt does not bind its runtime manifest."
            )
        state_payloads = {
            name: payload
            for name, payload in payloads.items()
            if name.startswith("state-") and name != "state-manifest"
        }
        state_arrays = decode_logical_arrays(payloads["state-manifest"], state_payloads)
        state_collection_record = self._json_object(
            payloads["state-manifest"], "state collection manifest"
        )
        if state_collection_record.get("collection_id") != runtime_record.get(
            "state_collection_id"
        ):
            raise ValueError("Runtime manifest does not bind its logical state chunks.")
        checkpoint_manifest = {
            name: value
            for name, value in runtime_record.items()
            if name
            in {
                "kind",
                "checkpoint_id",
                "runtime_id",
                "content_digest",
                "encoding_id",
                "mesh_id",
                "method_id",
                "precision_id",
                "topology_epoch_id",
                "partition_id",
                "state",
                "controller",
                "rng",
                "observers",
            }
        }
        envelope, source_checkpoint_id = restore_runtime_checkpoint_arrays(
            checkpoint_manifest,
            state_arrays,
            state_template=state_template,
            controller_template=controller_template,
            observer_templates=observer_templates,
            rng_template=rng_template,
            target_mesh_id=self.manifest.topology_id,
            target_method_id=self.manifest.method_id,
            target_precision_id=self.manifest.precision_id,
            target_topology_epoch_id=self.manifest.geometry_layout_id,
            target_runtime_id=self._runtime_id,
            restart_relation=self._restart_relation,
            encoding_plan=self.encoding_plan,
        )
        if int(np.asarray(envelope.step_index)) != committed_receipt.accepted_step:
            raise ValueError(
                "Repository accepted step does not bind its checkpoint envelope."
            )
        outbox_records = runtime_record.get("outbox")
        if not isinstance(outbox_records, list):
            raise ValueError("Repository checkpoint outbox schema is invalid.")
        if len(outbox_records) > self.checkpoint_resources.maximum_outbox_records:
            raise ValueError("Repository checkpoint outbox record limit exceeded.")
        if outbox_records:
            if "outbox-manifest" not in payloads:
                raise ValueError("Repository checkpoint outbox arrays are missing.")
            outbox_payloads = {
                name: payload
                for name, payload in payloads.items()
                if name.startswith("outbox-") and name != "outbox-manifest"
            }
            outbox_arrays = decode_logical_arrays(
                payloads["outbox-manifest"], outbox_payloads
            )
            outbox_collection_record = self._json_object(
                payloads["outbox-manifest"], "outbox collection manifest"
            )
            if outbox_collection_record.get("collection_id") != runtime_record.get(
                "outbox_collection_id"
            ):
                raise ValueError("Runtime manifest does not bind its outbox chunks.")
        else:
            if (
                runtime_record.get("outbox_collection_id") is not None
                or "outbox-manifest" in payloads
                or any(name.startswith("outbox-") for name in payloads)
            ):
                raise ValueError("Empty checkpoint outbox has unexpected logical chunks.")
            outbox_arrays = {}
        events = []
        seen_ids = set()
        for expected_cursor, record in enumerate(outbox_records):
            if not isinstance(record, Mapping) or set(record) != {
                "event_id",
                "cursor",
                "delivered",
                "state",
            }:
                raise ValueError("Repository checkpoint outbox record is invalid.")
            event_id = record["event_id"]
            cursor = record["cursor"]
            delivered = record["delivered"]
            if (
                not isinstance(event_id, str)
                or not event_id
                or event_id in seen_ids
                or type(cursor) is not int
                or cursor != expected_cursor
                or type(delivered) is not bool
            ):
                raise ValueError("Repository output cursor ordering is invalid.")
            event_state = unpack_array_tree(
                record["state"], outbox_arrays, state_template
            )
            events.append(
                _RepositoryOutboxEvent(event_id, cursor, event_state, delivered)
            )
            seen_ids.add(event_id)
        controller = envelope.controller_state
        if (
            not isinstance(controller, tuple)
            or len(controller) != 4
            or int(np.asarray(controller[2])) != len(events)
        ):
            raise ValueError("Checkpoint output cursor does not match its outbox.")
        self._events = tuple(events)
        self._last_envelope = envelope
        self._last_repository_manifest = repository_manifest
        self._last_generation = committed_receipt.generation
        self._last_receipt = committed_receipt
        terminal = runtime_record.get("terminal")
        self._terminal_payload = terminal if isinstance(terminal, Mapping) else None
        self.last_replay_classification = self._restart_relation.classification
        if envelope.checkpoint_id != source_checkpoint_id:
            self._terminal_payload = None
            self._write_snapshot(envelope, phase="restart-lineage")
        return envelope

    def commit_terminal(self, payload: Mapping[str, Any], /) -> ArtifactManifest:
        """Commit terminal metadata against the last complete logical checkpoint."""

        if self._last_envelope is None:
            raise RuntimeError("Terminal metadata requires a committed checkpoint.")
        self._terminal_payload = dict(payload)
        return self._write_snapshot(self._last_envelope, phase="terminal")


_FAILURE_CODES = {
    "state-invalid": frozenset({"PRODUCTION_STATE_INVALID"}),
    "step-rejected": frozenset({"PRODUCTION_STEP_REJECTED"}),
    "output-failed": frozenset(
        {
            "PRODUCTION_OUTPUT_PUBLISHER_UNAVAILABLE",
            "PRODUCTION_OUTPUT_PUBLISH_FAILED",
            "PRODUCTION_OUTPUT_DRAIN_FAILED",
        }
    ),
    "step-capacity-exhausted": frozenset({"PRODUCTION_STEP_CAPACITY_EXHAUSTED"}),
}


class ProductionFailureRecord(StrictModule, NonTrainableState):
    step_index: Array
    time: Array
    category: str = eqx.field(static=True)
    error_code: str = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)
    failure_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_index: ArrayLike,
        time: ArrayLike,
        category: str,
        error_code: str,
        last_checkpoint_id: str,
        /,
    ):
        category_ = str(category)
        code = str(error_code)
        if code not in _FAILURE_CODES.get(category_, ()):
            raise ValueError(
                "Production failure category and error_code are not recognized."
            )
        checkpoint = str(last_checkpoint_id)
        if checkpoint and (
            len(checkpoint) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint)
        ):
            raise ValueError("Production failure checkpoint identity is invalid.")
        self.step_index = jnp.asarray(step_index)
        self.time = jnp.asarray(time)
        self.category = category_
        self.error_code = code
        self.last_checkpoint_id = checkpoint
        self.failure_id = canonical_fingerprint(
            {
                "kind": "production-failure-record",
                "step": int(np.asarray(self.step_index)),
                "time": float(np.asarray(self.time)),
                "category": self.category,
                "error_code": self.error_code,
                "last_checkpoint": self.last_checkpoint_id,
            }
        )


class ProductionTerminalManifest(StrictModule, NonTrainableState):
    status: RunStatus = eqx.field(static=True)
    case_manifest_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)
    failure_id: str | None = eqx.field(static=True)
    failure_category: str | None = eqx.field(static=True)
    failure_error_code: str | None = eqx.field(static=True)
    iteration_session_id: str | None = eqx.field(static=True)
    iteration_session_cursor: int = eqx.field(static=True)
    iteration_stop_requested: bool = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: RunStatus,
        case_manifest_id: str,
        run_id: str,
        last_checkpoint_id: str,
        failure: ProductionFailureRecord | None,
        iteration_session_state: IterationSessionState | None = None,
        /,
    ):
        if status not in ("completed", "failed", "canceled"):
            raise ValueError("Terminal manifest status is not terminal.")
        if status == "failed":
            if not isinstance(failure, ProductionFailureRecord):
                raise ValueError("Failed terminal manifests require a failure record.")
        elif failure is not None:
            raise ValueError("Non-failed terminal manifests cannot carry failure.")
        self.status = status
        self.case_manifest_id = str(case_manifest_id)
        self.run_id = str(run_id)
        self.last_checkpoint_id = str(last_checkpoint_id)
        self.failure_id = None if failure is None else failure.failure_id
        self.failure_category = None if failure is None else failure.category
        self.failure_error_code = None if failure is None else failure.error_code
        self.iteration_session_id = (
            None
            if iteration_session_state is None
            else iteration_session_state.session_id
        )
        self.iteration_session_cursor = (
            0 if iteration_session_state is None else iteration_session_state.cursor
        )
        self.iteration_stop_requested = (
            False
            if iteration_session_state is None
            else iteration_session_state.stop_requested
        )
        self.terminal_id = canonical_fingerprint(
            {
                "kind": "production-terminal-manifest",
                "status": status,
                "case": self.case_manifest_id,
                "run": self.run_id,
                "last_checkpoint": self.last_checkpoint_id,
                "failure": self.failure_id,
                "failure_category": self.failure_category,
                "failure_error_code": self.failure_error_code,
                "iteration_session": self.iteration_session_id,
                "iteration_cursor": self.iteration_session_cursor,
                "iteration_stop_requested": self.iteration_stop_requested,
            }
        )

    def payload(self, /) -> dict[str, Any]:
        return {
            "status": self.status,
            "case_manifest_id": self.case_manifest_id,
            "run_id": self.run_id,
            "last_checkpoint_id": self.last_checkpoint_id,
            "failure_id": self.failure_id,
            "failure_category": self.failure_category,
            "failure_error_code": self.failure_error_code,
            "iteration_session_id": self.iteration_session_id,
            "iteration_session_cursor": self.iteration_session_cursor,
            "iteration_stop_requested": self.iteration_stop_requested,
            "terminal_id": self.terminal_id,
        }


class ProductionTriggerBinding(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    graph: AcceptedStepTriggerGraph
    moment_indices: tuple[int, ...] = eqx.field(static=True)
    moment_components: tuple[int, ...] = eqx.field(static=True)
    action: ProductionTriggerAction = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        graph: AcceptedStepTriggerGraph,
        moment_indices: Sequence[int],
        action: ProductionTriggerAction,
        action_id: str,
        /,
        *,
        moment_components: Sequence[int] = (),
    ):
        name_ = str(name)
        indices = tuple(moment_indices)
        components = tuple(moment_components)
        action_id_ = str(action_id)
        if (
            not name_
            or not isinstance(graph, AcceptedStepTriggerGraph)
            or len(indices) != len(graph.triggers)
            or any(value < 0 for value in indices)
            or (components and len(components) != len(indices))
            or any(value < 0 for value in components)
            or action not in ("checkpoint", "publish", "stop")
            or not action_id_
        ):
            raise ValueError("Production trigger binding is invalid.")
        self.name = name_
        self.graph = graph
        self.moment_indices = indices
        self.moment_components = components
        self.action = action
        self.action_id = action_id_
        self.binding_id = canonical_fingerprint(
            {
                "kind": "production-trigger-binding",
                "name": name_,
                "graph": graph.graph_id,
                "moment_indices": indices,
                "moment_components": components,
                "action": action,
                "action_id": action_id_,
            }
        )


class ProductionIterationMetrics(StrictModule):
    """Typed evidence emitted at one production host transaction boundary."""

    time: Array
    accepted_step_size: Array
    retry_count: Array
    method_successful: Array
    accepted: Array
    output_due: Array
    checkpoint_due: Array

    def __init__(
        self,
        time,
        accepted_step_size,
        retry_count,
        method_successful,
        accepted,
        output_due,
        checkpoint_due,
        /,
    ):
        self.time = jnp.asarray(time)
        self.accepted_step_size = jnp.asarray(accepted_step_size)
        self.retry_count = jnp.asarray(retry_count, dtype=jnp.int32)
        self.method_successful = jnp.asarray(method_successful, dtype=jnp.bool_)
        self.accepted = jnp.asarray(accepted, dtype=jnp.bool_)
        self.output_due = jnp.asarray(output_due, dtype=jnp.bool_)
        self.checkpoint_due = jnp.asarray(checkpoint_due, dtype=jnp.bool_)


def _production_iteration_record(
    phase,
    step_index,
    metrics: ProductionIterationMetrics,
    /,
    *,
    active=True,
    committed=False,
    terminal=False,
    status=0,
) -> IterationRecord:
    return IterationRecord(
        IterationCoordinates(
            phase,
            step_index,
            attempt=step_index,
            accepted=step_index,
            active=active,
            committed=committed,
            terminal=terminal,
        ),
        status,
        metrics,
    )


class ProductionRunState(StrictModule):
    step_index: Array
    time: Array
    accepted_state: PyTree[Array]
    controller_state: Any
    rng_state: Any
    schedule_cursor: Array
    moment_states: tuple[StreamingMomentState, ...]
    trigger_states: tuple[AcceptedStepTriggerGraphState, ...]
    output_cursor: Array
    status: RunStatus = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)


def _replace_run_metadata(
    state: ProductionRunState,
    /,
    *,
    status: RunStatus | None = None,
    last_checkpoint_id: str | None = None,
) -> ProductionRunState:
    return ProductionRunState(
        state.step_index,
        state.time,
        state.accepted_state,
        state.controller_state,
        state.rng_state,
        state.schedule_cursor,
        state.moment_states,
        state.trigger_states,
        state.output_cursor,
        state.status if status is None else status,
        state.last_checkpoint_id if last_checkpoint_id is None else last_checkpoint_id,
    )


def _replace_output_cursor(
    state: ProductionRunState, cursor: int, /
) -> ProductionRunState:
    return ProductionRunState(
        state.step_index,
        state.time,
        state.accepted_state,
        state.controller_state,
        state.rng_state,
        state.schedule_cursor,
        state.moment_states,
        state.trigger_states,
        jnp.asarray(cursor, dtype=state.output_cursor.dtype),
        state.status,
        state.last_checkpoint_id,
    )


class ProductionRunResult(StrictModule):
    state: ProductionRunState
    successful: Array
    failure: ProductionFailureRecord | None
    run_id: str = eqx.field(static=True)
    iteration_session_state: IterationSessionState | None = eqx.field(static=True)


class ProductionRunPlan(StrictModule, NonTrainableState):
    method: AbstractFixedStepMethod
    retry_policy: RobustRetryPolicy
    output_schedule: ExactTimeSchedule | None
    moments: tuple[StreamingMomentPlan, ...]
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    validator: Callable = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    checkpoint_interval: int = eqx.field(static=True)
    segment_steps: int = eqx.field(static=True)
    device_resident: bool = eqx.field(static=True)
    validator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        retry_policy: RobustRetryPolicy,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 32,
        output_schedule: ExactTimeSchedule | None = None,
        moments: Sequence[StreamingMomentPlan] = (),
        trigger_bindings: Sequence[ProductionTriggerBinding] = (),
        validator: Callable | None = None,
        validator_id: str | None = None,
        device_resident: bool = False,
    ):
        step = float(step_size)
        end = float(end_time)
        steps = int(maximum_steps)
        interval = int(checkpoint_interval)
        segment = int(segment_steps)
        moments_ = tuple(moments)
        bindings = tuple(trigger_bindings)
        if not isinstance(device_resident, bool):
            raise TypeError("device_resident must be Boolean.")
        if validator is None:
            if validator_id is not None:
                raise ValueError("validator_id requires a supplied validator.")
            validator_ = _finite_array_tree
            validator_identifier = "production-validator:finite-array-tree"
        else:
            validator_ = validator
            validator_identifier = "" if validator_id is None else str(validator_id)
            if not callable(validator_) or not validator_identifier:
                raise ValueError("A supplied validator requires a stable validator_id.")
        if (
            not isinstance(method, AbstractFixedStepMethod)
            or not isinstance(retry_policy, RobustRetryPolicy)
            or not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(end)
            or steps <= 0
            or interval <= 0
            or segment <= 0
            or (
                output_schedule is not None
                and not isinstance(output_schedule, ExactTimeSchedule)
            )
            or any(not isinstance(value, StreamingMomentPlan) for value in moments_)
            or any(not isinstance(value, ProductionTriggerBinding) for value in bindings)
        ):
            raise ValueError("Production run plan is invalid.")
        if any(
            index >= len(moments_)
            for binding in bindings
            for index in binding.moment_indices
        ):
            raise ValueError(
                "Trigger bindings reference an unavailable streaming moment."
            )
        for binding in bindings:
            if binding.moment_components:
                for index, component in zip(
                    binding.moment_indices,
                    binding.moment_components,
                    strict=True,
                ):
                    value_size = int(np.prod(moments_[index].value_shape, dtype=np.int64))
                    if component >= max(value_size, 1):
                        raise ValueError(
                            "Trigger component is outside its streaming moment."
                        )
            elif any(moments_[index].value_shape for index in binding.moment_indices):
                raise ValueError(
                    "Vector streaming moments require explicit trigger components."
                )
        if (
            output_schedule is not None
            and float(np.asarray(output_schedule.targets[-1]))
            > end + output_schedule.tolerance
        ):
            raise ValueError("Production output targets cannot exceed end_time.")
        required = method.required_step_size
        if required is not None and (
            not math.isfinite(float(required))
            or not np.isclose(step, float(required), rtol=0.0, atol=0.0)
        ):
            raise ValueError("Production step_size is incompatible with the method.")
        if retry_policy.maximum_retries and not method.allows_step_reduction:
            raise ValueError("Production retry policy requires forbidden step reduction.")
        self.method = method
        self.retry_policy = retry_policy
        self.output_schedule = output_schedule
        self.moments = moments_
        self.trigger_bindings = bindings
        self.validator = validator_
        self.step_size = step
        self.end_time = end
        self.maximum_steps = steps
        self.checkpoint_interval = interval
        self.segment_steps = segment
        self.device_resident = device_resident
        self.validator_id = validator_identifier
        identity = {
            "kind": "production-run-plan",
            "method": method.method_id,
            "retry": retry_policy.policy_id,
            "step_size": step,
            "end_time": end,
            "maximum_steps": steps,
            "checkpoint_interval": interval,
            "segment_steps": segment,
            "output_schedule": None
            if output_schedule is None
            else output_schedule.schedule_id,
            "moments": tuple(value.plan_id for value in moments_),
            "trigger_bindings": tuple(value.binding_id for value in bindings),
            "validator": validator_identifier,
        }
        if device_resident:
            identity["device_resident"] = True
        self.plan_id = canonical_fingerprint(identity)


class _SegmentState(StrictModule):
    step_index: Array
    time: Array
    accepted_state: PyTree[Array]
    schedule_cursor: Array
    moment_states: tuple[StreamingMomentState, ...]
    trigger_states: tuple[AcceptedStepTriggerGraphState, ...]
    output_cursor: Array
    running: Array
    stop_requested: Array


class _SegmentRecord(StrictModule):
    state: _SegmentState
    result: RetriedFixedStepResult
    attempted: Array
    accepted: Array
    method_successful: Array
    schedule_cursor_before: Array
    output_due: Array
    checkpoint_due: Array
    trigger_fires: tuple[Array, ...]


class PreparedProductionRun:
    """Bound production runtime with bounded compiled scan segments."""

    def __init__(
        self,
        manifest: ProductionCaseManifest,
        plan: ProductionRunPlan,
        checkpoint_store: DurableCheckpointStore | ArtifactCheckpointStore,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
        session: IterationSession | None = None,
        resolved_run_spec: ResolvedRunSpec | None = None,
        restart_relation: RuntimeRestartRelation | None = None,
        migration_report: MigrationReport | None = None,
    ):
        if (
            not isinstance(manifest, ProductionCaseManifest)
            or not isinstance(plan, ProductionRunPlan)
            or not isinstance(
                checkpoint_store, (DurableCheckpointStore, ArtifactCheckpointStore)
            )
            or checkpoint_store.manifest.manifest_id != manifest.manifest_id
        ):
            raise TypeError("Prepared production run inputs are incompatible.")
        if manifest.method_id != plan.method.method_id:
            raise ValueError(
                "Production manifest method identity does not match the plan."
            )
        if publisher is not None and not isinstance(publisher, ByteBoundedAsyncPublisher):
            raise TypeError("publisher must be ByteBoundedAsyncPublisher or None.")
        if session is not None and not isinstance(session, IterationSession):
            raise TypeError("session must be IterationSession or None.")
        if publisher is None and (
            plan.output_schedule is not None
            or any(value.action == "publish" for value in plan.trigger_bindings)
        ):
            raise ValueError("Scheduled or triggered publication requires a publisher.")
        if args_id is None:
            if args is not None:
                raise ValueError("Bound production args require a stable args_id.")
            args_identifier = "production-args:none"
        else:
            args_identifier = str(args_id)
            if not args_identifier:
                raise ValueError("args_id must be nonempty when supplied.")
        if isinstance(checkpoint_store, ArtifactCheckpointStore):
            resolved = checkpoint_store.resolved_run_spec
            if resolved_run_spec is not None and (
                not isinstance(resolved_run_spec, ResolvedRunSpec)
                or resolved_run_spec.spec_id != resolved.spec_id
            ):
                raise ValueError(
                    "Prepared resolved run does not match the repository store binding."
                )
            relation = (
                RuntimeRestartRelation.identity(manifest.topology_id)
                if restart_relation is None
                else restart_relation
            )
            if not isinstance(relation, RuntimeRestartRelation):
                raise TypeError(
                    "restart_relation must be RuntimeRestartRelation or None."
                )
            persistence_identity = {
                "kind": "artifact",
                "store": checkpoint_store.store_id,
                "resolved_run": resolved.spec_id,
                "repository_support_tuple": (
                    checkpoint_store.repository_support_tuple_id
                ),
                "deployment_support_tuples": (
                    checkpoint_store.deployment_support_tuple_ids
                ),
                "restart_relation": relation.relation_id,
                "replay_classification": relation.classification,
                "migration_report": None
                if migration_report is None
                else migration_report.report_id,
            }
        else:
            if resolved_run_spec is not None and not isinstance(
                resolved_run_spec, ResolvedRunSpec
            ):
                raise TypeError("resolved_run_spec must be ResolvedRunSpec or None.")
            if restart_relation is not None or migration_report is not None:
                raise ValueError(
                    "Topology and configuration migration require an artifact repository."
                )
            resolved = resolved_run_spec
            relation = RuntimeRestartRelation.identity(manifest.topology_id)
            persistence_identity = None
        self.manifest = manifest
        self.plan = plan
        self.checkpoint_store = checkpoint_store
        self.args = args
        self.args_id = args_identifier
        self.publisher = publisher
        self.iteration_session = session
        if session is None:
            self.iteration_scope = None
        else:
            iteration_plan = IterationPlan(granularity="step")
            iteration_capabilities = IterationCapabilities(
                ("terminal", "segment", "step"),
                host_stop=True,
                host_streaming=True,
                checkpointable=True,
            )
            self.iteration_scope = bind_iteration_scope(
                iteration_plan,
                iteration_capabilities,
                f"production:{plan.method.method_id}",
            )
        self.resolved_run_spec = resolved
        self.restart_relation = relation
        self.migration_report = migration_report
        self._terminal_iteration_emitted = False
        identity = {
            "kind": "prepared-production-run",
            "manifest": manifest.manifest_id,
            "plan": plan.plan_id,
            "checkpoint_policy": checkpoint_store.policy.policy_id,
            "checkpoint_encoding": checkpoint_store.encoding_plan.encoding_id,
            "args": args_identifier,
            "iteration_session": None if session is None else session.session_id,
            "iteration_control": None if session is None else session.control_id,
        }
        if persistence_identity is not None:
            identity["persistence"] = persistence_identity
        elif resolved is not None:
            identity["resolved_run"] = resolved.spec_id
        self.run_id = canonical_fingerprint(identity)
        if isinstance(checkpoint_store, ArtifactCheckpointStore):
            checkpoint_store.bind_runtime(
                self.run_id,
                relation,
                migration_report=migration_report,
            )
        self.last_replay_classification: str | None = None
        self._last_checkpoint_receipt: CheckpointCommitReceipt | None = None
        self._compiled_segment = self._compile_segment(plan.segment_steps)
        self._compiled_one_step = self._compile_segment(1)

    def _compile_segment(self, length: int, /):
        plan = self.plan
        args = self.args
        retry_decision_id = canonical_fingerprint(
            {
                "kind": "retried-fixed-step-decision",
                "method": plan.method.method_id,
                "retry_policy": plan.retry_policy.policy_id,
            }
        )
        attempt_count = plan.retry_policy.maximum_retries + 1

        def scan_step(carry: _SegmentState, unused: None):
            del unused
            tolerance = jnp.asarray(
                32.0 * jnp.finfo(carry.time.dtype).eps, dtype=carry.time.dtype
            )
            active = (
                carry.running
                & ~carry.stop_requested
                & (carry.step_index < plan.maximum_steps)
                & (carry.time < plan.end_time - tolerance)
            )
            proposed_step = jnp.minimum(
                jnp.asarray(plan.step_size, dtype=carry.time.dtype),
                jnp.asarray(plan.end_time, dtype=carry.time.dtype) - carry.time,
            )
            if plan.output_schedule is not None:
                proposed_step = plan.output_schedule.clamp_step(
                    carry.time, proposed_step, carry.schedule_cursor
                )

            def advance(_: None) -> RetriedFixedStepResult:
                return retry_fixed_step(
                    plan.method,
                    plan.retry_policy,
                    carry.step_index,
                    carry.time,
                    carry.accepted_state,
                    proposed_step,
                    args,
                )

            def inactive(_: None) -> RetriedFixedStepResult:
                return RetriedFixedStepResult(
                    carry.accepted_state,
                    carry.accepted_state,
                    jnp.asarray(True),
                    jnp.zeros((), dtype=carry.time.dtype),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros((attempt_count,), dtype=carry.time.dtype),
                    retry_decision_id,
                )

            result = jax.lax.cond(active, advance, inactive, operand=None)
            valid = jnp.asarray(plan.validator(result.accepted_state))
            if valid.shape != () or valid.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(
                    "Production validators must return a scalar Boolean array."
                )
            accepted = active & result.successful & valid
            next_time = carry.time + jnp.where(accepted, result.accepted_step_size, 0.0)
            next_step = carry.step_index + accepted.astype(carry.step_index.dtype)
            next_state = tree_where(accepted, result.accepted_state, carry.accepted_state)
            proposed_moments = tuple(
                moment.update(
                    next_time,
                    next_state,
                    moment_state,
                    previous_time=carry.time,
                    args=args,
                )
                for moment, moment_state in zip(
                    plan.moments, carry.moment_states, strict=True
                )
            )
            moment_states = tuple(
                tree_where(accepted, proposed, current)
                for proposed, current in zip(
                    proposed_moments, carry.moment_states, strict=True
                )
            )
            trigger_states = []
            trigger_fires = []
            for binding, trigger_state in zip(
                plan.trigger_bindings, carry.trigger_states, strict=True
            ):
                if binding.moment_components:
                    values = tuple(
                        moment_states[index].mean.reshape((-1,))[component]
                        for index, component in zip(
                            binding.moment_indices,
                            binding.moment_components,
                            strict=True,
                        )
                    )
                else:
                    values = tuple(
                        moment_states[index].mean.reshape(())
                        for index in binding.moment_indices
                    )
                fire, proposed_trigger = binding.graph.evaluate(
                    values, trigger_state, accepted=accepted
                )
                trigger_fires.append(fire)
                trigger_states.append(
                    tree_where(accepted, proposed_trigger, trigger_state)
                )
            if plan.output_schedule is None:
                schedule_cursor = carry.schedule_cursor
                output_due = jnp.asarray(False)
            else:
                proposed_cursor = plan.output_schedule.advance_cursor(
                    next_time, carry.schedule_cursor
                )
                schedule_cursor = jnp.where(
                    accepted, proposed_cursor, carry.schedule_cursor
                )
                output_due = accepted & (schedule_cursor > carry.schedule_cursor)
            stop_fire = jnp.asarray(False)
            publish_count = jnp.asarray(0, dtype=carry.output_cursor.dtype)
            for binding, fire in zip(plan.trigger_bindings, trigger_fires, strict=True):
                if binding.action == "stop":
                    stop_fire = stop_fire | fire
                elif binding.action == "publish":
                    publish_count = publish_count + fire.astype(publish_count.dtype)
            output_increment = (schedule_cursor - carry.schedule_cursor).astype(
                carry.output_cursor.dtype
            ) + publish_count
            running = carry.running & ~(active & ~accepted)
            next_carry = _SegmentState(
                next_step,
                next_time,
                next_state,
                schedule_cursor,
                moment_states,
                tuple(trigger_states),
                carry.output_cursor + output_increment,
                running,
                carry.stop_requested | stop_fire,
            )
            checkpoint_due = accepted & (next_step % plan.checkpoint_interval == 0)
            record = _SegmentRecord(
                next_carry,
                result,
                active,
                accepted,
                result.successful,
                carry.schedule_cursor,
                output_due,
                checkpoint_due,
                tuple(trigger_fires),
            )
            return next_carry, record

        @jax.jit
        def execute(initial: _SegmentState):
            return jax.lax.scan(scan_step, initial, xs=None, length=length)

        return execute

    def _preflight_horizon(self, time: ArrayLike, step_index: ArrayLike, /) -> None:
        start = float(np.asarray(time))
        step = int(np.asarray(step_index))
        tolerance = 32.0 * np.finfo(np.asarray(time).dtype).eps
        if not math.isfinite(start) or start > self.plan.end_time + tolerance:
            raise ValueError("Production start time exceeds the absolute end_time.")
        if step < 0 or step > self.plan.maximum_steps:
            raise ValueError("Production step index exceeds absolute step capacity.")
        remaining = max(self.plan.end_time - start, 0.0)
        nominal_steps = int(
            math.ceil(max(remaining - tolerance, 0.0) / self.plan.step_size)
        )
        if step + nominal_steps > self.plan.maximum_steps:
            raise ValueError(
                "Absolute step capacity cannot reach end_time from this state."
            )
        if not self.plan.method.allows_step_reduction:
            points = [self.plan.end_time]
            if self.plan.output_schedule is not None:
                points.extend(
                    float(value)
                    for value in np.asarray(self.plan.output_schedule.targets)
                    if float(value) > start + self.plan.output_schedule.tolerance
                )
            method_tolerance = self.plan.method.schedule_alignment_tolerance
            if method_tolerance is not None and (
                not math.isfinite(float(method_tolerance))
                or float(method_tolerance) < 0.0
            ):
                raise ValueError("Method schedule alignment tolerance is invalid.")
            for point in points:
                raw = (point - start) / self.plan.step_size
                rounded = round(raw)
                aligned = (
                    np.isclose(raw, rounded, rtol=1.0e-12, atol=1.0e-12)
                    if method_tolerance is None
                    else abs(point - (start + rounded * self.plan.step_size))
                    <= float(method_tolerance)
                )
                if raw < -tolerance or not aligned:
                    raise ValueError(
                        "Method step-reduction constraints are incompatible with the runtime horizon."
                    )

    def initial_state(
        self,
        state: Any,
        /,
        *,
        time: ArrayLike = 0.0,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        if self.iteration_session is not None and (
            self.iteration_session.cursor != 0 or self.iteration_session.stop_requested
        ):
            raise ValueError("A new production state requires a fresh iteration session.")
        value = _canonical_structured_state(state)
        dtype = _state_dtype(value)
        if str(jnp.dtype(dtype)) != self.manifest.dtype:
            raise ValueError("Initial state precision does not match the case manifest.")
        time_dtype = jnp.asarray(0, dtype=dtype).real.dtype
        time_ = jnp.asarray(time, dtype=time_dtype)
        if time_.shape != ():
            raise ValueError("Production start time must be scalar.")
        self._preflight_horizon(time_, jnp.asarray(0, dtype=jnp.int64))
        schedule_cursor = (
            jnp.asarray(0, dtype=jnp.int64)
            if self.plan.output_schedule is None
            else self.plan.output_schedule.initial_cursor(time_).astype(jnp.int64)
        )
        return ProductionRunState(
            jnp.asarray(0, dtype=jnp.int64),
            time_,
            value,
            _canonical_auxiliary_tree(controller_state, "controller state"),
            _canonical_auxiliary_tree(rng_state, "RNG state"),
            schedule_cursor,
            tuple(moment.initial_state(time_dtype) for moment in self.plan.moments),
            tuple(
                binding.graph.initial_state(time_dtype)
                for binding in self.plan.trigger_bindings
            ),
            jnp.asarray(0, dtype=jnp.int64),
            "ready",
            "",
        )

    def _iteration_session_checkpoint(self, /):
        if self.iteration_session is None:
            return ()
        state = self.iteration_session.snapshot()
        return (
            jnp.asarray(state.cursor, dtype=jnp.int64),
            jnp.asarray(state.stop_requested, dtype=jnp.bool_),
        )

    def _envelope(self, state: ProductionRunState, /) -> RuntimeCheckpointEnvelope:
        return RuntimeCheckpointEnvelope(
            state.accepted_state,
            time=state.time,
            step_index=state.step_index,
            schedule_cursor=state.schedule_cursor,
            mesh_id=self.manifest.topology_id,
            method_id=self.manifest.method_id,
            precision_id=self.manifest.precision_id,
            topology_epoch_id=self.manifest.geometry_layout_id,
            controller_state=(
                state.controller_state,
                state.trigger_states,
                state.output_cursor,
                self._iteration_session_checkpoint(),
            ),
            observer_states=state.moment_states,
            rng_state=state.rng_state,
            runtime_id=self.run_id,
            encoding_plan=self.checkpoint_store.encoding_plan,
        )

    def commit_checkpoint(
        self, state: ProductionRunState, /
    ) -> tuple[ProductionRunState, CheckpointCommitReceipt]:
        """Durably commit state and return store-verified commit evidence."""

        envelope = self._envelope(state)
        generation = self.checkpoint_store.generation_for_commit(envelope)
        receipt = self.checkpoint_store.commit(generation, envelope)
        verified = self.checkpoint_store.verify_commit(receipt)
        if (
            verified.checkpoint_id != envelope.checkpoint_id
            or verified.content_digest != envelope.content_digest
            or verified.runtime_id != self.run_id
            or verified.accepted_step != int(np.asarray(state.step_index))
        ):
            raise ValueError("Checkpoint commit receipt does not bind runtime state.")
        self._last_checkpoint_receipt = verified
        return (
            _replace_run_metadata(state, last_checkpoint_id=verified.checkpoint_id),
            verified,
        )

    def checkpoint(self, state: ProductionRunState, /) -> ProductionRunState:
        checkpointed, _receipt = self.commit_checkpoint(state)
        return checkpointed

    def resume(self, template: ProductionRunState, /) -> ProductionRunState:
        envelope = self.checkpoint_store.latest(
            template.accepted_state,
            controller_template=(
                template.controller_state,
                template.trigger_states,
                template.output_cursor,
                self._iteration_session_checkpoint(),
            ),
            observer_templates=template.moment_states,
            rng_template=template.rng_state,
            runtime_id=self.run_id,
        )
        if isinstance(self.checkpoint_store, ArtifactCheckpointStore):
            self.last_replay_classification = (
                self.checkpoint_store.last_replay_classification
            )
            self.checkpoint_store.dispatch_outbox(self.publisher)
        self._last_checkpoint_receipt = self.checkpoint_store.receipt_for(envelope)
        controller, triggers, output_cursor, iteration_session_state = (
            envelope.controller_state
        )
        if self.iteration_session is not None:
            cursor, stop_requested = iteration_session_state
            self.iteration_session.restore(
                IterationSessionState(
                    self.iteration_session.session_id,
                    int(np.asarray(cursor)),
                    bool(np.asarray(stop_requested)),
                )
            )
        self._preflight_horizon(envelope.time, envelope.step_index)
        expected_cursor = (
            0
            if self.plan.output_schedule is None
            else int(np.asarray(self.plan.output_schedule.initial_cursor(envelope.time)))
        )
        if int(np.asarray(envelope.schedule_cursor)) != expected_cursor:
            raise ValueError("Checkpoint output schedule cursor is stale.")
        return ProductionRunState(
            envelope.step_index,
            envelope.time,
            envelope.state,
            controller,
            envelope.rng_state,
            envelope.schedule_cursor,
            envelope.observer_states,
            triggers,
            output_cursor,
            "ready",
            envelope.checkpoint_id,
        )

    def _commit_terminal(
        self, state: ProductionRunState, failure: ProductionFailureRecord | None, /
    ) -> ProductionTerminalManifest:
        checkpoint_id = ""
        if state.last_checkpoint_id:
            receipt = self._last_checkpoint_receipt
            if receipt is None or receipt.checkpoint_id != state.last_checkpoint_id:
                raise ValueError(
                    "Terminal state checkpoint has no store-issued commit receipt."
                )
            checkpoint_id = self.checkpoint_store.verify_commit(receipt).checkpoint_id
        terminal = ProductionTerminalManifest(
            state.status,
            self.manifest.manifest_id,
            self.run_id,
            checkpoint_id,
            failure,
            (
                None
                if self.iteration_session is None
                else self.iteration_session.snapshot()
            ),
        )
        self.checkpoint_store.commit_terminal(terminal.payload())
        return terminal

    def _index_tree(self, tree: Any, index: int, /) -> Any:
        if self.plan.device_resident:
            return jax.tree.map(lambda leaf: leaf[index], tree)
        return jax.tree.map(lambda leaf: np.asarray(leaf)[index], tree)

    @staticmethod
    def _place_tree_like(value: Any, reference: Any, /) -> Any:
        if jax.tree.structure(value) != jax.tree.structure(reference):
            raise ValueError("Device-resident state changed its PyTree structure.")
        return jax.tree.map(
            lambda leaf, template: (
                jax.device_put(leaf, template.sharding)
                if isinstance(template, jax.Array)
                else leaf
            ),
            value,
            reference,
        )

    @classmethod
    def _place_production_state_like(
        cls,
        state: ProductionRunState,
        reference: ProductionRunState,
        /,
    ) -> ProductionRunState:
        return ProductionRunState(
            state.step_index,
            state.time,
            cls._place_tree_like(state.accepted_state, reference.accepted_state),
            state.controller_state,
            state.rng_state,
            state.schedule_cursor,
            state.moment_states,
            state.trigger_states,
            state.output_cursor,
            state.status,
            state.last_checkpoint_id,
        )

    def _segment_initial(self, state: ProductionRunState, /) -> _SegmentState:
        return _SegmentState(
            state.step_index,
            state.time,
            state.accepted_state,
            state.schedule_cursor,
            state.moment_states,
            state.trigger_states,
            state.output_cursor,
            jnp.asarray(True),
            jnp.asarray(
                False
                if self.iteration_session is None
                else self.iteration_session.stop_requested
            ),
        )

    def _production_state(
        self,
        segment: _SegmentState,
        source: ProductionRunState,
        status: RunStatus,
        last_checkpoint_id: str,
        /,
    ) -> ProductionRunState:
        return ProductionRunState(
            segment.step_index,
            segment.time,
            segment.accepted_state,
            source.controller_state,
            source.rng_state,
            segment.schedule_cursor,
            segment.moment_states,
            segment.trigger_states,
            segment.output_cursor,
            status,
            last_checkpoint_id,
        )

    def _publish(
        self,
        event_id: str,
        cursor: int,
        state: PyTree[Array],
        /,
    ) -> str | None:
        if self.publisher is None:
            return "PRODUCTION_OUTPUT_PUBLISHER_UNAVAILABLE"
        try:
            if isinstance(self.checkpoint_store, ArtifactCheckpointStore):
                self.checkpoint_store.stage_output(event_id, cursor, state)
            else:
                self.publisher.publish(event_id, state)
        except Exception as error:
            emit(
                "ERROR",
                "solver.production.output_publish_failed",
                "Production output publication failed",
                diagnostic=str(error),
                error_type=type(error).__name__,
            )
            return "PRODUCTION_OUTPUT_PUBLISH_FAILED"
        return None

    def _drain_outputs(self, /) -> str | None:
        if self.publisher is None:
            return None
        try:
            if isinstance(self.checkpoint_store, ArtifactCheckpointStore):
                self.checkpoint_store.dispatch_outbox(self.publisher)
            else:
                self.publisher.drain()
        except Exception as error:
            emit(
                "ERROR",
                "solver.production.output_drain_failed",
                "Production output drain failed",
                diagnostic=str(error),
                error_type=type(error).__name__,
            )
            return "PRODUCTION_OUTPUT_DRAIN_FAILED"
        return None

    def _emit_iteration_start(self, state: ProductionRunState, /) -> None:
        if self.iteration_session is None or self.iteration_session.cursor != 0:
            return
        assert self.iteration_scope is not None
        zero = jnp.zeros((), dtype=state.time.dtype)
        self.iteration_session.emit(
            self.iteration_scope,
            _production_iteration_record(
                IterationPhase.START,
                state.step_index,
                ProductionIterationMetrics(
                    state.time,
                    zero,
                    0,
                    True,
                    True,
                    False,
                    False,
                ),
            ),
        )

    def _emit_iteration_terminal(
        self,
        state: ProductionRunState,
        failure: ProductionFailureRecord | None,
        /,
    ) -> None:
        if self._terminal_iteration_emitted:
            return
        if self.iteration_session is None:
            return
        assert self.iteration_scope is not None
        zero = jnp.zeros((), dtype=state.time.dtype)
        status = 0 if state.status == "completed" and failure is None else 1
        self.iteration_session.emit(
            self.iteration_scope,
            _production_iteration_record(
                IterationPhase.TERMINAL,
                state.step_index,
                ProductionIterationMetrics(
                    state.time,
                    zero,
                    0,
                    failure is None,
                    state.status == "completed",
                    False,
                    False,
                ),
                committed=state.status == "completed",
                terminal=True,
                status=status,
            ),
        )
        self._terminal_iteration_emitted = True

    def _process_segment(
        self,
        source: ProductionRunState,
        final_segment: _SegmentState,
        records: _SegmentRecord,
        /,
    ) -> tuple[ProductionRunState, ProductionFailureRecord | None]:
        attempted = np.asarray(records.attempted, dtype=np.bool_)
        accepted = np.asarray(records.accepted, dtype=np.bool_)
        method_successful = np.asarray(records.method_successful, dtype=np.bool_)
        output_due = np.asarray(records.output_due, dtype=np.bool_)
        checkpoint_due = np.asarray(records.checkpoint_due, dtype=np.bool_)
        trigger_fires = tuple(
            np.asarray(value, dtype=np.bool_) for value in records.trigger_fires
        )
        last_checkpoint = source.last_checkpoint_id
        event_cursor = int(np.asarray(source.output_cursor))
        for index in np.flatnonzero(attempted):
            snapshot_segment = self._index_tree(records.state, int(index))
            snapshot = self._production_state(
                snapshot_segment, source, "running", last_checkpoint
            )
            if self.plan.device_resident:
                snapshot = self._place_production_state_like(snapshot, source)
            transition = self._index_tree(records.result, int(index))
            host_stop = False
            if self.iteration_session is not None:
                assert self.iteration_scope is not None
                host_stop = self.iteration_session.emit(
                    self.iteration_scope,
                    _production_iteration_record(
                        (
                            IterationPhase.COMMIT
                            if accepted[index]
                            else IterationPhase.ATTEMPT
                        ),
                        snapshot.step_index,
                        ProductionIterationMetrics(
                            snapshot.time,
                            transition.accepted_step_size,
                            transition.retry_count,
                            method_successful[index],
                            accepted[index],
                            output_due[index],
                            checkpoint_due[index],
                        ),
                        committed=accepted[index],
                        status=0 if accepted[index] else 1,
                    ),
                )
            if not accepted[index]:
                category = (
                    "state-invalid" if method_successful[index] else "step-rejected"
                )
                error_code = (
                    "PRODUCTION_STATE_INVALID"
                    if method_successful[index]
                    else "PRODUCTION_STEP_REJECTED"
                )
                failed = _replace_run_metadata(snapshot, status="failed")
                return failed, ProductionFailureRecord(
                    failed.step_index,
                    failed.time,
                    category,
                    error_code,
                    last_checkpoint,
                )
            if output_due[index]:
                schedule = self.plan.output_schedule
                if schedule is None:
                    raise RuntimeError(
                        "Output-due state has no bound production schedule."
                    )
                before = int(np.asarray(records.schedule_cursor_before)[index])
                after = int(np.asarray(snapshot.schedule_cursor))
                for cursor in range(before, after):
                    event_id = canonical_fingerprint(
                        {
                            "kind": "scheduled-production-output",
                            "run": self.run_id,
                            "schedule": schedule.schedule_id,
                            "cursor": cursor,
                        }
                    )
                    detail = self._publish(
                        event_id, event_cursor, snapshot.accepted_state
                    )
                    if detail is None:
                        event_cursor += 1
                    if detail is not None:
                        failed = _replace_run_metadata(snapshot, status="failed")
                        failed = _replace_output_cursor(failed, event_cursor)
                        return failed, ProductionFailureRecord(
                            failed.step_index,
                            failed.time,
                            "output-failed",
                            detail,
                            last_checkpoint,
                        )
            trigger_checkpoint = False
            for binding_index, binding in enumerate(self.plan.trigger_bindings):
                if not trigger_fires[binding_index][index]:
                    continue
                if binding.action == "checkpoint":
                    trigger_checkpoint = True
                elif binding.action == "publish":
                    fire_count = int(
                        np.asarray(snapshot.trigger_states[binding_index].fire_count)
                    )
                    event_id = canonical_fingerprint(
                        {
                            "kind": "triggered-production-output",
                            "run": self.run_id,
                            "binding": binding.binding_id,
                            "action": binding.action_id,
                            "fire_count": fire_count,
                        }
                    )
                    detail = self._publish(
                        event_id, event_cursor, snapshot.accepted_state
                    )
                    if detail is None:
                        event_cursor += 1
                    if detail is not None:
                        failed = _replace_run_metadata(snapshot, status="failed")
                        failed = _replace_output_cursor(failed, event_cursor)
                        return failed, ProductionFailureRecord(
                            failed.step_index,
                            failed.time,
                            "output-failed",
                            detail,
                            last_checkpoint,
                        )
            if event_cursor != int(np.asarray(snapshot.output_cursor)):
                raise RuntimeError(
                    "Production output cursor diverged from ordered output events."
                )
            tolerance = 32.0 * np.finfo(np.asarray(snapshot.time).dtype).eps
            terminal_status: RunStatus | None = None
            if host_stop or bool(np.asarray(snapshot_segment.stop_requested)):
                terminal_status = "canceled"
            elif float(np.asarray(snapshot.time)) >= self.plan.end_time - tolerance:
                terminal_status = "completed"
            elif int(np.asarray(snapshot.step_index)) >= self.plan.maximum_steps:
                terminal_status = "failed"
            if terminal_status is not None:
                snapshot = _replace_run_metadata(snapshot, status=terminal_status)
                self._emit_iteration_terminal(snapshot, None)
            if checkpoint_due[index] or trigger_checkpoint:
                if isinstance(self.checkpoint_store, ArtifactCheckpointStore):
                    snapshot = self.checkpoint(snapshot)
                    last_checkpoint = snapshot.last_checkpoint_id
                    detail = self._drain_outputs()
                else:
                    detail = self._drain_outputs()
                    if detail is None:
                        snapshot = self.checkpoint(snapshot)
                        last_checkpoint = snapshot.last_checkpoint_id
                if detail is not None:
                    failed = _replace_run_metadata(snapshot, status="failed")
                    return failed, ProductionFailureRecord(
                        failed.step_index,
                        failed.time,
                        "output-failed",
                        detail,
                        last_checkpoint,
                    )
            if host_stop:
                return _replace_run_metadata(snapshot, status="canceled"), None
        status: RunStatus
        tolerance = 32.0 * np.finfo(np.asarray(final_segment.time).dtype).eps
        if not bool(np.asarray(final_segment.running)):
            status = "failed"
        elif bool(np.asarray(final_segment.stop_requested)):
            status = "canceled"
        elif float(np.asarray(final_segment.time)) >= self.plan.end_time - tolerance:
            status = "completed"
        elif int(np.asarray(final_segment.step_index)) >= self.plan.maximum_steps:
            status = "failed"
        else:
            status = "running"
        current = self._production_state(final_segment, source, status, last_checkpoint)
        if status == "failed":
            return current, ProductionFailureRecord(
                current.step_index,
                current.time,
                "step-capacity-exhausted",
                "PRODUCTION_STEP_CAPACITY_EXHAUSTED",
                last_checkpoint,
            )
        return current, None

    def _execute(
        self, state: ProductionRunState, *, one_step: bool
    ) -> tuple[ProductionRunState, ProductionFailureRecord | None, _SegmentRecord]:
        if state.status not in ("ready", "running"):
            raise ValueError("Only ready or running production state can advance.")
        self._emit_iteration_start(state)
        executor = self._compiled_one_step if one_step else self._compiled_segment
        final_segment, records = executor(self._segment_initial(state))
        if self.plan.device_resident:
            processed_segment, processed_records = final_segment, records
        else:
            processed_segment, processed_records = jax.device_get(
                (final_segment, records)
            )
        current, failure = self._process_segment(
            state, processed_segment, processed_records
        )
        if self.plan.device_resident:
            current = self._place_production_state_like(current, state)
        return current, failure, processed_records

    def step(
        self, state: ProductionRunState, /
    ) -> tuple[ProductionRunState, RetriedFixedStepResult]:
        current, _failure, records = self._execute(state, one_step=True)
        attempted = np.flatnonzero(np.asarray(records.attempted, dtype=np.bool_))
        if attempted.size != 1:
            raise ValueError("Production state has already reached its horizon.")
        transition = self._index_tree(records.result, int(attempted[0]))
        if self.plan.device_resident:
            transition = RetriedFixedStepResult(
                candidate_state=self._place_tree_like(
                    transition.candidate_state, state.accepted_state
                ),
                accepted_state=self._place_tree_like(
                    transition.accepted_state, state.accepted_state
                ),
                successful=transition.successful,
                accepted_step_size=transition.accepted_step_size,
                retry_count=transition.retry_count,
                attempted_step_sizes=transition.attempted_step_sizes,
                decision_id=transition.decision_id,
            )
        return current, transition

    def run(self, state: ProductionRunState, /) -> ProductionRunResult:
        initial_output_cursor = int(np.asarray(state.output_cursor))
        acknowledged_before = (
            0 if self.publisher is None else len(self.publisher.acknowledged_event_ids)
        )
        current = state
        failure = None
        while current.status in ("ready", "running"):
            current, failure, _records = self._execute(current, one_step=False)
            if failure is not None or current.status in (
                "completed",
                "failed",
                "canceled",
            ):
                break
        publication_failed = failure is not None and failure.category == "output-failed"
        if isinstance(self.checkpoint_store, ArtifactCheckpointStore):
            if not publication_failed:
                current = self.checkpoint(current)
                drain_detail = self._drain_outputs()
            else:
                drain_detail = None
        else:
            drain_detail = self._drain_outputs()
        if drain_detail is not None:
            current = _replace_run_metadata(current, status="failed")
            if not isinstance(self.checkpoint_store, ArtifactCheckpointStore):
                acknowledged = initial_output_cursor + (
                    0
                    if self.publisher is None
                    else len(self.publisher.acknowledged_event_ids) - acknowledged_before
                )
                current = _replace_output_cursor(current, acknowledged)
            failure = ProductionFailureRecord(
                current.step_index,
                current.time,
                "output-failed",
                drain_detail,
                current.last_checkpoint_id,
            )
            publication_failed = True
        if publication_failed and self._last_checkpoint_receipt is None:
            current = self.checkpoint(current)
            if failure is None:
                raise RuntimeError(
                    "Publication failure checkpoint has no failure record."
                )
            failure = ProductionFailureRecord(
                failure.step_index,
                failure.time,
                failure.category,
                failure.error_code,
                current.last_checkpoint_id,
            )
        self._emit_iteration_terminal(current, failure)
        if not publication_failed and not isinstance(
            self.checkpoint_store, ArtifactCheckpointStore
        ):
            current = self.checkpoint(current)
        self._commit_terminal(current, failure)
        return ProductionRunResult(
            current,
            jnp.asarray(failure is None),
            failure,
            self.run_id,
            (
                None
                if self.iteration_session is None
                else self.iteration_session.snapshot()
            ),
        )


__all__ = [
    "ArtifactCheckpointStore",
    "CheckpointCommitReceipt",
    "CheckpointGenerationPolicy",
    "ProductionCaseManifest",
    "ProductionFailureRecord",
    "ProductionRunPlan",
    "ProductionRunResult",
    "ProductionIterationMetrics",
    "ProductionRunState",
    "ProductionTerminalManifest",
    "ProductionTriggerAction",
    "ProductionTriggerBinding",
    "DurableCheckpointStore",
    "PreparedProductionRun",
]
