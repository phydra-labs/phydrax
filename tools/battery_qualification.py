#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Offline, content-addressed scientific qualification for battery candidates.

The CLI accepts only explicit local records. It performs no discovery, fetches,
or release mutation. A successful output is therefore a candidate scientific
evidence bundle, not a release decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import phydrax
from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.artifacts import ArtifactManifest
from phydrax.lifecycle import ResolvedRunSpec
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    CapabilityProfile,
    QualificationCriterion,
    QualificationEvidence,
    ReferenceArtifactManifest,
    SupportTuple,
    validate_qualification_causality,
)
from tools.battery_campaign_registry import (
    get_campaign_entry,
    metric_key,
    prepare_builtin_campaign,
    PreparedCampaign,
)


_RIGHT_FIELDS = frozenset(
    {
        "commercial_use",
        "redistribution",
        "training_use",
        "export",
    }
)
_ARTIFACT_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "artifact_id",
        "producer",
        "version",
        "sha256",
        "byte_size",
        "source_uri",
        "license_id",
        "model",
        "coverage",
        "manifest_id",
    }
)
_REFERENCE_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "artifact_name",
        "checksum_algorithm",
        "checksum",
        "size_bytes",
        "license_id",
        "commercial_use_permitted",
        "redistribution_permitted",
        "training_use_permitted",
        "export_permitted",
        "export_classification",
        "nondimensionalization",
        "uncertainty",
        "lineage_ids",
        "manifest_id",
    }
)
_SUPPORT_FIELDS = frozenset({"kind", "capability", "attributes", "support_tuple_id"})
_PROFILE_FIELDS = frozenset(
    {
        "kind",
        "name",
        "provider",
        "version",
        "support_tuples",
        "dependencies",
        "required_gates",
        "release_evidence",
        "released",
        "profile_id",
    }
)
_RELEASE_EVIDENCE_FIELDS = frozenset(
    {
        "kind",
        "gate",
        "passed",
        "evidence_ids",
        "reviewer_id",
        "deviation_ids",
        "issued_at",
        "expires_at",
        "evidence_id",
    }
)
_DEPENDENCY_FIELDS = frozenset(
    {"kind", "profile_id", "support_tuple_id", "dependency_id"}
)
_CRITERION_FIELDS = frozenset(
    {
        "kind",
        "support_tuple_id",
        "metric",
        "unit",
        "comparison",
        "target",
        "aggregation",
        "uncertainty",
        "applicability",
        "approval_id",
        "issued_at",
        "valid_until",
        "criterion_id",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "kind",
        "sample_times_s",
        "not_before",
        "deadline",
        "evidence_validity_duration",
        "schedule_id",
    }
)
_PAYLOAD_FIELDS = frozenset({"path", "manifest", "required_rights"})
_CAMPAIGN_FIELDS = frozenset(
    {
        "kind",
        "campaign_kind",
        "candidate_profile",
        "registry_entry_id",
        "candidate_support",
        "resolved_run_spec",
        "criteria_set_id",
        "build_id",
        "environment_id",
        "backend_id",
        "device_id",
        "precision_id",
        "topology_id",
        "discretization_id",
        "parameter_id",
        "replay_id",
        "payloads",
        "split_id",
        "preprocessing_id",
        "noise_model_id",
        "model_selection_id",
        "rng_id",
        "reviewer_id",
        "planned_schedule",
        "campaign_spec_id",
    }
)
_CRITERIA_SET_FIELDS = frozenset({"kind", "criteria", "criteria_set_id"})
_PERFORMANCE_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "JAX_COMPILATION_CACHE_DIR",
)
_RUNTIME_DISTRIBUTIONS = (
    "coordax",
    "diffrax",
    "equinox",
    "jax",
    "jaxlib",
    "jaxtyping",
    "lineax",
    "numpy",
    "optimistix",
)


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _timestamp(value: object, name: str, /) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return value


def _strict_bool(value: object, name: str, /) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean.")
    return value


def _mapping(value: object, name: str, /) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} field names must be strings.")
    return value


def _exact_fields(
    value: object, expected: frozenset[str], name: str, /
) -> Mapping[str, object]:
    record = _mapping(value, name)
    missing = sorted(expected - set(record))
    unknown = sorted(set(record) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{name} has {'; '.join(details)}.")
    return record


def _sequence(value: object, name: str, /) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _finite_number(value: object, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return 0.0 if result == 0.0 else result


def _reject_nonfinite(value: str, /) -> None:
    raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")


def _unique_object(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate field {key!r}.")
        result[key] = value
    return result


def read_json_object(path: Path, /) -> Mapping[str, object]:
    """Read one strict finite JSON object, rejecting duplicate fields."""
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
        object_pairs_hook=_unique_object,
    )
    return _mapping(value, f"JSON document {path}")


def _validate_support_record(record: object, /) -> None:
    value = _exact_fields(record, _SUPPORT_FIELDS, "Candidate support record")
    if value["kind"] != "capability-support-tuple":
        raise ValueError("Candidate support has an unsupported kind.")
    _identifier(value["support_tuple_id"], "candidate support-tuple ID")


def _validate_profile_record(record: object, /) -> None:
    value = _exact_fields(record, _PROFILE_FIELDS, "Candidate profile record")
    if value["kind"] != "capability-profile":
        raise ValueError("Candidate profile has an unsupported kind.")
    _identifier(value["profile_id"], "candidate profile ID")
    _strict_bool(value["released"], "candidate released state")
    for support in _sequence(value["support_tuples"], "candidate support tuples"):
        _validate_support_record(support)
    for dependency in _sequence(value["dependencies"], "candidate dependencies"):
        if isinstance(dependency, Mapping):
            dependency_record = _exact_fields(
                dependency, _DEPENDENCY_FIELDS, "Candidate dependency"
            )
            if dependency_record["kind"] != "exact-support-dependency":
                raise ValueError("Candidate dependency has an unsupported kind.")
            _identifier(dependency_record["dependency_id"], "dependency ID")
        else:
            _identifier(dependency, "dependency profile ID")
    _sequence(value["required_gates"], "candidate required gates")
    for evidence in _sequence(value["release_evidence"], "candidate release evidence"):
        evidence_record = _exact_fields(
            evidence, _RELEASE_EVIDENCE_FIELDS, "Candidate release evidence"
        )
        if evidence_record["kind"] != "release-gate-evidence":
            raise ValueError("Candidate release evidence has an unsupported kind.")
        _identifier(evidence_record["evidence_id"], "release evidence ID")
        _strict_bool(evidence_record["passed"], "release evidence passed state")


def _artifact_manifest_record(manifest: ArtifactManifest, /) -> dict[str, object]:
    return {
        "kind": "artifact-manifest",
        "artifact_id": manifest.artifact_id,
        "producer": manifest.producer,
        "version": manifest.version,
        "sha256": manifest.sha256,
        "byte_size": manifest.byte_size,
        "source_uri": manifest.source_uri,
        "license_id": manifest.license_id,
        "model": manifest.model,
        "coverage": manifest.coverage,
        "manifest_id": manifest.manifest_id,
    }


def _artifact_manifest_from_record(record: object, /) -> ArtifactManifest:
    value = _exact_fields(record, _ARTIFACT_MANIFEST_FIELDS, "Artifact manifest")
    if value["kind"] != "artifact-manifest":
        raise ValueError("Artifact manifest has an unsupported kind.")
    if type(value["byte_size"]) is not int:
        raise TypeError("Artifact manifest byte_size must be an integer.")
    manifest = ArtifactManifest(
        artifact_id=_identifier(value["artifact_id"], "artifact ID"),
        producer=_identifier(value["producer"], "artifact producer"),
        version=_identifier(value["version"], "artifact version"),
        sha256=_identifier(value["sha256"], "artifact sha256"),
        byte_size=value["byte_size"],
        source_uri=_identifier(value["source_uri"], "artifact source URI"),
        license_id=_identifier(value["license_id"], "artifact license ID"),
        model=_identifier(value["model"], "artifact model"),
        coverage=_identifier(value["coverage"], "artifact coverage"),
    )
    if _identifier(value["manifest_id"], "artifact manifest ID") != manifest.manifest_id:
        raise ValueError("Artifact manifest has an invalid content address.")
    return manifest


def _reference_manifest_from_record(record: object, /) -> ReferenceArtifactManifest:
    value = _exact_fields(
        record, _REFERENCE_MANIFEST_FIELDS, "Reference-artifact manifest"
    )
    if value["kind"] != "reference-artifact-manifest":
        raise ValueError("Reference-artifact manifest has an unsupported kind.")
    for field in (
        "commercial_use_permitted",
        "redistribution_permitted",
        "training_use_permitted",
        "export_permitted",
    ):
        _strict_bool(value[field], field)
    _identifier(value["manifest_id"], "reference manifest ID")
    return ReferenceArtifactManifest.from_record(value)


@dataclass(frozen=True, slots=True)
class PayloadBinding:
    """One local payload, exact generic manifest, and explicit use request."""

    path: str
    manifest: ArtifactManifest | ReferenceArtifactManifest
    required_rights: tuple[tuple[str, bool], ...]

    @property
    def manifest_id(self) -> str:
        return self.manifest.manifest_id

    def to_record(self) -> dict[str, object]:
        manifest_record = (
            _artifact_manifest_record(self.manifest)
            if isinstance(self.manifest, ArtifactManifest)
            else self.manifest.to_record()
        )
        return {
            "path": self.path,
            "manifest": manifest_record,
            "required_rights": dict(self.required_rights),
        }

    @classmethod
    def from_record(cls, record: object, /) -> PayloadBinding:
        value = _exact_fields(record, _PAYLOAD_FIELDS, "Payload binding")
        path = _identifier(value["path"], "local payload path")
        if "\x00" in path or "://" in path:
            raise ValueError("Payload paths must be local filesystem paths, not URIs.")
        rights_record = _exact_fields(
            value["required_rights"], _RIGHT_FIELDS, "Payload required rights"
        )
        rights = tuple(
            sorted(
                (name, _strict_bool(rights_record[name], f"required right {name}"))
                for name in _RIGHT_FIELDS
            )
        )
        manifest_record = _mapping(value["manifest"], "Payload manifest")
        kind = manifest_record.get("kind")
        if kind == "artifact-manifest":
            manifest: ArtifactManifest | ReferenceArtifactManifest = (
                _artifact_manifest_from_record(manifest_record)
            )
            if any(required for _, required in rights):
                raise PermissionError(
                    "ArtifactManifest does not declare requested-use grants; "
                    "all required rights must be false."
                )
            if manifest.source_uri != path:
                raise ValueError(
                    "Artifact manifest source_uri must exactly equal its local payload path."
                )
        elif kind == "reference-artifact-manifest":
            manifest = _reference_manifest_from_record(manifest_record)
            manifest.require_rights(**dict(rights))
        else:
            raise ValueError("Payload manifest kind is unsupported.")
        return cls(path, manifest, rights)

    def local_path(self, base_directory: Path, /) -> Path:
        declared = Path(self.path)
        return declared if declared.is_absolute() else base_directory / declared

    def verify_local(self, base_directory: Path, /) -> str:
        """Verify rights, size, and checksum without resolving a remote source."""
        if isinstance(self.manifest, ReferenceArtifactManifest):
            self.manifest.require_rights(**dict(self.required_rights))
            expected_size = self.manifest.size_bytes
            algorithm = self.manifest.checksum_algorithm
            expected_digest = self.manifest.checksum
        else:
            if any(required for _, required in self.required_rights):
                raise PermissionError(
                    "ArtifactManifest does not carry affirmative use-right grants."
                )
            expected_size = self.manifest.byte_size
            algorithm = "sha256"
            expected_digest = self.manifest.sha256
        local_path = self.local_path(base_directory)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local payload does not exist: {local_path}")
        if local_path.stat().st_size != expected_size:
            raise ValueError(f"Local payload size does not match {self.manifest_id}.")
        digest = hashlib.new(algorithm)
        with local_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"Local payload checksum does not match {self.manifest_id}.")
        return self.manifest_id


def execution_source_build_id(root: Path, /) -> str:
    """Fingerprint the complete live Python source and lockfile closure."""
    if not isinstance(root, Path):
        raise TypeError("Execution source root must be a pathlib.Path.")
    source_root = root.resolve()
    package_root = source_root / "phydrax"
    if not source_root.is_dir() or not package_root.is_dir():
        raise FileNotFoundError(
            "Execution source root must contain the phydrax package directory."
        )
    required = (
        source_root / "tools/battery_qualification.py",
        source_root / "tools/battery_campaign_registry.py",
        source_root / "tools/_battery_ecm_campaign.py",
        source_root / "tools/battery_campaign_resources.py",
        source_root / "benchmarks/battery_performance.py",
        source_root / "benchmarks/_runtime.py",
        source_root / "benchmarks/_comparison.py",
        source_root / "pyproject.toml",
        source_root / "uv.lock",
    )
    python_sources = (*package_root.rglob("*.py"), *(source_root / "tools").rglob("*.py"))
    paths = tuple(
        sorted(set((*required, *python_sources)), key=lambda path: path.as_posix())
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            "Execution source closure requires the qualification tool, "
            "pyproject.toml, uv.lock, and every phydrax Python source."
        )
    sources: list[dict[str, object]] = []
    for path in paths:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if path.stat().st_size != size:
            raise RuntimeError(f"Execution source changed while hashing: {path}")
        sources.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return canonical_fingerprint(
        {
            "kind": "battery-live-execution-source-closure",
            "sources": sources,
        }
    )


def _python_tree_id(directory: Path, /) -> str:
    files = []
    for path in sorted(directory.rglob("*.py")):
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not files:
        raise ValueError("Imported Python package has no source files.")
    return canonical_fingerprint(files)


def verify_campaign_harness(source_root: Path, /) -> None:
    """The executing copied harness must equal the runner bytes in the frozen closure."""
    if source_root.resolve() == _PROJECT_ROOT:
        return
    if _python_tree_id(_PROJECT_ROOT / "tools") != _python_tree_id(source_root / "tools"):
        raise ValueError("Executing tools differ from the frozen source harness.")
    for name in ("battery_performance.py", "_runtime.py", "_comparison.py"):
        if (_PROJECT_ROOT / "benchmarks" / name).read_bytes() != (
            source_root / "benchmarks" / name
        ).read_bytes():
            raise ValueError(
                "Executing benchmark differs from the frozen source harness."
            )


@dataclass(frozen=True, slots=True)
class CampaignSchedule:
    """Finite sampling window and evidence-lifetime plan."""

    sample_times_s: tuple[float, ...]
    not_before: int
    deadline: int
    evidence_validity_duration: int
    schedule_id: str

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "battery-campaign-schedule",
            "sample_times_s": list(self.sample_times_s),
            "not_before": self.not_before,
            "deadline": self.deadline,
            "evidence_validity_duration": self.evidence_validity_duration,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "schedule_id": self.schedule_id}

    @classmethod
    def from_record(cls, record: object, /) -> CampaignSchedule:
        value = _exact_fields(record, _SCHEDULE_FIELDS, "Planned schedule")
        if value["kind"] != "battery-campaign-schedule":
            raise ValueError("Planned schedule has an unsupported kind.")
        times = tuple(
            _finite_number(item, "planned sample time")
            for item in _sequence(value["sample_times_s"], "planned sample times")
        )
        if not 2 <= len(times) <= 4096:
            raise ValueError(
                "Planned sample times must contain between 2 and 4096 nodes."
            )
        if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError(
                "Planned sample times must start at zero and be strictly increasing."
            )
        not_before = _timestamp(value["not_before"], "planned not_before")
        deadline = _timestamp(value["deadline"], "planned deadline")
        duration = _timestamp(
            value["evidence_validity_duration"],
            "planned evidence_validity_duration",
        )
        if deadline <= not_before:
            raise ValueError("Planned campaign deadline must follow not_before.")
        if duration < 1:
            raise ValueError("Planned evidence validity duration must be positive.")
        temporary = cls(times, not_before, deadline, duration, "pending")
        expected = canonical_fingerprint(temporary._content_record())
        if _identifier(value["schedule_id"], "schedule ID") != expected:
            raise ValueError("Planned schedule has an invalid content address.")
        return cls(times, not_before, deadline, duration, expected)


@dataclass(frozen=True, slots=True)
class QualificationCriteriaSet:
    """A content-addressed, canonical set of generic criteria."""

    criteria: tuple[QualificationCriterion, ...]
    criteria_set_id: str

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "battery-qualification-criteria-set",
            "criteria": [criterion.to_record() for criterion in self.criteria],
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "criteria_set_id": self.criteria_set_id}

    @classmethod
    def from_record(cls, record: object, /) -> QualificationCriteriaSet:
        value = _exact_fields(record, _CRITERIA_SET_FIELDS, "Criteria set")
        if value["kind"] != "battery-qualification-criteria-set":
            raise ValueError("Criteria set has an unsupported kind.")
        criteria: list[QualificationCriterion] = []
        for item in _sequence(value["criteria"], "criteria"):
            criterion_record = _exact_fields(
                item, _CRITERION_FIELDS, "Qualification criterion"
            )
            if criterion_record["kind"] != "qualification-criterion":
                raise ValueError("Qualification criterion has an unsupported kind.")
            _identifier(criterion_record["criterion_id"], "criterion ID")
            criteria.append(QualificationCriterion.from_record(criterion_record))
        if not criteria:
            raise ValueError("Criteria set must contain at least one criterion.")
        criteria.sort(key=lambda criterion: criterion.criterion_id)
        if len({criterion.criterion_id for criterion in criteria}) != len(criteria):
            raise ValueError("Criteria set contains duplicate criteria.")
        temporary = cls(tuple(criteria), "pending")
        expected = canonical_fingerprint(temporary._content_record())
        if _identifier(value["criteria_set_id"], "criteria-set ID") != expected:
            raise ValueError("Criteria set has an invalid content address.")
        return cls(tuple(criteria), expected)


def _runtime_dependency_identity(
    root: Path,
    installed_versions: Mapping[str, str] | None,
    /,
) -> tuple[dict[str, str], str]:
    lock_path = root.resolve() / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("Runtime identity requires the repository uv.lock.")
    lock_payload = lock_path.read_bytes()
    lock_document = tomllib.loads(lock_payload.decode("utf-8"))
    package_records = lock_document.get("package")
    if not isinstance(package_records, list):
        raise ValueError("uv.lock must contain package records.")
    locked: dict[str, str] = {}
    required = set(_RUNTIME_DISTRIBUTIONS)
    for package in package_records:
        if not isinstance(package, Mapping):
            raise TypeError("uv.lock package entries must be mappings.")
        name = package.get("name")
        if name not in required:
            continue
        version = package.get("version")
        if type(version) is not str or not version:
            raise ValueError(f"uv.lock package {name!r} has no exact version.")
        if name in locked:
            raise ValueError(f"uv.lock contains duplicate runtime package {name!r}.")
        locked[name] = version
    missing_lock = sorted(required - set(locked))
    if missing_lock:
        raise ValueError(
            "uv.lock is missing runtime dependencies: " + ", ".join(missing_lock)
        )
    observed: dict[str, str] = {}
    for name in _RUNTIME_DISTRIBUTIONS:
        if installed_versions is None:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(
                    f"Required runtime distribution {name!r} is not installed."
                ) from error
        else:
            if name not in installed_versions:
                raise ValueError(
                    f"Required runtime distribution {name!r} is not installed."
                )
            version = installed_versions[name]
        if type(version) is not str or not version:
            raise ValueError(
                f"Installed runtime distribution {name!r} has no exact version."
            )
        if version != locked[name]:
            raise ValueError(
                f"Installed runtime distribution {name!r} version {version!r} "
                f"does not match uv.lock version {locked[name]!r}."
            )
        observed[name] = version
    return observed, hashlib.sha256(lock_payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Observed software, backend, device, precision, and topology identity."""

    environment_id: str
    backend_id: str
    device_id: str
    precision_id: str
    topology_id: str
    environment_record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "environment_id": self.environment_id,
            "backend_id": self.backend_id,
            "device_id": self.device_id,
            "precision_id": self.precision_id,
            "topology_id": self.topology_id,
            "environment": dict(self.environment_record),
        }


def capture_runtime_identity(
    *,
    source_root: Path | None = None,
    distribution_manifest: Path | None = None,
    installed_versions: Mapping[str, str] | None = None,
) -> RuntimeIdentity:
    """Capture and lock-verify comparison-relevant runtime identity."""
    root = _PROJECT_ROOT if source_root is None else source_root
    runtime_dependencies, lock_sha256 = _runtime_dependency_identity(
        root, installed_versions
    )
    imported_package = Path(phydrax.__file__).resolve().parent
    distribution_identity = None
    if distribution_manifest is not None:
        from tools.battery_distribution import verify_installed_distribution

        distribution_identity = verify_installed_distribution(distribution_manifest, root)
    devices = tuple(
        sorted(
            (
                {
                    "id": int(device.id),
                    "process_index": int(device.process_index),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                }
                for device in jax.devices()
            ),
            key=lambda value: (
                value["process_index"],
                value["platform"],
                value["id"],
                value["device_kind"],
            ),
        )
    )
    backend_id = jax.default_backend()
    precision_id = str(jnp.asarray(0.0).dtype)
    device_content = {"kind": "jax-device-set", "devices": list(devices)}
    device_id = canonical_fingerprint(device_content)
    topology_content = {
        "kind": "jax-runtime-topology",
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "local_device_count": jax.local_device_count(),
        "device_id": device_id,
    }
    topology_id = canonical_fingerprint(topology_content)
    environment_record: dict[str, object] = {
        "kind": "battery-runtime-environment",
        "python_version": platform.python_version(),
        "phydrax_version": importlib.metadata.version("phydrax"),
        "phydrax_import_path": str(imported_package),
        "phydrax_python_tree_id": _python_tree_id(imported_package),
        "execution_mode": "development"
        if distribution_identity is None
        else "clean-installed-distribution",
        "distribution_identity": distribution_identity,
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "backend_id": backend_id,
        "device_id": device_id,
        "precision_id": precision_id,
        "topology_id": topology_id,
        "runtime_dependencies": runtime_dependencies,
        "lock_sha256": lock_sha256,
        "performance_environment": {
            key: os.environ.get(key) for key in _PERFORMANCE_ENVIRONMENT_KEYS
        },
    }
    return RuntimeIdentity(
        canonical_fingerprint(environment_record),
        backend_id,
        device_id,
        precision_id,
        topology_id,
        environment_record,
    )


@dataclass(frozen=True, slots=True)
class BatteryCampaignSpec:
    campaign_kind: str
    candidate_profile: CapabilityProfile
    candidate_support: SupportTuple
    resolved_run_spec: ResolvedRunSpec
    criteria_set_id: str
    build_id: str
    environment_id: str
    backend_id: str
    device_id: str
    precision_id: str
    topology_id: str
    discretization_id: str
    parameter_id: str
    replay_id: str
    payloads: tuple[PayloadBinding, ...]
    split_id: str
    preprocessing_id: str
    noise_model_id: str
    model_selection_id: str
    rng_id: str
    reviewer_id: str
    planned_schedule: CampaignSchedule
    campaign_spec_id: str

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "battery-qualification-campaign",
            "campaign_kind": self.campaign_kind,
            "registry_entry_id": get_campaign_entry(self.campaign_kind).entry_id,
            "candidate_profile": self.candidate_profile.to_record(),
            "candidate_support": self.candidate_support.to_record(),
            "resolved_run_spec": self.resolved_run_spec.to_record(),
            "criteria_set_id": self.criteria_set_id,
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend_id": self.backend_id,
            "device_id": self.device_id,
            "precision_id": self.precision_id,
            "topology_id": self.topology_id,
            "discretization_id": self.discretization_id,
            "parameter_id": self.parameter_id,
            "replay_id": self.replay_id,
            "payloads": [payload.to_record() for payload in self.payloads],
            "split_id": self.split_id,
            "preprocessing_id": self.preprocessing_id,
            "noise_model_id": self.noise_model_id,
            "model_selection_id": self.model_selection_id,
            "rng_id": self.rng_id,
            "reviewer_id": self.reviewer_id,
            "planned_schedule": self.planned_schedule.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "campaign_spec_id": self.campaign_spec_id}

    def expected_replay_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "battery-campaign-replay",
                "campaign_kind": self.campaign_kind,
                "registry_entry_id": get_campaign_entry(self.campaign_kind).entry_id,
                "candidate_profile_id": self.candidate_profile.profile_id,
                "candidate_support_tuple_id": self.candidate_support.support_tuple_id,
                "resolved_run_spec_id": self.resolved_run_spec.spec_id,
                "criteria_set_id": self.criteria_set_id,
                "build_id": self.build_id,
                "environment_id": self.environment_id,
                "backend_id": self.backend_id,
                "device_id": self.device_id,
                "precision_id": self.precision_id,
                "topology_id": self.topology_id,
                "discretization_id": self.discretization_id,
                "parameter_id": self.parameter_id,
                "payload_manifest_ids": sorted(
                    payload.manifest_id for payload in self.payloads
                ),
                "split_id": self.split_id,
                "preprocessing_id": self.preprocessing_id,
                "noise_model_id": self.noise_model_id,
                "model_selection_id": self.model_selection_id,
                "rng_id": self.rng_id,
                "schedule_id": self.planned_schedule.schedule_id,
            }
        )

    def workload_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "battery-performance-workload",
                "campaign_kind": self.campaign_kind,
                "registry_entry_id": get_campaign_entry(self.campaign_kind).entry_id,
                "candidate_profile_id": self.candidate_profile.profile_id,
                "candidate_support_tuple_id": self.candidate_support.support_tuple_id,
                "backend_id": self.backend_id,
                "device_id": self.device_id,
                "precision_id": self.precision_id,
                "topology_id": self.topology_id,
                "discretization_id": self.discretization_id,
                "parameter_id": self.parameter_id,
                "model_selection_id": self.model_selection_id,
                "sample_times_s": list(self.planned_schedule.sample_times_s),
            }
        )

    @classmethod
    def from_record(cls, record: object, /) -> BatteryCampaignSpec:
        value = _exact_fields(record, _CAMPAIGN_FIELDS, "Battery campaign spec")
        if value["kind"] != "battery-qualification-campaign":
            raise ValueError("Battery campaign spec has an unsupported kind.")
        kind = _identifier(value["campaign_kind"], "campaign kind")
        entry = get_campaign_entry(kind)
        if value["registry_entry_id"] != entry.entry_id:
            raise ValueError(
                "Campaign registry entry identity does not match the live executable."
            )
        _validate_profile_record(value["candidate_profile"])
        _validate_support_record(value["candidate_support"])
        profile = CapabilityProfile.from_record(
            _mapping(value["candidate_profile"], "candidate profile")
        )
        support = SupportTuple.from_record(
            _mapping(value["candidate_support"], "candidate support")
        )
        if (
            support.to_record() != entry.candidate_support.to_record()
            or profile.to_record() != entry.candidate_profile.to_record()
        ):
            raise ValueError(
                "Campaign requires its exact registered candidate and support."
            )
        run_spec = ResolvedRunSpec.from_record(
            _mapping(value["resolved_run_spec"], "resolved run spec")
        )
        matching_dependencies = tuple(
            dependency
            for dependency in (
                run_spec.scientific_dependencies + run_spec.deployment_dependencies
            )
            if dependency.profile_id == profile.profile_id
        )
        if (
            len(matching_dependencies) != 1
            or matching_dependencies[0].support_tuple_id != support.support_tuple_id
        ):
            raise ValueError(
                "ResolvedRunSpec must bind the exact candidate profile and support once."
            )
        payloads = tuple(
            PayloadBinding.from_record(item)
            for item in _sequence(value["payloads"], "campaign payloads")
        )
        if len({payload.path for payload in payloads}) != len(payloads):
            raise ValueError("Battery campaign payload paths must be unique.")
        if len({payload.manifest_id for payload in payloads}) != len(payloads):
            raise ValueError("Battery campaign payload manifest IDs must be unique.")
        schedule = CampaignSchedule.from_record(value["planned_schedule"])
        identifiers = {
            name: _identifier(value[name], name.replace("_", " "))
            for name in (
                "criteria_set_id",
                "build_id",
                "environment_id",
                "backend_id",
                "device_id",
                "precision_id",
                "topology_id",
                "discretization_id",
                "parameter_id",
                "replay_id",
                "split_id",
                "preprocessing_id",
                "noise_model_id",
                "model_selection_id",
                "rng_id",
                "reviewer_id",
            )
        }
        result = cls(
            kind,
            profile,
            support,
            run_spec,
            identifiers["criteria_set_id"],
            identifiers["build_id"],
            identifiers["environment_id"],
            identifiers["backend_id"],
            identifiers["device_id"],
            identifiers["precision_id"],
            identifiers["topology_id"],
            identifiers["discretization_id"],
            identifiers["parameter_id"],
            identifiers["replay_id"],
            payloads,
            identifiers["split_id"],
            identifiers["preprocessing_id"],
            identifiers["noise_model_id"],
            identifiers["model_selection_id"],
            identifiers["rng_id"],
            identifiers["reviewer_id"],
            schedule,
            _identifier(value["campaign_spec_id"], "campaign-spec ID"),
        )
        if result.replay_id != result.expected_replay_id():
            raise ValueError(
                "Battery campaign replay ID does not match its exact inputs."
            )
        expected_spec_id = canonical_fingerprint(result._content_record())
        if result.campaign_spec_id != expected_spec_id:
            raise ValueError("Battery campaign spec has an invalid content address.")
        return result


def load_campaign_spec(path: Path, /) -> BatteryCampaignSpec:
    return BatteryCampaignSpec.from_record(read_json_object(path))


def load_criteria_set(path: Path, /) -> QualificationCriteriaSet:
    return QualificationCriteriaSet.from_record(read_json_object(path))


def verify_campaign_preflight(
    spec: BatteryCampaignSpec,
    base_directory: Path,
    /,
    *,
    source_root: Path | None = None,
    distribution_manifest: Path | None = None,
) -> PreparedCampaign:
    """Verify every identity, right, and local byte before scientific dispatch."""
    root = _PROJECT_ROOT if source_root is None else source_root
    verify_campaign_harness(root)
    if (
        Path(phydrax.__file__).resolve().parent != (root / "phydrax").resolve()
        and distribution_manifest is None
    ):
        raise ValueError(
            "Clean installed execution requires an explicit distribution manifest."
        )
    entry = get_campaign_entry(spec.campaign_kind)
    references = tuple(
        payload
        for payload in spec.payloads
        if isinstance(payload.manifest, ReferenceArtifactManifest)
    )
    if len(references) != entry.reference_count or len(spec.payloads) != len(references):
        raise ValueError(
            "Campaign payloads must be exactly its registered reference inputs."
        )
    for reference in references:
        requested = dict(reference.required_rights)
        if any(not requested[right] for right in entry.required_rights):
            raise PermissionError(
                "Reference binding omits a required campaign use right."
            )
        reference.manifest.require_rights(
            **{right: True for right in entry.required_rights}
        )
    runtime = capture_runtime_identity(
        source_root=root, distribution_manifest=distribution_manifest
    )
    observed_identity = {
        "environment_id": runtime.environment_id,
        "backend_id": runtime.backend_id,
        "device_id": runtime.device_id,
        "precision_id": runtime.precision_id,
        "topology_id": runtime.topology_id,
    }
    declared_identity = {
        "environment_id": spec.environment_id,
        "backend_id": spec.backend_id,
        "device_id": spec.device_id,
        "precision_id": spec.precision_id,
        "topology_id": spec.topology_id,
    }
    if declared_identity != observed_identity:
        raise ValueError("Campaign runtime identity does not match the observed runtime.")
    manifest_ids = tuple(
        payload.verify_local(base_directory) for payload in spec.payloads
    )
    expected_build_id = execution_source_build_id(
        _PROJECT_ROOT if source_root is None else source_root
    )
    if spec.build_id != expected_build_id:
        raise ValueError(
            "Campaign build_id does not identify the verified execution-source closure."
        )
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Verified payload manifest identities are not unique.")
    builtin = prepare_builtin_campaign(
        spec.campaign_kind, spec.planned_schedule.sample_times_s
    )
    if spec.discretization_id != builtin.discretization_id:
        raise ValueError("Campaign discretization ID does not match the built-in case.")
    if spec.parameter_id != builtin.parameter_id:
        raise ValueError("Campaign parameter ID does not match the built-in case.")
    if spec.model_selection_id != builtin.model_selection_id:
        raise ValueError("Campaign model-selection ID does not match the built-in case.")
    expected_preparation = builtin.prepared_configuration_id(
        spec.planned_schedule.schedule_id
    )
    if spec.resolved_run_spec.prepared_configuration_id != expected_preparation:
        raise ValueError(
            "ResolvedRunSpec prepared configuration does not match the exact campaign."
        )
    if spec.replay_id != spec.expected_replay_id():
        raise ValueError("Campaign replay ID does not match the verified inputs.")
    return builtin


def _validate_criteria(
    criteria_set: QualificationCriteriaSet, spec: BatteryCampaignSpec, /
) -> None:
    if criteria_set.criteria_set_id != spec.criteria_set_id:
        raise ValueError("Campaign does not bind the supplied criteria set.")
    get_campaign_entry(spec.campaign_kind).validate_criteria(criteria_set.criteria)


def _validate_start_time(
    criteria_set: QualificationCriteriaSet,
    spec: BatteryCampaignSpec,
    started_at: int,
    /,
) -> None:
    schedule = spec.planned_schedule
    if not schedule.not_before <= started_at <= schedule.deadline:
        raise ValueError("Campaign start lies outside the planned schedule window.")
    if not (
        spec.resolved_run_spec.valid_from
        <= started_at
        <= spec.resolved_run_spec.valid_until
    ):
        raise ValueError("ResolvedRunSpec is not valid at campaign start.")
    for criterion in criteria_set.criteria:
        if not criterion.issued_at < started_at:
            raise ValueError("Qualification criterion is postdated at campaign start.")
        if not criterion.is_valid(started_at):
            raise ValueError("Qualification criterion is expired at campaign start.")


def _synchronize(value: object, /) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def run_qualification(
    spec: BatteryCampaignSpec,
    criteria_set: QualificationCriteriaSet,
    /,
    *,
    campaign_directory: Path,
    artifact_directory: Path | None = None,
    source_root: Path | None = None,
    distribution_manifest: Path | None = None,
    utc_timestamp_source: Callable[[], int] | None = None,
    monotonic_timestamp_source: Callable[[], int] | None = None,
) -> dict[str, object]:
    """Persist causal starts before dispatch, and every scientific outcome afterward."""
    destination = artifact_directory or campaign_directory / "qualification-artifacts"
    execution_root = _PROJECT_ROOT if source_root is None else source_root
    utc_clock = time.time_ns if utc_timestamp_source is None else utc_timestamp_source
    monotonic_clock = (
        time.monotonic_ns
        if monotonic_timestamp_source is None
        else monotonic_timestamp_source
    )
    try:
        # Reconstruct typed records too: direct dataclass construction cannot bypass preflight.
        BatteryCampaignSpec.from_record(spec.to_record())
        QualificationCriteriaSet.from_record(criteria_set.to_record())
        _validate_criteria(criteria_set, spec)
        builtin = verify_campaign_preflight(
            spec,
            campaign_directory,
            source_root=execution_root,
            distribution_manifest=distribution_manifest,
        )
        started_tick = _timestamp(monotonic_clock(), "campaign start monotonic tick")
        started_at = _timestamp(utc_clock(), "campaign start UTC timestamp")
        _validate_start_time(criteria_set, spec, started_at)
    except Exception as error:
        refusal = campaign_attempt_record(error, campaign_spec_id=spec.campaign_spec_id)
        write_json_immutable(destination / f"{refusal['attempt_id']}.json", refusal)
        return refusal

    entry = get_campaign_entry(spec.campaign_kind)
    schedule = spec.planned_schedule
    starts = tuple(
        CampaignStartRecord(
            campaign_spec_id=spec.campaign_spec_id,
            criterion_id=criterion.criterion_id,
            resolved_run_spec_id=spec.resolved_run_spec.spec_id,
            support_tuple_id=spec.candidate_support.support_tuple_id,
            started_at=started_at,
        )
        for criterion in criteria_set.criteria
    )
    for start in starts:
        write_json_immutable(
            destination / f"{start.start_record_id}.json", start.to_record()
        )
    # No scientific operation precedes the durable start boundary.
    failures: list[dict[str, object]] = []
    payload = None
    observed_runtime = None
    try:
        executed = builtin.execute()
        _synchronize(executed)
        observed_payload = entry.raw_output(spec, executed, campaign_directory)
        entry.validate_raw(observed_payload)
        payload = observed_payload
        for reference in spec.payloads:
            reference.verify_local(campaign_directory)
        if execution_source_build_id(execution_root) != spec.build_id:
            raise RuntimeError("Execution source closure changed after campaign start.")
        verify_campaign_harness(execution_root)
        observed_runtime = capture_runtime_identity(
            source_root=execution_root, distribution_manifest=distribution_manifest
        )
        if observed_runtime.environment_id != spec.environment_id:
            raise RuntimeError(
                "Imported package or runtime identity changed after campaign start."
            )
    except Exception as error:
        failures.append(
            {
                "stage": "execution-or-reference",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        if payload is None:
            payload = {
                "kind": "battery-campaign-execution-failure",
                "metrics": {
                    key: {
                        "value": None,
                        "unavailable_reason": "post-start-execution-or-reference-failure",
                    }
                    for key in entry.metrics()
                },
            }

    def boundary(previous_utc: int, previous_tick: int, stage: str) -> tuple[int, int]:
        # On a clock failure retain the observed bytes and an explicitly identified
        # monotonic-anchored boundary, never issue passed evidence from that clock.
        fallback_start = time.monotonic_ns()
        try:
            tick = _timestamp(monotonic_clock(), f"{stage} monotonic tick")
            utc = _timestamp(utc_clock(), f"{stage} UTC timestamp")
            if tick < previous_tick or utc < previous_utc:
                raise ValueError("Campaign clock reversed.")
        except Exception as error:
            failures.append(
                {
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "boundary_clock": "previous-utc-plus-system-monotonic-elapsed",
                }
            )
            return previous_utc + time.monotonic_ns() - fallback_start, previous_tick
        if utc > schedule.deadline or utc > spec.resolved_run_spec.valid_until:
            failures.append(
                {
                    "stage": stage,
                    "type": "ExpiredExecutionWindow",
                    "message": "Campaign execution window expired.",
                }
            )
        return utc, tick

    observed_at, observed_tick = boundary(started_at, started_tick, "observation")
    issued_at, _ = boundary(observed_at, observed_tick, "evidence")
    raw_content = {
        **payload,
        "registry_entry_id": entry.entry_id,
        "campaign_spec_id": spec.campaign_spec_id,
        "infrastructure_failures": failures,
        "runtime_identity": None
        if observed_runtime is None
        else observed_runtime.to_record(),
    }
    raw_artifact_id = canonical_fingerprint(raw_content)
    raw_output = {**raw_content, "raw_artifact_id": raw_artifact_id}
    write_json_immutable(destination / f"{raw_artifact_id}.json", raw_output)
    observations = tuple(
        CampaignObservationRecord(
            start_record_id=start.start_record_id,
            campaign_spec_id=spec.campaign_spec_id,
            criterion_id=start.criterion_id,
            resolved_run_spec_id=spec.resolved_run_spec.spec_id,
            support_tuple_id=spec.candidate_support.support_tuple_id,
            raw_artifact_ids=(raw_artifact_id,),
            observed_at=observed_at,
        )
        for start in starts
    )
    records = []
    for criterion, start, observation in zip(
        criteria_set.criteria, starts, observations, strict=True
    ):
        measurement = payload["metrics"][
            metric_key(criterion.applicability, criterion.metric)
        ]
        outcome, reason = entry.classify(criterion, measurement)
        if failures:
            outcome, reason = "inconclusive", "post-start-infrastructure-failure"
        elif not criterion.is_valid(observed_at) or not criterion.is_valid(issued_at):
            outcome, reason = "inconclusive", "criterion-expired-after-start"
        expiry = min(issued_at + schedule.evidence_validity_duration, 2**63 - 1)
        if outcome == "passed":
            expiry = min(expiry, spec.resolved_run_spec.valid_until)
            if criterion.valid_until is not None:
                expiry = min(expiry, criterion.valid_until)
            if expiry <= issued_at:
                outcome, reason = "inconclusive", "no-current-evidence-lifetime"
                expiry = min(issued_at + schedule.evidence_validity_duration, 2**63 - 1)
        evidence = QualificationEvidence(
            entry.evidence_kind(criterion),
            outcome,
            (spec.candidate_profile.profile_id, spec.candidate_support.support_tuple_id),
            build_id=spec.build_id,
            environment_id=spec.environment_id,
            backend=spec.backend_id,
            topology=spec.topology_id,
            precision=spec.precision_id,
            reduction=f"{criterion.uncertainty}:{criterion.aggregation}",
            replay_id=spec.replay_id,
            criteria_ids=(criterion.criterion_id,),
            raw_artifact_ids=(raw_artifact_id,),
            campaign_start_record_ids=(start.start_record_id,),
            campaign_observation_record_ids=(observation.observation_record_id,),
            reviewer_id=spec.reviewer_id,
            issued_at=issued_at,
            expires_at=expiry,
            reason=reason,
            requalification_triggers=(
                "build-change",
                "environment-change",
                "support-change",
                "parameter-change",
            ),
        )
        validate_qualification_causality(criterion, start, observation, evidence)
        write_json_immutable(
            destination / f"{observation.observation_record_id}.json",
            observation.to_record(),
        )
        write_json_immutable(
            destination / f"{evidence.evidence_id}.json", evidence.to_record()
        )
        records.append(evidence.to_record())
    outcomes = {record["outcome"] for record in records}
    result = {
        "kind": "battery-qualification-result",
        "outcome": "failed"
        if "failed" in outcomes
        else "inconclusive"
        if "inconclusive" in outcomes
        else "passed",
        "campaign_spec": spec.to_record(),
        "criteria_set": criteria_set.to_record(),
        "registry_entry": entry.to_record(),
        "campaign_start_records": [start.to_record() for start in starts],
        "raw_output": raw_output,
        "campaign_observation_records": [
            observation.to_record() for observation in observations
        ],
        "evidence_records": records,
    }
    write_json_immutable(destination / f"{canonical_fingerprint(result)}.json", result)
    return result


def campaign_input_identity(path: Path, /) -> dict[str, object]:
    """Unreadable input bytes are themselves auditable preflight refusal evidence."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        return {
            "path": str(path),
            "sha256": None,
            "read_error": f"{type(error).__name__}:{error}",
        }
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "read_error": None,
    }


def campaign_attempt_record(error: Exception, /, **inputs: object) -> dict[str, object]:
    """A refusal is an audit attempt, not a scientific observation or failed case."""
    content = {
        "kind": "battery-campaign-attempt",
        "outcome": "preflight-refused",
        "attempted_at": time.time_ns(),
        "inputs": inputs,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    return {**content, "attempt_id": canonical_fingerprint(content)}


def write_json_immutable(path: Path, result: Mapping[str, object], /) -> None:
    """Publish durable immutable bytes; never replace a prior attempt or observation."""
    payload = (canonical_json(result) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise FileExistsError(
                    f"Immutable campaign record already exists: {path}"
                ) from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, result: Mapping[str, object], /) -> None:
    """Atomically replace a JSON file through an exclusive destination-local temp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an offline governed battery scientific campaign."
    )
    parser.add_argument("--campaign-spec", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--distribution-manifest", type=Path)
    arguments = parser.parse_args(argv)
    try:
        spec = load_campaign_spec(arguments.campaign_spec)
        criteria_set = load_criteria_set(arguments.criteria)
    except Exception as error:
        inputs = {}
        for name, path in (
            ("campaign_spec", arguments.campaign_spec),
            ("criteria", arguments.criteria),
        ):
            inputs[name] = campaign_input_identity(path)
        result = campaign_attempt_record(error, **inputs)
    else:
        result = run_qualification(
            spec,
            criteria_set,
            campaign_directory=arguments.campaign_spec.parent,
            source_root=arguments.source_root,
            distribution_manifest=arguments.distribution_manifest,
        )
    write_json_immutable(arguments.output, result)
    return {"passed": 0, "failed": 1, "preflight-refused": 2, "inconclusive": 3}[
        result["outcome"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BatteryCampaignSpec",
    "CampaignSchedule",
    "PayloadBinding",
    "QualificationCriteriaSet",
    "PreparedCampaign",
    "RuntimeIdentity",
    "capture_runtime_identity",
    "execution_source_build_id",
    "load_campaign_spec",
    "load_criteria_set",
    "main",
    "prepare_builtin_campaign",
    "read_json_object",
    "run_qualification",
    "verify_campaign_preflight",
    "write_json_atomic",
    "write_json_immutable",
]
