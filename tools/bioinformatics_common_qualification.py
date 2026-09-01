#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared deterministic provenance and CLI support for bioinformatics qualification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._runtime import capture_environment


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_CLAIM_TAXONOMY = {
    "exact_model": "Exact objective or model under its declared assumptions.",
    "approximate_model": "Scientifically approximate model with explicit conditioning.",
    "relaxed_objective": "Continuous or surrogate relaxation of a discrete objective.",
    "heuristic": "Procedure without an exact or approximate-model optimality claim.",
    "learned": "Data-fitted behavior whose validity depends on training provenance.",
}


def _canonical(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Fingerprint payloads must contain only finite floats.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tolist"):
        array = np.asarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": _canonical(array.tolist()),
        }
    if isinstance(value, Sequence):
        return [_canonical(item) for item in value]
    if hasattr(value, "value"):
        return _canonical(value.value)
    raise TypeError(f"Unsupported fingerprint value {type(value).__name__}.")


def fingerprint(value: Any, /) -> str:
    """Hash one canonical finite JSON value."""

    payload = json.dumps(
        _canonical(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def method_contract_evidence(contract: Any, /) -> dict[str, Any]:
    """Serialize the public scientific/numerical method contract."""

    payload = {
        "absolute_tolerance": float(contract.absolute_tolerance),
        "assumptions": list(contract.assumptions),
        "capacity_semantics": contract.capacity_semantics,
        "compute_dtype": contract.compute_dtype,
        "conditioning_statement": contract.conditioning_statement,
        "differentiation_kind": contract.differentiation_kind.value,
        "execution_kind": contract.execution_kind.value,
        "input_dtype": contract.input_dtype,
        "method_kind": contract.method_kind.value,
        "method_name": contract.method_name,
        "nondifferentiable_outputs": list(contract.nondifferentiable_outputs),
        "output_dtype": contract.output_dtype,
        "output_kind": contract.output_kind.value,
        "relative_tolerance": float(contract.relative_tolerance),
        "truncation_statement": contract.truncation_statement,
    }
    return {
        "contract_id": contract.contract_id,
        "fingerprint": fingerprint(payload),
        **payload,
    }


def _file_sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(root: Path, /) -> tuple[str, str, int]:
    """Hash a file by bytes or a directory by canonical relative paths and file hashes."""

    if root.is_symlink():
        raise ValueError("External dataset roots may not be symbolic links.")
    if root.is_file():
        return _file_sha256(root), "sha256-file-bytes", 1
    if not root.is_dir():
        raise ValueError("External dataset root must be a regular file or directory.")

    files = sorted(
        (path for path in root.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("External dataset directory contains no regular files.")
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError(
                f"External dataset contains symbolic link {path.relative_to(root)}."
            )
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(_file_sha256(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest(), "sha256-relative-path-and-file-digest", len(files)


def external_dataset_campaign(
    name: str,
    root: Path | None,
    expected_sha256: str | None,
    /,
) -> dict[str, Any]:
    """Verify an explicitly supplied external campaign root without fetching data."""

    method = {
        "claim": "input_identity_only",
        "digest_algorithms": [
            "sha256-file-bytes",
            "sha256-relative-path-and-file-digest",
        ],
        "network_access": "none",
    }
    method_fingerprint = fingerprint(method)
    if root is None and expected_sha256 is None:
        return {
            "campaign": name,
            "scope": "external_campaign",
            "status": "not_requested",
            "passed": None,
            "method": method,
            "method_fingerprint": method_fingerprint,
        }
    if root is None or expected_sha256 is None:
        return {
            "campaign": name,
            "scope": "external_campaign",
            "status": "path_and_digest_required",
            "passed": False,
            "method": method,
            "method_fingerprint": method_fingerprint,
        }

    expected = str(expected_sha256).lower()
    if _SHA256.fullmatch(expected) is None:
        return {
            "campaign": name,
            "scope": "external_campaign",
            "root": str(root),
            "status": "invalid_expected_sha256",
            "passed": False,
            "method": method,
            "method_fingerprint": method_fingerprint,
        }
    try:
        resolved = root.expanduser().resolve(strict=True)
        observed, digest_method, file_count = dataset_digest(resolved)
    except (OSError, ValueError) as error:
        return {
            "campaign": name,
            "scope": "external_campaign",
            "root": str(root),
            "expected_sha256": expected,
            "status": "unreadable_input",
            "error": str(error),
            "passed": False,
            "method": method,
            "method_fingerprint": method_fingerprint,
        }
    matches = observed == expected
    return {
        "campaign": name,
        "scope": "external_campaign",
        "root": str(resolved),
        "file_count": file_count,
        "digest_method": digest_method,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "input_fingerprint": observed,
        "qualification_claim": "input_identity_verified",
        "status": "verified" if matches else "digest_mismatch",
        "passed": matches,
        "method": method,
        "method_fingerprint": method_fingerprint,
    }


def qualification_report(
    domain: str,
    cases: Mapping[str, Mapping[str, Any]],
    /,
    *,
    external_campaigns: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble unit and optional external evidence with comparison fingerprints."""

    unit_cases = dict(cases)
    campaigns = {} if external_campaigns is None else dict(external_campaigns)
    unit_passed = all(bool(case.get("passed")) for case in unit_cases.values())
    requested = [
        campaign
        for campaign in campaigns.values()
        if campaign.get("status") != "not_requested"
    ]
    external_passed = all(bool(campaign.get("passed")) for campaign in requested)
    inputs = {name: case.get("input_fingerprint") for name, case in unit_cases.items()}
    methods = {name: case.get("method_fingerprint") for name, case in unit_cases.items()}
    return {
        "domain": domain,
        "environment": capture_environment().to_dict(),
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": fingerprint(methods),
        "method_claim_taxonomy": METHOD_CLAIM_TAXONOMY,
        "execution_boundaries": {
            "host_interchange": (
                "Paths, digests, text identity, and capacity planning stay outside "
                "compiled kernels."
            ),
            "jax_kernel": (
                "Array-valued scientific evidence is produced by public JAX APIs."
            ),
        },
        "unit_qualification": {
            "scope": "unit_qualification",
            "cases": unit_cases,
            "passed": unit_passed,
        },
        "external_campaigns": campaigns,
        "external_campaigns_requested": len(requested),
        "passed": unit_passed and external_passed,
    }


def _json_output(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_output(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_output(value.item())
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        return _json_output(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def emit_report(report: Mapping[str, Any], output: Path | None, /) -> int:
    """Print canonical JSON, optionally persist it, and map evidence to an exit code."""

    payload = (
        json.dumps(_json_output(report), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if bool(report.get("passed")) else 1


__all__ = [
    "METHOD_CLAIM_TAXONOMY",
    "dataset_digest",
    "emit_report",
    "external_dataset_campaign",
    "fingerprint",
    "method_contract_evidence",
    "qualification_report",
]
