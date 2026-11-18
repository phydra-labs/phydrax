from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._meshcore import meshcore_available, meshcore_identity


def runtime_identity() -> dict[str, object]:
    """Canonical build, interpreter, backend, and precision identity for one run."""
    meshcore = meshcore_identity() if meshcore_available() else "unavailable"
    build_id = canonical_fingerprint(
        {
            "kind": "meshing-qualification-build",
            "phydrax": importlib.metadata.version("phydrax"),
            "phydrax_path": str(Path(phx.__file__).resolve().parent),
            "meshcore": meshcore,
        }
    )
    environment_id = canonical_fingerprint(
        {
            "kind": "meshing-qualification-environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
    )
    identity = phx.qualification.QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return {**identity.to_record(), "meshcore": meshcore}


def require_meshcore() -> str:
    if not meshcore_available():
        raise RuntimeError(
            "Native qualification requires the installed, ABI-matched phydrax-meshcore wheel."
        )
    return meshcore_identity()


def single_cell_kind(mesh: phx.discretization.CellMesh, family: str, /) -> str:
    """Return the one published cell kind of ``family`` (e.g. ``simplex:3`` tetrahedra)."""
    kinds = {block.cell_kind for block in mesh.blocks if block.cell_count}
    if len(kinds) != 1:
        raise RuntimeError(
            f"Expected one published {family} cell kind; got {sorted(kinds)}."
        )
    kind = kinds.pop()
    if phx.discretization.reference_cell_topology(kind).family != family:
        raise RuntimeError(f"Published cell kind {kind!r} is not in the {family} family.")
    return kind


def accepted_result_record(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    result.audit.require_passed()
    if result.certification is None:
        raise RuntimeError("Native result omitted computed scientific certification.")
    result.certification.require_passed()
    if not result.compliance.passed:
        raise RuntimeError("Native result failed its requested meshing compliance.")
    if result.provider.name != "phydrax-native":
        raise RuntimeError(f"Unexpected meshing backend: {result.provider.name!r}.")
    unspecified: dict[str, int] = {}
    construction_calls: dict[str, int] = {}
    construction_unknown: dict[str, int] = {}
    reported = False
    surface_fidelity_counts: dict[str, int] = {}
    surface_fidelity_unknown = 0
    surface_fidelity_reported = False
    transfer_rows: dict[str, int] = {}
    transfer_calls: dict[str, int] = {}
    transfer_unknown: dict[str, int] = {}
    transfer_reported = False
    for stage in result.trace.stages:
        for diagnostic in stage.diagnostics:
            code = diagnostic.provider_code
            if code == "stored_coordinate_unspecified_transfer":
                kind, rows, calls = diagnostic.logical_findings
                if kind != "none":
                    transfer_rows[kind] = transfer_rows.get(kind, 0) + int(rows)
                    transfer_calls[kind] = transfer_calls.get(kind, 0) + int(calls)
                transfer_reported = True
                continue
            if code == "stored_coordinate_unspecified_transfer_unknown":
                reason, count = diagnostic.logical_findings
                if reason != "none":
                    transfer_unknown[reason] = transfer_unknown.get(reason, 0) + int(
                        count
                    )
                transfer_reported = True
                continue
            if code not in (
                "stored_coordinate_unspecified_construction",
                "stored_coordinate_unspecified_construction_calls",
                "stored_coordinate_unspecified_construction_unknown",
                "stored_coordinate_unspecified_surface_fidelity",
                "stored_coordinate_unspecified_surface_fidelity_unknown",
            ):
                continue
            amounts = dict(diagnostic.achieved_quantities)
            if "count" not in amounts:
                raise RuntimeError(
                    "Stored-coordinate declaration lacks its evaluation count."
                )
            count = amounts["count"]
            if count < 0 or not count.is_integer():
                raise RuntimeError(
                    "Stored-coordinate evaluation count must be a nonnegative integer."
                )
            if code == "stored_coordinate_unspecified_surface_fidelity_unknown":
                surface_fidelity_unknown += int(count)
                surface_fidelity_reported = True
                continue
            if len(diagnostic.logical_findings) != 1:
                raise RuntimeError(
                    "Stored-coordinate declaration lacks an evaluator kind."
                )
            kind = diagnostic.logical_findings[0]
            if code == "stored_coordinate_unspecified_construction_calls":
                reported = True
                target = construction_calls
            elif code == "stored_coordinate_unspecified_construction_unknown":
                reported = True
                target = construction_unknown
            elif code == "stored_coordinate_unspecified_construction":
                reported = True
                target = unspecified
            else:
                surface_fidelity_reported = True
                target = surface_fidelity_counts
            if kind != "none":
                target[kind] = target.get(kind, 0) + int(count)
    certificate_counts: dict[str, int] = {}
    unknown_certificates = 0
    seen_certificates: set[int] = set()
    certificates = (
        result.certification.fidelity,
        *result.certification.scoped_fidelity,
    )
    for certificate in certificates:
        if certificate is None or id(certificate) in seen_certificates:
            continue
        seen_certificates.add(id(certificate))
        counts = certificate.unspecified_evaluations
        if counts is None:
            unknown_certificates += 1
            continue
        for kind, count in counts:
            if kind.startswith("unknown:"):
                unknown_certificates += count
            else:
                certificate_counts[kind] = certificate_counts.get(kind, 0) + count
    return {
        **(
            {
                "transfer_unspecified_evaluations": {
                    "rows": dict(sorted(transfer_rows.items())),
                    "calls": dict(sorted(transfer_calls.items())),
                    "unknown": dict(sorted(transfer_unknown.items())),
                }
            }
            if transfer_reported
            else {}
        ),
        **(
            {
                "construction_unspecified_evaluations": dict(sorted(unspecified.items())),
                "construction_unspecified_evaluation_calls": dict(
                    sorted(construction_calls.items())
                ),
                "construction_unspecified_evidence_unknown": sum(
                    construction_unknown.values()
                ),
                "construction_unspecified_unknown_reasons": dict(
                    sorted(construction_unknown.items())
                ),
            }
            if reported
            else {}
        ),
        **(
            {
                "source_surface_fidelity_unspecified_evaluations": dict(
                    sorted(surface_fidelity_counts.items())
                ),
                "source_surface_fidelity_unspecified_evidence_unknown": surface_fidelity_unknown,
            }
            if surface_fidelity_reported
            else {}
        ),
        "fidelity_unspecified_evaluations": dict(sorted(certificate_counts.items())),
        "fidelity_unspecified_evidence_unknown": unknown_certificates,
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "provider": result.provider.name,
        "provider_id": result.provider.provider_id,
        "actual_version": result.runtime.actual_version,
        "runtime_id": result.runtime.runtime_id,
        "vertices": int(result.mesh.coordinates.shape[0]),
        "cells": sum(block.cell_count for block in result.mesh.blocks),
        "audit_passed": result.audit.passed,
        "certification_id": result.certification.report_id,
        "certification_passed": result.certification.passed,
        "compliance_passed": result.compliance.passed,
    }
