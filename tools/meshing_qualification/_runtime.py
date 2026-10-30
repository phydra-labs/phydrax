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


def accepted_result_record(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    result.audit.require_passed()
    if result.certification is None:
        raise RuntimeError("Native result omitted computed scientific certification.")
    result.certification.require_passed()
    if not result.compliance.passed:
        raise RuntimeError("Native result failed its requested meshing compliance.")
    if result.provider.name != "phydrax-native":
        raise RuntimeError(f"Unexpected meshing backend: {result.provider.name!r}.")
    return {
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
