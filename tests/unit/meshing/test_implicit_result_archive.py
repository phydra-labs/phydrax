"""Native implicit meshing results archive and restore in independent processes."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import Sphere
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_archive,
    read_model_artifact,
    restore_model_archive_value,
    write_model_archive,
    write_model_artifact,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeImplicitSource


pytestmark = pytest.mark.meshcore

_REPOSITORY = Path(__file__).resolve().parents[3]
_POLICY = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.4, root_tolerance=1.0e-8)


def _generate(source: Any, maximum_deviation: float, /) -> CellMeshingResult:
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(),
        maximum_deviation=maximum_deviation,
    )
    return (
        NativeMeshingProvider(
            NativeMeshingOptions("implicit_surface", implicit_policy=_POLICY)
        )
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )


def _discovered_sphere() -> CellMeshingResult:
    """Adaptive discovery bound to its authoring profile."""
    source_id = "archive-implicit-sphere"
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 0.75, feature_id=source_id).compile(),
        SpatialCoordinateContract.si(),
        source_id=source_id,
        source_revision=f"{source_id}-r1",
    )
    bounds = np.asarray(profile.boundary_bounds, dtype=np.float64)
    padding = float(np.max(bounds[1] - bounds[0])) / 4.0
    discovery = discover_adaptive_implicit_surface(
        profile.geometry,
        domain=np.stack(
            (
                np.nextafter(bounds[0] - padding, -np.inf),
                np.nextafter(bounds[1] + padding, np.inf),
            )
        ),
        policy=_POLICY,
        source_id=profile.source_id,
    )
    source = NativeImplicitSource(
        profile.geometry,
        profile.source_id,
        profile.source_revision,
        discovery=discovery,
        authoring_profile=profile,
    )
    return _generate(source, 0.5)


def _fresh_process_result_id(path: Path, /) -> str:
    code = (
        "import sys\n"
        "from phydrax.meshing._model_codec import (\n"
        "    MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value)\n"
        "from phydrax.meshing._result import CellMeshingResult\n"
        "restored = read_model_artifact(sys.argv[1], expected_type=CellMeshingResult,\n"
        "    restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS)\n"
        "print(restored.result_id, restored.certification.report_id)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_REPOSITORY), *filter(None, (environment.get("PYTHONPATH"),)))
    )
    completed = subprocess.run(
        (sys.executable, "-c", code, str(path)),
        cwd=_REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1]


def test_native_implicit_result_round_trips_in_a_fresh_process(tmp_path: Path) -> None:
    original = _discovered_sphere()
    assert original.certification is not None
    assert type(original.certification.inputs.feature_source) is NativeImplicitSource
    path = write_model_artifact(
        tmp_path / "implicit.phx", original, limits=MODEL_ARCHIVE_LIMITS
    )
    assert _fresh_process_result_id(path) == (
        f"{original.result_id} {original.certification.report_id}"
    )


def test_analytic_profile_volume_restores_its_shared_profile_owner(
    tmp_path: Path,
) -> None:
    """The profile is domain, fidelity query and feature source at once."""
    from phydrax.meshing._contracts import VolumeFillStrategy, VolumeMeshingSpec
    from phydrax.meshing._volume_generation import NativeVolumeSchedule

    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 0.75, feature_id="archive-volume-sphere").compile(),
        contract,
        source_id="archive-volume-sphere",
        source_revision="archive-volume-sphere-r1",
    )
    original = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "implicit_restricted_delaunay", volume_schedule=NativeVolumeSchedule()
            )
        )
        .plan(
            profile,
            VolumeMeshingSpec(
                CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
                MeshingScope(
                    profile.source_id,
                    profile.source_revision,
                    MeshingEntityKind.GEOMETRY,
                    2,
                    profile.scope_entity_set_id(2),
                    profile.scope_entity_ids(2),
                ),
                VolumeFillStrategy.SIMPLEX,
                size_controls=(),
                maximum_deviation=0.8,
            ),
            coordinate_contract=contract,
        )
        .execute()
    )
    assert original.certification is not None
    inputs = original.certification.inputs
    assert inputs.domain is inputs.source
    path = write_model_artifact(
        tmp_path / "volume.phx", original, limits=MODEL_ARCHIVE_LIMITS
    )
    assert _fresh_process_result_id(path) == (
        f"{original.result_id} {original.certification.report_id}"
    )


def _refuses(path: Path, change: Callable[[dict[str, Any]], None], /) -> None:
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    found: list[dict[str, Any]] = []

    def visit(node: Any, /) -> None:
        if isinstance(node, dict):
            if str(node.get("type", "")).endswith(":NativeImplicitSourceArchiveRecipe"):
                found.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(manifest["model_recipe"])
    assert found
    for node in found:
        change(node)
    forged = write_model_archive(
        path.with_name("forged.phx"),
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            forged,
            expected_type=CellMeshingResult,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_native_implicit_result_refuses_changed_discovery_or_profile(
    tmp_path: Path,
) -> None:
    original = _discovered_sphere()
    path = write_model_artifact(
        tmp_path / "implicit.phx", original, limits=MODEL_ARCHIVE_LIMITS
    )

    def discovery(node: dict[str, Any], /) -> None:
        # Recipe items: geometry, grid, discovery, authoring profile, ...
        field = node["items"][2]["fields"]["mesh_source_id"]
        field["value"] = f"{field['value']}-forged"

    def profile(node: dict[str, Any], /) -> None:
        # Profile recipe items: geometry, contract, tube_radius, ...
        field = node["items"][3]["items"][2]
        field["value"] = float(field["value"]) * 2.0

    _refuses(path, discovery)
    _refuses(path, profile)
