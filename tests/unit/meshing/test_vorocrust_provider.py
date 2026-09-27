import json
import os
import shutil
import sys
from typing import Any

import manifold3d
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing.providers._vorocrust import _polyhedra


_DIGEST = "a" * 64


def _cube_surface() -> Any:
    arrays = manifold3d.Manifold.cube().to_mesh64()
    return phx.geometry.SurfaceModel.from_triangles(
        arrays.vert_properties[:, :3],
        arrays.tri_verts,
        phx.geometry.SurfaceMetadata(
            source_id="cube",
            source_revision="0",
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            provenance=("qualification",),
        ),
    )


def _fake_worker(tmp_path: Any, body: Any) -> Any:
    """A protocol-level worker script speaking the persistent worker protocol."""
    worker = tmp_path / "worker"
    worker.write_text(
        f"#!{sys.executable}\nimport json, sys\nPREFIX = '@phydrax-worker '\n{body}",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return worker


def _hello_worker(tmp_path: Any, identity: Any) -> Any:
    return _fake_worker(
        tmp_path,
        f"identity = json.loads({json.dumps(json.dumps(identity))})\n"
        "hello = {'hello': {'identity': identity, 'memory_enforcement': "
        "'peak-rss-audit', 'ranks': 1}, 'ok': True}\n"
        "sys.stdout.write(PREFIX + json.dumps(hello) + '\\n')\n"
        "sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['operation'] == 'close':\n"
        "        break\n",
    )


@pytest.mark.meshing_vorocrust
def test_real_vorocrust_preserves_closed_cube_volume() -> None:
    executable = shutil.which(os.environ.get("PHYDRAX_VOROCRUST_EXECUTABLE", "vc_mesh"))
    worker = shutil.which(
        os.environ.get("PHYDRAX_VOROCRUST_WORKER", "phydrax-vorocrust-worker")
    )
    if executable is None or worker is None:
        pytest.skip(
            "Set the VoroCrust mesher and extraction worker for real qualification."
        )
    with phx.meshing.VoroCrustProvider(executable, worker) as provider:
        info = provider.info()
        result = provider.execute(
            _cube_surface(),
            phx.meshing.VoroCrustOptions(1.0),
            limits=phx.meshing.MeshingLimits(maximum_wall_seconds=120.0),
        )
        assert provider.worker.launches == 1
    assert result.provider.version == info.version
    assert all(
        isinstance(block, phx.discretization.PolyhedralBlock)
        for block in result.mesh.blocks
    )
    assert float(np.sum(np.asarray(result.quality.evaluation.measures))) == pytest.approx(
        1.0, rel=1e-6
    )
    assert result.audit.passed
    assert result.compliance.passed
    assert "native_output_entities_preallocation" in result.runtime.unenforced_limits
    assert "native_output_connectivity_preallocation" in result.runtime.unenforced_limits
    assert result.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE


def test_vorocrust_contracts() -> None:
    coordinates = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype="float64",
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    )
    base = phx.meshing.certify_cell_mesh(
        # ty: ignore[invalid-argument-type]
        phx.discretization.CellMesh.from_polyhedra(coordinates, cells),
        phx.SpatialCoordinateContract.si(),
    )
    provider = phx.meshing.MeshingProviderInfo(
        "vorocrust",
        "qualification-fixture",
        "BSD-3-Clause",
        operations=(phx.meshing.MeshingOperation.MESH_VOLUME,),
        source_kinds=(phx.meshing.MeshingSourceKind.SURFACE,),
        capabilities=(phx.meshing.MeshingCapability.POLYHEDRAL,),
        cell_kinds=("polyhedron",),
        dimensions=(3,),
        execution_modes=(phx.meshing.MeshingExecutionMode.SUBPROCESS,),
    )
    runtime = phx.meshing.MeshingRuntimeInfo(
        provider.provider_id,
        "qualification-fixture",
        phx.meshing.MeshingExecutionMode.SUBPROCESS,
        deterministic=False,
    )
    result = phx.meshing.CellMeshingResult(
        base.mesh,
        base.geometry,
        base.coordinate_contract,
        base.audit,
        base.quality,
        base.compliance,
        base.trace,
        provider,
        runtime,
        phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE,
        base.provenance,
    )
    geospatial = phx.interchange.GeospatialContract.local_cartesian(
        result.coordinate_contract,
        vertical_datum="local-survey-datum",
    )
    # ty: ignore[unresolved-attribute]
    boundary = np.flatnonzero(np.asarray(result.mesh.connectivity.boundary_faces))
    qualified = phx.applications.porous_media.qualify_vorocrust_porous_mesh(
        result,
        geospatial,
        surface_faces=boundary,
    )
    assert qualified.discretization.geometry_id
    assert qualified.require_surface_trace().parent_faces.size == 6
    assert not qualified.generator_dual_available
    assert not qualified.tpfa_certified
    with pytest.raises(ValueError, match="does not certify TPFA"):
        qualified.require_tpfa()
    mesh, merged, collapsed = _polyhedra(
        _two_box_extraction(), phx.meshing.MeshingLimits(), 1e-12
    )
    certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    assert merged == 2
    assert collapsed == 1
    np.testing.assert_array_equal(mesh.vertex_global_ids, np.arange(12))
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks]), (0, 1)
    )
    np.testing.assert_allclose(
        np.asarray(certified.quality.evaluation.measures), (0.5, 0.5), rtol=1e-12
    )
    assert certified.audit.passed
    # Consecutive chain members lie within the tolerance of each other, but
    # the chain end is farther than the tolerance from its representative.
    step = 0.6 * 1e-9 * np.sqrt(3.0)
    arrays = _two_box_extraction(((step, 0.0, 0.0), (2 * step, 0.0, 0.0)))

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        _polyhedra(arrays, phx.meshing.MeshingLimits(), 1e-9)

    assert failure.value.category is phx.meshing.MeshingFailureCategory.CONVERSION_FAILED


def test_vorocrust_worker_startup_exit_is_an_execution_failure(tmp_path: Any) -> None:
    worker = _fake_worker(tmp_path, "sys.exit(7)\n")
    provider = phx.meshing.VoroCrustProvider(sys.executable, worker)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.info()

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
    )


def test_vorocrust_worker_identity_is_reported_once_per_session(tmp_path: Any) -> None:
    identity = {
        "provider": "vorocrust",
        "revision": "r1",
        "source_sha256": _DIGEST,
        "config_sha256": _DIGEST,
        "library_sha256": _DIGEST,
        "openmp": False,
        "operations": ["extract"],
    }
    with phx.meshing.VoroCrustProvider(
        sys.executable, _hello_worker(tmp_path, identity)
    ) as provider:
        assert provider.info().version == "r1"
        assert provider.info().version == "r1"
        assert provider.worker.launches == 1


def test_vorocrust_worker_identity_requires_binary_identities(tmp_path: Any) -> None:
    identity = {
        "provider": "vorocrust",
        "revision": "r1",
        "source_sha256": _DIGEST,
        "config_sha256": "not-a-digest",
        "library_sha256": _DIGEST,
        "operations": ["extract"],
    }
    with phx.meshing.VoroCrustProvider(
        sys.executable, _hello_worker(tmp_path, identity)
    ) as provider:
        with pytest.raises(phx.meshing.MeshingFailure) as failure:
            provider.info()

    assert (
        failure.value.category is phx.meshing.MeshingFailureCategory.PROVIDER_UNAVAILABLE
    )


def test_vorocrust_worker_bounds_control_output_before_decoding(tmp_path: Any) -> None:
    worker = _fake_worker(
        tmp_path,
        "sys.stdout.write(PREFIX + 'x' * (4 * 1024 * 1024))\n"
        "sys.stdout.flush()\n"
        "sys.stdin.read()\n",
    )
    provider = phx.meshing.VoroCrustProvider(sys.executable, worker)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.info()

    assert failure.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_vorocrust_fails_closed_when_native_preallocation_is_required(
    tmp_path: Any,
) -> None:
    worker = _fake_worker(tmp_path, "sys.exit(0)\n")

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.VoroCrustProvider(sys.executable, worker).execute(
            _cube_surface(),
            phx.meshing.VoroCrustOptions(
                1.0,
                require_native_output_preallocation=True,
            ),
        )

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    )


def _two_box_extraction(extra_vertices: Any = ()) -> Any:
    """Packed extraction output of two unit-height boxes split at x = 0.5.

    Vertex 12 aliases 7 and vertex 13 aliases 3 within the merge tolerance.
    Loop orientations are deliberately mixed, one face uses an alias, and one
    face degenerates to two vertices once aliases are normalized.
    """
    grid = np.stack(
        np.meshgrid((0.0, 0.5, 1.0), (0.0, 1.0), (0.0, 1.0), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    vertices = np.concatenate(
        (
            grid,
            grid[[7, 3]] + (1e-14, -1e-14, 1e-14),
            np.asarray(extra_vertices, dtype=np.float64).reshape(-1, 3),
        )
    )
    faces = (
        ((0, 2, 3, 1), (2, 0)),
        ((0, 4, 5, 1), (0, 2)),
        ((3, 7, 6, 2), (2, 0)),
        ((0, 4, 6, 2), (0, 2)),
        ((1, 5, 7, 3), (0, 2)),
        ((4, 6, 7, 5), (0, 1)),
        ((8, 10, 11, 9), (1, 2)),
        ((4, 8, 9, 5), (2, 1)),
        ((6, 10, 11, 12), (1, 2)),
        ((6, 4, 8, 10), (2, 1)),
        ((5, 9, 11, 7), (1, 2)),
        ((3, 13, 7), (0, 2)),
    )
    loops = [np.asarray(loop, dtype=np.int64) for loop, _ in faces]
    return {
        "vertices": vertices,
        "face_offsets": np.concatenate(([0], np.cumsum([loop.size for loop in loops]))),
        "face_vertices": np.concatenate(loops),
        "face_seeds": np.asarray([pair for _, pair in faces], dtype=np.int64),
        "seed_points": np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (-1.0, 0.5, 0.5))),
        "seed_regions": np.asarray((1, 1, 0), dtype=np.int64),
    }
