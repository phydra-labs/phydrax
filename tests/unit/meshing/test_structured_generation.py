#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry._mesh_certificates import ParametricCurveBoundarySource
from phydrax.geometry.brep._patches import CircleCurve, LineCurve, PlanePatch
from phydrax.meshing._contracts import MeshingFailure
from phydrax.meshing._controls import TransfiniteCurveControl, TransfiniteSurfaceControl
from phydrax.meshing._structured import generate_structured_block, TransfiniteBlock


def _square(n: int, /) -> TransfiniteBlock:
    return TransfiniteBlock(
        "square",
        tuple(
            TransfiniteCurveControl(LineCurve(origin, direction), name, (0.0, 1.0), n)
            for name, origin, direction in (
                ("left", (0.0, 0.0), (0.0, 1.0)),
                ("right", (1.0, 0.0), (0.0, 1.0)),
                ("bottom", (0.0, 0.0), (1.0, 0.0)),
                ("top", (0.0, 1.0), (1.0, 0.0)),
            )
        ),
    )


def test_native_route_cannot_substitute_quads_for_required_triangles() -> None:
    block = _square(4)
    controls = tuple(
        control
        for control in block.boundaries
        if isinstance(control, TransfiniteCurveControl)
    )
    revision = canonical_fingerprint(
        {"curves": tuple(array_tree_fingerprint(c.curve) for c in controls)}
    )
    query = ParametricCurveBoundarySource(
        tuple(c.curve for c in controls),
        tuple(c.parameter_range for c in controls),
        source_id="square",
        source_revision=revision,
    )
    source = phx.meshing.NativeStructuredSource(
        (block,), (), "square", revision, fidelity_source=query, maximum_deviation=0.08
    )
    scope = phx.meshing.MeshingScope(
        "square",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "square-domain",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.25, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("structured_transfinite")
    )
    with pytest.raises(MeshingFailure, match="required_cell_families"):
        provider.plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )


def test_opposing_curve_count_mismatch_rejected() -> None:
    boundaries = list(_square(2).boundaries)
    boundaries[0] = TransfiniteCurveControl(
        LineCurve((0.0, 0.0), (0.0, 1.0)), "left", (0.0, 1.0), 3
    )
    with pytest.raises(ValueError, match="Opposing"):
        TransfiniteBlock("mismatch", tuple(boundaries))


def test_curve_orientation_is_not_inferred_from_coordinates() -> None:
    boundaries = list(_square(2).boundaries)
    boundaries[1] = TransfiniteCurveControl(
        LineCurve((1.0, 0.0), (0.0, 1.0)), "right", (0.0, 1.0), 2, reversed=True
    )
    with pytest.raises(ValueError, match="incompatible boundary charts"):
        generate_structured_block(TransfiniteBlock("reversed", tuple(boundaries)))


def test_curved_annulus_sector_exact_boundary_nodes_and_positive_quads() -> None:
    angle = np.pi / 2.0
    inner = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0)
    block = TransfiniteBlock(
        "annulus-sector",
        (
            TransfiniteCurveControl(inner, "inner", (0.0, angle), 12),
            TransfiniteCurveControl(outer, "outer", (0.0, angle), 12),
            TransfiniteCurveControl(
                LineCurve((1.0, 0.0), (1.0, 0.0)), "start", (0.0, 1.0), 3
            ),
            TransfiniteCurveControl(
                LineCurve((0.0, 1.0), (0.0, 1.0)), "end", (0.0, 1.0), 3
            ),
        ),
    )
    controls = tuple(
        control
        for control in block.boundaries
        if isinstance(control, TransfiniteCurveControl)
    )
    revision = canonical_fingerprint(
        {
            "source": "annulus-sector",
            "curves": tuple(
                (
                    control.source_id,
                    array_tree_fingerprint(control.curve),
                    control.parameter_range,
                )
                for control in controls
            ),
        }
    )
    query = ParametricCurveBoundarySource(
        tuple(control.curve for control in controls),
        tuple(control.parameter_range for control in controls),
        source_id="annulus-sector",
        source_revision=revision,
        covering_radius=0.01,
    )
    source = phx.meshing.NativeStructuredSource(
        (block,),
        (),
        "annulus-sector",
        revision,
        fidelity_source=query,
        maximum_deviation=0.08,
    )
    scope = phx.meshing.MeshingScope(
        "annulus-sector",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "annulus-domain",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("quadrilateral",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.4, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("structured_transfinite")
    )
    result = provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()
    points = np.asarray(result.mesh.coordinates).reshape(4, 13, 2)
    np.testing.assert_allclose(np.linalg.norm(points[0], axis=-1), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(points[-1], axis=-1), 2.0, atol=1e-13)
    assert result.mesh.blocks[0].cell_kind == "quadrilateral"
    assert result.mesh.blocks[0].cell_count == 36
    assert np.all(np.asarray(result.audit.validity.determinant_lower) > 0.0)
    assert result.certification is not None
    assert result.certification.passed
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"


def test_six_oriented_source_faces_generate_pure_positive_hexes() -> None:
    faces = (
        PlanePatch((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        PlanePatch((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        PlanePatch((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        PlanePatch((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    oriented = [
        TransfiniteSurfaceControl(
            face,
            f"face:{i}",
            np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            (2, 2),
        )
        for i, face in enumerate(faces)
    ]
    oriented[0] = TransfiniteSurfaceControl(
        PlanePatch((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "face:0",
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        (2, 2),
        permutation=(1, 0),
    )
    oriented[1] = TransfiniteSurfaceControl(
        PlanePatch((1.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        "face:1",
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        (2, 2),
        flips=(True, False),
    )
    controls = tuple(oriented)
    result = generate_structured_block(TransfiniteBlock("cube", controls))
    assert result.mesh.blocks[0].cell_kind == "hexahedron"
    assert result.mesh.blocks[0].cell_count == 8
    np.testing.assert_allclose(
        np.asarray(result.mesh.coordinates).reshape(3, 3, 3, 3)[1, 1, 1],
        (0.5, 0.5, 0.5),
        atol=1e-14,
    )
    assert result.embedding.status == "certified"


def test_elliptic_optimization_preserves_declared_boundary() -> None:
    initial = generate_structured_block(_square(3))
    optimized = generate_structured_block(_square(3), optimization_steps=8)
    boundary = np.zeros((4, 4), dtype=np.bool_)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    np.testing.assert_array_equal(
        np.asarray(initial.mesh.coordinates)[boundary.reshape(-1)],
        np.asarray(optimized.mesh.coordinates)[boundary.reshape(-1)],
    )
    assert optimized.optimization is not None
    assert optimized.embedding.status == "certified"


def test_inverted_tensor_block_cannot_publish_positive_status() -> None:
    edges = tuple(
        TransfiniteCurveControl(LineCurve(origin, direction), name, (0.0, 1.0), 2)
        for name, origin, direction in (
            ("left", (1.0, 0.0), (0.0, 1.0)),
            ("right", (0.0, 0.0), (0.0, 1.0)),
            ("bottom", (1.0, 0.0), (-1.0, 0.0)),
            ("top", (1.0, 1.0), (-1.0, 0.0)),
        )
    )
    with pytest.raises(MeshingFailure, match="Jacobian"):
        generate_structured_block(TransfiniteBlock("inverted", edges))


def test_transfinite_lattice_refuses_against_the_original_live_storage_pool() -> None:
    from phydrax.meshing._structured import _boundary_lattice
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    block = _square(32)
    identity = block.block_id
    limits = phx.meshing.MeshingLimits(maximum_scratch_bytes=1024**2)
    with pytest.raises(MeshingFailure) as refused:
        with native_volume_execution_budget(limits) as execution:
            retained = execution.allocate_host_array((1024**2 - 4096,), np.uint8)
            retained.fill(23)
            _boundary_lattice(block)
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert retained[0] == retained[-1] == 23
    assert block.block_id == identity


def test_transfinite_source_queries_share_the_original_remaining_allowance() -> None:
    from phydrax.meshing._structured import _curve_values
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    block = _square(1)
    first, second = block.boundaries[:2]
    assert isinstance(first, TransfiniteCurveControl)
    assert isinstance(second, TransfiniteCurveControl)
    limits = phx.meshing.MeshingLimits(maximum_geometry_queries=3)
    with pytest.raises(MeshingFailure) as refused:
        with native_volume_execution_budget(limits):
            values = _curve_values(first)
            np.testing.assert_array_equal(values, ((0.0, 0.0), (0.0, 1.0)))
            _curve_values(second)
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED


@pytest.mark.parametrize(
    "shape,expected",
    (
        ((2, 3), ((0, 3, 4, 1), (1, 4, 5, 2))),
        ((2, 2, 3), ((0, 6, 9, 3, 1, 7, 10, 4), (1, 7, 10, 4, 2, 8, 11, 5))),
    ),
)
def test_logical_connectivity_retains_reference_corner_orientation(
    shape: tuple[int, ...], expected: tuple[tuple[int, ...], ...]
) -> None:
    from phydrax.meshing._structured import logical_cells

    np.testing.assert_array_equal(logical_cells(shape), expected)


def test_logical_connectivity_refuses_before_int32_node_identity_wraparound() -> None:
    from phydrax.meshing._structured import logical_cells

    with pytest.raises(MeshingFailure) as refused:
        logical_cells((2**31 + 1, 2))
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_independent_cold_oracle_preserves_original_source_and_rejects_changed_content(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import write_array_archive
    from tools.meshing_qualification import (
        _family_oracle_content_id,
        _family_oracle_payload,
        _family_read_oracle,
    )

    domain = phx.geometry.PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("original-material",),
        source_id="original-cold-oracle",
    )
    case = {
        "name": "independent",
        "domain": domain,
        "expected_regions": {"original-material": 2.0},
        "dimension": 2,
    }
    metadata, arrays = _family_oracle_payload(case)
    content_id = _family_oracle_content_id(metadata, arrays)
    path = write_array_archive(tmp_path / "oracle.zip", manifest=metadata, arrays=arrays)
    restored = _family_read_oracle(str(path), content_id, None)
    assert restored["domain"].domain_id == domain.domain_id
    assert restored["expected_regions"] == case["expected_regions"]
    np.testing.assert_array_equal(restored["domain"].vertices, domain.vertices)
    changed = dict(metadata)
    changed["case"] = {**metadata["case"], "expected_regions": {"original-material": 3.0}}
    write_array_archive(path, manifest=changed, arrays=arrays)
    with pytest.raises(RuntimeError, match="parent-authored content identity"):
        _family_read_oracle(str(path), content_id, None)
