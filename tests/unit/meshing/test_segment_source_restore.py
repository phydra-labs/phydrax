"""Segment query indexes authenticate retained hierarchy state at checkpoints."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._bvh import BVHBuildKind, BVHBuildPolicy
from phydrax._fingerprint import tree_content_equal
from phydrax.geometry.simplicial._bvh import SegmentBVH
from phydrax.geometry.simplicial._regions import SegmentMesh
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)


@pytest.fixture
def segment_index() -> SegmentBVH:
    vertices = np.asarray(
        tuple((index, index % 3) for index in range(12)), dtype=np.float64
    )
    edges = np.stack((np.arange(11), np.arange(1, 12)), axis=1).astype(np.int32)
    return SegmentBVH(
        vertices, edges, policy=BVHBuildPolicy(leaf_size=2), execution="bvh"
    )


@pytest.mark.parametrize("kind", tuple(BVHBuildKind))
@pytest.mark.parametrize("refitted", (False, True))
def test_segment_index_checkpoint_preserves_policy_and_refitted_topology(
    kind: BVHBuildKind, refitted: bool, tmp_path: Path
) -> None:
    vertices = np.asarray(
        tuple((index, index % 3) for index in range(12)), dtype=np.float64
    )
    edges = np.stack((np.arange(11), np.arange(1, 12)), axis=1).astype(np.int32)
    index = SegmentBVH(
        vertices,
        edges,
        policy=BVHBuildPolicy(kind, leaf_size=2, sah_bins=4, morton_code_bits=63),
        execution="bvh",
        dense_workspace_bytes=256,
    )
    if refitted:
        moved = vertices * np.asarray((0.5, 2.0)) + np.asarray((3.0, -1.0))
        index = index.refit(moved)
    register_meshing_source_artifacts()
    restored = reconstruct_meshing_source(index)
    assert tree_content_equal(restored, index)
    path = tmp_path / "segment-source.npz"
    receipt = write_meshing_source_closure(path, index)
    loaded = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert tree_content_equal(loaded, index)
    points = jnp.asarray(((0.2, 0.1), (3.1, 1.0)), dtype=jnp.float64)
    assert tree_content_equal(
        loaded.nearest_segments(points), index.nearest_segments(points)
    )


@pytest.mark.parametrize(
    "fault", ("children", "leaf_items", "node_bounds", "item_bounds", "edges")
)
def test_segment_restore_refuses_invalid_tree_or_geometry(
    segment_index: SegmentBVH, fault: str
) -> None:
    register_meshing_source_artifacts()
    match fault:
        case "children":
            altered = eqx.tree_at(
                lambda value: value.bvh.left,
                segment_index,
                -jnp.ones_like(segment_index.bvh.left),
            )
        case "leaf_items":
            altered = eqx.tree_at(
                lambda value: value.bvh.leaf_items,
                segment_index,
                jnp.zeros_like(segment_index.bvh.leaf_items),
            )
        case "node_bounds":
            altered = eqx.tree_at(
                lambda value: value.bvh.bbox_min,
                segment_index,
                segment_index.bvh.bbox_min + 100.0,
            )
        case "item_bounds":
            altered = eqx.tree_at(
                lambda value: value.bvh.item_bbox_max,
                segment_index,
                segment_index.bvh.item_bbox_max + 1.0,
            )
        case "edges":
            altered = eqx.tree_at(
                lambda value: value.edges, segment_index, segment_index.edges.at[0].set(0)
            )
        case _:
            raise ValueError(
                f"SegmentRestoreTest.fault must name an admitted fault; got {fault!r}"
            )
    expected = {
        "children": r"PackedBVH\.children",
        "leaf_items": r"PackedBVH\.leaf_items",
        "node_bounds": r"PackedBVH\.bounds",
        "item_bounds": r"PackedBVH\.leaf_bounds",
        "edges": "distinct non-negative vertices",
    }[fault]
    with pytest.raises(ValueError, match=expected):
        SegmentBVH.from_stored_fields(altered)


def test_segment_mesh_checkpoint_binds_derived_topology_edges(tmp_path: Path) -> None:
    vertices = jnp.asarray(((0, 0), (1, 0), (1, 1)), dtype=jnp.float64)
    edges = jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32)
    source = SegmentMesh(vertices, edges, source_id="checkpoint-segments")
    register_meshing_source_artifacts()
    restored = reconstruct_meshing_source(source)
    assert tree_content_equal(restored, source)
    path = tmp_path / "segment-mesh-source.npz"
    receipt = write_meshing_source_closure(path, source)
    loaded = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert tree_content_equal(loaded, source)
