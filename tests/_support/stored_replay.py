"""Two-process stored association replay at the lifecycle boundary."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax.lifecycle._meshing_source_records import (
    reconstruct_distributed_stored_fields,
)
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from phydrax.meshing import GeometryAssociation, GeometryAssociationKind
from tests._support.recipes import restore_with_identity


def _sharded_leaf(leaf: jax.Array, sharding: NamedSharding) -> jax.Array:
    host = np.asarray(leaf)

    def local_slice(index: tuple[slice, ...] | None) -> NDArray[np.generic]:
        return host if index is None else host[index]

    return jax.make_array_from_callback(host.shape, sharding, local_slice)


def association_round_trip() -> dict[str, int | bool]:
    register_meshing_source_artifacts()
    value = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        "implicit-source",
        "declared-revision",
        "face-set",
        np.asarray((11, 29), dtype=np.int64),
        ("implicit-zero-set", "implicit-zero-set"),
        np.zeros((2,), dtype=np.float64),
    )
    leaves, structure = jax.tree.flatten(value)
    mesh = Mesh(np.asarray(jax.devices()), ("process",))
    sharding = NamedSharding(mesh, PartitionSpec("process"))
    # Source-identity routes are immutable host metadata, not numerical banks.
    distributed = [
        _sharded_leaf(leaf, sharding) if isinstance(leaf, jax.Array) else leaf
        for leaf in leaves
    ]
    value = jax.tree.unflatten(structure, distributed)
    assert all(
        not leaf.is_fully_addressable
        for leaf in distributed if isinstance(leaf, jax.Array)
    )
    assert isinstance(value.source_entities.row_indices, np.ndarray)
    restored = reconstruct_distributed_stored_fields(
        value, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    assert jax.tree.structure(restored) == structure
    assert all(
        left is right
        for left, right in zip(jax.tree.leaves(restored), distributed, strict=True)
    )
    validate_meshing_source_closure(value)
    with pytest.raises(ValueError, match=r"stored_fields\.GeometryAssociation.*bytes"):
        reconstruct_distributed_stored_fields(
            value, limits=ArrayArchiveLimits(max_member_bytes=1)
        )
    forged = restore_with_identity(value, "association_id", "forged-association-id")
    with pytest.raises(
        ValueError, match=r"stored_fields\.GeometryAssociation.*association_id"
    ):
        reconstruct_distributed_stored_fields(forged, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
    with pytest.raises(
        ValueError, match=r"stored_fields\.GeometryAssociation.*association_id"
    ):
        validate_meshing_source_closure(forged)
    return {"process": jax.process_index(), "round_trip": True, "forged_refused": True}
