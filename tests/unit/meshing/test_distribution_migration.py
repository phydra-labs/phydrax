#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Real-process stable-ID forest migration, restart repacking, and replay."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import jax
import numpy as np
import pytest

from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
    refine_adaptive_simplex,
)
from phydrax.meshing._distribution import (
    ShardedOwnerDirectoryData,
    ShardedOwnerDirectoryWitness,
)
from phydrax.meshing._distribution_forest import (
    extract_simplex_forest,
    reconstruct_adaptive_simplex_state,
    SimplexForestRecords,
)
from phydrax.meshing._distribution_migration import (
    accept_ownerlocal_migration,
    OwnerLocalMigrationState,
    prepare_ownerlocal_migration,
)
from phydrax.meshing._restart_distribution import repack_simplex_restart


def _mesh_state() -> tuple[AdaptiveSimplexLayout, AdaptiveSimplexState, np.ndarray]:
    coordinates = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (3.0, 1.0),
            (6.0, 0.0),
            (7.0, 0.0),
            (6.0, 1.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8)), dtype=np.int32)
    vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    cell_ids = np.arange(cells.shape[0], dtype=np.int64)
    layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=32,
        cell_capacity=32,
        protected_edge_capacity=1,
        coordinate_dtype=np.dtype(np.float64),
    )
    initial = adaptive_simplex_state(
        layout,
        coordinates=coordinates,
        vertex_ids=vertex_ids,
        vertex_active=np.ones(vertex_ids.shape, dtype=np.bool_),
        vertex_parents=np.full((vertex_ids.size, 2), -1, dtype=np.int64),
        vertex_protected=np.zeros(vertex_ids.shape, dtype=np.bool_),
        cells=cells,
        tuples=cells,
        tags=np.full(cell_ids.shape, 2, dtype=np.int32),
        blocks=np.zeros(cell_ids.shape, dtype=np.int32),
        generations=np.zeros(cell_ids.shape, dtype=np.int32),
        parents=np.full(cell_ids.shape, -1, dtype=np.int64),
        children=np.full((cell_ids.size, 2), -1, dtype=np.int64),
        bisection_vertices=np.full(cell_ids.shape, -1, dtype=np.int64),
        cell_ids=cell_ids,
        cell_active=np.ones(cell_ids.shape, dtype=np.bool_),
        cell_classes=np.zeros(cell_ids.shape, dtype=np.int32),
        facet_classes=np.zeros((cell_ids.size, 3), dtype=np.int32),
        protected_edges=np.zeros((0, 2), dtype=np.int64),
        next_vertex_id=vertex_ids.size,
        next_cell_id=cell_ids.size,
    )
    update = refine_adaptive_simplex(
        layout,
        initial,
        np.asarray(initial.mesh.cell_active, dtype=np.bool_),
    )
    return update.layout, update.state, cell_ids


def test_live_forest_discards_retired_slots_without_reusing_ids() -> None:
    from phydrax.discretization import coarsen_adaptive_simplex

    layout, refined, root_ids = _mesh_state()
    live = extract_simplex_forest(refined)
    restored = reconstruct_adaptive_simplex_state(live, refined, layout)
    update = coarsen_adaptive_simplex(
        layout, restored, np.asarray(restored.mesh.cell_active, dtype=np.bool_)
    )
    compact = extract_simplex_forest(update.state)
    np.testing.assert_array_equal(compact.cell_ids, root_ids)
    np.testing.assert_array_equal(compact.vertex_ids, np.arange(9, dtype=np.int64))
    np.testing.assert_array_equal(compact.cell_field_map()["children_ids"], -1)
    assert compact.next_cell_id == live.next_cell_id
    assert compact.next_vertex_id == live.next_vertex_id
    next_state = reconstruct_adaptive_simplex_state(compact, update.state, update.layout)
    next_update = refine_adaptive_simplex(
        update.layout, next_state, np.asarray(next_state.mesh.cell_active, dtype=np.bool_)
    )
    next_live = extract_simplex_forest(next_update.state)
    new_ids = np.asarray(next_live.cell_ids)[
        np.asarray(next_live.cell_field_map()["active"])
    ]
    assert np.all(new_ids >= live.next_cell_id)
    np.testing.assert_array_equal(
        np.asarray(next_live.vertex_field_map()["coordinates"])[:9],
        np.asarray(compact.vertex_field_map()["coordinates"]),
    )


def test_live_forest_levels_follow_parent_depth_not_executor_rounds() -> None:
    import equinox as eqx
    import jax.numpy as jnp

    _, state, _ = _mesh_state()
    first = extract_simplex_forest(state)
    changed = eqx.tree_at(
        lambda value: value.vertex_levels,
        state,
        jnp.asarray(np.asarray(state.vertex_levels) + 7),
    )
    second = extract_simplex_forest(changed)
    np.testing.assert_array_equal(
        first.vertex_field_map()["levels"], second.vertex_field_map()["levels"]
    )
    assert first.forest_id == second.forest_id


def test_periodic_live_forest_checks_exact_restriction_and_tiling() -> None:
    from fractions import Fraction

    from phydrax.meshing._distribution_forest import PeriodicSimplexForest

    _, state, _ = _mesh_state()
    forest = extract_simplex_forest(state)
    exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(forest.vertex_field_map()["coordinates"])
    )
    roots = np.asarray(forest.vertex_ids)
    windings = np.zeros((roots.size, 2), dtype=np.int64)
    wrapper = PeriodicSimplexForest(
        forest, roots, windings, exact, ("triangles",), "source", "revision", "lattice"
    )
    assert wrapper.records.forest_id == forest.forest_id
    changed_exact = list(exact)
    changed_exact[-1] = (exact[-1][0] + Fraction(1, 100), exact[-1][1])
    changed_vertices = {
        name: np.asarray(values).copy() for name, values in forest.vertex_fields
    }
    changed_vertices["coordinates"][-1, 0] = float(changed_exact[-1][0])
    changed_forest = SimplexForestRecords(
        forest.cell_ids,
        forest.cell_owner_ranks,
        dict(forest.cell_fields),
        forest.vertex_ids,
        forest.vertex_owner_ranks,
        changed_vertices,
        forest.protected_edge_ids,
        forest.dimension,
        forest.ambient_dimension,
        forest.next_vertex_id,
        forest.next_cell_id,
    )
    with pytest.raises(ValueError, match="exact parent restriction"):
        PeriodicSimplexForest(
            changed_forest,
            roots,
            windings,
            tuple(changed_exact),
            ("triangles",),
            "source",
            "revision",
            "lattice",
        )
    changed_cells = {
        name: np.asarray(values).copy() for name, values in forest.cell_fields
    }
    children = changed_cells["children_ids"][0]
    first, second = np.searchsorted(np.asarray(forest.cell_ids), children)
    changed_cells["cells"][first] = changed_cells["cells"][second]
    invalid_tiling = SimplexForestRecords(
        forest.cell_ids,
        forest.cell_owner_ranks,
        changed_cells,
        forest.vertex_ids,
        forest.vertex_owner_ranks,
        dict(forest.vertex_fields),
        forest.protected_edge_ids,
        forest.dimension,
        forest.ambient_dimension,
        forest.next_vertex_id,
        forest.next_cell_id,
    )
    with pytest.raises(ValueError, match="exactly tile"):
        PeriodicSimplexForest(
            invalid_tiling,
            roots,
            windings,
            exact,
            ("triangles",),
            "source",
            "revision",
            "lattice",
        )


def test_periodic_forest_cold_restore_rejects_consistent_identity_tamper(
    tmp_path: Path,
) -> None:
    import copy
    import subprocess
    from fractions import Fraction

    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.meshing._distribution_forest import PeriodicSimplexForest
    from phydrax.meshing._model_codec import write_model_artifact

    _, state, _ = _mesh_state()
    records = extract_simplex_forest(state)
    exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(records.vertex_field_map()["coordinates"])
    )
    roots = np.asarray(records.vertex_ids, dtype=np.int64)
    windings = np.zeros((roots.size, 2), dtype=np.int64)
    valid = PeriodicSimplexForest(
        records, roots, windings, exact, ("triangles",), "source", "revision", "lattice"
    )
    changed = list(exact)
    changed[-1] = (exact[-1][0] + Fraction(1, 100), exact[-1][1])
    vertices = {name: np.asarray(value).copy() for name, value in records.vertex_fields}
    vertices["coordinates"][-1, 0] = float(changed[-1][0])
    altered = SimplexForestRecords(
        records.cell_ids,
        records.cell_owner_ranks,
        dict(records.cell_fields),
        records.vertex_ids,
        records.vertex_owner_ranks,
        vertices,
        records.protected_edge_ids,
        records.dimension,
        records.ambient_dimension,
        records.next_vertex_id,
        records.next_cell_id,
    )
    counterfeit = copy.copy(valid)
    object.__setattr__(counterfeit, "records", altered)
    object.__setattr__(counterfeit, "exact_vertex_coordinates", tuple(changed))
    object.__setattr__(
        counterfeit,
        "forest_id",
        canonical_fingerprint(
            {
                "kind": "live-periodic-simplex-forest",
                "records": altered.forest_id,
                "owners": (
                    np.asarray(altered.cell_owner_ranks),
                    np.asarray(altered.vertex_owner_ranks),
                ),
                "roots": roots,
                "windings": windings,
                "exact": tuple(
                    tuple((value.numerator, value.denominator) for value in row)
                    for row in changed
                ),
                "blocks": ("triangles",),
                "source": ("source", "revision", "lattice"),
            }
        ),
    )
    path = write_model_artifact(tmp_path / "false-forest.phx", counterfeit, limits=None)
    worker = (
        "import sys\n"
        "from phydrax.meshing._distribution_forest import PeriodicSimplexForest\n"
        "from phydrax.meshing._model_codec import read_model_artifact, restore_model_archive_value\n"
        "try:\n"
        " read_model_artifact(sys.argv[1], expected_type=PeriodicSimplexForest, restore=restore_model_archive_value, limits=None)\n"
        "except Exception as error:\n"
        " assert 'exact parent restriction' in str(error.__cause__), repr(error.__cause__)\n"
        "else:\n"
        " raise AssertionError('consistent-identity false midpoint was accepted')\n"
    )
    result = subprocess.run(
        (sys.executable, "-c", worker, str(path)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _root_owners(
    cell_ids: np.ndarray, cell_parent_ids: np.ndarray, part_count: int, /
) -> dict[int, int]:
    parent_by_id = {
        int(identifier): int(parent)
        for identifier, parent in zip(cell_ids, cell_parent_ids, strict=True)
    }
    owners: dict[int, int] = {}
    for identifier_value in cell_ids:
        identifier = int(identifier_value)
        root = identifier
        while parent_by_id[root] >= 0:
            root = parent_by_id[root]
        owners[identifier] = root % part_count
    return owners


def _owner_maps(
    forest: SimplexForestRecords, part_count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    cell_ids = np.asarray(forest.cell_ids, dtype=np.int64)
    vertex_ids = np.asarray(forest.vertex_ids, dtype=np.int64)
    cell_fields = dict((name, np.asarray(value)) for name, value in forest.cell_fields)
    cell_owner_map = _root_owners(cell_ids, cell_fields["parent_ids"], part_count)
    vertex_owner_map: dict[int, int] = {}
    for cell_id, cell_vertices in zip(
        cell_ids.tolist(), cell_fields["cells"], strict=True
    ):
        owner = cell_owner_map[int(cell_id)]
        for vertex_value in cell_vertices:
            vertex_id = int(vertex_value)
            previous = vertex_owner_map.get(vertex_id)
            if previous is not None and previous != owner:
                raise ValueError("Acceptance mesh families must not share vertices.")
            vertex_owner_map[vertex_id] = owner
    if set(vertex_owner_map) != set(int(identifier) for identifier in vertex_ids):
        raise ValueError("Every acceptance vertex must belong to one cell family.")
    return (
        np.asarray(
            [cell_owner_map[int(identifier)] for identifier in cell_ids], dtype=np.int32
        ),
        np.asarray(
            [vertex_owner_map[int(identifier)] for identifier in vertex_ids],
            dtype=np.int32,
        ),
    )


def _slice_forest(
    forest: SimplexForestRecords,
    cell_owners: np.ndarray,
    vertex_owners: np.ndarray,
    rank: int,
    /,
) -> SimplexForestRecords:
    cell_mask = cell_owners == rank
    vertex_mask = vertex_owners == rank
    vertex_ids = np.asarray(forest.vertex_ids, dtype=np.int64)[vertex_mask]
    protected_edges = np.asarray(forest.protected_edge_ids, dtype=np.int64)
    if protected_edges.size:
        protected_edges = protected_edges[
            np.all(np.isin(protected_edges, vertex_ids), axis=1)
        ]
    return SimplexForestRecords(
        np.asarray(forest.cell_ids, dtype=np.int64)[cell_mask],
        np.asarray(forest.cell_owner_ranks, dtype=np.int32)[cell_mask],
        {name: np.asarray(value)[cell_mask] for name, value in forest.cell_fields},
        vertex_ids,
        np.asarray(forest.vertex_owner_ranks, dtype=np.int32)[vertex_mask],
        {name: np.asarray(value)[vertex_mask] for name, value in forest.vertex_fields},
        protected_edges,
        forest.dimension,
        forest.ambient_dimension,
        forest.next_vertex_id,
        forest.next_cell_id,
    )


def _merge_states(
    states: Sequence[OwnerLocalMigrationState],
    reference: OwnerLocalMigrationState,
    next_vertex_id: int,
    next_cell_id: int,
    /,
) -> OwnerLocalMigrationState:
    records = states
    schema = reference.forest
    cell_names = tuple(name for name, _ in schema.cell_fields)
    vertex_names = tuple(name for name, _ in schema.vertex_fields)
    cell_payload_names = tuple(name for name, _ in reference.cell_payloads)
    vertex_payload_names = tuple(name for name, _ in reference.vertex_payloads)
    if records:
        cell_ids = np.concatenate(
            [np.asarray(state.forest.cell_ids, dtype=np.int64) for state in records]
        )
        vertex_ids = np.concatenate(
            [np.asarray(state.forest.vertex_ids, dtype=np.int64) for state in records]
        )
        cell_owners = np.concatenate(
            [
                np.asarray(state.forest.cell_owner_ranks, dtype=np.int32)
                for state in records
            ]
        )
        vertex_owners = np.concatenate(
            [
                np.asarray(state.forest.vertex_owner_ranks, dtype=np.int32)
                for state in records
            ]
        )
        cell_values = {
            name: np.concatenate(
                [np.asarray(dict(state.forest.cell_fields)[name]) for state in records]
            )
            for name in cell_names
        }
        vertex_values = {
            name: np.concatenate(
                [np.asarray(dict(state.forest.vertex_fields)[name]) for state in records]
            )
            for name in vertex_names
        }
        cell_payload_values = {
            name: np.concatenate(
                [np.asarray(dict(state.cell_payloads)[name]) for state in records]
            )
            for name in cell_payload_names
        }
        vertex_payload_values = {
            name: np.concatenate(
                [np.asarray(dict(state.vertex_payloads)[name]) for state in records]
            )
            for name in vertex_payload_names
        }
        protected = np.concatenate(
            [
                np.asarray(state.forest.protected_edge_ids, dtype=np.int64)
                for state in records
            ]
        )
    else:
        cell_ids = np.zeros((0,), dtype=np.int64)
        vertex_ids = np.zeros((0,), dtype=np.int64)
        cell_owners = np.zeros((0,), dtype=np.int32)
        vertex_owners = np.zeros((0,), dtype=np.int32)
        cell_values = {
            name: np.zeros(
                (0, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype
            )
            for name, value in schema.cell_fields
        }
        vertex_values = {
            name: np.zeros(
                (0, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype
            )
            for name, value in schema.vertex_fields
        }
        cell_payload_values = {
            name: np.zeros(
                (0, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype
            )
            for name, value in reference.cell_payloads
        }
        vertex_payload_values = {
            name: np.zeros(
                (0, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype
            )
            for name, value in reference.vertex_payloads
        }
        protected = np.zeros((0, 2), dtype=np.int64)
    cell_order = np.argsort(cell_ids, kind="stable")
    vertex_order = np.argsort(vertex_ids, kind="stable")
    if (
        np.unique(cell_ids).size != cell_ids.size
        or np.unique(vertex_ids).size != vertex_ids.size
    ):
        raise ValueError("Saved owner-local shards contain duplicate forest IDs.")
    protected = np.unique(protected, axis=0)
    forest = SimplexForestRecords(
        cell_ids[cell_order],
        cell_owners[cell_order],
        {name: values[cell_order] for name, values in cell_values.items()},
        vertex_ids[vertex_order],
        vertex_owners[vertex_order],
        {name: values[vertex_order] for name, values in vertex_values.items()},
        protected,
        schema.dimension,
        schema.ambient_dimension,
        next_vertex_id,
        next_cell_id,
    )
    return OwnerLocalMigrationState(
        forest,
        cell_payloads={
            name: values[cell_order] for name, values in cell_payload_values.items()
        },
        vertex_payloads={
            name: values[vertex_order] for name, values in vertex_payload_values.items()
        },
    )


def _archive_payload(
    state: OwnerLocalMigrationState,
    directory: ShardedOwnerDirectoryData,
    source_identity: str,
    /,
) -> dict[str, np.ndarray]:
    forest = state.forest
    arrays: dict[str, np.ndarray] = {
        "source_identity": np.asarray(source_identity),
        "rank": np.asarray(directory.rank, dtype=np.int64),
        "dimension": np.asarray(forest.dimension, dtype=np.int64),
        "ambient_dimension": np.asarray(forest.ambient_dimension, dtype=np.int64),
        "next_vertex_id": np.asarray(forest.next_vertex_id, dtype=np.int64),
        "next_cell_id": np.asarray(forest.next_cell_id, dtype=np.int64),
        "cell_field_names": np.asarray(
            json.dumps(tuple(name for name, _ in forest.cell_fields))
        ),
        "vertex_field_names": np.asarray(
            json.dumps(tuple(name for name, _ in forest.vertex_fields))
        ),
        "cell_payload_names": np.asarray(
            json.dumps(tuple(name for name, _ in state.cell_payloads))
        ),
        "vertex_payload_names": np.asarray(
            json.dumps(tuple(name for name, _ in state.vertex_payloads))
        ),
        "cell_ids": np.asarray(forest.cell_ids, dtype=np.int64),
        "cell_owner_ranks": np.asarray(forest.cell_owner_ranks, dtype=np.int32),
        "vertex_ids": np.asarray(forest.vertex_ids, dtype=np.int64),
        "vertex_owner_ranks": np.asarray(forest.vertex_owner_ranks, dtype=np.int32),
        "protected_edge_ids": np.asarray(forest.protected_edge_ids, dtype=np.int64),
        "source_manifest_id": np.asarray(directory.source_manifest_id),
        "directory_group_id": np.asarray(directory.group_id),
        "directory_rank": np.asarray(directory.rank, dtype=np.int64),
        "directory_part_count": np.asarray(directory.part_count, dtype=np.int64),
        "source_directory_ids": np.asarray(
            directory.source_directory_ids, dtype=np.int64
        ),
        "source_directory_owner_ranks": np.asarray(
            directory.source_directory_owner_ranks, dtype=np.int32
        ),
        "cell_directory_ids": np.asarray(directory.cell_directory_ids, dtype=np.int64),
        "cell_directory_owner_ranks": np.asarray(
            directory.cell_directory_owner_ranks, dtype=np.int32
        ),
        "vertex_directory_ids": np.asarray(
            directory.vertex_directory_ids, dtype=np.int64
        ),
        "vertex_directory_owner_ranks": np.asarray(
            directory.vertex_directory_owner_ranks, dtype=np.int32
        ),
        "source_shard_records": np.asarray(json.dumps(directory.source_shard_records)),
        "cell_shard_records": np.asarray(json.dumps(directory.cell_shard_records)),
        "vertex_shard_records": np.asarray(json.dumps(directory.vertex_shard_records)),
        "source_directory_id": np.asarray(directory.source_directory_id),
        "cell_directory_id": np.asarray(directory.cell_directory_id),
        "vertex_directory_id": np.asarray(directory.vertex_directory_id),
    }
    arrays.update(
        {f"cell_field_{name}": np.asarray(value) for name, value in forest.cell_fields}
    )
    arrays.update(
        {
            f"vertex_field_{name}": np.asarray(value)
            for name, value in forest.vertex_fields
        }
    )
    arrays.update(
        {f"cell_payload_{name}": np.asarray(value) for name, value in state.cell_payloads}
    )
    arrays.update(
        {
            f"vertex_payload_{name}": np.asarray(value)
            for name, value in state.vertex_payloads
        }
    )
    return arrays


def _load_archive(
    path: Path, /
) -> tuple[OwnerLocalMigrationState, ShardedOwnerDirectoryData, str, int]:
    with np.load(path, allow_pickle=False) as archive:
        archive_names = tuple(
            str(name) for name in json.loads(str(np.asarray(archive["arr_0"]).item()))
        )
        values = {
            name: np.asarray(archive[f"arr_{index + 1}"]).copy()
            for index, name in enumerate(archive_names)
        }

    def scalar_int(name: str, /) -> int:
        return int(values[name].item())

    def scalar_text(name: str, /) -> str:
        return str(values[name].item())

    cell_names = tuple(str(name) for name in json.loads(scalar_text("cell_field_names")))
    vertex_names = tuple(
        str(name) for name in json.loads(scalar_text("vertex_field_names"))
    )
    cell_payload_names = tuple(
        str(name) for name in json.loads(scalar_text("cell_payload_names"))
    )
    vertex_payload_names = tuple(
        str(name) for name in json.loads(scalar_text("vertex_payload_names"))
    )
    forest = SimplexForestRecords(
        values["cell_ids"],
        values["cell_owner_ranks"],
        {name: values[f"cell_field_{name}"] for name in cell_names},
        values["vertex_ids"],
        values["vertex_owner_ranks"],
        {name: values[f"vertex_field_{name}"] for name in vertex_names},
        values["protected_edge_ids"],
        scalar_int("dimension"),
        scalar_int("ambient_dimension"),
        scalar_int("next_vertex_id"),
        scalar_int("next_cell_id"),
    )
    state = OwnerLocalMigrationState(
        forest,
        cell_payloads={
            name: values[f"cell_payload_{name}"] for name in cell_payload_names
        },
        vertex_payloads={
            name: values[f"vertex_payload_{name}"] for name in vertex_payload_names
        },
    )

    def records(name: str, /) -> tuple[tuple[int, str, int, str], ...]:
        return tuple(
            (int(row[0]), str(row[1]), int(row[2]), str(row[3]))
            for row in json.loads(scalar_text(name))
        )

    directory = ShardedOwnerDirectoryData(
        scalar_text("source_manifest_id"),
        scalar_text("directory_group_id"),
        scalar_int("directory_rank"),
        scalar_int("directory_part_count"),
        values["source_directory_ids"],
        values["source_directory_owner_ranks"],
        values["cell_directory_ids"],
        values["cell_directory_owner_ranks"],
        values["vertex_directory_ids"],
        values["vertex_directory_owner_ranks"],
        records("source_shard_records"),
        records("cell_shard_records"),
        records("vertex_shard_records"),
        scalar_text("source_directory_id"),
        scalar_text("cell_directory_id"),
        scalar_text("vertex_directory_id"),
    )
    return state, directory, scalar_text("source_identity"), scalar_int("rank")


def _save_archive(
    path: Path,
    state: OwnerLocalMigrationState,
    directory: ShardedOwnerDirectoryData,
    source_identity: str,
    /,
) -> None:
    payload = _archive_payload(state, directory, source_identity)
    names = tuple(sorted(payload))
    np.savez_compressed(
        path,
        np.asarray(json.dumps(names)),
        *(payload[name] for name in names),
    )


def _group(parts: int, phase: str, /) -> ExecutionGroup:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    ranks = tuple(device.process_index for device in devices)
    if len(devices) != parts or jax.process_count() != parts:
        raise RuntimeError(
            "Owner-local simplex acceptance requires one CPU device per process."
        )
    axis_name = f"migration_{phase}"
    return ExecutionGroup(
        ExecutionGroupSpec(
            f"simplex-migration-{phase}-{parts}",
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=((axis_name, parts),),
        ),
        devices,
    )


def _launch_phase(
    phase: str,
    parts: int,
    old_part_count: int,
    source_directory: Path,
    target_directory: Path,
    /,
) -> list[dict[str, object]]:
    import socket

    from phydrax import config as pxconfig
    from phydrax._execution_launch import launch_local_processes, LocalProcessLaunchPlan

    repository = Path(__file__).resolve().parents[3]
    script = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])['_distributed_migration_worker']()"
    )
    environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        environment.pop(name, None)
    process_environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=environment,
    )
    process_environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    results = launch_local_processes(
        (
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            phase,
            str(source_directory),
            str(target_directory),
            str(old_part_count),
        ),
        LocalProcessLaunchPlan(parts, f"127.0.0.1:{port}", devices_per_process=1),
        environment=process_environment,
        cwd=str(repository),
        timeout_seconds=900.0,
    )
    for result in results:
        if result.returncode != 0:
            raise RuntimeError(
                f"Migration phase {phase} rank {result.process_id} failed:\n"
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )
    return [
        json.loads(result.stdout.decode().strip().splitlines()[-1]) for result in results
    ]


def _distributed_live_family_worker(
    group: ExecutionGroup, rank: int, /
) -> dict[str, object]:
    from fractions import Fraction
    from phydrax.discretization import coarsen_adaptive_simplex
    from phydrax.meshing._distribution_forest import PeriodicSimplexForest
    from phydrax.meshing._distribution_migration import migrate_periodic_simplex_forest

    layout, state, _ = _mesh_state()
    extracted = extract_simplex_forest(state)
    cell_owners, vertex_owners = _owner_maps(extracted, 3)
    full = SimplexForestRecords(
        extracted.cell_ids,
        cell_owners,
        dict(extracted.cell_fields),
        extracted.vertex_ids,
        vertex_owners,
        dict(extracted.vertex_fields),
        extracted.protected_edge_ids,
        2,
        2,
        extracted.next_vertex_id,
        extracted.next_cell_id,
    )
    local = _slice_forest(full, cell_owners, vertex_owners, rank)
    original = PeriodicSimplexForest(
        local,
        local.vertex_ids,
        np.zeros((local.vertex_ids.size, 2), dtype=np.int64),
        tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in np.asarray(local.vertex_field_map()["coordinates"])
        ),
        ("triangles",),
        "migration-source",
        "migration-revision",
        "migration-lattice",
    )
    original_id = original.forest_id
    cell_fields = full.cell_field_map()
    active = np.asarray(cell_fields["active"], dtype=np.bool_)
    active_ids = np.asarray(full.cell_ids)[active]
    roots = np.asarray(cell_fields["parent_ids"])[active]
    ordinals = np.asarray(cell_fields["parent_ordinals"])[active]
    split_ids = active_ids[(roots + ordinals) % 3 == rank]
    split = migrate_periodic_simplex_forest(
        original, split_ids, group, axis_name="live_family_split"
    )
    split_fields = split.records.cell_field_map()
    split_active = np.asarray(split_fields["active"], dtype=np.bool_)
    assert np.unique(np.asarray(split.records.cell_owner_ranks)[split_active]).size == 3
    reunited_ids = active_ids[roots % 2 == rank]
    reunited = migrate_periodic_simplex_forest(
        split, reunited_ids, group, axis_name="live_family_reunite"
    )
    np.testing.assert_array_equal(
        np.asarray(reunited.records.cell_ids)[
            np.asarray(reunited.records.cell_field_map()["active"])
        ],
        reunited_ids,
    )
    restored = reconstruct_adaptive_simplex_state(reunited.records, state, layout)
    update = coarsen_adaptive_simplex(
        layout, restored, np.asarray(restored.mesh.cell_active, dtype=np.bool_)
    )
    compact = extract_simplex_forest(update.state)
    expected_roots = np.arange(3, dtype=np.int64)[np.arange(3) % 2 == rank]
    np.testing.assert_array_equal(compact.cell_ids, expected_roots)
    assert original.forest_id == original_id
    assert compact.next_cell_id == full.next_cell_id
    return {
        "rank": rank,
        "coarsened_roots": expected_roots.tolist(),
        "split_owned_ids": split_ids.tolist(),
    }


def test_real_process_split_live_families_reunite_and_coarsen(tmp_path: Path) -> None:
    if (
        jax.process_count() != 1
        or os.name != "posix"
        or os.environ.get("PYTEST_XDIST_WORKER") is not None
    ):
        pytest.skip(
            "Real-process live-family acceptance requires an isolated POSIX coordinator."
        )
    reports = _launch_phase("live-family3", 3, 3, tmp_path, tmp_path)
    assert [report["coarsened_roots"] for report in reports] == [[0, 2], [1], []]
    assert all(
        isinstance(report["split_owned_ids"], list)
        and len(report["split_owned_ids"]) == 2
        for report in reports
    )


def _distributed_migration_worker() -> None:
    from phydrax.discretization import coarsen_adaptive_simplex

    phase = sys.argv[2]
    source_directory = Path(sys.argv[3])
    target_directory = Path(sys.argv[4])
    old_part_count = int(sys.argv[5])
    parts = jax.process_count()
    rank = jax.process_index()
    group = _group(parts, phase)
    if phase == "live-family3":
        print(json.dumps(_distributed_live_family_worker(group, rank), sort_keys=True))
        return
    if phase == "write3":
        _layout, state, _ = _mesh_state()
        full = extract_simplex_forest(state)
        cell_owners, vertex_owners = _owner_maps(full, parts)
        host_ids = np.asarray(state.mesh.cell_ids, dtype=np.int64)
        host_vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)
        cell_owner_slots = np.full(host_ids.shape, -1, dtype=np.int32)
        vertex_owner_slots = np.full(host_vertex_ids.shape, -1, dtype=np.int32)
        for slot, identifier in enumerate(host_ids):
            if identifier >= 0:
                cell_owner_slots[slot] = cell_owners[
                    np.searchsorted(np.asarray(full.cell_ids), identifier)
                ]
        for slot, identifier in enumerate(host_vertex_ids):
            if identifier >= 0:
                vertex_owner_slots[slot] = vertex_owners[
                    np.searchsorted(np.asarray(full.vertex_ids), identifier)
                ]
        full = extract_simplex_forest(
            state,
            cell_owner_ranks=cell_owner_slots,
            vertex_owner_ranks=vertex_owner_slots,
        )
        local = _slice_forest(full, cell_owners, vertex_owners, rank)
        if not isinstance(local, SimplexForestRecords):
            raise TypeError("Source owner does not have SimplexForestRecords.")
        cell_fields = dict((name, np.asarray(value)) for name, value in local.cell_fields)
        vertex_fields = dict(
            (name, np.asarray(value)) for name, value in local.vertex_fields
        )
        state_value = OwnerLocalMigrationState(
            local,
            cell_payloads={
                "temperature": np.asarray(local.cell_ids, dtype=np.float64) * 0.25
                + np.asarray(cell_fields["generations"], dtype=np.float64),
                "history": np.stack(
                    (
                        np.asarray(local.cell_ids, dtype=np.int64),
                        np.asarray(cell_fields["generations"], dtype=np.int64),
                    ),
                    axis=1,
                ),
            },
            vertex_payloads={
                "potential": np.asarray(local.vertex_ids, dtype=np.float64) + 0.125,
                "history": np.stack(
                    (
                        np.asarray(local.vertex_ids, dtype=np.int64),
                        np.asarray(vertex_fields["levels"], dtype=np.int64),
                    ),
                    axis=1,
                ),
            },
        )
        target_cell_owners = np.full(local.cell_ids.shape, rank, dtype=np.int32)
        target_vertex_owners = np.full(local.vertex_ids.shape, rank, dtype=np.int32)
        prepared = prepare_ownerlocal_migration(
            state_value,
            target_cell_owners,
            target_vertex_owners,
            local.cell_ids,
            local.vertex_ids,
            group,
            axis_name="simplex_initial_owner_commit",
        )
        accepted = accept_ownerlocal_migration(
            prepared, group, axis_name="simplex_initial_owner_accept"
        )
        witness = ShardedOwnerDirectoryWitness(
            group,
            rank,
            "simplex-migration-source-manifest",
            "simplex-migration-owner-directory",
            accepted.forest.vertex_ids,
            np.full(accepted.forest.vertex_ids.shape, rank, dtype=np.int32),
            np.ones(accepted.forest.vertex_ids.shape, dtype=np.bool_),
            accepted.forest.cell_ids,
            np.full(accepted.forest.cell_ids.shape, rank, dtype=np.int32),
            accepted.forest.vertex_ids,
            np.full(accepted.forest.vertex_ids.shape, rank, dtype=np.int32),
            axis_name="simplex_initial_owner_directory",
        )
        directory_data = ShardedOwnerDirectoryData.from_witness(witness)
        _save_archive(
            source_directory / f"rank-{rank}.npz",
            accepted,
            directory_data,
            prepared.evidence.target_identity,
        )
        report = {
            "phase": phase,
            "rank": rank,
            "source_identity": prepared.evidence.source_identity,
            "target_identity": prepared.evidence.target_identity,
            "directory_id": witness.directory_id,
        }
    else:
        all_records = tuple(
            _load_archive(source_directory / f"rank-{old_rank}.npz")
            for old_rank in range(old_part_count)
        )
        state_shards = tuple(value[0] for value in all_records)
        directory_shards = tuple(value[1] for value in all_records)
        source_ids = {value[2] for value in all_records}
        if len(source_ids) != 1:
            raise ValueError("Restart archive owner records disagree on source identity.")
        source_identity = source_ids.pop()
        all_state = _merge_states(
            state_shards,
            state_shards[0],
            max(state.forest.next_vertex_id for state in state_shards),
            max(state.forest.next_cell_id for state in state_shards),
        )
        if not isinstance(all_state, OwnerLocalMigrationState):
            raise TypeError("Merged restart archive did not produce migration state.")
        global_cell_owners, global_vertex_owners = _owner_maps(all_state.forest, parts)
        global_cell_owner_by_id = {
            int(identifier): int(owner)
            for identifier, owner in zip(
                all_state.forest.cell_ids, global_cell_owners, strict=True
            )
        }
        global_vertex_owner_by_id = {
            int(identifier): int(owner)
            for identifier, owner in zip(
                all_state.forest.vertex_ids, global_vertex_owners, strict=True
            )
        }
        local_sources = tuple(
            state
            for state, directory, _, _ in all_records
            if directory.rank % parts == rank
        )
        local_state = _merge_states(
            local_sources,
            state_shards[0],
            all_state.forest.next_vertex_id,
            all_state.forest.next_cell_id,
        )
        if not isinstance(local_state, OwnerLocalMigrationState):
            raise TypeError("Local archived owners did not produce migration state.")
        target_cell_owners = np.asarray(
            [
                global_cell_owner_by_id[int(value)]
                for value in local_state.forest.cell_ids
            ],
            dtype=np.int32,
        )
        target_vertex_owners = np.asarray(
            [
                global_vertex_owner_by_id[int(value)]
                for value in local_state.forest.vertex_ids
            ],
            dtype=np.int32,
        )
        target_cell_ids = np.asarray(all_state.forest.cell_ids, dtype=np.int64)[
            global_cell_owners == rank
        ]
        target_vertex_ids = np.asarray(all_state.forest.vertex_ids, dtype=np.int64)[
            global_vertex_owners == rank
        ]
        repacked = repack_simplex_restart(
            local_state,
            directory_shards,
            target_cell_owners,
            target_vertex_owners,
            target_cell_ids,
            target_vertex_ids,
            group,
            target_vertex_ids,
            np.full(target_vertex_ids.shape, rank, dtype=np.int32),
            np.ones(target_vertex_ids.shape, dtype=np.bool_),
            target_cell_ids,
            np.full(target_cell_ids.shape, rank, dtype=np.int32),
            target_vertex_ids,
            np.full(target_vertex_ids.shape, rank, dtype=np.int32),
            directory_entry_id="simplex-migration-owner-directory",
            axis_name=f"simplex-restart-{phase}",
        )
        if (
            repacked.migration_evidence.source_identity != source_identity
            or repacked.migration_evidence.target_identity != source_identity
        ):
            raise ValueError(
                "Changed-topology restart changed scientific forest or field history."
            )
        directory_data = ShardedOwnerDirectoryData.from_witness(repacked.owner_directory)
        _save_archive(
            target_directory / f"rank-{rank}.npz",
            repacked.state,
            directory_data,
            repacked.migration_evidence.target_identity,
        )
        report = {
            "phase": phase,
            "rank": rank,
            "source_identity": repacked.migration_evidence.source_identity,
            "target_identity": repacked.migration_evidence.target_identity,
            "directory_id": repacked.owner_directory.directory_id,
        }
        if phase == "coarsen3":
            layout, template_state, roots = _mesh_state()
            restored = reconstruct_adaptive_simplex_state(
                repacked.state.forest, template_state, layout
            )
            update = coarsen_adaptive_simplex(
                layout,
                restored,
                np.asarray(restored.mesh.cell_active, dtype=np.bool_),
            )
            active_ids = np.asarray(update.state.mesh.cell_ids, dtype=np.int64)[
                np.asarray(update.state.mesh.cell_active, dtype=np.bool_)
            ]
            expected_roots = np.asarray(roots, dtype=np.int64)[
                np.asarray(
                    [int(value) % parts == rank for value in roots], dtype=np.bool_
                )
            ]
            if not np.array_equal(np.sort(active_ids), np.sort(expected_roots)):
                raise ValueError(
                    "Coarsening did not restore this owner's original forest roots."
                )
            from jax.experimental import multihost_utils

            gathered_roots = np.asarray(multihost_utils.process_allgather(active_ids))
            if not np.array_equal(np.sort(gathered_roots.reshape(-1)), np.sort(roots)):
                raise ValueError(
                    "Collective coarsening did not restore all original roots."
                )
            report["coarsened_roots"] = np.sort(gathered_roots.reshape(-1)).tolist()
        if phase == "restore4":
            from jax import lax
            from jax.sharding import PartitionSpec

            from phydrax.meshing._migration_execution import (
                LogicalMigrationParts,
                partition_map,
            )

            logical_axis = "historical_owner"
            routes = ((0, 1), (1, 2), (2, 0))
            logical = LogicalMigrationParts(
                group, 3, axis_name=logical_axis, neighbor_pairs=routes
            )

            def route_values(values: jax.Array, /) -> jax.Array:
                return lax.ppermute(values, logical_axis, logical.neighbor_pairs)

            replay = partition_map(
                route_values,
                parts=logical,
                in_specs=(PartitionSpec(logical_axis),),
                out_specs=PartitionSpec(logical_axis),
            )
            observed = np.asarray(replay(np.asarray((10, 20, 30), dtype=np.int32)))
            if observed.tolist() != [30, 10, 20]:
                raise ValueError(
                    "Historical logical-rank replay changed its route semantics."
                )
            report["logical_replay"] = observed.tolist()
    print(json.dumps(report, sort_keys=True))


def test_real_process_simplex_migration_restart_and_logical_replay(
    tmp_path: Path,
) -> None:
    """Refine on three owners, migrate/restart on two and four, then coarsen."""
    if jax.process_count() != 1:
        pytest.skip(
            "Real-process migration coordination must run outside a live JAX group."
        )
    if os.name != "posix" or os.environ.get("PYTEST_XDIST_WORKER") is not None:
        pytest.skip(
            "Real-process migration acceptance requires POSIX supervision outside xdist."
        )
    source = tmp_path / "source-three"
    moved = tmp_path / "migrated-two"
    coarsened = tmp_path / "coarsened-three"
    restored_four = tmp_path / "restored-four"
    for path in (source, moved, coarsened, restored_four):
        path.mkdir()
    source_reports = _launch_phase("write3", 3, 0, source, source)
    moved_reports = _launch_phase("restore2", 2, 3, source, moved)
    coarsened_reports = _launch_phase("coarsen3", 3, 2, moved, coarsened)
    restored_reports = _launch_phase("restore4", 4, 3, source, restored_four)
    source_identity = source_reports[0]["target_identity"]
    assert all(report["target_identity"] == source_identity for report in source_reports)
    assert all(report["source_identity"] == source_identity for report in moved_reports)
    assert all(report["target_identity"] == source_identity for report in moved_reports)
    assert all(
        report["source_identity"] == source_identity for report in coarsened_reports
    )
    assert all(
        report["target_identity"] == source_identity for report in coarsened_reports
    )
    assert all(
        report["source_identity"] == source_identity for report in restored_reports
    )
    assert all(
        report["target_identity"] == source_identity for report in restored_reports
    )
    assert coarsened_reports[0]["coarsened_roots"] == [0, 1, 2]
    assert all(report["logical_replay"] == [30, 10, 20] for report in restored_reports)


def test_publication_lowering_preserves_field_roles_and_present_halos() -> None:
    from phydrax.meshing._publication_lowering import (
        lower_accepted_publication,
        PublicationEntityRole,
        PublicationFieldRole,
    )

    lowered = lower_accepted_publication(
        "accepted-state",
        "owner-distribution",
        {"cell_history": np.asarray(((80, 81), (30, 31), (50, 51)), dtype=np.int64)},
        {"cell_history": np.asarray((8, 3, 5), dtype=np.int64)},
        {"cell_history": np.asarray((0, 1, 2), dtype=np.int32)},
        {"cell_history": PublicationEntityRole.CELL},
        {"cell_history": PublicationFieldRole.HISTORY},
        1,
        3,
        entity_present={"cell_history": np.asarray((False, False, True))},
    )
    assert lowered.rank == 1
    assert lowered.part_count == 3
    assert lowered.roles() == (
        ("cell_history", PublicationEntityRole.CELL, PublicationFieldRole.HISTORY),
    )
    np.testing.assert_array_equal(lowered.fields[0].entity_ids, (3, 5))
    np.testing.assert_array_equal(lowered.fields[0].values, ((30, 31), (50, 51)))
    np.testing.assert_array_equal(lowered.fields[0].owned, (True, False))


def test_owner_local_migration_registered_archive_rebuilds_canonical_forest(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        read_model_archive,
        read_model_artifact,
        restore_model_archive_value,
        write_model_archive,
        write_model_artifact,
    )

    _, state, _ = _mesh_state()
    forest = extract_simplex_forest(state)
    original = OwnerLocalMigrationState(
        forest,
        cell_payloads={"temperature": np.asarray(forest.cell_ids, np.float64) + 0.25},
        vertex_payloads={"history": np.asarray(forest.vertex_ids, np.int64)[:, None]},
    )
    path = write_model_artifact(
        tmp_path / "migration.phx", original, limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path,
        expected_type=OwnerLocalMigrationState,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.state_id == original.state_id
    assert restored.forest.forest_id == forest.forest_id
    assert restored.forest.next_cell_id == forest.next_cell_id
    assert restored.forest.next_vertex_id == forest.next_vertex_id
    for (_, actual), (_, expected) in zip(
        restored.forest.cell_fields, forest.cell_fields, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for (_, actual), (_, expected) in zip(
        restored.cell_payloads, original.cell_payloads, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    manifest["model_recipe"]["fields"]["forest"]["fields"]["forest_id"]["value"] = (
        "0" * 64
    )
    changed = write_model_archive(
        tmp_path / "changed.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            changed,
            expected_type=OwnerLocalMigrationState,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )
