from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._assembly import MeshAssembly, MeshPart, MeshPartManifest
from phydrax.meshing._distribution import prepare_owner_local_cell_shard
from phydrax.meshing._motion import (
    advance_mesh_motion,
    advance_owner_local_mesh_motion,
    MeshMotionAdvance,
    MeshMotionMonitor,
)
from phydrax.meshing._overset import (
    OversetConnectivity,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.solver._moving_overset_poisson import (
    _owner_local_coefficient_ids,
    _owner_local_reconstruction,
    prepare_moving_overset_poisson_update,
    PreparedMovingOversetPoissonUpdate,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="moving overset overlap requires meshcore"
)


def _square_part(name: str, triangles: tuple[tuple[int, int, int], ...]) -> MeshPart:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                f"{name}-triangles",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
            ),
        ),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _p1_cell_content(
    discretization: FiniteElementDiscretization, coefficients: Array, /
) -> np.ndarray:
    dof_map = discretization.dof_maps[discretization._field_index("u")]
    measures = np.asarray(discretization.measures[0].weights, dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    content = np.empty((measures.size,), dtype=np.float64)
    offset = 0
    for block_index, block in enumerate(discretization.mesh.blocks):
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = offset + block.cell_count
        content[offset:stop] = measures[offset:stop] * np.mean(values[cell_dofs], axis=1)
        offset = stop
    return content


def _part(name: str, points: tuple[tuple[float, float], ...]) -> MeshPart:
    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(((0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int64)
    mesh = CellMesh(
        coordinates,
        (CellBlock(f"{name}-triangles", "triangle", cells),),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _cell_result(part: MeshPart, /) -> CellMeshingResult:
    if not isinstance(part.carrier, CellMeshingResult):
        raise TypeError("This focused FE test requires a certified cell mesh.")
    return part.carrier


def _cell_geometry(result: CellMeshingResult, /) -> CellGeometrySpec:
    if not isinstance(result.geometry, CellGeometrySpec):
        raise TypeError("This focused FE test requires affine cell geometry.")
    return result.geometry


def _owner_view(part: MeshPart, /) -> OwnerLocalCellGeometry:
    if not isinstance(part.carrier, OwnerLocalCellGeometry):
        raise TypeError("The three-process fixture requires owner-local views.")
    return part.carrier


def _boundary_ids(mesh: CellMesh, facet_dimension: int, /) -> np.ndarray:
    entity_ids = np.asarray(mesh.entity_set(facet_dimension).entity_ids, dtype=np.int64)
    if isinstance(mesh.connectivity, SimplicialConnectivity):
        boundary = mesh.connectivity.boundary_masks[facet_dimension]
    elif isinstance(mesh.connectivity, PolygonalConnectivity) and facet_dimension == 1:
        boundary = mesh.connectivity.boundary_edges
    else:
        raise TypeError("The focused FE test requires simplex or polygon connectivity.")
    return entity_ids[np.asarray(boundary, dtype=np.bool_)]


def _owned_boundary_ids(view: OwnerLocalCellGeometry, /) -> np.ndarray:
    """Facet IDs of owned cells met by no other resident cell (exact face halo)."""
    _, facet_ids = view.entity_binding(1)
    facet_by_key = {
        tuple(sorted(int(row) for row in np.asarray(loop))): int(identifier)
        for identifier, loop in zip(
            np.asarray(facet_ids).tolist(),
            view.entity_vertex_loops(1, np.asarray(facet_ids)),
            strict=True,
        )
    }
    counts: dict[tuple[int, ...], int] = {}
    owned_keys: set[tuple[int, ...]] = set()
    for block in view.blocks:
        owned_ids = set(
            np.asarray(view.cell_ids, dtype=np.int64)[
                np.asarray(view.owned, dtype=np.bool_)
            ].tolist()
        )
        for identifier, vertices in zip(
            np.asarray(block.cell_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices, dtype=np.int64).tolist(),
            strict=True,
        ):
            for skip in range(len(vertices)):
                key = tuple(sorted(vertices[:skip] + vertices[skip + 1 :]))
                counts[key] = counts.get(key, 0) + 1
                if identifier in owned_ids:
                    owned_keys.add(key)
    return np.asarray(
        sorted(facet_by_key[key] for key in owned_keys if counts[key] == 1),
        dtype=np.int64,
    )


def _case(
    *, bound: bool = True
) -> tuple[
    OversetConnectivity,
    dict[str, FiniteElementPlan],
    Composition,
    MeshPart,
]:
    donor = _part(
        "donor",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.5, 0.5)),
    )
    receptor = _part(
        "receptor",
        ((0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.36, 0.36)),
    )
    receptor_result = _cell_result(receptor)
    receptor_mesh = receptor_result.mesh
    boundary_ids = _boundary_ids(receptor_mesh, 1)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("donor"),
            OversetPartSpec("receptor", boundary=receptor.scope(1, boundary_ids)),
        ),
        policy=OversetPolicy(fringe_layers=1),
        partition_priority=("receptor", "donor") if bound else None,
        partition_sources={} if bound else None,
    )
    plans: dict[str, FiniteElementPlan] = {}
    entries = list(connectivity.composition_entries())
    part_entries = {entry.entry_id: entry for entry in entries}
    for part in (donor, receptor):
        result = _cell_result(part)
        geometry = _cell_geometry(result)
        plan = FiniteElementPlan(
            result.mesh,
            FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
            coordinate_spec=geometry,
        )
        discretization = plan.prepare()
        plans[part.name] = plan
        field_entry = CompositionEntry(
            discretization,
            entry_id=f"fe:{part.name}",
            role="discretization",
            owner_id="moving-overset-test",
            structure_id=discretization.dof_maps[0].dof_map_id,
            revision_id=discretization.prepared_id,
            semantics_id=f"poisson-space:{part.name}",
            dependencies=(part_entries[f"overset:part:{part.name}"].binding("revision"),),
        )
        state = jnp.zeros(
            (discretization.dof_maps[0].global_dof_count,), dtype=jnp.float64
        )
        state_entry = CompositionEntry(
            state,
            entry_id=f"poisson:{part.name}",
            role="physical-state",
            owner_id="moving-overset-test",
            structure_id=field_entry.structure_id,
            revision_id=f"poisson:{part.name}:0",
            semantics_id="scalar-poisson-potential",
            dependencies=(field_entry.binding("revision"),),
        )
        entries.extend((field_entry, state_entry))
    return connectivity, plans, Composition(entries, boundary_id="accepted-0"), receptor


def _advance(
    part: MeshPart,
    shift: tuple[float, float],
    /,
    *,
    move_all: bool = False,
) -> MeshMotionAdvance:
    result = _cell_result(part)
    monitor = MeshMotionMonitor(result.mesh)
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64).copy()
    displacement = np.asarray(shift, dtype=np.float64)
    if move_all:
        coordinates += displacement
    else:
        coordinates[-1] += displacement
    return advance_mesh_motion(
        monitor,
        result,
        coordinates,
        boundary_residual=0.0,
        numeric_version="moving-overset-test",
    )


def test_serial_composite_rejects_unbound_connectivity() -> None:
    connectivity, plans, composition, receptor = _case(bound=False)
    advance = _advance(receptor, (0.01, 0.0))
    with pytest.raises(ValueError, match="connectivity-bound partition"):
        prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {"receptor": advance},
            plans,
            "u",
            part_priority=("receptor", "donor"),
            boundary_values=lambda points: jnp.zeros(points.shape[0]),
        )


def test_serial_composite_rejects_partition_input_mismatch() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (0.01, 0.0))
    boundary = lambda points: jnp.zeros(points.shape[0])
    with pytest.raises(ValueError, match="part_priority must match"):
        prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {"receptor": advance},
            plans,
            "u",
            part_priority=("donor", "receptor"),
            boundary_values=boundary,
        )
    cavity = np.asarray(((0.1, 0.1), (0.2, 0.1), (0.1, 0.2)))
    with pytest.raises(ValueError, match="cavities must match"):
        prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {"receptor": advance},
            plans,
            "u",
            part_priority=("receptor", "donor"),
            cavities=(cavity,),
            boundary_values=boundary,
        )


def test_moving_overset_poisson_commits_one_physical_partition() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (0.01, 0.0))

    update = prepare_moving_overset_poisson_update(
        composition,
        connectivity,
        {"receptor": advance},
        plans,
        "u",
        part_priority=("receptor", "donor"),
        boundary_values=lambda points: jnp.zeros(points.shape[0]),
    )

    assert isinstance(update, PreparedMovingOversetPoissonUpdate)
    assert update.accepted
    expected_content = {"donor": 1.875, "receptor": 0.125}
    assert {part.part_name for part in update.parts} == set(expected_content)
    for part in update.parts:
        assert not part.remap_complete
        assert part.conservative_remap is None
        assert part.overlap_evidence is None
        assert part.reanalysis_id
        assert part.source_content == pytest.approx(expected_content[part.part_name])
        assert part.target_content == pytest.approx(expected_content[part.part_name])
        assert part.overlap_defect is None
        assert part.linear_residual <= 1e-11
        assert part.boundary_flux + part.overset_flux == pytest.approx(
            -expected_content[part.part_name], abs=1e-9
        )
        assert part.flux_defect <= 1e-9
    assert update.partition is not None
    assert update.partition is update.candidate.bound_partition
    assert update.composite is not None
    assert update.interfaces is not None
    assert float(update.partition.total_area) == pytest.approx(2.0)
    assert update.composite.flux_defect <= 1e-9
    assert sum(part.overset_flux for part in update.parts) == pytest.approx(
        0.0, abs=1e-12
    )
    assert any(edge.kind == "inter_part" for edge in update.interfaces)
    receptor_part = next(part for part in update.parts if part.part_name == "receptor")
    assert abs(receptor_part.overset_flux) > 0.1

    rejected = update.commit(accepted_boundary=False)
    assert not rejected.published
    assert rejected.composition is composition
    accepted = update.commit(accepted_boundary=True)
    assert accepted.published
    assert accepted.composition.composition_id != composition.composition_id
    assert accepted.composition.entry("poisson:donor").value is not None
    assert accepted.composition.entry("poisson:receptor").value is not None


def test_composite_solution_uses_current_boundary_not_predecessor_receptors() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (0.01, 0.0), move_all=True)
    zero = prepare_moving_overset_poisson_update(
        composition,
        connectivity,
        {"receptor": advance},
        plans,
        "u",
        boundary_values=lambda points: jnp.zeros(points.shape[0]),
        part_priority=("receptor", "donor"),
    )
    one = prepare_moving_overset_poisson_update(
        composition,
        connectivity,
        {"receptor": advance},
        plans,
        "u",
        boundary_values=lambda points: jnp.ones(points.shape[0]),
        part_priority=("receptor", "donor"),
    )
    assert zero.update_id != one.update_id
    assert zero.accepted and one.accepted
    assert zero.partition is not None
    assert float(zero.partition.total_area) == pytest.approx(2.0)
    for first, second in zip(zero.parts, one.parts, strict=True):
        assert first.part_name == second.part_name
        assert np.max(np.abs(np.asarray(second.coefficients - first.coefficients))) > 0.9
    receipt = zero.commit(accepted_boundary=True)
    assert receipt.published
    with pytest.raises(ValueError, match="current overset source result"):
        prepare_moving_overset_poisson_update(
            receipt.composition,
            zero.candidate,
            {"receptor": advance},
            plans,
            "u",
            boundary_values=lambda points: jnp.zeros(points.shape[0]),
            part_priority=("receptor", "donor"),
        )
    successor_parts = {part.name: part for part in zero.candidate.assembly.parts}
    successor_plans = {}
    for name, part in successor_parts.items():
        result = _cell_result(part)
        old = plans[name]
        successor_plans[name] = FiniteElementPlan(
            result.mesh,
            old.fields,
            precision_policy=old.precision_policy,
            coordinate_spec=result.geometry,
            coefficient_dtype=old.coefficient_dtype,
            coordinate_measure=old.coordinate_measure,
        )
    second = prepare_moving_overset_poisson_update(
        receipt.composition,
        zero.candidate,
        {
            "donor": _advance(successor_parts["donor"], (0.002, 0.0)),
            "receptor": _advance(
                successor_parts["receptor"], (0.005, 0.0), move_all=True
            ),
        },
        successor_plans,
        "u",
        boundary_values=lambda points: jnp.ones(points.shape[0]),
        part_priority=("receptor", "donor"),
    )
    assert second.previous.connectivity_id == zero.candidate.connectivity_id
    assert second.accepted
    assert second.commit(accepted_boundary=True).published


def test_composite_boundary_support_distinguishes_cavity_and_unmatched_edges() -> None:
    from phydrax.geometry._overset_partition import partition_overset_triangles
    from phydrax.solver._moving_overset_poisson import (
        _require_physical_partition_boundary,
    )

    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    background = ("background", points, triangles)
    cavity = np.asarray(((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)))
    partition = partition_overset_triangles((background,), (cavity,))
    interfaces = _require_physical_partition_boundary(partition, background, (cavity,))
    assert float(partition.total_area) == pytest.approx(0.75)
    assert any(edge.kind == "physical_boundary" for edge in interfaces)

    rogue = (
        "rogue",
        np.asarray(((1.0, 0.0), (1.5, 0.5), (1.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    partition = partition_overset_triangles((rogue, background), ())
    with pytest.raises(ValueError, match="unmatched edge"):
        _require_physical_partition_boundary(partition, background, ())


def test_moving_overset_orphan_refuses_before_composition_publication() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (5.0, 0.0), move_all=True)
    accepted_id = composition.composition_id

    with pytest.raises(OversetConnectivityError, match="incomplete"):
        prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {"receptor": advance},
            plans,
            "u",
        )

    assert composition.composition_id == accepted_id


def _square_part(name: str, triangles: tuple[tuple[int, int, int], ...]) -> MeshPart:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                f"{name}-triangles",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
            ),
        ),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _p1_cell_content(
    discretization: FiniteElementDiscretization, coefficients: Array, /
) -> np.ndarray:
    dof_map = discretization.dof_maps[discretization._field_index("u")]
    measures = np.asarray(discretization.measures[0].weights, dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    content = np.empty((measures.size,), dtype=np.float64)
    offset = 0
    for block_index, block in enumerate(discretization.mesh.blocks):
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = offset + block.cell_count
        content[offset:stop] = measures[offset:stop] * np.mean(values[cell_dofs], axis=1)
        offset = stop
    return content


def test_overset_interpolation_is_distinct_from_conservative_overlap() -> None:
    donor = _square_part("square-donor", ((0, 1, 2), (0, 2, 3)))
    receptor = _square_part("square-receptor", ((0, 1, 3), (1, 2, 3)))
    receptor_result = _cell_result(receptor)
    receptor_mesh = receptor_result.mesh
    boundary_ids = _boundary_ids(receptor_mesh, 1)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec(donor.name),
            OversetPartSpec(receptor.name, boundary=receptor.scope(1, boundary_ids)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    donor_result = _cell_result(donor)
    donor_discretization = FiniteElementPlan(
        donor_result.mesh,
        field,
        coordinate_spec=_cell_geometry(donor_result),
    ).prepare()
    receptor_discretization = FiniteElementPlan(
        receptor_mesh,
        field,
        coordinate_spec=_cell_geometry(receptor_result),
    ).prepare()

    def bilinear(points: Array, _args: object) -> Array:
        return points[..., 0] * points[..., 1]

    source_coefficients = donor_discretization.project("u", bilinear)
    interpolation = prepare_overset_field_transfer(
        connectivity, {donor.name: donor_discretization}, "u"
    )
    receptor_values = interpolation.apply({donor.name: source_coefficients})[
        receptor.name
    ]
    receptor_evidence = connectivity.receptors_of(receptor.name)
    receptor_dofs: dict[int, int] = {}
    receptor_dof_map = receptor_discretization.dof_maps[0]
    for block_index, block in enumerate(receptor_mesh.blocks):
        for vertex_row, dof_row in zip(
            np.asarray(block.vertices, dtype=np.int64).reshape(-1),
            np.asarray(receptor_dof_map.cell_dofs[block_index], dtype=np.int64).reshape(
                -1
            ),
            strict=True,
        ):
            identifier = int(receptor_mesh.vertex_global_ids[vertex_row])
            receptor_dofs.setdefault(identifier, int(dof_row))
    interpolated_coefficients = jnp.zeros(
        (receptor_dof_map.global_dof_count,), dtype=jnp.float64
    )
    for identifier, value in zip(
        np.asarray(receptor_evidence.receptor_ids, dtype=np.int64),
        np.asarray(receptor_values).reshape((-1,)),
        strict=True,
    ):
        interpolated_coefficients = interpolated_coefficients.at[
            receptor_dofs[int(identifier)]
        ].set(value)

    remap = prepare_overset_conservative_remap(donor, receptor)
    source_cell_content = _p1_cell_content(donor_discretization, source_coefficients)
    conservative_target = remap.apply_extensive(source_cell_content)
    interpolated_target_content = float(
        np.sum(_p1_cell_content(receptor_discretization, interpolated_coefficients))
    )

    assert not interpolation.conservative
    assert remap.complete
    assert float(np.sum(source_cell_content)) == pytest.approx(1.0 / 3.0)
    assert float(np.sum(np.asarray(conservative_target))) == pytest.approx(1.0 / 3.0)
    assert interpolated_target_content == pytest.approx(1.0 / 6.0)
    assert not np.isclose(
        interpolated_target_content,
        float(np.sum(source_cell_content)),
        rtol=0.0,
        atol=1e-12,
    )


def _grid_part(name: str, lower: float, upper: float, count: int, /) -> MeshPart:
    axis = np.linspace(lower, upper, count + 1)
    points = np.asarray([(x, y) for y in axis for x in axis], dtype=np.float64)
    triangles = []
    for j in range(count):
        for i in range(count):
            first = j * (count + 1) + i
            above = first + count + 1
            triangles.extend(((first, first + 1, above + 1), (first, above + 1, above)))
    mesh = CellMesh(
        points,
        (CellBlock(f"{name}-triangles", "triangle", np.asarray(triangles, np.int64)),),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _manufactured(points: np.ndarray, /) -> np.ndarray:
    """Nonconstant accepted state; its xy term is not reproduced by P1 transfer."""
    return 1.0 + points[:, 0] + 2.0 * points[:, 1] + 0.5 * points[:, 0] * points[:, 1]


def _sharded_part(
    serial: MeshPart, execution_group: ExecutionGroup, rank: int, /
) -> tuple[MeshPart, MeshPartManifest]:
    mesh = _cell_result(serial).mesh
    centroids = np.mean(
        np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)], axis=1
    )
    lower = float(np.min(centroids[:, 0]))
    width = float(np.max(centroids[:, 0])) - lower
    owners = np.minimum(((centroids[:, 0] - lower) / width * 3.0).astype(np.int32), 2)
    distribution = phx.meshing.MeshDistribution(
        serial, phx.discretization.CellPartition(owners, 3), halo_width=1
    )
    assembly = prepare_owner_local_cell_shard(
        serial.name,
        serial.name,
        serial.part_id,
        serial.coordinate_contract,
        mesh.blocks[0].name,
        "triangle",
        distribution.owner_local_cell_shard_rows(rank),
        execution_group,
        axis_name=f"poisson_shard_{serial.name}",
    )
    (part,) = assembly.parts
    (manifest,) = assembly.part_manifest
    return part, manifest


def _owner_local_poisson_case(
    execution_group: ExecutionGroup, rank: int
) -> tuple[
    OversetConnectivity,
    dict[str, FiniteElementPlan],
    Composition,
    dict[str, MeshPart],
    dict[str, MeshPart],
]:
    serial = {
        "owner-poisson-donor": _grid_part("owner-poisson-donor", 0.0, 2.0, 6),
        "owner-poisson-receptor": _grid_part("owner-poisson-receptor", 0.55, 1.35, 4),
    }
    sharded = {
        name: _sharded_part(part, execution_group, rank) for name, part in serial.items()
    }
    parts = {name: value[0] for name, value in sharded.items()}
    assembly = MeshAssembly(
        tuple(parts.values()),
        global_parts=tuple(value[1] for value in sharded.values()),
        local_rank=rank,
    )
    receptor_view = _owner_view(parts["owner-poisson-receptor"])
    # Owner-local scopes declare each global face once: on the owner of the
    # cell carrying that boundary face.
    boundary_ids = _owned_boundary_ids(receptor_view)
    specifications = (
        OversetPartSpec("owner-poisson-donor", owner_rank=0),
        OversetPartSpec(
            "owner-poisson-receptor",
            boundary=parts["owner-poisson-receptor"].scope(1, boundary_ids),
            owner_rank=1,
        ),
    )
    prepared = prepare_overset_connectivity(
        assembly,
        specifications,
        policy=OversetPolicy(fringe_layers=1),
        execution_group=execution_group,
        axis_name="moving_owner_local_poisson_initial",
    )
    field_spec = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    plans: dict[str, FiniteElementPlan] = {}
    discretizations: dict[str, FiniteElementDiscretization] = {}
    reconstructions: dict[str, Any] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    for part in parts.values():
        view = _owner_view(part)
        mesh, geometry = view.reconstruct_local_geometry()
        if mesh is None or geometry is None:
            raise ValueError(f"Owner-local part {part.name!r} must be non-empty.")
        plan = FiniteElementPlan(mesh, field_spec, coordinate_spec=geometry)
        discretization = plan.prepare()
        plans[part.name] = plan
        discretizations[part.name] = discretization
        reconstructions[part.name] = _owner_local_reconstruction(
            view, discretization, "u"
        )
        coefficient_ids[part.name] = _owner_local_coefficient_ids(discretization, "u")
    # Owner-local donor coverage is proved by the collective donor search of the
    # field exchange; only the resolved connectivity publishes entries.
    connectivity = (
        prepare_overset_field_transfer(prepared, reconstructions, "u")
        .prepare_owner_local_exchange(
            coefficient_ids,
            execution_group,
            {name: _owner_view(part) for name, part in parts.items()},
            axis_name="moving_owner_local_poisson_initial_field",
        )
        .connectivity
    )
    entries = list(connectivity.composition_entries())
    part_entries = {entry.entry_id: entry for entry in entries}
    for name, discretization in discretizations.items():
        fe_entry = CompositionEntry(
            discretization,
            entry_id=f"fe:{name}",
            role="discretization",
            owner_id="moving-owner-local-test",
            structure_id=discretization.dof_maps[0].dof_map_id,
            revision_id=discretization.prepared_id,
            semantics_id=f"moving-owner-local-space:{name}",
            dependencies=(part_entries[f"overset:part:{name}"].binding("revision"),),
        )
        state_entry = CompositionEntry(
            jnp.asarray(
                _manufactured(np.asarray(discretization.dof_maps[0].dof_coordinates))
            ),
            entry_id=f"poisson:{name}",
            role="physical-state",
            owner_id="moving-owner-local-test",
            structure_id=fe_entry.structure_id,
            revision_id=f"moving-owner-local-state:{name}:0",
            semantics_id="scalar-poisson-potential",
            dependencies=(fe_entry.binding("revision"),),
        )
        entries.extend((fe_entry, state_entry))
    return (
        connectivity,
        plans,
        Composition(entries, boundary_id="moving-owner-local-poisson-accepted"),
        parts,
        serial,
    )


_MOVED_VERTICES = {
    "owner-poisson-donor": (np.asarray((1.0, 1.0)), np.asarray((0.01, 0.0))),
    "owner-poisson-receptor": (np.asarray((0.95, 0.95)), np.asarray((0.0, 0.005))),
}


def _moved(points: np.ndarray, name: str, /) -> tuple[np.ndarray, np.ndarray]:
    anchor, shift = _MOVED_VERTICES[name]
    moved = np.all(np.isclose(points, anchor, rtol=0.0, atol=1e-12), axis=1)
    result = points.copy()
    result[moved] += shift
    return result, moved


def _locate_p1(
    points: np.ndarray, triangles: np.ndarray, values: np.ndarray, query: np.ndarray, /
) -> float:
    for triangle in triangles:
        corners = points[triangle]
        matrix = np.stack((corners[1] - corners[0], corners[2] - corners[0]), axis=1)
        local = np.linalg.solve(matrix, query - corners[0])
        weights = np.asarray((1.0 - local[0] - local[1], local[0], local[1]))
        if np.all(weights >= -1e-12):
            return float(weights @ values[triangle])
    raise ValueError("Receptor point lies outside the donor mesh.")


def _serial_poisson(
    points: np.ndarray, triangles: np.ndarray, dirichlet: dict[int, float], /
) -> np.ndarray:
    """Independent dense P1 oracle for -Laplace(u) = 1 with Dirichlet rows."""
    count = points.shape[0]
    stiffness = np.zeros((count, count))
    load = np.zeros((count,))
    reference = np.asarray(((-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)))
    for triangle in triangles:
        corners = points[triangle]
        jacobian = np.stack((corners[1] - corners[0], corners[2] - corners[0]), axis=1)
        area = 0.5 * abs(np.linalg.det(jacobian))
        gradients = np.linalg.inv(jacobian).T @ reference
        stiffness[np.ix_(triangle, triangle)] += area * gradients.T @ gradients
        load[triangle] += area / 3.0
    fixed = np.asarray(sorted(dirichlet), dtype=np.int64)
    free = np.setdiff1d(np.arange(count), fixed)
    solution = np.zeros((count,))
    solution[fixed] = [dirichlet[int(row)] for row in fixed]
    solution[free] = np.linalg.solve(
        stiffness[np.ix_(free, free)],
        load[free] - stiffness[np.ix_(free, fixed)] @ solution[fixed],
    )
    return solution


def _gathered_receptor_ids(update: PreparedMovingOversetPoissonUpdate) -> set[int]:
    from jax.experimental import multihost_utils

    evidence = update.candidate.receptors_of("owner-poisson-receptor")
    donor_index = update.candidate.part_index("owner-poisson-donor")
    local = np.asarray(
        [
            int(identifier)
            for identifier, found, part in zip(
                np.asarray(evidence.receptor_ids).tolist(),
                np.asarray(evidence.found).tolist(),
                np.asarray(evidence.donor_parts).tolist(),
                strict=True,
            )
            if found and part == donor_index
        ],
        dtype=np.int64,
    )
    counts = np.asarray(
        multihost_utils.process_allgather(np.asarray((local.size,), np.int64))
    ).reshape((-1,))
    padded = np.full((int(np.max(counts)),), -1, dtype=np.int64)
    padded[: local.size] = local
    gathered = np.asarray(multihost_utils.process_allgather(padded)).reshape(-1)
    return {int(value) for value in gathered if value >= 0}


def test_owner_local_moving_overset_poisson_commits_collectively() -> None:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    process_ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or process_ranks != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip("This route requires three native processes with one device each.")
    execution_group = ExecutionGroup(
        ExecutionGroupSpec(
            "moving-owner-local-poisson",
            process_ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("overset", 3),),
        ),
        devices,
    )
    rank = process_ranks.index(jax.process_index())
    previous, plans, composition, source_parts, serial = _owner_local_poisson_case(
        execution_group, rank
    )
    for part in source_parts.values():
        view = _owner_view(part)
        assert not view.serial_source_view
        assert np.count_nonzero(np.asarray(view.owned)) < sum(
            1
            for _ in np.asarray(_cell_result(serial[part.name]).mesh.blocks[0].global_ids)
        )
    advances: dict[str, MeshMotionAdvance] = {}
    target_parts: list[MeshPart] = []
    for index, (name, part) in enumerate(sorted(source_parts.items())):
        view = _owner_view(part)
        coordinates, moved = _moved(np.asarray(view.coordinates, dtype=np.float64), name)
        advance = advance_owner_local_mesh_motion(
            view,
            coordinates,
            execution_group,
            boundary_residual=0.0,
            fixed_vertices=~moved,
            driver_id=f"moving-owner-local-poisson-{name}",
            axis_name=f"moving_owner_local_poisson_motion_{index}",
        )
        assert advance.accepted and advance.local_result is not None, tuple(
            assessment.failure_detail for assessment in advance.assessments
        )
        advances[name] = advance
        target_parts.append(
            MeshPart(
                name,
                advance.local_result,
                coordinate_contract=part.coordinate_contract,
            )
        )
    successor_assembly = MeshAssembly(
        tuple(target_parts),
        global_parts=tuple(MeshPartManifest.from_part(part) for part in target_parts),
        local_rank=rank,
    )

    update = prepare_moving_overset_poisson_update(
        composition,
        previous,
        advances,
        plans,
        "u",
        execution_group=execution_group,
        successor_assembly=successor_assembly,
    )

    assert update.accepted
    assert update.candidate.complete
    assert {part.part_name for part in update.parts} == set(previous.global_part_names)
    for part in update.parts:
        assert part.remap_complete
        assert part.overlap_defect is not None
        assert part.overlap_defect <= 1e-9
        assert part.boundary_flux + part.overset_flux == pytest.approx(
            -part.target_content, abs=1e-9
        )
        assert part.flux_defect <= 1e-9

    # Independent serial oracle over the moved global meshes.
    receptor_ids = _gathered_receptor_ids(update)
    assert receptor_ids
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, part in serial.items():
        mesh = _cell_result(part).mesh
        source_points = np.asarray(mesh.coordinates, dtype=np.float64)
        target_points, _ = _moved(source_points, name)
        targets[name] = (
            source_points,
            target_points,
            np.asarray(mesh.blocks[0].vertices, dtype=np.int64),
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        )
    donor_source, donor_target, donor_cells, _ = targets["owner-poisson-donor"]
    donor_state = _manufactured(donor_source)
    expected: dict[str, dict[int, float]] = {}
    for name, (source_points, target_points, cells, vertex_ids) in targets.items():
        boundary_rows = set()
        mesh = _cell_result(serial[name]).mesh
        for row in np.flatnonzero(np.asarray(mesh.boundary_masks[0], dtype=np.bool_)):
            boundary_rows.add(int(row))
        dirichlet = {
            row: float(_manufactured(source_points[row : row + 1])[0])
            for row in boundary_rows
        }
        if name == "owner-poisson-receptor":
            for row, identifier in enumerate(vertex_ids.tolist()):
                if identifier in receptor_ids:
                    dirichlet[row] = _locate_p1(
                        donor_target, donor_cells, donor_state, target_points[row]
                    )
        solution = _serial_poisson(target_points, cells, dirichlet)
        expected[name] = dict(zip(vertex_ids.tolist(), solution.tolist(), strict=True))
    for part in update.parts:
        identifiers = _owner_local_coefficient_ids(part.discretization, "u")
        np.testing.assert_allclose(
            np.asarray(part.coefficients),
            [expected[part.part_name][int(identifier)] for identifier in identifiers],
            rtol=0.0,
            atol=1e-9,
        )

    rejected = update.commit(accepted_boundary=False)
    assert not rejected.published
    assert rejected.composition is composition
    accepted = update.commit(accepted_boundary=True)
    assert accepted.published
    assert accepted.composition.composition_id != composition.composition_id
