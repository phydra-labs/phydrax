"""Check integrable varying-frame extraction against an independent source oracle."""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._domain_coverage import certify_domain_coverage
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.meshing import CellMeshingResult, NativeMappedHexSource
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._hex_frame_map import (
    _extract_source_block_tensor_grid,
    _extract_source_tensor_frame_grid,
    chart_face_permutation,
    extract_source_block_integer_grid,
    extract_source_frame_grid,
    prepare_hex_frame_holonomy,
    prepare_source_block_connection,
    prepare_source_hex_frame,
)
from phydrax.meshing._hex_frame_records import (
    _SourceBlockTensorGrid,
    PreparedSourceHexFrame,
    SourceBlockIntegerGrid,
)
from phydrax.meshing._hex_generation import (
    MappedGridHexConstruction,
    NativeHexGridSchedule,
)
from phydrax.meshing.qualification import (
    check_stationary_euler,
    curved_occupied_source,
    family_transport_field_templates,
    solve_linear_diffusion,
)
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.hex_fields import allfield_inventory, allfield_state
from tests._support.hex_sources import (
    sheared_hex_specification,
    sheared_mapped_hex_source,
)


def _realize_test_grid(
    prepared: PreparedSourceHexFrame,
    grid: SourceBlockIntegerGrid | _SourceBlockTensorGrid,
    regions: np.ndarray,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> MappedGridHexConstruction:
    if isinstance(grid, _SourceBlockTensorGrid):
        return _extract_source_tensor_frame_grid(
            prepared, grid, regions, limits, scope=scope, budget=budget
        )
    return extract_source_frame_grid(
        prepared, grid, regions, limits, scope=scope, budget=budget
    )


# Preserve the mapped Q2 example's fixed empirical solver acceptance limits.
_DIFFUSION_ERROR_LIMIT = 1.0e-8
_EULER_RESIDUAL_LIMIT = 1.0e-10


type ShearedHexPublication = tuple[
    NativeMappedHexSource,
    SourceBlockIntegerGrid,
    MappedGridHexConstruction,
    CellMeshingResult,
]

@pytest.mark.parametrize(
    ("partition", "dtype"),
    (((1, 2, 5), np.int16), ((2, 3, 4), np.int32), ((5, 2, 1), np.int64)),
)
def test_compact_tensor_storage_bounds_cover_real_owners(
    partition: tuple[int, int, int], dtype: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from phydrax._host_storage import retained_host_storage_profile
    from phydrax.meshing import _hex_frame_map as owner
    from phydrax.meshing._hex_frame_records import SourceIntegerGridCounts

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    counts = np.asarray((partition, partition), dtype=dtype)
    cells = sum(int(np.prod(row)) for row in counts)
    vertices = sum(int(np.prod(row + 1)) for row in counts)
    external, managed = owner._source_tensor_storage_bounds(
        tuple(tuple(int(n) for n in row) for row in counts)
    )
    observed_external: list[int] = []
    observed_managed: list[int] = []
    original = owner._fill_tensor_cells

    def observe(nodes: np.ndarray, cells: np.ndarray, *, scope: NativeExecutionScope) -> None:
        frame = sys._getframe(1)
        local = frame.f_locals
        roots = tuple(local[name] for name in (
            "shared", "terms", "matrix", "origin", "corner_rows", "point", "support",
        ))
        _, declared, logical, _ = retained_host_storage_profile(roots, {}, lambda _: False)
        observed_external.append(declared + logical)
        observed_managed.append(sum(local[name].nbytes for name in (
            "points", "cells", "parents", "local",
        )))
        original(nodes, cells, scope=scope)

    monkeypatch.setattr(owner, "_fill_tensor_cells", observe)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        admission = SourceIntegerGridCounts(
            counts, np.asarray(source.domain.cell_regions),
            np.asarray(source.domain.reference_mesh.vertex_global_ids),
            cells, vertices, 0,
        )
        with scope.host_workspace() as workspace:
            workspace.set_bound(external)
            banks = owner._prepare_source_tensor_banks(prepared, admission, scope=scope)
            assert banks.cells.shape == (cells, 8)
            assert banks.points.dtype == np.dtype(np.float64)
            assert banks.cells.dtype == banks.parent_rows.dtype == np.dtype(np.int64)
    assert len(observed_external) == 2
    assert max(observed_external) <= external
    assert max(observed_managed) == managed


@pytest.mark.parametrize("dtype", (np.uint8, np.float64, np.longdouble))
def test_compact_tensor_storage_child_exact_capacity_and_one_byte_refusal(dtype: Any) -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionLimits
    from phydrax.meshing import _hex_frame_map as owner
    from phydrax.meshing._hex_frame_records import SourceIntegerGridCounts

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
    counts = np.asarray(((2, 3, 4),), dtype=np.int64)
    admission = SourceIntegerGridCounts(
        counts, np.asarray(source.domain.cell_regions),
        np.asarray(source.domain.reference_mesh.vertex_global_ids), 24, 60, 0,
    )
    external, managed = owner._source_tensor_storage_bounds(((2, 3, 4),))

    def run(capacity: int) -> int:
        limits = NativeExecutionLimits(100000, 0, 0, 100000000, float("inf"))
        with NativeExecutionScope(limits) as scope:
            with scope.batch(100000, bytes_upper=capacity) as parent:
                with parent.batch(100000, bytes_upper=capacity) as child:
                    with child.native_target(), scope.host_workspace() as workspace:
                        workspace.set_bound(external)
                        sibling = scope.allocate_host_array((3, 5), dtype)
                        assert sibling.ctypes.data % sibling.dtype.alignment == 0
                        banks = owner._prepare_source_tensor_banks(
                            prepared, admission, scope=scope,
                        )
                        required = child.managed_bytes
                        del banks, sibling
                    return required

    # Discover only the canonical native owners' metadata contribution. The
    # external and array payload bounds above are analytic, not fitted to RSS.
    required = run(10000000)
    assert required >= external + managed + 15 * np.dtype(dtype).itemsize
    assert run(required) == required
    with pytest.raises(MeshcoreError) as refused:
        run(required - 1)
    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    with pytest.raises(MeshcoreError) as external_refused:
        run(external - 1)
    assert external_refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED

def test_compact_tensor_combined_bound_refuses_before_bank_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _hex_frame_map as owner

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    required = sum(owner._source_tensor_storage_bounds(((2, 3, 4),)))
    limits = MeshingLimits(maximum_scratch_bytes=required - 1)
    entered = False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal entered
        entered = True
        raise AssertionError("Tensor banks were allocated before combined admission")

    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        monkeypatch.setattr(owner, "_prepare_source_tensor_banks", forbidden)
        with pytest.raises(MeshingFailure):
            _extract_source_block_tensor_grid(
                prepared, np.asarray(((2, 3, 4),), dtype=np.int64),
                source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
            )
    assert not entered


@pytest.mark.parametrize("curvature", (0, 1))
def test_source_corner_frames_retire_exact_jet_scratch(curvature: int) -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import Polynomial
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._hex_frame_map import _fill_source_corner_frames

    polynomials: tuple[Polynomial, ...] = (
        {
            (1, 0, 0): Fraction(1),
            **({(2, 1, 0): Fraction(curvature, 8)} if curvature else {}),
        },
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    frames = np.empty((8, 3, 3), dtype=np.float64)
    expected = np.asarray(
        [
            (
                (1 + curvature * u * v / 4, curvature * u * u / 8, 0),
                (0, 1, 0),
                (0, 0, 1),
            )
            for u, v, _ in reference_cell_topology("hexahedron").vertices
        ],
        dtype=np.float64,
    )
    with coordinate_reference_budget() as budget:
        before_temporary = budget.temporary_bytes_upper
        before_retained = budget.retained_basis_bytes
        for _ in range(2):
            before_work = budget.work_units
            varying = _fill_source_corner_frames(polynomials, frames, budget=budget)
            assert varying is bool(curvature)
            np.testing.assert_array_equal(frames, expected)
            assert budget.work_units > before_work
            assert budget.temporary_bytes_upper == before_temporary
            assert budget.retained_basis_bytes == before_retained


def test_source_corner_frames_release_scratch_after_resource_refusal() -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing._hex_frame_map import _fill_source_corner_frames

    frames = np.full((8, 3, 3), np.nan, dtype=np.float64)
    with coordinate_reference_budget(MeshingLimits(maximum_work_units=1)) as budget:
        before_temporary = budget.temporary_bytes_upper
        with pytest.raises(CoordinateEnclosureResourceError):
            _fill_source_corner_frames(
                ({(1, 0, 0): Fraction(1), (2, 1, 0): Fraction(1, 8)},),
                frames,
                budget=budget,
            )
        assert budget.temporary_bytes_upper == before_temporary
        assert np.all(np.isnan(frames))


def test_intermediate_source_grid_retires_consumed_validity_preparations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as owner

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        original = owner.certify_cell_geometry_validity
        retired = []
        original_restriction = owner._certify_restriction_cell_geometry_validity
        restrictions = []
        predecessors = []

        def capture(*args: Any, **kwargs: Any) -> Any:
            before = set(budget.validity_certificate_cache)
            result = original(*args, **kwargs)
            assert result.all_certified
            retired.extend(set(budget.validity_certificate_cache) - before)
            predecessors.append(kwargs.get("predecessor"))
            return result

        def capture_restriction(*args: Any, **kwargs: Any) -> Any:
            before = set(budget.validity_certificate_cache)
            result = original_restriction(*args, **kwargs)
            assert result.all_certified
            retired.extend(set(budget.validity_certificate_cache) - before)
            restrictions.append(result.certificate_id)
            return result

        monkeypatch.setattr(owner, "certify_cell_geometry_validity", capture)
        monkeypatch.setattr(
            owner, "_certify_restriction_cell_geometry_validity", capture_restriction
        )
        before_work = budget.work_units
        grid = extract_source_block_integer_grid(
            prepared, np.full((1, 3), 2, dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6,
            scope=scope, budget=budget,
        )
        assert sum(block.cell_count for block in grid.mesh.blocks) == 8
        assert retired
        assert predecessors == [None]
        assert len(restrictions) == 1
        assert len(grid.mesh.blocks) == 1
        assert all(key not in budget.validity_certificate_cache for key in retired)
        assert budget.work_units > before_work


@pytest.mark.parametrize(
    "occupied,materials",
    (
        (
            {(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 2, 0), (1, 1, 1), (1, 1, 2)},
            False,
        ),
        (set(product(range(3), repeat=3)) - {(1, 1, 1)}, True),
    ),
)
def test_original_nonconstant_frame_has_exact_embedded_hex_coverage(
    occupied: set[tuple[int, int, int]], materials: bool
) -> None:
    source = curved_occupied_source(occupied, materials=materials)
    domain = source.domain
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(prepared.varying_roots)
        assert np.any(prepared.corner_frames[:, 1:] != prepared.corner_frames[:, :1])
        assert prepared.face_connections.shape[0] > 0
        before = prepared.frame_id
        limits = MeshingLimits()
        counts = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
        grid = extract_source_block_integer_grid(
            prepared,
            counts,
            domain.cell_regions,
            limits,
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            limits,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert prepared.frame_id == before
        assert {block.cell_kind for block in mapped.mesh.blocks} == {"hexahedron"}
        validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
        embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        fidelity = certify_source_fidelity(
            mapped.mesh,
            mapped.geometry,
            MappedDomainBoundarySource(domain, mapped.cell_regions),
            tolerance=0.0,
        )
        assert embedding.status == coverage.status == fidelity.status == "certified"
        assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0


def test_incompatible_chart_face_parity_refuses_without_changing_identity() -> None:

    authored = (37, 11, 29, 43)
    assert chart_face_permutation(authored, (29, 11, 37, 43)) == (2, 1, 0, 3)
    with pytest.raises(MeshingFailure, match="source_frame.boundary_parity"):
        chart_face_permutation(authored, (11, 29, 43, 37))
    assert authored == (37, 11, 29, 43)


def test_original_source_cycle_has_exact_singularity_witness_and_tree_cut() -> None:

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    mesh = source.domain.reference_mesh
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    edges = np.asarray(
        [
            (left, right)
            for left in range(len(cells))
            for right in range(left + 1, len(cells))
            if len(set(cells[left]) & set(cells[right])) == 4
        ],
        dtype=np.int64,
    )
    actions = np.broadcast_to(np.eye(3, dtype=np.int64), (len(edges), 3, 3)).copy()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        regular = prepare_hex_frame_holonomy(mesh, edges, actions, scope=execution_scope)
        assert regular.cycle_edges.size == 1
        assert regular.singular_cycles.size == 0
        actions[regular.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        singular = prepare_hex_frame_holonomy(mesh, edges, actions, scope=execution_scope)
        assert singular.singular_cycles.tolist() == [0]
        assert not np.array_equal(singular.cycle_holonomy[0], np.eye(3, dtype=np.int64))
        assert singular.spanning_edges.size == len(cells) - 1
        assert regular.singular_cycles.size == 0


def test_block_extraction_capacity_refusal_preserves_prepared_and_requested_state() -> (
    None
):

    source = curved_occupied_source({(0, 0, 0)})
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        counts = np.full((1, 3), 2, dtype=np.int64)
        frames = prepared.corner_frames.copy()
        with pytest.raises(MeshingFailure, match="cells"):
            extract_source_block_integer_grid(
                prepared,
                counts,
                source.domain.cell_regions,
                MeshingLimits(maximum_cells=1),
                10,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert np.array_equal(counts, np.full((1, 3), 2))
        assert np.array_equal(prepared.corner_frames, frames)


@pytest.fixture(scope="module")
def sheared_hex_publication() -> Iterator[ShearedHexPublication]:
    source = sheared_mapped_hex_source()
    domain = source.domain
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        _,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        requested = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
        requested[0, 1] = 4
        grid = extract_source_block_integer_grid(
            prepared,
            requested,
            domain.cell_regions,
            MeshingLimits(),
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        request = sheared_hex_specification(source, prepared)
        plan = phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("mapped_frame_grid_hex")
        ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        yield source, grid, mapped, plan.execute()


def test_sheared_authored_reference_charts_do_not_need_one_orthogonal_grid_frame(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    source, grid, mapped, _ = sheared_hex_publication
    assert np.all(grid.root_intervals[:, 1] == 4)
    validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
    embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
    coverage = certify_domain_coverage(
        mapped.mesh,
        mapped.geometry,
        source.domain,
        mapped.cell_regions,
        embedding=embedding,
    )
    assert embedding.status == coverage.status == "certified"


def test_sheared_hex_publication_retains_materials_and_interface(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    _, _, _, result = sheared_hex_publication
    assert result.audit.passed and result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert {zone.material_id for zone in result.zones} == {
        "material-left",
        "material-right",
    }
    assert "original-material-interface" in {patch.name for patch in result.patches}


def test_sheared_hex_publication_has_zero_two_sided_fidelity(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    source, _, _, result = sheared_hex_publication
    domain = source.domain
    root_ids = np.asarray(domain.reference_mesh.topology.entities(3).entity_ids)
    ancestry = result.geometry.restriction_source
    assert ancestry is not None
    root_regions = dict(zip(root_ids.tolist(), domain.cell_regions.tolist(), strict=True))
    public_regions = np.asarray(
        [
            root_regions[int(parent)]
            for block in result.mesh.blocks
            for parent in np.asarray(ancestry.block_parent_cell_ids[block.name])
        ],
        dtype=np.int64,
    )
    fidelity = certify_source_fidelity(
        result.mesh,
        result.geometry,
        MappedDomainBoundarySource(domain, public_regions),
        tolerance=0.0,
    )
    assert fidelity.status == "certified"
    assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0


def test_sheared_hex_publication_supports_q2_diffusion(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    assert solve_linear_diffusion(sheared_hex_publication[3], 2) < _DIFFUSION_ERROR_LIMIT


def test_sheared_hex_publication_supports_stationary_euler(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    assert check_stationary_euler(sheared_hex_publication[3]) < _EULER_RESIDUAL_LIMIT


def test_sheared_hex_publication_retains_scalar_form_and_material_inventory(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    result = sheared_hex_publication[3]
    fields = allfield_state(result)
    inventory = allfield_inventory(result, fields)
    assert set(fields) == {"H1", "DG", "Hcurl", "Hdiv", "material_content"}
    for kind in ("H1", "DG", "Hcurl", "Hdiv"):
        form = inventory[kind]
        assert "components" in form, kind
        assert form["components"] == 2, kind
    assert fields["material_content"].shape[1] == 5
    material = inventory["material_content"]
    assert "region_inventories" in material
    regions = material["region_inventories"]
    assert isinstance(regions, list)
    assert len(regions) == 2


def test_source_authored_tensor_connection_retains_integer_offsets_and_regular_cycles() -> (
    None
):

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        connection = prepare_source_block_connection(prepared, scope=execution_scope)
        assert connection.frame_id == prepared.frame_id
        assert np.all(connection.transports == np.eye(3, dtype=np.int64))
        assert np.all(np.count_nonzero(connection.offsets, axis=1) == 1)
        assert connection.holonomy.singular_cycles.size == 0


def test_incompatible_singular_connection_cannot_replace_original_chart_transaction() -> (
    None
):

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        accepted = prepare_source_block_connection(prepared, scope=execution_scope)
        actions = accepted.transports.copy()
        actions[accepted.holonomy.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        holonomy = prepare_hex_frame_holonomy(
            prepared.reference_mesh,
            prepared.face_connections[:, (0, 2)],
            actions,
            scope=execution_scope,
        )
        proposed = eqx.tree_at(
            lambda value: (value.transports, value.holonomy),
            accepted,
            (actions, holonomy),
        )
        assert proposed.holonomy.singular_cycles.size == 1
        with pytest.raises(MeshingFailure, match="source_connection.offsets"):
            extract_source_block_integer_grid(
                prepared,
                np.full((len(prepared.root_cells), 3), 2),
                source.domain.cell_regions,
                MeshingLimits(),
                10,
                source_connection=proposed,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert accepted.holonomy.singular_cycles.size == 0
        assert np.all(accepted.transports == np.eye(3, dtype=np.int64))


@pytest.mark.parametrize("cell_kind,dimension", (("quadrilateral", 2), ("hexahedron", 3)))
def test_family_flux_templates_use_intrinsic_codimension_one_forms(
    cell_kind: str, dimension: int
) -> None:
    templates = family_transport_field_templates(cell_kind, kind="Hdiv")
    assert tuple(field.name for _, field in templates) == ("current", "history")
    for _, field in templates:
        element = field.elements[0]
        assert element.conformity == "Hdiv"
        assert element.form_basis is not None
        assert element.form_basis.dimension == dimension
        assert element.form_basis.form_degree == dimension - 1


def test_nonbinary_integer_charts_retain_exact_thirds_not_rounded_reference_fits() -> (
    None
):
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    domain = source.domain
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        grid = extract_source_block_integer_grid(
            prepared,
            np.full((1, 3), 3, dtype=np.int64),
            domain.cell_regions,
            MeshingLimits(),
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(grid.root_intervals == 3)
        assert np.all(grid.chart_denominators == 3)
        assert sum(block.cell_count for block in grid.mesh.blocks) == 27
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        original = np.asarray(
            domain.source_geometry.source_coordinates(budget=reference_budget),
            dtype=np.float64,
        )
        retained = np.asarray(
            mapped.geometry.source_coordinates(budget=reference_budget), dtype=np.float64
        )
        assert np.array_equal(original.view(np.uint64), retained.view(np.uint64))
        validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
        embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        fidelity = certify_source_fidelity(
            mapped.mesh,
            mapped.geometry,
            MappedDomainBoundarySource(domain, mapped.cell_regions),
            tolerance=0.0,
        )
        assert embedding.status == coverage.status == fidelity.status == "certified"
        assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0


@pytest.mark.parametrize("physical", (False, True))
def test_mapped_composition_retires_lookup_owners_before_mesh_construction(
    physical: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    from phydrax.meshing import (
        _hex_frame_map as frame_owner,
        _hex_generation as generation,
    )

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        transient: list[weakref.ReferenceType[Any]] = []
        results: list[weakref.ReferenceType[Any]] = []
        leases = []
        original_prepare = generation._prepare_mapped_source
        original_group = generation._group_mapped_restrictions
        original_compose = generation._compose_mapped_geometry
        original_mesh = generation.CellMesh
        original_banks = frame_owner._prepare_source_integer_banks
        original_proof = frame_owner.certify_source_grid_restrictions
        raw_banks: list[weakref.ReferenceType[Any]] = []
        chart_banks: list[weakref.ReferenceType[Any]] = []
        observations = []
        starting_temporary = []

        def capture_prepare(*args: Any, **kwargs: Any) -> Any:
            starting_temporary.append(budget.temporary_bytes_upper)
            value = original_prepare(*args, **kwargs)
            transient.append(weakref.ref(value))
            leases.append(kwargs["storage"])
            return value

        def capture_group(*args: Any, **kwargs: Any) -> Any:
            value = original_group(*args, **kwargs)
            transient.append(weakref.ref(value))
            return value

        def capture_compose(*args: Any, **kwargs: Any) -> Any:
            value = original_compose(*args, **kwargs)
            preparation = args[0]
            # Canonical blocks own their connectivity, never a view of a dense
            # lookup bank that would pin preparation after its last use.
            assert all(not np.shares_memory(
                np.asarray(block.vertices), preparation.reference_cells
            ) for block in value.blocks)
            assert not np.shares_memory(value.order, preparation.reference_cells)
            results.append(weakref.ref(value))
            return value

        def capture_mesh(*args: Any, **kwargs: Any) -> Any:
            assert transient and all(reference() is None for reference in transient)
            assert all(lease.bound == 0 and lease._closed for lease in leases)
            value = results[-1]()
            assert value is not None
            assert budget.retained_owner_objects[id(value)] is value
            assert args[1] is value.blocks
            observations.append(budget.temporary_bytes_upper)
            return original_mesh(*args, **kwargs)

        def capture_banks(*args: Any, **kwargs: Any) -> Any:
            value = original_banks(*args, **kwargs)
            raw_banks.extend((weakref.ref(value.points), weakref.ref(value.cells)))
            chart_banks.extend((
                weakref.ref(value.chart_numerators), weakref.ref(value.chart_denominators)
            ))
            return value

        def capture_proof(*args: Any, **kwargs: Any) -> None:
            assert raw_banks and all(reference() is None for reference in raw_banks)
            assert all(reference() is not None for reference in chart_banks)
            original_proof(*args, **kwargs)

        monkeypatch.setattr(generation, "_prepare_mapped_source", capture_prepare)
        monkeypatch.setattr(generation, "_group_mapped_restrictions", capture_group)
        monkeypatch.setattr(generation, "_compose_mapped_geometry", capture_compose)
        monkeypatch.setattr(generation, "CellMesh", capture_mesh)
        monkeypatch.setattr(frame_owner, "_prepare_source_integer_banks", capture_banks)
        monkeypatch.setattr(frame_owner, "certify_source_grid_restrictions", capture_proof)
        grid = extract_source_block_integer_grid(
            prepared, np.full((2, 3), 2, dtype=np.int64), source.domain.cell_regions,
            limits, 6, scope=scope, budget=budget,
        )
        # The entire private reference proof is discarded, including output
        # ownership. Its published grid remains the original one-block carrier.
        assert all(reference() is None for reference in results)
        assert len(grid.mesh.blocks) == 1
        if physical:
            results.clear()
            mapped = extract_source_frame_grid(
                prepared, grid, source.domain.cell_regions, limits,
                scope=scope, budget=budget,
            )
            realized = results[-1]()
            assert realized is not None
            assert all(a is b for a, b in zip(
                realized.blocks, mapped.mesh.blocks, strict=True
            ))
            np.testing.assert_array_equal(
                realized.physical_points.view(np.uint64),
                np.asarray(mapped.mesh.coordinates).view(np.uint64),
            )
        assert len(observations) == (2 if physical else 1)
        # Only the discarded-preparation snapshot can remain temporary here;
        # the root/candidate/group/image envelope has actually ended.
        assert all(after <= before for after, before in zip(
            observations, starting_temporary, strict=True
        ))
        if physical:
            assert observations[-1] == 0


def test_source_grid_failed_output_transfer_closes_all_preparation_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing import _hex_generation as generation

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        original_prepare = generation._prepare_mapped_source
        original_retain = budget.retain_owner
        references: list[weakref.ReferenceType[Any]] = []
        leases = []

        def capture_prepare(*args: Any, **kwargs: Any) -> Any:
            result = original_prepare(*args, **kwargs)
            references.append(weakref.ref(result))
            leases.append(kwargs["storage"])
            return result

        def refuse_transfer(value: Any, **kwargs: Any) -> None:
            if isinstance(value, generation._MappedGeometryPreparation):
                references.append(weakref.ref(value))
                assert kwargs["temporary_storage_credit"].bound > 0
                with budget.bound_stage(0, budget.maximum_memory_bytes):
                    original_retain(value, **kwargs)
                pytest.fail("A fresh retained graph must charge its owner visits.")
            original_retain(value, **kwargs)

        monkeypatch.setattr(generation, "_prepare_mapped_source", capture_prepare)
        monkeypatch.setattr(budget, "retain_owner", refuse_transfer)
        before = budget.temporary_bytes_upper
        with pytest.raises(CoordinateEnclosureResourceError, match="coefficient_work"):
            extract_source_block_integer_grid(
                prepared, np.full((1, 3), 2, dtype=np.int64),
                source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
            )
        assert budget.temporary_bytes_upper == before
        assert leases and all(lease.bound == 0 and lease._closed for lease in leases)
        assert references and all(reference() is None for reference in references)
        assert not any(isinstance(value, generation._MappedGeometryPreparation)
                       for value in budget.retained_owner_objects.values())


@pytest.mark.parametrize("resource", ("storage", "work"))
@pytest.mark.parametrize("compact", (False, True))
def test_source_grid_output_lifetime_exact_simultaneous_resource_boundary(
    resource: str, compact: bool,
) -> None:
    from phydrax._meshcore import MeshcoreError, NativeExecutionScope
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)

    def construct(storage: int, work_limit: int | None = None) -> tuple[Any, int, int]:
        limits = MeshingLimits(maximum_scratch_bytes=storage) if work_limit is None else MeshingLimits(
            maximum_scratch_bytes=storage, maximum_work_units=work_limit
        )
        with NativeExecutionScope(limits.native_limits()) as scope:
            scope.observe_managed_admission()
            ticket = scope.ticket_view()
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=storage,
            )
            with ticket, budget.activate(), ticket.native_target():
                prepared = prepare_source_hex_frame(
                    source.domain.reference_mesh, source.domain.source_geometry,
                    scope=scope, budget=budget,
                )
                extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
                grid = extract(
                    prepared, np.full((2, 3), 3, dtype=np.int64),
                    source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
                )
                if compact:
                    assert isinstance(grid, _SourceBlockTensorGrid)
                    _extract_source_tensor_frame_grid(
                        prepared, grid, source.domain.cell_regions, limits,
                        scope=scope, budget=budget,
                        quality_policy=NativeHexGridSchedule("frame_grid"),
                    )
        evidence = scope.completed_evidence()
        peak = evidence.managed_admission_peak_bytes_upper
        assert peak is not None
        return grid, peak, evidence.work_counters.construction_work

    expected, required, work = construct(64 << 20)
    actual, repeated, repeated_work = construct(
        required, work if resource == "work" else None
    )
    assert repeated == required
    assert repeated_work == work
    assert actual.grid_id == expected.grid_id
    assert actual.mesh.mesh_id == expected.mesh.mesh_id
    for name in (
        "cell_regions", "source_parent_rows", "root_intervals",
        *(("cell_integer_lower", "chart_numerators", "chart_denominators") if not compact else ()),
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(
        np.asarray(actual.mesh.coordinates).view(np.uint64),
        np.asarray(expected.mesh.coordinates).view(np.uint64),
    )
    if resource == "storage":
        with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
            construct(required - 1)
    else:
        with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
            construct(required, work - 1)


def _long_lived_source_grid_proof(
    mesh: Any, source_mesh: Any, parents: Any, numerators: Any, denominators: Any,
    regions: Any, limits: Any, *, scope: Any, budget: Any, tensor_counts: Any = None,
) -> None:
    """Keep composition scratch live, with the same atomic validity operation."""
    from phydrax.discretization._cell_geometry import (
        affine_geometry_construction_bounds,
        CellGeometrySpec,
    )
    from phydrax.discretization._cell_geometry_validity import (
        _certify_restriction_cell_geometry_validity,
    )
    from phydrax.discretization._cell_mesh import cell_mesh_construction_bounds, CellMesh
    from phydrax.meshing import _hex_generation as generation
    from phydrax.meshing._family_execution import coordinate_phase

    shapes = tuple((block.cell_kind, block.cell_count, block.arity)
                   for block in source_mesh.blocks)
    with scope.host_workspace() as workspace:
        workspace.set_bound(affine_geometry_construction_bounds(
            source_mesh.coordinates.shape[0], source_mesh.ambient_dimension, shapes
        ))
        root_geometry = CellGeometrySpec.affine(source_mesh)
    with (
        coordinate_phase(scope=scope, budget=budget),
        budget.discarded_preparation_scope(),
        budget.live_storage() as source_storage,
    ):
        root_validity = certify_cell_geometry_validity(
            root_geometry, mesh=source_mesh, budget=budget
        )
        assert root_validity.all_certified
        prepared = generation._prepare_mapped_source(
            mesh, source_mesh, root_geometry, regions, budget=budget, storage=source_storage
        )
        charts = generation._MappedRestrictionCharts(parents, numerators, denominators)
        grouped = generation._group_mapped_restrictions(
            prepared, charts, source_mesh, limits, budget=budget
        )
        with budget.live_storage() as placement:
            realized = generation._compose_mapped_geometry(
                prepared, charts, grouped, source_mesh, root_geometry,
                scope=scope, budget=budget, placement_storage=placement,
            )
        budget.reserve(mesh.coordinates.size)
        np.testing.assert_array_equal(
            realized.physical_points.view(np.uint64),
            np.asarray(mesh.coordinates).view(np.uint64),
        )
        del prepared, grouped
        shapes = tuple((block.cell_kind, block.cell_count, block.arity)
                       for block in realized.blocks)
        with scope.host_workspace() as workspace:
            workspace.set_bound(cell_mesh_construction_bounds(
                mesh.coordinates.shape[0], mesh.ambient_dimension, shapes
            ))
            proof_mesh = CellMesh(
                mesh.coordinates, realized.blocks,
                vertex_global_ids=mesh.vertex_global_ids, numeric_version=mesh.numeric_version,
            )
        validity = _certify_restriction_cell_geometry_validity(
            source_mesh, root_geometry, root_validity,
            proof_mesh, realized.geometry, budget=budget,
        )
        budget.reserve(len(realized.order), 8 * len(realized.order))
        status = np.empty(len(realized.order), dtype=np.int32)
        status[realized.order] = np.asarray(validity.status)
        assert validity.all_certified


@pytest.mark.parametrize("roots", (2, 6))
def test_source_grid_lifetime_preserves_old_arrays_and_accounts_every_owner_visit(
    roots: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization import _coordinate_budget as budget_owner
    from phydrax.meshing import (
        _hex_frame_map as frame_owner,
        _hex_generation as generation,
    )

    source = curved_occupied_source({(root, 0, 0) for root in range(roots)}, curvature=0.125)
    profile = budget_owner.retained_host_storage_profile
    retain = budget_owner.CoordinateEnclosureBudget.retain_owner
    visits: list[int] = []
    charged: list[int] = []

    def observed_profile(value: Any, *args: Any, **kwargs: Any) -> Any:
        result = profile(value, *args, **kwargs)
        if isinstance(value, generation._MappedGeometryPreparation):
            visits.append(result[3])
        return result

    def observed_retain(self: Any, value: Any, **kwargs: Any) -> None:
        before = self.work_units
        retain(self, value, **kwargs)
        if isinstance(value, generation._MappedGeometryPreparation):
            charged.append(self.work_units - before)
            assert charged[-1] == visits[-1] > 0
            # A second handoff of the very same owned graph must not scan or
            # charge its content again.
            retain(self, value, **kwargs)
            assert self.work_units == before + visits[-1]

    monkeypatch.setattr(budget_owner, "retained_host_storage_profile", observed_profile)
    monkeypatch.setattr(budget_owner.CoordinateEnclosureBudget, "retain_owner", observed_retain)

    def construct() -> tuple[SourceBlockIntegerGrid, int, int]:
        limits = MeshingLimits()
        with NativeExecutionScope(limits.native_limits()) as scope:
            scope.observe_managed_admission()
            ticket = scope.ticket_view()
            budget = budget_owner.CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket, budget.activate(), ticket.native_target():
                prepared = prepare_source_hex_frame(
                    source.domain.reference_mesh, source.domain.source_geometry,
                    scope=scope, budget=budget,
                )
                result = extract_source_block_integer_grid(
                    prepared, np.full((roots, 3), 3, dtype=np.int64),
                    source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
                )
        evidence = scope.completed_evidence()
        peak = evidence.managed_admission_peak_bytes_upper
        assert peak is not None
        return result, peak, evidence.work_counters.construction_work

    actual, peak, work = construct()
    assert len(visits) == len(charged) == 1
    monkeypatch.setattr(frame_owner, "certify_source_grid_restrictions", _long_lived_source_grid_proof)
    expected, old_peak, old_work = construct()
    assert actual.grid_id == expected.grid_id
    # The compact tensor theorem replaces independent child differential proofs;
    # the old path still independently authenticates and certifies each child.
    assert work < old_work
    assert peak < old_peak
    for name in (
        "cell_regions", "source_parent_rows", "root_intervals",
        "cell_integer_lower", "chart_numerators", "chart_denominators",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(
        np.asarray(actual.mesh.coordinates).view(np.uint64),
        np.asarray(expected.mesh.coordinates).view(np.uint64),
    )


@pytest.mark.parametrize("mutate", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_ordinary_source_callback_precedes_physical_composition_and_revalidation(
    mutate: bool, compact: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _hex_generation as generation
    from phydrax.meshing._measurements import NativeMeshingPhaseMeasurement

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    limits = MeshingLimits()
    extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
    realize = _realize_test_grid
    events: list[str] = []
    compose = generation._compose_mapped_geometry

    def capture_composition(*args: Any, **kwargs: Any) -> Any:
        physical = args[4] is source.domain.source_geometry
        if physical:
            assert events[-1] == "source_block_integer_extraction"
        events.append("physical" if physical else "reference")
        return compose(*args, **kwargs)

    def record(measurement: NativeMeshingPhaseMeasurement) -> None:
        events.append(measurement.phase)
        if measurement.phase == "source_block_integer_extraction":
            assert "reference" in events and "physical" not in events
            if mutate:
                geometry = source.domain.source_geometry
                object.__setattr__(
                    geometry, "coordinates", geometry.coordinates.at[0, 0].add(0.25)
                )

    monkeypatch.setattr(generation, "_compose_mapped_geometry", capture_composition)
    with native_meshing_request(limits) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        grid = extract(
            prepared, np.full((1, 3), 2, dtype=np.int64),
            source.domain.cell_regions, limits, 6,
            record_phase=record, scope=scope, budget=budget,
        )
        if mutate:
            with pytest.raises(MeshingFailure, match="prepared.source_id"):
                realize(
                    prepared, grid, source.domain.cell_regions, limits,
                    scope=scope, budget=budget,
                )
            assert "physical" not in events
        else:
            mapped = realize(
                prepared, grid, source.domain.cell_regions, limits,
                scope=scope, budget=budget,
            )
            assert events[-1] == "physical"
            assert mapped.source_parent_rows.shape == (8,)
        assert budget._polynomial_preparation_storage is None


@pytest.mark.parametrize("counts", ((2, 3, 1), (3, 1, 2), (1, 2, 3)))
@pytest.mark.parametrize("compact", (False, True))
def test_tensor_grid_certificate_matches_independent_ordinary_restrictions(
    counts: tuple[int, int, int], compact: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as owner

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    certify = owner._certify_restriction_cell_geometry_validity
    observations = []

    def compare(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["tensor_grid"] is not None
        with coordinate_reference_budget() as compact_budget:
            compact_arguments: dict[str, Any] = {**kwargs, "budget": compact_budget}
            compact = certify(*args, **compact_arguments)
            compact_work = compact_budget.work_units
        with coordinate_reference_budget() as ordinary_budget:
            ordinary_arguments: dict[str, Any] = {
                **kwargs, "budget": ordinary_budget, "tensor_grid": None,
            }
            ordinary = certify(*args, **ordinary_arguments)
            ordinary_work = ordinary_budget.work_units
        assert compact.certificate_id == ordinary.certificate_id
        assert compact.geometry_id == ordinary.geometry_id
        assert compact.geometry_layout_id == ordinary.geometry_layout_id
        assert compact.topology_id == ordinary.topology_id
        assert compact.policy_id == ordinary.policy_id
        assert compact.block_names == ordinary.block_names
        assert compact.block_offsets == ordinary.block_offsets
        assert compact.unresolved_reasons == ordinary.unresolved_reasons
        for name in ("status", "determinant_lower", "determinant_upper", "depth"):
            np.testing.assert_array_equal(getattr(compact, name), getattr(ordinary, name))
        assert compact_work < ordinary_work
        observations.append((compact_work, ordinary_work))
        return certify(*args, **kwargs)

    monkeypatch.setattr(owner, "_certify_restriction_cell_geometry_validity", compare)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
        grid = extract(
            prepared, np.asarray([counts], dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        assert grid.mesh.blocks[0].cell_count == 6
    assert len(observations) == 1


@pytest.mark.parametrize("fault", ("zero", "reversed", "gap", "overlap", "order", "missing"))
@pytest.mark.parametrize(("compact", "physical"), ((False, False), (True, False), (True, True)))
def test_tensor_grid_rejects_partition_and_order_faults_without_cached_verdict(
    fault: str, compact: bool, physical: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from phydrax.discretization import _cell_geometry_validity as owner

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    certify = owner._certify_restriction_cell_geometry_validity
    observed = []

    def faulted(*args: Any, **kwargs: Any) -> Any:
        theorem = kwargs["tensor_grid"]
        valid = certify(*args, **kwargs)
        assert valid.all_certified
        if physical and args[1] is not source.domain.source_geometry:
            return valid
        if fault == "zero":
            broken = replace(theorem, counts=((0, 2, 2),))
        elif fault == "reversed":
            broken = replace(theorem, counts=((-2, 2, 2),))
        elif fault == "gap":
            broken = replace(theorem, counts=((1, 4, 2),))
        elif fault == "overlap":
            broken = replace(theorem, order=(theorem.order[0],) * len(theorem.order))
        elif fault == "order":
            broken = replace(theorem, order=tuple(reversed(theorem.order)))
        else:
            broken = replace(theorem, order=theorem.order[:-1])
        # The first successful call must not make a malformed replay eligible.
        with pytest.raises(ValueError, match="tensor_grid"):
            certify(*args, **{**kwargs, "tensor_grid": broken})
        # Cold requests independently reject the same malformed input.
        with coordinate_reference_budget() as cold:
            cold_arguments: dict[str, Any] = {
                **kwargs, "tensor_grid": broken, "budget": cold,
            }
            with pytest.raises(ValueError, match="tensor_grid"):
                certify(*args, **cold_arguments)
        observed.append(fault)
        return valid

    monkeypatch.setattr(owner, "_certify_restriction_cell_geometry_validity", faulted)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
        grid = extract(
            prepared, np.full((1, 3), 2, dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        if physical:
            _realize_test_grid(
                prepared, grid, source.domain.cell_regions, MeshingLimits(),
                scope=scope, budget=budget,
            )
    assert observed == [fault]


@pytest.mark.parametrize("degree", (1, 2))
def test_tensor_chart_family_preserves_every_ordinary_element_field(degree: int) -> None:
    from phydrax.discretization._cell_geometry import (
        _TensorGridChartFamily,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    counts = (2, 3, 1)
    source = coordinate_lagrange_element("hexahedron", degree)
    corners = reference_cell_topology("hexahedron").vertices
    with coordinate_reference_budget() as budget:
        family = _TensorGridChartFamily.prepare(source, counts, budget=budget)
        for index in product(*(range(n) for n in counts)):
            integer = (index[0], index[1], index[2])
            actual = family.element(integer, budget=budget)
            numerators = np.empty((8, 3), dtype=np.int64)
            denominators = np.empty((8, 3), dtype=np.int64)
            for corner, node in zip(corners, family.nodes, strict=True):
                numerators[node] = tuple(integer[axis] + int(corner[axis]) for axis in range(3))
                denominators[node] = counts
            expected = PolynomialComposedCellGeometryElement(
                source, family.prototype.chart_element, numerators, denominators,
                budget=budget,
            )
            for name in (
                "element_id", "cell_kind", "conformity", "degree",
                "topological_dimension", "local_dof_count", "chart_coefficients",
            ):
                assert getattr(actual, name) == getattr(expected, name)
            assert actual.source_element is source
            assert actual.chart_element is expected.chart_element
            np.testing.assert_array_equal(
                np.asarray(actual.chart_coordinates).view(np.uint64),
                np.asarray(expected.chart_coordinates).view(np.uint64),
            )
        for integer in ((-1, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 1)):
            with pytest.raises(ValueError, match="tensor_grid.chart"):
                family.element(integer, budget=budget)


@pytest.mark.parametrize("curvature", (0.0, 0.125))
@pytest.mark.parametrize("counts", ((2, 3, 1), (3, 2, 3)))
def test_internal_tensor_grid_preserves_ordinary_publication_without_materialized_banks(
    curvature: float, counts: tuple[int, int, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as validity
    from phydrax.meshing import _hex_frame_map as frame

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=curvature)
    limits = MeshingLimits()

    def construct(compact: bool) -> tuple[Any, Any, list[str]]:
        events: list[str] = []
        with native_meshing_request(limits) as (scope, budget, _):
            prepared = prepare_source_hex_frame(
                source.domain.reference_mesh, source.domain.source_geometry,
                scope=scope, budget=budget,
            )
            extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
            realize = _realize_test_grid
            grid = extract(
                prepared, np.asarray([counts, counts], dtype=np.int64),
                source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
                record_phase=lambda measurement: events.append(measurement.phase),
            )
            mapped = realize(
                prepared, grid, source.domain.cell_regions, limits, scope=scope, budget=budget,
            )
        return grid, mapped, events

    expected_grid, expected, expected_events = construct(False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("The internal tensor route must not materialize ordinary chart banks or import a predecessor")

    monkeypatch.setattr(frame, "_prepare_source_integer_banks", forbidden)
    monkeypatch.setattr(validity, "_prepare_validity_import", forbidden)
    actual_grid, actual, events = construct(True)
    assert events == expected_events
    assert actual_grid.grid_id == expected_grid.grid_id
    assert actual_grid.mesh.mesh_id == expected_grid.mesh.mesh_id
    assert not hasattr(actual_grid, "chart_numerators")
    assert not hasattr(actual_grid, "chart_denominators")
    assert not hasattr(actual_grid, "cell_integer_lower")
    assert actual.mapped_id == expected.mapped_id
    assert actual.mesh.mesh_id == expected.mesh.mesh_id
    assert actual.reference_mesh.mesh_id == expected.reference_mesh.mesh_id
    assert validity.cell_geometry_id(actual.geometry) == validity.cell_geometry_id(expected.geometry)
    for mesh_name in ("mesh", "reference_mesh"):
        left, right = getattr(actual, mesh_name), getattr(expected, mesh_name)
        np.testing.assert_array_equal(
            np.asarray(left.coordinates).view(np.uint64),
            np.asarray(right.coordinates).view(np.uint64),
        )
        for a, b in zip(left.blocks, right.blocks, strict=True):
            assert a.name == b.name
            np.testing.assert_array_equal(a.vertices, b.vertices)
            np.testing.assert_array_equal(a.global_ids, b.global_ids)
    for name in ("cell_regions", "source_parent_rows", "target_reference_rows"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("scaled_jacobian_lower", "mean_ratio_lower", "aspect_ratio_upper"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name)).view(np.uint64),
            np.asarray(getattr(expected, name)).view(np.uint64),
        )


@pytest.mark.parametrize("fault", ("denominator", "nonmonotone", "gap", "overlap", "parent"))
def test_tensor_grid_rejects_changed_construction_banks(
    fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _hex_frame_map as frame_owner

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    prepare = frame_owner._prepare_source_integer_banks

    def changed(*args: Any, **kwargs: Any) -> Any:
        banks = prepare(*args, **kwargs)
        if fault == "denominator":
            banks.chart_denominators[0, 0, 0] = 0
        elif fault == "nonmonotone":
            banks.chart_numerators[[0, 1]] = banks.chart_numerators[[1, 0]]
        elif fault == "gap":
            banks.chart_numerators[0, 0, 0] += 1
        elif fault == "overlap":
            banks.chart_numerators[1] = banks.chart_numerators[0]
        else:
            banks.parent_rows[0] = 1
        return banks

    monkeypatch.setattr(frame_owner, "_prepare_source_integer_banks", changed)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        with pytest.raises((ValueError, MeshingFailure)):
            extract_source_block_integer_grid(
                prepared, np.full((2, 3), 2, dtype=np.int64),
                source.domain.cell_regions, MeshingLimits(), 6,
                scope=scope, budget=budget,
            )


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_tensor_grid_phase_callback_reauthenticates_source_before_construction(
    reference: bool, compact: bool,
) -> None:
    from phydrax.meshing._measurements import NativeMeshingPhaseMeasurement

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )

        def mutate(measurement: NativeMeshingPhaseMeasurement) -> None:
            if measurement.phase == "block_count_compatibility":
                geometry = prepared.reference_mesh if reference else prepared.geometry
                object.__setattr__(
                    geometry, "coordinates", geometry.coordinates.at[0, 0].add(0.25)
                )

        with pytest.raises(MeshingFailure, match="after the phase callback"):
            extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
            extract(
                prepared, np.full((1, 3), 2, dtype=np.int64),
                source.domain.cell_regions, MeshingLimits(), 6,
                record_phase=mutate, scope=scope, budget=budget,
            )


@pytest.mark.parametrize("compact", (False, True))
def test_reference_tensor_theorem_does_not_certify_reversed_physical_map(compact: bool) -> None:
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
        realize = _realize_test_grid
        grid = extract(
            prepared, np.full((1, 3), 2, dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        geometry = prepared.geometry
        assert geometry.exact_source is None
        object.__setattr__(
            geometry, "coordinates", geometry.coordinates.at[:, 0].multiply(-1)
        )
        # Even a freshly named physical source cannot inherit the reference
        # tensor theorem: its actual represented maps must certify separately.
        object.__setattr__(prepared, "source_id", cell_geometry_id(geometry))
        with pytest.raises(MeshingFailure, match="hex.mapped.validity"):
            realize(
                prepared, grid, source.domain.cell_regions, MeshingLimits(),
                scope=scope, budget=budget,
            )


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize(
    ("matrix", "positive"),
    (
        (((0, 1, 0), (0, 0, 1), (1, 0, 0)), True),
        (((-1, 0, 0), (0, -1, 0), (0, 0, 1)), True),
        (((-1, 0, 0), (0, 1, 0), (0, 0, 1)), False),
        (((0, 0, 0), (0, 1, 0), (0, 0, 1)), False),
    ),
)
def test_tensor_grid_reference_axes_require_exact_positive_orientation(
    matrix: tuple[tuple[int, ...], ...], positive: bool, compact: bool,
) -> None:
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    reference = source.domain.reference_mesh
    transformed = reference.with_coordinates(
        np.asarray(reference.coordinates) @ np.asarray(matrix).T,
        numeric_version=reference.numeric_version,
    )

    def construct(scope: Any, budget: Any) -> Any:
        prepared = prepare_source_hex_frame(
            transformed, source.domain.source_geometry, scope=scope, budget=budget
        )
        extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
        return extract(
            prepared, np.asarray([[2, 3, 1]], dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )

    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        if positive:
            grid = construct(scope, budget)
            assert grid.mesh.blocks[0].cell_count == 6
        else:
            with pytest.raises((ValueError, MeshingFailure)):
                construct(scope, budget)


@pytest.mark.parametrize("root_count", (2, 4))
@pytest.mark.parametrize("intervals", (2, 3))
def test_tensor_grid_bounded_multiroot_work_projection(
    root_count: int, intervals: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure root-only constants, not a scaled-down family workflow verdict."""
    import json
    from math import ceil

    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization import (
        _cell_geometry as geometry_owner,
        _cell_geometry_validity as validity_owner,
        _coordinate_budget as budget_owner,
    )
    from phydrax.meshing import _hex_frame_map as frame_owner

    source = curved_occupied_source({(2 * root, 0, 0) for root in range(root_count)})
    reference = source.domain.reference_mesh
    # Disjoint roots maximize source-bank rows per root. A dense positive
    # affine frame exercises all nine Jacobian entries, unlike a diagonal box.
    reference = reference.with_coordinates(
        np.asarray(reference.coordinates) @ np.asarray(((2, 1, 1), (1, 2, 1), (1, 1, 2))),
        numeric_version=reference.numeric_version,
    )
    tensor = validity_owner._tensor_grid_validity_blocks
    prepare_family = geometry_owner._TensorGridChartFamily.prepare
    proof = frame_owner.certify_source_grid_restrictions
    root_constants: list[int] = []
    prototype_constants: list[int] = []

    def observed_tensor(*args: Any, **kwargs: Any) -> Any:
        theorem, layout, _, _, _, _, budget = args
        before = budget.work_units
        result = tensor(*args, **kwargs)
        assert result is not None
        cells = sum(np.prod(row) for row in theorem.counts)
        endpoints = sum(n + 1 for row in theorem.counts for n in row)
        variable = 31 * int(cells) + 57 * len(layout.names) + 2 * endpoints
        root_constants.append(ceil((budget.work_units - before - variable) / root_count))
        return result

    def observed_family(cls: Any, *args: Any, **kwargs: Any) -> Any:
        budget = kwargs["budget"]
        before = budget.work_units
        result = prepare_family(*args, **kwargs)
        prototype_constants.append(
            budget.work_units - before - 2 * sum(n + 1 for n in result.counts)
        )
        return result

    monkeypatch.setattr(validity_owner, "_tensor_grid_validity_blocks", observed_tensor)
    monkeypatch.setattr(geometry_owner._TensorGridChartFamily, "prepare", classmethod(observed_family))

    def construct(compact: bool) -> tuple[SourceBlockIntegerGrid, int]:
        def selected(*args: Any, **kwargs: Any) -> None:
            if not compact:
                kwargs["tensor_counts"] = None
            proof(*args, **kwargs)

        monkeypatch.setattr(frame_owner, "certify_source_grid_restrictions", selected)
        limits = MeshingLimits()
        with NativeExecutionScope(limits.native_limits()) as scope:
            scope.observe_managed_admission()
            ticket = scope.ticket_view()
            budget = budget_owner.CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket, budget.activate(), ticket.native_target():
                prepared = prepare_source_hex_frame(
                    reference, source.domain.source_geometry, scope=scope, budget=budget
                )
                grid = extract_source_block_integer_grid(
                    prepared, np.full((root_count, 3), intervals, dtype=np.int64),
                    source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
                )
        return grid, scope.completed_evidence().work_counters.construction_work

    compact, compact_work = construct(True)
    ordinary, ordinary_work = construct(False)
    assert compact.grid_id == ordinary.grid_id
    assert compact_work < ordinary_work
    assert root_constants and prototype_constants
    # The ordinary child proof's conservative 232-work subtotal, the compact
    # 32*C+57*G binding, and >=26*G constructor saving give >=169*C when G<=C.
    # Positive integer axes satisfy a+b+c<=abc+2, hence A<=C+5*R.
    # Keep an additional 26-work/root allowance for singleton identity charts.
    roots, children = 26, 9359
    endpoints_upper = children + 5 * roots
    root_upper = max(root_constants) + max(prototype_constants) + 3 + 26
    projected_net = 169 * children - 4 * endpoints_upper - roots * root_upper
    assert projected_net >= 1_295_000
    print(json.dumps({
        "roots": root_count, "intervals": intervals,
        "ordinary_work": ordinary_work, "compact_work": compact_work,
        "observed_saving": ordinary_work - compact_work,
        "root_constant": max(root_constants),
        "prototype_constant": max(prototype_constants),
        "projected_net_saving": projected_net,
    }, sort_keys=True))


@pytest.mark.parametrize("fault", ("parent", "vertices", "route", "chart", "source"))
@pytest.mark.parametrize("physical", (False, True))
def test_compact_tensor_theorem_reauthenticates_mutated_owner_and_cold_replay(
    fault: str, physical: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as owner

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    certify = owner._certify_restriction_cell_geometry_validity
    observed: list[str] = []

    def changed(*args: Any, **kwargs: Any) -> Any:
        valid = certify(*args, **kwargs)
        if physical and args[1] is not source.domain.source_geometry:
            return valid
        geometry = args[4]
        record = geometry.restriction_source
        assert record is not None
        if fault == "parent":
            target, name = record, "parent_cell_ids"
            value = (record.parent_cell_ids[0].at[0].add(1), *record.parent_cell_ids[1:])
        elif fault == "vertices":
            target, name = record, "parent_vertex_ids"
            value = (record.parent_vertex_ids[0].at[0, 0].add(1), *record.parent_vertex_ids[1:])
        elif fault == "route":
            target, name = geometry, "geometry_dofs"
            value = (geometry.geometry_dofs[0].at[0, 0].set(geometry.geometry_dofs[0][0, 1]), *geometry.geometry_dofs[1:])
        elif fault == "chart":
            target, name = geometry.elements[0], "chart_coefficients"
            coefficients = target.chart_coefficients
            value = (((99, 1), *coefficients[0][1:]), *coefficients[1:])
        else:
            target, name = geometry, "coordinates"
            value = geometry.coordinates.at[0, 0].add(0.125)
        original = getattr(target, name)
        try:
            object.__setattr__(target, name, value)
            with pytest.raises(ValueError):
                certify(*args, **kwargs)
            with coordinate_reference_budget() as cold:
                with pytest.raises(ValueError):
                    cold_arguments: dict[str, Any] = {**kwargs, "budget": cold}
                    certify(*args, **cold_arguments)
        finally:
            object.__setattr__(target, name, original)
        observed.append(fault)
        return valid

    monkeypatch.setattr(owner, "_certify_restriction_cell_geometry_validity", changed)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        grid = _extract_source_block_tensor_grid(
            prepared, np.full((1, 3), 2, dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        if physical:
            _realize_test_grid(
                prepared, grid, source.domain.cell_regions, MeshingLimits(),
                scope=scope, budget=budget,
            )
    assert observed == [fault]


@pytest.mark.parametrize("root_count", (2, 4))
@pytest.mark.parametrize("intervals", (2, 3))
def test_compact_construction_observed_work_matches_static_projection(
    root_count: int, intervals: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure bounded construction, proof-entry and full realization on ABI0d11."""
    import json
    from math import ceil, prod

    from phydrax.discretization import _cell_geometry_validity as validity
    from phydrax.meshing import _hex_frame_map as frame

    source = curved_occupied_source({(2 * root, 0, 0) for root in range(root_count)}, curvature=0.125)
    requested = np.asarray(
        [(intervals + root % 2, intervals, intervals) for root in range(root_count)],
        dtype=np.int64,
    )
    stage: str | None = None
    tickets: dict[str, list[Any]] = {"banks": [], "identity": []}
    proof_entries: list[tuple[int, int]] = []
    physical_work: list[int] = []
    seal_work: list[int] = []
    proof_start = 0
    original_banks = frame._prepare_source_tensor_banks
    original_identity = frame._tensor_chart_fingerprint
    original_certify = validity._certify_restriction_cell_geometry_validity
    original_theorem = validity._tensor_grid_validity_blocks
    original_seal = validity._prepare_constructed_tensor_grid

    def seal(*args: Any, **kwargs: Any) -> Any:
        before = args[-1].work_units
        result = original_seal(*args, **kwargs)
        assert result is not None
        seal_work.append(args[-1].work_units - before)
        return result

    def banks(*args: Any, **kwargs: Any) -> Any:
        nonlocal stage
        stage = "banks"
        try:
            return original_banks(*args, **kwargs)
        finally:
            stage = None

    def identity(*args: Any, **kwargs: Any) -> Any:
        nonlocal stage
        stage = "identity"
        try:
            return original_identity(*args, **kwargs)
        finally:
            stage = None

    def certify(*args: Any, **kwargs: Any) -> Any:
        nonlocal proof_start
        proof_start = kwargs["budget"].work_units
        result = original_certify(*args, **kwargs)
        if args[1] is source.domain.source_geometry:
            physical_work.append(kwargs["budget"].work_units - proof_start)
        return result

    def theorem(*args: Any, **kwargs: Any) -> Any:
        if args[4].predecessor.source_geometry is not source.domain.source_geometry:
            proof_entries.append((args[-1].work_units - proof_start, len(args[1].names)))
        return original_theorem(*args, **kwargs)

    monkeypatch.setattr(frame, "_prepare_source_tensor_banks", banks)
    monkeypatch.setattr(frame, "_tensor_chart_fingerprint", identity)
    monkeypatch.setattr(validity, "_certify_restriction_cell_geometry_validity", certify)
    monkeypatch.setattr(validity, "_tensor_grid_validity_blocks", theorem)
    monkeypatch.setattr(validity, "_prepare_constructed_tensor_grid", seal)

    def construct(compact: bool) -> tuple[Any, Any, int, int, int]:
        limits = MeshingLimits()
        with NativeExecutionScope(limits.native_limits()) as scope:
            scope.observe_managed_admission()
            original_batch = scope.batch

            def batch(work: int, /, **kwargs: Any) -> Any:
                result = original_batch(work, **kwargs)
                if stage is not None:
                    tickets[stage].append(result)
                return result

            monkeypatch.setattr(scope, "batch", batch)
            ticket = scope.ticket_view()
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket, budget.activate(), ticket.native_target():
                prepared = prepare_source_hex_frame(
                    source.domain.reference_mesh, source.domain.source_geometry,
                    scope=scope, budget=budget,
                )
                extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
                grid = extract(
                    prepared, requested, source.domain.cell_regions, limits, 6,
                    scope=scope, budget=budget,
                )
                reference_work = limits.maximum_work_units - ticket.remaining_work_units
                mapped = _realize_test_grid(
                    prepared, grid, source.domain.cell_regions, limits,
                    scope=scope, budget=budget,
                )
        evidence = scope.completed_evidence()
        peak = evidence.managed_admission_peak_bytes_upper
        assert peak is not None
        return grid, mapped, reference_work, evidence.work_counters.construction_work, peak

    expected_grid, expected, ordinary_reference, ordinary_total, ordinary_peak = construct(False)
    actual_grid, actual, compact_reference, compact_total, compact_peak = construct(True)
    assert actual_grid.grid_id == expected_grid.grid_id
    assert actual.mapped_id == expected.mapped_id
    assert compact_reference < ordinary_reference
    assert compact_total < ordinary_total
    counts = tuple(tuple(int(n) for n in row) for row in actual_grid.root_intervals)
    cells = sum(prod(row) for row in counts)
    expected_node_work = sum(
        9 * sum(n + 1 for n in row) + 9 * prod(n + 1 for n in row)
        + 32 * (prod(row) - prod(n - 1 for n in row))
        for row in counts
    )
    observed_banks = sum(ticket.consumed_work_units for ticket in tickets["banks"])
    observed_identity = sum(ticket.consumed_work_units for ticket in tickets["identity"])
    assert observed_banks == expected_node_work + 8 * cells
    assert observed_identity == 48 * sum(prod(row) for row in set(counts))
    proof_work, groups = proof_entries[-1]
    source_work_per_root = ceil((proof_work - 4 * root_count - 4 * groups) / root_count)
    assert 26 * source_work_per_root <= 8192
    assert len(physical_work) == len(seal_work) == 1
    physical_groups = len(actual.mesh.blocks)
    # Five emitted certificate scalars/lookup operations per child, plus the
    # immutable theorem-order snapshot. Owner-edge authentication scales in
    # groups and roots, never in the 27-coefficient routes per child.
    group_upper = ceil((physical_work[0] + seal_work[0] - 6 * cells) / physical_groups)
    projected_physical = 6 * 9359 + 735 * group_upper + 26 * 1000
    assert projected_physical < 491_014
    print(json.dumps({
        "roots": root_count, "intervals": intervals, "cells": cells,
        "ordinary_reference_work": ordinary_reference,
        "compact_reference_work": compact_reference,
        "reference_saving": ordinary_reference - compact_reference,
        "ordinary_total_work": ordinary_total, "compact_total_work": compact_total,
        "total_saving": ordinary_total - compact_total,
        "ordinary_peak_bytes": ordinary_peak, "compact_peak_bytes": compact_peak,
        "observed_compact_banks": observed_banks,
        "observed_identity_worksets": observed_identity,
        "observed_pre_theorem_work": proof_work,
        "root_source_work_upper_at_26_roots": 26 * source_work_per_root,
        "observed_physical_validity_work": physical_work[0],
        "observed_physical_seal_work": seal_work[0],
        "physical_group_residual_upper": group_upper,
        "projected_r8_physical_validity_work": projected_physical,
    }, sort_keys=True))


@pytest.mark.parametrize("curvature", (0.0, 0.125))
@pytest.mark.parametrize("strict_quality", (False, True))
def test_physical_tensor_roots_match_ordinary_and_quality_falls_back(
    curvature: float, strict_quality: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as validity
    from phydrax.meshing import _hex_generation as generation

    source = curved_occupied_source({(0, 0, 0), (2, 0, 0)}, curvature=curvature)
    counts = np.asarray(((3, 2, 2), (2, 2, 3)), dtype=np.int64)
    policy = generation.NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0 if strict_quality else 0.25,
        minimum_mean_ratio=1.0 if strict_quality else 0.5,
        maximum_aspect_ratio=1.0 if strict_quality else 4.0,
    )
    theorem = validity._tensor_grid_validity_blocks
    root_source = generation._root_box_quality_source
    physical_roots: list[int] = []
    whole_calls: list[bool] = []

    def observe_theorem(*args: Any, **kwargs: Any) -> Any:
        result = theorem(*args, **kwargs)
        if args[4].predecessor.source_geometry is source.domain.source_geometry:
            assert result is not None
            physical_roots.append(len(args[0].counts))
        return result

    def observe_quality(*args: Any, **kwargs: Any) -> Any:
        whole_calls.append(kwargs.get("whole_root", False))
        return root_source(*args, **kwargs)

    monkeypatch.setattr(validity, "_tensor_grid_validity_blocks", observe_theorem)
    monkeypatch.setattr(generation, "_root_box_quality_source", observe_quality)

    def construct(compact: bool) -> Any:
        with native_meshing_request(MeshingLimits()) as (scope, budget, _):
            prepared = prepare_source_hex_frame(
                source.domain.reference_mesh, source.domain.source_geometry,
                scope=scope, budget=budget,
            )
            extract = _extract_source_block_tensor_grid if compact else extract_source_block_integer_grid
            grid = extract(
                prepared, counts, source.domain.cell_regions, MeshingLimits(), 6,
                scope=scope, budget=budget,
            )
            if compact:
                assert isinstance(grid, _SourceBlockTensorGrid)
                mapped = _extract_source_tensor_frame_grid(
                    prepared, grid, source.domain.cell_regions, MeshingLimits(),
                    scope=scope, budget=budget, quality_policy=policy,
                )
            else:
                mapped = _realize_test_grid(
                    prepared, grid, source.domain.cell_regions, MeshingLimits(),
                    scope=scope, budget=budget,
                )
            with generation.family_execution_owner.coordinate_phase(scope=scope, budget=budget):
                certificate = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh, budget=budget)
            assert certificate.all_certified
            return mapped, certificate

    ordinary, ordinary_certificate = construct(False)
    whole_calls.clear()
    actual, actual_certificate = construct(True)
    assert physical_roots == [2]
    assert whole_calls.count(True) == 2
    assert actual.mapped_id == ordinary.mapped_id
    assert actual_certificate.geometry_id == ordinary_certificate.geometry_id
    np.testing.assert_array_equal(actual_certificate.status, ordinary_certificate.status)
    np.testing.assert_array_equal(actual.mesh.coordinates, ordinary.mesh.coordinates)
    if strict_quality:
        assert False in whole_calls
        np.testing.assert_array_equal(actual.scaled_jacobian_lower, ordinary.scaled_jacobian_lower)
        np.testing.assert_array_equal(actual.mean_ratio_lower, ordinary.mean_ratio_lower)
        np.testing.assert_array_equal(actual.aspect_ratio_upper, ordinary.aspect_ratio_upper)
    else:
        for root in range(2):
            rows = actual.source_parent_rows == root
            accepted = (
                actual.scaled_jacobian_lower[rows].min() >= policy.minimum_scaled_jacobian
                and actual.mean_ratio_lower[rows].min() >= policy.minimum_mean_ratio
                and actual.aspect_ratio_upper[rows].max() <= policy.maximum_aspect_ratio
            )
            if accepted:
                assert np.all(actual.scaled_jacobian_lower[rows] <= ordinary.scaled_jacobian_lower[rows])
                assert np.all(actual.mean_ratio_lower[rows] <= ordinary.mean_ratio_lower[rows])
                assert np.all(actual.aspect_ratio_upper[rows] >= ordinary.aspect_ratio_upper[rows])
            else:
                # A failing root is not made acceptable by another root's bound.
                assert False in whole_calls
                np.testing.assert_array_equal(actual.scaled_jacobian_lower[rows], ordinary.scaled_jacobian_lower[rows])
                np.testing.assert_array_equal(actual.mean_ratio_lower[rows], ordinary.mean_ratio_lower[rows])
                np.testing.assert_array_equal(actual.aspect_ratio_upper[rows], ordinary.aspect_ratio_upper[rows])


def test_physical_tensor_root_rejects_localized_inversion() -> None:
    source = curved_occupied_source({(0, 0, 0), (2, 0, 0)}, curvature=0.0)
    geometry = source.domain.source_geometry
    coordinates = np.asarray(geometry.coordinates).copy()
    # Only the second root folds: z -> z + 2*z*(1-z), whose derivative
    # changes sign inside the root even though all corner positions agree.
    second = np.asarray(geometry.geometry_dofs[0])[1]
    z = coordinates[second, 2]
    coordinates[second, 2] = z + 2 * z * (1 - z)
    broken = eqx.tree_at(lambda value: value.coordinates, geometry, coordinates)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        with pytest.raises(MeshingFailure, match="source_geometry.validity"):
            prepare_source_hex_frame(
                source.domain.reference_mesh, broken, scope=scope, budget=budget
            )


@pytest.mark.parametrize("fault", ("root", "order", "parents", "target", "request", "replay", "failure"))
def test_physical_tensor_handoff_is_bound_single_use_and_cleans_failure(
    fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_validity as validity

    source = curved_occupied_source({(0, 0, 0), (2, 0, 0)}, curvature=0.125)
    certify = validity._certify_restriction_cell_geometry_validity
    tokens: list[Any] = []

    def intercept(*args: Any, **kwargs: Any) -> Any:
        token = kwargs.get("construction")
        if token is None:
            return certify(*args, **kwargs)
        tokens.append(token)
        if fault == "replay":
            result = certify(*args, **kwargs)
            with pytest.raises(ValueError, match="exactly once"):
                certify(*args, **kwargs)
            return result
        if fault == "failure":
            raise ValueError("injected handoff failure")
        changed = list(args)
        if fault == "root":
            changed[1] = eqx.tree_at(
                lambda value: value.coordinates, args[1], args[1].coordinates + 0.125
            )
        elif fault == "order":
            object.__setattr__(kwargs["tensor_grid"], "order", tuple(reversed(kwargs["tensor_grid"].order)))
        elif fault == "parents":
            token.parents = tuple(reversed(token.parents))
        elif fault == "target":
            object.__setattr__(args[4], "geometry_dofs", tuple(reversed(args[4].geometry_dofs)))
        elif fault == "request":
            with coordinate_reference_budget() as cold:
                cold_arguments: dict[str, Any] = {**kwargs, "budget": cold}
                return certify(*args, **cold_arguments)
        return certify(*changed, **kwargs)

    monkeypatch.setattr(validity, "_certify_restriction_cell_geometry_validity", intercept)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry, scope=scope, budget=budget
        )
        grid = _extract_source_block_tensor_grid(
            prepared, np.full((2, 3), 2, dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        if fault == "replay":
            _realize_test_grid(
                prepared, grid, source.domain.cell_regions, MeshingLimits(), scope=scope, budget=budget
            )
        else:
            with pytest.raises(ValueError, match="tensor_grid.construction|injected handoff failure"):
                _realize_test_grid(
                    prepared, grid, source.domain.cell_regions, MeshingLimits(), scope=scope, budget=budget
                )
    assert len(tokens) == 1
    assert not tokens[0].active
    assert tokens[0].owner_fields == tokens[0].parents == ()


@pytest.mark.parametrize("fault", ("source", "counts", "parents"))
def test_physical_tensor_reauthenticates_callback_visible_inputs(fault: str) -> None:
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry, scope=scope, budget=budget
        )
        grid = _extract_source_block_tensor_grid(
            prepared, np.asarray(((2, 3, 2),), dtype=np.int64),
            source.domain.cell_regions, MeshingLimits(), 6, scope=scope, budget=budget,
        )
        # These owners are callback-visible between reference and physical phases.
        if fault == "source":
            object.__setattr__(prepared.geometry, "coordinates", prepared.geometry.coordinates + 0.125)
        elif fault == "counts":
            grid.root_intervals[0] = (3, 2, 2)
        else:
            grid.source_parent_rows[0] = 1
        with pytest.raises((ValueError, MeshingFailure)):
            _realize_test_grid(
                prepared, grid, source.domain.cell_regions, MeshingLimits(), scope=scope, budget=budget
            )


@contextmanager
def _tensor_association_case(
    intervals: int = 2, *, limits: MeshingLimits | None = None,
) -> Iterator[tuple[Any, Any, Any, Any]]:
    from phydrax.meshing._family_execution import coordinate_phase

    source = curved_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    limits = MeshingLimits() if limits is None else limits
    with native_meshing_request(limits) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh, source.domain.source_geometry,
            scope=scope, budget=budget,
        )
        grid = _extract_source_block_tensor_grid(
            prepared, np.full((2, 3), intervals, dtype=np.int64),
            source.domain.cell_regions, limits, 6, scope=scope, budget=budget,
        )
        inputs: list[Any] = []
        mapped = _extract_source_tensor_frame_grid(
            prepared, grid, source.domain.cell_regions, limits,
            scope=scope, budget=budget, association_inputs=inputs,
        )
        assert len(inputs) == 1
        with coordinate_phase(scope=scope, budget=budget):
            yield source.domain, mapped, inputs.pop(), budget


@pytest.mark.parametrize("intervals", (2, 3))
def test_tensor_associations_match_ordinary_ids_and_evidence(intervals: int) -> None:
    from phydrax.meshing._association import MappedReferenceAssociationTransfer
    from phydrax.meshing._mapped_reference_association import (
        _tensor_reference_associations,
    )

    with _tensor_association_case(intervals) as (domain, mapped, inputs, budget):
        with budget.measurement() as measured:
            actual = _tensor_reference_associations(
                domain, mapped.mesh, mapped.geometry, inputs,
                maximum_support_queries=1_000_000, budget=budget,
            )
        with budget.measurement() as ordinary:
            expected = MappedReferenceAssociationTransfer(domain).associations(
                mapped.mesh, mapped.geometry, budget=budget,
            )
        assert measured.work_units < ordinary.work_units
        print({"compact_work": measured.work_units, "ordinary_work": ordinary.work_units,
               "compact_queries": measured.geometry_queries, "compact_storage": measured.peak_bytes_upper})
        for left, right in zip(actual, expected, strict=True):
            assert left.association_id == right.association_id
            assert left.source_entities.entity_ids() == right.source_entities.entity_ids()
            for field in (
                "target_global_ids", "parent_dimensions", "parent_ids",
                "orientations", "residuals", "resolved", "ambiguous",
            ):
                np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        with pytest.raises(ValueError, match="consumed exactly once"):
            _tensor_reference_associations(
                domain, mapped.mesh, mapped.geometry, inputs,
                maximum_support_queries=1_000_000, budget=budget,
            )


@pytest.mark.parametrize("fault", ("parent", "order", "source", "target", "mutation"))
def test_tensor_association_handoff_rejects_changed_owners(fault: str) -> None:
    from phydrax.meshing._mapped_reference_association import (
        _tensor_reference_associations,
    )

    with _tensor_association_case() as (domain, mapped, inputs, budget):
        mesh = mapped.mesh
        if fault == "parent":
            inputs.parents = tuple(reversed(inputs.parents))
        elif fault == "order":
            object.__setattr__(inputs.theorem, "order", tuple(reversed(inputs.theorem.order)))
        elif fault == "source":
            object.__setattr__(domain.source_geometry, "coordinates", domain.source_geometry.coordinates + 0.1)
        elif fault == "target":
            mesh = mesh.with_coordinates(mesh.coordinates + 0.1, numeric_version="changed")
        else:
            object.__setattr__(mapped.geometry, "coordinates", mapped.geometry.coordinates + 0.1)
        with pytest.raises(ValueError, match="current, unchanged"):
            _tensor_reference_associations(
                domain, mesh, mapped.geometry, inputs,
                maximum_support_queries=1_000_000, budget=budget,
            )


@pytest.mark.parametrize("ambiguous", (False, True))
def test_tensor_association_canonical_coverage_and_uniqueness(ambiguous: bool) -> None:
    from phydrax.meshing._mapped_reference_association import (
        _assemble_mapped_associations,
    )

    with _tensor_association_case() as (domain, mapped, _, budget):
        choices = [[set() for _ in range(mapped.mesh.entity_set(0).count)]]
        if ambiguous:
            choices[0][0] = {(0, 0, 0), (0, 1, 0)}
        with pytest.raises(ValueError, match="source_stratum" if ambiguous else "source_incidence"):
            _assemble_mapped_associations(
                domain, mapped.mesh, domain.reference_mesh, choices, budget=budget,
            )


@pytest.mark.parametrize("tensor", (False, True))
def test_mapped_association_work_and_storage_exact_boundaries(tensor: bool) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing._association import MappedReferenceAssociationTransfer
    from phydrax.meshing._mapped_reference_association import (
        _tensor_reference_associations,
    )

    def associate(domain: Any, mapped: Any, inputs: Any, budget: Any) -> None:
        if tensor:
            _tensor_reference_associations(
                domain, mapped.mesh, mapped.geometry, inputs,
                maximum_support_queries=1_000_000, budget=budget,
            )
        else:
            MappedReferenceAssociationTransfer(
                domain, maximum_support_queries=1_000_000,
            ).associations(mapped.mesh, mapped.geometry, budget=budget)

    with _tensor_association_case() as (domain, mapped, inputs, budget):
        baseline = budget.retained_basis_bytes + budget.temporary_bytes_upper
        with budget.measurement() as measured:
            associate(domain, mapped, inputs, budget)
        work, storage = measured.work_units, baseline + measured.peak_bytes_upper
    for work_limit, storage_limit, fails in (
        (work, storage, False), (work - 1, storage, True), (work, storage - 1, True),
    ):
        with _tensor_association_case() as (domain, mapped, inputs, budget):
            def run() -> None:
                with budget.bound_stage(work_limit, storage_limit):
                    associate(domain, mapped, inputs, budget)
            if fails:
                with pytest.raises((CoordinateEnclosureResourceError, MeshingFailure)):
                    run()
            else:
                run()


@pytest.mark.parametrize("scratch", (536_870_912, 1_536_000_000))
@pytest.mark.parametrize("tensor", (False, True))
def test_mapped_association_inherits_exact_request_scratch(
    scratch: int, tensor: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.meshing._association import MappedReferenceAssociationTransfer
    from phydrax.meshing._mapped_reference_association import (
        _tensor_reference_associations,
    )

    limits = MeshingLimits(maximum_scratch_bytes=scratch)
    with _tensor_association_case(limits=limits) as (domain, mapped, inputs, budget):
        original = CoordinateEnclosureBudget.bound_stage
        admitted: list[tuple[int, int]] = []

        @contextmanager
        def observe(
            self: CoordinateEnclosureBudget,
            work: int,
            memory: int,
            /,
            *,
            starting_work_units: int | None = None,
        ) -> Iterator[None]:
            assert self is budget
            admitted.append((work, memory))
            with original(
                self, work, memory, starting_work_units=starting_work_units,
            ):
                yield

        monkeypatch.setattr(CoordinateEnclosureBudget, "bound_stage", observe)
        ticket = budget.ticket
        if tensor:
            _tensor_reference_associations(
                domain, mapped.mesh, mapped.geometry, inputs,
                maximum_support_queries=1_000_000, budget=budget,
            )
        else:
            MappedReferenceAssociationTransfer(
                domain, maximum_support_queries=1_000_000,
            ).associations(mapped.mesh, mapped.geometry, budget=budget)
        assert admitted
        assert admitted[0] == (1_000_000, scratch)
        assert all(memory == scratch for _, memory in admitted)
        assert budget.ticket is ticket
        assert budget.maximum_memory_bytes == scratch
        assert budget.maximum_work_units == limits.maximum_work_units
