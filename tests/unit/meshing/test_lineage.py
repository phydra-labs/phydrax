import dataclasses
import warnings
from collections.abc import Callable
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.meshing._adaptation import _finalize_native
from phydrax.meshing._topology_edit import (
    CellTopologyEdit,
    EntityRelations,
    TopologyEditBlock,
)


def _source() -> phx.meshing.CellMeshingResult:
    return phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
            np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
            vertex_global_ids=np.asarray((1, 2, 3, 4, 5), dtype=np.int64),
            cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
        ),
        phx.SpatialCoordinateContract.si(),
    )


def test_triangle_transition_lineage_stencil_and_atomic_commit() -> None:
    source = _source()
    mesh = source.mesh
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    transition = adaptation.transition
    assert transition is not None
    stencil = transition.vertex_stencil
    assert stencil is not None
    constant = stencil.apply(mesh.vertex_global_ids, jnp.ones((5,)))
    linear = stencil.apply(mesh.vertex_global_ids, mesh.coordinates[:, 0])
    cells = transition.lineage.entity_lineage(2)
    refined = np.asarray(cells.relation_kinds) == int(
        phx.meshing.EntityLineageKind.REFINED_FROM
    )

    assert jnp.allclose(constant, 1.0)
    assert jnp.allclose(linear, transition.target.mesh.coordinates[:, 0])
    assert transition.lineage.source_topology_id == mesh.topology_id
    assert transition.lineage.target_topology_id == transition.target.mesh.topology_id

    geometry_transition = transition.geometry_transition
    assert geometry_transition is not None
    assert geometry_transition.reference_map is not None
    reference_groups = (
        geometry_transition.reference_map
        if isinstance(geometry_transition.reference_map, tuple)
        else (geometry_transition.reference_map,)
    )
    mapped_cell_ids = np.concatenate(
        tuple(
            np.asarray(group.target_cell_ids, dtype=np.int64)
            for group in reference_groups
        )
    )
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in transition.target.mesh.blocks
        )
    )
    np.testing.assert_array_equal(np.sort(mapped_cell_ids), np.sort(target_cell_ids))
    np.testing.assert_array_equal(np.asarray(cells.source_global_ids)[refined], (10, 10))
    np.testing.assert_array_equal(np.asarray(cells.deleted_source_ids), ())

    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0],),
        0.0,
        0,
        mesh.topology_id,
        "prepared",
        "compiled",
        field_layouts=(layout,),
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: (
            candidate.topology_id == lineage.target_topology_id
            and bool(jnp.all(jnp.isfinite(fields[0])))
        )
    )
    result = transaction.execute(accepted, mesh, adaptation)

    assert bool(result.committed)
    assert result.adaptation is adaptation
    assert result.mesh.topology_id == transition.target.mesh.topology_id
    assert jnp.allclose(result.state.fields[0], transition.target.mesh.coordinates[:, 0])


class _PointwiseTriangleTabulator(StrictModule):
    scale: Array

    def __call__(self, point: Array) -> tuple[Array, Array]:
        x, y = point[0], point[1]
        return (
            self.scale * jnp.stack((1.0 - x - y, x, y)),
            self.scale
            * jnp.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)), dtype=point.dtype),
        )


class _TabulatorChain(StrictModule):
    actions: tuple[Callable[[Array], tuple[Array, Array]], ...]

    def __init__(self, action: Callable[[Array], tuple[Array, Array]]) -> None:
        self.actions = (action,)

    def __call__(self, points: Array) -> tuple[Array, Array]:
        return self.actions[0](points)


@pytest.mark.strict_jax
def test_committed_topology_result_preserves_nested_callback_warning_and_fields() -> None:
    source = _source()
    mesh = source.mesh
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    base = phx.discretization.lagrange_element("triangle", 1)
    tabulator = _TabulatorChain(
        jax.vmap(_PointwiseTriangleTabulator(jnp.asarray(1.0, dtype=jnp.float64)))
    )
    element = phx.discretization.FiniteElementSpec(
        base.family,
        base.cell_kind,
        base.degree,
        base.reference_nodes,
        base.entity_dofs,
        value_spec=base.value_spec,
        continuity=base.continuity,
        representation=base.representation,
        tabulator=tabulator,
        tabulator_id="linear-triangle-callback",
    )
    field = phx.discretization.FiniteElementFieldSpec("u", element)
    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        field,
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    second_field = phx.discretization.FiniteElementFieldSpec("v", element)
    second_layout = phx.solver.FiniteElementAcceptedFieldLayout(
        second_field,
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    space = phx.discretization.FiniteElementPlan(
        mesh, (field, second_field), coordinate_spec=source.geometry
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "topology-callback",
        ("u", "v"),
        (
            phx.equations.DiffusionAction("u", 1.0, action_id="diffusion-u"),
            phx.equations.DiffusionAction("v", 1.0, action_id="diffusion-v"),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0], mesh.coordinates[:, 1]),
        0.0,
        0,
        mesh.topology_id,
        space.prepared_id,
        problem.compilation_id,
        field_layouts=(layout, second_layout),
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: (
            candidate.topology_id == lineage.target_topology_id
            and bool(jnp.allclose(fields[0], candidate.coordinates[:, 0]))
            and bool(jnp.allclose(fields[1], candidate.coordinates[:, 1]))
        )
    )
    with warnings.catch_warnings(record=True) as actual:
        warnings.simplefilter("always")
        result = transaction.execute(accepted, mesh, adaptation)
    assert bool(result.committed)
    assert result.adaptation is adaptation
    assert result.mesh is adaptation.target.mesh
    assert result.state.field_layouts[0] is layout
    assert result.state.field_layouts[1] is second_layout
    assert result.state.state_version == accepted.state_version + 1
    assert jnp.allclose(result.state.fields[0], result.mesh.coordinates[:, 0])
    assert jnp.allclose(result.state.fields[1], result.mesh.coordinates[:, 1])
    fields = {
        field.name: getattr(result, field.name) for field in dataclasses.fields(result)
    }
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        direct = phx.solver.FiniteElementTopologyResult(**fields)
    assert [(str(item.message), item.category) for item in actual] == [
        (str(item.message), item.category) for item in expected
    ]
    assert len(actual) == 2
    actual_leaves, actual_tree = jax.tree.flatten(result)
    direct_leaves, direct_tree = jax.tree.flatten(direct)
    assert any(leaf is result.state.fields[0] for leaf in actual_leaves)
    assert actual_tree == direct_tree
    assert all(
        left is right for left, right in zip(actual_leaves, direct_leaves, strict=True)
    )
    assert all(getattr(direct, name) is value for name, value in fields.items())
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "diagnostics", "changed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="JAX-transformed callable"):
            transaction.execute(accepted, mesh, adaptation)


def test_non_dyadic_stencil_weights_remain_exact_and_unique() -> None:
    exact = ((Fraction(1, 3), Fraction(2, 3)),)
    sources = np.asarray(((5, 7),), dtype=np.int64)
    weights = np.asarray(((float(exact[0][0]), float(exact[0][1])),), dtype=np.float64)
    valid = np.ones(sources.shape, dtype=np.bool_)
    stencil = phx.meshing.VertexInterpolationStencil(
        "source-vertices",
        "target-vertices",
        np.asarray((11,), dtype=np.int64),
        sources,
        weights,
        valid,
        exact_weights=exact,
    )

    assert stencil.preserves_constants
    assert stencil.exact_weights == exact
    assert (
        sum((Fraction.from_float(float(weight)) for weight in weights[0]), Fraction(0))
        != 1
    )
    assert len(set(np.asarray(stencil.source_global_ids)[0])) == 2
    expected = float(exact[0][0] * 2 + exact[0][1] * 4)
    value = np.asarray(stencil.apply((5, 7), np.asarray((2.0, 4.0))))
    np.testing.assert_allclose(value, (expected,), rtol=0.0, atol=1e-15)

    transfer = stencil.as_transfer(
        (5, 7),
        source_topology_id="source-topology",
        target_topology_id="target-topology",
    )
    assert transfer.preserves_constants
    np.testing.assert_allclose(transfer.apply(np.ones((2,))), np.ones((1,)))

    with pytest.raises(ValueError, match="one exact coefficient per source ID"):
        phx.meshing.VertexInterpolationStencil(
            "source-vertices",
            "target-vertices",
            np.asarray((12,), dtype=np.int64),
            np.asarray(((5, 5),), dtype=np.int64),
            np.asarray(((0.25, 0.75),), dtype=np.float64),
            np.ones((1, 2), dtype=np.bool_),
            exact_weights=((Fraction(1, 4), Fraction(3, 4)),),
        )


def test_owner_local_p0_uses_many_to_many_physical_cell_lineage() -> None:
    from dataclasses import replace

    from phydrax.meshing._lineage import EntityLineage, EntityLineageKind, MeshLineage
    from phydrax.solver._finite_element_adaptivity import (
        _FiniteElementOwnerLocalAdaptiveMetadata,
        _owner_local_cell_average_rows,
    )

    # One unequal-measure family coarsens while another cell refines.
    source_ids = np.asarray((10, 11, 20), dtype=np.int64)
    target_ids = np.asarray((1, 21, 22), dtype=np.int64)
    lineage = MeshLineage(
        "source-topology",
        "target-topology",
        (
            EntityLineage(
                2,
                "source-cells",
                "target-cells",
                (10, 11, 20, 20),
                (1, 1, 21, 22),
                (
                    EntityLineageKind.COARSENED_INTO,
                    EntityLineageKind.COARSENED_INTO,
                    EntityLineageKind.REFINED_FROM,
                    EntityLineageKind.REFINED_FROM,
                ),
            ),
        ),
    )
    metadata = _FiniteElementOwnerLocalAdaptiveMetadata(
        "result",
        "candidate",
        lineage,
        None,
        source_ids,
        target_ids,
        (Fraction(1, 3), Fraction(2, 3), Fraction(2)),
        (Fraction(1), Fraction(1, 2), Fraction(3, 2)),
        None,
    )
    rows = _owner_local_cell_average_rows(metadata, 2, target_ids)
    assert rows == (
        ((10, Fraction(1, 3)), (11, Fraction(2, 3))),
        ((20, Fraction(1)),),
        ((20, Fraction(1)),),
    )
    source_values = {10: Fraction(2), 11: Fraction(8), 20: Fraction(5)}
    values = tuple(
        sum((source_values[i] * w for i, w in row), Fraction(0)) for row in rows
    )
    assert values == (6, 5, 5)
    assert sum(
        (
            value * measure
            for value, measure in zip(values, metadata.target_cell_measures, strict=True)
        ),
        Fraction(0),
    ) == sum(
        (
            source_values[int(i)] * measure
            for i, measure in zip(source_ids, metadata.source_cell_measures, strict=True)
        ),
        Fraction(0),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        _owner_local_cell_average_rows(
            replace(
                metadata,
                target_cell_measures=(Fraction(2), Fraction(1, 2), Fraction(3, 2)),
            ),
            2,
            target_ids,
        )


def test_conservative_cell_average_uses_overlap_and_rolls_back() -> None:
    source = _source()
    mesh = source.mesh
    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "mass", phx.discretization.fem.discontinuous_element("triangle", 0)
        ),
        transfer_semantics=(
            phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
        ),
    )
    restored_layout = phx.solver.FiniteElementAcceptedFieldLayout.from_recipe(
        layout.recipe()
    )
    assert restored_layout.layout_id == layout.layout_id
    assert (
        restored_layout.transfer_semantics
        is phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
    )
    with pytest.raises(ValueError, match="conservative_cell_average"):
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec(
                "not-cell-average", phx.discretization.lagrange_element("triangle", 1)
            ),
            transfer_semantics=(
                phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
            ),
        )
    dof_map = (
        phx.discretization.fem.FiniteElementPlan(
            mesh,
            (layout.field_spec,),
            coordinate_spec=source.geometry,
        )
        .prepare()
        .dof_maps[0]
    )
    field = np.arange(1, dof_map.global_dof_count + 1, dtype=np.float64)
    accepted = phx.solver.FiniteElementAcceptedState(
        (field,),
        0.0,
        0,
        mesh.topology_id,
        "prepared-conservative",
        "compiled-conservative",
        field_layouts=(layout,),
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    certify = lambda candidate, fields, materials, lineage, args: (
        candidate.topology_id == lineage.target_topology_id
        and bool(jnp.all(jnp.isfinite(fields[0])))
    )
    result = phx.solver.FiniteElementTopologyTransaction(certify).execute(
        accepted, mesh, adaptation
    )
    assert bool(result.committed)
    (evidence,) = result.conservative_transfers
    assert evidence.transfer.conservative
    assert evidence.coverage_complete
    assert np.all(
        np.abs(np.asarray(evidence.conservation_defect)) <= evidence.conservation_bound
    )
    np.testing.assert_allclose(evidence.source_integral, evidence.target_integral)
    np.testing.assert_array_equal(
        result.state.fields[0], evidence.transfer.apply(accepted.fields[0])
    )

    refused = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: False
    ).execute(accepted, mesh, adaptation)
    assert not bool(refused.committed)
    assert refused.state is accepted
    assert refused.adaptation is None
    assert refused.conservative_transfers[0].transfer.transfer_id == (
        evidence.transfer.transfer_id
    )


def test_parentless_retriangulation_withholds_vertex_transfer() -> None:
    source = _source()
    mesh = source.mesh
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
        ),
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids[:4], dtype=np.int64)
    target_cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    empty_relations = tuple(
        EntityRelations(
            dimension,
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )
        for dimension, width in ((0, 1), (1, 2), (2, 1))
    )
    edit = CellTopologyEdit(
        coordinates=np.asarray(mesh.coordinates[:4], dtype=np.float64),
        vertex_global_ids=vertex_ids,
        blocks=(
            TopologyEditBlock(
                mesh.blocks[0].name,
                "triangle",
                "triangle",
                target_cells,
                np.asarray((101, 102), dtype=np.int64),
            ),
        ),
        stencil_sources=vertex_ids[:, None],
        stencil_weights=np.ones((4, 1), dtype=np.float64),
        stencil_valid=np.ones((4, 1), dtype=np.bool_),
        relations=empty_relations,
    )

    target = _finalize_native(
        prepared,
        edit,
        phx.meshing.MeshTransitionKind.REMESH,
        conservative=False,
    )

    assert target.stencil.preserves_constants
    assert target.transfer is None
    cell_lineage = target.lineage.entity_lineage(2)
    np.testing.assert_array_equal(np.asarray(cell_lineage.created_target_ids), (101, 102))


def test_material_field_requires_its_declared_chain_action() -> None:
    source = _source()
    mesh = source.mesh
    with pytest.raises(ValueError, match="explicitly declare transfer_semantics"):
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec(
                "ambiguous", phx.discretization.lagrange_element("triangle", 1)
            )
        )
    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "material_content", phx.discretization.lagrange_element("triangle", 1)
        ),
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT,
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0],),
        0.0,
        0,
        mesh.topology_id,
        "prepared-material",
        "compiled-material",
        field_layouts=(layout,),
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    result = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: True
    ).execute(accepted, mesh, adaptation)

    assert not bool(result.committed)
    assert result.state is accepted
    assert result.adaptation is None
    assert (
        result.diagnostics == "material-following-field-requires-validated-material-chain"
    )


@pytest.mark.parametrize("commit", (True, False), ids=("commit", "rollback"))
def test_declared_form_transfer_preserves_linear_state_and_evidence(commit: bool) -> None:
    source = _source()
    mesh = source.mesh
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "potential",
            phx.discretization.fem.form_element(
                "triangle", 0, 1, family="trimmed", twist="untwisted"
            ),
        ),
        periodic_quotient=False,
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0] + 2.0 * mesh.coordinates[:, 1],),
        0.0,
        0,
        mesh.topology_id,
        "prepared-form",
        "compiled-form",
        field_layouts=(layout,),
    )

    def certify(
        candidate_mesh: phx.discretization.CellMesh,
        fields: tuple[Array, ...],
        materials: phx.equations.MaterialTransaction | None,
        lineage: phx.meshing.MeshLineage,
        args: object,
    ) -> bool:
        return commit

    result = phx.solver.FiniteElementTopologyTransaction(certify).execute(
        accepted, mesh, adaptation
    )
    assert bool(result.committed) is commit
    expected_mesh = adaptation.target.mesh if commit else mesh
    np.testing.assert_allclose(
        result.state.fields[0],
        expected_mesh.coordinates[:, 0] + 2.0 * expected_mesh.coordinates[:, 1],
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert result.mesh.topology_id == expected_mesh.topology_id
    if not commit:
        assert result.state.accepted_id == accepted.accepted_id
    (transfer,) = result.field_transfers
    assert transfer.complex_map.source.space(0).size == mesh.coordinates.shape[0]
    assert (
        transfer.complex_map.target.space(0).size
        == adaptation.target.mesh.coordinates.shape[0]
    )
    assert np.all(
        np.asarray(transfer.commuting_defects)
        <= np.asarray(transfer.commuting_error_bounds)
    )
    assert transfer.enclosed_integral_count == 0
