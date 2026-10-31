from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh, PolygonalConnectivity
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._surface_chart_records import (
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from phydrax.discretization.finite_volume._material_chain_action import (
    material_chain_apply,
    material_chain_transpose,
    prepare_material_chain_conservative_remap,
)
from phydrax.discretization.finite_volume._material_schedule import (
    prepare_material_chain_schedule,
)
from phydrax.discretization.finite_volume._remap_csr_view import material_csr_rows
from phydrax.geometry._common_root_volume_overlap import CommonRootVolumeWitness
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
)
from phydrax.geometry._volume_material_deformation import (
    declare_material_reference,
    MaterialReferencePolicy,
)
from phydrax.geometry._volume_material_map import prepare_eulerian_volume_overlap
from phydrax.geometry.brep._patches import AbstractSurfacePatch, LineCurve
from phydrax.meshing._adaptation import (
    _renew_transition_certification_inputs,
    _unknown_lineage,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._bisection import BisectionHierarchy
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._lineage import CellMeshTransition, MeshTransitionKind
from phydrax.meshing._motion import advance_mesh_motion
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import (
    NativeCurveSchedule,
    NativeSurfaceSchedule,
)
from phydrax.meshing.providers._native_surface import prepare_native_surface
from phydrax.solver._finite_element_adaptivity import (
    FiniteElementFieldTransferContext,
)
from phydrax.solver._finite_element_schedule import (
    FiniteElementAcceptedFieldLayout,
    FiniteElementAcceptedState,
    FiniteElementFieldTransferSemantics,
)
from tests._support.volume_material_map_fixtures import (
    make_curved_material_refinement,
    make_material_coverage_pair,
    make_material_source,
)


def _certified_square() -> tuple[CellMeshingResult, PiecewiseLinearDomain]:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)
    mesh = CellMesh.from_triangles(coordinates, cells)
    domain = PiecewiseLinearDomain(
        coordinates[:4],
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("material",),
        source_id="transition-square",
    )
    inputs = MeshCertificationInputs(
        mesh,
        CellGeometrySpec.affine(mesh),
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.zeros((len(cells),), dtype=np.int64),
        junction_vertices=np.zeros((0,), dtype=np.int64),
        protected_features=(),
    )
    result = certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=CellGeometrySpec.affine(mesh),
        certification_inputs=inputs,
    )
    return result, domain


def _bisection_hierarchy(
    adaptation: MeshAdaptationResult,
    /,
) -> BisectionHierarchy:
    hierarchy = adaptation.hierarchy
    if not isinstance(hierarchy, BisectionHierarchy):
        raise ValueError("Adaptation did not retain its bisection hierarchy.")
    return hierarchy


def test_pwl_refinement_and_coarsening_renew_original_domain_acceptance() -> None:
    source, domain = _certified_square()
    assert source.certification is not None and source.certification.passed
    route = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    source_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks
        )
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(source_ids),
            policy=route,
        )
    )
    refined_certification = refined.target.certification
    assert refined_certification is not None and refined_certification.passed
    assert refined_certification.inputs.domain is domain
    assert refined_certification.inputs.cell_regions is not None
    np.testing.assert_array_equal(
        refined_certification.inputs.cell_regions,
        np.zeros(refined_certification.inputs.cell_regions.shape, dtype=np.int64),
    )

    refined_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in refined.target.mesh.blocks
        )
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                (), refined_ids, hierarchy=_bisection_hierarchy(refined)
            ),
            policy=route,
            prior_transition=refined.transition,
        )
    )
    with pytest.raises(ValueError, match="prior_transition target"):
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                (), refined_ids, hierarchy=_bisection_hierarchy(refined)
            ),
            policy=route,
            prior_transition=refined.transition,
        )
    coarsened_certification = coarsened.target.certification
    assert coarsened_certification is not None and coarsened_certification.passed
    assert coarsened_certification.inputs.domain is domain
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    source.certification.revalidate(source.mesh, source.geometry)


def _curved_material_refinement_case() -> tuple[
    CellMeshingResult,
    object,
    CellMeshTransition,
    MeshAdaptationPolicy,
    np.ndarray,
    MeshAdaptationResult,
    object,
    tuple[str, str],
]:
    """Certified curved two-tetrahedron source and its material-following refinement."""
    (
        root_mesh,
        root_geometry,
        _,
        source_mesh,
        source_geometry,
        _,
        _,
    ) = make_curved_material_refinement(exact_chart_arguments=True)
    occurrence = ("assembly-1", "occurrence-9")
    domain, _, _ = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        region_id="aluminum",
        occurrence_path=occurrence,
    )
    inputs = MeshCertificationInputs(
        source_mesh,
        source_geometry,
        MeshCertificationSchedule("volume_plc"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.zeros(
            (sum(block.cell_count for block in source_mesh.blocks),), dtype=np.int64
        ),
    )
    source = certify_cell_mesh(
        source_mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=source_geometry,
        certification_inputs=inputs,
    )
    assert source.certification is not None and source.certification.coverage is not None
    reference = declare_material_reference(
        source.mesh,
        source.geometry,
        make_material_source(domain, occurrence_path=occurrence),
        ("solid-cell", "solid-cell"),
        ("aluminum", "aluminum"),
        source.certification.coverage,
        frame_id=source.coordinate_contract.spatial_id,
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(occurrence, occurrence),
    )
    prior = CellMeshTransition(
        root_mesh.mesh_id,
        root_mesh.topology_id,
        source,
        _unknown_lineage(root_mesh, source.mesh),
        MeshTransitionKind.REMESH,
        material_reference=reference,
    )
    route = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    source_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks
        )
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(source_ids),
            policy=route,
            material_reference=reference,
        )
    )
    return source, reference, prior, route, source_ids, refined, domain, occurrence


def _tetrahedral_fv_plan(
    result: CellMeshingResult, component_names: tuple[str, ...], /
) -> UnstructuredFiniteVolumePlan:
    return UnstructuredFiniteVolumePlan(
        np.asarray(result.mesh.coordinates),
        tetrahedra=np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int32) for block in result.mesh.blocks
            )
        ),
        vertex_global_ids=np.asarray(result.mesh.vertex_global_ids, dtype=np.int64),
        cell_global_ids=np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in result.mesh.blocks
            )
        ),
        component_names=component_names,
    )


def test_curved_volume_material_chain_renews_exact_refinement_and_coarsening() -> None:
    (
        source,
        reference,
        prior,
        route,
        source_ids,
        refined,
        domain,
        occurrence,
    ) = _curved_material_refinement_case()
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(()),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
            prior_transition=prior,
            material_reference=reference,
        )
    refined_transition = refined.transition
    assert refined_transition is not None
    refined_chain = refined_transition.volume_material_chain
    assert refined_chain is not None and len(refined_chain.maps) == 1
    assert refined_chain.source_mesh_id == source.mesh.mesh_id
    assert refined_chain.target_mesh_id == refined.target.mesh.mesh_id
    assert refined_transition.material_reference is not None
    assert (
        refined_transition.material_reference.declaration_id == reference.declaration_id
    )

    def root_reference_maps(
        geometry: CellGeometrySpec, /
    ) -> dict[str, ReferenceCellMapWitnesses]:
        restriction = geometry.restriction_source
        assert restriction is not None
        assert restriction.source_topology_id == root_mesh.topology_id
        maps = {}
        for name, reference_map in zip(
            restriction.block_names, restriction.parent_reference_maps, strict=True
        ):
            assert reference_map is not None
            maps[name] = reference_map
        return maps

    source_count = sum(block.cell_count for block in source.mesh.blocks)
    target_count = sum(block.cell_count for block in refined.target.mesh.blocks)
    assert refined.target.certification is not None
    assert refined.target.certification.coverage is not None
    eulerian_overlap = prepare_eulerian_volume_overlap(
        source.mesh,
        source.geometry,
        refined.target.mesh,
        refined.target.geometry,
        make_material_source(domain, occurrence_path=occurrence),
        ("solid-cell",) * source_count,
        ("solid-cell",) * target_count,
        ("aluminum",) * source_count,
        ("aluminum",) * target_count,
        source_coverage=source.certification.coverage,
        target_coverage=refined.target.certification.coverage,
        source_occurrence_paths=(occurrence,) * source_count,
        target_occurrence_paths=(occurrence,) * target_count,
        common_root=CommonRootVolumeWitness(
            root_mesh,
            root_geometry,
            source.mesh,
            source.geometry,
            refined.target.mesh,
            refined.target.geometry,
            root_reference_maps(source.geometry),
            root_reference_maps(refined.target.geometry),
        ),
    )
    assert eulerian_overlap.refinement.succeeded
    source_measures = np.asarray(eulerian_overlap.refinement.source_measures)
    target_measures = np.asarray(eulerian_overlap.refinement.target_measures)
    np.testing.assert_allclose(
        np.sum(target_measures), np.sum(source_measures), rtol=1e-13, atol=0.0
    )

    material_semantics = FiniteElementFieldTransferSemantics.MATERIAL_FOLLOWING_CONTENT
    eulerian_semantics = FiniteElementFieldTransferSemantics.EULERIAN_PHYSICAL_OVERLAP
    nested_semantics = FiniteElementFieldTransferSemantics.NESTED_FE
    interpolation_semantics = (
        FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION
    )
    material_layout = FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "material-content",
            phx.discretization.discontinuous_element("tetrahedron", 0),
        ),
        transfer_semantics=material_semantics,
    )
    eulerian_layout = FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "eulerian-average",
            phx.discretization.discontinuous_element("tetrahedron", 0),
        ),
        transfer_semantics=eulerian_semantics,
    )
    compatible_layout = FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "circulation",
            phx.discretization.form_element("tetrahedron", 1, 1, proxy="circulation"),
        ),
        periodic_quotient=False,
        transfer_semantics=nested_semantics,
    )
    interpolation_layout = FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "nodal",
            phx.discretization.lagrange_element("tetrahedron", 1),
        ),
        transfer_semantics=interpolation_semantics,
    )
    source_cell_content = jnp.asarray((2.0, 8.0), dtype=jnp.float64)
    source_cell_average = jnp.asarray((2.0, 8.0), dtype=jnp.float64)
    source_circulation = jnp.ones(
        (source.mesh.entity_set(1).entity_ids.size,), dtype=jnp.float64
    )
    source_nodal = jnp.full((source.mesh.vertex_global_ids.size,), 7.0, dtype=jnp.float64)
    accepted = FiniteElementAcceptedState(
        (source_cell_content, source_cell_average, source_circulation, source_nodal),
        0.0,
        0,
        source.mesh.topology_id,
        "mixed-transfer-prepared",
        "mixed-transfer-compiled",
        field_layouts=(
            material_layout,
            eulerian_layout,
            compatible_layout,
            interpolation_layout,
        ),
    )
    selected_actions: list[str] = []

    def material_field_transfer(
        field: Array,
        layout: FiniteElementAcceptedFieldLayout,
        adaptation: FiniteElementFieldTransferContext,
        args: object,
    ) -> Array:
        del args
        if layout.transfer_semantics is not material_semantics:
            raise ValueError("Material transfer received another field semantics.")
        if not isinstance(adaptation, MeshAdaptationResult):
            raise ValueError("Material transfer requires a mesh adaptation result.")
        transition = adaptation.transition
        if transition is None or transition.volume_material_chain is None:
            raise ValueError("Material transfer requires the accepted material chain.")
        from_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in adaptation.source.mesh.blocks
            )
        )
        to_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in adaptation.target.mesh.blocks
            )
        )
        schedule = prepare_material_chain_schedule(
            transition.volume_material_chain, from_ids, to_ids
        )
        selected_actions.append("material")
        return material_chain_apply(schedule, field)

    def eulerian_field_transfer(
        field: Array,
        layout: FiniteElementAcceptedFieldLayout,
        adaptation: FiniteElementFieldTransferContext,
        args: object,
    ) -> Array:
        del adaptation, args
        if layout.transfer_semantics is not eulerian_semantics:
            raise ValueError("Eulerian transfer received another field semantics.")
        selected_actions.append("eulerian")
        return eulerian_overlap.apply_intensive(field)

    def nonconservative_field_transfer(
        field: Array,
        layout: FiniteElementAcceptedFieldLayout,
        adaptation: FiniteElementFieldTransferContext,
        args: object,
    ) -> Array:
        del args
        if not isinstance(adaptation, MeshAdaptationResult):
            raise ValueError("Interpolation requires a mesh adaptation result.")
        stencil = adaptation.stencil
        if stencil is None:
            raise ValueError("Nonconservative interpolation requires vertex lineage.")
        selected_actions.append("nonconservative")
        return stencil.apply(adaptation.source.mesh.vertex_global_ids, field)

    mixed = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: True,
        material_field_transfer=material_field_transfer,
        eulerian_field_transfer=eulerian_field_transfer,
        nonconservative_field_transfer=nonconservative_field_transfer,
    ).execute(accepted, source.mesh, refined)
    assert bool(mixed.committed)
    assert selected_actions[:2] == ["material", "eulerian"]
    assert selected_actions[2:] in ([], ["nonconservative"])
    material_target = np.asarray(mixed.state.fields[0])
    eulerian_target = np.asarray(mixed.state.fields[1])
    np.testing.assert_allclose(np.sum(material_target), np.sum(source_cell_content))
    np.testing.assert_allclose(
        np.sum(eulerian_target * target_measures),
        np.sum(np.asarray(source_cell_average) * source_measures),
        rtol=1e-13,
        atol=0.0,
    )
    assert not np.array_equal(material_target, eulerian_target)
    np.testing.assert_allclose(
        mixed.state.fields[3],
        np.full((refined.target.mesh.vertex_global_ids.size,), 7.0),
        rtol=1e-13,
        atol=1e-13,
    )
    assert mixed.field_transfers
    assert np.all(
        np.asarray(mixed.field_transfers[0].commuting_defects)
        <= np.asarray(mixed.field_transfers[0].commuting_error_bounds)
    )
    refined_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in refined.target.mesh.blocks
        )
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                (), refined_ids, hierarchy=_bisection_hierarchy(refined)
            ),
            policy=route,
            prior_transition=refined_transition,
        )
    )
    coarsened_transition = coarsened.transition
    assert coarsened_transition is not None
    chain = coarsened_transition.volume_material_chain
    assert chain is not None and len(chain.maps) == 2
    assert chain.source_mesh_id == source.mesh.mesh_id
    assert chain.target_mesh_id == coarsened.target.mesh.mesh_id
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    assert coarsened.target.certification is not None
    assert coarsened.target.certification.inputs.domain is domain
    assert chain.reference.declaration_id == reference.declaration_id
    assert chain.maps[-1].deformation.reference_witness is not None
    assert chain.maps[-1].source_occurrence_paths == (occurrence,) * len(refined_ids)
    assert chain.maps[-1].target_occurrence_paths == (occurrence, occurrence)
    target_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in coarsened.target.mesh.blocks
        )
    )
    schedule = prepare_material_chain_schedule(chain, source_ids, target_ids)
    source_contents = jnp.asarray(
        ((2.0**900, 2.0**899), (2.0**899, 2.0**898)), dtype=jnp.float64
    )
    carried_contents = np.asarray(material_chain_apply(schedule, source_contents))
    source_values = np.asarray(source_contents)
    csr_offsets, csr_sources, csr_weights = material_csr_rows(
        schedule, phx.linalg.SparseAssemblyPolicy()
    )
    csr_contents = np.zeros_like(carried_contents)
    for target_row in range(schedule.target_count):
        for position in range(
            int(csr_offsets[target_row]), int(csr_offsets[target_row + 1])
        ):
            csr_contents[target_row] += (
                csr_weights[position] * source_values[csr_sources[position]]
            )
    np.testing.assert_allclose(csr_contents, carried_contents, rtol=1e-15, atol=0.0)
    cotangents = jnp.asarray(((0.5, 0.25), (0.25, 0.5)), dtype=jnp.float64)
    transposed = np.asarray(material_chain_transpose(schedule, cotangents))
    np.testing.assert_allclose(
        np.vdot(carried_contents, np.asarray(cotangents)),
        np.vdot(source_values, transposed),
        rtol=1e-15,
        atol=0.0,
    )
    assert schedule.map_count == 2
    source_regions = chain.maps[0].source_region_ids
    target_regions = chain.maps[-1].target_region_ids
    for region in sorted(set(source_regions)):
        source_rows = tuple(
            row for row, value in enumerate(source_regions) if value == region
        )
        target_rows = tuple(
            row for row, value in enumerate(target_regions) if value == region
        )
        for material in range(source_values.shape[1]):
            expected = sum(
                (
                    Fraction.from_float(float(source_values[row, material]))
                    for row in source_rows
                ),
                Fraction(0),
            )
            actual = sum(
                (
                    Fraction.from_float(float(carried_contents[row, material]))
                    for row in target_rows
                ),
                Fraction(0),
            )
            # Non-dyadic overlap weights (a source split into 48 cells) make the
            # binary64 chain conservative to rounding only: each edit rounds every
            # edge product and every row sum once.
            rounding = Fraction(4 * schedule.map_count * schedule.maximum_edges, 2**53)
            assert abs(actual - expected) <= rounding * abs(expected)
    source.certification.revalidate(source.mesh, source.geometry)


def _euler_fv_runtime(
    result: CellMeshingResult, system: phx.equations.EulerSystem, /
) -> tuple[object, object]:
    discretization = _tetrahedral_fv_plan(result, system.component_names).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "material-remesh", "state", system, boundaries
        ),
        discretization,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
            topology_event_capacity=4,
            topology_event_policy="accepted_step",
        ),
    ).dynamics
    return discretization, dynamics


def test_fv_commit_remesh_routes_material_and_eulerian_content() -> None:
    source, _, prior, _, _, refined, _, _ = _curved_material_refinement_case()
    transition = refined.transition
    assert transition is not None and transition.volume_material_chain is not None
    system = phx.equations.EulerSystem(3)
    source_geometry, source_dynamics = _euler_fv_runtime(source, system)
    target_geometry, target_dynamics = _euler_fv_runtime(refined.target, system)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        source_dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    primitive = jnp.asarray(
        ((1.0, 0.1, 0.0, -0.05, 1.0), (2.0, -0.2, 0.05, 0.0, 1.5)), dtype=jnp.float64
    )
    accepted = runtime.advance(
        runtime.initialize_state(system.primitive_to_conserved(primitive), 0.0, 1.0e-4)
    )
    assert bool(np.asarray(accepted.accepted))
    source_content = accepted.runtime_state.content_state
    source_total = np.sum(np.asarray(source_content.conservative_content), axis=0)

    material = runtime.commit_remesh(
        accepted,
        transition,
        target_dynamics,
        content_semantics="material_following_content",
    )
    assert material.successor_runtime is not None
    material_epoch = material.runtime_state.topology_journal.epoch_table[-1]
    assert material_epoch.index == 1
    assert material_epoch.topology_id == target_geometry.topology_id
    oracle = prepare_material_chain_conservative_remap(
        transition.volume_material_chain,
        np.asarray(source_geometry.cell_global_ids, dtype=np.int64),
        np.asarray(target_geometry.cell_global_ids, dtype=np.int64),
        source_geometry.cell_volumes,
        target_geometry.cell_volumes,
    )
    material_content = np.asarray(
        material.runtime_state.content_state.conservative_content
    )
    np.testing.assert_allclose(
        material_content,
        np.asarray(oracle.apply_content(source_content.conservative_content)),
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(material_content, axis=0), source_total, rtol=1.0e-14
    )
    continued = material.successor_runtime.advance(material.runtime_state)
    assert bool(np.asarray(continued.accepted))

    eulerian = runtime.commit_remesh(accepted, transition, target_dynamics)
    assert eulerian.successor_runtime is not None
    overlap = prepare_unstructured_conservative_remap(
        source_geometry, target_geometry, provenance="eulerian-oracle"
    ).plan
    assert overlap is not None
    np.testing.assert_allclose(
        np.asarray(eulerian.runtime_state.content_state.cell_average()),
        np.asarray(overlap.apply(source_content.cell_average())),
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(
            np.asarray(eulerian.runtime_state.content_state.conservative_content),
            axis=0,
        ),
        source_total,
        rtol=1.0e-14,
    )

    assert prior.volume_material_chain is None
    with pytest.raises(ValueError, match="accepted volume material chain"):
        runtime.commit_remesh(
            accepted,
            prior,
            target_dynamics,
            content_semantics="material_following_content",
        )


def test_fixed_topology_motion_reproves_source_coverage_or_rolls_back() -> None:
    source, domain = _certified_square()
    policy = phx.meshing.MeshMotionMonitorPolicy(
        relocation_jacobian_ratio=0.01,
        remesh_jacobian_ratio=0.005,
        relocation_quality_ratio=0.01,
        remesh_quality_ratio=0.005,
        maximum_boundary_residual=0.0,
    )
    monitor = phx.meshing.MeshMotionMonitor(source.mesh, policy=policy)
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64).copy()
    coordinates[4] += np.asarray((0.03, -0.02))
    moved = advance_mesh_motion(monitor, source, coordinates, boundary_residual=0.0)
    assert moved.decision is phx.meshing.MeshMotionDecision.ACCEPT_MOTION
    assert moved.result is not None and moved.result.certification is not None
    assert moved.result.certification.passed
    assert moved.result.certification.inputs.domain is domain

    assert moved.source_mesh_id == source.mesh.mesh_id
    assert moved.source_result_id == source.result_id
    assert moved.geometry_transition is not None
    assert moved.geometry_transition.source_mesh.mesh_id == source.mesh.mesh_id
    assert moved.geometry_transition.target_mesh.mesh_id == moved.result.mesh.mesh_id

    changed_boundary = np.asarray(source.mesh.coordinates, dtype=np.float64).copy()
    changed_boundary[0, 0] -= 0.04
    rejected = advance_mesh_motion(
        monitor,
        source,
        changed_boundary,
        boundary_residual=0.0,
        fixed_vertices=np.zeros((changed_boundary.shape[0],), dtype=np.bool_),
    )
    assert rejected.decision is phx.meshing.MeshMotionDecision.REJECT
    assert rejected.result is None
    refusal = rejected.assessments[-1]
    assert refusal.source_certification is not None
    assert not refusal.source_certification.passed
    assert "source-acceptance" in refusal.reasons
    assert rejected.source_mesh_id == source.mesh.mesh_id
    assert rejected.source_result_id == source.result_id
    assert rejected.geometry_transition is not None
    assert rejected.geometry_transition.source_mesh.mesh_id == source.mesh.mesh_id
    assert rejected.geometry_transition.target_mesh.mesh_id != source.mesh.mesh_id
    assert source.certification is not None
    source.certification.revalidate(source.mesh, source.geometry)


class _TinyTrianglePatch(AbstractSurfacePatch):
    @property
    def parameter_domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.0, 0.0), (1.0, 1.0))

    def evaluate(self, parameters: jnp.ndarray, /) -> jnp.ndarray:
        uv = jnp.asarray(parameters, dtype=jnp.float64)
        return jnp.stack(
            (
                0.1 * uv[..., 0],
                0.1 * uv[..., 1],
                0.002 * uv[..., 0] * uv[..., 1],
            ),
            axis=-1,
        )


def _tiny_chart_domain() -> MeshingDomain:
    chart_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    endpoint_pairs = ((0, 1), (1, 2), (2, 0))
    curves = tuple(MeshingDomainCurve(first, last) for first, last in endpoint_pairs)
    uses = tuple(
        PatchCurveUse(
            curve,
            LineCurve(
                np.asarray(chart_points[first], dtype=np.float64),
                np.asarray(chart_points[last], dtype=np.float64)
                - np.asarray(chart_points[first], dtype=np.float64),
            ),
            0.0,
            1.0,
            sense=1,
        )
        for curve, (first, last) in enumerate(endpoint_pairs)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(_TinyTrianglePatch(), (uses,)),),
        curves,
        3,
        source_id="transition-chart-triangle",
        source_revision="chart-r1",
        regions=(MeshingDomainRegion("material", ((0, 1),)),),
    )


def tiny_chart_source() -> tuple[MeshingDomain, CellMeshingResult]:
    """Certified one-triangle parametric chart surface with exact curve intervals."""
    domain = _tiny_chart_domain()
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(
            UniformSizeControl(scope, 2.0, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=0.02,
    )
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(
                quality_angle_degrees=20.0,
                spacing_fraction=0.95,
                curve_schedule=NativeCurveSchedule(fidelity_samples=5),
            ),
        ),
        phx.SpatialCoordinateContract.si(),
    ).execute()
    request = construction.certification
    inputs = MeshCertificationInputs(
        construction.mesh,
        construction.geometry,
        request.schedule,
        source_id=request.source_id,
        source_revision=request.source_revision,
        domain=request.domain,
        cell_regions=request.cell_regions,
        source=request.fidelity_source,
        fidelity_tolerance=request.fidelity_tolerance,
        maximum_normal_deviation=request.maximum_normal_deviation,
        junction_vertices=request.junction_vertices,
        protected_features=request.protected_features,
        scoped_fidelity=request.scoped_fidelity,
        fidelity_witness=request.fidelity_witness,
        feature_source=request.feature_source,
        source_coverage_witness=request.source_coverage_witness,
    )
    organization = construction.organization
    source = certify_cell_mesh(
        construction.mesh,
        construction.coordinate_contract,
        geometry=construction.geometry,
        certification_inputs=inputs,
        boundary=organization.boundary,
        patches=organization.patches,
        zones=organization.zones,
        labels=organization.labels,
        attributes=organization.attributes,
        associations=organization.associations,
        surface_source=organization.surface_source,
        region_boundary_evidence=organization.region_boundary_evidence,
        planar_embedding=construction.planar_embedding,
    )
    return domain, source


def test_refinement_rebinds_exact_surface_chart_root_and_piece_ancestry() -> None:
    domain, source = tiny_chart_source()
    assert source.certification is not None and source.certification.passed
    original = source.certification.inputs.fidelity_witness
    assert isinstance(original, SurfaceChartWitness)
    cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks
        )
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(cell_ids),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=phx.meshing.SurfaceAssociationTransfer(domain),
            ),
        )
    )
    certification = refined.target.certification
    assert certification is not None and certification.passed
    assert certification.inputs.feature_source is domain
    target_witness = certification.inputs.fidelity_witness
    assert isinstance(target_witness, SurfaceChartWitness)
    # Root-mesh bindings name the actual target cells of the adapted mesh.
    assert target_witness.root_mesh_cell_bindings == tuple(
        SurfaceRootMeshCellBinding(
            cell_geometry_id(refined.target.geometry),
            refined.target.mesh.topology_id,
            int(identifier),
        )
        for identifier in np.asarray(target_witness.cell_global_ids, dtype=np.int64)
    )
    source_roots = set(
        zip(
            np.asarray(original.source_root_cell_ids, dtype=np.int64).tolist(),
            original.source_piece_ids,
            strict=True,
        )
    )
    target_roots = set(
        zip(
            np.asarray(target_witness.source_root_cell_ids, dtype=np.int64).tolist(),
            target_witness.source_piece_ids,
            strict=True,
        )
    )
    assert target_roots == source_roots
    assert refined.target.surface_source is source.surface_source

    refined_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in refined.target.mesh.blocks
        )
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                (), refined_ids, hierarchy=_bisection_hierarchy(refined)
            ),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=phx.meshing.SurfaceAssociationTransfer(domain),
            ),
        )
    )
    coarsened_report = coarsened.target.certification
    assert coarsened_report is not None and coarsened_report.passed
    coarsened_witness = coarsened_report.inputs.fidelity_witness
    assert isinstance(coarsened_witness, SurfaceChartWitness)
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    assert [
        (binding.root_topology_id, binding.root_cell_global_id)
        for binding in coarsened_witness.root_mesh_cell_bindings
        if binding is not None
    ] == [
        (binding.root_topology_id, binding.root_cell_global_id)
        for binding in original.root_mesh_cell_bindings
        if binding is not None
    ]
    assert coarsened_witness.domain_id == original.domain_id
    np.testing.assert_array_equal(
        coarsened_witness.source_root_cell_ids,
        original.source_root_cell_ids,
    )
    assert coarsened_witness.source_piece_ids == original.source_piece_ids
    assert coarsened_witness.geometry_entity_ids == original.geometry_entity_ids
    assert coarsened_witness.occurrence_paths == original.occurrence_paths


def _certified_curve_edge_triangle() -> tuple[
    CellMeshingResult, ParametricCurveBoundarySource
]:
    start = np.asarray((3.0, 0.0), dtype=np.float64)
    end = np.asarray((0.0, 3.0), dtype=np.float64)
    split = start + (1.0 / 3.0) * (end - start)
    points = np.stack((np.zeros((2,), dtype=np.float64), start, end, split))
    mesh = CellMesh.from_triangles(
        points, np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.affine(mesh)
    source = ParametricCurveBoundarySource(
        (LineCurve(start, end - start),),
        ((0.0, 1.0),),
        source_id="transition-boundary-edge",
        source_revision="edge-r1",
        curve_entity_ids=("authored-edge",),
        chart_ids=("authored-edge-chart",),
        occurrence_paths=((),),
    )
    topology = reference_cell_topology("triangle")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    parameter_by_vertex = {
        int(vertex_ids[1]): (Fraction(0),),
        int(vertex_ids[3]): (Fraction(1, 3),),
        int(vertex_ids[2]): (Fraction(1),),
    }
    selected_edges = {
        frozenset((int(vertex_ids[1]), int(vertex_ids[3]))),
        frozenset((int(vertex_ids[3]), int(vertex_ids[2]))),
    }
    cell_ids: list[int] = []
    entity_ids: list[int] = []
    local_entity_indices: list[int] = []
    reference_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    parameter_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    orientations: list[int] = []
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("The curve chart fixture requires polygonal edge incidence.")
    edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    edge_by_vertices = {
        frozenset(int(vertex_ids[int(value)]) for value in edge_vertices[row]): int(
            identifier
        )
        for row, identifier in enumerate(
            np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64).tolist()
        )
    }
    cell_vertex_rows = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    for cell_row in range(cell_vertex_rows.shape[0]):
        for local_edge, local_vertices in enumerate(topology.entities[1]):
            ordered_ids = tuple(
                int(vertex_ids[int(cell_vertex_rows[cell_row, int(vertex)])])
                for vertex in local_vertices
            )
            if frozenset(ordered_ids) not in selected_edges:
                continue
            reference_rows.append(
                tuple(
                    tuple(
                        Fraction(float(value)) for value in topology.vertices[int(vertex)]
                    )
                    for vertex in local_vertices
                )
            )
            parameters = tuple(
                parameter_by_vertex[identifier] for identifier in ordered_ids
            )
            parameter_rows.append(parameters)
            orientations.append(1 if parameters[1][0] > parameters[0][0] else -1)
            cell_ids.append(int(mesh.blocks[0].global_ids[cell_row]))
            entity_ids.append(edge_by_vertices[frozenset(ordered_ids)])
            local_entity_indices.append(local_edge)
    if len(cell_ids) != 2:
        raise ValueError(
            "The non-dyadic curve fixture must produce two source-edge charts."
        )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(entity_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_entity_indices, dtype=np.int32),
        target_reference_vertices=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in reference_rows
            ],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(reference_rows),
        source_chart_ids=source.chart_ids * 2,
        source_entity_ids=source.curve_entity_ids * 2,
        occurrence_paths=source.occurrence_paths * 2,
        source_parameters=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in parameter_rows
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(parameter_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(((0, 1), (1, 3), (3, 2), (2, 0)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("material",),
        source_id=source.source_id,
    )
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=source.source_id,
        source_revision=source.source_revision,
        domain=domain,
        cell_regions=np.zeros((2,), dtype=np.int64),
        source=source,
        fidelity_tolerance=0.0,
        fidelity_witness=witness,
        junction_vertices=np.zeros((0,), dtype=np.int64),
        protected_features=(),
    )
    result = certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        certification_inputs=inputs,
    )
    return result, source


def test_bisection_coarsening_preserves_nondyadic_source_chart_tokens() -> None:
    source, query = _certified_curve_edge_triangle()
    assert source.certification is not None and source.certification.passed
    source_cell_ids = np.asarray(source.mesh.blocks[0].global_ids, dtype=np.int64)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION)

    def cell_ids(result: CellMeshingResult) -> np.ndarray:
        return np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in result.mesh.blocks
            )
        )

    def parameter_values(result: CellMeshingResult) -> set[Fraction]:
        report = result.certification
        assert report is not None and report.passed
        assert report.inputs.source is query
        witness = report.inputs.fidelity_witness
        assert isinstance(witness, SourceBoundaryChartWitness)
        assert set(witness.source_entity_ids) == set(query.curve_entity_ids)
        assert set(witness.source_chart_ids) == set(query.chart_ids)
        assert any(
            (1, 3) in point
            for row in witness.source_parameter_rational_tokens
            for point in row
        )
        return {
            coordinate
            for row in range(len(witness.source_chart_ids))
            for point in witness.source_parameter_chart(row)
            for coordinate in point
        }

    # The first generation splits the straight legs; the second splits both
    # source-edge pieces at the non-dyadic chart midpoints 1/6 and 2/3.
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(source_cell_ids),
            policy=policy,
        )
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                cell_ids(refined.target), hierarchy=_bisection_hierarchy(refined)
            ),
            policy=policy,
        )
    )
    assert parameter_values(refined.target) == {
        Fraction(0),
        Fraction(1, 6),
        Fraction(1, 3),
        Fraction(2, 3),
        Fraction(1),
    }
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                (), cell_ids(refined.target), hierarchy=_bisection_hierarchy(refined)
            ),
            policy=policy,
        )
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    assert parameter_values(coarsened.target) == {
        Fraction(0),
        Fraction(1, 3),
        Fraction(1),
    }
    source.certification.revalidate(source.mesh, source.geometry)
    transition = coarsened.transition
    assert transition is not None

    with pytest.raises(ValueError, match="exact nested child-to-parent map"):
        _renew_transition_certification_inputs(
            refined.target,
            coarsened.target.mesh,
            coarsened.target.geometry,
            transition.lineage,
            reference_maps=(),
        )
