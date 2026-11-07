from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
    MaterialChartDeformation,
)
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses
from phydrax.geometry._common_root_volume_overlap import MaterialReferenceVolumeWitness
from phydrax.geometry._domain_certification import DomainCoverageCertificate
from phydrax.geometry._domain_coverage import certify_domain_coverage
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._volume_material_deformation import (
    declare_material_reference,
    MaterialReferenceDeclaration,
    MaterialReferencePolicy,
    VolumeMaterialDeformation,
)
from phydrax.geometry._volume_material_map import (
    MaterialSourceMapCapability,
    prepare_volume_material_map,
    VolumeMaterialChain,
    VolumeMaterialMap,
)
from tests._support.volume_material_map_fixtures import (
    affine_reference_chart,
    make_curved_root,
    make_material_coverage_pair,
    make_material_source,
    make_root_restriction,
    reference_map_witness,
    ROOT_REFERENCE_TETRAHEDRON,
    with_root_restriction,
)


_FRAME = "cad-world-frame"
_REGION = "aluminum"
_ENTITY = "solid-cell"
_OCCURRENCE = ("assembly-1", "occurrence-9")


def _move_endpoint(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    /,
) -> tuple[CellMesh, CellGeometrySpec]:
    moved_geometry_coordinates = transform(
        np.asarray(geometry.coordinates, dtype=np.float64)
    )
    moved_vertices = transform(np.asarray(mesh.coordinates, dtype=np.float64))
    moved_mesh = CellMesh(
        moved_vertices,
        mesh.blocks,
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
    )
    moved_geometry = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        moved_geometry_coordinates,
    )
    return moved_mesh, moved_geometry


def _coverage(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[MappedReferenceDomain, DomainCoverageCertificate]:
    domain, source_coverage, _ = make_material_coverage_pair(
        mesh,
        geometry,
        mesh,
        geometry,
        region_id=_REGION,
        occurrence_path=_OCCURRENCE,
    )
    return domain, source_coverage


def _identity_target_map() -> ReferenceCellMapWitnesses:
    return reference_map_witness(
        (0,),
        (7,),
        (ROOT_REFERENCE_TETRAHEDRON,),
        chart_arguments=(affine_reference_chart(ROOT_REFERENCE_TETRAHEDRON),),
    )


def _material_reference(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    coverage: DomainCoverageCertificate,
    /,
) -> tuple[MaterialSourceMapCapability, MaterialReferenceDeclaration]:
    domain = coverage.domain
    assert isinstance(domain, MappedReferenceDomain)
    source_map = make_material_source(domain)
    reference = declare_material_reference(
        mesh,
        geometry,
        source_map,
        (_ENTITY,),
        (_REGION,),
        coverage,
        frame_id=_FRAME,
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(_OCCURRENCE,),
    )
    return source_map, reference


def _deformation(
    reference: MaterialReferenceDeclaration,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_coverage: DomainCoverageCertificate,
    target_coverage: DomainCoverageCertificate,
    *,
    source_maps: Mapping[str, ReferenceCellMapWitnesses] | None = None,
    target_maps: Mapping[str, ReferenceCellMapWitnesses] | None = None,
    source_frame: str = _FRAME,
    target_frame: str = _FRAME,
    coordinate_transition: CellGeometryTransition | None = None,
) -> VolumeMaterialDeformation:
    return VolumeMaterialDeformation(
        reference,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        source_reference_maps=(
            dict(reference.identity_reference_maps)
            if source_maps is None
            else source_maps
        ),
        target_reference_maps=(
            {"root": _identity_target_map()} if target_maps is None else target_maps
        ),
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        source_frame_id=source_frame,
        target_frame_id=target_frame,
        coordinate_transition=coordinate_transition,
    )


def _material_map(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    deformation: VolumeMaterialDeformation,
    source_coverage: DomainCoverageCertificate,
    target_coverage: DomainCoverageCertificate,
    /,
) -> VolumeMaterialMap:
    return prepare_volume_material_map(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        deformation,
        (_ENTITY,),
        (_ENTITY,),
        (_REGION,),
        (_REGION,),
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        source_occurrence_paths=(_OCCURRENCE,),
        target_occurrence_paths=(_OCCURRENCE,),
    )


def _assert_sparse_action_adjoint(
    action: Callable[[Array], Array],
    jvp: Callable[[Array], Array],
    transpose: Callable[[Array], Array],
    /,
) -> None:
    primal = jnp.asarray((2.75,))
    tangent = jnp.asarray((-0.5,))
    cotangent = jnp.asarray((1.25,))
    _, actual_tangent = jax.jvp(action, (primal,), (tangent,))
    np.testing.assert_allclose(actual_tangent, jvp(tangent), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        jnp.vdot(action(primal), cotangent),
        jnp.vdot(primal, transpose(cotangent)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_material_volume_scaling_separates_content_density_and_scalar_actions() -> None:
    root_mesh, root_geometry, _ = make_curved_root()
    _, root_coverage = _coverage(root_mesh, root_geometry)
    _, reference = _material_reference(root_mesh, root_geometry, root_coverage)

    target_mesh, target_geometry = _move_endpoint(
        root_mesh,
        root_geometry,
        lambda points: points @ np.diag((2.0, 1.0, 1.0)).T,
    )
    _, target_coverage = _coverage(target_mesh, target_geometry)
    transition = CellGeometryTransition(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        evidence=CellGeometryTransitionEvidence("vertex_displacement"),
    )
    deformation = _deformation(
        reference,
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        root_coverage,
        target_coverage,
        coordinate_transition=transition,
    )
    material_map = _material_map(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        deformation,
        root_coverage,
        target_coverage,
    )

    assert isinstance(deformation, MaterialChartDeformation)
    assert material_map.deformation.frame_id == _FRAME
    assert material_map.material_overlap.succeeded
    assert not np.isclose(
        float(np.asarray(material_map.source_physical_measures)[0]),
        float(np.asarray(material_map.target_physical_measures)[0]),
    )
    np.testing.assert_allclose(
        material_map.apply_material_extensive(jnp.asarray((6.0,))), (6.0,)
    )
    np.testing.assert_allclose(
        material_map.apply_material_density(jnp.asarray((3.0,))), (1.5,)
    )
    np.testing.assert_allclose(
        material_map.apply_material_scalar(jnp.asarray((4.25,))), (4.25,)
    )
    density_evidence = material_map.apply_material_density_with_evidence(
        jnp.asarray((3.0,))
    )
    np.testing.assert_allclose(
        density_evidence.before_total, density_evidence.after_total, atol=1e-12
    )
    np.testing.assert_allclose(
        material_map.apply_material_extensive_transpose(jnp.asarray((2.0,))), (2.0,)
    )
    _assert_sparse_action_adjoint(
        material_map.apply_material_extensive,
        material_map.apply_material_extensive_jvp,
        material_map.apply_material_extensive_transpose,
    )
    _assert_sparse_action_adjoint(
        material_map.apply_material_density,
        material_map.apply_material_density_jvp,
        material_map.apply_material_density_transpose,
    )
    _assert_sparse_action_adjoint(
        material_map.apply_material_scalar,
        material_map.apply_material_scalar_jvp,
        material_map.apply_material_scalar_transpose,
    )


def test_curved_nonuniform_motion_composes_and_cold_revalidates() -> None:
    root_mesh, root_geometry, _ = make_curved_root()
    _, root_coverage = _coverage(root_mesh, root_geometry)
    _, reference = _material_reference(root_mesh, root_geometry, root_coverage)

    scaled_mesh, scaled_geometry = _move_endpoint(
        root_mesh,
        root_geometry,
        lambda points: points @ np.diag((2.0, 1.0, 1.0)).T,
    )
    _, scaled_coverage = _coverage(scaled_mesh, scaled_geometry)
    first_deformation = _deformation(
        reference,
        root_mesh,
        root_geometry,
        scaled_mesh,
        scaled_geometry,
        root_coverage,
        scaled_coverage,
    )
    first = _material_map(
        root_mesh,
        root_geometry,
        scaled_mesh,
        scaled_geometry,
        first_deformation,
        root_coverage,
        scaled_coverage,
    )

    def nonuniform_curved_motion(
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        moved = np.array(points, copy=True)
        moved[:, 0] += 0.1 * moved[:, 0] ** 2
        return moved

    target_mesh, target_geometry = _move_endpoint(
        root_mesh, root_geometry, nonuniform_curved_motion
    )
    _, target_coverage = _coverage(target_mesh, target_geometry)
    second_deformation = _deformation(
        reference,
        scaled_mesh,
        scaled_geometry,
        target_mesh,
        target_geometry,
        scaled_coverage,
        target_coverage,
        source_maps=dict(first.deformation.target_reference_maps),
    )
    second = _material_map(
        scaled_mesh,
        scaled_geometry,
        target_mesh,
        target_geometry,
        second_deformation,
        scaled_coverage,
        target_coverage,
    )
    chain = VolumeMaterialChain((first, second))

    expected_density = (
        3.0
        * float(np.asarray(first.source_physical_measures)[0])
        / float(np.asarray(second.target_physical_measures)[0])
    )
    np.testing.assert_allclose(
        chain.apply_material_extensive(jnp.asarray((8.0,))), (8.0,)
    )
    np.testing.assert_allclose(
        chain.apply_material_density(jnp.asarray((3.0,))), (expected_density,), rtol=1e-11
    )
    np.testing.assert_allclose(chain.apply_material_scalar(jnp.asarray((1.75,))), (1.75,))
    np.testing.assert_allclose(
        jnp.vdot(chain.apply_material_density(jnp.asarray((2.0,))), jnp.asarray((1.0,))),
        jnp.vdot(
            jnp.asarray((2.0,)),
            chain.apply_material_density_transpose(jnp.asarray((1.0,))),
        ),
        rtol=1e-12,
    )
    assert VolumeMaterialChain.revalidate(chain, chain.maps).chain_id == chain.chain_id
    rebuilt = VolumeMaterialMap.revalidate(
        first,
        root_mesh,
        root_geometry,
        scaled_mesh,
        scaled_geometry,
        root_mesh=root_mesh,
        root_geometry=root_geometry,
        root_coverage=root_coverage,
        source_coverage=root_coverage,
        target_coverage=scaled_coverage,
    )
    assert rebuilt.map_id == first.map_id

    identity_deformation = _deformation(
        reference,
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        root_coverage,
        root_coverage,
    )
    identity = _material_map(
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        identity_deformation,
        root_coverage,
        root_coverage,
    )
    np.testing.assert_allclose(
        identity.apply_material_extensive(jnp.asarray((5.0,))), (5.0,)
    )
    np.testing.assert_allclose(
        identity.apply_material_density(jnp.asarray((5.0,))), (5.0,)
    )
    np.testing.assert_allclose(
        identity.apply_material_scalar(jnp.asarray((5.0,))), (5.0,)
    )


def test_material_reference_refuses_foreign_frame_inverted_and_singular_maps() -> None:
    root_mesh, root_geometry, _ = make_curved_root()
    _, root_coverage = _coverage(root_mesh, root_geometry)
    _, reference = _material_reference(root_mesh, root_geometry, root_coverage)
    identity_maps = dict(reference.identity_reference_maps)

    with pytest.raises(ValueError, match="declared reference frame"):
        _deformation(
            reference,
            root_mesh,
            root_geometry,
            root_mesh,
            root_geometry,
            root_coverage,
            root_coverage,
            target_frame="foreign-frame",
        )

    for transform in (
        lambda points: points @ np.diag((-1.0, 1.0, 1.0)).T,
        lambda points: points @ np.diag((0.0, 1.0, 1.0)).T,
    ):
        bad_mesh, bad_geometry = _move_endpoint(root_mesh, root_geometry, transform)
        with pytest.raises(
            ValueError, match="(?i)(positive|jacobian|valid|inverted|singular)"
        ):
            MaterialReferenceVolumeWitness(
                root_mesh,
                root_geometry,
                root_mesh,
                root_geometry,
                bad_mesh,
                bad_geometry,
                identity_maps,
                {"root": _identity_target_map()},
            )

    # Endpoint maps are independent deformations of the root, so a translated
    # root is a legitimate reference by itself; a witness bound to its root
    # refuses a foreign one.
    foreign_root, foreign_geometry = _move_endpoint(
        root_mesh, root_geometry, lambda points: points + np.asarray((4.0, 0.0, 0.0))
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        identity_maps,
        identity_maps,
    )
    with pytest.raises(ValueError, match="does not match"):
        witness.revalidate(
            foreign_root,
            foreign_geometry,
            root_mesh,
            root_geometry,
            root_mesh,
            root_geometry,
        )


def test_descendant_material_reference_restores_motion_with_distinct_chart_parent() -> (
    None
):
    root, root_geometry, _ = make_curved_root()
    parent, parent_geometry, _ = make_root_restriction(
        root,
        root_geometry,
        cells=(ROOT_REFERENCE_TETRAHEDRON,),
        cell_ids=(91,),
        block_name="parent",
    )
    charts = (affine_reference_chart(ROOT_REFERENCE_TETRAHEDRON),)
    parent_maps = reference_map_witness(
        (0,),
        (7,),
        (ROOT_REFERENCE_TETRAHEDRON,),
        chart_arguments=charts,
    )
    parent_geometry = with_root_restriction(
        root,
        root_geometry,
        parent,
        parent_geometry,
        parent_maps,
    )
    domain, _, _ = make_material_coverage_pair(
        root,
        root_geometry,
        parent,
        parent_geometry,
        region_id=_REGION,
        occurrence_path=_OCCURRENCE,
    )
    source_mesh = CellMesh(
        parent.coordinates,
        (
            CellBlock(
                "child",
                "tetrahedron",
                parent.blocks[0].vertices,
                global_ids=np.asarray((101,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=parent.vertex_global_ids,
    )
    source_geometry = CellGeometrySpec(
        {"child": root_geometry.elements[0]},
        {"child": root_geometry.geometry_dofs[0]},
        root_geometry.coordinates,
    )
    source_geometry = with_root_restriction(
        parent,
        parent_geometry,
        source_mesh,
        source_geometry,
        reference_map_witness(
            (0,),
            (91,),
            (ROOT_REFERENCE_TETRAHEDRON,),
            chart_arguments=charts,
        ),
    )
    chart_parent = (parent, parent_geometry)
    validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    coverage = certify_domain_coverage(
        source_mesh,
        source_geometry,
        domain,
        np.zeros(1, dtype=np.int64),
        embedding=certify_global_embedding(source_mesh, source_geometry, validity),
        chart_parent=chart_parent,
    )
    assert coverage.status == "certified", coverage.findings
    source_map = make_material_source(domain)
    reference = declare_material_reference(
        source_mesh,
        source_geometry,
        source_map,
        (_ENTITY,),
        (_REGION,),
        coverage,
        frame_id=_FRAME,
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(_OCCURRENCE,),
        chart_parent=chart_parent,
    )
    target_mesh, target_geometry = _move_endpoint(
        source_mesh,
        source_geometry,
        lambda points: points @ np.diag((2.0, 1.0, 1.0)).T,
    )
    _, target_coverage = _coverage(target_mesh, target_geometry)
    maps = dict(reference.identity_reference_maps)
    deformation = VolumeMaterialDeformation(
        reference,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        source_reference_maps=maps,
        target_reference_maps=maps,
        source_coverage=coverage,
        target_coverage=target_coverage,
        source_frame_id=_FRAME,
        target_frame_id=_FRAME,
        source_chart_parent=chart_parent,
    )
    material_map = prepare_volume_material_map(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        deformation,
        (_ENTITY,),
        (_ENTITY,),
        (_REGION,),
        (_REGION,),
        source_coverage=coverage,
        target_coverage=target_coverage,
        source_occurrence_paths=(_OCCURRENCE,),
        target_occurrence_paths=(_OCCURRENCE,),
        source_chart_parent=chart_parent,
    )
    with pytest.raises(ValueError, match="source coverage is stale"):
        VolumeMaterialMap.revalidate(
            material_map,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            root_mesh=source_mesh,
            root_geometry=source_geometry,
            root_coverage=coverage,
            source_chart_parent=chart_parent,
        )
    restored = VolumeMaterialMap.revalidate(
        material_map,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        root_mesh=source_mesh,
        root_geometry=source_geometry,
        root_coverage=coverage,
        root_chart_parent=chart_parent,
        source_chart_parent=chart_parent,
    )
    np.testing.assert_allclose(
        restored.target_physical_measures,
        2.0 * restored.source_physical_measures,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        restored.apply_material_extensive(jnp.asarray((2.75,), dtype=jnp.float64)),
        np.asarray((2.75,), dtype=np.float64),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        restored.apply_material_density(jnp.asarray((6.0,), dtype=jnp.float64)),
        np.asarray((3.0,), dtype=np.float64),
        rtol=1e-13,
        atol=1e-13,
    )
