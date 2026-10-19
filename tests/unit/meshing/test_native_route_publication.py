"""Provider publication preserves scientific binding, organization, and evidence."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing import CellMeshingResult
from tests._support.meshing_fixtures import box_plc
from tests._support.structured_sources import material_multiblock_request


class _PublishedRoute(NamedTuple):
    result: CellMeshingResult
    source_id: str
    source_revision: str
    specification_id: str
    plan_id: str
    zone_names: tuple[str, ...]


def _quad_publication() -> _PublishedRoute:
    m = phx.meshing
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="publication-square",
    )
    source = m.NativePlanarSource(region, "publication-r1")
    scope = m.MeshingScope(
        source.source_id,
        source.source_revision,
        m.MeshingEntityKind.GEOMETRY,
        2,
        "publication-square-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = m.SurfaceMeshingSpec(
        m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            m.UniformSizeControl(scope, 1.0, strength=m.SizeControlStrength.SOFT),
        ),
    )
    plan = m.NativeMeshingProvider(m.NativeMeshingOptions("planar_dual_quad")).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    return _PublishedRoute(
        plan.execute(),
        source.source_id,
        source.source_revision,
        specification.specification_id,
        plan.plan_id,
        (),
    )


def _grid_publication() -> _PublishedRoute:
    m = phx.meshing
    source = m.NativePlcSource(box_plc(), "publication-cube", "publication-r1")
    scope = m.MeshingScope(
        source.source_id,
        source.source_revision,
        m.MeshingEntityKind.GEOMETRY,
        2,
        "publication-cube-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = m.VolumeMeshingSpec(
        m.CellMeshingTarget(3, 3, m.CellFamilyPolicy(required=("hexahedron",))),
        scope,
        m.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            m.UniformSizeControl(scope, 1.0, strength=m.SizeControlStrength.SOFT),
        ),
    )
    plan = m.NativeMeshingProvider(m.NativeMeshingOptions("plc_balanced_grid_hex")).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    return _PublishedRoute(
        plan.execute(),
        source.source_id,
        source.source_revision,
        specification.specification_id,
        plan.plan_id,
        source.complex.region_ids,
    )


def _polyhedral_publication() -> _PublishedRoute:
    m = phx.meshing
    source = m.NativePlcSource(box_plc(), "publication-power-cube", "publication-r1")
    scope = m.MeshingScope(
        source.source_id,
        source.source_revision,
        m.MeshingEntityKind.GEOMETRY,
        2,
        "publication-power-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = m.VolumeMeshingSpec(
        m.CellMeshingTarget(3, 3, m.CellFamilyPolicy(required=("polyhedron",))),
        scope,
        m.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            m.UniformSizeControl(scope, 10.0, strength=m.SizeControlStrength.SOFT),
        ),
    )
    plan = m.NativeMeshingProvider(m.NativeMeshingOptions("plc_restricted_power")).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    return _PublishedRoute(
        plan.execute(),
        source.source_id,
        source.source_revision,
        specification.specification_id,
        plan.plan_id,
        source.complex.region_ids,
    )


def _structured_publication() -> _PublishedRoute:
    source, specification, options = material_multiblock_request()
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    return _PublishedRoute(
        plan.execute(),
        source.source_id,
        source.source_revision,
        specification.specification_id,
        plan.plan_id,
        tuple(control.region_name for control in specification.region_controls),
    )


@pytest.fixture(
    scope="module",
    params=[
        _quad_publication,
        _grid_publication,
        _polyhedral_publication,
        _structured_publication,
    ],
    ids=["dual-quad", "grid-hex", "polyhedral", "controlled-structured"],
)
def published_route(request: pytest.FixtureRequest) -> _PublishedRoute:
    factory: Callable[[], _PublishedRoute] = request.param
    return factory()


def test_provider_publication_retains_source_and_specification_binding(
    published_route: _PublishedRoute,
) -> None:
    result = published_route.result
    assert result.compliance.specification_id == published_route.specification_id
    binding = result.trace.binding
    assert binding is not None
    assert binding.source_id == published_route.source_id
    assert binding.source_revision == published_route.source_revision
    assert binding.topology_id == result.mesh.topology_id


def test_provider_publication_keeps_certification_and_failure_evidence(
    published_route: _PublishedRoute,
) -> None:
    result = published_route.result
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert result.execution_evidence is not None
    assert result.execution_evidence.owner_id == published_route.plan_id


def test_provider_organization_binds_published_mesh_entity_sets(
    published_route: _PublishedRoute,
) -> None:
    result = published_route.result
    assert tuple(sorted(zone.name for zone in result.zones)) == tuple(
        sorted(published_route.zone_names)
    )
    for zone in result.zones:
        assert zone.scope.source_id == result.mesh.mesh_id
        assert zone.scope.source_revision == result.mesh.numeric_version
        assert (
            zone.scope.entity_set_id
            == result.mesh.entity_set(zone.scope.entity_dimension).entity_set_id
        )
    entity_sets = {
        result.mesh.entity_set(dimension).entity_set_id
        for dimension in range(result.mesh.topological_dimension + 1)
    }
    for association in result.associations:
        assert association.source_id == published_route.source_id
        assert association.source_revision == published_route.source_revision
        assert association.target_entity_set_id in entity_sets
