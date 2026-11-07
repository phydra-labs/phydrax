import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._brep_sphere_material import (
    _require_full_sphere_source,
    BRepSphereMaterialMap,
)
from phydrax.meshing._result import CellMeshingResult
from tools.meshing_qualification.scenarios import native_planar_profiles as profiles


@pytest.fixture(scope="module")
def _sphere_result() -> CellMeshingResult:
    domain = profiles._surface_definition("sphere-feature-surface", "sphere").domain
    scope = profiles.MeshingScope.whole(domain, 2)
    size = 0.4
    deviation = 0.25 * size * size
    spec = profiles.SurfaceMeshingSpec(
        profiles.CellMeshingTarget(
            2, 3, profiles.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            profiles.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=profiles.SizeControlStrength.HARD,
            ),
        ),
        protected_features=(
            profiles.ProtectedFeature(
                scope,
                profiles.FeatureKind.SURFACE,
                maximum_deviation=deviation,
            ),
        ),
        size_compliance=profiles.SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        ),
        maximum_deviation=min(0.02, deviation),
    )
    return (
        profiles.NativeMeshingProvider(
            profiles.NativeMeshingOptions("parametric_surface")
        )
        .plan(domain, spec, coordinate_contract=profiles.SpatialCoordinateContract.si())
        .execute()
    )


def test_brep_sphere_radial_map_edge_metric_and_jvp(
    _sphere_result: CellMeshingResult,
) -> None:
    result = _sphere_result
    certification = result.certification
    assert certification is not None
    source_query = certification.inputs.source
    assert source_query is not None
    source = BRepSphereMaterialMap(result, source_query)
    point = jnp.asarray((0.2, 0.3))
    physical, jacobian = source.evaluate(0, point[None, :])
    independent = jax.jacfwd(lambda x: source.evaluate_points(0, x[None, :])[0, 0])(point)
    np.testing.assert_allclose(
        np.asarray(jacobian[0, 0]), np.asarray(independent), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(physical[0, 0])), 1.0, atol=1e-12
    )
    assert np.linalg.det(np.asarray(jacobian[0, 0]).T @ np.asarray(jacobian[0, 0])) > 0
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    times = np.asarray((0.2, 0.5, 0.8))[:, None]
    uses = {}
    for block_index, block in enumerate(result.mesh.blocks):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            points = reference[a] * (1 - times) + reference[b] * times
            values = np.asarray(source.evaluate_points(block_index, points))
            for row, cell in enumerate(np.asarray(block.vertices)):
                first, last = int(cell[a]), int(cell[b])
                key = (min(first, last), max(first, last))
                uses.setdefault(key, []).append(
                    values[row] if first < last else values[row, ::-1]
                )
    assert uses and all(len(pair) == 2 for pair in uses.values())
    for first, second in uses.values():
        np.testing.assert_allclose(first, second, rtol=0, atol=2e-12)


def test_changed_spherical_seam_authority_is_rejected(
    _sphere_result: CellMeshingResult,
) -> None:
    charts = _sphere_result.surface_source
    assert charts is not None
    authority = charts.domain.brep_authority
    assert authority is not None
    geometry = authority.geometry
    seams = [
        coedge
        for coedge, edge in enumerate(geometry.coedge_edges)
        if geometry.edge_curves[edge] >= 0
    ]
    assert len(seams) == 2
    altered = list(geometry.pcurves)
    altered[seams[0]] = altered[seams[1]]
    forged = eqx.tree_at(
        lambda value: value.domain.brep_authority.geometry.pcurves,
        charts,
        tuple(altered),
    )
    with pytest.raises(ValueError):
        _require_full_sphere_source(forged)
