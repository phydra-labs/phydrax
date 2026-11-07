from fractions import Fraction as F

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._brep_torus_material import (
    _certify_full_torus,
    _normal_fiber_sign,
    BRepTorusMaterialMap,
)
from phydrax.meshing._result import CellMeshingResult
from tools.meshing_qualification.scenarios import native_planar_profiles as profiles


def test_torus_normal_fiber_proves_transversality_and_refuses_tangency() -> None:
    base, axis = (F(0), F(0), F(0)), (F(0), F(0), F(1))
    outer = ((F(3), F(-1, 4), F(-1, 4)), (F(3), F(1, 4), F(-1, 4)), (F(3), F(0), F(1, 4)))
    inner = (
        (F(1), outer[0][1], outer[0][2]),
        (F(1), outer[1][1], outer[1][2]),
        (F(1), outer[2][1], outer[2][2]),
    )
    assert _normal_fiber_sign(outer, base, axis, F(2)) == 1
    assert _normal_fiber_sign(inner, base, axis, F(2)) == -1
    tangent = (
        (F(2), outer[0][1], outer[0][2]),
        (F(2), outer[1][1], outer[1][2]),
        (F(2), outer[2][1], outer[2][2]),
    )
    with pytest.raises(ValueError, match="tangent"):
        _normal_fiber_sign(tangent, base, axis, F(2))


@pytest.fixture(scope="module")
def _torus_result() -> CellMeshingResult:
    domain = profiles._surface_definition("native-surface-pde:torus", "torus").domain
    scope = profiles.MeshingScope.whole(domain, 2)
    size = 0.45
    spec = profiles.SurfaceMeshingSpec(
        profiles.CellMeshingTarget(
            2, 3, profiles.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            profiles.UniformSizeControl(
                scope,
                size,
                strength=profiles.SizeControlStrength.SOFT,
            ),
        ),
        maximum_deviation=0.02,
    )
    return (
        profiles.NativeMeshingProvider(
            profiles.NativeMeshingOptions("parametric_surface")
        )
        .plan(domain, spec, coordinate_contract=profiles.SpatialCoordinateContract.si())
        .execute()
    )


def test_brep_torus_material_map_source_metric_jvp_and_edges(
    _torus_result: CellMeshingResult,
) -> None:
    result = _torus_result
    certification = result.certification
    assert certification is not None
    source_query = certification.inputs.source
    assert source_query is not None
    source = BRepTorusMaterialMap(result, source_query)
    point = jnp.asarray((0.2, 0.3))
    physical, jacobian = source.evaluate(0, point[None, :])
    independent = jax.jacfwd(lambda x: source.evaluate_points(0, x[None, :])[0, 0])(point)
    np.testing.assert_allclose(
        np.asarray(jacobian[0, 0]), np.asarray(independent), rtol=1e-10, atol=1e-10
    )
    x, y, z = np.asarray(physical[0, 0])
    np.testing.assert_allclose((np.hypot(x, y) - 2) ** 2 + z * z, 0.6**2, atol=1e-12)
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


def test_changed_torus_seam_authority_is_rejected(
    _torus_result: CellMeshingResult,
) -> None:
    charts = _torus_result.surface_source
    assert charts is not None
    authority = charts.domain.brep_authority
    assert authority is not None
    geometry = authority.geometry
    first, second = geometry.face_loops[0][0][:2]
    altered = list(geometry.pcurves)
    altered[first] = altered[second]
    forged = eqx.tree_at(
        lambda value: value.domain.brep_authority.geometry.pcurves,
        charts,
        tuple(altered),
    )
    with pytest.raises(ValueError):
        _certify_full_torus(forged)
