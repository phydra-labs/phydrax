from fractions import Fraction as F
from math import inf, nextafter, tau

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._approximate_source_coverage import _cad_trim_root_box
from phydrax.geometry._brep_torus_material import (
    _certify_full_torus,
    _normal_fiber_sign,
    BRepTorusMaterialMap,
)
from phydrax.geometry._domain_boundary import _rooted_patch_chart_junctions
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    PatchCurveUse,
    SourcePreparation,
)
from phydrax.geometry.brep import brep_torus
from phydrax.geometry.brep._boolean_core import PeriodicPCurve
from phydrax.geometry.brep._curve_approximation_topology import TrimReferenceSegment
from phydrax.geometry.brep._intersection import ConicParameterEndpoint
from phydrax.geometry.brep._patches import ConicParameterRoot
from phydrax.geometry.brep._source_embedding import (
    _torus_trim_sides,
    certify_brep_trim_sheet,
    certify_reference_trim_sheet,
)
from phydrax.geometry.brep._trim_helpers import PeriodOffset
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


@pytest.fixture(scope="module")
def _torus_source() -> MeshingDomain:
    return MeshingDomain.from_brep(
        brep_torus(1.0, 0.25, coordinate_contract=SpatialCoordinateContract.si())
    )


def test_torus_source_sheet_replays_both_exact_decks(
    _torus_source: MeshingDomain,
) -> None:
    proof = certify_brep_trim_sheet(_torus_source, 0)
    assert proof is not None and proof.replay(_torus_source, 0)
    turn = PeriodOffset(F(0), F(1))
    assert proof.embedding.bounds == ((F(0), F(0)), (turn, turn))
    assert proof.embedding.deck_axes == (0, 1)


@pytest.mark.parametrize("fault", ("missing", "extra-turn", "rounded", "short-range"))
def test_torus_deck_refuses_changed_profile_turn(
    _torus_source: MeshingDomain, fault: str
) -> None:
    model = _torus_source.brep_authority
    assert model is not None
    geometry = model.geometry
    coedge = geometry.face_loops[0][0][1]
    roots = list(geometry.coedge_endpoint_roots)
    start, stop = roots[coedge]
    assert isinstance(stop, ConicParameterEndpoint)
    if fault == "short-range":
        uses = list(_torus_source.patches[0].loops[0])
        use = uses[1]
        assert isinstance(use, PatchCurveUse)
        uses[1] = PatchCurveUse(
            use.curve, use.pcurve, use.first, nextafter(tau, 0.0), sense=use.sense
        )
        changed = eqx.tree_at(
            lambda domain: domain.patches[0].loops, _torus_source, (tuple(uses),)
        )
    else:
        if fault == "missing":
            end = None
        elif fault == "rounded":
            end = ConicParameterEndpoint(
                stop.root, stop.carrier, F.from_float(tau), stop.point_exact
            )
        else:
            conic = ConicParameterRoot(stop.root.conic, F(1), F(0), turns=2)
            end = ConicParameterEndpoint(conic, stop.carrier, None, stop.point_exact)
        roots[coedge] = start, end
        changed_model = eqx.tree_at(
            lambda authority: authority.geometry.coedge_endpoint_roots,
            model,
            tuple(roots),
        )
        changed = eqx.tree_at(
            lambda domain: domain.brep_authority, _torus_source, changed_model
        )
    # Exercise the deck evidence itself, not the stale-domain fingerprint.
    assert _torus_trim_sides(changed, 0) is None


@pytest.mark.parametrize("fault", ("edge", "sense", "shift"))
def test_torus_deck_refuses_mismatched_seam_authority(
    _torus_source: MeshingDomain, fault: str
) -> None:
    model = _torus_source.brep_authority
    assert model is not None
    geometry = model.geometry
    bottom, _, top, _ = geometry.face_loops[0][0]
    uses = list(_torus_source.patches[0].loops[0])
    if fault in ("edge", "sense"):
        use = uses[2]
        assert isinstance(use, PatchCurveUse)
        other = uses[1 if fault == "edge" else 0]
        assert isinstance(other, PatchCurveUse)
        uses[2] = PatchCurveUse(
            other.curve if fault == "edge" else use.curve,
            use.pcurve,
            use.first,
            use.last,
            sense=other.sense if fault == "sense" else use.sense,
        )
        changed = eqx.tree_at(
            lambda domain: domain.patches[0].loops, _torus_source, (tuple(uses),)
        )
    else:
        curves = list(geometry.pcurves)
        curves[top] = PeriodicPCurve(curves[bottom], model.patches[0], (0, 2))
        uses[2] = eqx.tree_at(lambda use: use.pcurve, uses[2], curves[top])
        changed_model = eqx.tree_at(
            lambda authority: authority.geometry.pcurves, model, tuple(curves)
        )
        changed = eqx.tree_at(
            lambda domain: (domain.brep_authority, domain.patches[0].loops),
            _torus_source,
            (changed_model, (tuple(uses),)),
        )
    assert _torus_trim_sides(changed, 0) is None


@pytest.fixture(scope="module")
def _torus_reference(_torus_source: MeshingDomain) -> tuple[TrimReferenceSegment, ...]:
    """Split each of the four authored sides at the same exact parameter."""
    model = _torus_source.brep_authority
    assert model is not None
    limit = F.from_float(tau)
    corners = ((F(0), F(0)), (limit, F(0)), (limit, limit), (F(0), limit))
    junctions = _rooted_patch_chart_junctions(_torus_source, 0)[0]
    segments = []
    for slot, (use, coedge) in enumerate(
        zip(
            _torus_source.patches[0].loops[0],
            model.geometry.face_loops[0][0],
            strict=True,
        )
    ):
        start, stop = corners[slot], corners[(slot + 1) % 4]
        midpoint = ((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2)
        a, b = (F(0), limit) if use.sense > 0 else (limit, F(0))
        roots = model.geometry.coedge_endpoint_roots[coedge]
        first, last = roots if use.sense > 0 else roots[::-1]
        assert first is not None and last is not None
        segments.extend(
            (
                TrimReferenceSegment(
                    use.pcurve,
                    a,
                    limit / 2,
                    start,
                    midpoint,
                    _cad_trim_root_box(junctions[slot - 1]),
                    None,
                    first.parameter_enclosure(),
                    None,
                ),
                TrimReferenceSegment(
                    use.pcurve,
                    limit / 2,
                    b,
                    midpoint,
                    stop,
                    None,
                    _cad_trim_root_box(junctions[slot]),
                    None,
                    last.parameter_enclosure(),
                ),
            )
        )
    return tuple(segments)


def test_torus_reference_sheet_proves_each_paired_chain(
    _torus_source: MeshingDomain, _torus_reference: tuple[TrimReferenceSegment, ...]
) -> None:
    native = certify_brep_trim_sheet(_torus_source, 0)
    assert native is not None
    certified = certify_reference_trim_sheet(
        _torus_source, 0, native, (_torus_reference,), SourcePreparation(_torus_source)
    )
    assert certified is not None
    assert certified[1].bounds == native.embedding.bounds
    assert certified[1].deck_axes == (0, 1)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("fault", ("chord", "cuts", "root", "outside"))
def test_torus_reference_sheet_refuses_unproved_correspondence(
    _torus_source: MeshingDomain,
    _torus_reference: tuple[TrimReferenceSegment, ...],
    axis: int,
    fault: str,
) -> None:
    native = certify_brep_trim_sheet(_torus_source, 0)
    assert native is not None
    segments = list(_torus_reference)
    slot = 2 if axis == 0 else 4  # The positive-side chord chain.
    left, right = segments[slot : slot + 2]
    if fault == "root":
        segments[slot] = left._replace(start_parameter_box=None)
    elif fault == "outside":
        outside = F.from_float(nextafter(tau, inf))
        point = (outside, left.stop[1]) if axis == 0 else (left.stop[0], outside)
        segments[slot] = left._replace(stop=point)
        segments[slot + 1] = right._replace(start=point)
    else:
        shifted = left.last + F(1, 8)
        point = (left.stop[0], shifted) if axis == 0 else (shifted, left.stop[1])
        if fault == "cuts":
            # Alter both the source cut and chord point on just one side.
            left, right = left._replace(last=shifted), right._replace(first=shifted)
        segments[slot] = left._replace(stop=point)
        segments[slot + 1] = right._replace(start=point)
    assert (
        certify_reference_trim_sheet(
            _torus_source, 0, native, (tuple(segments),), SourcePreparation(_torus_source)
        )
        is None
    )
