from fractions import Fraction
from pathlib import Path

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._patches import CircleCurve
from phydrax.geometry.brep._source_shell_admissibility import _trim_enclosure


def test_polygon_trim_bounds_use_each_native_face_chart() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    bounds_seen = set()
    for face, trim in enumerate(model.trim_domains):
        assert trim is not None and trim.outer is not None and not trim.curve_loops
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        points = np.asarray(trim.outer, dtype=np.float64)
        lower = tuple(Fraction.from_float(float(value)) for value in bounds[0])
        upper = tuple(Fraction.from_float(float(value)) for value in bounds[1])
        expected = tuple(
            tuple(
                lower[axis]
                + Fraction.from_float(float(reduce(points[:, axis])))
                * (upper[axis] - lower[axis])
                for axis in range(2)
            )
            for reduce in (np.min, np.max)
        )
        assert _trim_enclosure(model, face) == expected
        bounds_seen.add((lower, upper))
    assert len(bounds_seen) > 1


def test_iges_quadric_source_face_pair_replay(tmp_path: Path) -> None:
    from phydrax.geometry.brep._source_shell_admissibility import (
        certify_retained_branch_pair,
        certify_same_chart_face_separation,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case

    model, _, _, _ = _quadric_case("iges", tmp_path)
    separation = certify_same_chart_face_separation(model, 0, 1)
    assert separation is not None
    assert separation.replay(model)
    for first, second in ((0, 2), (1, 2)):
        proof = certify_retained_branch_pair(model, first, second)
        assert proof is not None
        assert proof.shared_edges
        assert proof.replay(model)


def test_original_step_source_pair_certificates_and_stale_refusal(tmp_path: Path) -> None:
    import equinox as eqx

    from phydrax.geometry.brep._source_shell_admissibility import (
        _opposed_source_uses,
        _shared_two_arc_circle,
        certify_coaxial_radial_separation,
        certify_plane_torus_pair,
        certify_source_ring_face_pair,
        certify_tangent_cylinder_torus_pair,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    model, _, _, _ = _external_case("external-routine", "step", tmp_path)
    for certify, pairs in (
        (certify_source_ring_face_pair, ((0, 1), (1, 2))),
        (certify_tangent_cylinder_torus_pair, ((0, 5), (2, 4))),
        (certify_plane_torus_pair, ((3, 4), (3, 5))),
    ):
        for first, second in pairs:
            proof = certify(model, first, second)
            assert proof is not None
            assert proof.replay(model)
            assert certify(model, second, first) is None
            assert certify(model, first, first) is None
    assert certify_tangent_cylinder_torus_pair(model, 0, 4) is None
    assert certify_plane_torus_pair(model, 1, 4) is None
    radial = certify_coaxial_radial_separation(model, 0, 4)
    assert radial is not None and radial.replay(model)
    assert certify_coaxial_radial_separation(model, 0, 5) is None
    assert _opposed_source_uses(model, (0, 1), (0, 6))
    assert not _opposed_source_uses(model, (0, 1), (0, 5))
    assert _shared_two_arc_circle(model, 0, 5) is not None
    shortened_ranges = model.geometry.edge_ranges.at[1, 1].set(1.0)
    shortened = eqx.tree_at(
        lambda value: value.geometry.edge_ranges, model, shortened_ranges
    )
    assert _shared_two_arc_circle(shortened, 0, 5) is None

    # A forged numeric circle shadow is not a current source edge, even if its
    # retained root or a mesh sample still resembles the old boundary.
    circle = model.geometry.curves[0]
    assert isinstance(circle, CircleCurve)
    stale = eqx.tree_at(
        lambda value: value.geometry.curves[0].radius,
        model,
        circle.radius * 1.01,
    )
    assert certify_source_ring_face_pair(stale, 0, 1) is None
