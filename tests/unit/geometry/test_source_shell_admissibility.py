from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phydrax.geometry._atlas import TrimDomain
from phydrax.geometry.brep._source_shell_admissibility import _trim_enclosure


def test_polygon_trim_bounds_use_each_native_face_chart() -> None:
    first = TrimDomain(np.asarray(((0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5))))
    second = TrimDomain(np.asarray(((0, 0), (1, 0), (1, 1), (0, 1))))
    model = SimpleNamespace(
        trim_domains=(first, second),
        parameter_bounds=np.asarray((((0, 0), (2, 2)), ((0, 0), (1, 1)))),
    )
    expected = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)))
    assert _trim_enclosure(model, 0) == expected
    assert _trim_enclosure(model, 1) == expected


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
