from fractions import Fraction
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
