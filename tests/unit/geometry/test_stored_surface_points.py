"""Golden D1 points, exact enclosures, and fault adequacy of the stored route.

The plane has no transcendental operation and no fault-adequacy requirement.
"""

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._meshing_domain import (
    _PlacedSurfacePatch,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    SourcePreparation,
)
from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._host_program import prepare_host_program
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    bucketed_query_rows,
    ConePatch,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)


type Interval = tuple[Fraction, Fraction]


def _add(left: Interval, right: Interval, /) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def _multiply(left: Interval, right: Interval, /) -> Interval:
    products = tuple(a * b for a in left for b in right)
    return min(products), max(products)


def _trigonometric(argument: Fraction, /, *, cosine: bool) -> Interval:
    # Taylor's theorem: the next derivative has absolute value at most one.
    degree = 48 if cosine else 49
    first = 0 if cosine else 1
    value = sum(
        (
            Fraction((-1) ** ((power - first) // 2), math.factorial(power))
            * argument**power
            for power in range(first, degree + 1, 2)
        ),
        Fraction(0),
    )
    remainder = abs(argument) ** (degree + 1) / math.factorial(degree + 1)
    return value - remainder, value + remainder


def _exact_coordinates(kind: str, u: Fraction, v: Fraction, /) -> tuple[Interval, ...]:
    cosine, sine = _trigonometric(u, cosine=True), _trigonometric(u, cosine=False)
    cosine_v = _trigonometric(v, cosine=True)
    sine_v = _trigonometric(v, cosine=False)
    one = (Fraction(1), Fraction(1))
    two = (Fraction(2), Fraction(2))
    three = (Fraction(3), Fraction(3))
    if kind == "plane":
        return (1 + u, 1 + u), (1 + v, 1 + v), one
    if kind == "cylinder":
        return _add((-one[0], -one[1]), cosine), sine, (v, v)
    if kind == "sphere":
        return (
            _add(one, _multiply(cosine, cosine_v)),
            _add(one, _multiply(sine, cosine_v)),
            _add(one, sine_v),
        )
    if kind == "cone":
        sine_a = _trigonometric(Fraction(1, 4), cosine=False)
        cosine_a = _trigonometric(Fraction(1, 4), cosine=True)
        tangent = (sine_a[0] / cosine_a[1], sine_a[1] / cosine_a[0])
        radius = _add(one, _multiply((v, v), tangent))
        return (
            _add(one, _multiply(cosine, radius)),
            _add(one, _multiply(sine, radius)),
            (1 + v, 1 + v),
        )
    if kind == "torus":
        radius = _add(two, cosine_v)
        return (
            _add(one, _multiply(cosine, radius)),
            _add(one, _multiply(sine, radius)),
            _add(one, sine_v),
        )
    if kind == "revolution":
        return (1 - cosine[1], 1 - cosine[0]), (-sine[1], -sine[0]), (v, v)
    if kind == "placed":
        return _add(two, cosine), _add(three, sine), (4 + v, 4 + v)
    raise ValueError(f"Unknown test kind: {kind}")


def _surfaces() -> dict[str, AbstractSurfacePatch]:
    origin, first, second, axis = (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    cylinder = CylinderPatch(origin, first, second, axis, 1)
    placement = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 1), (0, 1, 0, 2), (0, 0, 1, 3), (0, 0, 0, 1))
    )
    return {
        "plane": PlanePatch(origin, first, second),
        "cylinder": CylinderPatch((-1, 0, 0), first, second, axis, 1),
        "cone": ConePatch(origin, first, second, axis, 1, 0.25),
        "sphere": SpherePatch(origin, first, second, axis, 1),
        "torus": TorusPatch(origin, first, second, axis, 2, 1),
        "revolution": RevolutionSurface(LineCurve((0, 0, 0), axis), (1, 0, 0), axis),
        "placed": _PlacedSurfacePatch(cylinder, placement),
    }


def _domain(surface: AbstractSurfacePatch, /) -> MeshingDomain:
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    uses = tuple(
        PatchCurveUse(
            index,
            LineCurve(corners[first], np.subtract(corners[last], corners[first])),
            0.0,
            1.0,
        )
        for index, (first, last) in enumerate(edges)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        tuple(MeshingDomainCurve(first, last) for first, last in edges),
        4,
        source_id="stored-golden-points",
        source_revision="exact-dyadic-fixture",
        tolerance=1.0e-9,
    )


# Rows are dyadic (numerators in sixteenths); values are native lower endpoints.
_GOLDEN = {
    "plane": (
        ((1, 1), "0x1.1000000000000p+0 0x1.1000000000000p+0 0x1.0000000000000p+0"),
        ((1, 13), "0x1.1000000000000p+0 0x1.d000000000000p+0 0x1.0000000000000p+0"),
        ((6, 4), "0x1.6000000000000p+0 0x1.4000000000000p+0 0x1.0000000000000p+0"),
        ((15, 15), "0x1.f000000000000p+0 0x1.f000000000000p+0 0x1.0000000000000p+0"),
    ),
    "cylinder": (
        ((3, 1), "-0x1.1f2840c263ca0p-6 0x1.7dc102fbaf2b5p-3 0x1.0000000000000p-4"),
        ((3, 2), "-0x1.1f2840c263ca0p-6 0x1.7dc102fbaf2b5p-3 0x1.0000000000000p-3"),
        ((3, 3), "-0x1.1f2840c263ca0p-6 0x1.7dc102fbaf2b5p-3 0x1.8000000000000p-3"),
        ((3, 4), "-0x1.1f2840c263ca0p-6 0x1.7dc102fbaf2b5p-3 0x1.0000000000000p-2"),
    ),
    "cone": (
        ((2, 6), "0x1.0b297e67584d3p+1 0x1.22f90d752a4a0p+0 0x1.6000000000000p+0"),
        ((2, 10), "0x1.1344ef259a369p+1 0x1.2502a1ceec874p+0 0x1.a000000000000p+0"),
        ((2, 14), "0x1.1b605fe3dc1ffp+1 0x1.270c3628aec48p+0 0x1.e000000000000p+0"),
        ((3, 1), "0x1.ff86ebb563387p+0 0x1.307b1500055a2p+0 0x1.1000000000000p+0"),
    ),
    "sphere": (
        ((1, 3), "0x1.fb05a7c7ed29bp+0 0x1.0fb5975daf9e5p+0 0x1.2fb8205f75e56p+0"),
        ((1, 8), "0x1.e038f4eed5497p+0 0x1.0e083d087644bp+0 0x1.7abba1d12c17bp+0"),
        ((1, 9), "0x1.d82243cfb2b33p+0 0x1.0d86a6c18afc8p+0 0x1.88868625b4e1dp+0"),
        ((1, 10), "0x1.cf33826f01b6fp+0 0x1.0cf78af46270bp+0 0x1.95c8ef544210dp+0"),
    ),
    "torus": (
        ((1, 1), "0x1.ff00354f4a5e2p+1 0x1.2ff0026630c5fp+0 0x1.0ffd557776a76p+0"),
        ((1, 2), "0x1.fe40e514212c5p+1 0x1.2fd81062ecedcp+0 0x1.1feaaeee86ee2p+0"),
        ((1, 3), "0x1.fd02de8e463b6p+1 0x1.2fb0424c9ced2p+0 0x1.2fb8205f75e56p+0"),
        ((1, 4), "0x1.fb475fa9bfbd2p+1 0x1.2f78bfee0603dp+0 0x1.3f55dda9e62aep+0"),
    ),
    "revolution": (
        ((1, 1), "0x1.ffd556c165800p-10 -0x1.ffaaaeeed4edbp-5 0x1.0000000000000p-4"),
        ((1, 2), "0x1.ffd556c165800p-10 -0x1.ffaaaeeed4edbp-5 0x1.0000000000000p-3"),
        ((1, 3), "0x1.ffd556c165800p-10 -0x1.ffaaaeeed4edbp-5 0x1.8000000000000p-3"),
        ((1, 4), "0x1.ffd556c165800p-10 -0x1.ffaaaeeed4edbp-5 0x1.0000000000000p-2"),
    ),
    "placed": (
        ((10, 1), "0x1.67cda3b644be1p+1 0x1.cae477aa21085p+1 0x1.0400000000000p+2"),
        ((10, 2), "0x1.67cda3b644be1p+1 0x1.cae477aa21085p+1 0x1.0800000000000p+2"),
        ((10, 3), "0x1.67cda3b644be1p+1 0x1.cae477aa21085p+1 0x1.0c00000000000p+2"),
        ((10, 4), "0x1.67cda3b644be1p+1 0x1.cae477aa21085p+1 0x1.1000000000000p+2"),
    ),
}


@pytest.mark.parametrize("kind", tuple(_GOLDEN))
def test_stored_surface_golden_exact_enclosure_and_fault_adequacy(kind: str) -> None:
    surface = _surfaces()[kind]
    program = prepare_host_program(surface.evaluate, 2)
    assert program is not None
    domain = _domain(surface)
    preparation = SourcePreparation(domain)
    rows = np.asarray([(u / 16, v / 16) for (u, v), _ in _GOLDEN[kind]])
    expected = np.asarray(
        [
            [float.fromhex(value) for value in vector.split()]
            for _, vector in _GOLDEN[kind]
        ]
    )
    lower, upper = program.evaluate(rows, rows)
    actual = domain.evaluate(
        np.zeros(len(rows), dtype=np.int64), rows, preparation=preparation
    )
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(actual.view(np.uint64), lower.view(np.uint64))
    adequate = 0
    for index, (u, v) in enumerate(rows):
        eager = np.asarray(surface.evaluate(jnp.asarray(rows[index])), dtype=np.float64)
        separated = False
        for axis, (exact_lower, exact_upper) in enumerate(
            _exact_coordinates(kind, Fraction(float(u)), Fraction(float(v)))
        ):
            assert Fraction(float(lower[index, axis])) <= exact_lower
            assert exact_upper <= Fraction(float(upper[index, axis]))
            ulp = math.ulp(float((exact_lower + exact_upper) / 2))
            assert math.ulp(float(exact_lower)) == math.ulp(float(exact_upper)) == ulp
            if exact_lower - Fraction(float(actual[index, axis])) > 2 * Fraction(ulp):
                # Any faithful evaluation lies within two doubles of the true
                # coordinate, so cannot equal this more distant lower endpoint.
                assert eager[axis] != actual[index, axis]
                separated = True
        adequate += int(separated)
    minimum = 0 if kind == "plane" else 4
    assert adequate >= minimum
    assert not preparation.unsupported_points


def test_declined_bspline_surface_is_declared_and_uses_prior_jax_values() -> None:
    surface = BSplineSurfacePatch(
        np.asarray(
            (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
        ),
        np.ones((2, 2)),
        np.asarray((0.0, 0.0, 1.0, 1.0)),
        np.asarray((0.0, 0.0, 1.0, 1.0)),
        1,
        1,
    )
    assert prepare_host_program(surface.evaluate, 2) is None
    domain = _domain(surface)
    preparation = SourcePreparation(domain)
    rows = np.asarray(((0.125, 0.25), (0.5, 0.75), (0.875, 0.5)))
    actual = domain.evaluate(
        np.zeros(len(rows), dtype=np.int64), rows, preparation=preparation
    )
    expected = bucketed_query_rows(surface.evaluate, rows)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert preparation.unsupported_points == {
        f"{type(surface).__module__}.{type(surface).__qualname__}": len(rows)
    }
