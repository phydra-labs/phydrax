"""Exact elimination and source-branch projection contracts."""

import math
from fractions import Fraction as F

import numpy as np

from phydrax.algebraic._exact_univariate import isolate_real_roots, RealRootIsolator
from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import (
    orthogonal_cylinder_family,
    SurfaceRegion,
)
from phydrax.geometry.brep._intersection_projection import (
    _stationary_polynomials,
    project_intersection_curve,
)
from phydrax.geometry.brep._patches import CylinderPatch
from phydrax.geometry.brep._projection_contracts import (
    BRepProjectionPolicy,
    BRepProjectionStatus,
)


def _eval(poly: tuple[F, ...], t: F) -> F:
    return sum((coefficient * t**i for i, coefficient in enumerate(poly)), F(0))


def test_primitive_integer_remainder_matches_exact_rational_division() -> None:
    from phydrax.algebraic._exact_univariate import _divmod, _primitive_remainder

    rng = np.random.default_rng(2026)
    for _ in range(200):
        left = tuple(int(value) for value in rng.integers(-20, 21, size=7))
        right = tuple(int(value) for value in rng.integers(-20, 21, size=4))
        left = (*left[:-1], left[-1] or 1)
        right = (*right[:-1], right[-1] or -1)
        _, exact = _divmod(tuple(map(F, left)), tuple(map(F, right)))
        remainder = _primitive_remainder(left, right)
        assert len(remainder) == len(exact)
        if exact == (0,):
            assert remainder == (0,)
            continue
        factor = F(remainder[-1]) / exact[-1]
        assert factor > 0
        assert tuple(F(value) for value in remainder) == tuple(factor * value for value in exact)


def test_interval_only_isolation_preserves_nondyadic_rational_root() -> None:
    polynomial = (2, -6, -1, 3)  # (3x - 1)(x² - 2)
    tolerance = F(1, 1024)
    exact = isolate_real_roots(polynomial, tolerance=tolerance)
    intervals = isolate_real_roots(
        polynomial, tolerance=tolerance, recognize_rational_roots=False
    )
    assert (F(1, 3), F(1, 3), 1) in exact
    assert len(intervals) == len(exact) == 3
    root = next((lo, hi) for lo, hi, _ in intervals if lo <= F(1, 3) <= hi)
    assert root[0] < F(1, 3) < root[1]
    assert all(hi - lo <= tolerance for lo, hi, _ in intervals)
    assert all(left[1] < right[0] for left, right in zip(intervals, intervals[1:]))


def test_primitive_sturm_retains_distinct_roots_and_multiplicities() -> None:
    polynomial = [1]
    for factor in ((-1, 3), (-1, 3), (2, 1), (2, 1), (2, 1), (-2, 0, 1)):
        product = [0] * (len(polynomial) + len(factor) - 1)
        for i, first in enumerate(polynomial):
            for j, second in enumerate(factor):
                product[i + j] += first * second
        polynomial = product
    roots = isolate_real_roots(tuple(polynomial), tolerance=F(1, 4096))
    assert len(roots) == 4
    assert (F(-2), F(-2), 3) in roots
    assert (F(1, 3), F(1, 3), 2) in roots
    assert sum(multiplicity for _, _, multiplicity in roots) == 7
    assert all(left[1] < right[0] for left, right in zip(roots, roots[1:]))
    assert isolate_real_roots(tuple(-value for value in polynomial), tolerance=F(1, 4096)) == roots
    assert isolate_real_roots((3,), tolerance=F(1, 4096)) == ()
    from phydrax.algebraic._exact_univariate import _gcd

    assert _gcd((F(0),), (F(0),)) == (F(0),)
    assert _gcd((F(0),), (F(-6), F(3))) == (F(-2), F(1))


def test_stationary_elimination_matches_unsquared_derivative() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0)
    small = CylinderPatch((0, 0.125, -0.25), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -3.0), (math.tau, 3.0)))
    family = orthogonal_cylinder_family(SurfaceRegion(large, box), SurfaceRegion(small, box))
    assert family is not None
    for sign in (-1, 1):
        aa, bb, h, polynomial = _stationary_polynomials(
            family, sign, F(1, 4), F(2, 7), F(-3, 8)
        )
        for t in (F(-2), F(-1, 3), F(0), F(3, 2)):
            assert (
                _eval(polynomial, t)
                == _eval(aa, t) ** 2 * _eval(h, t) - _eval(bb, t) ** 2
            )
            # Independently differentiate the two rational terms before
            # squaring: A = D L' - 2 D' L, B = -s qx D(D H' - 2 D' H).
            d, dp = 1 + t * t, 2 * t
            a, b = family.small_normal_coefficients
            c, e = family.small_large_axis_coefficients
            n = family.delta_n * d + family.small_radius * (a * (1 - t * t) + 2 * b * t)
            np_ = family.delta_n * dp + family.small_radius * (-2 * a * t + 2 * b)
            z = family.delta_large_axis * d + family.small_radius * (
                c * (1 - t * t) + 2 * e * t
            )
            zp = family.delta_large_axis * dp + family.small_radius * (-2 * c * t + 2 * e)
            h0 = family.large_radius**2 * d * d - n * n
            hp = 2 * family.large_radius**2 * d * dp - 2 * n * np_
            w, wp = z - F(-3, 8) * d, zp - F(-3, 8) * dp
            ell = -2 * F(2, 7) * n * d + w * w
            ellp = -2 * F(2, 7) * (np_ * d + n * dp) + 2 * w * wp
            assert _eval(aa, t) == ellp * d - 2 * ell * dp
            assert _eval(bb, t) == -sign * F(1, 4) * d * (hp * d - 2 * h0 * dp)


def test_projection_on_certified_source_intersection() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    assert result.complete and len(result.curves) == 2
    curve = result.curves[0]
    parameter = 0.37
    query = np.asarray(curve.evaluate_coupled(parameter).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.2, 0.6),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert 0.0 <= projection.lower < 1e-8
    assert projection.upper < 1e-8
    assert abs(projection.parameter - parameter) < 1e-7
    assert projection.root_id is not None


def test_stationary_root_at_exact_large_seam_node() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) > 0)
    query = np.asarray((1.0, 0.0, 0.5))
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert projection.upper < 1e-8
    assert projection.lower <= 0.0 <= projection.upper
    np.testing.assert_allclose(projection.point, query, rtol=0.0, atol=1e-12)


def test_projection_of_singleton_endpoint() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = result.curves[0]
    query = np.asarray(curve.evaluate_coupled(0.0).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.0, 0.0),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.SEAM)
    assert projection.parameter == 0.0
    assert projection.upper < 1e-8


def test_pi_chart_node_does_not_confuse_symbolic_closing_turn() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    theta = np.asarray(curve.chart_start)[:, 2]
    node = int(np.argmin(abs(theta - math.pi)))
    assert node > 0 and theta[node] == math.pi
    query = np.asarray(curve.evaluate_coupled(float(node)).point, dtype=np.float64)
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.UNIQUE)
    assert abs(projection.parameter - node) < 1e-7
    assert projection.upper < 1e-8


def test_projection_preserves_sheet_for_query_on_other_branch() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    positive = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) > 0)
    negative = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    query = np.asarray(negative.evaluate_coupled(0.37).point, dtype=np.float64)
    projection = project_intersection_curve(
        positive,
        (0.0, float(positive.num_charts)),
        query,
        BRepProjectionPolicy(),
        edge_range=(0.0, float(positive.num_charts)),
    )
    assert projection.status in (
        int(BRepProjectionStatus.UNIQUE),
        int(BRepProjectionStatus.AMBIGUOUS),
    )
    assert projection.point[0] > 0
    assert projection.lower > 1.7


def test_two_distinct_source_minima_are_ambiguous() -> None:
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)))
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    curve = next(curve for curve in result.curves if float(curve.evaluate_coupled(0.0).point[0]) < 0)
    projection = project_intersection_curve(
        curve,
        (0.0, float(curve.num_charts)),
        np.zeros(3),
        BRepProjectionPolicy(),
        edge_range=(0.0, float(curve.num_charts)),
    )
    assert projection.status == int(BRepProjectionStatus.AMBIGUOUS)
    assert projection.lower <= 1.0 <= projection.upper


# Stationary polynomial A²H-B² of the original STEP quadric-routine seam node
# (-0x1.7p-1, 0x1.52dc7707ed4d5p-53, 0x1.3333333333333p-1) against its
# transverse-cylinder intersection branch, the policy isolation width, and the
# intervals the unmemoized Sturm isolator returned for it (both modes agree).
# Its two stationary roots near t = +-1.15e-16 straddle the periodic seam.
_SEAM_STATIONARY = tuple(
    map(
        F,
        (
            "1037923454694657707547740903698505470201695960420676147869640225/8343699359066055009355553539724812947666814540455674882605631280555545803830627148527195652096",
            "0",
            "-725135032491349965400611976040025070049358707893520474348913109306750748384465126196775208277867061252308947979616639648708425/169230328010303641331690318856389386196071598838855992136870091590247882556495704531248437872567112920983350278405979725889536",
            "940946008271763830249767273609194956780826011023312978083058375/231584178474632390847141970017375815706539969331281128078915168015826259279872",
            "-1037923454694657707547740903698505470201695960420676147869640225/8343699359066055009355553539724812947666814540455674882605631280555545803830627148527195652096",
            "10687405318625801058044774711723591118467685894585324751108699016881030997951343406264042158625/4697085165547666455778961193578674054751365097816639741414581943064418050229216886927397996769537406063869952",
            "3532463097788674595155722234886897446433238136337678631216055941093154384207258680000783422503584661904694585859745115931588425/84615164005151820665845159428194693098035799419427996068435045795123941278247852265624218936283556460491675139202989862944768",
            "0",
            "155021395118995976696408606608792351106335058950144746208801362059195170277333713181301460829775/8343699359066055009355553539724812947666814540455674882605631280555545803830627148527195652096",
            "-10687405318625801058044774711723591118467685894585324751108699016881030997951343406264042158625/4697085165547666455778961193578674054751365097816639741414581943064418050229216886927397996769537406063869952",
            "4889521098103299294109608541653719682718400148994795839385372554266056523261121981411241220173568140052462327780640312917051575/169230328010303641331690318856389386196071598838855992136870091590247882556495704531248437872567112920983350278405979725889536",
            "-940946008271763830249767273609194956780826011023312978083058375/231584178474632390847141970017375815706539969331281128078915168015826259279872",
            "1037923454694657707547740903698505470201695960420676147869640225/8343699359066055009355553539724812947666814540455674882605631280555545803830627148527195652096",
        ),
    )
)
_SEAM_TOLERANCE = F("4835703278458517/77371252455336267181195264")
_SEAM_ROOTS = tuple(
    (F(lower), F(upper), 1)
    for lower, upper in (
        (
            "-12374501611002927339608217157291025202891932515/22835963083295358096932575511191922182123945984",
            "-3093625402750731179432093888867989492692219915/5708990770823839524233143877797980545530986496",
        ),
        (
            "-2621879841601819067232123052855/11417981541647679048466287755595961091061972992",
            "-2621879841601819067232123052855/22835963083295358096932575511191922182123945984",
        ),
        (
            "2621879841601819067232123052855/22835963083295358096932575511191922182123945984",
            "2621879841601819067232123052855/11417981541647679048466287755595961091061972992",
        ),
        (
            "12374501611002922095848533953652890738645826805/22835963083295358096932575511191922182123945984",
            "3093625402750731179432093888867989492692219915/5708990770823839524233143877797980545530986496",
        ),
        (
            "238914417780237850362672349438667804304272266374771187976773135/22835963083295358096932575511191922182123945984",
            "119457208890118925181336174719335213092056934096919210049912995/11417981541647679048466287755595961091061972992",
        ),
        (
            "253482370083910890018932858550782558112529395411218397820752315/11417981541647679048466287755595961091061972992",
            "506964740167821780037865717101567738104900392641504027764557485/22835963083295358096932575511191922182123945984",
        ),
    )
)


def test_isolator_reuse_reproduces_fresh_isolation_on_seam_operand() -> None:
    isolator = RealRootIsolator(_SEAM_STATIONARY)
    # The projection's first pass, then its rational-recognizing refinements.
    ladder = [(_SEAM_TOLERANCE, False), (_SEAM_TOLERANCE, True)]
    ladder += [(_SEAM_TOLERANCE / 16**k, True) for k in (1, 2, 3)]
    for tolerance, recognize in ladder:
        reused = isolator.isolate(tolerance=tolerance, recognize_rational_roots=recognize)
        # A fresh isolator shares no memo and replays the full isolation.
        fresh = isolate_real_roots(
            _SEAM_STATIONARY, tolerance=tolerance, recognize_rational_roots=recognize
        )
        assert reused == fresh
        again = isolator.isolate(tolerance=tolerance, recognize_rational_roots=recognize)
        assert again is reused
        if tolerance == _SEAM_TOLERANCE:
            assert reused == _SEAM_ROOTS
        assert len(reused) == 6
        assert all(hi - lo <= tolerance for lo, hi, _ in reused)
        assert all(left[1] < right[0] for left, right in zip(reused, reused[1:]))
    # The two stationary roots straddling the seam stay separated across t = 0.
    assert _SEAM_ROOTS[1][1] < 0 < _SEAM_ROOTS[2][0]
    # A different polynomial is a different isolator: its roots are recomputed.
    shifted = (_SEAM_STATIONARY[0] + 1, *_SEAM_STATIONARY[1:])
    replayed = RealRootIsolator(shifted).isolate(tolerance=_SEAM_TOLERANCE)
    assert replayed == isolate_real_roots(shifted, tolerance=_SEAM_TOLERANCE)
    assert replayed != _SEAM_ROOTS


def test_isolator_reuse_matches_fresh_isolation_with_repeated_rational_roots() -> None:
    def product(*factors: tuple[F, ...]) -> tuple[F, ...]:
        result = (F(1),)
        for factor in factors:
            out = [F(0)] * (len(result) + len(factor) - 1)
            for i, a in enumerate(result):
                for j, b in enumerate(factor):
                    out[i + j] += a * b
            result = tuple(out)
        return result

    near = F(1, 3) + F(1, 1 << 44)
    polynomial = product(
        (F(-1, 3), F(1)),
        (F(-1, 3), F(1)),
        (-near, F(1)),
        (F(5, 8), F(1)),
        (F(5, 8), F(1)),
        (F(5, 8), F(1)),
        (F(-2), F(0), F(1)),
        (F(-7, 11), F(1)),
    )
    isolator = RealRootIsolator(polynomial)
    for exponent in (2, 6, 12, 20):
        for recognize in (False, True):
            tolerance = F(1, 1 << exponent)
            assert isolator.isolate(
                tolerance=tolerance, recognize_rational_roots=recognize
            ) == isolate_real_roots(
                polynomial, tolerance=tolerance, recognize_rational_roots=recognize
            )
    roots = isolator.isolate(tolerance=F(1, 1 << 20))
    assert (F(-5, 8), F(-5, 8), 3) in roots and (F(1, 3), F(1, 3), 2) in roots
    assert (F(7, 11), F(7, 11), 1) in roots and (near, near, 1) in roots
