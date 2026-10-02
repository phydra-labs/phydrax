from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import pytest
from jax.typing import ArrayLike
from numpy.polynomial import chebyshev, legendre

from phydrax.discretization import AxisDomain
from phydrax.discretization.spectral import (
    ChebyshevBasisPlan,
    ConstrainedBasisPlan,
    CosineBasisPlan,
    FourierBasisPlan,
    LegendreBasisPlan,
    periodic_trace_row,
    PreparedSpectralAxis,
    RationalChebyshevLineBasisPlan,
    SineBasisPlan,
    SpectralBoundaryConditionPlan,
    SpectralPrecisionPolicy,
    SpectralTraceTerm,
)


# Bounded and periodic seams deliberately avoid the reference interval so the
# oracle exercises the physical-coordinate derivative scaling.
_LOWER, _UPPER = 0.25, 2.0
_LENGTH = _UPPER - _LOWER
_PERIOD_LOWER, _PERIOD_UPPER = 0.3, 2.3
_PERIOD = _PERIOD_UPPER - _PERIOD_LOWER
_PHASE = complex(np.exp(0.7j))
_JETS = {
    "order0": ({0: 0.75}, {0: -1.25}),
    "order1": ({1: 0.75}, {1: -1.25}),
    "order2": ({2: 0.75}, {2: -1.25}),
    "order3": ({3: 0.75}, {3: -1.25}),
    "mixed": ({0: 1.0, 1: 0.5, 3: -0.2}, {0: 0.3, 2: 1.5, 3: 0.1}),
}
_TRANSPORTS = pytest.mark.parametrize(
    "transport", (1.0, -1.0, _PHASE), ids=("identity", "anti", "bloch")
)

type Terms = Mapping[int, float]


def _random_coefficients(count: int, seed: int, /) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=count) + 1j * rng.normal(size=count)


def _polynomial_axis(
    family: Literal["chebyshev", "legendre"], count: int, /
) -> PreparedSpectralAxis:
    plan = (
        ChebyshevBasisPlan(count)
        if family == "chebyshev"
        else LegendreBasisPlan(count, node_rule="lobatto")
    )
    return plan.prepare(
        AxisDomain.interval(_LOWER, _UPPER), precision=SpectralPrecisionPolicy()
    )


def _polynomial_jet(
    family: Literal["chebyshev", "legendre"],
    coefficients: np.ndarray,
    terms: Terms,
    reference: np.ndarray | float,
    /,
) -> np.ndarray:
    """NumPy series oracle: Chebyshev `T_n`, orthonormal Legendre `sqrt((2n+1)/L) P_n`."""
    degrees = np.arange(coefficients.size)
    if family == "legendre":
        series = coefficients * np.sqrt((2.0 * degrees + 1.0) / _LENGTH)
        value, derivative = legendre.legval, legendre.legder
    else:
        series = coefficients
        value, derivative = chebyshev.chebval, chebyshev.chebder
    return sum(
        (
            weight
            * (2.0 / _LENGTH) ** order
            * value(reference, derivative(series, order))
            for order, weight in terms.items()
        ),
        start=np.zeros_like(np.asarray(reference), dtype=np.complex128),
    )


def _fourier_axis(count: int, /) -> PreparedSpectralAxis:
    return FourierBasisPlan(count).prepare(
        AxisDomain.periodic(_PERIOD_LOWER, _PERIOD_UPPER),
        precision=SpectralPrecisionPolicy(),
    )


def _fourier_jet(
    coefficients: np.ndarray, terms: Terms, points: np.ndarray, /
) -> np.ndarray:
    """Direct series oracle `exp(2 pi i m (x - a) / L) / sqrt(L)`, FFT mode order."""
    count = coefficients.size
    wave = 2.0j * np.pi * np.rint(np.fft.fftfreq(count) * count) / _PERIOD
    modes = np.exp(np.outer(points - _PERIOD_LOWER, wave)) / np.sqrt(_PERIOD)
    return sum(
        (
            weight * (modes * wave**order) @ coefficients
            for order, weight in terms.items()
        ),
        start=np.zeros(points.shape, dtype=np.complex128),
    )


def _trigonometric_jet(
    family: Literal["sine", "cosine"],
    coefficients: np.ndarray,
    terms: Terms,
    points: np.ndarray,
    /,
) -> np.ndarray:
    """Orthonormal DST-II/DCT-I series oracle with closed-form derivative phases."""
    count = coefficients.size
    if family == "sine":
        numbers = np.arange(1, count + 1, dtype=np.float64)
        edge = np.where(numbers == count, np.sqrt(0.5), 1.0)
        shape = np.sin
    else:
        numbers = np.arange(count, dtype=np.float64)
        edge = np.where((numbers == 0) | (numbers == count - 1), np.sqrt(0.5), 1.0)
        shape = np.cos
    wave = np.pi * numbers / _LENGTH
    angle = np.outer(points - _LOWER, wave)
    amplitude = np.sqrt(2.0 / _LENGTH) * edge
    return sum(
        (
            weight
            * (amplitude * wave**order * shape(angle + 0.5 * np.pi * order))
            @ coefficients
            for order, weight in terms.items()
        ),
        start=np.zeros(points.shape, dtype=np.complex128),
    )


@pytest.mark.parametrize("jet", tuple(_JETS), ids=tuple(_JETS))
@pytest.mark.parametrize("family", ("chebyshev", "legendre"))
@_TRANSPORTS
def test_polynomial_periodic_row_matches_paired_endpoint_jets(
    family: Literal["chebyshev", "legendre"], jet: str, transport: complex
) -> None:
    source, target = _JETS[jet]
    prepared = _polynomial_axis(family, 9)
    coefficients = _random_coefficients(9, 11)
    reference_nodes = (2.0 * np.asarray(prepared.nodes) - (_LOWER + _UPPER)) / _LENGTH

    row = periodic_trace_row(
        prepared, source_terms=source, target_terms=target, transport=transport
    )
    expected = _polynomial_jet(
        family, coefficients, target, 1.0
    ) - transport * _polynomial_jet(family, coefficients, source, -1.0)

    np.testing.assert_allclose(
        np.asarray(prepared.synthesize(coefficients)),
        _polynomial_jet(family, coefficients, {0: 1.0}, reference_nodes),
        rtol=1e-12,
        atol=1e-12,
    )
    assert row.shape == (prepared.mode_count,)
    np.testing.assert_allclose(row @ coefficients, expected, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("family", ("sine", "cosine"))
@_TRANSPORTS
def test_trigonometric_periodic_row_matches_paired_endpoint_jets(
    family: Literal["sine", "cosine"], transport: complex
) -> None:
    source, target = _JETS["mixed"]
    plan = SineBasisPlan(8) if family == "sine" else CosineBasisPlan(8)
    prepared = plan.prepare(
        AxisDomain.interval(_LOWER, _UPPER), precision=SpectralPrecisionPolicy()
    )
    coefficients = _random_coefficients(8, 5)
    endpoints = np.asarray([_LOWER, _UPPER])

    row = periodic_trace_row(
        prepared, source_terms=source, target_terms=target, transport=transport
    )
    upper = _trigonometric_jet(family, coefficients, target, endpoints)[1]
    lower = _trigonometric_jet(family, coefficients, source, endpoints)[0]

    np.testing.assert_allclose(
        np.asarray(prepared.synthesize(coefficients)),
        _trigonometric_jet(family, coefficients, {0: 1.0}, np.asarray(prepared.nodes)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        row @ coefficients, upper - transport * lower, rtol=1e-10, atol=1e-9
    )


@pytest.mark.parametrize("count", (8, 9), ids=("even", "odd"))
def test_fourier_identity_periodic_row_is_structurally_zero(count: int) -> None:
    jet = {0: 1.0, 1: 0.5, 2: -0.25, 3: 0.125}

    row = periodic_trace_row(
        _fourier_axis(count), source_terms=jet, target_terms=jet, transport=1.0
    )

    assert row.shape == (count,)
    assert np.all(row == 0.0)


@pytest.mark.parametrize("count", (8, 9), ids=("even", "odd"))
@_TRANSPORTS
def test_fourier_transported_row_matches_synthesized_seam_jets(
    count: int, transport: complex
) -> None:
    source, target = _JETS["mixed"]
    prepared = _fourier_axis(count)
    coefficients = _random_coefficients(count, 3)
    seam = np.asarray([_PERIOD_LOWER, _PERIOD_UPPER])

    row = periodic_trace_row(
        prepared, source_terms=source, target_terms=target, transport=transport
    )
    upper = _fourier_jet(coefficients, target, seam)[1]
    lower = _fourier_jet(coefficients, source, seam)[0]

    np.testing.assert_allclose(
        np.asarray(prepared.synthesize(coefficients)),
        _fourier_jet(coefficients, {0: 1.0}, np.asarray(prepared.nodes)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        row @ coefficients, upper - transport * lower, rtol=1e-10, atol=1e-9
    )


@pytest.fixture(scope="module")
def dtype_axes() -> tuple[PreparedSpectralAxis, PreparedSpectralAxis]:
    # Module scope prepares the axes before the strict context; only the row
    # construction is the strict dtype surface under test.
    return _polynomial_axis("chebyshev", 6), _fourier_axis(6)


@pytest.mark.strict_jax
def test_periodic_row_dtype_follows_basis_and_term_values(
    dtype_axes: tuple[PreparedSpectralAxis, PreparedSpectralAxis],
) -> None:
    chebyshev_axis, fourier_axis = dtype_axes

    real = periodic_trace_row(
        chebyshev_axis, source_terms={1: 1.0}, target_terms={1: 1.0}, transport=-1.0
    )
    phased = periodic_trace_row(
        chebyshev_axis, source_terms={1: 1.0}, target_terms={1: 1.0}, transport=_PHASE
    )
    fourier = periodic_trace_row(
        fourier_axis, source_terms={0: 1.0}, target_terms={0: 1.0}, transport=-1.0
    )

    assert real.dtype == np.float64
    assert phased.dtype == np.complex128
    assert fourier.dtype == np.complex128
    np.testing.assert_allclose(fourier, np.full((6,), 2.0 / np.sqrt(_PERIOD)))


def _rational_axis() -> PreparedSpectralAxis:
    return RationalChebyshevLineBasisPlan(8, 1.0).prepare(
        AxisDomain.real_line(), precision=SpectralPrecisionPolicy()
    )


def _constrained_axis() -> PreparedSpectralAxis:
    return ConstrainedBasisPlan(
        ChebyshevBasisPlan(8), SpectralBoundaryConditionPlan.dirichlet()
    ).prepare(AxisDomain.interval(_LOWER, _UPPER), precision=SpectralPrecisionPolicy())


@pytest.mark.parametrize(
    "axis", (_rational_axis, _constrained_axis), ids=("rational", "constrained")
)
def test_periodic_row_refuses_axes_without_unconstrained_finite_seam(
    axis: Callable[[], PreparedSpectralAxis],
) -> None:
    with pytest.raises(ValueError, match="unconstrained Fourier, Chebyshev"):
        periodic_trace_row(axis(), source_terms={0: 1.0}, target_terms={0: 1.0})


@pytest.mark.parametrize(
    ("source", "target", "transport", "error"),
    (
        ({}, {0: 1.0}, 1.0, TypeError),
        ({True: 1.0}, {0: 1.0}, 1.0, TypeError),
        ({-1: 1.0}, {0: 1.0}, 1.0, ValueError),
        ({0: 1.0}, {0: 0.0}, 1.0, ValueError),
        ({0: 1.0}, {0: float("nan")}, 1.0, ValueError),
        ({0: 1.0}, {0: np.ones((2,))}, 1.0, ValueError),
        ({0: 1.0}, {0: "1.0"}, 1.0, TypeError),
        ({0: 1.0}, {0: 1.0}, 0.0, ValueError),
        ({0: 1.0}, {0: 1.0}, np.inf, ValueError),
    ),
    ids=(
        "empty",
        "bool-order",
        "negative-order",
        "zero-coefficient",
        "nan-coefficient",
        "array-coefficient",
        "string-coefficient",
        "zero-transport",
        "infinite-transport",
    ),
)
def test_periodic_row_refuses_invalid_seam_terms(
    source: Mapping[int, ArrayLike],
    target: Mapping[int, ArrayLike],
    transport: ArrayLike,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        periodic_trace_row(
            _polynomial_axis("chebyshev", 4),
            source_terms=source,
            target_terms=target,
            transport=transport,
        )


def test_periodic_row_refuses_non_axis_owner() -> None:
    with pytest.raises(TypeError, match="PreparedSpectralAxis"):
        periodic_trace_row(
            "axis",  # ty: ignore[invalid-argument-type]
            source_terms={0: 1.0},
            target_terms={0: 1.0},
        )


@pytest.fixture(scope="module")
def float32_axes() -> Mapping[str, PreparedSpectralAxis]:
    precision = SpectralPrecisionPolicy(np.float32)
    bounded = AxisDomain.interval(_LOWER, _UPPER)
    periodic = AxisDomain.periodic(_PERIOD_LOWER, _PERIOD_UPPER)
    return {
        "chebyshev": ChebyshevBasisPlan(9).prepare(bounded, precision=precision),
        "legendre": LegendreBasisPlan(9, node_rule="lobatto").prepare(
            bounded, precision=precision
        ),
        "fourier": FourierBasisPlan(9).prepare(periodic, precision=precision),
        "sine": SineBasisPlan(9).prepare(bounded, precision=precision),
        "cosine": CosineBasisPlan(9).prepare(bounded, precision=precision),
    }


@pytest.mark.parametrize("family", ("chebyshev", "legendre", "fourier", "sine", "cosine"))
@pytest.mark.strict_jax
def test_periodic_trace_rows_retain_host_precision_for_float32_axes(
    family: Literal["chebyshev", "legendre", "fourier", "sine", "cosine"],
    float32_axes: Mapping[str, PreparedSpectralAxis],
) -> None:
    prepared = float32_axes[family]
    coefficients = _random_coefficients(9, 29)
    source, target = _JETS["mixed"]
    row = periodic_trace_row(
        prepared, source_terms=source, target_terms=target, transport=_PHASE
    )
    if family in ("chebyshev", "legendre"):
        expected = _polynomial_jet(
            family, coefficients, target, 1.0
        ) - _PHASE * _polynomial_jet(family, coefficients, source, -1.0)
    elif family == "fourier":
        endpoints = np.asarray((_PERIOD_LOWER, _PERIOD_UPPER), dtype=np.float64)
        expected = _fourier_jet(coefficients, target, endpoints)[1] - (
            _PHASE * _fourier_jet(coefficients, source, endpoints)[0]
        )
    else:
        endpoints = np.asarray((_LOWER, _UPPER), dtype=np.float64)
        expected = _trigonometric_jet(family, coefficients, target, endpoints)[1] - (
            _PHASE * _trigonometric_jet(family, coefficients, source, endpoints)[0]
        )
    np.testing.assert_allclose(row @ coefficients, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(
    ("family", "order"), (("sine", 0), ("cosine", 1)), ids=("sine-value", "cosine-slope")
)
@pytest.mark.strict_jax
def test_trigonometric_endpoint_structural_zeros_are_exact(
    family: str, order: int, float32_axes: Mapping[str, PreparedSpectralAxis]
) -> None:
    row = periodic_trace_row(
        float32_axes[family],
        source_terms={order: 1.0},
        target_terms={order: 1.0},
        transport=-1.0,
    )
    np.testing.assert_array_equal(row, np.zeros((9,), dtype=np.float64))


@pytest.mark.strict_jax
def test_polynomial_trace_above_mode_capacity_is_zero_before_scaling(
    float32_axes: Mapping[str, PreparedSpectralAxis],
) -> None:
    row = periodic_trace_row(
        float32_axes["chebyshev"], source_terms={1000: 1.0}, target_terms={1000: 1.0}
    )
    np.testing.assert_array_equal(row, np.zeros((9,), dtype=np.float64))


@pytest.mark.parametrize("order", (1.5, True), ids=("fractional", "boolean"))
def test_spectral_trace_term_refuses_noninteger_orders(order: float) -> None:
    with pytest.raises(TypeError, match="order must be an integer"):
        # ty: ignore[invalid-argument-type]
        SpectralTraceTerm(order)


def test_periodic_trace_row_refuses_transport_overflow(
    dtype_axes: tuple[PreparedSpectralAxis, PreparedSpectralAxis],
) -> None:
    with pytest.raises(ValueError, match="not representable"):
        periodic_trace_row(
            dtype_axes[0],
            source_terms={1: 1.0},
            target_terms={1: 1.0},
            transport=1e308,
        )
