#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent periodic pair-average oracles on oversampled trig fields."""

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy import special

from phydrax.discretization import (
    AxisDomain,
    ChebyshevBasisPlan,
    FourierBasisPlan,
    PaddingDealiasingPlan,
    PeriodicIncrementMomentEstimatorPlan,
    PeriodicLatticeIncrementMomentPlan,
    PeriodicRadialIncrementMomentPlan,
    PeriodicThirdIncrementMomentPlan,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import (
    IncrementAverage,
    IncrementProjection,
    ThirdIncrementStatistic,
)
from phydrax.execution import MemoryBudgetError


pytestmark = pytest.mark.strict_jax


def _space(
    dimension: int, n: int = 6, lengths: tuple[float, ...] | None = None
) -> TensorSpectralDiscretization:
    box = (2 * np.pi,) * dimension if lengths is None else lengths
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(n) for _ in box), axis_names=tuple("xyz"[:dimension])
    ).prepare(tuple(AxisDomain.periodic(0, length) for length in box))


def _coefficients(space: TensorSpectralDiscretization, members: int = 3) -> np.ndarray:
    rng = np.random.default_rng(6421)
    shape = space.modal_shape
    dimension = len(shape)
    volume = prod(float(axis.length) for axis in space.axes)
    values = rng.normal(size=(members,) + shape + (dimension,))
    c = (
        np.fft.fftn(values, axes=tuple(range(1, dimension + 1)))
        * np.sqrt(volume)
        / prod(shape)
    )
    for a, n in enumerate(shape, start=1):
        if n % 2 == 0:
            index = [slice(None)] * c.ndim
            index[a] = n // 2
            c[tuple(index)] = 0
    return c


def _physical(
    c: np.ndarray, space: TensorSpectralDiscretization, refinement: int
) -> np.ndarray:
    """NumPy trig synthesis: copy modes by signed integer labels, not package transfer."""
    shape = space.modal_shape
    target = tuple(refinement * n for n in shape)
    modes = tuple((np.fft.fftfreq(n) * n).astype(np.int64) for n in shape)
    padded = np.zeros(target + (c.shape[-1],), dtype=np.complex128)
    for index in np.ndindex(shape):
        destination = tuple(int(modes[a][i]) % target[a] for a, i in enumerate(index))
        padded[destination] = c[index]
    volume = prod(float(axis.length) for axis in space.axes)
    return (
        np.fft.ifftn(padded, axes=tuple(range(len(shape)))).real
        * prod(target)
        / np.sqrt(volume)
    )


def _pair_oracle(
    c: np.ndarray,
    space: TensorSpectralDiscretization,
    *,
    statistic: Literal["second", "longitudinal_cube", "energy_flux"],
    refinement: int = 4,
) -> np.ndarray:
    shape = space.modal_shape
    dimension = len(shape)
    u = _physical(c, space, refinement)
    result = np.zeros(shape, dtype=np.float64)
    for index in np.ndindex(shape):
        shift = tuple(refinement * i for i in index)
        delta = np.roll(u, tuple(-i for i in shift), axis=tuple(range(dimension))) - u
        h = np.asarray(
            [
                ((i if i <= n // 2 else i - n) * float(axis.length) / n)
                for i, n, axis in zip(index, shape, space.axes, strict=True)
            ]
        )
        norm = np.linalg.norm(h)
        if statistic == "second":
            result[index] = np.mean(np.sum(delta**2, axis=-1))
        elif norm != 0:
            longitudinal = np.sum(delta * (h / norm), axis=-1)
            result[index] = np.mean(
                longitudinal**3
                if statistic == "longitudinal_cube"
                else np.sum(delta**2, axis=-1) * longitudinal
            )
    return result


@pytest.mark.parametrize(
    "dimension", [2, 3], ids=["two-dimensional", "three-dimensional"]
)
def test_lattice_second_matches_fine_pair_average(dimension: int) -> None:
    space = _space(dimension, n=6, lengths=(3.1, 4.7) if dimension == 2 else None)
    c = _coefficients(space)
    plan = PeriodicLatticeIncrementMomentPlan(
        space, component_shape=(dimension,), member_chunk_size=2
    )
    result = eqx.filter_jit(plan)(c)
    expected = np.stack([_pair_oracle(member, space, statistic="second") for member in c])
    np.testing.assert_allclose(result.values, expected, atol=2e-13, rtol=2e-13)
    assert result.exact
    assert np.all(result.finite)
    np.testing.assert_array_equal(result.forbidden_coefficient_energy, 0)


@pytest.mark.parametrize(
    "dimension", [2, 3], ids=["two-dimensional", "three-dimensional"]
)
@pytest.mark.parametrize(
    "statistic", ["longitudinal_cube", "energy_flux"], ids=["longitudinal", "energy-flux"]
)
def test_third_matches_fine_pair_average(
    dimension: int, statistic: ThirdIncrementStatistic
) -> None:
    space = _space(dimension)
    c = _coefficients(space, members=1)
    plan = PeriodicThirdIncrementMomentPlan(
        space,
        statistic=statistic,
        dealiasing=PaddingDealiasingPlan(2),
        member_chunk_size=2,
        maximum_workspace_bytes=2**24,
    )
    result = eqx.filter_jit(plan)(c)
    expected = _pair_oracle(c[0], space, statistic=statistic)
    np.testing.assert_allclose(result.values[0], expected, atol=2e-13, rtol=2e-12)
    assert result.exact and np.all(result.finite)


def test_native_aliased_quadratic_product_is_detected() -> None:
    space = _space(2)
    c = _coefficients(space, members=1)
    exact = _pair_oracle(c[0], space, statistic="longitudinal_cube", refinement=4)
    native = _pair_oracle(c[0], space, statistic="longitudinal_cube", refinement=1)
    # Deliberate fault: native-grid products alias quadratic modes back into
    # the retained band. Correlating this Q with c reproduces native cubic
    # quadrature, but disagrees with the independent continuum pair average.
    u = _physical(c[0], space, 1)
    q = u[..., :, None] * u[..., None, :]
    volume = (2 * np.pi) ** 2
    q_modal = np.fft.fftn(q, axes=(0, 1)) * np.sqrt(volume) / 36
    cross = q_modal[..., :, :, None] * np.conj(c[0])[..., None, None, :]
    correlation = np.fft.ifftn(cross, axes=(0, 1)).real * 36 / volume
    h = np.asarray([0, 1, 2, 3, -2, -1], dtype=np.float64) * (2 * np.pi / 6)
    vectors = np.stack(np.meshgrid(h, h, indexing="ij"), axis=-1)
    norm = np.sqrt(np.sum(vectors**2, axis=-1, keepdims=True))
    directions = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm != 0)
    negative = (-np.arange(6)) % 6
    reversed_correlation = correlation[negative[:, None], negative[None, :]]
    aliased = 3 * np.einsum(
        "...i,...j,...k,...ijk->...",
        directions,
        directions,
        directions,
        reversed_correlation - correlation,
    )
    np.testing.assert_allclose(aliased, native, atol=2e-13)
    assert np.max(np.abs(exact - aliased)) > 1e-3
    result = PeriodicThirdIncrementMomentPlan(
        space,
        statistic="longitudinal_cube",
        dealiasing=PaddingDealiasingPlan(2),
        member_chunk_size=1,
        maximum_workspace_bytes=2**24,
    )(c)
    np.testing.assert_allclose(result.values[0], exact, atol=2e-13)


@pytest.mark.parametrize("dimension", [1, 2, 3], ids=["line", "plane", "space"])
@pytest.mark.parametrize("averaging", ["sphere", "ball"], ids=["sphere", "ball"])
def test_radial_single_mode_closed_form(
    dimension: int, averaging: IncrementAverage
) -> None:
    space = _space(dimension)
    radii = np.asarray(
        [0.0, 1e-7, 9.99e-4, 1.001e-3, 9.99e-3, 1.001e-2, 0.0999, 0.1001, 0.2, 1.0]
    )
    c = np.zeros((1,) + space.modal_shape, dtype=np.complex128)
    volume = (2 * np.pi) ** dimension
    c[(0, 1) + (0,) * (dimension - 1)] = np.sqrt(volume) / 2
    c[(0, 5) + (0,) * (dimension - 1)] = np.sqrt(volume) / 2
    plan = PeriodicRadialIncrementMomentPlan(
        space, radii, averaging=averaging, component_shape=(), member_chunk_size=1
    )
    result = eqx.filter_jit(plan)(c)
    x = radii
    if averaging == "sphere":
        kernel = (
            np.cos(x)
            if dimension == 1
            else special.j0(x)
            if dimension == 2
            else np.sinc(x / np.pi)
        )
    else:
        # Independent power series avoids cancellation in the near-zero oracle.
        if dimension == 1:
            kernel = np.sinc(x / np.pi)
        elif dimension == 2:
            kernel = np.asarray(
                [
                    sum(
                        (-1) ** k
                        * (r / 2) ** (2 * k)
                        / (special.factorial(k) * special.factorial(k + 1))
                        for k in range(14)
                    )
                    for r in x
                ]
            )
        else:
            kernel = np.asarray(
                [
                    sum(
                        3
                        * (-1) ** k
                        * r ** (2 * k)
                        / (special.factorial(2 * k + 1) * (2 * k + 3))
                        for k in range(14)
                    )
                    for r in x
                ]
            )
    np.testing.assert_allclose(result.values[0], 1 - kernel, atol=2e-12, rtol=1e-9)
    np.testing.assert_array_equal(result.radii, radii)
    assert result.exact and np.all(result.finite)


@pytest.mark.parametrize("dimension", [1, 2, 3], ids=["line", "plane", "space"])
@pytest.mark.parametrize("averaging", ["sphere", "ball"], ids=["sphere", "ball"])
def test_radial_small_radius_retains_quadratic_limit(
    dimension: int, averaging: IncrementAverage
) -> None:
    space = _space(dimension)
    radii = np.asarray([1e-10, 1e-8], dtype=np.float64)
    c = np.zeros((1,) + space.modal_shape, dtype=np.complex128)
    amplitude = np.sqrt((2 * np.pi) ** dimension) / 2
    c[(0, 1) + (0,) * (dimension - 1)] = amplitude
    c[(0, 5) + (0,) * (dimension - 1)] = amplitude
    plan = PeriodicRadialIncrementMomentPlan(
        space, radii, averaging=averaging, component_shape=(), member_chunk_size=1
    )
    result = plan(c)
    denominator = 2 * dimension if averaging == "sphere" else 2 * (dimension + 2)
    np.testing.assert_allclose(
        np.asarray(result.values[0]) / radii**2, 1 / denominator, rtol=2e-14
    )
    assert plan.radii == tuple(radii)


@pytest.mark.parametrize("dimension", [2, 3], ids=["plane", "space"])
@pytest.mark.parametrize("averaging", ["sphere", "ball"], ids=["sphere", "ball"])
def test_radial_random_field_matches_ungrouped_modal_sum(
    dimension: int, averaging: IncrementAverage
) -> None:
    space = _space(dimension)
    c = _coefficients(space, members=2)
    forbidden_index = (slice(None), 3) + (0,) * (dimension - 1) + (0,)
    c[forbidden_index] = np.asarray([2, 3], dtype=np.float64)
    radii = np.asarray([0.0, 0.15, 0.45, 0.9], dtype=np.float64)
    plan = PeriodicRadialIncrementMomentPlan(
        space,
        radii,
        averaging=averaging,
        component_shape=(dimension,),
        member_chunk_size=1,
    )
    result = eqx.filter_jit(plan)(c)
    # Independent ungrouped physical-wavenumber sum, retaining every mode's
    # separate power. L=2π makes physical wavenumbers equal integer labels.
    modes = np.fft.fftfreq(6) * 6
    wave = np.stack(np.meshgrid(*(modes,) * dimension, indexing="ij"), axis=-1)
    retained = np.all(np.abs(wave) != 3, axis=-1)
    power = np.sum(np.abs(c) ** 2, axis=-1) * retained[None, ...]
    magnitude = np.sqrt(np.sum(wave**2, axis=-1)).reshape((-1,))
    x = radii[:, None] * magnitude[None, :]
    if averaging == "sphere":
        kernel = special.j0(x) if dimension == 2 else np.sinc(x / np.pi)
    elif dimension == 2:
        kernel = np.divide(2 * special.j1(x), x, out=np.ones_like(x), where=x != 0)
    else:
        kernel = np.divide(
            3 * (np.sin(x) - x * np.cos(x)),
            x**3,
            out=np.ones_like(x),
            where=x != 0,
        )
    expected = (2 / (2 * np.pi) ** dimension) * power.reshape((2, -1)) @ (1 - kernel).T
    np.testing.assert_allclose(result.values, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(
        result.forbidden_coefficient_energy, np.asarray([4, 9], dtype=np.float64)
    )
    assert result.exact and np.all(result.finite)


@pytest.mark.parametrize("route", ["radial", "estimator"], ids=["radial", "estimator"])
def test_route_workspace_cap_counts_components_once(route: str) -> None:
    space = _space(3)
    coefficients = _coefficients(space, members=1)

    def prepare(
        cap: int,
    ) -> PeriodicRadialIncrementMomentPlan | PeriodicIncrementMomentEstimatorPlan:
        if route == "radial":
            return PeriodicRadialIncrementMomentPlan(
                space,
                np.asarray([0.1, 0.3], dtype=np.float64),
                averaging="sphere",
                component_shape=(3,),
                member_chunk_size=1,
                maximum_workspace_bytes=cap,
            )
        return PeriodicIncrementMomentEstimatorPlan(
            space,
            np.asarray([[1, 0, 0]], dtype=np.int32),
            order=2,
            projection="magnitude",
            member_chunk_size=1,
            maximum_displacements=1,
            maximum_workspace_bytes=cap,
        )

    bounded = prepare(64_000)
    larger = prepare(96_000)
    result = eqx.filter_jit(bounded)(coefficients)
    reference = eqx.filter_jit(larger)(coefficients)
    assert bounded.plan_id != larger.plan_id
    assert np.all(result.finite)
    np.testing.assert_array_equal(result.values, reference.values)
    with pytest.raises(MemoryBudgetError, match="maximum_workspace_bytes"):
        eqx.filter_jit(prepare(50_000))(coefficients)


@pytest.mark.parametrize(
    "projection,order",
    [("magnitude", 2.7), ("longitudinal", 3), ("absolute_longitudinal", 1.5)],
    ids=["magnitude", "signed", "absolute"],
)
def test_estimator_matches_native_pair_quadrature(
    projection: IncrementProjection, order: float
) -> None:
    space = _space(2, lengths=(3.1, 4.7))
    c = _coefficients(space, members=1)
    shifts = np.asarray([[1, 0], [-1, 2], [3, 0]], dtype=np.int32)
    result = eqx.filter_jit(
        PeriodicIncrementMomentEstimatorPlan(
            space,
            shifts,
            order=order,
            projection=projection,
            member_chunk_size=2,
            maximum_displacements=8,
        )
    )(c)
    u = _physical(c[0], space, 1)
    expected = []
    for shift in shifts:
        delta = np.roll(u, tuple(-int(i) for i in shift), axis=(0, 1)) - u
        h = shift * np.asarray([3.1, 4.7]) / 6
        longitudinal = np.sum(delta * h / np.linalg.norm(h), axis=-1)
        projected = (
            np.linalg.norm(delta, axis=-1)
            if projection == "magnitude"
            else longitudinal
            if projection == "longitudinal"
            else np.abs(longitudinal)
        )
        expected.append(np.mean(projected**order))
    np.testing.assert_allclose(
        result.values[0], np.asarray(expected, dtype=np.float64), atol=2e-13
    )
    assert not result.exact and np.all(result.finite)


@pytest.mark.parametrize(
    "kind",
    ["lattice", "radial", "third", "estimator"],
    ids=["lattice", "radial", "third", "estimator"],
)
def test_chunk_invariance_and_jit_replay(kind: str) -> None:
    space = _space(2, n=4)
    c = _coefficients(space)

    def evaluate(chunk: int) -> Array:
        if kind == "lattice":
            plan = PeriodicLatticeIncrementMomentPlan(
                space, component_shape=(2,), member_chunk_size=chunk
            )
        elif kind == "radial":
            plan = PeriodicRadialIncrementMomentPlan(
                space,
                np.asarray([0.1, 0.5], dtype=np.float64),
                averaging="ball",
                component_shape=(2,),
                member_chunk_size=chunk,
            )
        elif kind == "third":
            plan = PeriodicThirdIncrementMomentPlan(
                space,
                statistic="energy_flux",
                dealiasing=PaddingDealiasingPlan(2),
                member_chunk_size=chunk,
                maximum_workspace_bytes=2**24,
            )
        else:
            plan = PeriodicIncrementMomentEstimatorPlan(
                space,
                np.asarray([[1, 0], [1, 1]], dtype=np.int32),
                order=3,
                projection="longitudinal",
                member_chunk_size=chunk,
                maximum_displacements=4,
            )
        first, second = plan(c), plan(c)
        run = eqx.filter_jit(lambda prepared, inputs: prepared(inputs))
        dynamic = run(plan, c)
        np.testing.assert_array_equal(first.values, dynamic.values)
        np.testing.assert_array_equal(first.values, second.values)
        assert np.all(first.finite)
        return first.values

    np.testing.assert_allclose(evaluate(1), evaluate(2), atol=2e-13, rtol=2e-12)


def test_third_radial_bins_and_empty_mask() -> None:
    space = _space(2)
    c = _coefficients(space, members=1)
    edges = np.asarray([0, 0.4, 2, 4, 5, 6], dtype=np.float64)
    kwargs = {"member_chunk_size": 1, "maximum_workspace_bytes": 2**24}
    unbinned = PeriodicThirdIncrementMomentPlan(
        space, statistic="energy_flux", dealiasing=PaddingDealiasingPlan(2), **kwargs
    )(c)
    plan = PeriodicThirdIncrementMomentPlan(
        space,
        statistic="energy_flux",
        dealiasing=PaddingDealiasingPlan(2),
        radial_bin_edges=edges,
        **kwargs,
    )
    result = plan(c)
    h = np.asarray([0, 1, 2, 3, -2, -1]) * (2 * np.pi / 6)
    distance = np.sqrt(h[:, None] ** 2 + h[None, :] ** 2)
    for b in range(edges.size - 1):
        mask = (distance >= edges[b]) & (distance < edges[b + 1])
        count = np.count_nonzero(mask)
        np.testing.assert_equal(np.asarray(result.bin_counts[b]), count)
        assert bool(result.valid_bins[b]) == (count > 0)
        expected = np.mean(np.asarray(unbinned.values[0])[mask]) if count else 0
        np.testing.assert_allclose(result.values[0, b], expected, atol=2e-13)
    np.testing.assert_array_equal(result.radii, 0.5 * (edges[:-1] + edges[1:]))
    np.testing.assert_array_equal(plan.radial_bin_edges, edges)


def test_forbidden_energy_is_reported_without_changing_retained_moment() -> None:
    space = _space(2)
    c = _coefficients(space, members=1)
    plan = PeriodicLatticeIncrementMomentPlan(
        space, component_shape=(2,), member_chunk_size=1
    )
    baseline = plan(c)
    c[0, 3, 0, 0] = 2
    result = plan(c)
    np.testing.assert_array_equal(result.values, baseline.values)
    np.testing.assert_array_equal(
        result.forbidden_coefficient_energy, np.asarray([4], dtype=np.float64)
    )


def test_finite_evidence_is_member_local() -> None:
    space = _space(2)
    c = _coefficients(space)
    c[1, 1, 0, 0] = np.nan
    plan = PeriodicLatticeIncrementMomentPlan(
        space, component_shape=(2,), member_chunk_size=2
    )
    result = eqx.filter_jit(plan)(c)
    np.testing.assert_array_equal(
        result.finite, np.asarray([True, False, True], dtype=np.bool_)
    )


@pytest.mark.parametrize(
    "case,error",
    [
        ("workspace", MemoryBudgetError),
        ("output-workspace", MemoryBudgetError),
        ("bin-workspace", MemoryBudgetError),
        ("chunk", ValueError),
        ("bool-chunk", TypeError),
        ("fractional-chunk", TypeError),
        ("bool-workspace", TypeError),
        ("component", ValueError),
        ("bool-component", TypeError),
        ("nonfourier", ValueError),
        ("radial-box", ValueError),
        ("radial-kernel", MemoryBudgetError),
        ("bool-kernel", TypeError),
        ("radii", ValueError),
        ("bool-radii", TypeError),
        ("cubic-padding", ValueError),
        ("edges", ValueError),
        ("bool-edges", TypeError),
        ("order", ValueError),
        ("bool-order", TypeError),
        ("string-order", TypeError),
        ("signed-order", ValueError),
        ("zero-shift", ValueError),
        ("float-shift", TypeError),
        ("too-many-shifts", ValueError),
        ("bool-maximum-displacements", TypeError),
        ("coefficient-shape", ValueError),
        ("nonhermitian", eqx.EquinoxRuntimeError),
        ("scalar-longitudinal", ValueError),
    ],
    ids=[
        "workspace",
        "output-workspace",
        "bin-workspace",
        "chunk",
        "bool-chunk",
        "fractional-chunk",
        "bool-workspace",
        "component",
        "bool-component",
        "nonfourier",
        "radial-box",
        "radial-kernel",
        "bool-kernel",
        "radii",
        "bool-radii",
        "cubic-padding",
        "edges",
        "bool-edges",
        "order",
        "bool-order",
        "string-order",
        "signed-order",
        "zero-shift",
        "float-shift",
        "too-many-shifts",
        "bool-maximum-displacements",
        "coefficient-shape",
        "nonhermitian",
        "scalar-longitudinal",
    ],
)
def test_refusal_matrix(case: str, error: type[Exception]) -> None:
    space = _space(2)
    with pytest.raises(error) as refusal:
        match case:
            case "workspace":
                PeriodicLatticeIncrementMomentPlan(
                    space, component_shape=(2,), maximum_workspace_bytes=1
                )
            case "output-workspace":
                plan = PeriodicLatticeIncrementMomentPlan(
                    space,
                    component_shape=(2,),
                    member_chunk_size=1,
                    maximum_workspace_bytes=8_100,
                )
                plan(_coefficients(space, members=2))
            case "bin-workspace":
                PeriodicThirdIncrementMomentPlan(
                    space,
                    statistic="energy_flux",
                    dealiasing=PaddingDealiasingPlan(2),
                    radial_bin_edges=np.linspace(0, 6, 9),
                    member_chunk_size=1,
                    maximum_workspace_bytes=1_000,
                )
            case "chunk" | "bool-chunk" | "fractional-chunk":
                PeriodicLatticeIncrementMomentPlan(
                    space,
                    component_shape=(2,),
                    member_chunk_size=0
                    if case == "chunk"
                    else True
                    if case == "bool-chunk"
                    else 1.5,  # ty: ignore[invalid-argument-type]
                )
            case "bool-workspace":
                PeriodicLatticeIncrementMomentPlan(
                    space, component_shape=(2,), maximum_workspace_bytes=True
                )
            case "component" | "bool-component":
                PeriodicLatticeIncrementMomentPlan(
                    space, component_shape=(0,) if case == "component" else (True,)
                )
            case "nonfourier":
                other = TensorSpectralPlan(
                    (ChebyshevBasisPlan(6),), axis_names=("x",)
                ).prepare((AxisDomain.interval(-1, 1),))
                PeriodicLatticeIncrementMomentPlan(other, component_shape=())
            case "radial-box":
                PeriodicRadialIncrementMomentPlan(
                    _space(2, lengths=(2, 3)),
                    np.asarray([1], dtype=np.float64),
                    averaging="sphere",
                    component_shape=(),
                )
            case "radial-kernel" | "bool-kernel":
                PeriodicRadialIncrementMomentPlan(
                    space,
                    np.asarray([1], dtype=np.float64),
                    averaging="sphere",
                    component_shape=(),
                    maximum_kernel_bytes=1 if case == "radial-kernel" else True,
                )
            case "radii" | "bool-radii":
                PeriodicRadialIncrementMomentPlan(
                    space,
                    np.asarray([-1], dtype=np.float64)
                    if case == "radii"
                    else np.asarray([True, False]),
                    averaging="sphere",
                    component_shape=(),
                )
            case "cubic-padding":
                PeriodicThirdIncrementMomentPlan(
                    space,
                    statistic="energy_flux",
                    dealiasing=PaddingDealiasingPlan(3),
                    member_chunk_size=1,
                    maximum_workspace_bytes=2**24,
                )
            case "edges" | "bool-edges":
                PeriodicThirdIncrementMomentPlan(
                    space,
                    statistic="energy_flux",
                    dealiasing=PaddingDealiasingPlan(2),
                    radial_bin_edges=np.asarray([1, 0], dtype=np.float64)
                    if case == "edges"
                    else np.asarray([False, True]),
                    member_chunk_size=1,
                    maximum_workspace_bytes=2**24,
                )
            case "order" | "signed-order" | "bool-order" | "string-order":
                PeriodicIncrementMomentEstimatorPlan(
                    space,
                    np.asarray([[1, 0]], dtype=np.int32),
                    order={
                        "order": -1,
                        "signed-order": 1.2,
                        "bool-order": True,
                        "string-order": "2",
                    }[case],  # ty: ignore[invalid-argument-type]
                    projection="longitudinal",
                    member_chunk_size=1,
                    maximum_displacements=1,
                )
            case (
                "zero-shift"
                | "float-shift"
                | "too-many-shifts"
                | "bool-maximum-displacements"
            ):
                shifts = (
                    [[6, 0]]
                    if case == "zero-shift"
                    else [[1.0, 0.0]]
                    if case == "float-shift"
                    else [[1, 0], [0, 1]]
                )
                PeriodicIncrementMomentEstimatorPlan(
                    space,
                    np.asarray(shifts),
                    order=2,
                    projection="magnitude",
                    member_chunk_size=1,
                    maximum_displacements=True
                    if case == "bool-maximum-displacements"
                    else 1,
                )
            case "coefficient-shape":
                PeriodicLatticeIncrementMomentPlan(space, component_shape=(2,))(
                    jnp.zeros((6, 6, 2), dtype=jnp.complex128)
                )
            case "nonhermitian":
                c = _coefficients(space, members=1)
                c[0, 1, 0, 0] += 1j
                eqx.filter_jit(
                    PeriodicLatticeIncrementMomentPlan(space, component_shape=(2,))
                )(c)
            case "scalar-longitudinal":
                PeriodicIncrementMomentEstimatorPlan(
                    space,
                    np.asarray([[1, 0]], dtype=np.int32),
                    order=2,
                    projection="longitudinal",
                    member_chunk_size=1,
                    maximum_displacements=1,
                )(jnp.zeros((1, 6, 6), dtype=jnp.complex128))
            case _:
                raise RuntimeError("Unknown refusal scenario.")
    if isinstance(refusal.value, MemoryBudgetError):
        expected_budget = (
            "maximum_kernel_bytes"
            if case == "radial-kernel"
            else "maximum_workspace_bytes"
        )
        assert refusal.value.budget == expected_budget
        assert refusal.value.required_bytes > refusal.value.limit_bytes
