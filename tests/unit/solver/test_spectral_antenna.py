#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The B5 sampled plane antenna driving Cartesian PSATD.

References are independent of the implementation: the retarded plane wave
``f(t − (z − z_a)/c)`` of the sampled envelope, the launched energy ``A∫E² dt``,
the relativistic Doppler factor ``γ(1 + sβ)`` with the rest spectrum by direct
quadrature, the paraxial Gaussian-beam width and Gouy phase, and NumPy spectral
divergences for the declared sheet charges.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


mx = phx.solver.maxwell
sp = phx.solver.maxwell.spectral
D = phx.discretization


def _bridge(counts: tuple[int, int, int], spacing: tuple[float, float, float]) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            [[0.0, 0.0, 0.0], [n * h for n, h in zip(counts, spacing, strict=True)]]
        )
    )
    return D.StructuredCochainBridge(grid)


def _x_polarized(envelope: np.ndarray, shape: tuple[int, int] = (2, 2)) -> np.ndarray:
    electric = np.zeros((*shape, envelope.shape[-1], 2), dtype=np.complex128)
    electric[..., 0] = envelope
    return electric


def _run(solver: Any, step: float, steps: int) -> tuple[Any, Any]:
    """Vacuum run from zero fields; returns the final state and per-step diagnostics."""
    counts = solver.plan.counts
    source = sp.SpectralMaxwellSource(jnp.zeros((1, *counts, 3)), jnp.zeros((1, *counts)))

    def body(carry: tuple[Any, Array], _: None) -> tuple[tuple[Any, Array], Any]:
        field, time = carry
        result = solver.advance(time, field, source, jnp.asarray(step))
        return (result.field, time + step), result.diagnostics

    (final, _), diagnostics = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=steps)
    )((solver.field_with_charge(jnp.zeros(counts)), jnp.asarray(0.0)))
    return final, diagnostics


# -- one-way plane wave and work ledger -------------------------------------------------

_H = 0.05
_CELLS = 240
_PLANE = 100 * _H
_OMEGA = 2.0 * np.pi  # 20 cells per vacuum wavelength
_TIMES = np.linspace(0.0, 6.0, 601)
_ENVELOPE = np.exp(-(((_TIMES - 2.0) / 0.5) ** 2))
_STOP = 4.6


def _plane_wave_run(
    direction: mx.AntennaEmissionDirection,
) -> tuple[Any, Any, Any, float]:
    bridge = _bridge((2, 2, _CELLS), (_H, _H, _H))
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        _PLANE,
        [-1.0, 1.0],
        [-1.0, 1.0],
        _TIMES,
        _x_polarized(_ENVELOPE),
        carrier_angular_frequency=_OMEGA,
        direction=direction,
    )
    solver = sp.SpectralMaxwellPlan(
        bridge, grid="staggered", antennas=(antenna,)
    ).prepare()
    step = 0.9 * float(solver.stable_step)
    steps = int(_STOP / step)
    final, diagnostics = _run(solver, step, steps)
    return solver, final, diagnostics, steps * step


@pytest.fixture(scope="module", params=["positive", "negative"])
def plane_wave_run(request: Any) -> tuple[str, Any, Any, Any, float]:
    return (request.param, *_plane_wave_run(request.param))


def test_plane_antenna_launches_the_retarded_wave_one_way(plane_wave_run: Any) -> None:
    direction, solver, final, diagnostics, stop = plane_wave_run
    sign = 1.0 if direction == "positive" else -1.0
    z = np.arange(_CELLS) * _H
    field = np.asarray(final.electric)[0, 0, :, 0]
    ahead = sign * (z - _PLANE) > 0.5
    behind = sign * (z - _PLANE) < -0.5
    # Independent reference: the sampled wave f(τ) = Re[A(τ)e^{−iωτ}] retarded
    # to τ = t − s(z − z_a)/c on the emission side.
    retarded = stop - sign * (z - _PLANE)
    envelope = np.interp(retarded, _TIMES, _ENVELOPE, left=0.0, right=0.0)
    expected = envelope * np.cos(_OMEGA * retarded)

    assert np.sum(field[behind] ** 2) / np.sum(field[ahead] ** 2) < 1e-10
    # The constant-in-step sheet current is second order in ωΔt.
    np.testing.assert_allclose(field[ahead], expected[ahead], rtol=0, atol=3e-3)
    assert float(solver.antennas[0].evidence.cells_per_wavelength) == pytest.approx(20.0)
    # A transversely uniform sheet declares no charge.
    assert float(jnp.max(diagnostics.electric_constraint)) < 1e-12
    assert float(jnp.max(diagnostics.magnetic_constraint)) < 1e-12


def test_antenna_work_equals_injected_field_energy(plane_wave_run: Any) -> None:
    _, solver, final, diagnostics, _ = plane_wave_run
    energy = float(solver.field_energy(final))
    work = float(jnp.sum(diagnostics.antenna_work))
    carrier = _ENVELOPE * np.cos(_OMEGA * _TIMES)
    # ε = μ = c = 1: a launched plane pulse carries A ∫ E(t)² dt.
    analytic = (2 * _H) ** 2 * np.trapezoid(carrier**2, _TIMES)

    # The ledger integrates the exact in-step fields: it closes to roundoff.
    assert work == pytest.approx(energy, rel=1e-12)
    assert energy == pytest.approx(analytic, rel=5e-3)


# -- moving antenna ----------------------------------------------------------------------

_BETA = -0.9
_WAVELENGTH = 2.4e-6


def test_moving_antenna_emits_the_doppler_shifted_wave() -> None:
    # A boosted-frame antenna: the sheet recedes at 0.9c while emitting forward.
    scale = phx.ElectromagneticScaleContract.si()
    light = float(scale.speed_of_light)
    omega = 2.0 * np.pi * light / _WAVELENGTH
    period = 2.0 * np.pi / omega
    gamma = 1.0 / np.sqrt(1.0 - _BETA**2)
    doppler = gamma * (1.0 + _BETA)
    cells, plane = 768, 600
    spacing = _WAVELENGTH / doppler / 24.0
    bridge = _bridge((2, 2, cells), (spacing, spacing, spacing))
    times = np.linspace(0.0, 8.0 * period, 801)
    envelope = np.exp(-(((times - 4.0 * period) / (1.2 * period)) ** 2))
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        plane * spacing,
        [-1.0, 1.0],
        [-1.0, 1.0],
        times,
        _x_polarized(envelope),
        carrier_angular_frequency=omega,
        beta=_BETA,
        scale=scale,
    )
    solver = sp.SpectralMaxwellPlan(
        bridge,
        grid="staggered",
        antennas=(antenna,),
        permittivity=float(scale.vacuum_permittivity),
        permeability=float(scale.vacuum_permeability),
    ).prepare()
    step = 0.9 * float(solver.stable_step)
    steps = int(1.05 * gamma * times[-1] / step) + 1
    final, diagnostics = _run(solver, step, steps)
    z = np.arange(cells) * spacing
    sheet = plane * spacing + _BETA * light * steps * step
    field = np.asarray(final.electric)[0, 0, :, 0]
    ahead, behind = z > sheet + 5 * spacing, z < sheet - 5 * spacing
    # Spatial spectrum of the forward pulse f(t − z/c): c·F(ck) in time.
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(cells, d=spacing)
    spatial = np.abs(np.fft.rfft(np.where(ahead, field, 0.0))) * spacing / light
    fine = np.linspace(times[0], times[-1], 20001)
    signal = np.interp(fine, times, envelope) * np.cos(omega * fine)
    frequencies = np.linspace(0.2, 2.5, 231) * omega
    rest = np.abs(
        [np.trapezoid(signal * np.exp(1j * w * fine), fine) for w in frequencies]
    )
    rest_centroid = np.sum(frequencies * rest**2) / np.sum(rest**2)
    centroid = np.sum(light * wavenumbers * spatial**2) / np.sum(spatial**2)
    evidence = solver.antennas[0].evidence

    assert float(evidence.emitted_carrier_angular_frequency) == pytest.approx(
        doppler * omega
    )
    assert centroid == pytest.approx(doppler * rest_centroid, rel=1e-3)
    # F_sim(ω) = F′(ω/D): the field drops by D while the pulse stretches by 1/D.
    assert np.max(spatial) == pytest.approx(np.max(rest), rel=5e-3)
    assert np.sum(field[behind] ** 2) / np.sum(field[ahead] ** 2) < 1e-12
    assert float(jnp.sum(diagnostics.antenna_work)) == pytest.approx(
        float(solver.field_energy(final)), rel=1e-12
    )


# -- Gaussian beam -----------------------------------------------------------------------


def test_paraxial_gaussian_beam_reaches_its_waist_with_gouy_phase() -> None:
    columns, cells, waist = 180, 256, 24.0
    bridge = _bridge((columns, 2, cells), (1.0, 4.0, 1.0))
    omega = 2.0 * np.pi / 12.0  # PSATD vacuum: k = ω/c exactly
    rayleigh = 0.5 * omega * waist**2
    antenna_plane = 10.0
    focus = antenna_plane + 0.5 * rayleigh
    first = np.linspace(0.0, columns, 361)
    offset = first - 0.5 * columns
    distance = focus - antenna_plane
    width = waist * np.sqrt(1.0 + (distance / rayleigh) ** 2)
    profile = np.sqrt(waist / width) * np.exp(
        -((offset / width) ** 2)
        - 0.5j * omega * distance * offset**2 / (distance**2 + rayleigh**2)
        + 0.5j * np.arctan(distance / rayleigh)
    )
    times = np.linspace(0.0, 80.0, 801)
    electric = np.zeros((first.size, 2, times.size, 2), dtype=np.complex128)
    electric[..., 0] = profile[:, None, None] * np.exp(-(((times - 36.0) / 12.0) ** 2))
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        antenna_plane,
        first,
        [-10.0, 20.0],
        times,
        electric,
        carrier_angular_frequency=omega,
    )
    solver = sp.SpectralMaxwellPlan(bridge, antennas=(antenna,)).prepare()
    planes = np.asarray([16, 34, 52, 70, 88, 106, 124])
    step = 0.9 * float(solver.stable_step)
    steps = int(190.0 / step)
    source = sp.SpectralMaxwellSource(
        jnp.zeros((1, columns, 2, cells, 3)), jnp.zeros((1, columns, 2, cells))
    )

    def body(
        carry: tuple[Any, Array, Array], _: None
    ) -> tuple[tuple[Any, Array, Array], None]:
        field, time, phasor = carry
        field = solver.advance(time, field, source, jnp.asarray(step)).field
        time = time + step
        # Time-integral DFT at ω (positive sign) of E_x on the probe planes.
        phasor = phasor + field.electric[:, 0, planes, 0] * jnp.exp(1j * omega * time)
        return (field, time, phasor), None

    (_, _, phasor), _ = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=steps)
    )(
        (
            solver.field_with_charge(jnp.zeros((columns, 2, cells))),
            jnp.asarray(0.0),
            jnp.zeros((columns, planes.size), dtype=jnp.complex128),
        )
    )
    phasors = np.asarray(phasor).T
    centers = np.arange(columns) - 0.5 * columns
    intensity = np.abs(phasors) ** 2
    widths = 2.0 * np.sqrt(np.sum(centers**2 * intensity, 1) / np.sum(intensity, 1))
    gouy = np.angle(
        phasors[:, columns // 2] * np.exp(-1j * omega * (planes - antenna_plane))
    )
    expected_widths = waist * np.sqrt(1.0 + ((planes - focus) / rayleigh) ** 2)
    expected_gouy = -0.5 * np.arctan((planes - focus) / rayleigh)

    assert planes[np.argmin(widths)] == planes[np.argmin(np.abs(planes - focus))]
    np.testing.assert_allclose(widths, expected_widths, rtol=0.01)
    np.testing.assert_allclose(gouy, expected_gouy, atol=6e-3)


# -- declared sheet charges ----------------------------------------------------------------


def _divergence(field: np.ndarray, spacing: float, shift: float) -> np.ndarray:
    """Independent staggered spectral divergence; ``shift`` ∓½ maps E/B onto nodes/cells."""
    result = np.zeros(field.shape[:3])
    for axis in range(3):
        count = field.shape[axis]
        k = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
        symbol = 1j * k * np.exp(1j * shift * k * spacing)
        shape = [1, 1, 1]
        shape[axis] = count
        result += np.real(
            np.fft.ifft(
                np.fft.fft(field[..., axis], axis=axis) * symbol.reshape(shape), axis=axis
            )
        )
    return result


def test_focused_sheet_declares_its_electric_and_magnetic_charges() -> None:
    # A beam of finite aperture: ∇ₜ·K ≠ 0 and ∇ₜ·K_m ≠ 0 put charges on the sheet.
    counts, h = (24, 24, 48), 0.25
    bridge = _bridge(counts, (h, h, h))
    samples = np.linspace(0.0, 6.0, 49)
    profile = np.exp(-(((samples - 3.0) / 0.8) ** 2))
    times = np.linspace(0.0, 4.0, 201)
    envelope = np.exp(-(((times - 2.0) / 0.6) ** 2))
    electric = np.zeros((49, 49, times.size, 2), dtype=np.complex128)
    electric[..., 0] = profile[:, None, None] * profile[None, :, None] * envelope
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        3.0,
        samples,
        samples,
        times,
        electric,
        carrier_angular_frequency=0.8 * np.pi,
    )
    solver = sp.SpectralMaxwellPlan(
        bridge, grid="staggered", antennas=(antenna,)
    ).prepare()
    step = 0.9 * float(solver.stable_step)
    final, diagnostics = _run(solver, step, int(3.0 / step))
    charge = np.asarray(final.antenna_charge)
    magnetic_charge = np.asarray(final.antenna_magnetic_charge)
    divergence = _divergence(np.asarray(final.electric), h, -0.5)
    magnetic_divergence = _divergence(np.asarray(final.magnetic), h, 0.5)

    assert float(solver.antennas[0].evidence.magnetic_closure_defect) > 1e-3
    assert np.max(np.abs(charge)) > 1e-3
    assert np.max(np.abs(magnetic_charge)) > 1e-3
    # The fields carry exactly the declared charges (ε = 1).
    scale = np.max(np.abs(divergence))
    np.testing.assert_allclose(divergence, charge, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(
        magnetic_divergence, magnetic_charge, rtol=0, atol=1e-12 * scale
    )
    assert float(jnp.max(diagnostics.electric_constraint)) < 1e-12 * scale
    assert float(jnp.max(diagnostics.magnetic_constraint)) < 1e-12 * scale


# -- admissibility -------------------------------------------------------------------------


def _refusal_antenna(bridge: Any, **options: Any) -> Any:
    times = np.linspace(0.0, 4.0, 9)
    coordinate = options.pop("plane_coordinate", 1.0)
    window = options.pop("window", [-1.0, 3.0])
    return mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        coordinate,
        window,
        window,
        times,
        _x_polarized(np.ones(times.size)),
        **options,
    )


def test_multi_j_antenna_executes_its_wide_exact_transform_payload() -> None:
    counts = (8, 8, 16)
    bridge = _bridge(counts, (0.25, 0.25, 0.25))
    solver = sp.SpectralMaxwellPlan(
        bridge,
        time_dependency="multi-j",
        current_substeps=4,
        charge_conservation="update-with-rho",
        antennas=(_refusal_antenna(bridge),),
    ).prepare()
    intervals = solver.plan.current_intervals
    source = sp.SpectralMaxwellSource(
        jnp.zeros((intervals, *counts, 3)),
        jnp.zeros((intervals, *counts)),
    )
    field = solver.field_with_charge(jnp.zeros(counts))
    result = solver.advance(jnp.asarray(0.0), field, source, jnp.asarray(0.05))
    assert bool(result.successful)

    width = 15 + 4 * intervals
    values = jnp.arange(np.prod(counts) * width, dtype=jnp.float64).reshape(
        (*counts, width)
    )
    restored = solver.transform.inverse(solver.transform.forward(values))
    np.testing.assert_allclose(
        np.asarray(restored), np.asarray(values), rtol=2e-12, atol=5e-12
    )
    with pytest.raises(ValueError, match="payload"):
        solver.transform.forward(jnp.zeros((*counts, width - 1)))


_ANTENNA_REFUSALS: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
    "local-guarded": (
        {
            "stencil": "finite-order",
            "stencil_order": 4,
            "charge_conservation": "vay-deposition",
            "grid": "staggered",
            "decomposition": "local-guarded",
            "subdomains": (1, 1, 2),
            "guard_cells": (4, 4, 4),
        },
        {},
        "global-fft",
    ),
    "linear-j": (
        {"time_dependency": "linear-j", "charge_conservation": "update-with-rho"},
        {},
        "constant-j or multi-j",
    ),
    "medium": ({"permittivity": 2.0}, {}, "vacuum medium"),
    "tangential-galilean": (
        {
            "variant": "galilean",
            "galilean_velocity": (0.3, 0.0, 0.0),
            "charge_conservation": "update-with-rho",
        },
        {},
        "along their normals",
    ),
}


@pytest.mark.parametrize(
    ("options", "antenna", "message"),
    list(_ANTENNA_REFUSALS.values()),
    ids=list(_ANTENNA_REFUSALS),
)
def test_spectral_antenna_refuses_inadmissible_plans(
    options: dict[str, Any], antenna: dict[str, Any], message: str
) -> None:
    bridge = _bridge((8, 8, 16), (0.25, 0.25, 0.25))
    with pytest.raises(ValueError, match=message):
        sp.SpectralMaxwellPlan(
            bridge, antennas=(_refusal_antenna(bridge, **antenna),), **options
        )


def test_spectral_antenna_refuses_absorbers_huygens_and_unresolved_carriers() -> None:
    bridge = _bridge((8, 8, 32), (0.25, 0.25, 0.25))
    pml = sp.SpectralPMLPlan((0, 0, 4))
    with pytest.raises(ValueError, match="PML layer along its normal"):
        sp.SpectralMaxwellPlan(
            bridge,
            absorber="psatd-pml",
            pml=pml,
            antennas=(_refusal_antenna(bridge, plane_coordinate=0.5),),
        ).prepare()
    with pytest.raises(ValueError, match="aperture reaches into the PML"):
        sp.SpectralMaxwellPlan(
            bridge,
            absorber="psatd-pml",
            pml=sp.SpectralPMLPlan((2, 0, 4)),
            antennas=(_refusal_antenna(bridge, plane_coordinate=4.0),),
        ).prepare()
    with pytest.raises(ValueError, match="eight cells"):
        sp.SpectralMaxwellPlan(
            bridge,
            antennas=(
                _refusal_antenna(
                    bridge, plane_coordinate=4.0, carrier_angular_frequency=4.0 * np.pi
                ),
            ),
        ).prepare()
    box = sp.SpectralHuygensBoxPlan(
        (2, 2, 8),
        (6, 6, 24),
        mx.MaxwellSpectralAcquisition(
            jnp.asarray([1.0]), sign="positive", measure="time-integral"
        ),
        mx.HomogeneousMaxwellExterior(),
    )
    with pytest.raises(ValueError, match="Huygens"):
        sp.SpectralMaxwellPlan(
            bridge, observers=(box,), antennas=(_refusal_antenna(bridge),)
        )
    other = _bridge((8, 8, 16), (0.25, 0.25, 0.25))
    with pytest.raises(ValueError, match="spectral bridge"):
        sp.SpectralMaxwellPlan(bridge, antennas=(_refusal_antenna(other),))
