#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-domain prescribed charges against the B2b frequency-domain solve.

Periodic channel (Cherenkov, Smith–Purcell). A charge ``q`` moves at
``v = βc`` along the periodic axis ``z`` (length ``L``); ``y`` is periodic with
two cells and the charge sits midway between its two vertex planes, so both
routes carry the exact line charge ``λ = q/L_y`` of the lattice. ``x`` is open
(CPML). The time-domain charge advances ``h/r`` per step, so harmonic ``m`` of
``T = L/v`` is an exact Bloch wave with ``k_z = ω_m/v`` and ``T c_m``
(sample-mean phasor ``c_m`` over whole periods) is the single-charge
transform ``Ẽ(ω_m) = ∫ E e^{iω_m t} dt`` that `FrequencyMovingChargePlan`
solves for on the same bridge, material, absorber, and conductors. Projecting
both on ``e^{i k z}`` keeps the harmonic's Bloch content (each Smith–Purcell
order ``k_z − 2πn/d`` separately) and discards the start transient's other
wavenumbers; nested whole-period windows (a triangular window) suppress the
undamped waves grazing along ``z`` that the ``x`` absorber cannot reach. What
remains between the routes is the leapfrog and step-deposit error
``O((ωΔt)²)`` at fixed ``h``: halving Δt must cut it by ≈ 4.

Slit (diffraction radiation). A line charge crosses a thin PEC screen through
a slit of width ``a`` in an open ``(x, z)`` box. The time-domain path starts and
stops at rest with erf ramps far from the screen; a delayed opposite charge on
the same path cancels every static field. The time-domain diffraction field is
the screened run minus the unscreened run and is compared with B2b's
scattered-field solve of the same planar geometry (analytic incident field of
the infinite path).

References (code units ``ε₀ = μ₀ = c = 1``):

* Smith–Purcell (Smith & Purcell, Phys. Rev. 92, 1069 (1953)): order ``n`` of a
  grating of period ``d`` radiates at ``λ = (d/n)(1/β − cos θ)``, i.e. with
  ``k_z = ω/v − 2πn/d = (ω/c) cos θ``. On the Yee lattice of the executed
  update the transverse wavenumber obeys
  ``sin²(k_x h/2) = (h/cΔt)² sin²(ωΔt/2) − sin²(k_z h/2)`` (``k_y = 0``;
  Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 4).
* Kazantsev–Surdutovich diffraction radiation (Dokl. Akad. Nauk SSSR 147, 74
  (1962); Sov. Phys. Dokl. 7, 990 (1963)),
  in the line-charge form derived here: the charge's transform
  ``H_y = ±(λ/2) e^{−κ|x|} e^{i(ω/v)z}``, ``κ = ω/(βγ)``, is on each side a
  single evanescent plane wave, i.e. an incidence angle with
  ``cos φ₀ = −i/(βγ)``. Sommerfeld's exact PEC half-plane solution for
  ``H`` parallel to the edge has the far field
  ``H_d = H_edge D(φ) e^{ikρ}/√ρ``,
  ``|D|² = √(1 + b²) cos²(φ/2)/(πk (cos²φ + b²))``, ``b = 1/(βγ)``
  (Keller, J. Opt. Soc. Am. 52, 116 (1962)), so one edge at distance ``h``
  radiates ``dW/dω = (1/π)(λ/2)² e^{−2κh} ∫|D|² dφ = λ² βγ e^{−2κh}/(4πω)``.
  A slit is two such edges at ``a/2`` (their coupling vanishes along the
  screen plane for this polarization).
"""

import functools
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_PERIOD_LENGTH = 1.0
_CHARGE = 1.0
# Periods simulated, and the whole-period length of each nested sample-mean
# window; the windows start one period apart and end with the run.
_PERIODS = 48
_WINDOW = 16


@functools.cache
def _channel_bridge(count: int, cells: int) -> Any:
    spacing = _PERIOD_LENGTH / cells
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(count, periodic=False),
            D.UniformCellAxisSpec(2, periodic=True),
            D.UniformCellAxisSpec(cells, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray([[0.0, 0.0, 0.0], [count * spacing, 2.0 * spacing, _PERIOD_LENGTH]])
    )
    return D.StructuredCochainBridge(grid)


@dataclass(frozen=True)
class _Channel:
    """Periodic line-charge channel: ``count`` x cells, ``cells`` per period ``L``."""

    count: int
    cells: int
    absorber: int
    column: int
    speed: float
    constitutive: Any
    conductor: np.ndarray | None = None

    @property
    def spacing(self) -> float:
        return _PERIOD_LENGTH / self.cells

    @property
    def period(self) -> float:
        return _PERIOD_LENGTH / self.speed

    @property
    def bridge(self) -> Any:
        return _channel_bridge(self.count, self.cells)

    @property
    def options(self) -> dict[str, Any]:
        boundaries = (
            ()
            if self.conductor is None
            else (mx.MaxwellBoundaryPlan("pec", support=self.conductor),)
        )
        return {
            "constitutive": self.constitutive,
            "pml": mx.MaxwellCPMLPlan((self.absorber, 0, 0)),
            "boundaries": boundaries,
        }

    def frequencies(self, harmonics: tuple[int, ...]) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(harmonics, dtype=np.float64) / self.period

    def electric_rows(self) -> np.ndarray:
        """``E_z`` edges on vertex columns, then ``E_x`` edges, on the ``y = 0`` layer."""
        offsets = self.bridge.orientation_offsets[1]
        stride = 2 * self.cells
        along = np.arange(self.cells)
        axial = [offsets[2] + ix * stride + along for ix in range(self.count + 1)]
        transverse = [offsets[0] + ix * stride + along for ix in range(self.count)]
        return np.concatenate(axial + transverse)

    def magnetic_rows(self) -> np.ndarray:
        """``(x, z)`` faces at ``(x_{i+1/2}, z_{j+1/2})`` carrying ``−B_y h²``."""
        offset = self.bridge.orientation_offsets[2][1]
        stride = 2 * self.cells
        along = np.arange(self.cells)
        return np.concatenate([offset + ix * stride + along for ix in range(self.count)])

    def phasors(self, electric: np.ndarray, magnetic: np.ndarray) -> "_Phasors":
        """Split probe rows into ``E_z``, ``E_x``, ``H_y`` fields (``M, x, z``)."""
        split = (self.count + 1) * self.cells
        shape = (electric.shape[0], -1, self.cells)
        return _Phasors(
            electric[:, :split].reshape(shape) / self.spacing,
            electric[:, split:].reshape(shape) / self.spacing,
            -magnetic.reshape(shape) / self.spacing**2,
            self.spacing,
        )


@dataclass(frozen=True)
class _Phasors:
    """Single-charge transforms on the ``y = 0`` layer, one row per frequency."""

    axial: np.ndarray
    transverse: np.ndarray
    magnetic: np.ndarray
    spacing: float

    def bloch(self, wavenumbers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Amplitudes of ``e^{i k z}`` (one ``k`` per frequency) on every column."""
        cells = self.axial.shape[-1]
        centers = (np.arange(cells) + 0.5) * self.spacing
        nodes = np.arange(cells) * self.spacing
        k = np.asarray(wavenumbers)[:, None, None]
        centered = np.exp(-1j * k * centers) / cells
        noded = np.exp(-1j * k * nodes) / cells
        return (
            np.sum(self.axial * centered, axis=-1),
            np.sum(self.transverse * noded, axis=-1),
            np.sum(self.magnetic * centered, axis=-1),
        )

    def poynting(self, wavenumbers: np.ndarray, column: int) -> np.ndarray:
        """One-sided ``dW/(dω dA) = −Re(E_z H_y^*)/π`` of the Bloch wave at a column."""
        axial, _, magnetic = self.bloch(wavenumbers)
        average = 0.5 * (magnetic[:, column - 1] + magnetic[:, column])
        return -np.real(axial[:, column] * np.conj(average)) / np.pi


def _time_domain(
    channel: _Channel, subdivisions: int, harmonics: tuple[int, ...]
) -> tuple[_Phasors, Any]:
    """Drive the channel with ``subdivisions`` steps per cell; ``T c_m`` phasors."""
    spacing, speed, period = channel.spacing, channel.speed, channel.period
    step = spacing / (subdivisions * speed)
    stable = phx.solver.CompatibleMaxwellPlan(channel.bridge, **channel.options).prepare()
    assert step <= float(stable.stable_dt)
    steps = _PERIODS * subdivisions * channel.cells
    times = step * np.arange(steps + 1, dtype=np.float64)
    positions = np.zeros((steps + 1, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = channel.column * spacing
    positions[:, 0, 1] = 0.5 * spacing
    # Unwrapped z: the deposit wraps the chord across the periodic seam.
    positions[:, 0, 2] = speed * times
    particles = D.ParticleSetPlan(
        jnp.arange(1), jnp.ones((1,), dtype=jnp.float64), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((1,), dtype=jnp.float64), "channel").prepare(
        particles
    )
    current = PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(channel.bridge).prepare(charged)
    )
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    # _WINDOW sample means over whole periods, shifted by one period: each is
    # the harmonic's Fourier coefficient in steady state, and their mean is a
    # triangular window whose leakage from undamped non-harmonic modes (waves
    # grazing along z that the x absorber cannot reach) falls as 1/Δω².
    observers = []
    for shift in range(_WINDOW):
        start = (_PERIODS - 2 * _WINDOW + 1 + shift) * period
        acquisition = mx.MaxwellSpectralAcquisition(
            channel.frequencies(harmonics),
            sign="positive",
            measure="sample-mean",
            start_time=start + 0.5 * step,
            stop_time=start + _WINDOW * period + 0.5 * step,
        )
        observers.extend(
            mx.DFTObserverPlan(mx.FieldProbePlan(kind, rows), acquisition)
            for kind, rows in (
                ("electric", channel.electric_rows()),
                ("magnetic", channel.magnetic_rows()),
            )
        )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        channel.bridge,
        sources=(mx.PrescribedChargeCurrentSourcePlan(trajectory, current),),
        observers=tuple(observers),
        **channel.options,
    ).prepare()
    result = mx.solve_prescribed_charge_maxwell(
        mx.PrescribedChargeMaxwellPlan(
            maxwell, current, trajectory, np.asarray([_CHARGE], dtype=np.float64)
        )
    )
    values = [np.asarray(value) * period for value in result.observations]
    electric = np.mean(values[0::2], axis=0)
    magnetic = np.mean(values[1::2], axis=0)
    return channel.phasors(electric, magnetic), result.evidence


def _frequency_domain(
    channel: _Channel, harmonics: tuple[int, ...]
) -> tuple[_Phasors, list[Any]]:
    """B2b total-field solves of the same channel at the harmonic frequencies."""
    bridge = channel.bridge
    layout = mx.MaxwellCochainLayout(bridge, "full_3d")
    options = channel.options
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=_CHARGE,
        speed=channel.speed,
        origin=[channel.column * channel.spacing, 0.5 * channel.spacing, 0.0],
        direction=[0.0, 0.0, 1.0],
    )
    prepared = mx.FrequencyMovingChargePlan(
        source,
        options["constitutive"].prepare(bridge.cochain, layout),
        stretching=options["pml"],
        boundaries=options["boundaries"],
    ).prepare()
    results = [prepared.solve(omega) for omega in channel.frequencies(harmonics)]
    electric = np.stack([np.asarray(value.electric) for value in results])
    magnetic = np.stack([np.asarray(value.magnetic_flux) for value in results])
    return (
        channel.phasors(
            electric[:, channel.electric_rows()], magnetic[:, channel.magnetic_rows()]
        ),
        [value.evidence for value in results],
    )


def _bloch_error(
    first: _Phasors, second: _Phasors, wavenumbers: np.ndarray, columns: slice
) -> np.ndarray:
    """Per-frequency ``‖first − second‖/‖second‖`` of the Bloch ``E_z``, ``E_x`` profiles."""
    difference = np.zeros(wavenumbers.shape)
    reference = np.zeros(wavenumbers.shape)
    for mine, other in zip(
        first.bloch(wavenumbers)[:2], second.bloch(wavenumbers)[:2], strict=True
    ):
        difference += np.sum(np.abs(mine - other)[:, columns] ** 2, axis=1)
        reference += np.sum(np.abs(other)[:, columns] ** 2, axis=1)
    return np.sqrt(difference / reference)


@dataclass(frozen=True)
class _Comparison:
    """One channel solved by B2b and driven in time at two step sizes ``Δt, Δt/2``."""

    channel: _Channel
    harmonics: tuple[int, ...]
    frequency: _Phasors
    frequency_evidence: list[Any]
    coarse: _Phasors
    coarse_evidence: Any
    fine: _Phasors
    fine_evidence: Any


def _compare(
    channel: _Channel, harmonics: tuple[int, ...], subdivisions: int
) -> _Comparison:
    frequency, frequency_evidence = _frequency_domain(channel, harmonics)
    coarse, coarse_evidence = _time_domain(channel, subdivisions, harmonics)
    fine, fine_evidence = _time_domain(channel, 2 * subdivisions, harmonics)
    return _Comparison(
        channel,
        harmonics,
        frequency,
        frequency_evidence,
        coarse,
        coarse_evidence,
        fine,
        fine_evidence,
    )


def _runtime_constraints_hold(comparison: _Comparison) -> None:
    for evidence in comparison.frequency_evidence:
        assert bool(evidence.converged)
        assert float(evidence.continuity_defect) < 1e-12
        assert evidence.periodic and evidence.open_endpoints == 0
    # At Δt = h/(rβ) the CPML ledger of the coarse run keeps its O(Δt²)
    # residual above the 1e-2 default; every other bit is clear, and the fine
    # run closes the ledger too.
    ledger = int(mx.PrescribedChargeStatus.LEDGER_OPEN)
    assert int(comparison.coarse_evidence.status) & ~ledger == 0
    assert int(comparison.fine_evidence.status) == 0


# --- Cherenkov: homogeneous dielectric ---------------------------------------

_CHERENKOV_BETA = 0.9
_CHERENKOV_CELLS = 16
_CHERENKOV_ABSORBER = 20
_CHERENKOV_HALF = 32
_CHERENKOV_PLANE = 8


@pytest.fixture(scope="module")
def cherenkov() -> _Comparison:
    channel = _Channel(
        count=2 * _CHERENKOV_HALF,
        cells=_CHERENKOV_CELLS,
        absorber=_CHERENKOV_ABSORBER,
        column=_CHERENKOV_HALF,
        speed=_CHERENKOV_BETA,
        constitutive=mx.DiagonalMaxwellConstitutivePlan(permittivity=2.25),
    )
    return _compare(channel, (1,), 2)


def test_cherenkov_time_domain_equals_frequency_domain(cherenkov: _Comparison) -> None:
    _runtime_constraints_hold(cherenkov)
    channel = cherenkov.channel
    assert bool(cherenkov.frequency_evidence[0].radiating)
    wavenumbers = channel.frequencies(cherenkov.harmonics) / channel.speed
    interior = slice(channel.absorber, channel.count - channel.absorber)
    coarse, fine = (
        _bloch_error(run, cherenkov.frequency, wavenumbers, interior)[0]
        for run in (cherenkov.coarse, cherenkov.fine)
    )
    # Measured 9.2e-3 → 2.3e-3: second order in Δt at fixed h.
    assert fine < 3e-3
    assert 3.3 < coarse / fine < 4.8
    # Spectral Poynting dW/(dω dA) through the planes x = ±8 h, both routes.
    for plane in (channel.column + _CHERENKOV_PLANE, channel.column - _CHERENKOV_PLANE):
        reference = cherenkov.frequency.poynting(wavenumbers, plane)[0]
        assert np.sign(plane - channel.column) * reference > 0.0
        errors = [
            abs(run.poynting(wavenumbers, plane)[0] / reference - 1.0)
            for run in (cherenkov.coarse, cherenkov.fine)
        ]
        # Measured 3.5e-3 → 9.1e-4.
        assert errors[1] < 2e-3
        assert 3.0 < errors[0] / errors[1] < 5.0


# --- Smith–Purcell: grating of perfectly conducting bars in vacuum -------------

_GRATING_BETA = 0.8
# Grating period d = 16 h, two periods per channel (L = 2d, 32 cells); bars
# d/2 wide and d/4 tall on node rows 18–22, the charge d/4 above them.
_GRATING_PERIOD = 16
_GRATING_ABSORBER = 12
_GRATING_BARS = (18, 22)
_GRATING_COLUMN = 26
_GRATING_COUNT = 60
_GRATING_PLANE = 16
_GRATING_SUBDIVISIONS = 3
# (harmonic m, order n): radiating orders of β = 0.8, L = 2d, with
# k_z = 2π(m − 2n)/L: (2, 1) normal to the grating, (3, 1) forward, (3, 2)
# backward. Order 0 of both harmonics is bound (βn < 1).
_RADIATING = ((2, 1), (3, 1), (3, 2))
_BOUND = ((2, 0), (3, 0))


def _bar_grating(count: int, cells: int) -> np.ndarray:
    """PEC support of the closed bars ``x ∈ [18h, 22h]``, ``z mod d ∈ [0, d/2]``."""
    low, high = _GRATING_BARS

    def inside(column: np.ndarray, row: np.ndarray) -> np.ndarray:
        bar = (row % _GRATING_PERIOD) <= _GRATING_PERIOD // 2
        return (column >= low) & (column <= high) & bar

    masks = []
    for axis, shape in enumerate(_channel_bridge(count, cells).orientation_shapes[1]):
        column, _, row = np.indices(shape).reshape(3, -1)
        match axis:
            case 0:
                masks.append(inside(column, row) & inside(column + 1, row))
            case 1:
                masks.append(inside(column, row))
            case _:
                masks.append(inside(column, row) & inside(column, (row + 1) % cells))
    return np.concatenate(masks)


@pytest.fixture(scope="module")
def smith_purcell() -> _Comparison:
    cells = 2 * _GRATING_PERIOD
    channel = _Channel(
        count=_GRATING_COUNT,
        cells=cells,
        absorber=_GRATING_ABSORBER,
        column=_GRATING_COLUMN,
        speed=_GRATING_BETA,
        constitutive=mx.DiagonalMaxwellConstitutivePlan(),
        conductor=_bar_grating(_GRATING_COUNT, cells),
    )
    return _compare(channel, (2, 3), _GRATING_SUBDIVISIONS)


def _order(comparison: _Comparison, harmonic: int, order: int) -> tuple[int, float]:
    """Row of harmonic ``m`` and the wavenumber ``ω_m/v − 2πn/d`` of order ``n``."""
    channel = comparison.channel
    row = comparison.harmonics.index(harmonic)
    grating = 2.0 * np.pi / (_GRATING_PERIOD * channel.spacing)
    omega = channel.frequencies(comparison.harmonics)[row]
    return row, float(omega / channel.speed - order * grating)


def _order_wavenumbers(comparison: _Comparison, order: int) -> np.ndarray:
    return np.asarray(
        [_order(comparison, harmonic, order)[1] for harmonic in comparison.harmonics]
    )


def test_smith_purcell_time_domain_equals_frequency_domain(
    smith_purcell: _Comparison,
) -> None:
    _runtime_constraints_hold(smith_purcell)
    channel = smith_purcell.channel
    above = slice(channel.column + 2, channel.count - channel.absorber)
    plane = channel.column + _GRATING_PLANE
    for harmonic, order in _RADIATING:
        row, _ = _order(smith_purcell, harmonic, order)
        wavenumbers = _order_wavenumbers(smith_purcell, order)
        coarse, fine = (
            _bloch_error(run, smith_purcell.frequency, wavenumbers, above)[row]
            for run in (smith_purcell.coarse, smith_purcell.fine)
        )
        # Measured 3.7e-3 → 9.2e-4, 6.9e-3 → 1.7e-3, 2.6e-2 → 6.4e-3.
        assert fine < 1e-2
        assert 3.3 < coarse / fine < 4.8
        reference = smith_purcell.frequency.poynting(wavenumbers, plane)[row]
        assert reference > 0.0
        errors = [
            abs(run.poynting(wavenumbers, plane)[row] / reference - 1.0)
            for run in (smith_purcell.coarse, smith_purcell.fine)
        ]
        # Measured 7.7e-3 → 1.9e-3, 1.1e-2 → 2.8e-3, 1.1e-2 → 2.6e-3.
        assert errors[1] < 5e-3
        assert 3.0 < errors[0] / errors[1] < 5.0


def test_smith_purcell_wavelength_relation_in_time_domain(
    smith_purcell: _Comparison,
) -> None:
    channel = smith_purcell.channel
    spacing, run = channel.spacing, smith_purcell.fine
    step = spacing / (2 * _GRATING_SUBDIVISIONS * channel.speed)
    period = _GRATING_PERIOD * spacing
    near, far = channel.column + 4, channel.column + _GRATING_PLANE
    fluxes = {}
    for harmonic, order in _RADIATING + _BOUND:
        row, axial_wavenumber = _order(smith_purcell, harmonic, order)
        omega = channel.frequencies(smith_purcell.harmonics)[row]
        wavenumbers = _order_wavenumbers(smith_purcell, order)
        axial = run.bloch(wavenumbers)[0][row, near : far + 1]
        fluxes[harmonic, order] = run.poynting(wavenumbers, far)[row]
        if (harmonic, order) in _BOUND:
            continue
        # Outgoing plane wave above the charge: constant amplitude, and k_x from
        # the unwrapped phase across the 12 columns (k_x h < π per column).
        amplitude = np.abs(axial)
        assert np.ptp(amplitude) < 1e-2 * np.max(amplitude)
        phase = np.unwrap(np.angle(axial))
        transverse = (phase[-1] - phase[0]) / ((far - near) * spacing)
        lattice = (
            2.0
            / spacing
            * np.arcsin(
                np.sqrt(
                    (spacing / step) ** 2 * np.sin(0.5 * omega * step) ** 2
                    - np.sin(0.5 * axial_wavenumber * spacing) ** 2
                )
            )
        )
        cosine = axial_wavenumber / np.hypot(transverse, axial_wavenumber)
        wavelength = 2.0 * np.pi / omega
        relation = period / order * (1.0 / channel.speed - cosine)
        # The executed Yee branch holds to ≈ 1e-5; its dispersion at
        # ω h ≤ 0.47 moves k_x by 0.8 % from the continuum, which shifts the
        # Smith–Purcell wavelength by ≤ 0.4 % (measured 3.1e-3 and −1.6e-3).
        np.testing.assert_allclose(transverse, lattice, rtol=1e-4)
        np.testing.assert_allclose(relation, wavelength, rtol=5e-3)
    radiated = sum(fluxes[key] for key in _RADIATING)
    assert radiated > 0.0
    for key in _BOUND:
        assert abs(fluxes[key]) < 1e-4 * radiated


# --- Diffraction radiation: line charge through a slit in a thin PEC screen ---

_SLIT_BETA = 0.7
_SLIT_OMEGA = 2.0 * np.pi
_SLIT_SPACING = 1.0 / 16.0
# Box [−52h, 52h] × [−100h, 100h] in (x, z) with a 12-cell CPML on every face.
_SLIT_HALF_WIDTH = 52
_SLIT_HALF_LENGTH = 100
_SLIT_ABSORBER = 12
# Half-openings a/2 of the two slits, in cells.
_SLIT_OPENINGS = (4, 6)
_SLIT_RAMP = 3.0
_SLIT_TRAVEL = 5.2


@functools.cache
def _slit_bridge(planar: bool) -> Any:
    """Open ``x``, ``z`` box; the 3-D bridge adds two periodic ``y`` cells."""
    width = _SLIT_HALF_WIDTH * _SLIT_SPACING
    length = _SLIT_HALF_LENGTH * _SLIT_SPACING
    x = D.UniformCellAxisSpec(2 * _SLIT_HALF_WIDTH)
    z = D.UniformCellAxisSpec(2 * _SLIT_HALF_LENGTH)
    if planar:
        specs: tuple[Any, ...] = (x, z)
        names: tuple[str, ...] = ("x", "y")
        box = [[-width, -length], [width, length]]
    else:
        specs = (x, D.UniformCellAxisSpec(2, periodic=True), z)
        names = ("x", "y", "z")
        box = [[-width, 0.0, -length], [width, 2.0 * _SLIT_SPACING, length]]
    grid = D.TensorGridPlan(specs, axis_names=names).prepare(jnp.asarray(box))
    return D.StructuredCochainBridge(grid)


def _slit_screen(planar: bool, opening: int) -> Any:
    """PEC screen on the node plane ``z = 0`` with ``|x| ≥ opening·h`` (closed)."""
    bridge = _slit_bridge(planar)
    masks = []
    for axis, shape in enumerate(bridge.orientation_shapes[1]):
        nodes = np.indices(shape).reshape(len(shape), -1)
        across = nodes[-1] == _SLIT_HALF_LENGTH
        column = nodes[0] - _SLIT_HALF_WIDTH
        match axis:
            case 0:
                inside = (np.abs(column) >= opening) & (np.abs(column + 1) >= opening)
            case 1 if not planar:
                inside = np.abs(column) >= opening
            case _:
                inside = np.zeros_like(across)
        masks.append(across & inside)
    return mx.MaxwellBoundaryPlan("pec", support=np.concatenate(masks))


def _slit_pml(planar: bool) -> Any:
    widths = (
        (_SLIT_ABSORBER, _SLIT_ABSORBER)
        if planar
        else (_SLIT_ABSORBER, 0, _SLIT_ABSORBER)
    )
    return mx.MaxwellCPMLPlan(widths, alpha_max=1.0)


def _slit_heights(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Rest, erf ramp to ``β``, cruise through ``z = 0``, erf ramp to rest.

    Returns the heights and the cruise origin ``z₀`` (``z = z₀ + v t`` while
    cruising): ``v_z = β[Φ((t − t₁)/σ) − Φ((t − t₂)/σ)]``, ``t₁ = 5σ``. The
    ramp radiation reaching the screen is Doppler-compressed, so its spectrum
    at ``ω`` is ``~exp(−(ωσ(1 − β))²/4) ≈ 3e−4`` (``σ = 3``, ``β = 0.7``),
    and the ramps end ``1.0`` (``e^{−κ} ≈ 2e−3`` of the bound field) before
    the screen.
    """
    speed, sigma = _SLIT_BETA, _SLIT_RAMP
    rise = 5.0 * sigma
    fall = rise + 2.0 * _SLIT_TRAVEL / speed

    def integral(value: np.ndarray) -> np.ndarray:
        scaled = value / sigma
        return (
            0.5
            * sigma
            * (scaled * (1.0 + erf(scaled)) + np.exp(-(scaled**2)) / np.sqrt(np.pi))
        )

    def shape(value: np.ndarray) -> np.ndarray:
        return integral(value - rise) - integral(value - fall)

    start = shape(np.zeros((1,)))
    heights = -_SLIT_TRAVEL + speed * (shape(times) - start)
    return heights, float(-_SLIT_TRAVEL - speed * rise - speed * start[0])


def _slit_rows() -> np.ndarray:
    """3-D ``E_x`` then ``E_z`` rows of the ``y = 0`` layer, in planar edge order."""
    bridge = _slit_bridge(False)
    offsets = bridge.orientation_offsets[1]
    rows = []
    for axis in (0, 2):
        shape = bridge.orientation_shapes[1][axis]
        index = np.arange(int(np.prod(shape))).reshape(shape)[:, 0, :]
        rows.append(offsets[axis] + index.reshape(-1))
    return np.concatenate(rows)


def _slit_time_domain(opening: int | None) -> tuple[np.ndarray, Any]:
    """Line-charge (``λ = 1``) transform ``Ẽ(ω)`` of the ramped run (``None``: no screen).

    Two charges ``±q`` share the ramped path, ``−q`` delayed by ``τ = π/ω``:
    every compensator and every resting pair coincides, so no static field is
    left, and the pair's transform is ``(1 − e^{iωτ}) Ẽ = 2Ẽ``. A line source
    in 2-D has an algebraic wake, so a sharp time cut leaves an end term
    ``~ E(T) e^{iωT}/(iω)``; the transform uses a Hann taper over
    ``[40, 56]`` (64 time-integral cuts a quarter period apart; a sharp cut at
    ``T = 52`` is off by 13 %, the taper by 4 %).
    """
    bridge = _slit_bridge(False)
    half_period = np.pi / _SLIT_OMEGA
    step = half_period / 16
    options: dict[str, Any] = {"pml": _slit_pml(False)}
    if opening is not None:
        options["boundaries"] = (_slit_screen(False, opening),)
    stable = phx.solver.CompatibleMaxwellPlan(bridge, **options).prepare().stable_dt
    assert step <= float(stable)
    cuts = 40.0 + 0.5 * half_period * np.arange(64)
    steps = round(cuts[-1] / step) + 1
    times = step * np.arange(steps + 1, dtype=np.float64)
    positions = np.zeros((steps + 1, 2, 3), dtype=np.float64)
    positions[:, :, 1] = 0.5 * _SLIT_SPACING
    positions[:, 0, 2] = _slit_heights(times)[0]
    positions[:, 1, 2] = _slit_heights(times - half_period)[0]
    particles = D.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,), dtype=jnp.float64), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((2,), dtype=jnp.float64), "slit").prepare(
        particles
    )
    current = PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
    )
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    probe = mx.FieldProbePlan("electric", _slit_rows())
    observers = tuple(
        mx.DFTObserverPlan(
            probe,
            mx.MaxwellSpectralAcquisition(
                np.asarray([_SLIT_OMEGA]),
                sign="positive",
                measure="time-integral",
                stop_time=cut + 0.5 * step,
            ),
        )
        for cut in cuts
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(mx.PrescribedChargeCurrentSourcePlan(trajectory, current),),
        observers=observers,
        **options,
    ).prepare()
    # Charge ±2h over the two periodic y cells is the line charge λ = ±1.
    charge = 2.0 * _SLIT_SPACING
    result = mx.solve_prescribed_charge_maxwell(
        mx.PrescribedChargeMaxwellPlan(
            maxwell, current, trajectory, np.asarray([charge, -charge])
        )
    )
    taper = np.hanning(cuts.shape[0] + 2)[1:-1]
    taper /= np.sum(taper)
    phasors = np.stack([np.asarray(value)[0] for value in result.observations])
    return 0.5 * np.tensordot(taper, phasors, axes=1), result.evidence


def _slit_frequency_domain(opening: int) -> Any:
    """B2b scattered-field solve of the planar slit for the cruise line."""
    bridge = _slit_bridge(True)
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    vacuum = mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout)
    _, origin = _slit_heights(np.zeros((1,)))
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=_SLIT_BETA,
        origin=[0.0, origin],
        direction=[0.0, 1.0],
    )
    return (
        mx.FrequencyMovingChargePlan(
            source,
            vacuum,
            formulation="scattered-field",
            background=vacuum,
            stretching=_slit_pml(True),
            boundaries=(_slit_screen(True, opening),),
        )
        .prepare()
        .solve(_SLIT_OMEGA)
    )


def _kazantsev_surdutovich(beta: float, omega: float, distance: float) -> float:
    """One-sided ``dW/dω`` per unit length of one PEC half-plane edge (λ = 1)."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    decay = omega / (beta * gamma)
    return beta * gamma * np.exp(-2.0 * decay * distance) / (4.0 * np.pi * omega)


def _planar_edge_energy(cells: int, beta: float, opening: int, both: bool) -> float:
    """B2b scattered-field ``dW/dω`` of a half-plane (or a slit) at ``λ/cells``.

    Screen on ``z = 0`` with ``x ≥ opening·h`` (and ``x ≤ −opening·h`` for a
    slit), box ``[−3.25, 3.25]²`` with a 0.75 CPML, ``ω = 2π``.
    """
    spacing = 1.0 / cells
    half = round(3.25 * cells)
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(2 * half), D.UniformCellAxisSpec(2 * half)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[-half * spacing] * 2, [half * spacing] * 2]))
    bridge = D.StructuredCochainBridge(grid)
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    vacuum = mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout)
    masks = []
    for axis, shape in enumerate(bridge.orientation_shapes[1]):
        column, row = np.indices(shape).reshape(2, -1) - half
        if axis == 1:
            masks.append(np.zeros_like(row, dtype=np.bool_))
            continue
        low, high = np.minimum(column, column + 1), np.maximum(column, column + 1)
        inside = low >= opening
        if both:
            inside |= high <= -opening
        masks.append((row == 0) & inside)
    source = mx.MaxwellMovingChargePlan(
        bridge, layout, charge=1.0, speed=beta, origin=[0.0, 0.0], direction=[0.0, 1.0]
    )
    width = round(0.75 * cells)
    result = (
        mx.FrequencyMovingChargePlan(
            source,
            vacuum,
            formulation="scattered-field",
            background=vacuum,
            stretching=mx.MaxwellCPMLPlan((width, width)),
            boundaries=(mx.MaxwellBoundaryPlan("pec", support=np.concatenate(masks)),),
        )
        .prepare()
        .solve(2.0 * np.pi)
    )
    assert bool(result.evidence.converged)
    return 2.0 / np.pi * float(result.ledger.absorbed_power)


def _slit_energy(electric: np.ndarray) -> float:
    """``(2/π)`` × the stretched-layer absorbed power of a planar scattered field."""
    bridge = _slit_bridge(True)
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    operator = mx.FrequencyMaxwellOperator(
        bridge,
        layout,
        mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout),
        jnp.asarray(_SLIT_OMEGA),
        stretching=_slit_pml(True),
        boundaries=(_slit_screen(True, _SLIT_OPENINGS[0]),),
    )
    field = jnp.asarray(electric, dtype=jnp.complex128)
    ledger = operator.power_ledger(field, jnp.zeros_like(field))
    return 2.0 / np.pi * float(ledger.absorbed_power)


@dataclass(frozen=True)
class _Slit:
    scattered: np.ndarray
    vacuum: np.ndarray
    evidence: tuple[Any, Any]
    reference: Any


@pytest.fixture(scope="module")
def slit() -> _Slit:
    screened, screened_evidence = _slit_time_domain(_SLIT_OPENINGS[0])
    vacuum, vacuum_evidence = _slit_time_domain(None)
    return _Slit(
        screened - vacuum,
        vacuum,
        (screened_evidence, vacuum_evidence),
        _slit_frequency_domain(_SLIT_OPENINGS[0]),
    )


def _slit_positions() -> tuple[np.ndarray, np.ndarray]:
    """Edge-midpoint ``(x, z)`` of every planar electric edge, in cells."""
    bridge = _slit_bridge(True)
    points = []
    for axis, shape in enumerate(bridge.orientation_shapes[1]):
        nodes = np.indices(shape).reshape(2, -1).astype(np.float64)
        nodes[axis] += 0.5
        points.append(nodes.T)
    stacked = np.concatenate(points)
    return stacked[:, 0] - _SLIT_HALF_WIDTH, stacked[:, 1] - _SLIT_HALF_LENGTH


def test_slit_diffraction_radiation_matches_the_scattered_field_reference(
    slit: _Slit,
) -> None:
    ledger = int(mx.PrescribedChargeStatus.LEDGER_OPEN)
    for evidence in slit.evidence:
        # Δt = h/2 keeps a 9 % CPML ledger residual (the pair's stop ramps sit
        # 5 cells from the z layers); every constraint bit is clear.
        assert int(evidence.status) & ~ledger == 0
    reference = slit.reference
    evidence = reference.evidence
    assert bool(evidence.converged) and not bool(evidence.radiating)
    assert bool(evidence.bound_field_contained)
    assert float(evidence.source_distance) == pytest.approx(_SLIT_OPENINGS[0])
    assert reference.scattered_electric is not None
    assert reference.incident_electric is not None
    scattered = np.asarray(reference.scattered_electric)
    incident = np.asarray(reference.incident_electric)
    conductor = np.asarray(_slit_screen(True, _SLIT_OPENINGS[0]).support)
    # The routes differ in the incident field on the screen: the time domain
    # carries the lattice field of its Whitney deposit, the reference the
    # analytic field integrated on the edges; they differ at O((kh)²).
    mismatch = np.linalg.norm((slit.vacuum - incident)[conductor]) / np.linalg.norm(
        incident[conductor]
    )
    assert mismatch < 4e-2  # measured 2.5e-2
    x, z = _slit_positions()
    interior = _SLIT_HALF_WIDTH - _SLIT_ABSORBER
    inside = (np.abs(x) <= interior) & (np.abs(z) <= _SLIT_HALF_LENGTH - _SLIT_ABSORBER)
    inside &= ~conductor
    error = np.linalg.norm((slit.scattered - scattered)[inside]) / np.linalg.norm(
        scattered[inside]
    )
    assert error < 2.0 * mismatch  # measured 4.4e-2
    # Diffraction-radiation spectral energy into the absorbers, both routes.
    energy = _slit_energy(slit.scattered)
    np.testing.assert_allclose(
        _slit_energy(scattered), 2.0 / np.pi * float(reference.ledger.absorbed_power)
    )
    assert abs(energy / _slit_energy(scattered) - 1.0) < 3.0 * mismatch  # 6.5e-2


def test_slit_diffraction_radiation_against_kazantsev_surdutovich(slit: _Slit) -> None:
    # Two independent half-plane edges at a/2 (the slit form of the exact
    # half-plane solution). The staircased edge acts ≈ 0.3 h nearer the path
    # (first order in h, see the test below), within half a cell of the node
    # edge: 1 ≤ W/W_KS ≤ e^{κh}.
    beta, omega = _SLIT_BETA, _SLIT_OMEGA
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    distance = _SLIT_OPENINGS[0] * _SLIT_SPACING
    asymptotic = 2.0 * _kazantsev_surdutovich(beta, omega, distance)
    bound = np.exp(omega / (beta * gamma) * _SLIT_SPACING)
    assert slit.reference.scattered_electric is not None
    for energy in (
        _slit_energy(slit.scattered),
        _slit_energy(np.asarray(slit.reference.scattered_electric)),
    ):
        # Measured 1.39 (time domain) and 1.31 (reference) against 1.49.
        assert 1.0 < energy / asymptotic < bound


def test_kazantsev_surdutovich_is_the_limit_of_the_scattered_field_reference() -> None:
    beta, omega = 0.9, 2.0 * np.pi
    asymptotic = _kazantsev_surdutovich(beta, omega, 0.25)
    coarse, fine = (
        _planar_edge_energy(cells, beta, cells // 4, both=False) / asymptotic
        for cells in (16, 24)
    )
    # Staircased-edge offset ≈ 0.3 h: first order (measured 1.119 → 1.083, and
    # 1.064 at λ/32), and the Richardson limit (3·fine − 2·coarse) is KS to 1 %.
    assert 1.0 < fine < coarse
    assert 1.3 < (coarse - 1.0) / (fine - 1.0) < 1.8
    assert abs(3.0 * fine - 2.0 * coarse - 1.0) < 1.5e-2
    # H-polarized edges do not couple along the screen plane: the slit radiates
    # as two independent edges (measured 0.995).
    slit = _planar_edge_energy(16, beta, 4, both=True)
    np.testing.assert_allclose(slit / asymptotic, 2.0 * coarse, rtol=1e-2)
