#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-domain Cherenkov radiation of a prescribed uniformly moving charge.

Geometry. A charge ``q`` moves at ``v = βc`` along the periodic axis ``z``
(length ``L``); ``y`` is periodic with two cells and the charge sits midway
between its two vertex planes, so the deposit is uniform in ``y`` and the
source is an exact line charge ``λ = q/L_y`` on the lattice. ``x`` is open
(CPML of fixed physical thickness). The charge advances ``h/r`` per step, so
the discrete source is invariant under ``(z, t) → (z + h, t + rΔt)``: harmonic
``m`` of the time period ``T = L/v`` is an exact Bloch wave with
``k_z = ω_m/v``. Sample-mean phasors over an integer number of periods after
the start transient are the Fourier coefficients ``c_m`` (single-charge
transform ``T c_m``).

References (independent of the runtime, code units ``ε₀ = μ₀ = c = 1``):

* 2-D Frank–Tamm (derived here from Lorenz-gauge ``A_z``): the line charge
  current ``J̃_z = λ δ(x) e^{i k_z z}`` gives ``A_z = iμλ e^{i k_x|x|}/(2k_x)``
  with ``k_x = (ω/v)√(β²n² − 1)``, so ``H_y(0±) = ±λ/2`` and the energy per unit
  area of each plane ``x = ±a`` is ``dW/(dA dω) = k_x λ²/(4π ω ε)``; both sides
  give ``λ²√(β²n² − 1)/(2π ε v)`` per unit ``z`` and ``y``
  (Frank & Tamm, Dokl. Akad. Nauk SSSR 14, 109 (1937); Jackson, *Classical
  Electrodynamics*, 3rd ed., §13.4, reduced to a line source). With
  ``F(ω_m) = T c_m``, the periodic train carries the time-averaged power per
  area ``P_m = 2 Re(c_E c_H^*) = (ω₁/T) dW/(dA dω)`` through each plane.
* Cone: ``cos θ = 1/(βn)`` for the wavevector (and, in an isotropic medium,
  the energy flow); the lattice cone comes from the dispersion audit of the
  executed update (`CherenkovRegimePlan`, same ``h``, ``Δt``, and medium).
* Below threshold (``βn < 1``) the line field is evanescent,
  ``|E(x)| ∝ exp(−(ω/v)√(1 − β²n²)|x|)``, and carries no flux.
* Negative index (Veselago, Sov. Phys. Usp. 10, 509 (1968)): with
  ``Re ε, Re μ < 0`` the outgoing (``Im k_x > 0``) branch has
  ``Re k_x < 0``; the energy flows outward and backward, ``S_z < 0``.

The Lorentz–Drude medium is ``ε = 1 − ω_p²/(ω² + iγω)``,
``μ = 1 + F/(ω₀² − ω² − iγω)`` evaluated here in closed form.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_PERIOD_LENGTH = 1.0
_CHARGE = 1.0
_BETA = 0.9
_PERMITTIVITY = 2.25
_INDEX = 1.5
# Transverse layout in units of the periodic length: probe planes at ±a and
# ±a/2, a gap, and a CPML of fixed physical thickness (its reflection enters
# the flux at first order through the wave re-emitted from the far side).
_PLANE = 0.5
_GAP = 0.25
_ABSORBER = 1.25
# Periods simulated and averaged (the tail of the start transient decays).
_PERIODS = 24
_AVERAGED = 12
# The CPML power ledger carries a known O(1e-2) residual owned by the core
# prescribed-charge tests; every other status bit must be clear here.
_STATUS_MASK = ~int(mx.PrescribedChargeStatus.LEDGER_OPEN)


@dataclass(frozen=True)
class _Cherenkov:
    """Harmonic-one phasors of one run, projected on its Bloch wave ``e^{ik_z z}``."""

    spacing: float
    speed: float
    omega: float
    line_charge: float
    electric: dict[int, complex]
    magnetic: dict[int, complex]
    axial_electric: dict[int, complex]
    axial_magnetic: dict[int, complex]
    status: int

    @property
    def period(self) -> float:
        return _PERIOD_LENGTH / self.speed

    @property
    def plane_cells(self) -> int:
        return max(self.electric)

    def outward_flux(self, plane: int) -> float:
        """Time-averaged ``S·x̂ sign(x)`` at the vertex plane ``plane`` (cells)."""
        product = self.electric[plane] * np.conj(self.magnetic[plane])
        return float(-2.0 * np.real(product) * np.sign(plane))

    def axial_flux(self, plane: int) -> float:
        """Time-averaged ``S_z`` on the cell-center plane beside vertex ``plane``."""
        product = self.axial_electric[plane] * np.conj(self.axial_magnetic[plane])
        return float(2.0 * np.real(product))

    def transverse_wavenumber(self, side: int) -> complex:
        """``k_x`` from the two planes on one side: ``E(a)/E(a/2) = e^{i k_x a/2}``."""
        far = side * self.plane_cells
        ratio = self.electric[far] / self.electric[far // 2]
        distance = (self.plane_cells - self.plane_cells // 2) * self.spacing
        return complex(-1j * np.log(ratio) / distance)


def _run(
    cells: int,
    beta: float,
    constitutive: Any,
    subdivisions: int,
) -> _Cherenkov:
    """Drive one line charge ``cells`` per period with ``subdivisions`` steps per cell."""
    spacing = _PERIOD_LENGTH / cells
    plane = round(_PLANE / spacing)
    absorber = round(_ABSORBER / spacing)
    half = plane + round(_GAP / spacing) + absorber
    count = 2 * half
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
    bridge = D.StructuredCochainBridge(grid)
    options: dict[str, Any] = {
        "constitutive": constitutive,
        "pml": mx.MaxwellCPMLPlan((absorber, 0, 0)),
    }
    stable = float(
        phx.solver.CompatibleMaxwellPlan(bridge, **options).prepare().stable_dt
    )
    speed = beta
    step = spacing / (subdivisions * speed)
    assert step <= stable
    period = _PERIOD_LENGTH / speed
    steps = _PERIODS * subdivisions * cells
    times = step * np.arange(steps + 1, dtype=np.float64)
    positions = np.zeros((steps + 1, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = half * spacing
    positions[:, 0, 1] = 0.5 * spacing
    # Unwrapped z: the deposit wraps the chord across the periodic seam.
    positions[:, 0, 2] = speed * times
    particles = D.ParticleSetPlan(
        jnp.arange(1), jnp.ones((1,), dtype=jnp.float64), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(
        jnp.ones((1,), dtype=jnp.float64), "cherenkov"
    ).prepare(particles)
    current = PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
    )
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    source = mx.PrescribedChargeCurrentSourcePlan(trajectory, current)
    omega = 2.0 * np.pi / period
    # Window (start, stop] holds exactly the last _AVERAGED periods of samples.
    acquisition = mx.MaxwellSpectralAcquisition(
        np.asarray([omega], dtype=np.float64),
        sign="positive",
        measure="sample-mean",
        start_time=(_PERIODS - _AVERAGED) * period + 0.5 * step,
        stop_time=_PERIODS * period + 0.5 * step,
    )
    edges = bridge.orientation_offsets[1]
    faces = bridge.orientation_offsets[2]
    along = np.arange(cells)
    stride = 2 * cells

    def z_edges(ix: int) -> np.ndarray:
        # Edge (ix, iy, iz) spans z_{iz} → z_{iz+1} on vertex column (x_ix, y_iy).
        return np.stack([edges[2] + ix * stride + iy * cells + along for iy in (0, 1)], 1)

    def x_edges(ix: int) -> np.ndarray:
        return np.stack([edges[0] + ix * stride + iy * cells + along for iy in (0, 1)], 1)

    def y_faces(ix: int, shift: int) -> list[np.ndarray]:
        # The (x, z) face at (x_{ix+1/2}, z_{iz+1/2}) carries −B_y h².
        return [
            faces[1] + ix * stride + iy * cells + (along + shift) % cells for iy in (0, 1)
        ]

    observers = []
    labels = []
    for offset in (plane, plane // 2, -plane // 2, -plane):
        ix = half + offset
        # E_z at (x_ix, z_{iz+1/2}); H_y averaged over the faces at x_ix ± h/2.
        observers.append(mx.FieldProbePlan("electric", z_edges(ix)))
        observers.append(
            mx.FieldProbePlan(
                "magnetic", np.stack(y_faces(ix - 1, 0) + y_faces(ix, 0), 1)
            )
        )
        labels.append(("transverse", offset))
    for offset in (plane, -plane):
        ix = half + offset if offset > 0 else half + offset - 1
        # E_x at (x_{ix+1/2}, z_iz); H_y averaged over the faces at z_iz ± h/2.
        observers.append(mx.FieldProbePlan("electric", x_edges(ix)))
        observers.append(
            mx.FieldProbePlan("magnetic", np.stack(y_faces(ix, -1) + y_faces(ix, 0), 1))
        )
        labels.append(("axial", offset))
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(source,),
        observers=tuple(mx.DFTObserverPlan(probe, acquisition) for probe in observers),
        **options,
    ).prepare()
    plan = mx.PrescribedChargeMaxwellPlan(
        maxwell, current, trajectory, np.asarray([_CHARGE], dtype=np.float64)
    )
    result = mx.solve_prescribed_charge_maxwell(plan)
    kz = omega / speed
    centers = (along + 0.5) * spacing
    nodes = along * spacing
    values: dict[str, dict[int, complex]] = {
        "electric": {},
        "magnetic": {},
        "axial_electric": {},
        "axial_magnetic": {},
    }
    for index, (kind, offset) in enumerate(labels):
        electric = np.asarray(result.observations[2 * index])[0]
        magnetic = np.asarray(result.observations[2 * index + 1])[0]
        phase = np.exp(-1j * kz * (centers if kind == "transverse" else nodes))
        prefix = "" if kind == "transverse" else "axial_"
        values[prefix + "electric"][offset] = complex(np.mean(electric * phase) / spacing)
        values[prefix + "magnetic"][offset] = complex(
            -np.mean(magnetic * phase) / spacing**2
        )
    return _Cherenkov(
        spacing=spacing,
        speed=speed,
        omega=omega,
        line_charge=_CHARGE / (2.0 * spacing),
        electric=values["electric"],
        magnetic=values["magnetic"],
        axial_electric=values["axial_electric"],
        axial_magnetic=values["axial_magnetic"],
        status=int(result.evidence.status),
    )


def _regime(cells: int, beta: float, constitutive: Any, subdivisions: int) -> Any:
    """Dispersion-audited Cherenkov regime at harmonic one (azimuth in the x–z plane)."""
    spacing = _PERIOD_LENGTH / cells
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(8, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [8.0 * spacing] * 3]))
    runtime = phx.solver.CompatibleMaxwellPlan(
        D.StructuredCochainBridge(grid), constitutive=constitutive
    ).prepare()
    audit = mx.CompatibleMaxwellDispersionAudit(
        runtime,
        mx.MaxwellMaterialRegion((0, 0, 0), (8, 8, 8)),
        spacing / (subdivisions * beta),
    )
    omega = 2.0 * np.pi * beta / _PERIOD_LENGTH
    return mx.CherenkovRegimePlan(
        audit,
        jnp.asarray([0.0, 0.0, beta]),
        jnp.asarray([omega]),
        1,
    ).evaluate()


def _dielectric() -> Any:
    return mx.DiagonalMaxwellConstitutivePlan(permittivity=_PERMITTIVITY)


def _frank_tamm(line_charge: float, beta: float, transverse: float) -> float:
    """Per-plane power per area ``(ω₁/T) k_x λ²/(4π ω₁ ε)`` of harmonic one."""
    period = _PERIOD_LENGTH / beta
    return transverse * line_charge**2 / (4.0 * np.pi * period * _PERMITTIVITY)


def _continuum_transverse(beta: float) -> float:
    omega = 2.0 * np.pi * beta / _PERIOD_LENGTH
    return omega / beta * np.sqrt((beta * _INDEX) ** 2 - 1.0)


def _numerical_cone(evidence: Any) -> float:
    angles = np.asarray(evidence.numerical_cone_angle)[0, 0]
    finite = angles[np.isfinite(angles)]
    # Degenerate TE/TM branches in the x–z plane share one cone.
    np.testing.assert_allclose(finite, finite[0], rtol=1e-9)
    return float(finite[0])


@pytest.fixture(scope="module")
def coarse() -> _Cherenkov:
    return _run(16, _BETA, _dielectric(), 2)


@pytest.fixture(scope="module")
def fine() -> _Cherenkov:
    return _run(32, _BETA, _dielectric(), 2)


@pytest.fixture(scope="module")
def coarse_regime() -> Any:
    return _regime(16, _BETA, _dielectric(), 2)


def test_line_frank_tamm_flux_matches_audited_lattice_then_the_continuum(
    coarse: _Cherenkov, fine: _Cherenkov, coarse_regime: Any
) -> None:
    assert coarse.status & _STATUS_MASK == 0
    assert fine.status & _STATUS_MASK == 0
    lattice = coarse.omega / coarse.speed * np.tan(_numerical_cone(coarse_regime))
    for result in (coarse, fine):
        plane = result.plane_cells
        # Mirror symmetry and a lossless medium: equal outward flux on both
        # sides and through the nested planes (CPML re-emission is O(1e-4)).
        np.testing.assert_allclose(
            result.outward_flux(-plane), result.outward_flux(plane), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.outward_flux(plane // 2), result.outward_flux(plane), rtol=1e-3
        )
    # Lattice index from the dispersion audit; the remaining O((kh)²) terms of
    # the discrete impedance, the edge/step top-hat form factors
    # sinc²(k_z h/2) sinc²(ωΔt/2), and ω/Ω of leapfrog partly cancel (≈0.4 %).
    np.testing.assert_allclose(
        coarse.outward_flux(coarse.plane_cells),
        _frank_tamm(coarse.line_charge, _BETA, lattice),
        rtol=6e-3,
    )
    continuum = _continuum_transverse(_BETA)
    coarse_error = abs(
        coarse.outward_flux(coarse.plane_cells)
        / _frank_tamm(coarse.line_charge, _BETA, continuum)
        - 1.0
    )
    fine_error = abs(
        fine.outward_flux(fine.plane_cells)
        / _frank_tamm(fine.line_charge, _BETA, continuum)
        - 1.0
    )
    # Second order in h: halving h must cut the continuum error by ≳ 4.
    assert fine_error < coarse_error / 3.0
    assert fine_error < 4e-3
    # Ampère's jump |H_y| = λ/2 of the single-charge transform T·c_H.
    np.testing.assert_allclose(
        abs(fine.magnetic[fine.plane_cells]) * fine.period,
        0.5 * fine.line_charge,
        rtol=5e-3,
    )


def test_cherenkov_cone_from_the_measured_phase(
    coarse: _Cherenkov, fine: _Cherenkov, coarse_regime: Any
) -> None:
    kz = coarse.omega / coarse.speed
    for side in (1, -1):
        transverse = coarse.transverse_wavenumber(side)
        # Lossless outgoing wave: real k_x > 0 (no decay between the planes).
        assert abs(transverse.imag) < 1e-3 * transverse.real
        # The executed lattice cone: k_x/k_z against the audited Bloch branch.
        np.testing.assert_allclose(
            np.arctan2(transverse.real, kz), _numerical_cone(coarse_regime), rtol=1e-3
        )
    continuum = np.arccos(1.0 / (_BETA * _INDEX))
    np.testing.assert_allclose(
        np.asarray(coarse_regime.physical_cone_angle)[0, 0], continuum, rtol=1e-9
    )
    # Coarse lattice dispersion shifts the cone by ≈0.7 % (k h ≈ 0.53); at
    # h/2 the shift falls to ≈0.2 %.
    np.testing.assert_allclose(
        np.arctan2(coarse.transverse_wavenumber(1).real, kz), continuum, rtol=1.5e-2
    )
    np.testing.assert_allclose(
        np.arctan2(fine.transverse_wavenumber(1).real, fine.omega / fine.speed),
        continuum,
        rtol=4e-3,
    )
    # Isotropic medium: the energy flows along the wavevector cone.
    plane = coarse.plane_cells
    for side in (1, -1):
        np.testing.assert_allclose(
            coarse.axial_flux(side * plane) / coarse.outward_flux(side * plane),
            kz / coarse.transverse_wavenumber(side).real,
            rtol=1e-2,
        )


@pytest.fixture(scope="module")
def below() -> tuple[_Cherenkov, Any]:
    beta = 0.6
    return _run(16, beta, _dielectric(), 3), _regime(16, beta, _dielectric(), 3)


def test_below_threshold_line_charge_does_not_radiate(
    below: tuple[_Cherenkov, Any],
) -> None:
    result, regime = below
    assert result.status & _STATUS_MASK == 0
    # βn = 0.9: neither the continuum nor the executed lattice has a resonance.
    np.testing.assert_allclose(np.asarray(regime.continuum_index), _INDEX, rtol=1e-7)
    assert not bool(np.any(regime.physical_emission))
    assert not bool(np.any(regime.numerical_emission))
    assert bool(np.all(regime.resolved))
    # Same line charge above threshold (β = 0.9) radiates the Frank–Tamm power.
    above = _frank_tamm(result.line_charge, _BETA, _continuum_transverse(_BETA))
    for plane in (result.plane_cells, result.plane_cells // 2, -result.plane_cells):
        assert abs(result.outward_flux(plane)) < 1e-3 * above
    # Evanescent field: κ = (ω/v)√(1 − β²n²). The lattice κ² is a difference
    # of two O(k²) terms, so the O((k_z h)²) lattice shift is amplified (≈3 %).
    decay = (result.omega / result.speed) * np.sqrt(1.0 - (0.6 * _INDEX) ** 2)
    for side in (1, -1):
        transverse = result.transverse_wavenumber(side)
        np.testing.assert_allclose(transverse.imag, decay, rtol=5e-2)
        assert abs(transverse.real) < 1e-2 * decay


def _negative_index_material(omega: float) -> tuple[Any, complex, complex]:
    """Drude ε and Lorentz μ with ``Re ε ≈ Re μ ≈ −1.5`` at ``omega``."""
    plasma = 2.5 * omega**2
    resonance = 0.8 * omega
    strength = 0.9 * omega**2
    damping = 0.05 * omega
    material = mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([0.0], [damping], [plasma]),
        magnetic_poles=mx.MaxwellLorentzPoles([resonance], [damping], [strength]),
    )
    permittivity = 1.0 - plasma / (omega**2 + 1j * damping * omega)
    permeability = 1.0 + strength / (resonance**2 - omega**2 - 1j * damping * omega)
    return material, complex(permittivity), complex(permeability)


@pytest.fixture(scope="module")
def negative() -> tuple[_Cherenkov, Any, complex, complex]:
    omega = 2.0 * np.pi * _BETA / _PERIOD_LENGTH
    material, permittivity, permeability = _negative_index_material(omega)
    return (
        _run(16, _BETA, material, 3),
        _regime(16, _BETA, material, 3),
        permittivity,
        permeability,
    )


def test_negative_index_medium_radiates_the_reversed_cone(
    negative: tuple[_Cherenkov, Any, complex, complex],
) -> None:
    result, regime, permittivity, permeability = negative
    assert result.status & _STATUS_MASK == 0
    assert permittivity.real < 0.0 and permeability.real < 0.0
    index = -np.sqrt(permittivity * permeability)
    np.testing.assert_allclose(np.asarray(regime.continuum_index)[0], index, rtol=1e-6)
    assert bool(np.all(regime.physical_emission))
    kz = result.omega / result.speed
    # Outgoing (decaying) branch of k_x² = ω²εμ − k_z²: Re k_x < 0, the phase
    # runs inward while the energy leaves.
    root = np.sqrt(result.omega**2 * permittivity * permeability - kz**2 + 0j)
    expected = root if root.imag > 0.0 else -root
    assert expected.real < 0.0
    plane = result.plane_cells
    for side in (1, -1):
        transverse = result.transverse_wavenumber(side)
        # O((kh)²) lattice dispersion at k h ≈ 0.36 (≈1.5 %).
        np.testing.assert_allclose(transverse, expected, rtol=3e-2)
        assert transverse.real < 0.0
        outer = result.outward_flux(side * plane)
        inner = result.outward_flux(side * plane // 2)
        assert outer > 0.0 and inner > 0.0
        # Material loss: |S| decays as exp(−2 Im k_x Δx) between the planes.
        np.testing.assert_allclose(
            outer / inner,
            np.exp(-2.0 * expected.imag * (plane - plane // 2) * result.spacing),
            rtol=3e-2,
        )
        # Backward energy flow: the time-averaged S has S_z < 0 against v.
        axial = result.axial_flux(side * plane)
        assert axial < 0.0
        assert np.arctan2(outer, axial) > 0.5 * np.pi
