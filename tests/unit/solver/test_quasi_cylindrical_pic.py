#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts of the quasi-cylindrical PSATD spectral PIC field solver.

References are independent of the solver: Bessel-function eigenmodes of the
Laplacian and exact cylindrical TM vacuum modes, paraxial Gaussian-beam optics
(Siegman, *Lasers*, ch. 17), the Hertzian dipole (Jackson 9.4), the Cartesian
P2 PSATD solver, and — when a pinned interpreter is configured — FBPIC.
"""

import os
import shutil
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.special import jv

import phydrax as phx
from phydrax._external_runtime import pin_executable


sp = phx.solver.maxwell.spectral
mx = phx.solver.maxwell
D = phx.discretization


def _grid(
    radius: float, radial: int, length: float, axial: int, modes: int
) -> D.pic.QuasiCylindricalGrid:
    return D.pic.QuasiCylindricalGrid(radius, radial, 0.0, length, axial, modes)


def _zero_source(grid: D.pic.QuasiCylindricalGrid) -> Any:
    shape = (grid.mode_count, grid.radial_count, grid.axial_count)
    return sp.QuasiCylindricalSource(
        jnp.zeros((*shape, 3), dtype=jnp.complex128),
        jnp.zeros(shape, dtype=jnp.complex128),
    )


def _vacuum_run(solver: Any, field: Any, step: float, steps: int) -> Any:
    source = _zero_source(solver.grid)

    def body(value: Any, _: None) -> tuple[Any, tuple[Array, Array, Array]]:
        result = solver.advance(jnp.asarray(0.0), value, source, jnp.asarray(step))
        return result.field, (
            result.electric_constraint,
            result.magnetic_constraint,
            result.diagnostics.absorbed_energy,
        )

    return eqx.filter_jit(lambda value: jax.lax.scan(body, value, None, length=steps))(
        field
    )


def _species(
    grid: D.pic.QuasiCylindricalGrid,
    weights: np.ndarray,
    *,
    shape_order: D.pic.PICShapeOrder = 1,
    ion_mass_ratio: float = 100.0,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Electrons (q/m = −1) and ions (q/m = 1/ratio) of per-particle macrocharge ``w``."""
    count = weights.size
    transfer = D.pic.AzimuthalTransferPlan(grid, shape_order=shape_order)
    species, transfers = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", 1.0),
        (10**7, 1.0 / ion_mass_ratio, "ions", ion_mass_ratio),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + count),
            jnp.asarray(mass * weights),
            ambient_dimension=3,
        ).prepare()
        species.append(
            D.pic.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                D.pic.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
        transfers.append(transfer.prepare(support))
    return tuple(species), tuple(transfers)


def _cylinder_points(
    rng: np.random.Generator, count: int, radius: float, length: float
) -> np.ndarray:
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, count))
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    return np.stack(
        (r * np.cos(angle), r * np.sin(angle), rng.uniform(0.0, length, count)), axis=-1
    )


# -- shared-grid Hankel transforms ----------------------------------------------------


@pytest.mark.parametrize("mode", [0, 1, 3], ids=["m0", "m1", "m3"])
def test_shared_grid_hankel_diagonalizes_the_mode_laplacian(mode: int) -> None:
    radius, count = 1.5, 24
    prepared = D.SharedGridHankelPlan(radius, count, mode + 1).prepare()
    evidence = prepared.evidence
    assert bool(evidence.successful)
    # Full rank except orders m and m + 1 at m ≥ 1, whose k = 0 column vanishes.
    expected = (
        np.full(3, count) if mode == 0 else np.asarray([count, count - 1, count - 1])
    )
    np.testing.assert_array_equal(np.asarray(evidence.rank)[mode], expected)
    assert np.all(np.asarray(evidence.condition)[mode] < 100.0)
    r = (np.arange(count) + 0.5) * radius / count
    k = np.asarray(prepared.wavenumbers)[mode]
    # Independent k-grid: zeros of J_m (k = 0 leads for m ≥ 1).
    np.testing.assert_allclose(jv(mode, k * radius), 0.0, atol=1e-12)
    for offset in (-1, 0, 1):
        order = mode + offset
        for index in range(0 if mode == 0 or offset == -1 else 1, count, 5):
            # J_p(k r) e^{ipθ} is a Laplacian eigenfunction with eigenvalue −k²;
            # the analysis must return exactly that one spectral node.
            if mode > 0 and index == 0:
                samples = (r / radius) ** (mode - 1)
            else:
                samples = jv(order, k[index] * r)
            values = np.zeros((mode + 1, count))
            values[mode] = samples
            coefficients = np.asarray(prepared.forward(values, offset))[mode]
            unit = np.zeros(count)
            unit[index] = 1.0
            np.testing.assert_allclose(coefficients, unit, atol=1e-9)
            laplacian = np.asarray(
                prepared.inverse(
                    np.where(np.arange(mode + 1)[:, None] == mode, -(k**2) * unit, 0.0),
                    offset,
                )
            )[mode]
            np.testing.assert_allclose(
                laplacian, -(k[index] ** 2) * samples, atol=1e-9 * (1.0 + k[index] ** 2)
            )


def test_shared_grid_hankel_refuses_invalid_plans() -> None:
    with pytest.raises(ValueError, match="radius"):
        D.SharedGridHankelPlan(0.0, 8, 1)
    with pytest.raises(ValueError, match="mode_count"):
        D.SharedGridHankelPlan(1.0, 8, 0)
    with pytest.raises(ValueError, match="maximum_matrix_elements"):
        D.SharedGridHankelPlan(1.0, 64, 4, maximum_matrix_elements=1000)


# -- vacuum dispersion -------------------------------------------------------------


def _tm_mode(
    grid: D.pic.QuasiCylindricalGrid, mode: int, k: float, kz: float, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact traveling TM mode ``E_z = J_m(kr) e^{i(mθ + k_z z − ωt)}``, ``ω = |(k, k_z)|``.

    ``E_⊥ = (ik_z/k²)∇_⊥E_z`` and ``B_⊥ = (iω/k²) ẑ × ∇_⊥E_z`` (c = 1), in
    circular components ``F_± = (F_r ∓ iF_θ)/2``.
    """
    r = grid.radial_coordinates[:, None]
    z = grid.axial_coordinates[None, :]
    omega = np.hypot(k, kz)
    phase = np.exp(1j * (kz * z - omega * time))
    electric = np.zeros((grid.mode_count, r.size, z.size, 3), complex)
    magnetic = np.zeros_like(electric)
    electric[mode, :, :, 0] = 1j * kz / (2 * k) * jv(mode - 1, k * r) * phase
    electric[mode, :, :, 1] = -1j * kz / (2 * k) * jv(mode + 1, k * r) * phase
    electric[mode, :, :, 2] = jv(mode, k * r) * phase
    magnetic[mode, :, :, 0] = omega / (2 * k) * jv(mode - 1, k * r) * phase
    magnetic[mode, :, :, 1] = omega / (2 * k) * jv(mode + 1, k * r) * phase
    return electric, magnetic


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["m0", "m1", "m2"])
def test_vacuum_modes_follow_exact_dispersion(mode: int) -> None:
    grid = _grid(2.0, 32, 4.0, 32, 3)
    solver = sp.QuasiCylindricalMaxwellPlan(grid).prepare()
    k = D.SharedGridHankelPlan(2.0, 32, 3).prepare().wavenumbers[mode, 4]
    kz = 2.0 * np.pi * 3 / 4.0
    assert float(
        solver.dispersion_frequency(np.asarray([[float(k), 0.0, kz]]), 0.01)[0]
    ) == pytest.approx(np.hypot(float(k), kz), rel=1e-14)
    step, steps = 0.9 * float(solver.stable_step), 20
    initial = solver.circular_field(*_tm_mode(grid, mode, float(k), kz, 0.0))
    final, (electric_constraint, magnetic_constraint, _) = _vacuum_run(
        solver, initial, step, steps
    )
    electric, magnetic = _tm_mode(grid, mode, float(k), kz, steps * step)
    np.testing.assert_allclose(np.asarray(final.electric), electric, atol=1e-12)
    np.testing.assert_allclose(np.asarray(final.magnetic), magnetic, atol=1e-12)
    assert float(jnp.max(electric_constraint)) < 1e-12
    assert float(jnp.max(magnetic_constraint)) < 1e-12


# -- antennas and lasers ------------------------------------------------------------


def test_gaussian_laser_antenna_matches_paraxial_beam() -> None:
    """An m = 1 CW Gaussian launched at focus diffracts as a paraxial beam.

    Waist ``w(z) = w₀√(1 + z²/z_R²)``, on-axis amplitude ``∝ w₀/w``, and the
    phase advance ``kΔz − Δarctan(z/z_R)`` (Gouy) are measured by lock-in at two
    planes; ``w₀ = 1.5λ`` gives a paraxial parameter ``θ² ≈ 0.045``, so the
    tolerances cover the ``O(θ²)`` nonparaxial corrections. The sheet is
    one-way: the field behind it stays below 2% of the launched amplitude.
    """
    k0, waist, plane = 2.0 * np.pi, 1.5, 2.0
    rayleigh = k0 * waist**2 / 2.0
    grid = _grid(6.0, 48, 16.0, 192, 2)
    xs = np.linspace(-6.5, 6.5, 131)
    times = np.linspace(0.0, 16.0, 161)
    ramp = np.where(
        times < 3.0, np.sin(0.5 * np.pi * np.clip(times, 0.0, 3.0) / 3.0) ** 2, 1.0
    )
    x, y = np.meshgrid(xs, xs, indexing="ij")
    envelope = np.zeros((xs.size, xs.size, times.size, 2), complex)
    envelope[..., 0] = np.exp(-(x**2 + y**2) / waist**2)[:, :, None] * ramp
    antenna = sp.QuasiCylindricalAntennaPlan(
        plane, xs, xs, times, envelope, carrier_angular_frequency=k0
    )
    solver = sp.QuasiCylindricalMaxwellPlan(grid, antennas=(antenna,)).prepare()
    step = 0.9 * float(solver.stable_step)
    advance = eqx.filter_jit(
        lambda value, time: solver.advance(
            time, value, _zero_source(grid), jnp.asarray(step)
        )
    )
    near_index = int(round((plane + 0.5) / grid.axial_spacing))
    far_index = int(round((plane + rayleigh) / grid.axial_spacing))
    back_index = int(round((plane - 1.5) / grid.axial_spacing))
    near = np.zeros(grid.radial_count, complex)
    far = np.zeros(grid.radial_count, complex)
    behind = 0.0
    field = solver.field_with_charge(jnp.zeros((2, 48, 192), dtype=jnp.complex128))
    time = 0.0
    for _ in range(int(13.0 / step)):
        result = advance(field, jnp.asarray(time))
        assert bool(result.successful)
        field, time = result.field, time + step
        if time > 10.0:
            # Physical E_x at θ = 0 of mode 1: Re(E_+ + E_−).
            ex = np.real(
                np.asarray(field.electric[1, :, :, 0] + field.electric[1, :, :, 1])
            )
            near += ex[:, near_index] * np.exp(1j * k0 * time) * step
            far += ex[:, far_index] * np.exp(1j * k0 * time) * step
            behind = max(behind, float(np.max(np.abs(ex[:, back_index]))))
    assert float(result.electric_constraint) < 1e-12
    r = grid.radial_coordinates
    z_near = near_index * grid.axial_spacing - plane
    z_far = far_index * grid.axial_spacing - plane

    def width(amplitude: np.ndarray) -> float:
        # For exp(−2r²/w²): ⟨r²⟩ = w²/2.
        weight = np.abs(amplitude) ** 2
        return float(np.sqrt(2.0 * np.sum(weight * r**3) / np.sum(weight * r)))

    def beam(z: float) -> float:
        return waist * np.sqrt(1.0 + (z / rayleigh) ** 2)

    assert width(far) == pytest.approx(beam(z_far), rel=0.02)
    assert width(near) == pytest.approx(beam(z_near), rel=0.02)
    assert abs(far[0]) / abs(near[0]) == pytest.approx(
        beam(z_near) / beam(z_far), rel=0.05
    )
    gouy = np.arctan(z_far / rayleigh) - np.arctan(z_near / rayleigh)
    advance_phase = np.angle(
        far[0] / near[0] * np.exp(-1j * (k0 * (z_far - z_near) - gouy))
    )
    assert abs(advance_phase) < 0.05
    launched = 2.0 * np.max(np.abs(near)) / (time - 10.0)
    assert behind < 0.02 * launched


# -- charge conservation and deposition ------------------------------------------------


_GAUSS_CASES: dict[str, dict[str, Any]] = {
    "update-with-rho": {"charge_conservation": "update-with-rho"},
    "spectral-correction": {"charge_conservation": "spectral-correction"},
    "galilean-rho": {"variant": "galilean", "galilean_velocity": 0.4},
    "galilean-correction": {
        "variant": "galilean",
        "galilean_velocity": -0.3,
        "charge_conservation": "spectral-correction",
    },
    "averaged-galilean": {"variant": "averaged-galilean", "galilean_velocity": 0.5},
    "quadratic-shape": {"shape_order": 2},
    "cubic-shape": {"shape_order": 3},
}


@pytest.mark.parametrize("options", list(_GAUSS_CASES.values()), ids=list(_GAUSS_CASES))
def test_gauss_law_and_charge_conservation_hold_to_roundoff(
    options: dict[str, Any],
) -> None:
    settings = dict(options)
    shape_order = settings.pop("shape_order", 1)
    grid = _grid(2.0, 24, 3.0, 24, 3)
    rng = np.random.default_rng(3)
    count = 32
    positions = _cylinder_points(rng, count, 1.2, 3.0)
    species, transfers = _species(grid, np.ones(count), shape_order=shape_order)
    solver = sp.QuasiCylindricalMaxwellPlan(grid, **settings).prepare(transfers)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    # Independent deposit↔Gauss pairing: advancing with the deposited current lands
    # the solver's Gauss charge on the deposited end charge.
    assert pic.pairing_defect < 1e-12
    step = 0.5 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.2, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)
    advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step))
    for _ in range(3):
        result = advance(state)
        assert bool(result.successful)
        charge = np.asarray(result.accepted_state.field.charge)
        scale = np.max(np.abs(charge))
        assert scale > 0.0
        assert float(result.diagnostics.electric_constraint) < 1e-11 * scale
        assert float(result.diagnostics.magnetic_constraint) < 1e-11 * scale
        state = result.accepted_state
    # The advanced Gauss charge is the redeposited particle charge.
    deposited, _ = pic.species_charge(state.species)
    np.testing.assert_allclose(
        np.asarray(state.field.charge), np.asarray(deposited), atol=1e-11 * scale
    )


def _ring_lattice(rings: int, angles: int, planes: int, radius: float) -> np.ndarray:
    """Equal-area rings of equally spaced particles (a uniform areal density)."""
    r = radius * np.sqrt((np.arange(rings) + 0.5) / rings)
    theta = 2.0 * np.pi * (np.arange(angles) + 0.5) / angles
    z = (np.arange(planes) + 0.5) / planes
    rr, tt, zz = np.meshgrid(r, theta, z, indexing="ij")
    return np.stack((rr * np.cos(tt), rr * np.sin(tt), zz), axis=-1).reshape(-1, 3)


@pytest.mark.parametrize("shape_order", [1, 2, 3])
def test_near_axis_deposition_recovers_regular_modal_densities(
    shape_order: D.pic.PICShapeOrder,
) -> None:
    """Uniform density deposits uniformly and ``ρ = x`` deposits ``ρ₁ = r``, to the axis.

    Without the near-axis volume correction the linear-shape axis node of a
    uniform density reads ``13/12`` of the density.
    """
    grid = D.pic.QuasiCylindricalGrid(1.0, 16, 0.0, 1.0, 8, 2)
    radius = 0.75
    positions = _ring_lattice(400, 16, 16, radius)
    count = positions.shape[0]
    support = D.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    transfer = D.pic.AzimuthalTransferPlan(grid, shape_order=shape_order).prepare(support)
    active = jnp.ones((count,), dtype=jnp.bool_)
    uniform, successful = transfer.deposit(
        jnp.asarray(positions), active, jnp.ones((count, 1)), (0,)
    )
    assert bool(successful)
    density = count / (np.pi * radius**2)
    profile = np.real(np.asarray(uniform[0, :, :, 0])).mean(axis=1) / density
    np.testing.assert_allclose(profile[:8], 1.0, atol=3e-3)
    assert np.max(np.abs(np.asarray(uniform[1]))) < 1e-12 * density
    linear, _ = transfer.deposit(
        jnp.asarray(positions), active, jnp.asarray(positions[:, :1]), (0,)
    )
    ramp = np.real(np.asarray(linear[1, :, :, 0])).mean(axis=1) / density
    np.testing.assert_allclose(ramp[:8], grid.radial_coordinates[:8], atol=3e-3)


# -- cross-solver and observation -------------------------------------------------


def test_axisymmetric_field_matches_cartesian_psatd() -> None:
    """An m = 0 solenoidal pulse evolves as in the 3-D Cartesian P2 solver.

    ``E = ∇ × (ẑψ)``, ``ψ = exp(−r²/a² − (z − z₀)²/b²)``; the quasi-cylindrical
    field is synthesized at the Cartesian nodes from its own spectrum.
    """
    a, b, center, duration = 0.35, 0.5, 2.0, 1.0
    count, axial, h = 48, 32, 0.1
    length = 4.0
    tensor = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(count, periodic=True),
            D.UniformCellAxisSpec(count, periodic=True),
            D.UniformCellAxisSpec(axial, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [count * h, count * h, length]]))
    cartesian = sp.SpectralMaxwellPlan(D.StructuredCochainBridge(tensor)).prepare()
    x = np.arange(count) * h - 0.5 * count * h
    z = np.arange(axial) * length / axial
    xx, yy, zz = np.meshgrid(x, x, z, indexing="ij")
    psi = np.exp(-(xx**2 + yy**2) / a**2 - (zz - center) ** 2 / b**2)
    electric = np.stack(
        (-2.0 * yy / a**2 * psi, 2.0 * xx / a**2 * psi, np.zeros_like(psi)), axis=-1
    )
    initial = eqx.tree_at(
        lambda state: state.electric,
        cartesian.field_with_charge(jnp.zeros((count, count, axial))),
        jnp.asarray(electric),
    )
    steps = int(np.ceil(duration / (0.9 * float(cartesian.stable_step))))
    source = sp.SpectralMaxwellSource(
        jnp.zeros((1, count, count, axial, 3)), jnp.zeros((1, count, count, axial))
    )
    reference, _ = eqx.filter_jit(
        lambda value: jax.lax.scan(
            lambda state, _: (
                cartesian.advance(
                    jnp.asarray(0.0), state, source, jnp.asarray(duration / steps)
                ).field,
                None,
            ),
            value,
            None,
            length=steps,
        )
    )(initial)
    radius, radial = 2.4, 48
    grid = _grid(radius, radial, length, axial, 1)
    solver = sp.QuasiCylindricalMaxwellPlan(grid).prepare()
    r = grid.radial_coordinates[:, None]
    theta_field = (
        2.0 * r / a**2 * np.exp(-(r**2) / a**2 - (z[None, :] - center) ** 2 / b**2)
    )
    circular = np.zeros((1, radial, axial, 3), complex)
    circular[0, :, :, 0] = -0.5j * theta_field
    circular[0, :, :, 1] = 0.5j * theta_field
    quasi_steps = int(np.ceil(duration / (0.9 * float(solver.stable_step))))
    final, _ = _vacuum_run(
        solver,
        solver.circular_field(circular, np.zeros_like(circular)),
        duration / quasi_steps,
        quasi_steps,
    )
    # Synthesize E_θ = i(E_+ − E_−) and B_z on the Cartesian x ≥ 0, y = 0 nodes.
    k = np.asarray(D.SharedGridHankelPlan(radius, radial, 1).prepare().wavenumbers[0])
    spectrum_e = np.asarray(final.spectral_electric)[0]
    spectrum_b = np.asarray(final.spectral_magnetic)[0]
    plus = 0.5 * (-1j * spectrum_e[..., 0] - spectrum_e[..., 1])
    minus = 0.5 * (1j * spectrum_e[..., 0] - spectrum_e[..., 1])
    radii = x[count // 2 :]
    kz = 2.0 * np.pi * np.fft.fftfreq(axial, d=length / axial)
    axial_phase = np.exp(1j * np.outer(kz, z))
    bessel = np.outer(radii, k)
    azimuthal = np.real(
        1j * ((-jv(1, bessel) @ plus - jv(1, bessel) @ minus) @ axial_phase)
    )
    along = np.real((jv(0, bessel) @ spectrum_b[..., 2]) @ axial_phase)
    expected_e = np.asarray(reference.electric)[count // 2 :, count // 2, :, 1]
    expected_b = np.asarray(reference.magnetic)[count // 2 :, count // 2, :, 2]
    assert np.max(np.abs(azimuthal - expected_e)) < 1e-5 * np.max(np.abs(expected_e))
    assert np.max(np.abs(along - expected_b)) < 1e-5 * np.max(np.abs(expected_b))


_OMEGA0 = 4.0 * np.pi
_TAU = 0.15
_T0 = 3.0 * _TAU
_OMEGAS = np.asarray([0.8 * _OMEGA0, _OMEGA0, 1.2 * _OMEGA0])


def test_hertzian_dipole_far_field_through_huygens_cylinder() -> None:
    """An on-axis Gaussian ``J_z`` blob radiates the Hertzian-dipole far field.

    The moment is ``π^{3/2}s³`` times the pulse spectrum and the blob's form
    factor ``exp(−k²s²/4)``; charge follows continuity ``ρ̇ = −∂_zJ_z``.
    """
    grid = _grid(2.0, 36, 4.0, 72, 1)
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray(_OMEGAS), sign="positive", measure="time-integral", stop_time=2.0
    )
    exterior = mx.HomogeneousMaxwellExterior()
    box = sp.QuasiCylindricalHuygensPlan(9, 27, 45, acquisition, exterior)
    solver = sp.QuasiCylindricalMaxwellPlan(grid, observers=(box,)).prepare()
    center, size = 2.0, 0.1
    r = grid.radial_coordinates[:, None]
    z = grid.axial_coordinates[None, :]
    distance = r**2 + (z - center) ** 2
    blob = np.where(distance < (4.0 * size) ** 2, np.exp(-distance / size**2), 0.0)
    current = np.zeros((1, 36, 72, 3), complex)
    current[0, :, :, 2] = blob
    divergence = np.zeros((1, 36, 72), complex)
    divergence[0] = -2.0 * (z - center) / size**2 * blob
    step = 0.9 * float(solver.stable_step)
    steps = int(np.ceil(2.0 / step))

    def envelope(time: Array) -> Array:
        return jnp.exp(-(((time - _T0) / _TAU) ** 2)) * jnp.sin(_OMEGA0 * (time - _T0))

    def body(field: Any, index: Array) -> tuple[Any, Array]:
        time = index * step
        amplitude = envelope(time + 0.5 * step)
        source = sp.QuasiCylindricalSource(
            jnp.asarray(amplitude * current), jnp.asarray(-step * amplitude * divergence)
        )
        result = solver.advance(time, field, source, jnp.asarray(step))
        return result.field, result.successful

    final, successful = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, jnp.arange(steps))
    )(solver.field_with_charge(jnp.zeros((1, 36, 72), dtype=jnp.complex128)))
    assert bool(jnp.all(successful))
    theta = np.linspace(0.2, np.pi - 0.2, 9)
    directions = np.stack(
        (np.sin(theta) * np.cos(0.3), np.sin(theta) * np.sin(0.3), np.cos(theta)), axis=-1
    )
    far_field = mx.MaxwellFarFieldPlan(directions, jnp.asarray([0.0, 0.0, 1.0]), exterior)
    spectrum = np.asarray(
        far_field.evaluate(solver.huygens_phasors(final)[0]).field_spectrum[..., 0]
    )
    moment = (
        np.exp(1j * _OMEGAS * _T0)
        * (_TAU * np.sqrt(np.pi) / 2j)
        * (
            np.exp(-(_TAU**2) * (_OMEGAS + _OMEGA0) ** 2 / 4.0)
            - np.exp(-(_TAU**2) * (_OMEGAS - _OMEGA0) ** 2 / 4.0)
        )
        * np.pi**1.5
        * size**3
        * np.exp(-((_OMEGAS * size) ** 2) / 4.0)
    )
    exact = (
        -1j
        * _OMEGAS[:, None]
        * moment[:, None]
        * np.sin(theta)[None]
        * np.exp(-1j * _OMEGAS[:, None] * (directions[:, 2] * center)[None])
        / (4.0 * np.pi)
    )
    error = np.linalg.norm(spectrum - exact, axis=1) / np.linalg.norm(exact, axis=1)
    # Fourth-order side-wall interpolation at 9, 7.5, and 6 cells per wavelength.
    assert np.all(error < np.asarray([0.015, 0.025, 0.035]))


# -- absorber, moving window, restart ----------------------------------------------


def _solenoidal_pulse(
    grid: D.pic.QuasiCylindricalGrid, center: float = 2.0, width: float = 0.5
) -> np.ndarray:
    """``E = ∇ × (ẑψ)``, ``ψ = exp(−r²/a² − (z − center)²/width²)`` in mode 0."""
    r = grid.radial_coordinates[:, None]
    z = grid.axial_coordinates[None, :]
    theta_field = (
        2.0 * r / 0.35**2 * np.exp(-(r**2) / 0.35**2 - (z - center) ** 2 / width**2)
    )
    circular = np.zeros((1, grid.radial_count, grid.axial_count, 3), complex)
    circular[0, :, :, 0] = -0.5j * theta_field
    circular[0, :, :, 1] = 0.5j * theta_field
    return circular


def test_radial_damping_absorbs_an_outgoing_pulse_without_breaking_gauss() -> None:
    grid = _grid(3.0, 60, 4.0, 32, 1)
    energies = {}
    configurations: tuple[tuple[str, dict[str, Any]], ...] = (
        ("wall", {}),
        (
            "damped",
            {
                "absorber": "radial-damping",
                "damping": sp.RadialDampingPlan(16, attenuation=1e-6),
            },
        ),
    )
    for name, options in configurations:
        solver = sp.QuasiCylindricalMaxwellPlan(grid, **options).prepare()
        pulse = _solenoidal_pulse(grid)
        initial = solver.circular_field(pulse, np.zeros_like(pulse))
        step = 0.9 * float(solver.stable_step)
        final, (electric, magnetic, absorbed) = _vacuum_run(
            solver, initial, step, int(np.ceil(6.0 / step))
        )
        start = float(solver.field_energy(initial))
        energies[name] = (
            float(solver.field_energy(final)) / start,
            float(jnp.sum(absorbed)) / start,
        )
        assert float(jnp.max(electric)) < 1e-12 * np.max(np.abs(pulse))
        assert float(jnp.max(magnetic)) < 1e-12 * np.max(np.abs(pulse))
    remaining, absorbed = energies["damped"]
    # The grounded wall reflects everything; the layer keeps < 5% and closes the ledger.
    assert energies["wall"][0] > 0.99
    assert remaining < 0.05
    assert remaining + absorbed == pytest.approx(1.0, abs=1e-3)


def test_window_shift_translates_fields_and_pic_continues_with_gauss() -> None:
    grid = _grid(2.0, 24, 6.0, 48, 2)
    solver = sp.QuasiCylindricalMaxwellPlan(grid).prepare()
    # A pulse far from the seams translates exactly by whole nodes.
    pulse = _solenoidal_pulse(grid, 3.0, 0.3)
    pulse = np.concatenate((pulse, np.zeros_like(pulse)))
    field = solver.circular_field(pulse, 0.5 * pulse)
    shifted = solver.shift_window(field, 2, 3)
    np.testing.assert_allclose(
        np.asarray(shifted.electric)[:, :, :-3],
        np.asarray(field.electric)[:, :, 3:],
        atol=1e-12 * np.max(np.abs(pulse)),
    )
    np.testing.assert_allclose(np.asarray(shifted.electric)[:, :, -3:], 0.0, atol=1e-12)
    assert float(shifted.window_offset) == pytest.approx(3 * grid.axial_spacing)
    # Through the PIC moving window: particles and Gauss charge stay paired and
    # the projected field keeps the Gauss law, so stepping continues.
    rng = np.random.default_rng(7)
    count = 32
    positions = _cylinder_points(rng, count, 1.0, 2.0) + np.asarray([0.0, 0.0, 2.0])
    species, transfers = _species(grid, np.ones(count))
    solver = sp.QuasiCylindricalMaxwellPlan(grid).prepare(transfers)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    step = 0.5 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.1, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)
    state = pic.step_detailed(state, step).accepted_state
    window = phx.solver.PICMovingWindowPlan(pic, 2, shift_cells=3)
    moved = window.shift(window.initialize(state))
    assert bool(moved.successful)
    assert float(moved.particle_field_charge_defect) < 1e-12
    result = pic.step_detailed(moved.accepted_state.pic, step)
    assert bool(result.successful)
    scale = np.max(np.abs(np.asarray(result.accepted_state.field.charge)))
    assert float(result.diagnostics.electric_constraint) < 1e-11 * scale


def test_restart_round_trip_continues_bitwise_and_refuses_other_solvers() -> None:
    grid = _grid(2.0, 16, 3.0, 16, 2)
    rng = np.random.default_rng(5)
    count = 16
    positions = _cylinder_points(rng, count, 1.0, 3.0)
    species, transfers = _species(grid, np.ones(count))
    solver = sp.QuasiCylindricalMaxwellPlan(
        grid, variant="galilean", galilean_velocity=0.3
    ).prepare(transfers)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    step = 0.4 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.1, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)
    advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step).accepted_state)
    state = advance(state)
    restored = pic.restore(pic.checkpoint(state))
    for left, right in zip(
        jax.tree.leaves(advance(state)), jax.tree.leaves(advance(restored)), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    other = sp.QuasiCylindricalMaxwellPlan(grid).prepare(transfers)
    with pytest.raises(ValueError, match="another plan"):
        other.restore_component(solver.restart_component(state.field))


def _box() -> Any:
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray([1.0]), sign="positive", measure="time-integral", stop_time=1.0
    )
    return sp.QuasiCylindricalHuygensPlan(
        4, 2, 10, acquisition, mx.HomogeneousMaxwellExterior()
    )


def _antenna() -> Any:
    axis = np.linspace(-1.0, 1.0, 5)
    return sp.QuasiCylindricalAntennaPlan(
        1.0, axis, axis, np.linspace(0.0, 1.0, 3), np.ones((5, 5, 3, 2), complex)
    )


_REFUSALS: dict[str, tuple[dict[str, Any], str]] = {
    "vay-deposition": ({"charge_conservation": "vay-deposition"}, "Cartesian-stencil"),
    "standard-velocity": ({"galilean_velocity": 0.2}, "takes no galilean_velocity"),
    "galilean-superluminal": (
        {"variant": "galilean", "galilean_velocity": 1.5},
        "subluminal",
    ),
    "damping-without-plan": ({"absorber": "radial-damping"}, "RadialDampingPlan"),
    "damping-whole-grid": (
        {"absorber": "radial-damping", "damping": sp.RadialDampingPlan(16)},
        "leaves no interior",
    ),
    "averaged-damping": (
        {
            "variant": "averaged-galilean",
            "galilean_velocity": 0.3,
            "absorber": "radial-damping",
            "damping": sp.RadialDampingPlan(4),
        },
        "averaged-galilean",
    ),
    "galilean-huygens": (
        {"variant": "galilean", "galilean_velocity": 0.3, "observers": ("box",)},
        "standard",
    ),
    "antenna-huygens": ({"antennas": ("antenna",), "observers": ("box",)}, "antennas"),
}


@pytest.mark.parametrize("case", list(_REFUSALS.values()), ids=list(_REFUSALS))
def test_incompatible_configurations_are_refused(
    case: tuple[dict[str, Any], str],
) -> None:
    options, message = case
    resolved = dict(options)
    if "observers" in resolved:
        resolved["observers"] = (_box(),)
    if "antennas" in resolved:
        resolved["antennas"] = (_antenna(),)
    with pytest.raises((ValueError, TypeError), match=message):
        sp.QuasiCylindricalMaxwellPlan(_grid(2.0, 16, 3.0, 16, 2), **resolved)


def test_transfers_on_another_grid_are_refused() -> None:
    support = D.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=3
    ).prepare()
    transfer = D.pic.AzimuthalTransferPlan(_grid(2.0, 16, 3.0, 16, 2)).prepare(support)
    with pytest.raises(ValueError, match="solver's grid"):
        sp.QuasiCylindricalMaxwellPlan(_grid(2.0, 16, 4.0, 16, 2)).prepare((transfer,))


# -- external oracle -----------------------------------------------------------------


def _pinned_fbpic() -> Any:
    path = os.environ.get("PHYDRAX_FBPIC_PYTHON")
    version = os.environ.get("PHYDRAX_FBPIC_PYTHON_VERSION")
    if path is None or version is None or shutil.which(path) is None:
        return None
    return pin_executable(
        shutil.which(path) or path, version=version, license_id="BSD-3-Clause"
    )


def test_laser_wakefield_matches_pinned_fbpic() -> None:
    """Linear-regime LWFA (a₀ = 0.5, k₀ = 5k_p) on-axis wake equals FBPIC's.

    Both codes run the same grid, time step, evenly spaced cold plasma, and
    focused Gaussian laser (FBPIC ``GaussianLaser`` at focus); the on-axis
    mode-0 ``E_z`` behind the laser agrees within particle-sampling differences.
    """
    executable = _pinned_fbpic()
    if executable is None:
        pytest.skip(
            "set PHYDRAX_FBPIC_PYTHON and PHYDRAX_FBPIC_PYTHON_VERSION to a pinned "
            "FBPIC interpreter"
        )
    k0 = 5.0
    wavelength = 2.0 * np.pi / k0
    axial = 120
    length = axial * wavelength / 12.0
    grid = D.pic.QuasiCylindricalGrid(6.0, 48, 0.0, length, axial, 2)
    a0, waist, duration, center = 0.5, 2.5, 1.0, 3.0
    lower, upper, plasma_radius = 5.0, length - 0.2, 4.5
    per_cell = (2, 2, 4)
    spacing_z = (upper - lower) / (
        round((upper - lower) / grid.axial_spacing) * per_cell[0]
    )
    spacing_r = plasma_radius / (round(plasma_radius / grid.radial_spacing) * per_cell[1])
    z = np.arange(lower + 0.5 * spacing_z, upper, spacing_z)
    r = np.arange(0.5 * spacing_r, plasma_radius, spacing_r)
    theta = 2.0 * np.pi / per_cell[2] * np.arange(per_cell[2])
    zz, rr, tt = np.meshgrid(z, r, theta, indexing="ij")
    tt = tt + 2.0 * np.pi / per_cell[2] * (
        (
            np.arange(z.size)[:, None, None] * 0.618
            + np.arange(r.size)[None, :, None] * 0.414
        )
        % 1.0
    )
    positions = np.stack((rr * np.cos(tt), rr * np.sin(tt), zz), axis=-1).reshape(-1, 3)
    weights = (rr * 2.0 * np.pi / per_cell[2] * spacing_r * spacing_z).reshape(-1)
    count = weights.size
    species, transfers = _species(grid, weights, ion_mass_ratio=1.0e9)
    solver = sp.QuasiCylindricalMaxwellPlan(
        grid, charge_conservation="spectral-correction"
    ).prepare(transfers)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    step = 0.9 * float(solver.stable_step)
    steps = int(round(5.5 / step))
    state = pic.initialize((positions, positions), (np.zeros((count, 3)),) * 2, step)
    nodes_r = grid.radial_coordinates[:, None]
    nodes_z = grid.axial_coordinates[None, :]
    laser = np.zeros((2, 48, axial, 3), complex)
    laser[1, :, :, 0] = (
        a0
        * k0
        * np.exp(-(nodes_r**2) / waist**2 - (nodes_z - center) ** 2 / duration**2)
        * np.cos(k0 * (nodes_z - center))
    )
    state = eqx.tree_at(
        lambda value: value.field, state, solver.add_propagating_field(state.field, laser)
    )
    advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step))
    for _ in range(steps):
        result = advance(state)
        assert bool(result.successful)
        state = result.accepted_state
    ours = np.real(np.asarray(state.field.electric[0, 0, :, 2]))
    theirs = sp.fbpic_laser_wakefield(
        sp.FBPICProvider(executable),
        grid,
        density=1.0e24,
        a0=a0,
        wavelength=wavelength,
        waist=waist,
        length=duration,
        center=center,
        plasma_lower=lower,
        plasma_upper=upper,
        plasma_radius=plasma_radius,
        particles_per_cell=per_cell,
        time_step=step,
        steps=steps,
    ).longitudinal_field[:, 0]
    wake = (grid.axial_coordinates > lower + 0.4) & (grid.axial_coordinates < 9.5)
    assert np.max(np.abs(theirs[wake])) > 0.03
    assert np.linalg.norm((ours - theirs)[wake]) < 0.1 * np.linalg.norm(theirs[wake])
    assert np.max(np.abs(ours[wake])) == pytest.approx(
        np.max(np.abs(theirs[wake])), rel=0.1
    )
