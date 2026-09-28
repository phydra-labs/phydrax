#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import ColdPlasmaDielectric


mx = phx.solver.maxwell

_CELLS = 8
_SPACING = 1.0 / _CELLS
_REGION = mx.MaxwellMaterialRegion((0, 0, 0), (_CELLS, _CELLS, _CELLS))


def _bridge(count: int = _CELLS, *, periodic: bool = True) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return phx.discretization.StructuredCochainBridge(grid)


def _audit(constitutive: Any = None, fraction: float = 0.9) -> Any:
    runtime = phx.solver.CompatibleMaxwellPlan(
        _bridge(), constitutive=constitutive
    ).prepare()
    return mx.CompatibleMaxwellDispersionAudit(
        runtime, _REGION, fraction * float(runtime.cfl_limit)
    )


def _positive_branches(result: Any) -> np.ndarray:
    omega = np.asarray(result.angular_frequencies)[0]
    return omega.real[omega.real > 1e-6]


def _electron_plasma(cyclotron: float, collision: float = 0.0) -> Any:
    return mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
        jnp.asarray([1.0]),
        jnp.asarray([[0.0, 0.0, cyclotron]]),
        collision_frequency=collision,
    )


def _reference_plasma(cyclotron: float) -> ColdPlasmaDielectric:
    """SI electron plasma with ω_p = 1 rad/s scaled units and Ω_e = cyclotron·ω_p."""
    scale = ElectromagneticScaleContract.si()
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    permittivity = float(scale.vacuum_permittivity)
    omega_p = 1.0e9
    density = omega_p**2 * permittivity * mass / charge**2
    field = -cyclotron * omega_p * mass / charge
    return ColdPlasmaDielectric(
        scale,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
    )


def test_vacuum_audit_matches_yee_leapfrog_dispersion() -> None:
    audit = _audit()
    wavevectors = np.asarray([[2.0, 1.0, 0.5], [10.0, 5.0, 0.0], [0.0, 0.0, 20.0]])
    result = audit.dispersion(wavevectors)
    dt = float(audit.step_size)
    for k, omega in zip(wavevectors, np.asarray(result.angular_frequencies), strict=True):
        spatial = np.sum((2.0 / _SPACING * np.sin(0.5 * k * _SPACING)) ** 2)
        expected = 2.0 / dt * np.arcsin(0.5 * dt * np.sqrt(spatial))
        np.testing.assert_allclose(np.sort(omega.real)[-2:], expected, rtol=1e-11)
        np.testing.assert_allclose(omega.imag, 0.0, atol=1e-9)
    assert bool(jnp.all(result.stable))
    assert float(audit.stencil_defect) < 1e-12


def test_audit_multiplier_equals_runtime_single_mode_phase_advance() -> None:
    plasma = _electron_plasma(-0.6, collision=0.05)
    runtime = phx.solver.CompatibleMaxwellPlan(_bridge(), constitutive=plasma).prepare()
    step = 0.9 * float(runtime.stable_dt)
    audit = mx.CompatibleMaxwellDispersionAudit(runtime, _REGION, step)
    wavevector = np.asarray([2.0 * np.pi, 0.0, 2.0 * np.pi])
    result = audit.dispersion(wavevector[None, :])
    multiplier = complex(np.asarray(result.multipliers)[0, -1])
    mode = np.asarray(result.modes)[0, :, -1]
    state = audit.bloch_state(wavevector, mode)
    stepped = runtime.leapfrog_step(0.0, state, step)
    np.testing.assert_allclose(
        stepped.primary.electric_displacement,
        multiplier * state.primary.electric_displacement,
        atol=1e-9 * float(jnp.max(jnp.abs(state.primary.electric_displacement))),
    )
    np.testing.assert_allclose(
        stepped.primary.magnetic_flux,
        multiplier * state.primary.magnetic_flux,
        atol=1e-9 * float(jnp.max(jnp.abs(state.primary.magnetic_flux))),
    )
    np.testing.assert_allclose(
        stepped.auxiliary.material.current,
        multiplier * state.auxiliary.material.current,
        atol=1e-9 * float(jnp.max(jnp.abs(state.auxiliary.material.current))),
    )
    assert abs(multiplier) < 1.0


def test_lorentz_ade_dispersion_converges_to_continuum_permittivity() -> None:
    poles = mx.MaxwellLorentzPoles([2.0], [0.0], [3.0])
    material = mx.LorentzDrudeMaxwellConstitutivePlan(poles, permittivity_infinity=1.5)
    wavevector = np.asarray([[1.5, 0.0, 0.0]])
    # Yee spatial symbol isolates the time discretization of field and ADE.
    symbol = (2.0 / _SPACING * np.sin(0.5 * wavevector[0, 0] * _SPACING)) ** 2
    errors: dict[float, float] = {}
    for fraction in (0.4, 0.1):
        audit = _audit(material, fraction)
        worst = 0.0
        for omega in _positive_branches(audit.dispersion(wavevector)):
            permittivity = 1.5 + 3.0 / (4.0 - omega**2)
            if abs(permittivity) > 0.1:
                worst = max(worst, abs(symbol / omega**2 / permittivity - 1.0))
        errors[fraction] = worst
    # Second order in Δt: a fourfold smaller step reduces the error sixteenfold.
    assert errors[0.1] < 2e-4
    assert errors[0.1] < 0.1 * errors[0.4]


def test_audit_detects_the_courant_limit() -> None:
    corner = np.full((1, 3), np.pi / _SPACING)
    below = _audit(fraction=0.999).dispersion(corner)
    above = _audit(fraction=1.001).dispersion(corner)
    assert bool(below.stable[0])
    assert not bool(above.stable[0])
    assert float(above.spectral_radius[0]) > 1.01


def test_cherenkov_regime_masks_separate_numerical_from_physical_emission() -> None:
    speed = jnp.asarray([0.95, 0.0, 0.0])
    frequencies = jnp.asarray([1.0, 16.0])
    vacuum = mx.CherenkovRegimePlan(_audit(), speed, frequencies, 4).evaluate()
    assert not bool(jnp.any(vacuum.physical_emission))
    assert not bool(jnp.any(vacuum.numerical_emission[0]))
    assert bool(jnp.all(vacuum.numerical_only[1]))
    dielectric = mx.CherenkovRegimePlan(
        _audit(mx.DiagonalMaxwellConstitutivePlan(permittivity=4.0)),
        speed,
        frequencies[:1],
        4,
    ).evaluate()
    # Degenerate isotropic Booker roots carry O(sqrt(eps)) rounding.
    np.testing.assert_allclose(dielectric.continuum_index, 2.0, rtol=1e-7)
    expected = np.arccos(1.0 / (0.95 * 2.0))
    np.testing.assert_allclose(
        dielectric.physical_cone_angle[0], expected, rtol=1e-9, atol=1e-9
    )
    assert bool(jnp.all(dielectric.physical_emission))
    assert not bool(jnp.any(dielectric.numerical_only))
    numerical = np.asarray(dielectric.numerical_cone_angle[0])
    np.testing.assert_allclose(numerical[np.isfinite(numerical)], expected, rtol=5e-3)


def test_dispersion_audit_refusals() -> None:
    nonuniform = phx.discretization.TensorGridPlan(
        (
            phx.discretization.NonuniformCellAxisSpec(
                np.concatenate((np.linspace(0.0, 0.5, 5), np.linspace(0.55, 1.0, 5)))
            ),
            phx.discretization.UniformCellAxisSpec(_CELLS),
            phx.discretization.UniformCellAxisSpec(_CELLS),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    runtime = phx.solver.CompatibleMaxwellPlan(
        phx.discretization.StructuredCochainBridge(nonuniform)
    ).prepare()
    with pytest.raises(ValueError, match="uniform structured axes"):
        mx.CompatibleMaxwellDispersionAudit(runtime, _REGION, 0.01)
    bridge = _bridge(10, periodic=False)
    runtime = phx.solver.CompatibleMaxwellPlan(bridge).prepare()
    with pytest.raises(ValueError, match="must span more than"):
        mx.CompatibleMaxwellDispersionAudit(
            runtime, mx.MaxwellMaterialRegion((2, 2, 2), (7, 7, 7)), 0.01
        )
    absorbing = phx.solver.CompatibleMaxwellPlan(
        bridge, pml=mx.MaxwellCPMLPlan(2)
    ).prepare()
    with pytest.raises(ValueError, match="intersects the absorbing CPML"):
        mx.CompatibleMaxwellDispersionAudit(
            absorbing, mx.MaxwellMaterialRegion((1, 1, 1), (9, 9, 9)), 0.01
        )
    projected = phx.solver.CompatibleMaxwellPlan(
        _bridge(), magnetic_constraint=mx.MaxwellMagneticConstraintPolicy("project")
    ).prepare()
    with pytest.raises(ValueError, match="magnetic projection is global"):
        mx.CompatibleMaxwellDispersionAudit(projected, _REGION, 0.01)
    layout = mx.MaxwellCochainLayout(_bridge())
    permittivity = jnp.ones((layout.electric_count,)).at[0].set(2.0)
    heterogeneous = phx.solver.CompatibleMaxwellPlan(
        _bridge(),
        constitutive=mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity),
    ).prepare()
    with pytest.raises(ValueError, match="not translation invariant"):
        mx.CompatibleMaxwellDispersionAudit(heterogeneous, _REGION, 0.01)
    planar = phx.discretization.StructuredCochainBridge(
        phx.discretization.TensorGridPlan(
            tuple(
                phx.discretization.UniformCellAxisSpec(_CELLS, periodic=True)
                for _ in range(2)
            ),
            axis_names=("x", "y"),
        ).prepare(jnp.asarray([[0.0] * 2, [1.0] * 2]))
    )
    tez = phx.solver.CompatibleMaxwellPlan(planar, polarization="tez").prepare()
    audit = mx.CompatibleMaxwellDispersionAudit(
        tez, mx.MaxwellMaterialRegion((0, 0), (_CELLS, _CELLS)), 0.01
    )
    with pytest.raises(ValueError, match="full_3d"):
        mx.CherenkovRegimePlan(audit, jnp.asarray([0.5, 0.0, 0.0]), jnp.asarray([1.0]), 2)


@pytest.mark.parametrize("angle", [0.0, 0.5 * np.pi], ids=["parallel", "perpendicular"])
def test_magnetized_plasma_continuum_matches_appleton_hartree(angle: float) -> None:
    audit = _audit(_electron_plasma(-0.6), 0.05)
    wavenumber = 0.3
    wavevector = wavenumber * np.asarray([[np.sin(angle), 0.0, np.cos(angle)]])
    reference = _reference_plasma(-0.6)
    branches = [
        omega
        for omega in _positive_branches(audit.dispersion(wavevector))
        # The flat longitudinal branch at ω_p carries no transverse wave.
        if abs(omega - 1.0) > 1e-3
    ]
    assert len(branches) >= 2
    for omega in branches:
        numerical = (wavenumber / omega) ** 2
        roots = np.asarray(
            reference.refractive_indices(omega * 1.0e9, angle).n_squared
        ).real
        root = int(np.argmin(np.abs(roots - numerical)))

        def mismatch(frequency: float) -> float:
            value = reference.refractive_indices(frequency * 1.0e9, angle).n_squared
            return float(np.asarray(value).real[root]) - (wavenumber / frequency) ** 2

        # Continuum frequency of the same branch at the same k by bisection.
        lower, upper = 0.99 * omega, 1.01 * omega
        assert mismatch(lower) * mismatch(upper) < 0.0
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if mismatch(lower) * mismatch(middle) <= 0.0:
                upper = middle
            else:
                lower = middle
        # O((kh)^2) Yee error at kh = 0.0375.
        np.testing.assert_allclose(omega, 0.5 * (lower + upper), rtol=5e-4)


def test_magnetized_plasma_whistler_branch_and_cutoffs() -> None:
    audit = _audit(_electron_plasma(-2.0), 0.05)
    reference = _reference_plasma(-2.0)
    cutoffs = reference.characteristic_frequencies()
    expected = np.sort(
        np.concatenate(
            (
                np.asarray(cutoffs.right_cutoffs).real,
                np.asarray(cutoffs.left_cutoffs).real,
                np.asarray(cutoffs.plasma_cutoffs).real,
            )
        )
    )
    expected = expected[expected > 0.0] / 1.0e9
    at_rest = _positive_branches(audit.dispersion(np.zeros((1, 3))))
    for value in expected:
        assert np.min(np.abs(at_rest - value)) < 1e-4 * value
    parallel = 0.5 * np.asarray([[0.0, 0.0, 1.0]])
    left_cutoff = float(np.min(expected))
    whistler = [
        omega
        for omega in _positive_branches(audit.dispersion(parallel))
        if omega < left_cutoff
    ]
    assert len(whistler) == 1
    roots = np.asarray(reference.refractive_indices(whistler[0] * 1.0e9, 0.0).n_squared)
    np.testing.assert_allclose(np.max(roots.real), (0.5 / whistler[0]) ** 2, rtol=2e-3)


def test_negative_index_band_carries_a_backward_wave() -> None:
    material = mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([0.0], [0.0], [1.0]),
        magnetic_poles=mx.MaxwellLorentzPoles([0.6], [0.0], [0.4]),
    )
    audit = _audit(material, 0.1)

    def band(k: float) -> float:
        values = _positive_branches(audit.dispersion(np.asarray([[k, 0.0, 0.0]])))
        inside = values[(values > 0.6) & (values < np.sqrt(0.76) - 1e-3)]
        assert inside.size >= 1
        return float(inside[0])

    near, far = band(0.2), band(0.4)
    assert far < near
    permittivity = 1.0 - 1.0 / near**2
    permeability = 1.0 + 0.4 / (0.36 - near**2)
    assert permittivity < 0.0 and permeability < 0.0
    np.testing.assert_allclose((0.2 / near) ** 2, permittivity * permeability, rtol=2e-3)
    evidence = mx.CherenkovRegimePlan(
        audit, jnp.asarray([0.9, 0.0, 0.0]), jnp.asarray([near]), 1
    ).evaluate()
    np.testing.assert_allclose(
        evidence.continuum_index[0].real,
        -np.sqrt(permittivity * permeability),
        rtol=1e-7,
    )


def test_localized_drude_pole_is_exactly_zero_outside_its_support() -> None:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    support = np.asarray(bridge.cochain.coordinates[1])[:, 0] > 0.5
    plasma = jnp.where(jnp.asarray(support), 2.0, 0.0)[None, :]
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        constitutive=mx.drude_maxwell_constitutive(plasma, jnp.asarray([0.2])),
    ).prepare()
    displacement = jnp.sin(jnp.arange(layout.electric_count, dtype=jnp.float64))
    state = runtime.initialize(electric_displacement=displacement)
    for index in range(20):
        state = runtime.leapfrog_step(
            index * 0.5 * runtime.stable_dt, state, 0.5 * runtime.stable_dt
        )
    material = state.auxiliary.material
    assert bool(jnp.all(material.polarization[:, ~support] == 0.0))
    assert bool(jnp.all(material.velocity[:, ~support] == 0.0))
    assert float(jnp.max(jnp.abs(material.polarization[:, support]))) > 0.0
    assert bool(jnp.isfinite(runtime.energy(state)))


def test_lossy_media_close_power_balance_and_are_passive() -> None:
    rng = np.random.default_rng(3)
    bridge = _bridge(3, periodic=False)
    layout = mx.MaxwellCochainLayout(bridge)
    lorentz = mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([1.0, 0.0], [0.3, 0.1], [0.8, 0.5]),
        magnetic_poles=mx.MaxwellLorentzPoles([1.5], [0.2], [0.4]),
    )
    plasma = mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
        jnp.asarray([1.0, 0.3]),
        jnp.asarray([[0.0, 0.2, -0.7], [0.0, 0.0, 0.1]]),
        collision_frequency=jnp.asarray([0.3, 0.1]),
    )
    source = mx.MaxwellElectricCurrentSourcePlan(jnp.asarray([3]), jnp.asarray([1.0]))
    for material in (lorentz, plasma):
        runtime = phx.solver.CompatibleMaxwellPlan(
            bridge, constitutive=material, sources=(source,)
        ).prepare()
        auxiliary = runtime.constitutive.initialize_state()
        leaves = [
            jnp.asarray(rng.normal(size=leaf.shape))
            for leaf in jax.tree_util.tree_leaves(auxiliary)
        ]
        auxiliary = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(auxiliary), leaves
        )
        state = runtime.pack(
            jnp.asarray(rng.normal(size=(layout.electric_count,))),
            bridge.exterior_derivative(
                1, jnp.asarray(rng.normal(size=(layout.electric_count,)))
            ),
            material_state=auxiliary,
        )
        dissipation = runtime.material_dissipation(state)
        assert float(dissipation) > 0.0
        residual = runtime.power_balance_residual(0.0, state)
        scale = float(dissipation) + abs(float(runtime.source_power(0.0, state)))
        assert abs(float(residual)) < 1e-10 * scale
    lossy = _audit(_electron_plasma(-0.6, collision=0.2), 0.5)
    sampled = lossy.dispersion(
        np.asarray([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [25.0, 25.0, 25.0]])
    )
    assert bool(jnp.all(sampled.stable))
    omega = jnp.linspace(0.2, 3.0, 7)
    prepared = lorentz.prepare(bridge.cochain, layout)
    for value in omega:
        assert bool(
            jnp.all(jnp.imag(prepared.continuum_relative_permittivity(value)) > 0.0)
        )
        assert bool(
            jnp.all(jnp.imag(prepared.continuum_relative_permeability(value)) > 0.0)
        )


def test_exponential_gyration_has_no_cyclotron_resonance_shift() -> None:
    audit = _audit(_electron_plasma(-40.0), 0.9)
    phase = float(audit.cyclotron_step_phase[0])
    assert phase > 1.0
    np.testing.assert_allclose(audit.cyclotron_resonance_shift, 0.0, atol=1e-10)
    cayley = 2.0 / float(audit.step_size) * np.arctan(0.5 * phase) - 40.0
    assert abs(cayley) > 1.0
