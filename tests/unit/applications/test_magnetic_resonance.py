#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import magnetic_resonance as mr


def _density(state: Any) -> Any:
    state = jnp.asarray(state, dtype="complex128")
    return jnp.outer(state, jnp.conj(state))


def _test_isotope(name: Any, gamma_hz_t: Any) -> Any:
    return mr.ResonanceIsotope(name, "nucleus", 0.5, 2.0 * math.pi * gamma_hz_t)


def test_magnetic_resonance_scenario_1() -> None:
    isotope = _test_isotope("L", 100.0)
    profile = mr.ExactSingleCrystalNMRProfile(
        mr.MagneticResonanceSpinSystem(
            (mr.SpinSite("L", isotope),),
            # ty: ignore[invalid-argument-type]
            (0.0, 0.0, 1.0),
        )
    )
    prepared = profile.prepare()
    transverse = _density(jnp.asarray([1.0, 1.0]) / jnp.sqrt(2.0))
    fid = mr.acquire_fid(
        prepared,
        transverse,
        mr.AcquisitionPlan(1.0 / 1024.0, 1024),
    )
    instrument = mr.apply_instrument(
        fid,
        mr.InstrumentPlan(
            gain=2.0,
            receiver_phase_rad=0.5 * math.pi,
            frequency_offset_hz=10.0,
        ),
    )
    spectrum = instrument.spectrum
    peak_hz = spectrum.frequency_hz[jnp.argmax(jnp.abs(spectrum.amplitude))]

    assert fid.valid
    assert jnp.isclose(peak_hz, 90.0, atol=1.0e-5)
    assert jnp.max(fid.trace_residuals) < 1.0e-8
    assert fid.step_unitarity_residual < 1.0e-8
    assert jnp.allclose(instrument.detected_fid, -2.0j * fid.signal)
    assert jnp.array_equal(instrument.bare_fid.signal, fid.signal)
    isotope = _test_isotope("R", 100.0)
    prepared = mr.ExactSingleCrystalNMRProfile(
        mr.MagneticResonanceSpinSystem(
            (mr.SpinSite("R", isotope),),
            # ty: ignore[invalid-argument-type]
            (0.0, 0.0, 0.0),
        )
    ).prepare()
    pulse = mr.prepare_pulse_sequence(
        prepared,
        mr.FixedPulseSequence(
            # ty: ignore[invalid-argument-type]
            (0.0, 0.5),
            # ty: ignore[invalid-argument-type]
            ((0.01, 0.0, 0.0),),
        ),
    )
    result = mr.evolve_density_exact(prepared, pulse, _density((1.0, 0.0)))

    assert result.evidence.valid
    assert jnp.isclose(jnp.real(result.density_matrices[-1, 1, 1]), 1.0, atol=1.0e-6)
    assert jnp.isclose(jnp.real(result.density_matrices[-1, 0, 0]), 0.0, atol=1.0e-6)
    isotope_a = _test_isotope("A", 10.0)
    isotope_x = _test_isotope("X", 4.0)
    coupling_hz = 2.5
    system = mr.MagneticResonanceSpinSystem(
        (
            mr.SpinSite("A", isotope_a),
            mr.SpinSite("X", isotope_x),
        ),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0),
        interactions=(mr.ScalarJCoupling("A", "X", coupling_hz),),
    )
    dense = mr.ExactSingleCrystalNMRProfile(system).prepare().dense_hamiltonian_rad_s
    spin = mr.spin_operators(0.5)
    identity = spin.identity
    expected = (
        -isotope_a.gyromagnetic_ratio_rad_s_t * jnp.kron(spin.z, identity)
        - isotope_x.gyromagnetic_ratio_rad_s_t * jnp.kron(identity, spin.z)
        + 2.0
        * math.pi
        * coupling_hz
        * (jnp.kron(spin.x, spin.x) + jnp.kron(spin.y, spin.y) + jnp.kron(spin.z, spin.z))
    )

    assert jnp.allclose(dense, expected, atol=1.0e-8)


def test_magnetic_resonance_scenario_2() -> None:
    isotope = _test_isotope("Hahn", 1.0)
    system = mr.MagneticResonanceSpinSystem(
        (mr.SpinSite("Hahn", isotope),),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0),
    )
    reference_system = mr.MagneticResonanceSpinSystem(
        (mr.SpinSite("Hahn", isotope),),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 0.0),
    )
    prepared = mr.ExactSingleCrystalNMRProfile(system).prepare()
    reference = mr.ExactSingleCrystalNMRProfile(reference_system).prepare()
    t90 = 0.25 / 1000.0
    t180 = 0.5 / 1000.0
    tau = 0.125
    times = (
        0.0,
        t90,
        t90 + tau,
        t90 + tau + t180,
        t90 + 2.0 * tau + t180,
    )
    fields = (
        (1000.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1000.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    # ty: ignore[invalid-argument-type]
    sequence = mr.FixedPulseSequence(times, fields)
    pulse = mr.prepare_pulse_sequence(prepared, sequence)
    reference_pulse = mr.prepare_pulse_sequence(reference, sequence)
    initial = _density((1.0, 0.0))
    result = mr.evolve_density_exact(prepared, pulse, initial)
    reference_result = mr.evolve_density_exact(reference, reference_pulse, initial)
    receiver = mr.receiver_operator(prepared)
    echo = jnp.trace(result.density_matrices[-1] @ receiver)
    reference_echo = jnp.trace(reference_result.density_matrices[-1] @ receiver)

    assert result.evidence.valid
    assert reference_result.evidence.valid
    assert jnp.allclose(echo, reference_echo, atol=2.0e-3)
    with pytest.raises(ValueError, match="integer or half-integer"):
        mr.ResonanceIsotope("bad", "nucleus", 0.7, 1.0)

    spin_half = mr.MagneticResonanceSpinSystem(
        (mr.SpinSite("q", mr.HYDROGEN_1),),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0),
        interactions=(
            mr.QuadrupolarInteraction(
                "q",
                # ty: ignore[invalid-argument-type]
                (
                    (1.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="I >= 1"):
        mr.prepare_spin_system(spin_half)

    dimension_guarded = mr.MagneticResonanceSpinSystem(
        (
            mr.SpinSite("a", mr.HYDROGEN_1),
            mr.SpinSite("b", mr.HYDROGEN_1),
        ),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 0.0),
        resource_policy=mr.MagneticResonanceResourcePolicy(
            maximum_hilbert_dimension=3,
            maximum_density_elements=16,
        ),
    )
    with pytest.raises(ValueError, match="D=4"):
        mr.prepare_spin_system(dimension_guarded)

    guarded = mr.MagneticResonanceSpinSystem(
        (
            mr.SpinSite("a", mr.HYDROGEN_1),
            mr.SpinSite("b", mr.HYDROGEN_1),
        ),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 0.0),
        resource_policy=mr.MagneticResonanceResourcePolicy(
            maximum_hilbert_dimension=8,
            maximum_density_elements=15,
        ),
    )
    with pytest.raises(ValueError, match=r"D\^2=16"):
        mr.prepare_spin_system(guarded)
    electron_system = mr.MagneticResonanceSpinSystem(
        (mr.SpinSite("e", mr.ELECTRON),),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0e-3),
    )
    epr = mr.ExactSingleCrystalEPRProfile(electron_system).prepare()
    electron_levels = np.linalg.eigvalsh(np.asarray(epr.dense_hamiltonian_rad_s))
    assert electron_levels[1] - electron_levels[0] == pytest.approx(
        abs(mr.ELECTRON.gyromagnetic_ratio_rad_s_t) * 1.0e-3
    )

    muon_system = mr.MagneticResonanceSpinSystem(
        (mr.SpinSite("mu", mr.POSITIVE_MUON),),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0e-3),
    )
    musr = mr.ExactStaticSiteMuonSpinRotationProfile(muon_system).prepare()
    assert musr.system.sites[0].isotope.gyromagnetic_ratio_rad_s_t > 0.0
    with pytest.raises(ValueError, match="nuclear spin sites only"):
        mr.ExactSingleCrystalNMRProfile(electron_system)
    mixed = mr.MagneticResonanceSpinSystem(
        (
            mr.SpinSite("e", mr.ELECTRON),
            mr.SpinSite("mu", mr.POSITIVE_MUON),
        ),
        # ty: ignore[invalid-argument-type]
        (0.0, 0.0, 1.0e-3),
    )
    with pytest.raises(ValueError, match="electron and nuclear"):
        mr.ExactSingleCrystalEPRProfile(mixed)
