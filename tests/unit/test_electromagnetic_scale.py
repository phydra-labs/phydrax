from fractions import Fraction
from math import isclose, pi

import pytest

from phydrax import (
    DimensionalScaleContract,
    ElectromagneticScaleContract,
    RelativityScaleContract,
)
from phydrax.units import (
    CHARGE,
    COULOMB,
    KILOGRAM,
    LENGTH,
    METER,
    SECOND,
    TIME,
    UnitDefinition,
)


# CODATA 2022 recommended values (NIST SP 961, May 2024).
_CODATA_2022_EXACT = {
    "elementary_charge": "1.602176634e-19",
    "electron_mass": "9.1093837139e-31",
    "vacuum_permittivity": "8.8541878188e-12",
}
_CODATA_2022_DERIVED = {
    "vacuum_permeability": 1.25663706127e-6,
    "vacuum_impedance": 376.730313412,
    "inverse_fine_structure": 137.035999177,
    "classical_electron_radius": 2.8179403205e-15,
}


def test_si_realization_matches_codata_2022() -> None:
    scale = ElectromagneticScaleContract.si()

    assert scale.constant_set_id == "codata-2022"
    assert scale.charge_unit == COULOMB
    assert scale.speed_of_light == Fraction(299_792_458)
    for name, value in _CODATA_2022_EXACT.items():
        assert getattr(scale, name) == Fraction(value)
    assert isclose(
        float(scale.vacuum_permeability),
        _CODATA_2022_DERIVED["vacuum_permeability"],
        rel_tol=1e-10,
    )
    assert isclose(
        float(scale.vacuum_impedance),
        _CODATA_2022_DERIVED["vacuum_impedance"],
        rel_tol=1e-10,
    )
    assert isclose(
        float(scale.classical_electron_radius),
        _CODATA_2022_DERIVED["classical_electron_radius"],
        rel_tol=1e-10,
    )
    # hbar = h / (2 pi) from exact SI h; CODATA 2022 alpha^-1 = 137.035999177(21).
    assert abs(1.0 / float(scale.fine_structure) - 137.035999177) <= 21e-9


def test_vacuum_constants_satisfy_maxwell_identity() -> None:
    scale = ElectromagneticScaleContract.si()

    assert (
        scale.vacuum_permeability * scale.vacuum_permittivity * scale.speed_of_light**2
        == 1
    )
    assert (
        scale.vacuum_impedance**2 == scale.vacuum_permeability / scale.vacuum_permittivity
    )


def test_schwinger_field_matches_independent_float_formula() -> None:
    scale = ElectromagneticScaleContract.si()
    mass = 9.1093837139e-31
    charge = 1.602176634e-19
    light = 299_792_458.0
    hbar = 6.62607015e-34 / (2.0 * pi)

    expected = mass**2 * light**3 / (charge * hbar)
    assert isclose(float(scale.schwinger_field), expected, rel_tol=1e-12)
    assert isclose(float(scale.schwinger_field), 1.3232855e18, rel_tol=1e-7)


def _laser_code_units(
    scale: ElectromagneticScaleContract, length_si: Fraction
) -> ElectromagneticScaleContract:
    """Electron-normalized code units: c = e = m_e = 1, length unit ``length_si``."""
    light = scale.speed_of_light
    time_si = length_si / light
    mass_si = scale.electron_mass
    charge_si = scale.elementary_charge
    length = UnitDefinition("L0", LENGTH, "si", length_si)
    mass = UnitDefinition("m_e", KILOGRAM.dimension, "si", mass_si)
    time = UnitDefinition("T0", TIME, "si", time_si)
    charge = UnitDefinition("q_e", CHARGE, "si", charge_si)
    relativity = scale.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(length, mass, time),
        charge,
        gravitational_constant=relativity.gravitational_constant
        * mass_si
        * time_si**2
        / length_si**3,
        speed_of_light=light * time_si / length_si,
        reduced_planck_constant=scale.reduced_planck_constant
        * time_si
        / (mass_si * length_si**2),
        boltzmann_constant=relativity.boltzmann_constant
        * time_si**2
        / (mass_si * length_si**2),
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=scale.vacuum_permittivity
        * length_si**3
        * mass_si
        / (charge_si**2 * time_si**2),
        constant_set_id="codata-2022",
    )


def test_quantum_parameter_is_invariant_under_unit_change() -> None:
    si = ElectromagneticScaleContract.si()
    code = _laser_code_units(si, Fraction(8, 10**7))

    assert code.speed_of_light == 1
    assert isclose(float(code.fine_structure), float(si.fine_structure), rel_tol=1e-14)

    gamma = 2.0e3
    field_si = 3.0e14
    unit_si, dimension = code.unit_si_map()["electric_field"]
    assert dimension == (1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0)
    field_code = field_si / unit_si
    chi_si = gamma * field_si / float(si.schwinger_field)
    chi_code = gamma * field_code / float(code.schwinger_field)
    assert isclose(chi_code, chi_si, rel_tol=1e-12)
    # With c = e = m_e = 1, E_S = m_e^2 c^3 / (e hbar) reduces to 1 / hbar.
    assert isclose(
        float(code.schwinger_field),
        1.0 / float(code.reduced_planck_constant),
        rel_tol=1e-14,
    )


def test_unit_si_map_reports_openpmd_dimensions() -> None:
    mapping = ElectromagneticScaleContract.si().unit_si_map()

    assert mapping["magnetic_field"] == (1.0, (0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0))
    assert mapping["charge"] == (1.0, (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    assert mapping["current_density"] == (1.0, (-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

    code = _laser_code_units(ElectromagneticScaleContract.si(), Fraction(1, 10**6))
    code_map = code.unit_si_map()
    assert isclose(code_map["length"][0], 1.0e-6, rel_tol=1e-15)
    assert isclose(code_map["time"][0], 1.0e-6 / 299_792_458.0, rel_tol=1e-15)
    assert isclose(code_map["velocity"][0], 299_792_458.0, rel_tol=1e-15)


def test_dict_round_trip_preserves_fingerprint_and_refuses_tampering() -> None:
    code = _laser_code_units(ElectromagneticScaleContract.si(), Fraction(1, 10**6))
    for scale in (ElectromagneticScaleContract.si(), code):
        restored = ElectromagneticScaleContract.from_dict(scale.to_dict())
        assert restored.scale_id == scale.scale_id
        assert restored.vacuum_permittivity == scale.vacuum_permittivity

    payload = ElectromagneticScaleContract.si().to_dict()
    payload["constant_set_id"] = "codata-2018"
    with pytest.raises(ValueError, match="fingerprint"):
        ElectromagneticScaleContract.from_dict(payload)
    assert code.scale_id != ElectromagneticScaleContract.si().scale_id


def test_reference_system_mismatch_is_refused() -> None:
    relativity = RelativityScaleContract.si()
    foreign_charge = UnitDefinition("q", CHARGE, "code")
    with pytest.raises(ValueError, match="reference system"):
        ElectromagneticScaleContract(
            relativity, foreign_charge, 1, 1, 1, constant_set_id="code"
        )
    with pytest.raises(ValueError, match="charge dimension"):
        ElectromagneticScaleContract(relativity, METER, 1, 1, 1, constant_set_id="code")
    code_scale = ElectromagneticScaleContract(
        RelativityScaleContract(
            DimensionalScaleContract(
                UnitDefinition("l", LENGTH, "code"),
                UnitDefinition("m", KILOGRAM.dimension, "code"),
                UnitDefinition("t", TIME, "code"),
            ),
            1,
            1,
            1,
            1,
        ),
        foreign_charge,
        1,
        1,
        1,
        constant_set_id="dimensionless",
    )
    with pytest.raises(ValueError, match="SI system"):
        code_scale.unit_si_map()


@pytest.mark.parametrize(
    ("charge", "mass", "permittivity"),
    [(0, 1, 1), (1, -1, 1), (1, 1, 0), (1, 1, float("nan"))],
)
def test_nonpositive_constants_are_refused(
    charge: float, mass: float, permittivity: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ElectromagneticScaleContract(
            RelativityScaleContract.si(),
            COULOMB,
            charge,
            mass,
            permittivity,
            constant_set_id="codata-2022",
        )


def test_hbar_free_relativity_scale_is_refused() -> None:
    relativity = RelativityScaleContract(
        DimensionalScaleContract(METER, KILOGRAM, SECOND),
        "6.67430e-11",
        299_792_458,
        1,
        1,
        quantum_constants_explicit=False,
    )
    with pytest.raises(ValueError, match="hbar"):
        ElectromagneticScaleContract(
            relativity, COULOMB, 1, 1, 1, constant_set_id="codata-2022"
        )


def test_code_unit_constants_reproduce_si_dimensionless_ratios() -> None:
    si = ElectromagneticScaleContract.si()
    code = _laser_code_units(si, Fraction(1, 10**6))
    # Classical electron radius over reduced Compton wavelength equals alpha.
    for scale in (si, code):
        compton = scale.reduced_planck_constant / (
            scale.electron_mass * scale.speed_of_light
        )
        assert isclose(
            float(scale.classical_electron_radius / compton),
            float(scale.fine_structure),
            rel_tol=1e-15,
        )
    assert isclose(
        float(code.classical_electron_radius) * 1.0e-6,
        float(si.classical_electron_radius),
        rel_tol=1e-14,
    )
