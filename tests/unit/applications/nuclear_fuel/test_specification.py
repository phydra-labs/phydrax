from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._strict import StrictModule
from phydrax.applications.nuclear_fuel import (
    admit_fuel_rod,
    IrradiationHistory,
    SingleChannelCoolant,
)
from phydrax.applications.nuclear_fuel._bundles import (
    BoreEmissivity,
    CoulombFriction,
    EnergyDeposition,
    FuelApplicabilityPolicy,
    FuelModelBundle,
    PlenumTemperatureClosure,
    RelocationHistory,
)
from phydrax.applications.nuclear_fuel._specification import (
    CladdingSpecification,
    FuelRodSpecification,
    PelletSpecification,
    RodSpecification,
)
from phydrax.units import KELVIN, KILOPASCAL, MILLIMETER
from tests._support.nuclear_fuel import (
    synthetic_model_bundle,
    synthetic_rod_specification,
)


# Closure-record constructors keyed by the source-ID input each one validates.
_CLOSURE_SOURCE_INPUTS: dict[str, Callable[[str], StrictModule]] = {
    "RelocationHistory.source_id": lambda source: RelocationHistory(
        policy="fission-activated-permanent-power-recoverable", source_id=source
    ),
    "PlenumTemperatureClosure.source_id": lambda source: PlenumTemperatureClosure(
        offset=0.0, source_id=source, prescribed_source_id="declared-plenum"
    ),
    "PlenumTemperatureClosure.prescribed_source_id": lambda source: (
        PlenumTemperatureClosure(
            offset=0.0, source_id="declared-plenum", prescribed_source_id=source
        )
    ),
    "EnergyDeposition.source_id": lambda source: EnergyDeposition(
        fuel_fraction=0.95, cladding_fraction=0.025, source_id=source
    ),
    "BoreEmissivity.source_id": lambda source: BoreEmissivity(
        value=0.3, temperature_range=(293.0, 2500.0), source_id=source
    ),
    "CoulombFriction.source_id": lambda source: CoulombFriction(
        coefficient=0.5, source_id=source
    ),
}


def _pellet() -> PelletSpecification:
    return PelletSpecification(
        radius=0.004,
        height=0.01,
        density_fraction=0.95,
        uranium_235_weight_fraction=0.04,
        grain_radius=5.0e-6,
        resintering_density_change=0.0,
        roughness=1.0e-6,
    )


def test_fabrication_units_convert_before_geometry_validation() -> None:
    pellet = PelletSpecification(
        radius=4.0,
        height=10.0,
        density_fraction=0.95,
        uranium_235_weight_fraction=0.04,
        grain_radius=5.0e-6,
        resintering_density_change=0.0,
        roughness=1.0e-6,
        units={"radius": MILLIMETER, "height": MILLIMETER},
    )
    clad = CladdingSpecification(
        inner_radius=4.1,
        outer_radius=4.7,
        roughness=1.0e-6,
        units={"inner_radius": MILLIMETER, "outer_radius": MILLIMETER},
    )
    rod = RodSpecification(
        stack_length=1000.0,
        plenum_length=200.0,
        fill_pressure=2000.0,
        units={
            "stack_length": MILLIMETER,
            "plenum_length": MILLIMETER,
            "fill_pressure": KILOPASCAL,
        },
    )
    design = FuelRodSpecification(pellet, clad, rod)
    np.testing.assert_allclose(design.cold_gap, 0.0001, rtol=0.0, atol=1e-18)
    assert rod.fill_pressure == 2.0e6
    assert rod.stack_length == 1.0
    assert pellet.to_record() == _pellet().to_record()


def test_fabrication_dimension_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="matching dimensions"):
        RodSpecification(
            stack_length=1.0,
            plenum_length=0.1,
            fill_pressure=2.0e6,
            units={"fill_pressure": KELVIN},
        )


def test_fabrication_open_porosity_cannot_exceed_total_porosity() -> None:
    with pytest.raises(ValueError, match="Open porosity"):
        PelletSpecification(
            radius=0.004,
            height=0.01,
            density_fraction=0.95,
            open_porosity_fraction=0.06,
            uranium_235_weight_fraction=0.04,
            grain_radius=5.0e-6,
            resintering_density_change=0.0,
            roughness=1.0e-6,
        )


def test_fabrication_noncommunicating_gap_is_refused() -> None:
    clad = CladdingSpecification(
        inner_radius=0.0039,
        outer_radius=0.0047,
        roughness=1.0e-6,
    )
    rod = RodSpecification(stack_length=1.0, plenum_length=0.1, fill_pressure=2.0e6)
    with pytest.raises(ValueError, match="gap must be open"):
        FuelRodSpecification(_pellet(), clad, rod)


@pytest.mark.parametrize(
    "composition",
    [(0.9, 0.0, 0.0, 0.0, 0.0), (1.1, -0.1, 0.0, 0.0, 0.0)],
    ids=["inventory-fraction-deficit", "negative-inventory"],
)
def test_fabrication_invalid_gas_composition_is_refused(
    composition: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="nonnegative, sum to one"):
        RodSpecification(
            stack_length=1.0,
            plenum_length=0.1,
            fill_pressure=2.0e6,
            fill_gas_mole_fractions=composition,
        )


def test_unadmitted_fuel_form_is_refused_at_runtime() -> None:
    with pytest.raises(ValueError, match="form"):
        PelletSpecification(
            radius=0.004,
            height=0.01,
            density_fraction=0.95,
            uranium_235_weight_fraction=0.04,
            grain_radius=5e-6,
            resintering_density_change=0.0,
            roughness=1e-6,
            form="mox",  # ty: ignore[invalid-argument-type]
        )


def test_fabrication_smooth_pellet_is_refused() -> None:
    with pytest.raises(ValueError, match="roughness must be finite and positive"):
        PelletSpecification(
            radius=0.004,
            height=0.01,
            density_fraction=0.95,
            uranium_235_weight_fraction=0.04,
            grain_radius=5.0e-6,
            resintering_density_change=0.0,
            roughness=0.0,
        )


def test_fabrication_smooth_cladding_is_refused() -> None:
    with pytest.raises(ValueError, match="roughness must be finite and positive"):
        CladdingSpecification(inner_radius=0.0041, outer_radius=0.0047, roughness=0.0)


def test_fabrication_cold_work_reports_closed_interval() -> None:
    with pytest.raises(ValueError, match=r"cold_work must be a fraction in \[0, 1\]\."):
        CladdingSpecification(
            inner_radius=0.0041, outer_radius=0.0047, roughness=1.0e-6, cold_work=1.5
        )


def _bundle(
    *,
    applicability: FuelApplicabilityPolicy = "equation-reproduction",
    deposition: EnergyDeposition | None = None,
    friction: CoulombFriction | None = None,
) -> FuelModelBundle:
    return synthetic_model_bundle(
        mode_count=2,
        applicability=applicability,
        deposition=deposition,
        friction=friction,
    )


def test_bundle_friction_is_a_coefficient_not_an_energy_fraction() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        # The deposition fractions sum to 0.925 + 0.025 < 1, while the friction
        # coefficient exceeds one: it is not part of the energy partition.
        bundle = _bundle(
            deposition=EnergyDeposition(
                fuel_fraction=0.925,
                cladding_fraction=0.025,
                source_id="declared-energy-partition",
            ),
            friction=CoulombFriction(coefficient=1.2, source_id="declared-friction"),
        )
        assert bundle.friction is not None
        assert float(bundle.friction.coefficient) == 1.2
        assert _bundle().friction is None


@pytest.mark.parametrize(
    "coefficient", [0.0, -0.1, float("nan")], ids=["zero", "negative", "nan"]
)
def test_coulomb_friction_refuses_nonpositive_or_nonfinite_coefficient(
    coefficient: float,
) -> None:
    with pytest.raises(ValueError, match="CoulombFriction coefficient"):
        CoulombFriction(coefficient=coefficient, source_id="declared-friction")


def test_energy_deposition_refuses_fractions_summing_above_one() -> None:
    with pytest.raises(ValueError, match="sum to at most one"):
        EnergyDeposition(
            fuel_fraction=0.98,
            cladding_fraction=0.025,
            source_id="declared-energy-partition",
        )


def test_strict_source_support_refuses_statically_unsourced_closures() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Strict source support requires sourced closures for: "
                "relocation_history, plenum_temperature. "
                'Use applicability="equation-reproduction"'
            ),
        ):
            _bundle(applicability="strict-source-support")
        assert _bundle().applicability == "equation-reproduction"


@pytest.mark.parametrize("source_input", tuple(_CLOSURE_SOURCE_INPUTS))
@pytest.mark.parametrize("source_id", ["", "   ", " padded-source "])
def test_closures_refuse_noncanonical_source_ids(
    source_input: str, source_id: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(source_input)):
        _CLOSURE_SOURCE_INPUTS[source_input](source_id)


def test_closure_refuses_non_string_source_id() -> None:
    with pytest.raises(TypeError, match=re.escape("CoulombFriction.source_id")):
        CoulombFriction(coefficient=0.5, source_id=7)  # ty: ignore[invalid-argument-type]


def test_single_channel_admission_refuses_stagnant_coolant() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        specification = synthetic_rod_specification()
        times = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
        power = np.full((3, 1), 100.0, dtype=np.float64)
        # Forced flow stops at the last breakpoint.
        stagnant = IrradiationHistory(
            times=times,
            zone_edges=(0.0, specification.rod.stack_length),
            linear_power=power,
            fast_flux=power * 5.0e13,
            coolant_inlet_temperature=np.full((3,), 560.0, dtype=np.float64),
            coolant_pressure=np.full((3,), 15.5e6, dtype=np.float64),
            coolant_mass_flux=np.asarray((3000.0, 3000.0, 0.0), dtype=np.float64),
        )
        coolant = SingleChannelCoolant(
            hydraulic_diameter=0.0118, flow_area_per_rod=8.8e-5
        )
        with pytest.raises(ValueError, match="positive coolant_mass_flux"):
            admit_fuel_rod(
                specification,
                stagnant,
                coolant,
                bundle=synthetic_model_bundle(mode_count=2),
            )
