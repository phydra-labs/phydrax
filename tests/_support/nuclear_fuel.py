"""Explicit synthetic fuel-rod setups and closed-form references for rod-level tests.

Every rod is a declared numerical design, not an IFPE/FUMEX rod: a 0.3 m stack
of 8.19 mm diameter UO2 pellets in a Zircaloy tube, a single axial zone, a
prescribed cladding wall temperature, and a synthetic fast flux proportional to
linear power. Source-law coefficients keep their owners' provenance; the
helium production network, spectrum, energy partition, and plenum closures are
synthetic declarations with source IDs that say so.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.applications.nuclear_fuel import (
    admit_fuel_rod,
    AthermalRelease,
    BoreEmissivity,
    CladdingSpecification,
    CoulombFriction,
    EnergyDeposition,
    FissionGasModel,
    FuelApplicabilityPolicy,
    FuelGeometryReduction,
    FuelModelBundle,
    FuelRodSpecification,
    GapConductanceLaw,
    GasAccommodationLaw,
    HeliumGasModel,
    HeliumSource,
    HeliumTransportTable,
    IrradiationHistory,
    PelletSpecification,
    PlenumTemperatureClosure,
    prepare_fuel_rod,
    PreparedFuelRod,
    PrescribedWallTemperature,
    RadialPowerProfileModel,
    RelocationHistory,
    RodSpecification,
)
from phydrax.materials import PropertyValidityRange
from phydrax.nuclear import (
    AVOGADRO_PER_MOL,
    FissionGasYieldTable,
    FISSIONING_NUCLIDES,
    HeliumProductionPlan,
    InventoryTransition,
    NuclearDataProvenance,
    NuclideInventory,
    NuclideKey,
)
from phydrax.qualification import ReferenceArtifactManifest


PELLET_RADIUS = 4.095e-3  # [m]
PELLET_HEIGHT = 10.0e-3  # [m]
STACK_LENGTH = 0.3  # [m]
# Synthetic fast flux per unit linear power [n/(m^2 s) per W/m].
FAST_FLUX_PER_LINEAR_POWER = 5.0e13
ATMOSPHERIC_COOLANT_PRESSURE = 0.103e6  # [Pa]
COOLANT_MASS_FLUX = 3000.0  # [kg/(m^2 s)]


def synthetic_helium_source() -> HeliumSource:
    """Declared alpha-decay and ternary-fission helium network, not evaluated data.

    One fixed U-238 -> Th-234 + He-4 transition produces 1e10 atoms/(m^3 s) in a
    1 m^3 reference volume; every fissioning nuclide has ternary yield 1e-4.
    """
    parent, daughter, helium = NuclideKey(92, 238), NuclideKey(90, 234), NuclideKey(2, 4)
    reference_volume, parent_amount = 1.0, 1.0
    decay_rate = 1e10 * reference_volume / (parent_amount * AVOGADRO_PER_MOL)
    ternary_yields = {name: 1e-4 for name in FISSIONING_NUCLIDES}
    payload = canonical_json(
        {
            "kind": "synthetic-numerical-helium-production",
            "parent": parent.nuclide_id,
            "daughter": daughter.nuclide_id,
            "helium": helium.nuclide_id,
            "ternary_yields": ternary_yields,
            "parent_amount_mol": parent_amount,
            "reference_volume_m3": reference_volume,
            "declared_decay_rate_s": decay_rate,
            "target_decay_rate_atoms_per_m3_s": 1e10,
            "inventory_policy": "fixed-declared-actinide-approximation-no-depletion",
        }
    ).encode("utf-8")
    reference = ReferenceArtifactManifest(
        "synthetic-numerical-helium-production",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic-owned-numerical-input",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="synthetic-fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic-numerical-verification:not-evaluated-decay-data",),
    )
    reference.verify_bytes(payload)
    provenance = NuclearDataProvenance(
        reference,
        "synthetic://nuclear-fuel/numerical-helium-production",
        "synthetic-numerical-verification",
        "declared-control",
        "not-evaluated-physical-decay-data",
    )
    transition = InventoryTransition(
        "synthetic-U238-to-Th234-plus-He4-control",
        parent,
        ((daughter, 1.0), (helium, 1.0)),
        decay_rate,
        np.zeros((1,), dtype=np.float64),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        provenance,
    )
    inventory = NuclideInventory(
        np.asarray((parent_amount,), dtype=np.float64),
        0.0,
        (parent.nuclide_id,),
        canonical_fingerprint(
            {
                "kind": "synthetic-fixed-alpha-network",
                "transition": transition.transition_id,
                "nuclides": (parent.nuclide_id,),
            }
        ),
    )
    plan = HeliumProductionPlan(
        ternary_yields, inventory, (transition,), provenance, helium=helium
    )
    return HeliumSource(
        production=plan, inventory=inventory, reference_volume=reference_volume
    )


def synthetic_rod_specification(
    *, pellet_radius: float = PELLET_RADIUS, density_fraction: float = 0.95
) -> FuelRodSpecification:
    """3.5 wt% U-235 pellets in a cold-worked tube with a 0.1 MPa helium fill."""
    return FuelRodSpecification(
        PelletSpecification(
            radius=pellet_radius,
            height=PELLET_HEIGHT,
            density_fraction=density_fraction,
            uranium_235_weight_fraction=0.035,
            grain_radius=5.0e-6,
            resintering_density_change=100.0,
            roughness=1.0e-6,
            open_porosity_fraction=0.005,
        ),
        CladdingSpecification(
            inner_radius=4.18e-3, outer_radius=4.75e-3, roughness=0.5e-6, cold_work=0.5
        ),
        RodSpecification(
            stack_length=STACK_LENGTH,
            plenum_length=0.03,
            fill_pressure=ATMOSPHERIC_COOLANT_PRESSURE,
            fill_temperature=300.0,
        ),
    )


def synthetic_model_bundle(
    *,
    mode_count: int,
    scattering_cross_section_barn: float = 5.0,
    applicability: FuelApplicabilityPolicy = "equation-reproduction",
    deposition: EnergyDeposition | None = None,
    friction: CoulombFriction | None = None,
) -> FuelModelBundle:
    """Source-law bundle with synthetic spectrum, helium, and closure declarations.

    ``mode_count`` is the intragranular gas modal resolution. Helium transport
    is a slow, weakly soluble synthetic table over 40 radial cells; the
    athermal escape surface is the uncracked pellet surface. ``deposition=None``
    deposits 95 % of fission energy in the fuel and 2.5 % in the cladding;
    ``friction=None`` is frictionless axial contact.
    """
    surface = 2.0 / PELLET_RADIUS + 2.0 / PELLET_HEIGHT
    transport_source = "synthetic-numerical-verification:helium-transport"
    return FuelModelBundle(
        applicability=applicability,
        radial_power=RadialPowerProfileModel(
            energy_per_fission=FissionGasYieldTable.endf_b_viii_0().energy_per_fission,
            scattering_cross_section_barn=scattering_cross_section_barn,
            scattering_source_id="synthetic-numerical-verification:not-spectrum-data",
            form_depth_unit_m=1.0e-3,
        ),
        fission_gas=FissionGasModel(
            mode_count=mode_count,
            athermal=AthermalRelease(
                geometric_surface_to_volume=surface, total_surface_to_volume=surface
            ),
        ),
        helium=HeliumGasModel(
            radial_cell_count=40,
            transport=HeliumTransportTable(
                temperature=(293.0, 2500.0),
                diffusivity=(1e-20, 1e-20),
                henry_constant=(1e12, 1e12),
                validity=PropertyValidityRange(
                    {"temperature": (293.0, 2500.0, "K")}, source_id=transport_source
                ),
                source_id=transport_source,
            ),
        ),
        helium_source=synthetic_helium_source(),
        relocation_history=RelocationHistory(
            policy="fission-activated-permanent-power-recoverable",
            source_id="declared-numerical-translation:not-FAST-caller",
        ),
        gap=GapConductanceLaw(accommodation=GasAccommodationLaw.gap_comparison()),
        bore_emissivity=BoreEmissivity(
            value=0.3,
            temperature_range=(293.0, 2500.0),
            source_id="synthetic-numerical-verification:gray-bore",
        ),
        plenum_temperature=PlenumTemperatureClosure(
            offset=0.0,
            source_id="declared-numerical-plenum:coolant-outlet-zero-offset",
            prescribed_source_id="declared-numerical-plenum:terminal-wall-proxy",
        ),
        deposition=EnergyDeposition(
            fuel_fraction=0.95,
            cladding_fraction=0.025,
            source_id="synthetic-numerical-verification:declared-energy-partition",
        )
        if deposition is None
        else deposition,
        friction=friction,
    )


def wall_temperature_history(
    *,
    times: Sequence[float],
    linear_power: Sequence[float],
    wall_temperature: Sequence[float],
    stack_length: float = STACK_LENGTH,
) -> tuple[IrradiationHistory, PrescribedWallTemperature]:
    """Single-zone history [s, W/m, K] whose coolant boundary is the wall temperature.

    The coolant inlet temperature equals the wall temperature at atmospheric
    pressure and forced flow, and fast flux is proportional to linear power.
    """
    time = np.asarray(times, dtype=np.float64)
    power = np.asarray(linear_power, dtype=np.float64)[:, None]
    wall = np.asarray(wall_temperature, dtype=np.float64)
    history = IrradiationHistory(
        times=time,
        zone_edges=np.asarray((0.0, stack_length), dtype=np.float64),
        linear_power=power,
        fast_flux=power * FAST_FLUX_PER_LINEAR_POWER,
        coolant_inlet_temperature=wall,
        coolant_pressure=np.full(time.shape, ATMOSPHERIC_COOLANT_PRESSURE),
        coolant_mass_flux=np.full(time.shape, COOLANT_MASS_FLUX),
    )
    return history, PrescribedWallTemperature(time, wall[:, None])


def prepare_rod(
    specification: FuelRodSpecification,
    bundle: FuelModelBundle,
    history: IrradiationHistory,
    coolant: PrescribedWallTemperature,
    /,
    *,
    reduction: FuelGeometryReduction,
    fuel_cells: int,
    cladding_cells: int,
    quadrature_order: int = 2,
) -> PreparedFuelRod:
    """Admit with frictionless axial contact and prepare one cell per axial zone."""
    return prepare_fuel_rod(
        admit_fuel_rod(
            specification,
            history,
            coolant,
            bundle=bundle,
            reduction=reduction,
            axial_interaction="frictionless",
        ),
        fuel_cells=fuel_cells,
        cladding_cells=cladding_cells,
        quadrature_order=quadrature_order,
    )


def prepare_low_power_startup_rod(
    *,
    reduction: FuelGeometryReduction = "layered-radial",
    fuel_cells: int = 2,
    cladding_cells: int = 2,
) -> PreparedFuelRod:
    """Zero-power wall cycling 300-305 K for 2 s, then 100 W/m for 1 s.

    The coarse two-mode gas model and two-cell bodies keep each coupled step
    cheap while every thermal, mechanical, gas, and coolant history is live.
    Prepared rods are immutable, so one float64 preparation per argument set is
    shared by every caller in the process; call it with x64 enabled.
    """
    return _low_power_startup_rod(reduction, fuel_cells, cladding_cells)


@functools.cache
def _low_power_startup_rod(
    reduction: FuelGeometryReduction, fuel_cells: int, cladding_cells: int, /
) -> PreparedFuelRod:
    specification = synthetic_rod_specification()
    history, coolant = wall_temperature_history(
        times=(0.0, 1.0, 2.0, 3.0),
        linear_power=(0.0, 0.0, 100.0, 100.0),
        wall_temperature=(300.0, 305.0, 300.0, 305.0),
        stack_length=specification.rod.stack_length,
    )
    return prepare_rod(
        specification,
        synthetic_model_bundle(mode_count=2),
        history,
        coolant,
        reduction=reduction,
        fuel_cells=fuel_cells,
        cladding_cells=cladding_cells,
    )


def solid_cylinder_temperature(
    radius: ArrayLike,
    /,
    *,
    outer_radius: float,
    surface_temperature: float,
    volumetric_heat: float,
    conductivity: float,
) -> NDArray[np.float64]:
    """Steady solid cylinder with uniform source and insulated axis [K].

    Carslaw & Jaeger, Conduction of Heat in Solids:
    ``T(r) = T_s + q''' (a^2 - r^2) / (4 k)``.
    """
    r = np.asarray(radius, dtype=np.float64)
    return surface_temperature + volumetric_heat * (outer_radius**2 - r**2) / (
        4.0 * conductivity
    )


def closed_end_thick_cylinder(
    radius: ArrayLike,
    /,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_pressure: float,
    outer_pressure: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[NDArray[np.float64], float]:
    """Radial displacement [m] and axial strain of a closed-end elastic tube.

    Timoshenko & Goodier, Theory of Elasticity: ``sigma_rr = A - B/r^2`` and
    ``sigma_tt = A + B/r^2`` with ``A = (p_i a^2 - p_o b^2)/(b^2 - a^2)`` and
    ``B = a^2 b^2 (p_i - p_o)/(b^2 - a^2)``; closed ends give ``sigma_zz = A``.
    """
    a2, b2 = inner_radius**2, outer_radius**2
    lame_a = (inner_pressure * a2 - outer_pressure * b2) / (b2 - a2)
    lame_b = a2 * b2 * (inner_pressure - outer_pressure) / (b2 - a2)
    r = np.asarray(radius, dtype=np.float64)
    nu, young = poisson_ratio, youngs_modulus
    displacement = ((1 - 2 * nu) * lame_a * r + (1 + nu) * lame_b / r) / young
    return displacement, (1 - 2 * nu) * lame_a / young
