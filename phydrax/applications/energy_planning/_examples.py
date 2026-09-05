# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Small complete examples, using one consistent carrier-amount/time basis."""

from ...units import ENERGY, KILOGRAM, SI_REFERENCE_SYSTEM_ID, UnitDefinition
from ._spec import (
    BalancePoint,
    Carrier,
    Chronology,
    Converter,
    ConverterPort,
    Demand,
    EnergySystem,
    Horizon,
    Inventory,
    InventoryBoundary,
    Source,
)


def electricity_heat_storage_example(*, exact: bool = False) -> EnergySystem:
    """Two physical intervals; explicit ambient heat and independent store E/P."""
    return EnergySystem(
        Chronology((Horizon("day", (1.0, 2.0)),)),
        (Carrier("electricity"), Carrier("heat"), Carrier("ambient", environmental=True)),
        (
            BalancePoint("grid", "electricity"),
            BalancePoint("building", "heat"),
            BalancePoint("environment", "ambient"),
        ),
        sources=(
            Source("grid-import", "grid", 8.0, marginal_cost=(1.0, 5.0), emissions=0.2),
            Source("ambient-input", "environment", 20.0),
        ),
        demands=(
            Demand("electric-load", "grid", (1.0, 1.0)),
            Demand("space-heat", "building", (3.0, 6.0)),
        ),
        inventories=(
            Inventory(
                "heat-store",
                "building",
                4.0,
                4.0,
                2.0,
                (InventoryBoundary("day"),),
                retention=0.99,
                exclusive=exact,
            ),
        ),
        converters=(
            Converter(
                "heat-pump",
                "grid",
                "input",
                (
                    ConverterPort("grid", -1.0),
                    ConverterPort("environment", -2.0),
                    ConverterPort("building", 3.0),
                ),
                4.0,
            ),
        ),
    )


def electricity_hydrogen_example(*, exact: bool = False) -> EnergySystem:
    """Electrolyzer with useful heat output, hydrogen storage and a fuel cell.

    Hydrogen is in kg with an explicitly selected 120 MJ/kg lower-heating-value
    basis. Electricity/heat amounts are MWh; all rates are amounts per physical
    chronology time unit. Explicit unit factors keep the optimization well scaled.
    """
    lhv = 120e6
    mwh = UnitDefinition("MWh", ENERGY, SI_REFERENCE_SYSTEM_ID, 3600000000)
    kg_per_mwh = float(mwh.scale_to_reference) / lhv
    return EnergySystem(
        Chronology((Horizon("day", (1.0, 1.0)),)),
        (
            Carrier("electricity", mwh),
            Carrier("hydrogen", KILOGRAM, energy_content=lhv),
            Carrier("heat", mwh),
        ),
        (
            BalancePoint("electric", "electricity"),
            BalancePoint("hydrogen", "hydrogen"),
            BalancePoint("heat", "heat", spill_capacity=10.0),
        ),
        sources=(Source("import", "electric", 10.0, marginal_cost=(1.0, 10.0)),),
        demands=(Demand("electric-load", "electric", (0.0, 1.0)),),
        inventories=(
            Inventory(
                "hydrogen-tank",
                "hydrogen",
                5.0 * kg_per_mwh,
                5.0 * kg_per_mwh,
                2.0 * kg_per_mwh,
                (InventoryBoundary("day"),),
                exclusive=exact,
            ),
        ),
        converters=(
            Converter(
                "electrolyzer",
                "electric",
                "input",
                (
                    ConverterPort("electric", -1.0),
                    ConverterPort("hydrogen", 0.7 * kg_per_mwh),
                    ConverterPort("heat", 0.2),
                ),
                5.0,
            ),
            Converter(
                "fuel-cell",
                "electric",
                "output",
                (
                    ConverterPort("hydrogen", -2.0 * kg_per_mwh),
                    ConverterPort("electric", 1.0),
                    ConverterPort("heat", 0.8),
                ),
                2.0,
            ),
        ),
    )
