"""Source-bound UO2 / stress-relief-annealed Zircaloy closure bundle."""

from __future__ import annotations

from typing import assert_never, final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier
from ...nuclear import (
    FissionGasYieldTable,
    HeliumProductionPlan,
    NuclideInventory,
    NuclideKey,
)
from ...typing import parse
from ._cladding_deformation import (
    ZircaloyCreep,
    ZircaloyIrradiationGrowth,
    ZircaloyPlasticity,
)
from ._cladding_properties import (
    ZircaloyElasticity,
    ZircaloyHeatCapacity,
    ZircaloyThermalConductivity,
    ZircaloyThermalExpansion,
)
from ._cladding_surface import (
    ZircaloyEmissivity,
    ZircaloyHydrogenPickup,
    ZircaloyWatersideCorrosion,
    ZirconiaThermalConductivity,
)
from ._fission_gas import FissionGasModel
from ._fuel_creep import UO2Creep
from ._fuel_eigenstrain import FuelRelocation, UO2Densification, UO2SolidSwelling
from ._fuel_properties import (
    UO2Elasticity,
    UO2Emissivity,
    UO2HeatCapacity,
    UO2ThermalConductivity,
    UO2ThermalExpansion,
)
from ._gap import GapConductanceLaw
from ._helium import HeliumGasModel
from ._layout import STATICALLY_UNSOURCED_SLOTS
from ._radial_power import RadialPowerProfileModel


type FuelApplicabilityPolicy = Literal["strict-source-support", "equation-reproduction"]
type FuelRelocationHistoryPolicy = Literal[
    "fission-activated-permanent-power-recoverable"
]

# Strain-free reference temperature [K] of the UO2 thermal-expansion law.
_FUEL_EXPANSION_REFERENCE_TEMPERATURE_K = 300.0


@final
class EnergyDeposition(StrictModule):
    """Shares of fission energy deposited directly in the fuel and the cladding.

    The remainder ``1 - fuel_fraction - cladding_fraction`` deposits directly in
    the coolant.
    """

    fuel_fraction: Array
    cladding_fraction: Array
    source_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, fuel_fraction: ArrayLike, cladding_fraction: ArrayLike, source_id: str
    ) -> None:
        fractions = np.asarray([fuel_fraction, cladding_fraction], dtype=np.float64)
        if (
            fractions.shape != (2,)
            or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0)
        ):
            raise ValueError("Deposition fractions must be finite nonnegative scalars.")
        if fractions[0] + fractions[1] > 1.0:
            raise ValueError(
                "Fuel and cladding deposition fractions must sum to at most one."
            )
        canonical_identifier(source_id, "EnergyDeposition.source_id")
        self.fuel_fraction = jnp.asarray(fractions[0])
        self.cladding_fraction = jnp.asarray(fractions[1])
        self.source_id = source_id
        # Fingerprint the float64 host inputs: with JAX x64 disabled the leaves are
        # float32, and the identity must not depend on it.
        self.closure_id = canonical_fingerprint(
            {
                "kind": "energy-deposition",
                "fractions": fractions.tolist(),
                "source": source_id,
            }
        )


@final
class CoulombFriction(StrictModule):
    """Coulomb coefficient of pellet-cladding axial contact (dimensionless)."""

    coefficient: Array
    source_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(self, /, *, coefficient: ArrayLike, source_id: str) -> None:
        value = np.asarray(coefficient, dtype=np.float64)
        if value.shape != () or not np.isfinite(value) or value <= 0:
            raise ValueError(
                "CoulombFriction coefficient must be a finite positive scalar."
            )
        canonical_identifier(source_id, "CoulombFriction.source_id")
        self.coefficient = jnp.asarray(value)
        self.source_id = source_id
        self.closure_id = canonical_fingerprint(
            {"kind": "coulomb-friction", "coefficient": float(value), "source": source_id}
        )


@final
class BoreEmissivity(StrictModule):
    """Gray-surface emissivity of the plenum bore over its sourced temperatures [K]."""

    value: Array
    temperature_range: tuple[float, float] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        value: ArrayLike,
        temperature_range: tuple[float, float],
        source_id: str,
    ) -> None:
        emissivity = np.asarray(value, dtype=np.float64)
        low, high = temperature_range
        if emissivity.shape != () or not 0 < emissivity <= 1 or not 0 < low < high:
            raise ValueError(
                "Bore emissivity must lie in (0, 1] over an increasing positive "
                "temperature range."
            )
        canonical_identifier(source_id, "BoreEmissivity.source_id")
        self.value = jnp.asarray(emissivity)
        self.temperature_range = (low, high)
        self.source_id = source_id
        self.closure_id = canonical_fingerprint(
            {
                "kind": "bore-emissivity",
                "value": float(emissivity),
                "temperature_range": self.temperature_range,
                "source": source_id,
            }
        )


@final
class PlenumTemperatureClosure(StrictModule):
    """Plenum gas temperature: coolant outlet temperature plus ``offset`` [K].

    ``source_id`` sources the coolant-outlet closure of channel coolants;
    ``prescribed_source_id`` sources the terminal wall temperature that stands in
    for the outlet temperature under a prescribed wall-temperature boundary.
    """

    offset: Array
    source_id: str = eqx.field(static=True)
    prescribed_source_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, offset: ArrayLike, source_id: str, prescribed_source_id: str
    ) -> None:
        value = np.asarray(offset, dtype=np.float64)
        if value.shape != () or not np.isfinite(value):
            raise ValueError("PlenumTemperatureClosure offset must be a finite scalar.")
        canonical_identifier(source_id, "PlenumTemperatureClosure.source_id")
        canonical_identifier(
            prescribed_source_id, "PlenumTemperatureClosure.prescribed_source_id"
        )
        self.offset = jnp.asarray(value)
        self.source_id = source_id
        self.prescribed_source_id = prescribed_source_id
        self.closure_id = canonical_fingerprint(
            {
                "kind": "coolant-outlet-offset;prescribed-terminal-wall-offset",
                "offset": float(value),
                "source": source_id,
                "prescribed_source": prescribed_source_id,
            }
        )


@final
class HeliumSource(StrictModule):
    """Helium-4 production plan, its fixed decay inventory, and reference volume [m^3]."""

    production: HeliumProductionPlan
    inventory: NuclideInventory
    reference_volume: Array
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        production: HeliumProductionPlan,
        inventory: NuclideInventory,
        reference_volume: ArrayLike,
    ) -> None:
        if not isinstance(production, HeliumProductionPlan):
            raise TypeError("HeliumSource production must be HeliumProductionPlan.")
        if not isinstance(inventory, NuclideInventory):
            raise TypeError("HeliumSource inventory must be NuclideInventory.")
        if production.helium != NuclideKey(2, 4):
            raise ValueError(
                "The fuel helium transport profile requires ground-state helium-4."
            )
        if (
            inventory.network_id != production.inventory_network_id
            or inventory.nuclide_ids != production.inventory_nuclide_ids
        ):
            raise ValueError(
                "Helium production and fixed inventory identities must agree."
            )
        volume = np.asarray(reference_volume, dtype=np.float64)
        if volume.shape != () or not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                "The fixed helium decay inventory requires a finite positive reference volume."
            )
        self.production = production
        self.inventory = inventory
        self.reference_volume = jnp.asarray(volume)
        self.closure_id = canonical_fingerprint(
            {
                "kind": "helium-source",
                "production": production.plan_id,
                "inventory_network": inventory.network_id,
                "inventory_nuclides": inventory.nuclide_ids,
                "inventory": array_tree_fingerprint(inventory),
                "reference_volume": float(volume),
            }
        )


@final
class RelocationHistory(StrictModule):
    """History translation of the relocation correlation and its source."""

    policy: FuelRelocationHistoryPolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, /, *, policy: FuelRelocationHistoryPolicy, source_id: str) -> None:
        self.policy = parse(
            policy, FuelRelocationHistoryPolicy, "RelocationHistory.policy"
        )
        self.source_id = canonical_identifier(source_id, "RelocationHistory.source_id")


@final
class FuelModelBundle(StrictModule):
    """Reference UO2 / Zircaloy closure with mechanistic gas swelling.

    Fission gas uses effective diffusivity, and the bundle holds the single gas
    swelling state: empirical swelling, explicit trap populations, or smeared
    cracking laws would count the same gas twice and are not accepted. Every
    closure is a source-bound law or a closure record carrying its source ID.

    ``applicability="strict-source-support"`` refuses construction while any
    slot of ``STATICALLY_UNSOURCED_SLOTS`` lacks a sourced closure; per-step
    source-range checks apply to every constructed bundle.
    """

    applicability: FuelApplicabilityPolicy = eqx.field(static=True)
    radial_power: RadialPowerProfileModel
    yields: FissionGasYieldTable
    fission_gas: FissionGasModel
    helium: HeliumGasModel
    helium_source: HeliumSource
    fuel_heat: UO2HeatCapacity
    fuel_conductivity: UO2ThermalConductivity
    fuel_expansion: UO2ThermalExpansion
    fuel_elasticity: UO2Elasticity
    fuel_emissivity: UO2Emissivity
    densification: UO2Densification
    solid_swelling: UO2SolidSwelling
    relocation: FuelRelocation
    relocation_history: RelocationHistory
    fuel_creep: UO2Creep
    cladding_heat: ZircaloyHeatCapacity
    cladding_conductivity: ZircaloyThermalConductivity
    cladding_expansion: ZircaloyThermalExpansion
    cladding_elasticity: ZircaloyElasticity
    cladding_growth: ZircaloyIrradiationGrowth
    cladding_creep: ZircaloyCreep
    cladding_plasticity: ZircaloyPlasticity
    cladding_emissivity: ZircaloyEmissivity
    corrosion: ZircaloyWatersideCorrosion
    hydrogen: ZircaloyHydrogenPickup
    oxide_conductivity: ZirconiaThermalConductivity
    gap: GapConductanceLaw
    bore_emissivity: BoreEmissivity
    plenum_temperature: PlenumTemperatureClosure
    deposition: EnergyDeposition
    # ``None`` is frictionless axial contact: a zero Coulomb limit.
    friction: CoulombFriction | None
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        applicability: FuelApplicabilityPolicy,
        radial_power: RadialPowerProfileModel,
        fission_gas: FissionGasModel,
        helium: HeliumGasModel,
        helium_source: HeliumSource,
        relocation_history: RelocationHistory,
        gap: GapConductanceLaw,
        bore_emissivity: BoreEmissivity,
        plenum_temperature: PlenumTemperatureClosure,
        deposition: EnergyDeposition,
        friction: CoulombFriction | None = None,
        cladding_creep: ZircaloyCreep | None = None,
        cladding_plasticity: ZircaloyPlasticity | None = None,
    ) -> None:
        applicability_ = parse(applicability, FuelApplicabilityPolicy, "applicability")
        match applicability_:
            case "strict-source-support":
                if STATICALLY_UNSOURCED_SLOTS:
                    raise ValueError(
                        "Strict source support requires sourced closures for: "
                        + ", ".join(STATICALLY_UNSOURCED_SLOTS)
                        + '. Use applicability="equation-reproduction" for '
                        "synthetic or equation-verification work."
                    )
            case "equation-reproduction":
                pass
            case unsupported:
                assert_never(unsupported)
        if not isinstance(radial_power, RadialPowerProfileModel):
            raise TypeError("radial_power must be RadialPowerProfileModel.")
        if not isinstance(fission_gas, FissionGasModel) or not isinstance(
            gap, GapConductanceLaw
        ):
            raise TypeError("fission_gas and gap must be source-bound law objects.")
        if not isinstance(helium, HeliumGasModel):
            raise TypeError("helium must be HeliumGasModel.")
        if not isinstance(helium_source, HeliumSource):
            raise TypeError("helium_source must be HeliumSource.")
        if not isinstance(relocation_history, RelocationHistory):
            raise TypeError("relocation_history must be RelocationHistory.")
        if not isinstance(bore_emissivity, BoreEmissivity):
            raise TypeError("bore_emissivity must be BoreEmissivity.")
        if not isinstance(plenum_temperature, PlenumTemperatureClosure):
            raise TypeError("plenum_temperature must be PlenumTemperatureClosure.")
        if not isinstance(deposition, EnergyDeposition):
            raise TypeError("deposition must be EnergyDeposition.")
        if friction is not None and not isinstance(friction, CoulombFriction):
            raise TypeError("friction must be CoulombFriction or None.")
        if cladding_creep is not None and not isinstance(cladding_creep, ZircaloyCreep):
            raise TypeError("cladding_creep must be ZircaloyCreep or None.")
        if cladding_plasticity is not None and not isinstance(
            cladding_plasticity, ZircaloyPlasticity
        ):
            raise TypeError("cladding_plasticity must be ZircaloyPlasticity or None.")
        yields = FissionGasYieldTable.endf_b_viii_0()
        if not np.array_equal(
            np.asarray(radial_power.energy_per_fission),
            np.asarray(yields.energy_per_fission),
        ):
            raise ValueError(
                "Radial buildup and exposure must use the same canonical fission-energy table."
            )
        self.applicability = applicability_
        self.radial_power = radial_power
        self.yields = yields
        self.fission_gas = fission_gas
        self.helium = helium
        self.helium_source = helium_source
        self.fuel_heat = UO2HeatCapacity()
        self.fuel_conductivity = UO2ThermalConductivity()
        self.fuel_expansion = UO2ThermalExpansion(
            reference_temperature=_FUEL_EXPANSION_REFERENCE_TEMPERATURE_K
        )
        self.fuel_elasticity = UO2Elasticity()
        self.fuel_emissivity = UO2Emissivity()
        self.densification = UO2Densification()
        self.solid_swelling = UO2SolidSwelling()
        self.relocation = FuelRelocation()
        self.relocation_history = relocation_history
        self.fuel_creep = UO2Creep()
        self.cladding_heat = ZircaloyHeatCapacity()
        self.cladding_conductivity = ZircaloyThermalConductivity()
        self.cladding_expansion = ZircaloyThermalExpansion()
        self.cladding_elasticity = ZircaloyElasticity()
        self.cladding_growth = ZircaloyIrradiationGrowth()
        self.cladding_creep = (
            ZircaloyCreep() if cladding_creep is None else cladding_creep
        )
        self.cladding_plasticity = (
            ZircaloyPlasticity() if cladding_plasticity is None else cladding_plasticity
        )
        self.cladding_emissivity = ZircaloyEmissivity()
        self.corrosion = ZircaloyWatersideCorrosion()
        self.hydrogen = ZircaloyHydrogenPickup()
        self.oxide_conductivity = ZirconiaThermalConductivity()
        self.gap = gap
        self.bore_emissivity = bore_emissivity
        self.plenum_temperature = plenum_temperature
        self.deposition = deposition
        self.friction = friction
        self.bundle_id = self._canonical_identity()

    def _canonical_identity(self) -> str:
        laws = (
            self.fuel_heat,
            self.fuel_conductivity,
            self.fuel_expansion,
            self.fuel_elasticity,
            self.fuel_emissivity,
            self.densification,
            self.solid_swelling,
            self.relocation,
            self.fuel_creep,
            self.cladding_heat,
            self.cladding_conductivity,
            self.cladding_expansion,
            self.cladding_elasticity,
            self.cladding_growth,
            self.cladding_creep,
            self.cladding_plasticity,
            self.cladding_emissivity,
            self.corrosion,
            self.hydrogen,
            self.oxide_conductivity,
        )
        return canonical_fingerprint(
            {
                "kind": "uo2-sra-zircaloy-reference-bundle",
                "applicability": self.applicability,
                "radial_power": self.radial_power.law_id,
                "gas": self.fission_gas.law_id,
                "helium": self.helium.law_id,
                "helium_source": self.helium_source.closure_id,
                "closures": tuple(law.law_id for law in laws),
                "validity": tuple(law.validity.range_id for law in laws),
                "relocation_history": (
                    self.relocation_history.policy,
                    self.relocation_history.source_id,
                ),
                "gap": self.gap.law_id,
                "bore_emissivity": self.bore_emissivity.closure_id,
                "plenum_temperature": self.plenum_temperature.closure_id,
                "deposition": self.deposition.closure_id,
                "friction": None if self.friction is None else self.friction.closure_id,
            }
        )
