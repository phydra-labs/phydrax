#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._ablating_material import PorousAblatingMaterialPlan
from ...equations._ionized_gas import (
    IonizedMultitemperatureEulerSystem,
    IonizedMultitemperatureNavierStokesSystem,
)
from ...equations._plasma_transport import AmbipolarPlasmaTransportPlan
from ...solver._aerothermal_material import (
    ConjugateAerothermalInterfacePlan,
    FixedConnectivityRecessionPlan,
)
from ...solver._continuum_dsmc import (
    DynamicHybridOwnershipPlan,
    FixedContinuumDSMCInterfacePlan,
)
from ...solver._dsmc_runtime import DSMCProductionPlan
from ...solver._plasma_electrostatic import ElectrostaticPlasmaCouplingPlan
from ...solver._radiation_balance_law import MultigroupRadiationMatterProcessPlan
from ...solver._thermochemical_source import FixedWorkThermochemicalSourcePlan
from ._wall import ReactingPlasmaWallPlan


class IonizedContinuumProfile(StrictModule, NonTrainableState):
    system: IonizedMultitemperatureEulerSystem | IonizedMultitemperatureNavierStokesSystem
    thermochemical_source: FixedWorkThermochemicalSourcePlan
    plasma_transport: AmbipolarPlasmaTransportPlan
    electrostatic: ElectrostaticPlasmaCouplingPlan | None
    profile_id: str = eqx.field(static=True)

    def __init__(
        self, system, thermochemical_source, plasma_transport, /, *, electrostatic=None
    ):
        if (
            not isinstance(
                system,
                (
                    IonizedMultitemperatureEulerSystem,
                    IonizedMultitemperatureNavierStokesSystem,
                ),
            )
            or not isinstance(thermochemical_source, FixedWorkThermochemicalSourcePlan)
            or not isinstance(plasma_transport, AmbipolarPlasmaTransportPlan)
            or plasma_transport.schema.schema_id != system.thermodynamics.schema.schema_id
            or (
                electrostatic is not None
                and (
                    not isinstance(electrostatic, ElectrostaticPlasmaCouplingPlan)
                    or electrostatic.system.system_id != system.system_id
                )
            )
        ):
            raise ValueError("Ionized continuum profile components do not match.")
        self.system = system
        self.thermochemical_source = thermochemical_source
        self.plasma_transport = plasma_transport
        self.electrostatic = electrostatic
        self.profile_id = canonical_fingerprint(
            {
                "kind": "ionized-continuum-profile",
                "system": system.system_id,
                "source": thermochemical_source.plan_id,
                "transport": plasma_transport.transport_id,
                "electrostatic": None if electrostatic is None else electrostatic.plan_id,
            }
        )


class RadiatingContinuumProfile(StrictModule, NonTrainableState):
    continuum: IonizedContinuumProfile
    radiation: MultigroupRadiationMatterProcessPlan
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        continuum: IonizedContinuumProfile,
        radiation: MultigroupRadiationMatterProcessPlan,
        /,
    ):
        if (
            not isinstance(continuum, IonizedContinuumProfile)
            or not isinstance(radiation, MultigroupRadiationMatterProcessPlan)
            or radiation.coefficients.populations.species_count
            != continuum.system.species_count
        ):
            raise ValueError("Radiating continuum species or processes do not match.")
        self.continuum = continuum
        self.radiation = radiation
        self.profile_id = canonical_fingerprint(
            {
                "kind": "radiating-continuum-profile",
                "continuum": continuum.profile_id,
                "radiation": radiation.plan_id,
            }
        )


class AblatingEntryProfile(StrictModule, NonTrainableState):
    radiating: RadiatingContinuumProfile
    wall: ReactingPlasmaWallPlan
    material: PorousAblatingMaterialPlan
    interface: ConjugateAerothermalInterfacePlan
    recession: FixedConnectivityRecessionPlan
    profile_id: str = eqx.field(static=True)

    def __init__(self, radiating, wall, material, interface, recession, /):
        if (
            not isinstance(radiating, RadiatingContinuumProfile)
            or not isinstance(wall, ReactingPlasmaWallPlan)
            or not isinstance(material, PorousAblatingMaterialPlan)
            or not isinstance(interface, ConjugateAerothermalInterfacePlan)
            or not isinstance(recession, FixedConnectivityRecessionPlan)
            or wall.mechanism.gas_schema.schema_id
            != radiating.continuum.system.thermodynamics.schema.schema_id
        ):
            raise ValueError("Ablating entry profile components do not match.")
        self.radiating = radiating
        self.wall = wall
        self.material = material
        self.interface = interface
        self.recession = recession
        self.profile_id = canonical_fingerprint(
            {
                "kind": "ablating-entry-profile",
                "radiating": radiating.profile_id,
                "wall": wall.plan_id,
                "material": material.material_id,
                "interface": interface.plan_id,
                "recession": recession.plan_id,
            }
        )


class RarefiedDSMCProfile(StrictModule, NonTrainableState):
    dsmc: DSMCProductionPlan
    profile_id: str = eqx.field(static=True)

    def __init__(self, dsmc: DSMCProductionPlan, /):
        if not isinstance(dsmc, DSMCProductionPlan):
            raise TypeError("Rarefied profile requires DSMCProductionPlan.")
        self.dsmc = dsmc
        self.profile_id = canonical_fingerprint(
            {"kind": "rarefied-dsmc-profile", "dsmc": dsmc.plan_id}
        )


class FixedContinuumDSMCProfile(StrictModule, NonTrainableState):
    continuum: IonizedContinuumProfile
    rarefied: RarefiedDSMCProfile
    interface: FixedContinuumDSMCInterfacePlan
    profile_id: str = eqx.field(static=True)

    def __init__(self, continuum, rarefied, interface, /):
        if (
            not isinstance(continuum, IonizedContinuumProfile)
            or not isinstance(rarefied, RarefiedDSMCProfile)
            or not isinstance(interface, FixedContinuumDSMCInterfacePlan)
            or interface.component_count != continuum.system.component_count
        ):
            raise ValueError("Fixed continuum-DSMC profile components do not match.")
        self.continuum = continuum
        self.rarefied = rarefied
        self.interface = interface
        self.profile_id = canonical_fingerprint(
            {
                "kind": "fixed-continuum-dsmc-profile",
                "continuum": continuum.profile_id,
                "rarefied": rarefied.profile_id,
                "interface": interface.plan_id,
            }
        )


class DynamicContinuumDSMCProfile(StrictModule, NonTrainableState):
    fixed: FixedContinuumDSMCProfile
    ownership: DynamicHybridOwnershipPlan
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        fixed: FixedContinuumDSMCProfile,
        ownership: DynamicHybridOwnershipPlan,
        /,
    ):
        if not isinstance(fixed, FixedContinuumDSMCProfile) or not isinstance(
            ownership, DynamicHybridOwnershipPlan
        ):
            raise TypeError(
                "Dynamic continuum-DSMC profile requires fixed bridge and ownership."
            )
        self.fixed = fixed
        self.ownership = ownership
        self.profile_id = canonical_fingerprint(
            {
                "kind": "dynamic-continuum-dsmc-profile",
                "fixed": fixed.profile_id,
                "ownership": ownership.policy_id,
            }
        )


__all__ = [
    "AblatingEntryProfile",
    "DynamicContinuumDSMCProfile",
    "FixedContinuumDSMCProfile",
    "IonizedContinuumProfile",
    "RadiatingContinuumProfile",
    "RarefiedDSMCProfile",
]
