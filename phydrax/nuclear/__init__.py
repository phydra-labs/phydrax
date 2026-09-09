#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nuclear identities, data, reactions, inventories, and interoperability."""

from . import interchange
from ._activation import (
    ActivationLedger,
    ActivationNetworkPlan,
    ActivationScheduleResult,
    ActivationStepResult,
    AVOGADRO_PER_MOL,
    InventoryTransition,
    IrradiationSchedulePlan,
    NuclideInventory,
    PreparedActivationNetwork,
)
from ._composition import (
    CompositionBasis,
    CompositionConversionResult,
    NuclearMaterialState,
    NuclideComposition,
)
from ._energy import (
    EnergyGroupLocation,
    EnergyGroupStructure,
    PreparedEnergyGroupStructure,
)
from ._fusion import (
    FusionProductSource,
    ReactivityEvaluation,
    TabulatedMaxwellianReactivity,
    ThermalFusionReactionPlan,
    ThermalFusionReactionResult,
)
from ._identity import (
    NuclearParticleKind,
    NuclearSpeciesKey,
    NuclearSpeciesTable,
    NuclideKey,
    PreparedNuclearSpeciesTable,
)
from ._provenance import NuclearDataProvenance
from ._qualification import nuclear_candidate_profile, nuclear_candidate_profiles
from ._quantity import resolve_nuclear_quantity
from ._reaction import (
    NuclearReactionChannel,
    NuclearReactionConservation,
    NuclearReactionParticipant,
    SPEED_OF_LIGHT_M_S,
)
from ._spectrum import (
    group_reaction_rate,
    MultigroupParticleSource,
    MultigroupScalarFlux,
    PreparedMultigroupParticleSource,
    PreparedMultigroupScalarFlux,
)


__all__ = [
    "AVOGADRO_PER_MOL",
    "ActivationLedger",
    "ActivationNetworkPlan",
    "ActivationScheduleResult",
    "ActivationStepResult",
    "CompositionBasis",
    "CompositionConversionResult",
    "EnergyGroupLocation",
    "EnergyGroupStructure",
    "FusionProductSource",
    "InventoryTransition",
    "IrradiationSchedulePlan",
    "MultigroupParticleSource",
    "MultigroupScalarFlux",
    "NuclearDataProvenance",
    "NuclearMaterialState",
    "NuclearParticleKind",
    "NuclearReactionChannel",
    "NuclearReactionConservation",
    "NuclearReactionParticipant",
    "NuclearSpeciesKey",
    "NuclearSpeciesTable",
    "NuclideComposition",
    "NuclideInventory",
    "NuclideKey",
    "PreparedEnergyGroupStructure",
    "PreparedMultigroupParticleSource",
    "PreparedMultigroupScalarFlux",
    "PreparedNuclearSpeciesTable",
    "PreparedActivationNetwork",
    "ReactivityEvaluation",
    "SPEED_OF_LIGHT_M_S",
    "TabulatedMaxwellianReactivity",
    "ThermalFusionReactionPlan",
    "ThermalFusionReactionResult",
    "nuclear_candidate_profile",
    "nuclear_candidate_profiles",
    "group_reaction_rate",
    "resolve_nuclear_quantity",
    "interchange",
]
