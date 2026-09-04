#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bounded skeletal surface-electromyography models."""

from ._planar_conductor import (
    PETERSEN_ROSTALSKI_2019_DOI,
    PETERSEN_ROSTALSKI_2019_DRYAD_DOI,
    PetersenRostalski2019PlanarConductorPlan,
    PlanarConductorEvidence,
    PlanarConductorParameters,
    PlanarConductorResult,
)
from ._templates import (
    MotorUnitActionPotentialTemplatePlan,
    PreparedMotorUnitActionPotentialTemplates,
    TemplateEMGEvidence,
    TemplateEMGResult,
)


__all__ = [
    "PETERSEN_ROSTALSKI_2019_DOI",
    "PETERSEN_ROSTALSKI_2019_DRYAD_DOI",
    "MotorUnitActionPotentialTemplatePlan",
    "PetersenRostalski2019PlanarConductorPlan",
    "PlanarConductorEvidence",
    "PlanarConductorParameters",
    "PlanarConductorResult",
    "PreparedMotorUnitActionPotentialTemplates",
    "TemplateEMGEvidence",
    "TemplateEMGResult",
]
