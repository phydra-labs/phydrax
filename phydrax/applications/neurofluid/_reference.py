#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from enum import StrEnum


class BrainCompartmentRole(StrEnum):
    WHITE_MATTER = "white_matter"
    GRAY_MATTER = "gray_matter"
    SUBCORTICAL_GRAY_MATTER = "subcortical_gray_matter"
    BRAINSTEM = "brainstem"
    SUBARACHNOID_SPACE = "subarachnoid_space"
    VENTRICLES = "ventricles"
    DURA_BOUNDARY = "dura_boundary"
    SPINAL_OUTLET = "spinal_outlet"


__all__ = ["BrainCompartmentRole"]
