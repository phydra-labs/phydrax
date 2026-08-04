#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed reference-space designs shared by domain sampling and integration."""

from .._sampling import (
    DESIGN_ALGORITHM_VERSION,
    AntitheticDesign,
    DesignCapabilities,
    design_signature,
    design_capabilities,
    design_name,
    HaltonDesign,
    HammersleyDesign,
    IIDDesign,
    LatinHypercubeDesign,
    materialize_design,
    RandomizedQMCDesign,
    resolve_design,
    SobolDesign,
)


__all__ = [
    "AntitheticDesign",
    "DESIGN_ALGORITHM_VERSION",
    "DesignCapabilities",
    "HaltonDesign",
    "HammersleyDesign",
    "IIDDesign",
    "LatinHypercubeDesign",
    "RandomizedQMCDesign",
    "SobolDesign",
    "design_capabilities",
    "design_name",
    "materialize_design",
    "design_signature",
    "resolve_design",
]
