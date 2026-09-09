#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Directed regional neural dynamics, physical delays and persistent BOLD."""

from ._bold import balloon_equilibrium, BalloonWindkessel, NeuralBOLDDrive
from ._observation import BOLDObservation
from ._regional import Hopf, regional_coupling, RegionalConnectivity, WilsonCowan
from ._workflow import (
    regional_bold_problem,
    regional_problem,
    RegionalSolution,
    solve_regional,
)


__all__ = [
    "BalloonWindkessel",
    "BOLDObservation",
    "Hopf",
    "NeuralBOLDDrive",
    "RegionalConnectivity",
    "RegionalSolution",
    "WilsonCowan",
    "balloon_equilibrium",
    "regional_bold_problem",
    "regional_coupling",
    "regional_problem",
    "solve_regional",
]
