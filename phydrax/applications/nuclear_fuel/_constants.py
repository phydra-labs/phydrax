"""Exact SI constants and unit conversions shared by the fuel models.

A correlation fitted with a rounded constant (for example FRAPCON's
8.314 J/(mol K) creep gas constant or the 1.38e-23 J/K Boltzmann constant in
FTHEXP) keeps that value beside the correlation, because its coefficients were
fitted with it. Everything else uses the exact 2019 SI values below.
"""

from typing import Final


AVOGADRO_PER_MOLE: Final[float] = 6.02214076e23
BOLTZMANN_J_PER_K: Final[float] = 1.380649e-23
# Product of the two exact constants; equal in float64 to 8.31446261815324.
MOLAR_GAS_CONSTANT_J_PER_MOL_K: Final[float] = BOLTZMANN_J_PER_K * AVOGADRO_PER_MOLE
JOULE_PER_ELECTRON_VOLT: Final[float] = 1.602176634e-19

SECONDS_PER_HOUR: Final[float] = 3600.0
SECONDS_PER_DAY: Final[float] = 86400.0
# 1 MWd = 1e6 W * 86400 s.
JOULE_PER_MEGAWATT_DAY: Final[float] = 8.64e10
PASCAL_PER_MEGAPASCAL: Final[float] = 1.0e6
