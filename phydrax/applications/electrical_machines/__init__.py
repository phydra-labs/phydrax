#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Linear planar magnetostatic machine analysis and fixed-angle design studies."""

from ._design import (
    MachineAngleResult,
    MachineAngleStudy,
    MachineDesignResult,
    optimize_machine_design,
    polar_machine_study,
    scan_machine_angles,
)
from ._magnetostatic import (
    machine_coenergy,
    MachineFieldResult,
    MachineSolvePolicy,
    solve_planar_machine,
)
from ._model import LinearMagneticRegion, PlanarMachine, polar_machine


__all__ = [
    "LinearMagneticRegion",
    "PlanarMachine",
    "polar_machine",
    "MachineSolvePolicy",
    "MachineFieldResult",
    "machine_coenergy",
    "solve_planar_machine",
    "MachineAngleStudy",
    "MachineAngleResult",
    "MachineDesignResult",
    "polar_machine_study",
    "scan_machine_angles",
    "optimize_machine_design",
]
