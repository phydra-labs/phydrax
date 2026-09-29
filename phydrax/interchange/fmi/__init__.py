#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only FMI 2.0 synchronous Co-Simulation; optional FMPy imports are lazy.

Coupling bindings load `phydrax.solver.coupling` on first use, so importing the
session layer does not pull in the partitioned-coupling runtime.
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from ._session import (
    FMICoSimulationSession,
    FMIModelDescription,
    FMIState,
    FMIStepResult,
    FMIUnit,
    FMIVariable,
    inspect_fmu,
)


if TYPE_CHECKING:
    from ._binding import (
        fmi_unit_definition,
        FMICouplingBinding,
        FMICouplingParticipant,
        FMIPortRealization,
        FMIVariableBinding,
        FMIWindowStatus,
    )

_BINDING_EXPORTS = (
    "FMICouplingBinding",
    "FMICouplingParticipant",
    "FMIPortRealization",
    "FMIVariableBinding",
    "FMIWindowStatus",
    "fmi_unit_definition",
)


def __getattr__(name: str) -> Any:
    if name in _BINDING_EXPORTS:
        return getattr(import_module("._binding", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FMICoSimulationSession",
    "FMICouplingBinding",
    "FMICouplingParticipant",
    "FMIModelDescription",
    "FMIPortRealization",
    "FMIState",
    "FMIStepResult",
    "FMIUnit",
    "FMIVariable",
    "FMIVariableBinding",
    "FMIWindowStatus",
    "fmi_unit_definition",
    "inspect_fmu",
]
