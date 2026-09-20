#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Balanced positive-sequence RMS networks, native optimization and machine DAEs."""

from importlib import import_module

from ._dynamics import (
    ClassicalMachine,
    DroopGovernor,
    FirstOrderAVR,
    FixedExciter,
    FixedGovernor,
    initialize_power_dynamics,
    initialize_smib,
    Order4Machine,
    PowerDynamicsInitialization,
    PowerDynamicsModel,
    PowerDynamicsResult,
    PowerEvent,
    PowerEventEvidence,
    PowerSegmentResult,
    PowerTopology,
    simulate_power_dynamics,
)
from ._interchange import (
    parse_cgmes,
    parse_matpower,
    parse_psse,
    PowerCaseAdaptation,
    PowerImportError,
    PowerParserLimits,
)
from ._network import (
    Branch,
    Bus,
    BusControl,
    compile_network,
    CompiledNetwork,
    Generator,
    Load,
    PowerBase,
    PowerNetwork,
    PowerStudy,
    Shunt,
)
from ._opf import (
    ACOPFCompilation,
    ACOPFResult,
    compile_ac_opf,
    compile_dc_opf,
    DCFlowResult,
    DCOPFCompilation,
    DCOPFResult,
    solve_ac_opf,
    solve_dc_opf,
    solve_dc_power_flow,
)
from ._polynomial_power_flow import __all__ as _polynomial_power_flow_all
from ._power_flow import (
    fixed_mode_power_flow,
    FixedModePowerFlowResult,
    PowerFlowResult,
    solve_power_flow,
)


_FACADE_EXPORT_MODULES = ("._polynomial_power_flow",)


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "PowerBase",
    "Bus",
    "BusControl",
    "PowerStudy",
    "Branch",
    "Shunt",
    "Generator",
    "Load",
    "PowerNetwork",
    "CompiledNetwork",
    "compile_network",
    "FixedModePowerFlowResult",
    "PowerFlowResult",
    "fixed_mode_power_flow",
    "solve_power_flow",
    "DCFlowResult",
    "solve_dc_power_flow",
    "DCOPFCompilation",
    "DCOPFResult",
    "compile_dc_opf",
    "solve_dc_opf",
    "ACOPFCompilation",
    "ACOPFResult",
    "compile_ac_opf",
    "solve_ac_opf",
    "FixedExciter",
    "FirstOrderAVR",
    "FixedGovernor",
    "DroopGovernor",
    "ClassicalMachine",
    "Order4Machine",
    "PowerTopology",
    "PowerDynamicsModel",
    "PowerDynamicsInitialization",
    "PowerEvent",
    "PowerEventEvidence",
    "PowerSegmentResult",
    "PowerDynamicsResult",
    "initialize_power_dynamics",
    "initialize_smib",
    "simulate_power_dynamics",
    "PowerCaseAdaptation",
    "PowerImportError",
    "PowerParserLimits",
    "parse_matpower",
    "parse_psse",
    "parse_cgmes",
]
__all__ += [name for name in _polynomial_power_flow_all if name not in __all__]
