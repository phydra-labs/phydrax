#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._applications import (
    AllenCahnModel,
    CahnHilliardModel,
    CrystalPlasticityModel,
    CrystalSlipSystem,
    FrictionlessContactLaw,
    j2_radial_return,
    J2PlasticityParameters,
    PhaseFieldFractureModel,
)
from ._execution import ElementTensorOperator, PartialAssemblyOperator
from ._interpreter import evaluate_differential_operator, execute_local_action
from ._ir import (
    ActionKind,
    DifferentialOperator,
    FieldSlot,
    FieldSlotRole,
    LocalActionIR,
    LocalActionTermIR,
    RegionIR,
    RegionKind,
)
from ._lowering import compile_workset_program, lower_weak_form
from ._materials import LocalImplicitDiagnostics, LocalImplicitMaterial
from ._operators import (
    average,
    curl,
    divergence,
    FieldJet,
    jump,
    normal_trace,
    symmetric_gradient,
    tangential_trace,
)
from ._worksets import CompiledWorkset, WorksetProgram, WorksetSignature


__all__ = [
    "AllenCahnModel",
    "CahnHilliardModel",
    "ElementTensorOperator",
    "PartialAssemblyOperator",
    "CrystalPlasticityModel",
    "CrystalSlipSystem",
    "FrictionlessContactLaw",
    "J2PlasticityParameters",
    "PhaseFieldFractureModel",
    "j2_radial_return",
    "ActionKind",
    "CompiledWorkset",
    "DifferentialOperator",
    "LocalImplicitDiagnostics",
    "LocalImplicitMaterial",
    "FieldJet",
    "FieldSlot",
    "FieldSlotRole",
    "LocalActionIR",
    "LocalActionTermIR",
    "RegionIR",
    "RegionKind",
    "WorksetProgram",
    "WorksetSignature",
    "average",
    "compile_workset_program",
    "curl",
    "divergence",
    "evaluate_differential_operator",
    "execute_local_action",
    "jump",
    "lower_weak_form",
    "normal_trace",
    "symmetric_gradient",
    "tangential_trace",
]
