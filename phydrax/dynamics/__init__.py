#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamical-system, pathwise-evolution, analysis, and identification contracts."""

from . import analysis, identification
from ._differential_algebraic import (
    DAERole,
    DAEStructure,
    DifferentialAlgebraicResidual,
    DifferentialAlgebraicSystem,
)
from ._evolution import (
    AbstractDifferentiableEvolution,
    AbstractEvolution,
    DiscreteEvolution,
    EVOLUTION_BACKEND_FAILED,
    EVOLUTION_NONFINITE,
    EVOLUTION_OUTSIDE_GEOMETRY,
    EVOLUTION_SUCCESS,
    EvolutionStep,
    EvolutionTangentStep,
    EvolutionTrajectory,
    evolve,
)
from ._grid import EvolutionGrid, IterationGrid, TimeGrid
from ._layout import InputLayout, InputRole, StateLayout
from ._linearization import EvolutionJacobianAction
from ._model_system import continuous_model_system, ContinuousModelVectorField
from ._second_order import (
    SecondOrderDifferentialProblem,
    SecondOrderDifferentialSystem,
    SecondOrderResidual,
)
from ._system import (
    AbstractInputPolicy,
    AutonomousContinuousVectorField,
    AutonomousDiscreteTransition,
    CallableInputPolicy,
    ContinuousSystem,
    DiscreteSystem,
    InputContinuousVectorField,
    InputDiscreteTransition,
    SystemTransition,
    SystemVectorField,
)
from ._trajectory import (
    CaseAxisRole,
    InputAlignment,
    TrajectoryData,
    TrajectoryTransitions,
)


__all__ = [
    "analysis",
    "identification",
    "AbstractDifferentiableEvolution",
    "AbstractEvolution",
    "AbstractInputPolicy",
    "AutonomousContinuousVectorField",
    "AutonomousDiscreteTransition",
    "CallableInputPolicy",
    "CaseAxisRole",
    "ContinuousModelVectorField",
    "ContinuousSystem",
    "DAERole",
    "DAEStructure",
    "DifferentialAlgebraicResidual",
    "DifferentialAlgebraicSystem",
    "DiscreteEvolution",
    "DiscreteSystem",
    "EVOLUTION_BACKEND_FAILED",
    "EVOLUTION_NONFINITE",
    "EVOLUTION_OUTSIDE_GEOMETRY",
    "EVOLUTION_SUCCESS",
    "EvolutionGrid",
    "EvolutionJacobianAction",
    "EvolutionStep",
    "EvolutionTangentStep",
    "EvolutionTrajectory",
    "InputContinuousVectorField",
    "InputDiscreteTransition",
    "InputLayout",
    "InputAlignment",
    "InputRole",
    "IterationGrid",
    "SecondOrderDifferentialProblem",
    "SecondOrderDifferentialSystem",
    "SecondOrderResidual",
    "StateLayout",
    "SystemTransition",
    "SystemVectorField",
    "TimeGrid",
    "TrajectoryData",
    "TrajectoryTransitions",
    "continuous_model_system",
    "evolve",
]
