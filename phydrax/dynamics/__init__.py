#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamical-system, pathwise-evolution, analysis, and identification contracts."""

from . import analysis, identification
from ._cell_enclosure import CellMapEnclosure
from ._conley import (
    compute_conley_homology_index,
    compute_conley_index,
    ConleyHomologyIndex,
    ConleyIndexResult,
)
from ._differential_algebraic import (
    AutonomousDifferentialAlgebraicResidual,
    DAERole,
    DAEStructure,
    DifferentialAlgebraicResidual,
    DifferentialAlgebraicSystem,
    InputDifferentialAlgebraicResidual,
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
from ._linear_descriptor import DescriptorSystemEvidence, LinearDescriptorSystem
from ._linearization import EvolutionJacobianAction
from ._model_system import (
    continuous_model_system,
    ContinuousModelVectorField,
    discrete_model_system,
    DiscreteModelTransition,
)
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
    HeldInputPolicy,
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
    "CellMapEnclosure",
    "ConleyHomologyIndex",
    "compute_conley_homology_index",
    "ConleyIndexResult",
    "compute_conley_index",
    "AbstractDifferentiableEvolution",
    "AbstractEvolution",
    "AbstractInputPolicy",
    "AutonomousContinuousVectorField",
    "AutonomousDiscreteTransition",
    "AutonomousDifferentialAlgebraicResidual",
    "CallableInputPolicy",
    "HeldInputPolicy",
    "CaseAxisRole",
    "ContinuousModelVectorField",
    "ContinuousSystem",
    "DAERole",
    "DAEStructure",
    "DifferentialAlgebraicResidual",
    "DifferentialAlgebraicSystem",
    "DescriptorSystemEvidence",
    "DiscreteEvolution",
    "DiscreteModelTransition",
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
    "InputDifferentialAlgebraicResidual",
    "InputContinuousVectorField",
    "InputDiscreteTransition",
    "InputLayout",
    "LinearDescriptorSystem",
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
    "discrete_model_system",
    "evolve",
]
