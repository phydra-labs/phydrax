#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Solver

Solvers assemble constraints, data, and attached model losses into a loss and
provide utilities for training or evaluation. The main entry point is
`FunctionalSolver`.

## Enforced constraints

Enforced constraint pipelines modify functions by construction so that boundary
and initial conditions are satisfied exactly. This is useful for enforcing
$u|_{\\partial \\Omega} = g$ or $u|_{t=0} = u_0$ without penalty terms.

!!! example
    ```python
    import jax.random as jr
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    structure = phx.domain.ProductStructure((("x",),))
    constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: f,
        num_points=32,
        structure=structure,
    )

    solver = phx.solver.FunctionalSolver(
        functions={"u": u},
        constraints=[constraint],
    )

    loss = solver.loss(key=jr.key(0))
    print(loss)
    ```
"""

from ._convergence import (
    coupled_strong_error,
    NoiseTruncationLevel,
    NoiseTruncationStudy,
    SPDEConvergenceLevel,
    SPDEConvergenceMetric,
    SPDEConvergenceStudy,
    SPDEErrorBudget,
    SPDERefinementAxis,
    weak_observable_estimate,
    WeakObservableEstimate,
)
from ._differential import (
    DifferentialInterpretation,
    DifferentialProblem,
    DifferentialSolution,
    DifferentialVectorField,
    NoiseStructure,
    WienerTerm,
)
from ._diffrax_backend import solve_diffrax, solve_diffrax_ensemble
from ._enforced_constraint_pipeline import (
    EnforcedConstraintPipeline,
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    MultiFieldEnforcedConstraint,
    SingleFieldEnforcedConstraint,
)
from ._fbsde import (
    CoupledFBSDEProblem,
    CoupledFBSDEResult,
    solve_coupled_fbsde_explicit,
)
from ._functional_solver import FunctionalSolver
from ._integration import spatial_measure
from ._jump import (
    finite_state_generator,
    FiniteStateGenerator,
    GeneratorBoundaryPolicy,
    JumpAlgorithm,
    JumpDifferentialProblem,
    JumpDifferentialSolution,
    JumpSolution,
    solve_direct_ssa,
    solve_jump_differential,
    solve_next_reaction,
)
from ._matrix_functions import (
    matrix_exponential_action,
    matrix_function_action,
    matrix_phi1_action,
    MatrixFunctionDifferentiation,
    MatrixFunctionKind,
    MatrixFunctionMethod,
    MatrixFunctionPolicy,
    SpectralMatrixRepresentation,
)
from ._noise import SpatialNoiseApproximation, SpatialNoiseBasis
from ._semilinear import (
    exact_modal_stochastic_convolution,
    SemilinearFallback,
    solve_semilinear_spde,
)
from ._semilinear_drift import SemilinearDrift
from ._spatial import (
    AbstractSpatialDiscretization,
    SpectralSpatialDiscretization,
    TensorGridDiscretization,
)
from ._spde import (
    SemidiscreteSPDE,
    semidiscretize_reaction_diffusion,
    semidiscretize_semilinear_spde,
    semidiscretize_spde,
)


__all__ = [
    "AbstractSpatialDiscretization",
    "CoupledFBSDEProblem",
    "CoupledFBSDEResult",
    "DifferentialInterpretation",
    "DifferentialProblem",
    "DifferentialSolution",
    "DifferentialVectorField",
    "NoiseStructure",
    "finite_state_generator",
    "FiniteStateGenerator",
    "GeneratorBoundaryPolicy",
    "JumpAlgorithm",
    "JumpDifferentialProblem",
    "JumpDifferentialSolution",
    "JumpSolution",
    "MatrixFunctionDifferentiation",
    "MatrixFunctionKind",
    "MatrixFunctionMethod",
    "MatrixFunctionPolicy",
    "NoiseTruncationLevel",
    "NoiseTruncationStudy",
    "SemidiscreteSPDE",
    "SemilinearDrift",
    "SemilinearFallback",
    "SPDEConvergenceLevel",
    "SPDEConvergenceMetric",
    "SPDEConvergenceStudy",
    "SPDEErrorBudget",
    "SPDERefinementAxis",
    "SpatialNoiseApproximation",
    "SpatialNoiseBasis",
    "SpectralSpatialDiscretization",
    "SpectralMatrixRepresentation",
    "TensorGridDiscretization",
    "spatial_measure",
    "WeakObservableEstimate",
    "WienerTerm",
    "semidiscretize_reaction_diffusion",
    "coupled_strong_error",
    "exact_modal_stochastic_convolution",
    "matrix_exponential_action",
    "matrix_function_action",
    "matrix_phi1_action",
    "semidiscretize_spde",
    "semidiscretize_semilinear_spde",
    "solve_diffrax",
    "solve_diffrax_ensemble",
    "solve_direct_ssa",
    "solve_jump_differential",
    "solve_next_reaction",
    "solve_coupled_fbsde_explicit",
    "solve_semilinear_spde",
    "weak_observable_estimate",
    "FunctionalSolver",
    "EnforcedConstraintPipeline",
    "EnforcedConstraintPipelines",
    "EnforcedInteriorData",
    "SingleFieldEnforcedConstraint",
    "MultiFieldEnforcedConstraint",
]
