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

from ._collocation import (
    assemble_stochastic_collocation,
    COLLOCATION_NONFINITE,
    COLLOCATION_SOLVER_FAILURE,
    COLLOCATION_SUCCESS,
    CollocationAxisRule,
    evaluate_stochastic_collocation,
    materialize_stochastic_collocation,
    run_stochastic_collocation,
    StochasticCollocationDesign,
    StochasticCollocationDiagnostics,
    StochasticCollocationNode,
    StochasticCollocationNodeEvaluation,
    StochasticCollocationPlan,
    StochasticCollocationResult,
)
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
from ._coupled import (
    CoupledCost,
    CoupledHierarchyResult,
    CoupledLevelResult,
    CoupledLevelSolver,
    CoupledObservable,
    CoupledValidity,
    solve_coupled_hierarchy,
)
from ._deep_bsde import DeepBSDEResult, solve_deep_bsde
from ._deep_picard import (
    DeepPicardDiagnostics,
    DeepPicardInitialSource,
    DeepPicardResult,
    PicardSourceContext,
    solve_deep_picard,
    StructuredPicardSource,
    StructuredSourceBuilder,
)
from ._deep_splitting import (
    DeepSplittingDiagnostics,
    DeepSplittingInterpolation,
    DeepSplittingResult,
    DeepSplittingSamplingMode,
    DeepSplittingSolution,
    solve_deep_splitting,
)
from ._differential import (
    DifferentialInterpretation,
    DifferentialProblem,
    DifferentialSolution,
    DifferentialVectorField,
    NoiseStructure,
    WienerTerm,
)
from ._geometric import (
    AbstractGeometricSolver,
    CommutatorFreeSolver,
    CommutatorFreeTableau,
    commutator_free_midpoint_tableau,
    GeometricEuler,
    GeometricLocalInterpolation,
    RKMK,
    SRKMK,
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
from ._levy import (
    LevySDEProblem,
    LevySDEScheme,
    LevySDESolution,
    LevySDESolverDiagnostics,
    LevySDEVectorField,
    LevySmallJumpApproximation,
    solve_levy_sde,
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
from ._memory import (
    DelayHistory,
    DelayVectorField,
    MemoryEquationSolution,
    solve_stochastic_delay,
    solve_stochastic_volterra,
    StochasticDelayProblem,
    StochasticVolterraProblem,
    VolterraFreeTerm,
    VolterraKernel,
    VolterraVectorField,
)
from ._noise import SpatialNoiseApproximation, SpatialNoiseBasis
from ._particles import (
    InteractingParticleProblem,
    InteractingParticleSolution,
    ParticleVectorField,
    solve_interacting_particles,
)
from ._reflected_bsde import (
    predict_reflected_path_dependent_control,
    predict_reflected_path_dependent_value,
    reflected_path_dependent_bsde_diagnostics,
    ReflectedPathDependentBSDEDiagnostics,
    ReflectedPathDependentBSDEResult,
    solve_reflected_path_dependent_bsde,
)
from ._regression_bsde import (
    AbstractBSDERegressionBasis,
    BSDERegressionScheme,
    CallableBSDERegressionBasis,
    least_squares_bsde_diagnostics,
    LeastSquaresBSDEDiagnostics,
    LeastSquaresBSDEResult,
    PolynomialBSDERegressionBasis,
    predict_bsde_least_squares_control,
    predict_bsde_least_squares_value,
    solve_bsde_least_squares,
)
from ._rough import (
    RoughDifferentialProblem,
    RoughDifferentialScheme,
    RoughDifferentialSolution,
    RoughDrift,
    RoughVectorFields,
    solve_rough_differential,
)
from ._semilinear import (
    exact_modal_stochastic_convolution,
    SemilinearFallback,
    SemilinearSPDEScheme,
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
from ._state_transfer import (
    AbstractStateTransfer,
    IdentityStateTransfer,
    SpectralCoefficientStateTransfer,
    TensorGridBoundary,
    TensorGridRestriction,
    TensorGridStateTransfer,
)


__all__ = [
    "AbstractGeometricSolver",
    "AbstractSpatialDiscretization",
    "AbstractBSDERegressionBasis",
    "assemble_stochastic_collocation",
    "COLLOCATION_NONFINITE",
    "COLLOCATION_SOLVER_FAILURE",
    "COLLOCATION_SUCCESS",
    "CollocationAxisRule",
    "AbstractStateTransfer",
    "CoupledCost",
    "CoupledHierarchyResult",
    "CoupledLevelResult",
    "CoupledLevelSolver",
    "CoupledObservable",
    "CoupledValidity",
    "CoupledFBSDEProblem",
    "CoupledFBSDEResult",
    "BSDERegressionScheme",
    "CommutatorFreeSolver",
    "CommutatorFreeTableau",
    "commutator_free_midpoint_tableau",
    "DeepBSDEResult",
    "DeepPicardDiagnostics",
    "DeepPicardInitialSource",
    "DeepPicardResult",
    "DeepSplittingDiagnostics",
    "DeepSplittingInterpolation",
    "DeepSplittingResult",
    "DeepSplittingSamplingMode",
    "DeepSplittingSolution",
    "PicardSourceContext",
    "CallableBSDERegressionBasis",
    "DelayHistory",
    "DelayVectorField",
    "DifferentialInterpretation",
    "DifferentialProblem",
    "DifferentialSolution",
    "DifferentialVectorField",
    "evaluate_stochastic_collocation",
    "GeometricEuler",
    "GeometricLocalInterpolation",
    "NoiseStructure",
    "finite_state_generator",
    "FiniteStateGenerator",
    "GeneratorBoundaryPolicy",
    "JumpAlgorithm",
    "JumpDifferentialProblem",
    "JumpDifferentialSolution",
    "JumpSolution",
    "InteractingParticleProblem",
    "InteractingParticleSolution",
    "LevySDEProblem",
    "LevySDEScheme",
    "LevySDESolution",
    "LevySDESolverDiagnostics",
    "LevySDEVectorField",
    "LevySmallJumpApproximation",
    "IdentityStateTransfer",
    "MatrixFunctionDifferentiation",
    "MatrixFunctionKind",
    "MatrixFunctionMethod",
    "MatrixFunctionPolicy",
    "MemoryEquationSolution",
    "NoiseTruncationLevel",
    "NoiseTruncationStudy",
    "materialize_stochastic_collocation",
    "least_squares_bsde_diagnostics",
    "LeastSquaresBSDEDiagnostics",
    "RKMK",
    "SRKMK",
    "LeastSquaresBSDEResult",
    "ParticleVectorField",
    "RoughDifferentialProblem",
    "RoughDifferentialScheme",
    "RoughDifferentialSolution",
    "RoughDrift",
    "RoughVectorFields",
    "SemidiscreteSPDE",
    "SemilinearDrift",
    "SemilinearFallback",
    "SemilinearSPDEScheme",
    "SPDEConvergenceLevel",
    "SPDEConvergenceMetric",
    "SPDEConvergenceStudy",
    "SPDEErrorBudget",
    "SPDERefinementAxis",
    "SpatialNoiseApproximation",
    "SpatialNoiseBasis",
    "SpectralSpatialDiscretization",
    "SpectralMatrixRepresentation",
    "SpectralCoefficientStateTransfer",
    "PolynomialBSDERegressionBasis",
    "predict_bsde_least_squares_control",
    "predict_bsde_least_squares_value",
    "predict_reflected_path_dependent_control",
    "predict_reflected_path_dependent_value",
    "reflected_path_dependent_bsde_diagnostics",
    "ReflectedPathDependentBSDEDiagnostics",
    "ReflectedPathDependentBSDEResult",
    "run_stochastic_collocation",
    "StochasticCollocationDesign",
    "StochasticCollocationDiagnostics",
    "StochasticCollocationNode",
    "StochasticCollocationNodeEvaluation",
    "StochasticCollocationPlan",
    "StochasticCollocationResult",
    "StochasticDelayProblem",
    "StochasticVolterraProblem",
    "TensorGridDiscretization",
    "TensorGridBoundary",
    "TensorGridRestriction",
    "TensorGridStateTransfer",
    "spatial_measure",
    "WeakObservableEstimate",
    "WienerTerm",
    "VolterraFreeTerm",
    "VolterraKernel",
    "VolterraVectorField",
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
    "solve_levy_sde",
    "solve_coupled_hierarchy",
    "solve_coupled_fbsde_explicit",
    "solve_interacting_particles",
    "solve_rough_differential",
    "solve_deep_picard",
    "solve_deep_bsde",
    "solve_deep_splitting",
    "StructuredPicardSource",
    "StructuredSourceBuilder",
    "solve_semilinear_spde",
    "solve_stochastic_delay",
    "solve_stochastic_volterra",
    "solve_bsde_least_squares",
    "solve_reflected_path_dependent_bsde",
    "weak_observable_estimate",
    "FunctionalSolver",
    "EnforcedConstraintPipeline",
    "EnforcedConstraintPipelines",
    "EnforcedInteriorData",
    "SingleFieldEnforcedConstraint",
    "MultiFieldEnforcedConstraint",
]
