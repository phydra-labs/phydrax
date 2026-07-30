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

from ._differential import (
    DifferentialInterpretation,
    DifferentialProblem,
    DifferentialSolution,
    DifferentialVectorField,
    LevyAreaKind,
    WienerDriver,
)
from ._diffrax_backend import solve_diffrax, solve_diffrax_ensemble
from ._enforced_constraint_pipeline import (
    EnforcedConstraintPipeline,
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    MultiFieldEnforcedConstraint,
    SingleFieldEnforcedConstraint,
)
from ._functional_solver import FunctionalSolver
from ._noise import SpatialNoiseBasis
from ._spatial import (
    AbstractSpatialDiscretization,
    SpectralSpatialDiscretization,
    TensorGridDiscretization,
)
from ._spde import (
    SemidiscreteSPDE,
    semidiscretize_reaction_diffusion,
    semidiscretize_spde,
)


__all__ = [
    "AbstractSpatialDiscretization",
    "DifferentialInterpretation",
    "DifferentialProblem",
    "DifferentialSolution",
    "DifferentialVectorField",
    "LevyAreaKind",
    "WienerDriver",
    "SemidiscreteSPDE",
    "SpatialNoiseBasis",
    "SpectralSpatialDiscretization",
    "TensorGridDiscretization",
    "semidiscretize_reaction_diffusion",
    "semidiscretize_spde",
    "solve_diffrax",
    "solve_diffrax_ensemble",
    "FunctionalSolver",
    "EnforcedConstraintPipeline",
    "EnforcedConstraintPipelines",
    "EnforcedInteriorData",
    "SingleFieldEnforcedConstraint",
    "MultiFieldEnforcedConstraint",
]
