#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax.numpy as jnp
import pytest
from evosax.algorithms import DifferentialEvolution, Open_ES

from phydrax.solver import FunctionalSolver
from phydrax.solver._functional_train import solve


_DUMMY_SOLVER = cast(FunctionalSolver, object())


def test_population_based_evosax_is_rejected_with_search_space_guidance():
    algorithm = DifferentialEvolution(
        population_size=4,
        solution=jnp.zeros((2,)),
    )

    with pytest.raises(
        NotImplementedError,
        match=r"initial population.*DesignConstraintSystem\.search",
    ):
        solve(_DUMMY_SOLVER, num_iter=1, optim=algorithm)


def test_unrelated_optimizer_object_is_rejected_before_training():
    with pytest.raises(TypeError, match="Optax transformation"):
        solve(_DUMMY_SOLVER, num_iter=1, optim=object())


def test_evaluation_parameters_remains_optax_only_for_evosax():
    algorithm = Open_ES(
        population_size=8,
        solution=jnp.zeros((2,)),
    )

    with pytest.raises(ValueError, match="only for Optax"):
        solve(
            _DUMMY_SOLVER,
            num_iter=1,
            optim=algorithm,
            evaluation_parameters=lambda state, parameters: parameters,
        )
