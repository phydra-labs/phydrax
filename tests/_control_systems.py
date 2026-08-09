#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Any

from jaxtyping import Array, ArrayLike

from phydrax.control import DifferentialControlDynamics, DiscreteControlDynamics
from phydrax.dynamics import (
    ContinuousSystem,
    DiscreteSystem,
    InputLayout,
    StateLayout,
)


def make_discrete_control_dynamics(
    transition: Callable[[Array, Array, Array, Any], ArrayLike],
    /,
    *,
    state_shape: Sequence[int],
    control_shape: Sequence[int],
    dynamics_id: str,
    method_id: str = "explicit-discrete-transition",
) -> DiscreteControlDynamics:
    system = DiscreteSystem(
        transition,
        state_layout=StateLayout(state_shape),
        input_layout=InputLayout(control_shape),
        system_id=dynamics_id,
    )
    return DiscreteControlDynamics(system, method_id=method_id)


def make_differential_control_dynamics(
    vector_field: Callable[[Array, Array, Array, Any], ArrayLike],
    /,
    *,
    state_shape: Sequence[int],
    control_shape: Sequence[int],
    dynamics_id: str,
    method_id: str = "canonical-differential-problem",
) -> DifferentialControlDynamics:
    system = ContinuousSystem(
        vector_field,
        state_layout=StateLayout(state_shape),
        input_layout=InputLayout(control_shape),
        system_id=dynamics_id,
    )
    return DifferentialControlDynamics(system, method_id=method_id)
