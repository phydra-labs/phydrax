#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native box programs of an implicit boundary field for zero-set certificates."""

from __future__ import annotations

import jax
from jax import Array

from ..._meshcore import IntervalBoxProgram, IntervalBranchProgram
from .._contracts import CompiledGeometry
from .._interval_enclosure import prepare_interval_function


def implicit_zero_set_programs(
    geometry: CompiledGeometry, /
) -> tuple[IntervalBranchProgram, IntervalBoxProgram]:
    """Lower f, with its branch-evidence events, and grad f of the boundary field.

    f's ``IntervalBranchProgram`` is the one program a zero-set certificate
    decides corner signs, continuity and admissions by. Raises
    ``IntervalExtensionError`` when either does not lower natively.
    """
    kernel, state = geometry.kernel, geometry.state

    def value(point: Array) -> Array:
        return kernel.boundary_field(state, point[None, :])[0]

    programs = tuple(
        prepare_interval_function(function, 3, batch_capacity=1, execution="native")
        for function in (value, jax.grad(value))
    )
    value_program, gradient_program = (program.native_program for program in programs)
    if value_program is None or gradient_program is None:
        raise RuntimeError("Native interval preparation returned no native program.")
    lowered = programs[0].native_branches().branch_box_program()
    return lowered, gradient_program.box_program()
