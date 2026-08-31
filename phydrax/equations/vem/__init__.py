#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Projection-aware virtual-element forms and compiled systems."""

from ._compiler import compile_virtual_element_problem, CompiledVirtualElementProblem
from ._form import (
    VirtualElementAction,
    VirtualElementExecutionContext,
    VirtualElementExecutionPolicy,
    VirtualElementForm,
    VirtualElementRobinAction,
)
from ._reconstruction import (
    evaluate_virtual_element_reconstruction,
    evaluate_virtual_element_trace,
    project_virtual_element_field,
    VirtualElementReconstruction,
)


__all__ = [
    "CompiledVirtualElementProblem",
    "VirtualElementAction",
    "VirtualElementExecutionContext",
    "VirtualElementExecutionPolicy",
    "VirtualElementForm",
    "VirtualElementReconstruction",
    "VirtualElementRobinAction",
    "compile_virtual_element_problem",
    "evaluate_virtual_element_reconstruction",
    "evaluate_virtual_element_trace",
    "project_virtual_element_field",
]
