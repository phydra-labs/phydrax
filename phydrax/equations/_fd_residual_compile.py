#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import PreparedTensorGrid
from ..linalg import (
    ArraySpace,
    JacobianLinearOperator,
    LinearSystem,
    prepare_linearization,
)
from ..typing import checked
from ._fd_compile import (
    _FiniteDifferenceExpressionEvaluator,
    _prepare_fd_lowering,
    FiniteDifferenceCompilationPolicy,
)
from ._ir import depends_on_fields, field_polynomial_degree, PDEExpression, PDEProblemIR
from ._validate import validate_pde_ir


@final
class ReducedStencilStateLayout(StrictModule):
    """Host-prepared topology for field-identified Dirichlet elimination."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    free_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    total_free: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    @checked
    def __init__(self, evaluator: _FiniteDifferenceExpressionEvaluator, /) -> None:
        names = evaluator.layout.field_names
        shape = evaluator.layout.spatial_shape
        indices = []
        offsets = [0]
        for name in names:
            mask = np.ones(shape, dtype=np.bool_)
            for pair in evaluator.boundary_program.pairs_for_field(name):
                runtime = pair.nodal_runtime
                if runtime is None:
                    continue
                for side, kind in ((0, runtime.lower_kind), (-1, runtime.upper_kind)):
                    if kind == "dirichlet":
                        index = [slice(None)] * len(shape)
                        index[runtime.axis] = (
                            slice(0, 1) if side == 0 else slice(-1, None)
                        )
                        mask[tuple(index)] = False
            free = tuple(np.flatnonzero(mask.reshape((-1,))).tolist())
            indices.append(free)
            offsets.append(offsets[-1] + len(free))
        if offsets[-1] == 0:
            raise ValueError("Stationary FD compilation requires at least one free node.")
        self.field_names = names
        self.free_indices = tuple(indices)
        self.offsets = tuple(offsets)
        self.spatial_shape = shape
        self.total_free = offsets[-1]
        self.layout_id = canonical_fingerprint(
            {
                "kind": "reduced-stencil-state-layout",
                "fields": names,
                "free_indices": self.free_indices,
                "offsets": self.offsets,
                "spatial_shape": shape,
            }
        )

    def scatter(self, state: Array, /) -> dict[str, Array]:
        return {
            name: jnp.zeros(self.spatial_shape, dtype=state.dtype)
            .reshape((-1,))
            .at[jnp.asarray(indices, dtype=jnp.int32)]
            .set(state[start:stop])
            .reshape(self.spatial_shape)
            for name, indices, start, stop in zip(
                self.field_names,
                self.free_indices,
                self.offsets[:-1],
                self.offsets[1:],
                strict=True,
            )
        }

    def gather(self, fields: Mapping[str, Array], /) -> Array:
        return jnp.concatenate(
            tuple(
                fields[name].reshape((-1,))[jnp.asarray(indices, dtype=jnp.int32)]
                for name, indices in zip(self.field_names, self.free_indices, strict=True)
            )
        )


@final
class CompiledFiniteDifferenceResidual(StrictModule):
    """Stationary native FD residual in explicit, Dirichlet-reduced coordinates.

    Neumann and Robin nodes remain unknowns. Their coordinate-derivative laws
    are realized by the same boundary evaluator used for native FD dynamics.
    """

    evaluator: _FiniteDifferenceExpressionEvaluator
    layout: ReducedStencilStateLayout
    state_space: ArraySpace
    residual_space: ArraySpace
    equation_targets: tuple[tuple[str, str], ...] = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        evaluator: _FiniteDifferenceExpressionEvaluator,
        equation_targets: tuple[tuple[str, str], ...],
        /,
    ) -> None:
        layout = ReducedStencilStateLayout(evaluator)
        space = ArraySpace(
            (layout.total_free,), dtype=evaluator.discretization.grid.points.dtype
        )
        affine = all(
            field_polynomial_degree(expression) in (0, 1)
            for _, expression in evaluator.equations
        )
        affine = affine and all(
            not depends_on_fields(binding.target.expression)
            for pair in evaluator.boundary_program.pairs
            for binding in (pair.lower, pair.upper)
            if binding.kind != "dirichlet"
        )
        affine = affine and all(
            not depends_on_fields(binding.target.expression)
            for interface in evaluator.interfaces
            for binding in (interface.field_binding, interface.flux_binding)
            if binding is not None
        )
        self.evaluator = evaluator
        self.layout = layout
        self.state_space = space
        self.residual_space = space
        self.equation_targets = equation_targets
        self.affine = affine
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-native-fd-residual",
                "problem": evaluator.problem.canonical_hash,
                "grid": evaluator.discretization.grid.prepared_id,
                "discretization": evaluator.discretization.prepared_id,
                "policy": evaluator.policy.policy_id,
                "boundary_program": evaluator.boundary_program.program_id,
                "interfaces": [value.interface_id for value in evaluator.interfaces],
                "equation_targets": equation_targets,
                "reduced_layout": layout.layout_id,
            }
        )

    def expand(self, state: Array, args: Any, /) -> dict[str, Array]:
        """Scatter free coordinates and impose runtime Dirichlet state rows."""
        fields = self.layout.scatter(self.state_space.validate(state))
        program = self.evaluator.boundary_program
        context = program.stage_context(
            jnp.zeros((), dtype=state.dtype), fields, args, stage_id="native-fd-residual"
        )
        return {
            name: program.constrain_state(name, value, context)
            for name, value in fields.items()
        }

    def residual(self, state: Array, args: Any, /) -> Array:
        """Evaluate lhs minus rhs at each explicitly targeted field's free nodes."""
        fields = self.expand(state, args)
        evaluator = self.evaluator
        program = evaluator.boundary_program
        context = program.stage_context(
            jnp.zeros((), dtype=state.dtype), fields, args, stage_id="native-fd-residual"
        )
        workspaces = {
            name: program.workspace(name, value, context)
            for name, value in fields.items()
        }
        values = {
            name: jnp.broadcast_to(
                evaluator.evaluate(expression, fields, args, context, workspaces),
                self.layout.spatial_shape,
            )
            for name, expression in evaluator.equations
        }
        return self.residual_space.validate(self.layout.gather(values))

    def linearization(self, state: Array, args: Any, /) -> JacobianLinearOperator:
        """Prepare reusable matrix-free JVP and transpose actions at one state."""

        def evaluate(value: Array) -> Array:
            return self.residual(value, args)

        prepared = prepare_linearization(
            evaluate,
            state,
            source=self.state_space,
            target=self.residual_space,
            linearization_id=canonical_fingerprint(
                {"kind": "fd-residual-linearization", "compilation": self.compilation_id}
            ),
        )
        return JacobianLinearOperator(prepared)

    def linear_system(self, args: Any, /) -> tuple[LinearSystem, Array]:
        """Return the exact affine system A x = b, refusing nonlinear contracts."""
        if not self.affine:
            raise ValueError(
                "linear_system requires affine field dependence and field-independent boundary targets."
            )
        operator = self.linearization(self.state_space.zeros(), args)
        return LinearSystem(operator), -operator.linearization.primal


def _require_boundary_target(
    expression: PDEExpression, axis_names: tuple[str, ...], /
) -> None:
    if expression.op not in (
        "constant",
        "coordinate",
        "field",
        "parameter",
        "negate",
        "add",
        "multiply",
        "divide",
        "power",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
    ):
        raise ValueError(
            f"Unrepresentable FD boundary target operation {expression.op!r}."
        )
    if expression.op == "coordinate" and expression.symbol not in axis_names:
        raise ValueError("Stationary FD boundary target coordinates must be grid axes.")
    for argument in expression.args:
        _require_boundary_target(argument, axis_names)


@checked
def compile_finite_difference_residual(
    problem: PDEProblemIR,
    grid: PreparedTensorGrid,
    /,
    *,
    equation_targets: Mapping[str, str],
    policy: FiniteDifferenceCompilationPolicy | None = None,
) -> CompiledFiniteDifferenceResidual:
    """Compile a stationary scalar PDE with explicit equation-to-field targets.

    Runtime arguments supply forcing, coefficients, and boundary data. Dirichlet
    nodes are eliminated; the remaining state and residual are flat ArraySpaces.
    """
    validate_pde_ir(problem)
    if any(coordinate.kind == "time" for coordinate in problem.coordinates):
        raise ValueError(
            "Stationary FD residual compilation does not admit a time coordinate."
        )
    if any(
        field.components != 1 or field.representation not in ("scalar", "pseudoscalar")
        for field in problem.fields
    ):
        raise ValueError(
            "Stationary FD residual compilation supports only scalar fields."
        )
    if not isinstance(equation_targets, Mapping):
        raise TypeError("equation_targets must be an explicit equation-to-field mapping.")
    names = {equation.name for equation in problem.equations}
    fields = {field.name for field in problem.fields}
    targets = dict(equation_targets)
    if (
        set(targets) != names
        or set(targets.values()) != fields
        or len(targets) != len(fields)
    ):
        raise ValueError(
            "equation_targets must define a bijection from equations to fields."
        )
    policy_ = FiniteDifferenceCompilationPolicy() if policy is None else policy
    equations = tuple(
        (targets[equation.name], equation.residual) for equation in problem.equations
    )
    evaluator = _prepare_fd_lowering(problem, grid, equations, policy_)
    if any(
        binding.kind == "dirichlet" and depends_on_fields(binding.target.expression)
        for pair in evaluator.boundary_program.pairs
        for binding in (pair.lower, pair.upper)
    ):
        raise ValueError("Dirichlet targets must be independent of solved fields.")
    for pair in evaluator.boundary_program.pairs:
        for binding in (pair.lower, pair.upper):
            _require_boundary_target(binding.target.expression, grid.axis_names)
    for interface in evaluator.interfaces:
        for interface_binding in (interface.field_binding, interface.flux_binding):
            if interface_binding is not None:
                _require_boundary_target(
                    interface_binding.target.expression, grid.axis_names
                )
    return CompiledFiniteDifferenceResidual(evaluator, tuple(sorted(targets.items())))


__all__ = ["CompiledFiniteDifferenceResidual", "compile_finite_difference_residual"]
