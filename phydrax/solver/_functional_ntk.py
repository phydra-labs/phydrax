#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import partition_trainable
from ..linalg import ArraySpace, LinearizationPolicy
from ..nn.neural_tangent import (
    analyze_ntk,
    NTKDiagnostics,
    NTKDiagnosticsPolicy,
    prepare_empirical_ntk,
    PreparedEmpiricalNTK,
)
from ..nn.parameters import ParameterSubspace
from ..terms import ResidualBlockRef
from ._functional_residual import prepare_functional_residual, PreparedFunctionalResidual
from ._functional_surrogate import PreparedFunctionalUpdate


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


FunctionalNTKView = Literal["physical", "surrogate"]


class PreparedFunctionalNTK(StrictModule):
    """Empirical NTK of one prepared physical or optimizer residual map."""

    ntk: PreparedEmpiricalNTK
    residual: PreparedFunctionalResidual
    parameters: PyTree[Any]
    view: FunctionalNTKView = eqx.field(static=True)
    discretization_bundle_id: str = eqx.field(static=True)
    parameter_paths: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        ntk: PreparedEmpiricalNTK,
        residual: PreparedFunctionalResidual,
        parameters: PyTree[Any],
        /,
        *,
        view: FunctionalNTKView,
        discretization_bundle_id: str,
        parameter_paths: tuple[str, ...],
    ):
        if not isinstance(ntk, PreparedEmpiricalNTK):
            raise TypeError("ntk must be a PreparedEmpiricalNTK.")
        if not isinstance(residual, PreparedFunctionalResidual):
            raise TypeError("residual must be a PreparedFunctionalResidual.")
        if view not in ("physical", "surrogate"):
            raise ValueError("Unknown functional NTK view.")
        self.ntk = ntk
        self.residual = residual
        self.parameters = parameters
        self.view = view
        self.discretization_bundle_id = str(discretization_bundle_id)
        self.parameter_paths = tuple(parameter_paths)

    @property
    def kernel(self):
        return self.ntk.kernel

    @property
    def layout(self):
        return self.residual.layout

    def diagnostics(
        self,
        /,
        *,
        policy: NTKDiagnosticsPolicy | None = None,
        key: Key[Array, ""] | None = None,
    ) -> NTKDiagnostics:
        return analyze_ntk(self.ntk, policy=policy, key=key)

    def block(self, reference: ResidualBlockRef, /) -> PreparedEmpiricalNTK:
        """Prepare one term or named residual-block kernel from shared roots."""
        if not isinstance(reference, ResidualBlockRef):
            raise TypeError("reference must be a ResidualBlockRef.")
        if reference.block_name is None:
            pieces = tuple(
                jnp.arange(entry.start, entry.stop, dtype=jnp.int32)
                for entry in self.layout.entries
                if entry.term_index == reference.term_index
            )
            if not pieces:
                raise KeyError(f"Unknown residual term {reference.term_index}.")
            indices = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)
        else:
            indices = self.layout.logical_indices(
                reference.term_index, reference.block_name
            )

        def block_roots(parameters):
            return self.residual.roots(parameters)[indices]

        output = block_roots(self.parameters)
        return prepare_empirical_ntk(
            block_roots,
            self.parameters,
            parameter_space=self.ntk.parameter_space,
            output_space=ArraySpace(output.shape, dtype=output.dtype),
            linearization=self.ntk.linearization.policy,
            ntk_id=(
                f"{self.ntk.ntk_id}:term={reference.term_index}:"
                f"block={reference.block_name or '*'}"
            ),
        )


def prepare_functional_ntk(
    solver: FunctionalSolver,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    step: int | Array | None = None,
    term_indices: tuple[int, ...] | None = None,
    parameter_subspace: ParameterSubspace | None = None,
    prepared_update: PreparedFunctionalUpdate | None = None,
    view: FunctionalNTKView = "physical",
    linearization: LinearizationPolicy | None = None,
) -> PreparedFunctionalNTK:
    """Prepare the finite-width NTK of measure-weighted functional residuals."""
    if view not in ("physical", "surrogate"):
        raise ValueError("view must be 'physical' or 'surrogate'.")
    if prepared_update is not None and not isinstance(
        prepared_update, PreparedFunctionalUpdate
    ):
        raise TypeError("prepared_update must be PreparedFunctionalUpdate or None.")
    if view == "surrogate" and prepared_update is None:
        raise ValueError("A surrogate NTK requires its exact PreparedFunctionalUpdate.")
    if prepared_update is not None and parameter_subspace is not None:
        raise ValueError(
            "An already prepared update owns its exact parameter partition; "
            "parameter_subspace must be None."
        )
    if parameter_subspace is None:
        parameters, non_trainable = partition_trainable(solver.functions)
        paths = ()
    else:
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be ParameterSubspace or None.")
        parameter_subspace.validate_root(solver.functions)
        parameters = parameter_subspace.initial
        non_trainable = parameter_subspace.frozen
        paths = parameter_subspace.leaf_paths
    if prepared_update is None:
        indices = (
            tuple(range(len(solver.terms)))
            if term_indices is None
            else tuple(int(index) for index in term_indices)
        )
        physical = solver.objective.prepare_training(
            indices,
            scale=1.0,
            evaluation_key=key,
            sampling_key=jr.fold_in(key, 1),
            iteration=step,
        )
        residual = prepare_functional_residual(
            physical,
            parameters,
            non_trainable,
            solver.enforcement,
        )
    else:
        if view == "physical":
            residual = prepare_functional_residual(
                prepared_update.physical,
                parameters,
                non_trainable,
                solver.enforcement,
            )
        else:
            residual = prepared_update.residual
            if residual is None:
                raise ValueError("Prepared update contains no residual-root objective.")
        if term_indices is not None:
            raise ValueError("term_indices cannot modify an already prepared update.")
    initial_roots = residual.roots(parameters)
    ntk = prepare_empirical_ntk(
        residual.roots,
        parameters,
        output_space=ArraySpace(initial_roots.shape, dtype=initial_roots.dtype),
        linearization=linearization,
        ntk_id=(
            f"functional:{solver.discretization_bundle.bundle_id}:{view}:"
            f"step={step}"
        ),
    )
    return PreparedFunctionalNTK(
        ntk,
        residual,
        parameters,
        view=view,
        discretization_bundle_id=solver.discretization_bundle.bundle_id,
        parameter_paths=paths,
    )


__all__ = [
    "FunctionalNTKView",
    "PreparedFunctionalNTK",
    "prepare_functional_ntk",
]
