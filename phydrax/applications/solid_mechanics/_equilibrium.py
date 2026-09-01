#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ...linalg import AbstractVectorSpace
from ...nonlinear import NonlinearSystemProblem
from ...solver._field_equilibrium import PreparedFieldEquilibrium


MechanicsRootCoordinates: TypeAlias = Literal[
    "physical-state",
    "field-parameters",
]


def _identifier(value: str | None, name: str, /) -> str:
    if value is None:
        raise ValueError(f"{name} must be supplied.")
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class MechanicsEquilibriumProblem(StrictModule):
    """Mechanical semantics around exactly one authoritative nonlinear root.

    The wrapper never copies or reconstructs the residual. Optional physical
    admissibility is evaluated as an additional acceptance condition while the
    wrapped ``root_problem`` remains the sole equation and numerical-validity
    authority.
    """

    root_problem: NonlinearSystemProblem
    admissibility_function: Callable[[PyTree[Any], PyTree[Any], Any, Any], Any] | None
    realization_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    admissibility_id: str | None = eqx.field(static=True)
    root_coordinates: MechanicsRootCoordinates = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: NonlinearSystemProblem | PreparedFieldEquilibrium,
        /,
        *,
        realization_id: str | None = None,
        provenance_id: str | None = None,
        admissibility: (
            Callable[[PyTree[Any], PyTree[Any], Any, Any], Any] | None
        ) = None,
        admissibility_id: str | None = None,
        root_coordinates: MechanicsRootCoordinates | None = None,
        problem_id: str | None = None,
    ):
        if isinstance(problem, PreparedFieldEquilibrium):
            root = problem.problem
            prepared_realization = problem.realization_id
            prepared_provenance = problem.provenance_id
            if realization_id is not None and str(realization_id) != prepared_realization:
                raise ValueError(
                    "Mechanics realization does not match the prepared field root."
                )
            if provenance_id is not None and str(provenance_id) != prepared_provenance:
                raise ValueError(
                    "Mechanics provenance does not match the prepared field root."
                )
            realization = prepared_realization
            provenance = prepared_provenance
            coordinates: MechanicsRootCoordinates = "field-parameters"
            if root_coordinates is not None and root_coordinates != coordinates:
                raise ValueError(
                    "Prepared field equilibrium has field-parameter root coordinates."
                )
        elif isinstance(problem, NonlinearSystemProblem):
            root = problem
            realization = _identifier(realization_id, "realization_id")
            provenance = _identifier(provenance_id, "provenance_id")
            coordinates = (
                "physical-state" if root_coordinates is None else root_coordinates
            )
        else:
            raise TypeError(
                "problem must be a NonlinearSystemProblem or PreparedFieldEquilibrium."
            )
        if coordinates not in ("physical-state", "field-parameters"):
            raise ValueError(
                "root_coordinates must be 'physical-state' or 'field-parameters'."
            )
        if admissibility is not None and not callable(admissibility):
            raise TypeError("admissibility must be callable or None.")
        if (admissibility is None) != (admissibility_id is None):
            raise ValueError(
                "admissibility and admissibility_id must be supplied together."
            )
        admissibility_identity = (
            None
            if admissibility_id is None
            else _identifier(admissibility_id, "admissibility_id")
        )
        identifier = root.problem_id if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.root_problem = root
        self.admissibility_function = admissibility
        self.realization_id = realization
        self.provenance_id = provenance
        self.admissibility_id = admissibility_identity
        self.root_coordinates = coordinates
        self.problem_id = identifier

    @property
    def state_space(self) -> AbstractVectorSpace | None:
        return self.root_problem.state_space

    @property
    def residual_space(self) -> AbstractVectorSpace | None:
        return self.root_problem.residual_space

    def evaluate(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], Any]:
        return self.root_problem.evaluate(state, args)

    def residual(self, state: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        return self.root_problem.residual(state, args)

    def valid(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        args: Any = None,
        /,
    ) -> Array:
        numerical = jnp.asarray(
            self.root_problem.valid(state, residual, auxiliary, args),
            dtype=bool,
        )
        if numerical.shape != ():
            raise ValueError("Mechanics root validity must return one scalar boolean.")
        if self.admissibility_function is None:
            return numerical.reshape(())
        physical = jnp.asarray(
            self.admissibility_function(state, residual, auxiliary, args),
            dtype=bool,
        )
        if physical.shape != ():
            raise ValueError("Mechanics admissibility must return one scalar boolean.")
        return numerical & physical.reshape(())

    def admissible(self, state: PyTree[Any], args: Any = None, /) -> Array:
        """Evaluate the root once and combine numerical and physical validity."""
        residual, auxiliary = self.root_problem.evaluate(state, args)
        return self.valid(state, residual, auxiliary, args)


__all__ = [
    "MechanicsEquilibriumProblem",
    "MechanicsRootCoordinates",
]
