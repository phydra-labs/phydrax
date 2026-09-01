#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from ..._strict import StrictModule
from ...linalg import AbstractLinearOperator, AbstractVectorSpace, LinearSubspace
from ...linalg.eigen import Eigenproblem, GeneralizedEigenproblem
from ._equilibrium import MechanicsEquilibriumProblem


StaticEigenvalueQuantity: TypeAlias = Literal["physical-tangent-curvature"]
DynamicEigenvalueQuantity: TypeAlias = Literal["squared-angular-frequency"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _equilibrium(value: MechanicsEquilibriumProblem, /) -> MechanicsEquilibriumProblem:
    if not isinstance(value, MechanicsEquilibriumProblem):
        raise TypeError("equilibrium must be a MechanicsEquilibriumProblem.")
    return value


def _physical_space(
    equilibrium: MechanicsEquilibriumProblem,
    space: AbstractVectorSpace,
    /,
) -> AbstractVectorSpace:
    if not isinstance(space, AbstractVectorSpace):
        raise TypeError("physical_space must be an AbstractVectorSpace.")
    parameter_space = equilibrium.state_space
    if (
        equilibrium.root_coordinates == "field-parameters"
        and parameter_space is not None
        and space.compatible(parameter_space)
    ):
        raise ValueError(
            "A field-parameter space cannot serve as physical stability space; "
            "supply an explicitly identified physical space and operator."
        )
    return space


def _physical_endomorphism(
    operator: AbstractLinearOperator,
    space: AbstractVectorSpace,
    name: str,
    /,
) -> AbstractLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError(f"{name} must be an AbstractLinearOperator.")
    if not operator.source.compatible(space) or not operator.target.compatible(space):
        raise ValueError(f"{name} must be an endomorphism on physical_space.")
    return operator


def _constraints(
    value: LinearSubspace | None,
    space: AbstractVectorSpace,
    /,
) -> LinearSubspace | None:
    if value is not None and not isinstance(value, LinearSubspace):
        raise TypeError("constraints must be a LinearSubspace or None.")
    if value is not None and not value.space.compatible(space):
        raise ValueError("constraints must belong to physical_space.")
    return value


class PhysicalStaticStabilityProblem(StrictModule):
    """Certified physical tangent spectrum at one mechanics equilibrium.

    Construction requires a separately identified physical vector space and a
    self-adjoint operator carrying non-unknown symmetry evidence. Parameter-space
    Hessians of field models are deliberately not promoted to this contract.
    """

    physical_space: AbstractVectorSpace
    tangent_operator: AbstractLinearOperator
    constraints: LinearSubspace | None
    equilibrium_problem_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    equilibrium_provenance_id: str = eqx.field(static=True)
    tangent_provenance_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: MechanicsEquilibriumProblem,
        physical_space: AbstractVectorSpace,
        tangent_operator: AbstractLinearOperator,
        /,
        *,
        tangent_provenance_id: str,
        constraints: LinearSubspace | None = None,
        problem_id: str | None = None,
    ):
        equilibrium_ = _equilibrium(equilibrium)
        space = _physical_space(equilibrium_, physical_space)
        tangent = _physical_endomorphism(
            tangent_operator,
            space,
            "tangent_operator",
        )
        constraints_ = _constraints(constraints, space)
        identifier = (
            f"{equilibrium_.problem_id}:physical-static-stability"
            if problem_id is None
            else _identifier(problem_id, "problem_id")
        )
        Eigenproblem(tangent, constraints=constraints_, problem_id=identifier)
        self.physical_space = space
        self.tangent_operator = tangent
        self.constraints = constraints_
        self.equilibrium_problem_id = equilibrium_.problem_id
        self.realization_id = equilibrium_.realization_id
        self.equilibrium_provenance_id = equilibrium_.provenance_id
        self.tangent_provenance_id = _identifier(
            tangent_provenance_id,
            "tangent_provenance_id",
        )
        self.problem_id = identifier

    @property
    def eigenvalue_quantity(self) -> StaticEigenvalueQuantity:
        return "physical-tangent-curvature"

    def as_eigenproblem(self, /) -> Eigenproblem:
        return Eigenproblem(
            self.tangent_operator,
            constraints=self.constraints,
            problem_id=self.problem_id,
        )


class DynamicStabilityProblem(StrictModule):
    """Physical vibration pencil ``K phi = omega^2 M phi``.

    ``stiffness_operator`` must be certified self-adjoint and ``mass_operator``
    must additionally carry certified positive-definite evidence on the same
    explicit physical space.
    """

    physical_space: AbstractVectorSpace
    stiffness_operator: AbstractLinearOperator
    mass_operator: AbstractLinearOperator
    constraints: LinearSubspace | None
    equilibrium_problem_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    equilibrium_provenance_id: str = eqx.field(static=True)
    stiffness_provenance_id: str = eqx.field(static=True)
    mass_provenance_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: MechanicsEquilibriumProblem,
        physical_space: AbstractVectorSpace,
        stiffness_operator: AbstractLinearOperator,
        mass_operator: AbstractLinearOperator,
        /,
        *,
        stiffness_provenance_id: str,
        mass_provenance_id: str,
        constraints: LinearSubspace | None = None,
        problem_id: str | None = None,
    ):
        equilibrium_ = _equilibrium(equilibrium)
        space = _physical_space(equilibrium_, physical_space)
        stiffness = _physical_endomorphism(
            stiffness_operator,
            space,
            "stiffness_operator",
        )
        mass = _physical_endomorphism(mass_operator, space, "mass_operator")
        constraints_ = _constraints(constraints, space)
        identifier = (
            f"{equilibrium_.problem_id}:dynamic-stability"
            if problem_id is None
            else _identifier(problem_id, "problem_id")
        )
        GeneralizedEigenproblem(
            stiffness,
            mass,
            constraints=constraints_,
            problem_id=identifier,
        )
        if stiffness.batch_shape != mass.batch_shape:
            raise ValueError("Dynamic pencil operators must share one batch shape.")
        self.physical_space = space
        self.stiffness_operator = stiffness
        self.mass_operator = mass
        self.constraints = constraints_
        self.equilibrium_problem_id = equilibrium_.problem_id
        self.realization_id = equilibrium_.realization_id
        self.equilibrium_provenance_id = equilibrium_.provenance_id
        self.stiffness_provenance_id = _identifier(
            stiffness_provenance_id,
            "stiffness_provenance_id",
        )
        self.mass_provenance_id = _identifier(
            mass_provenance_id,
            "mass_provenance_id",
        )
        self.problem_id = identifier

    @property
    def eigenvalue_quantity(self) -> DynamicEigenvalueQuantity:
        return "squared-angular-frequency"

    def as_generalized_eigenproblem(self, /) -> GeneralizedEigenproblem:
        return GeneralizedEigenproblem(
            self.stiffness_operator,
            self.mass_operator,
            constraints=self.constraints,
            problem_id=self.problem_id,
        )


__all__ = [
    "DynamicEigenvalueQuantity",
    "DynamicStabilityProblem",
    "PhysicalStaticStabilityProblem",
    "StaticEigenvalueQuantity",
]
