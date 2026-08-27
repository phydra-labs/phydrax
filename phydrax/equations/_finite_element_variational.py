#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._compensated import compensated_sum
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization._cell_complex import PolygonalConnectivity
from ..discretization.fem import (
    FiniteElementDirichletConstraint,
    FiniteElementDiscretization,
)
from ..dynamics import DAEStructure, DifferentialAlgebraicSystem
from ..linalg import DualSpace, FunctionLinearOperator, LinearSystem
from ..nonlinear import NonlinearSystemProblem
from ..sparse import SparseCoordinateOperator


class ResolvedCoefficient(StrictModule, NonTrainableState):
    """Typed constant or pure staged coefficient used by FE terms."""

    value: Array
    evaluator: Callable[[Array, object], ArrayLike] | None
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike | Callable[[Array, object], ArrayLike],
        /,
        *,
        coefficient_id: str | None = None,
    ):
        if callable(value):
            if coefficient_id is None or not str(coefficient_id):
                raise ValueError(
                    "Callable coefficients require an explicit coefficient_id."
                )
            self.value = jnp.asarray(0.0)
            self.evaluator = value
            self.coefficient_id = str(coefficient_id)
        else:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.inexact):
                array = array.astype(float)
            self.value = array
            self.evaluator = None
            self.coefficient_id = (
                canonical_fingerprint(
                    {
                        "kind": "finite-element-constant-coefficient",
                        "value": array_tree_fingerprint(np.asarray(array)),
                    }
                )
                if coefficient_id is None
                else str(coefficient_id)
            )
            if not self.coefficient_id:
                raise ValueError("coefficient_id must be non-empty.")

    @property
    def constant(self) -> bool:
        return self.evaluator is None

    def evaluate(self, points: Array, args: object = None, /) -> Array:
        if self.evaluator is None:
            return jnp.broadcast_to(self.value, points.shape[:-1] + self.value.shape)
        return jnp.asarray(self.evaluator(points, args))


def coefficient(
    value: ArrayLike | Callable[[Array, object], ArrayLike],
    /,
    *,
    coefficient_id: str | None = None,
) -> ResolvedCoefficient:
    return ResolvedCoefficient(value, coefficient_id=coefficient_id)


class DiffusionTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    diffusivity: ResolvedCoefficient
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        diffusivity: ResolvedCoefficient | ArrayLike = 1.0,
        /,
        *,
        term_id: str = "diffusion",
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Diffusion field and term IDs must be non-empty.")
        self.field_name = field
        self.diffusivity = (
            diffusivity
            if isinstance(diffusivity, ResolvedCoefficient)
            else coefficient(diffusivity)
        )
        self.term_id = identifier


class MassTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    coefficient: ResolvedCoefficient
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        value: ResolvedCoefficient | ArrayLike = 1.0,
        /,
        *,
        term_id: str = "mass",
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Mass field and term IDs must be non-empty.")
        self.field_name = field
        self.coefficient = (
            value if isinstance(value, ResolvedCoefficient) else coefficient(value)
        )
        self.term_id = identifier


class SourceTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    source: ResolvedCoefficient
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        source: ResolvedCoefficient | ArrayLike,
        /,
        *,
        term_id: str = "source",
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Source field and term IDs must be non-empty.")
        self.field_name = field
        self.source = (
            source if isinstance(source, ResolvedCoefficient) else coefficient(source)
        )
        self.term_id = identifier


class BoundaryLoadTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    load: ResolvedCoefficient
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        load: ResolvedCoefficient | ArrayLike,
        /,
        *,
        term_id: str = "boundary-load",
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Boundary-load field and term IDs must be non-empty.")
        self.field_name = field
        self.load = load if isinstance(load, ResolvedCoefficient) else coefficient(load)
        self.term_id = identifier


FiniteElementTerm = DiffusionTerm | MassTerm | SourceTerm | BoundaryLoadTerm


class _FiniteElementWorkBlock(StrictModule, NonTrainableState):
    block_name: str = eqx.field(static=True)
    cell_dofs: Array
    basis_values: Array
    physical_gradients: Array
    physical_points: Array
    physical_weights: Array
    work_id: str = eqx.field(static=True)


class WeakForm(StrictModule, NonTrainableState):
    form_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    terms: tuple[FiniteElementTerm, ...]

    def __init__(
        self,
        form_id: str,
        field_name: str,
        terms: Sequence[FiniteElementTerm],
        /,
    ):
        identifier = str(form_id)
        field = str(field_name)
        term_values = tuple(terms)
        if not identifier or not field:
            raise ValueError("Weak-form and field IDs must be non-empty.")
        if not term_values:
            raise ValueError("WeakForm requires at least one term.")
        if not all(
            isinstance(term, (DiffusionTerm, MassTerm, SourceTerm, BoundaryLoadTerm))
            for term in term_values
        ):
            raise TypeError("WeakForm contains an unsupported term type.")
        if any(term.field_name != field for term in term_values):
            raise ValueError("Every weak-form term must target the declared field.")
        term_ids = tuple(term.term_id for term in term_values)
        if len(set(term_ids)) != len(term_ids):
            raise ValueError("Weak-form term IDs must be unique.")
        self.form_id = identifier
        self.field_name = field
        self.terms = term_values


class FiniteElementFunctional(StrictModule, NonTrainableState):
    functional_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    density: Callable[[Array, Array, Array, object], ArrayLike]

    def __init__(
        self,
        functional_id: str,
        field_name: str,
        density: Callable[[Array, Array, Array, object], ArrayLike],
        /,
    ):
        identifier = str(functional_id)
        field = str(field_name)
        if not identifier or not field or not callable(density):
            raise ValueError("Functional ID, field, and callable density are required.")
        self.functional_id = identifier
        self.field_name = field
        self.density = density

    def evaluate(
        self,
        discretization: FiniteElementDiscretization,
        state: ArrayLike,
        args: object = None,
        /,
    ) -> Array:
        field_index = discretization._field_index(self.field_name)
        values = jnp.asarray(state)
        contributions = []
        for dofs, geometry in zip(
            discretization.dof_maps[field_index].cell_dofs,
            discretization.block_geometries[field_index],
            strict=True,
        ):
            local = values[dofs]
            field_values = oe.contract("qi,ci->cq", geometry.basis_values, local)
            gradients = oe.contract("cqid,ci->cqd", geometry.physical_gradients, local)
            density = jnp.asarray(
                self.density(field_values, gradients, geometry.physical_points, args)
            )
            if density.shape != geometry.physical_weights.shape:
                raise ValueError(
                    "Finite-element functional density must return one value per quadrature point."
                )
            contributions.append((density * geometry.physical_weights).reshape((-1,)))
        combined = jnp.concatenate(tuple(contributions))
        if discretization.precision_policy.compensated_accumulation:
            return compensated_sum(combined)
        return jnp.sum(combined)


class CompiledFiniteElementProblem(StrictModule, NonTrainableState):
    form: WeakForm
    discretization: FiniteElementDiscretization
    constraint: FiniteElementDirichletConstraint | None
    work_blocks: tuple[_FiniteElementWorkBlock, ...]
    lift: Array
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        form: WeakForm,
        discretization: FiniteElementDiscretization,
        /,
        *,
        constraint: FiniteElementDirichletConstraint | None = None,
        dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
    ):
        if not isinstance(form, WeakForm):
            raise TypeError("form must be a WeakForm.")
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        field_index = discretization._field_index(form.field_name)
        full_space = discretization.field_spaces[field_index].vector_space
        if constraint is None:
            if dirichlet_values is not None:
                raise ValueError("dirichlet_values require a finite-element constraint.")
            lift = jnp.zeros(
                full_space.structure().shape, dtype=full_space.structure().dtype
            )
        else:
            if not isinstance(constraint, FiniteElementDirichletConstraint):
                raise TypeError(
                    "constraint must be FiniteElementDirichletConstraint or None."
                )
            if constraint.field_name != form.field_name:
                raise ValueError("Constraint field must match the weak form field.")
            if dirichlet_values is None:
                raise ValueError("Constrained compilation requires dirichlet_values.")
            lift = constraint.lift(dirichlet_values)
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-finite-element-problem",
                "form": form.form_id,
                "discretization": discretization.prepared_id,
                "constraint": None if constraint is None else constraint.constraint_id,
            }
        )
        form_key = DiscretizationKey(
            "finite_element_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        self.form = form
        self.discretization = discretization
        self.constraint = constraint
        self.lift = lift
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                    precision_evidence_id=discretization.precision_evidence_id,
                    resource_evidence_id=discretization.resource_evidence_id,
                ),
                DiscretizationRecord(
                    form_key,
                    "compiled-finite-element-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        work_blocks = tuple(
            _FiniteElementWorkBlock(
                block_name=geometry.block_name,
                cell_dofs=dofs,
                basis_values=geometry.basis_values,
                physical_gradients=geometry.physical_gradients,
                physical_points=geometry.physical_points,
                physical_weights=geometry.physical_weights,
                work_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-work-block",
                        "compilation": compilation_id,
                        "block": geometry.block_name,
                        "cell_dofs": array_tree_fingerprint(np.asarray(dofs)),
                    }
                ),
            )
            for dofs, geometry in zip(
                discretization.dof_maps[field_index].cell_dofs,
                discretization.block_geometries[field_index],
                strict=True,
            )
        )
        self.work_blocks = work_blocks
        self.compilation_id = compilation_id

    @property
    def field_index(self) -> int:
        return self.discretization._field_index(self.form.field_name)

    @property
    def full_space(self):
        return self.discretization.field_spaces[self.field_index].vector_space

    @property
    def state_space(self):
        if self.constraint is None:
            return self.full_space
        return self.constraint.constraint_map.reduced_space

    @property
    def residual_space(self):
        return DualSpace(self.state_space)

    def expand(self, state: ArrayLike, /) -> Array:
        if self.constraint is None:
            return self.full_space.validate(state)
        return self.constraint.constraint_map.expand(state, self.lift)

    def full_residual(self, state: ArrayLike, args: object = None, /) -> Array:
        full = self.full_space.validate(state)
        return _full_residual(
            self.form,
            self.discretization,
            self.work_blocks,
            full,
            args,
        )

    def residual(self, state: ArrayLike, args: object = None, /) -> Array:
        full_residual = self.full_residual(self.expand(state), args)
        if self.constraint is None:
            return DualSpace(self.full_space).validate(full_residual)
        return self.constraint.constraint_map.pullback_dual(full_residual)

    def as_nonlinear_problem(self) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            lambda state, args: self.residual(state, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
            problem_id=self.compilation_id,
        )

    def affine_operator(self, args: object = None, /):
        field_index = self.field_index
        dof_map = self.discretization.dof_maps[field_index]
        coefficients = None
        relation = None
        for term in self.form.terms:
            if isinstance(term, DiffusionTerm):
                if not term.diffusivity.constant or term.diffusivity.value.shape != ():
                    raise ValueError(
                        "Sparse affine diffusion requires a scalar constant."
                    )
                operator = self.discretization.stiffness_operators[field_index]
                term_values = term.diffusivity.value * operator.coefficients
            elif isinstance(term, MassTerm):
                if not term.coefficient.constant or term.coefficient.value.shape != ():
                    raise ValueError("Sparse affine mass requires a scalar constant.")
                operator = self.discretization.mass_operators[field_index]
                term_values = term.coefficient.value * operator.coefficients
            else:
                continue
            if relation is None:
                relation = operator.relation
                coefficients = term_values
            else:
                if relation.route_shape != operator.relation.route_shape:
                    raise ValueError("Affine FE term sparse relations are incompatible.")
                coefficients = coefficients + term_values
        if relation is None or coefficients is None:
            raise ValueError("Weak form contains no affine operator term.")
        full_operator = SparseCoordinateOperator(
            relation,
            coefficients,
            source=self.full_space,
            target=DualSpace(self.full_space),
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-affine-operator",
                    "compilation": self.compilation_id,
                    "dof_map": dof_map.dof_map_id,
                }
            ),
        )
        if self.constraint is None:
            return full_operator
        constraint_map = self.constraint.constraint_map
        return FunctionLinearOperator(
            lambda reduced: constraint_map.pullback_dual(
                full_operator.mv(constraint_map.homogeneous_correction(reduced))
            ),
            source=constraint_map.reduced_space,
            target=DualSpace(constraint_map.reduced_space),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constrained-finite-element-affine-operator",
                    "compilation": self.compilation_id,
                }
            ),
        )

    def linear_system(self, args: object = None, /) -> tuple[LinearSystem, Array]:
        raw_operator = self.affine_operator(args)
        primal_operator = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_operator.mv(state)),
            source=self.state_space,
            target=self.state_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "riesz-finite-element-affine-operator",
                    "compilation": self.compilation_id,
                }
            ),
        )
        structure = self.state_space.structure()
        zero = jnp.zeros(structure.shape, dtype=structure.dtype)
        right_hand_side = self.state_space.inverse_riesz(-self.residual(zero, args))
        return LinearSystem(primal_operator), right_hand_side

    def as_dae_system(
        self,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
        system_id: str | None = None,
    ) -> DifferentialAlgebraicSystem:
        coefficient_ = jnp.asarray(mass_coefficient)
        if coefficient_.shape != ():
            raise ValueError("FE DAE mass_coefficient must be scalar.")
        field_index = self.field_index
        assembled = self.discretization.mass_operators[field_index]
        full_mass = SparseCoordinateOperator(
            assembled.relation,
            coefficient_ * assembled.coefficients,
            source=self.full_space,
            target=DualSpace(self.full_space),
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-dae-mass",
                    "compilation": self.compilation_id,
                }
            ),
        )
        if self.constraint is None:
            mass_operator = full_mass
        else:
            constraint_map = self.constraint.constraint_map
            mass_operator = FunctionLinearOperator(
                lambda reduced: constraint_map.pullback_dual(
                    full_mass.mv(constraint_map.homogeneous_correction(reduced))
                ),
                source=constraint_map.reduced_space,
                target=DualSpace(constraint_map.reduced_space),
                operator_id=canonical_fingerprint(
                    {
                        "kind": "constrained-finite-element-dae-mass",
                        "compilation": self.compilation_id,
                    }
                ),
            )
        structure = self.state_space.structure()
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-element-dae",
                    "compilation": self.compilation_id,
                }
            )
            if system_id is None
            else str(system_id)
        )
        return DifferentialAlgebraicSystem.from_mass_matrix(
            mass_operator,
            lambda time, state, args: -self.residual(state, (time, args)),
            state_shape=structure.shape,
            structure=DAEStructure(("differential",), component_axis=None),
            system_id=identifier,
        )


def _coefficient_values(
    coefficient_: ResolvedCoefficient,
    points: Array,
    args: object,
    /,
) -> Array:
    values = coefficient_.evaluate(points, args)
    expected = points.shape[:-1]
    if values.shape == ():
        return jnp.broadcast_to(values, expected)
    if values.shape != expected:
        raise ValueError(
            f"Finite-element scalar coefficient must return shape {expected}; got {values.shape}."
        )
    return values


def _scatter_local(residual: Array, dofs: Array, local: Array, /) -> Array:
    return residual.at[dofs.reshape((-1,))].add(local.reshape((-1,)))


def _full_residual(
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    work_blocks: tuple[_FiniteElementWorkBlock, ...],
    state: Array,
    args: object,
    /,
) -> Array:
    field_index = discretization._field_index(form.field_name)
    dof_map = discretization.dof_maps[field_index]
    residual = jnp.zeros((dof_map.global_dof_count,), dtype=state.dtype)
    for term in form.terms:
        if isinstance(term, BoundaryLoadTerm):
            residual = residual - _boundary_load(
                discretization, field_index, term.load, args
            )
            continue
        for work in work_blocks:
            dofs = work.cell_dofs
            local_state = state[dofs]
            if isinstance(term, DiffusionTerm):
                field_gradient = oe.contract(
                    "cqid,ci->cqd", work.physical_gradients, local_state
                )
                values = _coefficient_values(term.diffusivity, work.physical_points, args)
                local = oe.contract(
                    "cq,cq,cqid,cqd->ci",
                    work.physical_weights,
                    values,
                    work.physical_gradients,
                    field_gradient,
                )
            elif isinstance(term, MassTerm):
                field_value = oe.contract("qi,ci->cq", work.basis_values, local_state)
                values = _coefficient_values(term.coefficient, work.physical_points, args)
                local = oe.contract(
                    "cq,cq,qi,cq->ci",
                    work.physical_weights,
                    values,
                    work.basis_values,
                    field_value,
                )
            elif isinstance(term, SourceTerm):
                values = _coefficient_values(term.source, work.physical_points, args)
                local = -oe.contract(
                    "cq,cq,qi->ci",
                    work.physical_weights,
                    values,
                    work.basis_values,
                )
            else:
                raise TypeError("Unsupported finite-element term.")
            residual = _scatter_local(residual, dofs, local)
    return DualSpace(discretization.field_spaces[field_index].vector_space).validate(
        residual
    )


def _boundary_load(
    discretization: FiniteElementDiscretization,
    field_index: int,
    load: ResolvedCoefficient,
    args: object,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("BoundaryLoadTerm currently supports polygonal meshes.")
    boundary = np.flatnonzero(np.asarray(connectivity.boundary_edges, dtype=bool))
    edge_vertices = jnp.asarray(connectivity.edges)[boundary]
    points = discretization.mesh.coordinates[edge_vertices]
    midpoints = 0.5 * (points[:, 0] + points[:, 1])
    measures = jnp.sqrt(jnp.sum((points[:, 1] - points[:, 0]) ** 2, axis=-1))
    values = _coefficient_values(load, midpoints, args)
    result = jnp.zeros(
        (discretization.dof_maps[field_index].global_dof_count,),
        dtype=values.dtype,
    )
    vertex_values = 0.5 * measures * values
    result = result.at[edge_vertices.reshape((-1,))].add(jnp.repeat(vertex_values, 2))
    if (
        discretization.dof_maps[field_index].global_dof_count
        > discretization.mesh.coordinates.shape[0]
    ):
        edge_dofs = int(discretization.mesh.coordinates.shape[0]) + jnp.asarray(boundary)
        result = result.at[edge_vertices.reshape((-1,))].add(
            -jnp.repeat(2.0 * vertex_values / 3.0, 2)
        )
        result = result.at[edge_dofs].add(4.0 * vertex_values / 3.0)
    return result


def compile_finite_element_problem(
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    /,
    *,
    constraint: FiniteElementDirichletConstraint | None = None,
    dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
) -> CompiledFiniteElementProblem:
    return CompiledFiniteElementProblem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=dirichlet_values,
    )


__all__ = [
    "BoundaryLoadTerm",
    "CompiledFiniteElementProblem",
    "DiffusionTerm",
    "FiniteElementFunctional",
    "MassTerm",
    "ResolvedCoefficient",
    "SourceTerm",
    "WeakForm",
    "coefficient",
    "compile_finite_element_problem",
]
