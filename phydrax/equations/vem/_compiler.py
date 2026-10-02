#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._polynomial import ScaledMonomialBasis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    AbstractSideFluxEvaluator,
    BoundaryImposition,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    ImpositionKind,
    IntegrationDomain,
    PreparedFluxAction,
    PreparedTraceAction,
    SideActionDescriptor,
)
from ...discretization._polygon_query import (
    polygon_facet_domain,
    polygon_side_frame,
    polygon_side_revision,
    polygonal_connectivity_of,
)
from ...discretization.vem import (
    FactorizedVirtualElementOperator,
    stabilize_virtual_element_tensor,
    VirtualElementDirichletConstraint,
    VirtualElementDiscretization,
    VirtualElementProjectionData,
    VirtualElementRuntimeData,
    VirtualElementStabilizationPolicy,
)
from ...discretization.vem._space import virtual_element_runtime_matches
from ...dynamics import DAEStructure, DifferentialAlgebraicSystem
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ConstraintMap,
    DualSpace,
    FunctionLinearOperator,
    LinearSubspace,
    LinearSystem,
    NullspacePolicy,
    OperatorProperties,
    plan_sparse_assembly,
    prepare_sparse_assembly,
    PreparedSparseAssembly,
    SparseAssemblyPlan,
    SparseAssemblyPolicy,
)
from ...linalg.eigen import GeneralizedEigenproblem
from ...sparse import EdgeRelation, scatter_local, SparseCoordinateOperator
from ...typing import checked
from .._variational import (
    BoundaryLoadAction,
    DiffusionAction,
    MassAction,
    SourceAction,
    VariationalCoefficient,
)
from ._form import (
    VirtualElementAction,
    VirtualElementExecutionContext,
    VirtualElementExecutionPolicy,
    VirtualElementForm,
    VirtualElementRobinAction,
)
from ._reconstruction import virtual_element_projection_basis


def _cell_indices(
    discretization: VirtualElementDiscretization, block_index: int, /
) -> Array:
    offset = sum(block.cell_count for block in discretization.mesh.blocks[:block_index])
    count = discretization.mesh.blocks[block_index].cell_count
    return jnp.arange(offset, offset + count, dtype=jnp.int32)


def _cell_mask(action: VirtualElementAction, indices: Array, /) -> Array:
    if action.domain is None:
        return jnp.ones(indices.shape, dtype=jnp.bool_)
    selected = jnp.asarray(action.domain.entity_indices, dtype=jnp.int32)
    return jnp.any(indices[:, None] == selected[None, :], axis=1)


def _coefficient_values(
    coefficient: VariationalCoefficient,
    points: Array,
    indices: Array,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    /,
) -> Array:
    return coefficient.evaluate(
        points,
        context.user_args,
        entity_indices=indices,
        support_id=discretization.support.support_id,
        entity_set_id=discretization.mesh.topology.entity_sets[2].entity_set_id,
    )


def _diffusion_polynomial_matrices(
    action: DiffusionAction,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    /,
) -> tuple[Array, ...]:
    family = discretization.field.element.family
    if family == "DiscontinuousL2":
        raise ValueError("Diffusion is undefined on discontinuous L2 VEM spaces.")
    result = []
    for block_index, (geometry, cubature, projection) in enumerate(
        zip(
            context.runtime.geometries,
            context.runtime.cubatures,
            context.runtime.projections,
            strict=True,
        )
    ):
        indices = _cell_indices(discretization, block_index)
        values = _coefficient_values(
            action.diffusivity, cubature.points, indices, discretization, context
        )
        mask = _cell_mask(action, indices)
        if family == "ConformingH1":
            basis_gradient = projection.basis.gradient(
                cubature.points,
                geometry.centroids,
                geometry.characteristic_lengths,
            )
            if values.shape == cubature.weights.shape:
                matrix = ein.contract(
                    "cq,cq,cqad,cqbd->cab",
                    cubature.weights,
                    values,
                    basis_gradient,
                    basis_gradient,
                )
            elif values.shape == cubature.weights.shape + (2, 2):
                matrix = ein.contract(
                    "cq,cqad,cqde,cqbe->cab",
                    cubature.weights,
                    basis_gradient,
                    values,
                    basis_gradient,
                )
            else:
                raise ValueError(
                    "H1 VEM diffusivity must be scalar or a 2x2 tensor per cell point."
                )
        else:
            if values.shape != cubature.weights.shape:
                raise ValueError(
                    f"{discretization.field.element.differential_kind} VEM diffusivity must be scalar at cell points."
                )
            differential_basis = ScaledMonomialBasis(2, projection.differential_degree)
            basis_values = differential_basis.evaluate(
                cubature.points,
                geometry.centroids,
                geometry.characteristic_lengths,
            )
            matrix = ein.contract(
                "cq,cq,cqa,cqb->cab",
                cubature.weights,
                values,
                basis_values,
                basis_values,
            )
        result.append(jnp.where(mask[:, None, None], matrix, 0.0))
    return tuple(result)


def _mass_polynomial_matrices(
    coefficient: VariationalCoefficient,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    /,
    *,
    domain: IntegrationDomain | None = None,
) -> tuple[Array, ...]:
    result = []
    for block_index, (geometry, cubature, projection) in enumerate(
        zip(
            context.runtime.geometries,
            context.runtime.cubatures,
            context.runtime.projections,
            strict=True,
        )
    ):
        indices = _cell_indices(discretization, block_index)
        values = _coefficient_values(
            coefficient, cubature.points, indices, discretization, context
        )
        if values.shape != cubature.weights.shape:
            raise ValueError("VEM mass coefficient must be scalar at cell points.")
        basis_values = projection.basis.evaluate(
            cubature.points,
            geometry.centroids,
            geometry.characteristic_lengths,
        )
        scalar_matrix = ein.contract(
            "cq,cq,cqa,cqb->cab",
            cubature.weights,
            values,
            basis_values,
            basis_values,
        )
        if projection.polynomial_value_shape == (2,):
            polynomial_count = projection.basis.feature_count
            matrix = jnp.zeros(
                (
                    scalar_matrix.shape[0],
                    2 * polynomial_count,
                    2 * polynomial_count,
                ),
                dtype=scalar_matrix.dtype,
            )
            for component in range(2):
                start = component * polynomial_count
                matrix = matrix.at[
                    :, start : start + polynomial_count, start : start + polynomial_count
                ].set(scalar_matrix)
        else:
            matrix = scalar_matrix
        if domain is not None:
            selected = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
            mask = jnp.any(indices[:, None] == selected[None, :], axis=1)
            matrix = jnp.where(mask[:, None, None], matrix, 0.0)
        result.append(matrix)
    return tuple(result)


def _factorized_action(
    projections: Sequence[VirtualElementProjectionData],
    coefficient_maps: Sequence[Array],
    polynomial_matrices: Sequence[Array],
    discretization: VirtualElementDiscretization,
    policy: VirtualElementExecutionPolicy,
    /,
    *,
    kernel_projector: str,
    stabilization_policy: VirtualElementStabilizationPolicy,
    operator_id: str,
) -> FactorizedVirtualElementOperator:
    oriented_coefficients = []
    stabilization_matrices = []
    for projection, coefficient, polynomial, orientation in zip(
        projections,
        coefficient_maps,
        polynomial_matrices,
        discretization.dof_map.orientations,
        strict=True,
    ):
        consistent = ein.contract(
            "cai,cab,cbj->cij", coefficient, polynomial, coefficient
        )
        stabilized = stabilize_virtual_element_tensor(
            projection,
            consistent,
            stabilization_policy,
            projector=kernel_projector,
        )
        oriented_coefficients.append(coefficient * orientation[:, None, :])
        stabilization_matrices.append(
            stabilized.stabilization * orientation[:, :, None] * orientation[:, None, :]
        )
    return FactorizedVirtualElementOperator(
        tuple(oriented_coefficients),
        tuple(polynomial_matrices),
        tuple(stabilization_matrices),
        discretization.dof_map.cell_dofs,
        discretization.dof_map.global_dof_count,
        accumulation=policy.accumulation,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=operator_id,
    )


def _local_coordinate_operator(
    matrices: Array,
    routes: Array,
    full_space: AbstractVectorSpace,
    /,
    *,
    operator_id: str,
) -> SparseCoordinateOperator:
    """Canonical coordinates of `(entity, row, column)` tensors on prepared routes.

    The routes are prepared DOF structure and only the tensors are runtime data,
    so this stays traceable: no host route validation, fingerprinting, or
    relation re-planning runs inside a runtime or gradient evaluation.
    """
    return SparseCoordinateOperator(
        EdgeRelation(
            jnp.broadcast_to(routes[:, None, :], matrices.shape).reshape((-1,)),
            jnp.broadcast_to(routes[:, :, None], matrices.shape).reshape((-1,)),
            source_size=full_space.size,
            target_size=full_space.size,
        ),
        matrices.reshape((-1,)),
        source=full_space,
        target=DualSpace(full_space),
        properties=OperatorProperties(),
        operator_id=operator_id,
    )


def _merge_sparse(
    operators: Sequence[SparseCoordinateOperator],
    full_space: AbstractVectorSpace,
    /,
    *,
    properties: OperatorProperties,
    operator_id: str,
) -> SparseCoordinateOperator:
    if not operators:
        raise ValueError("Sparse VEM merge requires at least one operator.")
    sources = []
    targets = []
    valid = []
    coefficients = []
    for operator in operators:
        relation = operator.relation
        if not isinstance(relation, EdgeRelation):
            relation = relation.as_edge_relation()
        sources.append(relation.source_indices)
        targets.append(relation.target_indices)
        valid.append(relation.valid)
        coefficients.append(operator.coefficients)
    relation = EdgeRelation(
        jnp.concatenate(tuple(sources)),
        jnp.concatenate(tuple(targets)),
        source_size=full_space.size,
        target_size=full_space.size,
        valid=jnp.concatenate(tuple(valid)),
    )
    return SparseCoordinateOperator(
        relation,
        jnp.concatenate(tuple(coefficients)),
        source=full_space,
        target=DualSpace(full_space),
        properties=OperatorProperties(),
        operator_id=operator_id,
    )


def _realize_factorized(
    factorized: FactorizedVirtualElementOperator,
    full_space: AbstractVectorSpace,
    policy: VirtualElementExecutionPolicy,
    /,
) -> AbstractLinearOperator:
    if policy.realization == "sparse":
        sparse = tuple(
            _local_coordinate_operator(
                local, gather, full_space, operator_id=factorized.operator_id
            )
            for local, gather in zip(
                factorized.local_tensors(), factorized.gathers, strict=True
            )
        )
        return _merge_sparse(
            sparse,
            full_space,
            properties=factorized.properties,
            operator_id=factorized.operator_id,
        )
    raw = factorized.as_linear_operator()
    return FunctionLinearOperator(
        raw.mv,
        source=full_space,
        target=DualSpace(full_space),
        transpose_action=raw.transpose_mv,
        properties=OperatorProperties(),
        operator_id=factorized.operator_id,
        closure_convert=False,
    )


def _sum_operators(
    operators: Sequence[AbstractLinearOperator],
    full_space: AbstractVectorSpace,
    /,
    *,
    operator_id: str,
) -> AbstractLinearOperator:
    values = tuple(operators)
    if not values:
        raise ValueError("VEM form contains no bilinear action.")
    sparse = tuple(
        value for value in values if isinstance(value, SparseCoordinateOperator)
    )
    if len(sparse) == len(values):
        return _merge_sparse(
            sparse,
            full_space,
            properties=OperatorProperties(),
            operator_id=operator_id,
        )
    return FunctionLinearOperator(
        lambda state: sum(
            (value.mv(state) for value in values), start=jnp.zeros_like(state)
        ),
        source=full_space,
        target=DualSpace(full_space),
        transpose_action=lambda state: sum(
            (value.transpose_mv(state) for value in values), start=jnp.zeros_like(state)
        ),
        properties=OperatorProperties(),
        operator_id=operator_id,
        closure_convert=False,
    )


def _boundary_data(
    coefficient: VariationalCoefficient,
    points: Array,
    edges: Array,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    /,
) -> Array:
    return coefficient.evaluate(
        points,
        context.user_args,
        entity_indices=edges,
        support_id=discretization.support.support_id,
        entity_set_id=discretization.mesh.topology.entity_sets[1].entity_set_id,
    )


def _boundary_operator_and_rhs(
    action: BoundaryLoadAction | VirtualElementRobinAction,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    policy: VirtualElementExecutionPolicy,
    /,
) -> tuple[SparseCoordinateOperator | None, Array, Array]:
    from ...integration import GaussLegendreRule, interval_rule_data

    trace_kind = discretization.field.element.trace_kind
    if trace_kind == "none":
        raise ValueError("Discontinuous L2 virtual elements have no boundary trace.")
    domain = (
        discretization.exterior_facet_domain if action.domain is None else action.domain
    )
    if (
        domain.support_id != discretization.support.support_id
        or domain.entity_set_id
        != discretization.mesh.topology.entity_sets[1].entity_set_id
    ):
        raise ValueError("VEM boundary domain belongs to another facet support.")
    edges = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    routes = discretization.edge_trace_routes(edges)
    degree = discretization.field.element.degree
    quadrature = interval_rule_data(GaussLegendreRule(degree + 2))
    axis = jnp.asarray(quadrature.nodes)
    weights = jnp.asarray(quadrature.weights)
    trace_basis = discretization.field.element.edge_trace_basis(axis)
    connectivity = polygonal_connectivity_of(discretization.mesh)
    if trace_kind == "value":
        basis = jnp.broadcast_to(trace_basis[None], (edges.size,) + trace_basis.shape)
    else:
        owner = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
        owner_local = jnp.asarray(domain.owner_local_entities, dtype=jnp.int32)
        signs = jnp.asarray(connectivity.cell_edge_signs)[owner, owner_local]
        basis = signs[:, None, None] * trace_basis[None]
    connectivity_edges = jnp.asarray(connectivity.edges, dtype=jnp.int32)[edges]
    start = context.runtime.coordinates[connectivity_edges[:, 0]]
    stop = context.runtime.coordinates[connectivity_edges[:, 1]]
    points = (
        0.5 * (1.0 - axis[None, :, None]) * start[:, None, :]
        + 0.5 * (1.0 + axis[None, :, None]) * stop[:, None, :]
    )
    length = jnp.sqrt(jnp.sum((stop - start) ** 2, axis=-1))
    weighted = 0.5 * length[:, None] * weights[None, :]
    if isinstance(action, VirtualElementRobinAction):
        alpha = _boundary_data(action.coefficient, points, edges, discretization, context)
        value = _boundary_data(action.value, points, edges, discretization, context)
        if alpha.shape != weighted.shape or value.shape != weighted.shape:
            raise ValueError("VEM Robin data must be scalar on boundary quadrature.")
        matrices = ein.contract("eq,eq,eqi,eqj->eij", weighted, alpha, basis, basis)
        rhs = ein.contract("eq,eq,eqi->ei", weighted, value, basis)
        operator = _local_coordinate_operator(
            matrices,
            routes,
            discretization.field_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "virtual-element-robin",
                    "action": action.action_id,
                    "runtime": context.runtime.runtime_id,
                    "field_space": discretization.field_space.field_space_id,
                    "trace": trace_kind,
                }
            ),
        )
        return operator, routes, rhs
    value = _boundary_data(action.load, points, edges, discretization, context)
    if value.shape != weighted.shape:
        raise ValueError("VEM boundary load must be scalar on boundary quadrature.")
    rhs = ein.contract("eq,eq,eqi->ei", weighted, value, basis)
    return None, routes, rhs


class CompiledVirtualElementProblem(StrictModule, NonTrainableState):
    form: VirtualElementForm
    discretization: VirtualElementDiscretization
    constraint: VirtualElementDirichletConstraint | None
    execution_policy: VirtualElementExecutionPolicy
    lift: object
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        form: VirtualElementForm,
        discretization: VirtualElementDiscretization,
        /,
        *,
        constraint: VirtualElementDirichletConstraint | None = None,
        dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
        execution_policy: VirtualElementExecutionPolicy | None = None,
    ) -> None:
        if form.field_name != discretization.field.name:
            raise ValueError("VEM form field does not match the discretization.")
        family = discretization.field.element.family
        if family == "DiscontinuousL2":
            for action in form.actions:
                if isinstance(action, DiffusionAction):
                    raise ValueError(
                        "Diffusion is undefined on discontinuous L2 VEM spaces."
                    )
                if isinstance(action, (BoundaryLoadAction, VirtualElementRobinAction)):
                    raise ValueError(
                        "Discontinuous L2 virtual elements have no boundary trace."
                    )
            if constraint is not None:
                raise ValueError(
                    "Discontinuous L2 virtual elements have no Dirichlet trace."
                )
        policy = (
            VirtualElementExecutionPolicy()
            if execution_policy is None
            else execution_policy
        )
        if constraint is None:
            if dirichlet_values is not None:
                raise ValueError("Dirichlet values require a VEM constraint.")
            lift = discretization.field_space.vector_space.zeros()
        else:
            if not isinstance(constraint, VirtualElementDirichletConstraint):
                raise TypeError("constraint must be VirtualElementDirichletConstraint.")
            if constraint.field_space_id != discretization.field_space.field_space_id:
                raise ValueError("VEM constraint belongs to another field space.")
            if dirichlet_values is None:
                raise ValueError("VEM constraint requires Dirichlet values.")
            lift = constraint.lift(dirichlet_values)
        for action in form.actions:
            if (
                isinstance(
                    action,
                    (DiffusionAction, MassAction, SourceAction, BoundaryLoadAction),
                )
                and action.rules
            ):
                raise ValueError(
                    "Qualified VEM uses its prepared polygon/edge quadrature rules."
                )
        self.form = form
        self.discretization = discretization
        self.constraint = constraint
        self.execution_policy = policy
        self.lift = lift
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-virtual-element-problem",
                "form": form.form_id,
                "discretization": discretization.prepared_id,
                "constraint": None if constraint is None else constraint.constraint_id,
                "execution_policy": policy.policy_id,
            }
        )
        form_key = DiscretizationKey(
            "virtual_element_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
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
                    "compiled-virtual-element-form",
                    self.compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )

    @property
    def full_space(self) -> AbstractVectorSpace:
        return self.discretization.field_space.vector_space

    @property
    def state_space(self) -> AbstractVectorSpace:
        return (
            self.full_space
            if self.constraint is None
            else self.constraint.constraint_map.reduced_space
        )

    @property
    def residual_space(self) -> DualSpace:
        return DualSpace(self.state_space)

    @property
    def constraint_map(self) -> ConstraintMap | None:
        return None if self.constraint is None else self.constraint.constraint_map

    def _context(self, args: object = None, /) -> VirtualElementExecutionContext:
        if isinstance(args, VirtualElementExecutionContext):
            context = args
        else:
            context = VirtualElementExecutionContext(
                self.discretization.default_runtime,
                lift=self.lift,
                user_args=args,
            )
        runtime = context.runtime
        if not isinstance(runtime, VirtualElementRuntimeData):
            raise TypeError("VEM execution context runtime has the wrong type.")
        if not virtual_element_runtime_matches(self.discretization, runtime):
            raise ValueError("VEM execution context is incompatible with the space.")
        return context

    def expand(self, state: PyTree[Any], context: object = None, /) -> PyTree[Array]:
        context_ = self._context(context)
        lift = self.lift if context_.lift is None else context_.lift
        if self.constraint is None:
            return self.full_space.validate(state)
        return self.constraint.constraint_map.expand(state, lift)

    def _action_operator(
        self, action: VirtualElementAction, context: VirtualElementExecutionContext
    ) -> AbstractLinearOperator | None:
        if isinstance(action, DiffusionAction):
            polynomial = _diffusion_polynomial_matrices(
                action, self.discretization, context
            )
            family = self.discretization.field.element.family
            if family == "ConformingH1":
                coefficients = tuple(
                    value.h1_coefficients for value in context.runtime.projections
                )
                kernel_projector = "h1"
            else:
                coefficients = tuple(
                    value.differential_coefficients
                    for value in context.runtime.projections
                )
                kernel_projector = "l2"
            factorized = _factorized_action(
                context.runtime.projections,
                coefficients,
                polynomial,
                self.discretization,
                self.execution_policy,
                kernel_projector=kernel_projector,
                stabilization_policy=self.execution_policy.stiffness_stabilization,
                operator_id=canonical_fingerprint(
                    {
                        "kind": f"VEM-{self.discretization.field.element.differential_kind}",
                        "action": action.action_id,
                        "runtime": context.runtime.runtime_id,
                    }
                ),
            )
            return _realize_factorized(factorized, self.full_space, self.execution_policy)
        if isinstance(action, MassAction):
            polynomial = _mass_polynomial_matrices(
                action.coefficient,
                self.discretization,
                context,
                domain=action.domain,
            )
            factorized = _factorized_action(
                context.runtime.projections,
                tuple(value.l2_coefficients for value in context.runtime.projections),
                polynomial,
                self.discretization,
                self.execution_policy,
                kernel_projector="l2",
                stabilization_policy=self.execution_policy.mass_stabilization,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "VEM-mass",
                        "action": action.action_id,
                        "runtime": context.runtime.runtime_id,
                    }
                ),
            )
            return _realize_factorized(factorized, self.full_space, self.execution_policy)
        if isinstance(action, VirtualElementRobinAction):
            operator, _, _ = _boundary_operator_and_rhs(
                action,
                self.discretization,
                context,
                self.execution_policy,
            )
            return operator
        return None

    def full_affine_operator(self, args: object = None, /) -> AbstractLinearOperator:
        context = self._context(args)
        operators = tuple(
            operator
            for action in self.form.actions
            if (operator := self._action_operator(action, context)) is not None
        )
        return _sum_operators(
            operators,
            self.full_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "VEM-affine-operator",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )

    def affine_operator(self, args: object = None, /) -> AbstractLinearOperator:
        full = self.full_affine_operator(args)
        if self.constraint is None:
            return full
        mapping = self.constraint.constraint_map
        return FunctionLinearOperator(
            lambda reduced: mapping.pullback_dual(
                full.mv(mapping.homogeneous_correction(reduced))
            ),
            source=mapping.reduced_space,
            target=DualSpace(mapping.reduced_space),
            transpose_action=lambda reduced_dual: mapping.prolongation.transpose_mv(
                full.transpose_mv(mapping.prolongation.mv(reduced_dual))
            ),
            properties=OperatorProperties(),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constrained-VEM-operator",
                    "operator": full.operator_id,
                    "constraint": mapping.constraint_id,
                }
            ),
        )

    def full_right_hand_side(self, args: object = None, /) -> Array:
        context = self._context(args)
        result = jnp.zeros(
            (self.full_space.size,), dtype=context.runtime.coordinates.dtype
        )
        for action in self.form.actions:
            if isinstance(action, SourceAction):
                for block_index, (
                    geometry,
                    cubature,
                    projection,
                    gathers,
                    orientations,
                ) in enumerate(
                    zip(
                        context.runtime.geometries,
                        context.runtime.cubatures,
                        context.runtime.projections,
                        self.discretization.dof_map.cell_dofs,
                        self.discretization.dof_map.orientations,
                        strict=True,
                    )
                ):
                    indices = _cell_indices(self.discretization, block_index)
                    values = _coefficient_values(
                        action.source,
                        cubature.points,
                        indices,
                        self.discretization,
                        context,
                    )
                    basis = projection.basis.evaluate(
                        cubature.points,
                        geometry.centroids,
                        geometry.characteristic_lengths,
                    )
                    if projection.polynomial_value_shape == (2,):
                        expected = cubature.weights.shape + (2,)
                        if values.shape != expected:
                            raise ValueError(
                                "Vector VEM source must have two components at cell quadrature."
                            )
                        component_moments = ein.contract(
                            "cq,cqd,cqa->cda", cubature.weights, values, basis
                        )
                        moments = component_moments.reshape(
                            (component_moments.shape[0], -1)
                        )
                    else:
                        if values.shape != cubature.weights.shape:
                            raise ValueError(
                                "Scalar VEM source must be scalar at cell quadrature."
                            )
                        moments = ein.contract(
                            "cq,cq,cqa->ca", cubature.weights, values, basis
                        )
                    local = ein.contract(
                        "cai,ca->ci", projection.l2_coefficients, moments
                    )
                    local = (
                        jnp.where(_cell_mask(action, indices)[:, None], local, 0.0)
                        * orientations
                    )
                    result = scatter_local(
                        result,
                        gathers,
                        local,
                        self.execution_policy.accumulation,
                    )
            elif isinstance(action, (BoundaryLoadAction, VirtualElementRobinAction)):
                _, routes, local = _boundary_operator_and_rhs(
                    action,
                    self.discretization,
                    context,
                    self.execution_policy,
                )
                result = scatter_local(
                    result,
                    routes,
                    local,
                    self.execution_policy.accumulation,
                )
        return result

    def right_hand_side(self, args: object = None, /) -> Array:
        context = self._context(args)
        full_rhs = self.full_right_hand_side(context)
        if self.constraint is None:
            return full_rhs
        full_operator = self.full_affine_operator(context)
        lift = self.lift if context.lift is None else context.lift
        return self.constraint.constraint_map.pullback_dual(
            full_rhs - full_operator.mv(lift)
        )

    def full_residual(self, state: PyTree[Any], args: object = None, /) -> Array:
        context = self._context(args)
        return self.full_affine_operator(context).mv(state) - self.full_right_hand_side(
            context
        )

    def residual(self, state: PyTree[Any], args: object = None, /) -> Array:
        context = self._context(args)
        full = self.expand(state, context)
        residual = self.full_residual(full, context)
        return (
            residual
            if self.constraint is None
            else self.constraint.constraint_map.pullback_dual(residual)
        )

    @checked
    def _require_side_trace(self, trace: PreparedTraceAction, /) -> None:
        if (
            trace.descriptor.owner_id != self.discretization.prepared_id
            or trace.descriptor.field_space_id
            != self.discretization.field_space.field_space_id
        ):
            raise ValueError(
                "The trace belongs to another discretization field; prepare it with "
                "this problem's VirtualElementDiscretization.prepare_side_trace."
            )
        if (
            self.discretization.field.element.family != "ConformingH1"
            or trace.descriptor.quantity != "value"
        ):
            raise ValueError(
                "VEM conormal fluxes are defined for value traces of scalar "
                "ConformingH1 fields."
            )

    def _trace_domain(self, trace: PreparedTraceAction, /) -> IntegrationDomain:
        match trace.descriptor.domain_kind:
            case "exterior_facet":
                base = self.discretization.exterior_facet_domain
            case "interior_facet":
                base = self.discretization.interior_facet_domain
            case kind:
                raise ValueError(f"Side traces do not act on {kind!r} domains.")
        return polygon_facet_domain(base, trace.descriptor.facets)

    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        """Publish the exact-edge conormal flux as the residual reaction.

        The flux is the discrete Lagrange multiplier of the side: the full weak
        residual `full_residual(state, args)` restricted to `trace.support_rows`
        (the vertex and edge DOFs of the selected exterior edges). At a state
        satisfying the remaining rows it equals the outward conormal flux
        tested against the edge trace basis, exactly and without projecting the
        interior gradient (`approximation="variational-reaction"`). Only value
        traces of scalar `ConformingH1` fields on exterior facets are accepted:
        interior-facet rows also carry the residual of the other cell.
        """
        self._require_side_trace(trace)
        if trace.descriptor.domain_kind != "exterior_facet":
            raise ValueError(
                "Reaction fluxes live on exterior facets; interior-facet rows also "
                "carry the other cell's residual. Use prepare_projected_flux for "
                "a one-sided interior flux."
            )
        descriptor = SideActionDescriptor(
            owner_id=self.compilation_id,
            field_space_id=trace.descriptor.field_space_id,
            quantity="conormal-flux",
            representation="residual-reaction",
            orientation="outward",
            approximation="variational-reaction",
            side=trace.descriptor.side,
            domain=self._trace_domain(trace),
            revision_id=trace.descriptor.revision_id,
            rule=None,
            trace_degree=trace.descriptor.trace_degree,
            quadrature_exact_degree=None,
        )
        return PreparedFluxAction(
            descriptor, trace, _VirtualElementReactionFlux(self, trace.support_rows)
        )

    def prepare_projected_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        """Publish the projected conormal flux `kappa grad(Pi^nabla u) . n`.

        Densities at `trace.sites` use the energy projection of the side cell
        (owner or neighbor of the trace), the form's `DiffusionAction`
        diffusivities (scalar or 2x2 tensor, summed over the actions whose
        domain contains the side cell), and the trace's outward normals. The
        virtual interior gradient is not computable, so the flux is labeled
        `approximation="h1-projection"`; it is exact when the discrete field is
        a polynomial of degree `k`. It is evaluated on the geometry revision of
        its trace, which must be the problem's default runtime.
        """
        self._require_side_trace(trace)
        if not any(isinstance(action, DiffusionAction) for action in self.form.actions):
            raise ValueError(
                "The form has no DiffusionAction; it defines no conormal flux."
            )
        runtime = self.discretization.default_runtime
        if trace.descriptor.revision_id != polygon_side_revision(
            runtime.runtime_id, runtime.coordinates
        ):
            raise ValueError(
                "The trace was prepared on another geometry revision; prepare it "
                "on the problem's default runtime."
            )
        domain = self._trace_domain(trace)
        side_cells, _, normals = polygon_side_frame(
            self.discretization.mesh, runtime.coordinates, domain, trace.descriptor.side
        )
        if not np.allclose(
            normals, np.asarray(trace.normals[:, 0]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "The trace's facet sides differ from the owner's canonical facet "
                "domain; prepare the trace from the discretization's facet domains."
            )
        descriptor = SideActionDescriptor(
            owner_id=self.compilation_id,
            field_space_id=trace.descriptor.field_space_id,
            quantity="conormal-flux",
            representation="quadrature-values",
            orientation="outward",
            approximation="h1-projection",
            side=trace.descriptor.side,
            domain=domain,
            revision_id=trace.descriptor.revision_id,
            rule=None,
            trace_degree=None,
            quadrature_exact_degree=trace.descriptor.quadrature_exact_degree,
        )
        return PreparedFluxAction(
            descriptor,
            trace,
            _projected_flux_evaluator(self, trace, runtime, side_cells),
        )

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        """Report the provenance of every boundary law of this problem.

        The order is deterministic: the strong Dirichlet rows of `constraint`
        first (source: its constraint ID), then the natural boundary loads and
        Robin terms in form-action order on their exterior facets (source:
        the action ID).
        """
        field_space_id = self.discretization.field_space.field_space_id
        impositions: list[BoundaryImposition] = []
        kind: ImpositionKind
        if self.constraint is not None:
            impositions.append(
                BoundaryImposition(
                    "strong",
                    field_space_id=field_space_id,
                    source_id=self.constraint.constraint_id,
                    rows=self.constraint.constrained_dofs,
                )
            )
        for action in self.form.actions:
            if isinstance(action, BoundaryLoadAction):
                kind = "natural"
                domain = (
                    self.discretization.exterior_facet_domain
                    if action.domain is None
                    else action.domain
                )
            elif isinstance(action, VirtualElementRobinAction):
                kind = "robin"
                domain = action.domain
            else:
                continue
            impositions.append(
                BoundaryImposition(
                    kind,
                    field_space_id=field_space_id,
                    source_id=action.action_id,
                    entity_set_id=domain.entity_set_id,
                    facets=domain.entity_indices,
                )
            )
        return tuple(impositions)

    def _default_nullspace_policy(
        self, operator: AbstractLinearOperator
    ) -> NullspacePolicy | None:
        if (
            self.constraint is not None
            or self.discretization.field.element.family != "ConformingH1"
        ):
            return None
        has_diffusion = any(
            isinstance(action, DiffusionAction) for action in self.form.actions
        )
        removes_constant = any(
            isinstance(action, (MassAction, VirtualElementRobinAction))
            for action in self.form.actions
        )
        if not has_diffusion or removes_constant:
            return None
        right_basis = jnp.ones(
            (operator.source.size, 1), dtype=operator.source.zeros().dtype
        )
        left_basis = jnp.ones(
            (operator.target.size, 1), dtype=operator.target.zeros().dtype
        )
        return NullspacePolicy(
            right=LinearSubspace(operator.source, right_basis),
            left=LinearSubspace(operator.target, left_basis),
            compatibility="error",
            gauge="minimum-norm",
        )

    def linear_system(
        self,
        args: object = None,
        /,
        *,
        nullspace_policy: NullspacePolicy | None = None,
    ) -> tuple[LinearSystem, Array]:
        weak = self.affine_operator(args)
        operator = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(weak.mv(state)),
            source=self.state_space,
            target=self.state_space,
            transpose_action=lambda state: self.state_space.inverse_riesz(
                weak.transpose_mv(state)
            ),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {"kind": "VEM-linear-system-operator", "weak": weak.operator_id}
            ),
        )
        selected = (
            self._default_nullspace_policy(operator)
            if nullspace_policy is None
            else nullspace_policy
        )
        rhs = self.state_space.inverse_riesz(self.right_hand_side(args))
        return LinearSystem(operator, nullspace_policy=selected), rhs

    def sparse_assembly_plan(
        self,
        args: object = None,
        /,
        *,
        policy: SparseAssemblyPolicy | None = None,
    ) -> SparseAssemblyPlan:
        return plan_sparse_assembly(self.affine_operator(args), policy)

    def prepare_sparse_assembly(
        self,
        args: object = None,
        /,
        *,
        plan: SparseAssemblyPlan | None = None,
        policy: SparseAssemblyPolicy | None = None,
    ) -> PreparedSparseAssembly:
        operator = self.affine_operator(args)
        selected = plan_sparse_assembly(operator, policy) if plan is None else plan
        return prepare_sparse_assembly(selected, operator)

    def mass_operator(
        self,
        args: object = None,
        /,
        *,
        coefficient: ArrayLike | VariationalCoefficient = 1.0,
        return_full: bool = False,
    ) -> AbstractLinearOperator:
        context = self._context(args)
        from .._variational import coefficient as bind_coefficient

        # A coefficient bound on the host (identity fingerprinted once) may be
        # passed through, so the mass action stays usable inside traced code.
        bound = (
            coefficient
            if isinstance(coefficient, VariationalCoefficient)
            else bind_coefficient(coefficient)
        )
        polynomial = _mass_polynomial_matrices(bound, self.discretization, context)
        factorized = _factorized_action(
            context.runtime.projections,
            tuple(value.l2_coefficients for value in context.runtime.projections),
            polynomial,
            self.discretization,
            self.execution_policy,
            kernel_projector="l2",
            stabilization_policy=self.execution_policy.mass_stabilization,
            operator_id=canonical_fingerprint(
                {
                    "kind": "VEM-unit-mass",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )
        full_action = _realize_factorized(
            factorized, self.full_space, self.execution_policy
        )
        if return_full or self.constraint is None:
            return full_action
        mapping = self.constraint.constraint_map
        return FunctionLinearOperator(
            lambda reduced: mapping.pullback_dual(
                full_action.mv(mapping.homogeneous_correction(reduced))
            ),
            source=mapping.reduced_space,
            target=DualSpace(mapping.reduced_space),
            transpose_action=lambda value: mapping.pullback_dual(
                full_action.transpose_mv(mapping.homogeneous_correction(value))
            ),
            properties=OperatorProperties(),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constrained-VEM-mass",
                    "mass": full_action.operator_id,
                    "constraint": mapping.constraint_id,
                }
            ),
        )

    def as_dae_system(
        self,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
        system_id: str | None = None,
    ) -> DifferentialAlgebraicSystem:
        identifier = (
            canonical_fingerprint(
                {"kind": "virtual-element-dae", "compilation": self.compilation_id}
            )
            if system_id is None
            else str(system_id)
        )
        structure = self.state_space.structure()

        def context(time: Array, args: object) -> VirtualElementExecutionContext:
            base = self._context(args)
            return VirtualElementExecutionContext(
                base.runtime,
                time=time,
                lift=base.lift,
                lift_rate=base.lift_rate,
                user_args=base.user_args,
            )

        def mass_matrix(
            time: Array, state: Array, args: object
        ) -> AbstractLinearOperator:
            return self.mass_operator(context(time, args), coefficient=mass_coefficient)

        def vector_field(time: Array, state: Array, args: object) -> Array:
            current = context(time, args)
            result = -self.residual(state, current)
            if self.constraint is not None and current.lift_rate is not None:
                full_mass = self.mass_operator(
                    current, coefficient=mass_coefficient, return_full=True
                )
                result = result - self.constraint.constraint_map.pullback_dual(
                    full_mass.mv(current.lift_rate)
                )
            return result

        return DifferentialAlgebraicSystem.from_mass_matrix(
            mass_matrix,
            vector_field,
            state_shape=structure.shape,
            structure=DAEStructure(("differential",), component_axis=None),
            system_id=identifier,
        )

    def as_generalized_eigenproblem(
        self,
        args: object = None,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
    ) -> GeneralizedEigenproblem:
        if any(
            isinstance(
                action, (SourceAction, BoundaryLoadAction, VirtualElementRobinAction)
            )
            for action in self.form.actions
        ):
            raise ValueError(
                "VEM eigenproblems require homogeneous volume/boundary forms."
            )
        raw_stiffness = self.affine_operator(args)
        raw_mass = self.mass_operator(args, coefficient=mass_coefficient)
        stiffness = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_stiffness.mv(state)),
            source=self.state_space,
            target=self.state_space,
            transpose_action=lambda state: self.state_space.inverse_riesz(
                raw_stiffness.transpose_mv(state)
            ),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {"kind": "VEM-eigen-stiffness", "compilation": self.compilation_id}
            ),
        )
        mass = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_mass.mv(state)),
            source=self.state_space,
            target=self.state_space,
            transpose_action=lambda state: self.state_space.inverse_riesz(
                raw_mass.transpose_mv(state)
            ),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {"kind": "VEM-eigen-mass", "compilation": self.compilation_id}
            ),
        )
        return GeneralizedEigenproblem(stiffness, mass)


@final
class _VirtualElementReactionFlux(AbstractSideFluxEvaluator, NonTrainableState):
    """Full weak residual of one compiled problem on the rows of one side."""

    problem: CompiledVirtualElementProblem
    rows: Array

    @property
    def state_space(self) -> AbstractVectorSpace:
        return self.problem.full_space

    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        return self.problem.full_residual(state, args)[self.rows]


@final
class _VirtualElementProjectedFlux(AbstractSideFluxEvaluator, NonTrainableState):
    """`kappa grad(Pi^nabla u) . n` at fixed trace sites of known side cells.

    `gradient_weights` has shape `(facets, sites, local, 2)` and maps the
    gathered DOFs `dofs` of each side cell to the projected gradient.
    """

    problem: CompiledVirtualElementProblem
    dofs: Array
    gradient_weights: Array
    sites: Array
    normals: Array
    side_cells: Array
    runtime_id: str = eqx.field(static=True)

    @property
    def state_space(self) -> AbstractVectorSpace:
        return self.problem.full_space

    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        context = self.problem._context(args)
        if context.runtime.runtime_id != self.runtime_id:
            raise ValueError(
                "The projected flux was prepared on another VEM runtime; prepare the "
                "trace and flux on the evaluating runtime."
            )
        gathered = jnp.asarray(state)[self.dofs]
        gradient = ein.contract("fqld,fl->fqd", self.gradient_weights, gathered)
        flux = jnp.zeros(self.sites.shape[:2], dtype=gradient.dtype)
        for action in self.problem.form.actions:
            if not isinstance(action, DiffusionAction):
                continue
            kappa = _coefficient_values(
                action.diffusivity,
                self.sites,
                self.side_cells,
                self.problem.discretization,
                context,
            )
            if kappa.shape == self.sites.shape[:2]:
                conormal = kappa * jnp.sum(gradient * self.normals, axis=-1)
            elif kappa.shape == self.sites.shape[:2] + (2, 2):
                conormal = ein.contract("fqd,fqde,fqe->fq", self.normals, kappa, gradient)
            else:
                raise ValueError(
                    "H1 VEM diffusivity must be scalar or a 2x2 tensor per trace site."
                )
            inside = _cell_mask(action, self.side_cells)
            flux = flux + jnp.where(inside[:, None], conormal, 0.0)
        return flux


def _projected_flux_evaluator(
    problem: CompiledVirtualElementProblem,
    trace: PreparedTraceAction,
    runtime: VirtualElementRuntimeData,
    side_cells: np.ndarray,
    /,
) -> _VirtualElementProjectedFlux:
    """Prepare the fixed projected-gradient routes of the trace sites."""
    basis, routes = virtual_element_projection_basis(
        problem.discretization, runtime, "h1-projection"
    )
    facets, sites_per_facet = trace.sites.shape[:2]
    cells = jnp.asarray(side_cells)
    flat_cells = jnp.repeat(cells, sites_per_facet)
    points = trace.sites.reshape((-1, 2))
    gradient = jnp.stack(
        tuple(
            basis.cell_weights(flat_cells, points, derivative)
            for derivative in ((1, 0), (0, 1))
        ),
        axis=-1,
    )
    return _VirtualElementProjectedFlux(
        problem,
        jnp.asarray(routes)[cells],
        gradient.reshape((facets, sites_per_facet) + gradient.shape[1:]),
        trace.sites,
        trace.normals,
        cells,
        runtime_id=runtime.runtime_id,
    )


def compile_virtual_element_problem(
    form: VirtualElementForm,
    discretization: VirtualElementDiscretization,
    /,
    *,
    constraint: VirtualElementDirichletConstraint | None = None,
    dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
    execution_policy: VirtualElementExecutionPolicy | None = None,
) -> CompiledVirtualElementProblem:
    return CompiledVirtualElementProblem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=dirichlet_values,
        execution_policy=execution_policy,
    )


__all__ = ["CompiledVirtualElementProblem", "compile_virtual_element_problem"]
