#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ...discretization.vem import (
    FactorizedVirtualElementOperator,
    stabilize_virtual_element_tensor,
    VirtualElementDirichletConstraint,
    VirtualElementDiscretization,
)
from ...dynamics import DAEStructure, DifferentialAlgebraicSystem
from ...linalg import (
    AbstractLinearOperator,
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
from .._variational import BoundaryLoadAction, DiffusionAction, MassAction, SourceAction
from ._form import (
    VirtualElementExecutionContext,
    VirtualElementExecutionPolicy,
    VirtualElementForm,
    VirtualElementRobinAction,
)


def _cell_indices(discretization: VirtualElementDiscretization, block_index: int, /):
    offset = sum(block.cell_count for block in discretization.mesh.blocks[:block_index])
    count = discretization.mesh.blocks[block_index].cell_count
    return jnp.arange(offset, offset + count, dtype=jnp.int32)


def _cell_mask(action, indices: Array, /) -> Array:
    if action.domain is None:
        return jnp.ones(indices.shape, dtype=bool)
    selected = jnp.asarray(action.domain.entity_indices, dtype=jnp.int32)
    return jnp.any(indices[:, None] == selected[None, :], axis=1)


def _coefficient_values(
    coefficient,
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
):
    if action.diffusivity.evaluator is not None:
        raise ValueError("Callable diffusivity is not supported by qualified VEM.")
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
        if values.shape == cubature.weights.shape:
            scalar = values
            basis_gradient = projection.basis.gradient(
                cubature.points,
                geometry.centroids,
                geometry.characteristic_lengths,
            )
            matrix = oe.contract(
                "cq,cq,cqad,cqbd->cab",
                cubature.weights,
                scalar,
                basis_gradient,
                basis_gradient,
            )
        elif values.shape == cubature.weights.shape + (2, 2):
            basis_gradient = projection.basis.gradient(
                cubature.points,
                geometry.centroids,
                geometry.characteristic_lengths,
            )
            matrix = oe.contract(
                "cq,cqad,cqde,cqbe->cab",
                cubature.weights,
                basis_gradient,
                values,
                basis_gradient,
            )
        else:
            raise ValueError(
                "VEM diffusivity must be scalar or a 2x2 tensor per cell point."
            )
        result.append(jnp.where(mask[:, None, None], matrix, 0.0))
    return tuple(result)


def _mass_polynomial_matrices(
    coefficient,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    /,
    *,
    domain=None,
):
    if coefficient.evaluator is not None:
        raise ValueError("Callable mass coefficients are not supported by qualified VEM.")
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
        matrix = oe.contract(
            "cq,cq,cqa,cqb->cab",
            cubature.weights,
            values,
            basis_values,
            basis_values,
        )
        if domain is not None:
            selected = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
            mask = jnp.any(indices[:, None] == selected[None, :], axis=1)
            matrix = jnp.where(mask[:, None, None], matrix, 0.0)
        result.append(matrix)
    return tuple(result)


def _factorized_action(
    projections,
    coefficient_maps,
    polynomial_matrices,
    discretization: VirtualElementDiscretization,
    policy: VirtualElementExecutionPolicy,
    /,
    *,
    projector: str,
    operator_id: str,
):
    stabilization_matrices = []
    for projection, polynomial in zip(
        projections,
        polynomial_matrices,
        strict=True,
    ):
        coefficient = (
            projection.h1_coefficients
            if projector == "h1"
            else projection.l2_coefficients
        )
        consistent = oe.contract("cai,cab,cbj->cij", coefficient, polynomial, coefficient)
        stabilized = stabilize_virtual_element_tensor(
            projection,
            consistent,
            policy.stiffness_stabilization
            if projector == "h1"
            else policy.mass_stabilization,
            projector=projector,
        )
        stabilization_matrices.append(stabilized.stabilization)
    factorized = FactorizedVirtualElementOperator(
        tuple(coefficient_maps),
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
    return factorized


def _merge_sparse(
    operators: Sequence[SparseCoordinateOperator],
    full_space,
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
    full_space,
    policy: VirtualElementExecutionPolicy,
    /,
) -> AbstractLinearOperator:
    if policy.realization == "sparse":
        sparse = tuple(
            operator.as_sparse_coordinate()
            for operator in factorized.materialize_buckets()
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
    full_space,
    /,
    *,
    operator_id: str,
) -> AbstractLinearOperator:
    values = tuple(operators)
    if not values:
        raise ValueError("VEM form contains no bilinear action.")
    if all(isinstance(value, SparseCoordinateOperator) for value in values):
        return _merge_sparse(
            values,
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


def _lagrange_values(nodes: Array, points: Array, /) -> Array:
    values = []
    for index in range(nodes.size):
        basis = jnp.ones_like(points)
        for other in range(nodes.size):
            if other != index:
                basis = basis * (points - nodes[other]) / (nodes[index] - nodes[other])
        values.append(basis)
    return jnp.stack(tuple(values), axis=-1)


def _edge_routes(discretization: VirtualElementDiscretization, edges: Array, /) -> Array:
    endpoints = jnp.asarray(discretization.mesh.connectivity.edges, dtype=jnp.int32)[
        edges
    ]
    degree = discretization.field.element.degree
    routes = [endpoints[:, 0]]
    offset = discretization.dof_map.vertex_dof_count
    for interior in range(degree - 1):
        routes.append(offset + edges * (degree - 1) + interior)
    routes.append(endpoints[:, 1])
    return jnp.stack(tuple(routes), axis=1)


def _boundary_data(
    coefficient,
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
    action,
    discretization: VirtualElementDiscretization,
    context: VirtualElementExecutionContext,
    policy: VirtualElementExecutionPolicy,
    /,
):
    from ...integration import (
        GaussLegendreRule,
        GaussLobattoLegendreRule,
        interval_rule_data,
    )

    domain = (
        discretization.exterior_facet_domain if action.domain is None else action.domain
    )
    edges = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    routes = _edge_routes(discretization, edges)
    degree = discretization.field.element.degree
    nodal_data = interval_rule_data(GaussLobattoLegendreRule(degree + 1))
    nodes = jnp.asarray(nodal_data.nodes)
    quadrature = interval_rule_data(GaussLegendreRule(degree + 2))
    axis = jnp.asarray(quadrature.nodes)
    weights = jnp.asarray(quadrature.weights)
    basis = _lagrange_values(nodes, axis)
    connectivity_edges = jnp.asarray(
        discretization.mesh.connectivity.edges, dtype=jnp.int32
    )[edges]
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
        matrices = oe.contract("eq,eq,qi,qj->eij", weighted, alpha, basis, basis)
        rhs = oe.contract("eq,eq,qi->ei", weighted, value, basis)
        operator = SparseCoordinateOperator(
            EdgeRelation(
                jnp.broadcast_to(routes[:, None, :], matrices.shape).reshape((-1,)),
                jnp.broadcast_to(routes[:, :, None], matrices.shape).reshape((-1,)),
                source_size=discretization.dof_map.global_dof_count,
                target_size=discretization.dof_map.global_dof_count,
            ),
            matrices.reshape((-1,)),
            source=discretization.field_space.vector_space,
            target=DualSpace(discretization.field_space.vector_space),
            properties=OperatorProperties(),
            operator_id=canonical_fingerprint(
                {"kind": "virtual-element-robin", "action": action.action_id}
            ),
        )
        return operator, routes, rhs
    value = _boundary_data(action.load, points, edges, discretization, context)
    if value.shape != weighted.shape:
        raise ValueError("VEM boundary load must be scalar on boundary quadrature.")
    rhs = oe.contract("eq,eq,qi->ei", weighted, value, basis)
    return None, routes, rhs


class CompiledVirtualElementProblem(StrictModule, NonTrainableState):
    form: VirtualElementForm
    discretization: VirtualElementDiscretization
    constraint: object
    execution_policy: VirtualElementExecutionPolicy
    lift: object
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        form: VirtualElementForm,
        discretization: VirtualElementDiscretization,
        /,
        *,
        constraint: VirtualElementDirichletConstraint | None = None,
        dirichlet_values=None,
        execution_policy: VirtualElementExecutionPolicy | None = None,
    ):
        if not isinstance(form, VirtualElementForm):
            raise TypeError("form must be VirtualElementForm.")
        if not isinstance(discretization, VirtualElementDiscretization):
            raise TypeError("discretization must be VirtualElementDiscretization.")
        if form.field_name != discretization.field.name:
            raise ValueError("VEM form field does not match the discretization.")
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
            if dirichlet_values is None:
                raise ValueError("VEM constraint requires Dirichlet values.")
            lift = constraint.lift(dirichlet_values)
        for action in form.actions:
            if isinstance(action, (DiffusionAction, MassAction)) and action.rules:
                raise ValueError(
                    "Qualified VEM uses its prepared polygon quadrature rules."
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
    def full_space(self):
        return self.discretization.field_space.vector_space

    @property
    def state_space(self):
        return (
            self.full_space
            if self.constraint is None
            else self.constraint.constraint_map.reduced_space
        )

    @property
    def residual_space(self):
        return DualSpace(self.state_space)

    @property
    def constraint_map(self):
        return None if self.constraint is None else self.constraint.constraint_map

    def _context(self, args=None, /) -> VirtualElementExecutionContext:
        if isinstance(args, VirtualElementExecutionContext):
            context = args
        else:
            context = VirtualElementExecutionContext(
                self.discretization.default_runtime,
                lift=self.lift,
                user_args=args,
            )
        if context.runtime.topology_id != self.discretization.mesh.topology_id:
            raise ValueError("VEM execution context has incompatible topology.")
        return context

    def expand(self, state, context=None, /):
        context_ = self._context(context)
        lift = self.lift if context_.lift is None else context_.lift
        if self.constraint is None:
            return self.full_space.validate(state)
        return self.constraint.constraint_map.expand(state, lift)

    def _action_operator(self, action, context):
        if isinstance(action, DiffusionAction):
            polynomial = _diffusion_polynomial_matrices(
                action, self.discretization, context
            )
            factorized = _factorized_action(
                context.runtime.projections,
                tuple(value.h1_coefficients for value in context.runtime.projections),
                polynomial,
                self.discretization,
                self.execution_policy,
                projector="h1",
                operator_id=canonical_fingerprint(
                    {
                        "kind": "VEM-diffusion",
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
                projector="l2",
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

    def full_affine_operator(self, args=None, /) -> AbstractLinearOperator:
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

    def affine_operator(self, args=None, /) -> AbstractLinearOperator:
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

    def full_right_hand_side(self, args=None, /) -> Array:
        context = self._context(args)
        result = jnp.zeros(
            (self.full_space.size,), dtype=context.runtime.coordinates.dtype
        )
        for action in self.form.actions:
            if isinstance(action, SourceAction):
                for block_index, (geometry, cubature, projection, gathers) in enumerate(
                    zip(
                        context.runtime.geometries,
                        context.runtime.cubatures,
                        context.runtime.projections,
                        self.discretization.dof_map.cell_dofs,
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
                    if values.shape != cubature.weights.shape:
                        raise ValueError("VEM source must be scalar at cell quadrature.")
                    basis = projection.basis.evaluate(
                        cubature.points,
                        geometry.centroids,
                        geometry.characteristic_lengths,
                    )
                    moments = oe.contract(
                        "cq,cq,cqa->ca", cubature.weights, values, basis
                    )
                    local = oe.contract("cai,ca->ci", projection.l2_coefficients, moments)
                    local = jnp.where(_cell_mask(action, indices)[:, None], local, 0.0)
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

    def right_hand_side(self, args=None, /) -> Array:
        context = self._context(args)
        full_rhs = self.full_right_hand_side(context)
        if self.constraint is None:
            return full_rhs
        full_operator = self.full_affine_operator(context)
        lift = self.lift if context.lift is None else context.lift
        return self.constraint.constraint_map.pullback_dual(
            full_rhs - full_operator.mv(lift)
        )

    def full_residual(self, state, args=None, /) -> Array:
        context = self._context(args)
        return self.full_affine_operator(context).mv(state) - self.full_right_hand_side(
            context
        )

    def residual(self, state, args=None, /) -> Array:
        context = self._context(args)
        full = self.expand(state, context)
        residual = self.full_residual(full, context)
        return (
            residual
            if self.constraint is None
            else self.constraint.constraint_map.pullback_dual(residual)
        )

    def _default_nullspace_policy(self, operator) -> NullspacePolicy | None:
        if self.constraint is not None:
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
        args=None,
        /,
        *,
        nullspace_policy: NullspacePolicy | None = None,
    ):
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
        args=None,
        /,
        *,
        policy: SparseAssemblyPolicy | None = None,
    ) -> SparseAssemblyPlan:
        return plan_sparse_assembly(self.affine_operator(args), policy)

    def prepare_sparse_assembly(
        self,
        args=None,
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
        args=None,
        /,
        *,
        coefficient: ArrayLike = 1.0,
        return_full: bool = False,
    ):
        context = self._context(args)
        from .._variational import coefficient as bind_coefficient

        bound = bind_coefficient(coefficient)
        polynomial = _mass_polynomial_matrices(bound, self.discretization, context)
        factorized = _factorized_action(
            context.runtime.projections,
            tuple(value.l2_coefficients for value in context.runtime.projections),
            polynomial,
            self.discretization,
            self.execution_policy,
            projector="l2",
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

        def context(time, args):
            base = self._context(args)
            return VirtualElementExecutionContext(
                base.runtime,
                time=time,
                lift=base.lift,
                lift_rate=base.lift_rate,
                user_args=base.user_args,
            )

        def mass_matrix(time, state, args):
            return self.mass_operator(context(time, args), coefficient=mass_coefficient)

        def vector_field(time, state, args):
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
        args=None,
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


def compile_virtual_element_problem(
    form: VirtualElementForm,
    discretization: VirtualElementDiscretization,
    /,
    *,
    constraint: VirtualElementDirichletConstraint | None = None,
    dirichlet_values=None,
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
