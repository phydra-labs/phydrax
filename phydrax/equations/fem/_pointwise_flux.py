#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pointwise conormal flux and trace-inverse stability of compiled FE owners.

The conormal flux of a compiled finite-element problem is the physical flux
``q = n . K grad(u)`` of its declared ``DiffusionAction`` and
``TensorDiffusionAction`` terms, summed over the actions whose cell domain
contains a facet's side cell. It is evaluated from the discrete field of the
side cell: the discretization owns the whole-cell gradient route at the facet
sites, and this owner contracts it with the diffusivity-weighted outward
normal. The flux is linear in the full coefficients. A form with any other
term acting on the traced field (besides mass, source, and boundary terms,
none of which contributes a volume flux) has no declared boundary flux law and
is refused.

Certification solves, per facet, the local pencil between the flux Gram matrix
on an exact Gauss--Legendre facet rule and the side cell's energy of the same
diffusion terms on the owner's own cell rules (see
``phydrax.discretization.certify_trace_inverse``).
"""

from __future__ import annotations

from math import isqrt
from typing import final, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

import phydrax.ein as ein

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    AbstractSideFluxEvaluator,
    certify_trace_inverse,
    FacetRuleFamily,
    FacetTraceRule,
    PreparedFluxAction,
    PreparedTraceAction,
    SideActionDescriptor,
    TraceInverseEvidence,
)
from ...discretization.fem import FiniteElementDiscretization
from ...discretization.fem._point_interpolation import (
    finite_element_side_revision,
    FiniteElementSideGradient,
    prepare_finite_element_side_gradient,
)
from ...linalg import AbstractVectorSpace
from .._finite_element_variational import (
    _action_domain,
    _action_output_fields,
    _action_rule,
    _facet_subdomain,
    _finite_element_runtime,
    _reference_rule_data,
    _rule_coefficient_values,
    BoundaryLoadAction,
    CompiledFiniteElementProblem,
    DiffusionAction,
    ExteriorFacetAction,
    FiniteElementExecutionContext,
    MassAction,
    SourceAction,
    TensorDiffusionAction,
)
from .._variational import VariationalCoefficient


type _Diffusion = DiffusionAction | TensorDiffusionAction

# Terms that act on the traced field without contributing a volume flux.
_FLUX_FREE = (MassAction, SourceAction, BoundaryLoadAction, ExteriorFacetAction)
# Facet polynomial degree of a degree-k DOF coefficient per cell kind.
_FACET_DEGREE_FACTOR = {
    "interval": 1,
    "triangle": 1,
    "quadrilateral": 1,
    "tetrahedron": 1,
    "hexahedron": 2,
}


def _diffusion_actions(
    problem: CompiledFiniteElementProblem, field_name: str, /
) -> tuple[_Diffusion, ...]:
    """The declared diffusion terms of one field, refusing undeclared flux laws."""
    diffusion: list[_Diffusion] = []
    for action in problem.form.actions:
        if field_name not in _action_output_fields(action):
            continue
        if isinstance(action, (DiffusionAction, TensorDiffusionAction)):
            diffusion.append(action)
        elif not isinstance(action, _FLUX_FREE):
            raise ValueError(
                f"Action {action.action_id!r} ({type(action).__name__}) acts on field "
                f"{field_name!r} without a declared conormal flux law; the pointwise "
                "flux is published for diffusion operators with mass, source, and "
                "boundary terms only."
            )
    if not diffusion:
        raise ValueError(
            f"The form declares no DiffusionAction or TensorDiffusionAction for "
            f"field {field_name!r}; it defines no conormal flux."
        )
    for action in diffusion:
        match action.diffusivity.location:
            case "point" | "cell" | "dof":
                pass
            case location:
                raise ValueError(
                    f"A {location!r} diffusivity is bound to its own rule or entities "
                    "and has no values at facet trace sites."
                )
    return tuple(diffusion)


def _trace_rule(trace: PreparedTraceAction, /) -> FacetTraceRule:
    """The facet rule a trace was prepared on, recovered from its identity."""
    if trace.sites.shape[-1] == 1:
        # Point facets carry one site whatever the declared rule.
        return FacetTraceRule(points=1)
    sites = trace.output_shape[1]
    for family in get_args(FacetRuleFamily):
        for points in sorted({sites, isqrt(sites)}):
            if family == "gauss-lobatto-legendre" and points < 2:
                continue
            rule = FacetTraceRule(family, points=points)
            if rule.rule_id == trace.descriptor.rule_id:
                return rule
    raise ValueError("The trace was not prepared on a facet trace rule.")


def _coefficient_field(
    discretization: FiniteElementDiscretization, coefficient: VariationalCoefficient, /
) -> int:
    fields = [
        index
        for index, space in enumerate(discretization.field_spaces)
        if space.field_space_id == coefficient.field_space_id
    ]
    if len(fields) != 1:
        raise ValueError("The DOF diffusivity field is not uniquely available.")
    return fields[0]


@final
class _SiteCoefficient(StrictModule):
    """Side-cell gathers and basis of a DOF diffusivity at the facet sites."""

    dofs: Array | None
    basis: Array | None
    orientations: Array | None


def _site_coefficient(
    discretization: FiniteElementDiscretization,
    coefficient: VariationalCoefficient,
    gradient: FiniteElementSideGradient,
    /,
) -> _SiteCoefficient:
    if coefficient.location != "dof":
        return _SiteCoefficient(None, None, None)
    index = _coefficient_field(discretization, coefficient)
    blocks = np.asarray(gradient.side_blocks)
    cells = np.asarray(gradient.side_block_cells)
    reference = np.asarray(gradient.reference)
    count, sites, dimension = reference.shape
    dof_map = discretization.dof_maps[index]
    width = max(
        np.asarray(dof_map.cell_dofs[block]).shape[1] for block in np.unique(blocks)
    )
    dofs = np.zeros((count, width), dtype=np.int32)
    basis = np.zeros((count, sites, width))
    orientations = np.zeros((count, width))
    for block in np.unique(blocks).tolist():
        rows = np.flatnonzero(blocks == block)
        element = discretization.elements[index][block]
        values = np.asarray(
            element.tabulate(jnp.asarray(reference[rows].reshape((-1, dimension))))[0]
        ).reshape((rows.size, sites, -1))
        local = values.shape[-1]
        dofs[rows, :local] = np.asarray(dof_map.cell_dofs[block])[cells[rows]]
        basis[rows, :, :local] = values
        orientations[rows, :local] = np.asarray(dof_map.orientations[block])[cells[rows]]
    return _SiteCoefficient(
        jnp.asarray(dofs), jnp.asarray(basis), jnp.asarray(orientations)
    )


def _site_values(
    coefficient: VariationalCoefficient,
    route: _SiteCoefficient,
    discretization: FiniteElementDiscretization,
    gradient: FiniteElementSideGradient,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    """Diffusivity at the facet sites of the side cells."""
    return coefficient.evaluate(
        gradient.sites,
        context,
        entity_indices=gradient.side_cells,
        dof_indices=route.dofs,
        dof_orientations=route.orientations,
        basis_values=route.basis,
        support_id=(
            discretization.support.support_id
            if coefficient.support_id is not None
            else None
        ),
        entity_set_id=(
            discretization.cell_domain.entity_set_id
            if coefficient.entity_set_id is not None
            else None
        ),
        field_space_id=coefficient.field_space_id,
    )


def _conormal_weights(action: _Diffusion, values: Array, normals: Array, /) -> Array:
    """Weights ``w`` with ``q = w . grad(u)``: ``K^T n`` of the physical flux ``K grad(u)``."""
    leading = normals.shape[:-1]
    dimension = normals.shape[-1]
    if isinstance(action, TensorDiffusionAction):
        tensor = action.physical_tensor(values, dimension, leading_shape=leading)
        return ein.contract("fqi,fqij->fqj", normals, tensor)
    if values.shape != leading:
        raise ValueError("A DiffusionAction diffusivity must be scalar at every site.")
    return values[..., None] * normals


def _cell_tensor(action: _Diffusion, values: Array, points: Array, /) -> Array:
    """Physical ``(flux, gradient)`` tensor of one diffusion term at cell points."""
    dimension = points.shape[-1]
    if isinstance(action, TensorDiffusionAction):
        return action.physical_tensor(values, dimension, leading_shape=points.shape[:-1])
    return values[..., None, None] * jnp.eye(dimension, dtype=values.dtype)


@final
class _FiniteElementPointwiseFlux(AbstractSideFluxEvaluator, NonTrainableState):
    """``n . K grad(u)`` of the declared diffusion terms at fixed facet sites."""

    problem: CompiledFiniteElementProblem
    gradient: FiniteElementSideGradient
    actions: tuple[_Diffusion, ...]
    active: tuple[Array, ...]
    coefficients: tuple[_SiteCoefficient, ...]
    field_position: int = eqx.field(static=True)

    @property
    def state_space(self) -> AbstractVectorSpace:
        return self.problem.full_space

    def conormal(self, args: object, /) -> Array:
        """Site weights ``w`` of the flux ``q = w . grad(u)`` for one argument set."""
        context = self.problem._execution_context(args)
        if _finite_element_runtime(context).runtime_id != self.gradient.runtime_id:
            raise ValueError(
                "The pointwise flux was prepared on another finite-element runtime; "
                "prepare the trace and flux on the evaluating runtime."
            )
        discretization = self.problem.discretization
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("Pointwise fluxes belong to finite-element discretizations.")
        normals = self.gradient.normals
        weights = jnp.zeros(normals.shape, dtype=normals.dtype)
        for action, active, route in zip(
            self.actions, self.active, self.coefficients, strict=True
        ):
            values = _site_values(
                action.diffusivity, route, discretization, self.gradient, context
            )
            weights = weights + jnp.where(
                active[:, None, None], _conormal_weights(action, values, normals), 0.0
            )
        return weights

    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        full = (
            state
            if len(self.problem.form.field_names) == 1
            else state[self.field_position]
        )
        gradient = self.gradient.route.apply(jnp.asarray(full))
        return jnp.sum(self.conormal(args) * gradient, axis=-1)


def _pointwise_evaluator(
    problem: CompiledFiniteElementProblem,
    field_position: int,
    gradient: FiniteElementSideGradient,
    actions: tuple[_Diffusion, ...],
    /,
) -> _FiniteElementPointwiseFlux:
    discretization = problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("Pointwise fluxes belong to finite-element discretizations.")
    cells = np.asarray(gradient.side_cells)
    active = tuple(
        jnp.asarray(
            np.isin(
                cells, np.asarray(_action_domain(action, discretization).entity_indices)
            )
        )
        for action in actions
    )
    coefficients = tuple(
        _site_coefficient(discretization, action.diffusivity, gradient)
        for action in actions
    )
    return _FiniteElementPointwiseFlux(
        problem, gradient, actions, active, coefficients, field_position
    )


def _coefficient_degree(
    discretization: FiniteElementDiscretization, coefficient: VariationalCoefficient, /
) -> int | None:
    """Polynomial degree of a diffusivity along facets (None when not polynomial)."""
    match coefficient.location:
        case "point":
            return 0 if coefficient.constant else None
        case "cell":
            return 0
        case "dof":
            elements = discretization.elements[
                _coefficient_field(discretization, coefficient)
            ]
            factors = [
                _FACET_DEGREE_FACTOR.get(element.cell_kind) for element in elements
            ]
            if any(factor is None for factor in factors):
                return None
            return max(
                element.degree * factor
                for element, factor in zip(elements, factors, strict=True)
                if factor is not None
            )
        case _:
            return None


def _flux_degree(
    discretization: FiniteElementDiscretization,
    gradient: FiniteElementSideGradient,
    actions: tuple[_Diffusion, ...],
    /,
) -> int | None:
    degrees = [
        _coefficient_degree(discretization, action.diffusivity) for action in actions
    ]
    if gradient.gradient_degree is None or any(degree is None for degree in degrees):
        return None
    return gradient.gradient_degree + max(
        degree for degree in degrees if degree is not None
    )


def _require_trace_sites(
    gradient: FiniteElementSideGradient, trace: PreparedTraceAction, /
) -> None:
    sites = np.asarray(trace.sites)
    scale = max(float(np.max(np.abs(sites))), 1.0)
    if (
        np.max(np.abs(np.asarray(gradient.sites) - sites)) > 1.0e-10 * scale
        or np.max(np.abs(np.asarray(gradient.normals) - np.asarray(trace.normals)))
        > 1.0e-10
    ):
        raise ValueError(
            "The trace's sites or normals differ from the owner's facet embedding; "
            "prepare the trace from this problem's discretization facet domains."
        )


def prepare_finite_element_pointwise_flux(
    problem: CompiledFiniteElementProblem,
    field_position: int,
    trace: PreparedTraceAction,
    /,
) -> PreparedFluxAction:
    """Publish the exact pointwise conormal flux of one validated value trace."""
    descriptor = trace.descriptor
    if descriptor.quantity != "value" or descriptor.representation != "quadrature-values":
        raise ValueError("Pointwise fluxes are published at the sites of value traces.")
    discretization = problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise ValueError(
            "Exact pointwise fluxes are published for finite-element discretizations; "
            "this owner's discretization publishes no whole-cell facet gradient route."
        )
    field_name = problem.form.field_names[field_position]
    actions = _diffusion_actions(problem, field_name)
    runtime = discretization.default_runtime
    if descriptor.revision_id != finite_element_side_revision(runtime):
        raise ValueError(
            "The trace was prepared on another geometry revision; prepare it on the "
            "problem's default runtime."
        )
    domain = _facet_subdomain(
        discretization, descriptor.domain_kind, np.asarray(descriptor.facets)
    )
    gradient = prepare_finite_element_side_gradient(
        discretization,
        field_name,
        domain,
        rule=_trace_rule(trace),
        side=descriptor.side,
        runtime=runtime,
    )
    _require_trace_sites(gradient, trace)
    flux = SideActionDescriptor(
        owner_id=problem.compilation_id,
        field_space_id=descriptor.field_space_id,
        quantity="conormal-flux",
        representation="quadrature-values",
        orientation="outward",
        approximation="exact",
        side=descriptor.side,
        domain=domain,
        revision_id=descriptor.revision_id,
        rule=_trace_rule(trace),
        trace_degree=_flux_degree(discretization, gradient, actions),
        quadrature_exact_degree=descriptor.quadrature_exact_degree,
    )
    return PreparedFluxAction(
        flux, trace, _pointwise_evaluator(problem, field_position, gradient, actions)
    )


def _block_energy(
    problem: CompiledFiniteElementProblem,
    field_index: int,
    block: int,
    cells: np.ndarray,
    side_cells: np.ndarray,
    actions: tuple[_Diffusion, ...],
    /,
) -> tuple[Array, Array]:
    """Unoriented energy matrices and basis moments of selected cells of one block."""
    discretization = problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("Pointwise fluxes belong to finite-element discretizations.")
    context = problem._execution_context(None)
    coordinates = _finite_element_runtime(context).coordinates
    mesh_block = discretization.mesh.blocks[block]
    start = sum(item.cell_count for item in discretization.mesh.blocks[:block])
    entities = np.arange(start, start + mesh_block.cell_count, dtype=np.int32)
    field_name = discretization.field_spaces[field_index].name
    energy: Array | None = None
    moments: Array | None = None
    for action in actions:
        rule = _action_rule(action, mesh_block.name, mesh_block.cell_kind)
        rule_data = _reference_rule_data(rule)
        geometry = discretization.evaluate_block_geometry(
            field_name, block, coordinates, rule_data.points, rule_data.weights
        )
        values = _rule_coefficient_values(
            action.diffusivity,
            discretization,
            context,
            block,
            rule,
            rule_data,
            geometry.physical_points,
            entities,
        )
        # The energy a_K(v, v) = int grad(v) . K grad(v) sees only the symmetric part.
        tensor = _cell_tensor(action, values, geometry.physical_points)[cells]
        tensor = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
        weights = geometry.physical_weights[cells]
        gradients = geometry.physical_gradients[cells]
        local = ein.contract(
            "cq,cqid,cqde,cqje->cij", weights, gradients, tensor, gradients
        )
        inside = np.isin(
            side_cells, np.asarray(_action_domain(action, discretization).entity_indices)
        )
        local = jnp.where(jnp.asarray(inside)[:, None, None], local, 0.0)
        energy = local if energy is None else energy + local
        if moments is None:
            # Any positive cell rule gives r . c = |K|_h > 0 for the constant c.
            moments = ein.contract("cq,qi->ci", weights, geometry.basis_values)
    if energy is None or moments is None:
        raise ValueError("Certification needs at least one diffusion term.")
    return energy, moments


def _side_cell_energy(
    problem: CompiledFiniteElementProblem,
    field_index: int,
    gradient: FiniteElementSideGradient,
    actions: tuple[_Diffusion, ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Oriented cell energies ``A_K`` and moments ``int_K phi`` in the gradient layout."""
    discretization = problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("Pointwise fluxes belong to finite-element discretizations.")
    blocks = np.asarray(gradient.side_blocks)
    cells = np.asarray(gradient.side_block_cells)
    side_cells = np.asarray(gradient.side_cells)
    count, width = gradient.valid.shape
    energy = np.zeros((count, width, width))
    moments = np.zeros((count, width))
    for block in np.unique(blocks).tolist():
        rows = np.flatnonzero(blocks == block)
        local, cell_moments = _block_energy(
            problem, field_index, block, cells[rows], side_cells[rows], actions
        )
        orientation = np.asarray(
            discretization.dof_maps[field_index].orientations[block]
        )[cells[rows]]
        size = orientation.shape[1]
        energy[rows, :size, :size] = (
            np.asarray(local) * orientation[:, :, None] * orientation[:, None, :]
        )
        moments[rows, :size] = np.asarray(cell_moments) * orientation
    return energy, moments


def _exact_gradient(
    discretization: FiniteElementDiscretization,
    field_name: str,
    flux: PreparedFluxAction,
    degree: int,
    /,
) -> tuple[FiniteElementSideGradient, int]:
    """Gradient on a Gauss-Legendre facet rule exact for the squared flux."""
    descriptor = flux.descriptor
    domain = _facet_subdomain(
        discretization, descriptor.domain_kind, np.asarray(descriptor.facets)
    )
    points = degree + 1
    while True:
        gradient = prepare_finite_element_side_gradient(
            discretization,
            field_name,
            domain,
            rule=FacetTraceRule("gauss-legendre", points=points),
            side=descriptor.side,
            runtime=discretization.default_runtime,
        )
        exact = gradient.exact_degree
        if exact is None:
            return gradient, 2 * degree
        if exact >= 2 * degree:
            return gradient, exact
        points += 1


def certify_finite_element_flux_stability(
    problem: CompiledFiniteElementProblem, flux: PreparedFluxAction, /
) -> TraceInverseEvidence:
    """Certify the trace-inverse constants of one pointwise flux of ``problem``."""
    if not isinstance(flux, PreparedFluxAction):
        raise TypeError("flux must be a PreparedFluxAction.")
    evaluator = flux.evaluator
    if flux.descriptor.owner_id != problem.compilation_id or not isinstance(
        evaluator, _FiniteElementPointwiseFlux
    ):
        raise ValueError(
            "The flux is not a pointwise flux published by this problem's "
            "prepare_pointwise_flux."
        )
    degree = flux.descriptor.trace_degree
    if degree is None:
        raise ValueError(
            "Certified trace-inverse constants need a flux that is polynomial along "
            "the facets; this flux publishes no facet degree (callable diffusivity "
            "or side cells that are not affine along the facets)."
        )
    discretization = problem.discretization
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("Pointwise fluxes belong to finite-element discretizations.")
    field_name = problem.form.field_names[evaluator.field_position]
    gradient, exact = _exact_gradient(discretization, field_name, flux, degree)
    probe = _pointwise_evaluator(
        problem, evaluator.field_position, gradient, evaluator.actions
    )
    rows = jnp.sqrt(gradient.weights)[..., None] * ein.contract(
        "fqd,fqld->fql", probe.conormal(None), gradient.route.weights
    )
    energy, moments = _side_cell_energy(
        problem,
        discretization._field_index(field_name),
        gradient,
        evaluator.actions,
    )
    return certify_trace_inverse(
        rows,
        energy,
        moments,
        gradient.valid,
        owner_id=problem.compilation_id,
        flux_action_id=flux.action_id,
        facets=np.asarray(flux.descriptor.facets),
        side_cells=np.asarray(gradient.side_cells),
        facet_rule_id=gradient.rule_id,
        facet_exact_degree=exact,
    )


__all__ = [
    "certify_finite_element_flux_stability",
    "prepare_finite_element_pointwise_flux",
]
