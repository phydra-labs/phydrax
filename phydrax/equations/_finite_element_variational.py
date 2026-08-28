#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax
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
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ..discretization.fem import (
    FiniteElementDirichletConstraint,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
    IntegrationDomain,
)
from ..dynamics import (
    DAEStructure,
    DifferentialAlgebraicSystem,
    SecondOrderDifferentialSystem,
)
from ..linalg import (
    adjoint,
    BlockSpace,
    DualSpace,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    NullspacePolicy,
    OperatorProperties,
    plan_sparse_assembly,
    prepare_sparse_assembly,
    PreparedSparseAssembly,
    solve,
    SparseAssemblyPlan,
    SparseAssemblyPolicy,
)
from ..linalg.eigen import GeneralizedEigenproblem
from ..nonlinear import LaggedLinearSolveUpdate, NonlinearSystemProblem
from ..sparse import SparseCoordinateOperator


ReferenceRule: TypeAlias = Any


def _reference_rule_data(rule: ReferenceRule, /):
    from ..integration import reference_rule_data

    return reference_rule_data(rule)


def _interval_rule():
    from ..integration import ReferenceIntervalRule

    return ReferenceIntervalRule()


def _triangle_rule():
    from ..integration import ReferenceTriangleRule

    return ReferenceTriangleRule()


def _quadrilateral_rule():
    from ..integration import ReferenceQuadrilateralRule

    return ReferenceQuadrilateralRule()


def _tetrahedron_rule():
    from ..integration import ReferenceTetrahedronRule

    return ReferenceTetrahedronRule()


def _rule_id(rule: ReferenceRule, /) -> str:
    data = _reference_rule_data(rule)
    return canonical_fingerprint(
        {
            "kind": "finite-element-reference-rule",
            "rule_type": type(rule).__name__,
            "reference_domain": data.cell,
            "points": array_tree_fingerprint(np.asarray(data.points)),
            "weights": array_tree_fingerprint(np.asarray(data.weights)),
        }
    )


def _normalize_rules(
    rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]],
    /,
) -> tuple[tuple[str, ReferenceRule], ...]:
    items = tuple(rules.items()) if isinstance(rules, Mapping) else tuple(rules)
    normalized = tuple(sorted(((str(name), rule) for name, rule in items)))
    names = tuple(name for name, _ in normalized)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Reference-rule block names must be unique and non-empty.")
    for _, rule in normalized:
        _reference_rule_data(rule)
    return normalized


def _default_rule(cell_kind: str, /) -> ReferenceRule:
    if cell_kind == "triangle":
        return _triangle_rule()
    if cell_kind == "quadrilateral":
        return _quadrilateral_rule()
    if cell_kind == "tetrahedron":
        return _tetrahedron_rule()
    raise ValueError(f"No finite-element rule exists for cell kind {cell_kind!r}.")


class _ResolvedCoefficient(StrictModule, NonTrainableState):
    """Typed constant or pure staged coefficient used by FE terms."""

    value: Array
    evaluator: Callable[[Array, object], ArrayLike] | None
    location: str = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike | Callable[[Array, object], ArrayLike],
        /,
        *,
        coefficient_id: str | None = None,
        location: str = "point",
    ):
        location_ = str(location)
        if location_ not in ("point", "cell", "facet", "quadrature"):
            raise ValueError(
                "Coefficient location must be point, cell, facet, or quadrature."
            )
        if callable(value) and location_ != "point":
            raise ValueError("Callable coefficients currently require point location.")
        if callable(value):
            if coefficient_id is None or not str(coefficient_id):
                raise ValueError(
                    "Callable coefficients require an explicit coefficient_id."
                )
            self.value = jnp.asarray(0.0)
            self.evaluator = value
            self.location = location_
            self.coefficient_id = str(coefficient_id)
        else:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.inexact):
                array = array.astype(float)
            self.value = array
            self.evaluator = None
            self.location = location_
            self.coefficient_id = (
                canonical_fingerprint(
                    {
                        "kind": "finite-element-constant-coefficient",
                        "value": array_tree_fingerprint(np.asarray(array)),
                        "location": location_,
                    }
                )
                if coefficient_id is None
                else str(coefficient_id)
            )
            if not self.coefficient_id:
                raise ValueError("coefficient_id must be non-empty.")

    @property
    def constant(self) -> bool:
        return self.evaluator is None and self.location == "point"

    def evaluate(
        self,
        points: Array,
        args: object = None,
        /,
        *,
        entity_indices: ArrayLike | None = None,
    ) -> Array:
        if self.evaluator is not None:
            return jnp.asarray(self.evaluator(points, args))
        if self.location == "point":
            return jnp.broadcast_to(
                self.value,
                points.shape[:-1] + self.value.shape,
            )
        if entity_indices is None:
            raise ValueError("Entity/quadrature coefficients require entity indices.")
        indices = jnp.asarray(entity_indices, dtype=jnp.int32)
        selected = self.value[indices]
        if self.location in ("cell", "facet"):
            shape = (selected.shape[0],) + (1,) * (points.ndim - 2) + selected.shape[1:]
            selected = selected.reshape(shape)
            return jnp.broadcast_to(
                selected,
                points.shape[:-1] + selected.shape[points.ndim - 1 :],
            )
        if selected.shape[: points.ndim - 1] != points.shape[:-1]:
            raise ValueError(
                "Quadrature coefficient leading shape must match selected points."
            )
        return selected


def coefficient(
    value: ArrayLike | Callable[[Array, object], ArrayLike],
    /,
    *,
    coefficient_id: str | None = None,
    location: str = "point",
) -> _ResolvedCoefficient:
    return _ResolvedCoefficient(
        value,
        coefficient_id=coefficient_id,
        location=location,
    )


class FiniteElementExecutionContext(StrictModule, NonTrainableState):
    """Dynamic geometry, time, lift, and user arguments for FE execution."""

    runtime: FiniteElementRuntimeData
    time: Array
    lift: Array | None
    lift_rate: Array | None
    lift_acceleration: Array | None
    user_args: object

    def __init__(
        self,
        runtime: FiniteElementRuntimeData,
        /,
        *,
        time: ArrayLike = 0.0,
        lift: ArrayLike | None = None,
        lift_rate: ArrayLike | None = None,
        lift_acceleration: ArrayLike | None = None,
        user_args: object = None,
    ):
        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        self.runtime = runtime
        self.time = jnp.asarray(time)
        self.lift = None if lift is None else jnp.asarray(lift)
        self.lift_rate = None if lift_rate is None else jnp.asarray(lift_rate)
        self.lift_acceleration = (
            None if lift_acceleration is None else jnp.asarray(lift_acceleration)
        )
        self.user_args = user_args


class FiniteElementExecutionPolicy(StrictModule, NonTrainableState):
    """Operator realization and reduction policy for compiled FE forms."""

    realization: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        realization: str = "sparse",
        accumulation: str = "fast",
    ):
        realization_ = str(realization)
        accumulation_ = str(accumulation)
        if realization_ not in ("matrix_free", "sparse"):
            raise ValueError("Unknown finite-element operator realization.")
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown finite-element accumulation policy.")
        self.realization = realization_
        self.accumulation = accumulation_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-execution-policy",
                "realization": realization_,
                "accumulation": accumulation_,
            }
        )


class DiffusionTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    diffusivity: _ResolvedCoefficient
    term_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        field_name: str,
        diffusivity: _ResolvedCoefficient | ArrayLike = 1.0,
        /,
        *,
        term_id: str = "diffusion",
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Diffusion field and term IDs must be non-empty.")
        self.field_name = field
        self.diffusivity = (
            diffusivity
            if isinstance(diffusivity, _ResolvedCoefficient)
            else coefficient(diffusivity)
        )
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("DiffusionTerm requires a cell integration domain.")
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class MassTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    coefficient: _ResolvedCoefficient
    term_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        field_name: str,
        value: _ResolvedCoefficient | ArrayLike = 1.0,
        /,
        *,
        term_id: str = "mass",
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Mass field and term IDs must be non-empty.")
        self.field_name = field
        self.coefficient = (
            value if isinstance(value, _ResolvedCoefficient) else coefficient(value)
        )
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("MassTerm requires a cell integration domain.")
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class SourceTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    source: _ResolvedCoefficient
    term_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        field_name: str,
        source: _ResolvedCoefficient | ArrayLike,
        /,
        *,
        term_id: str = "source",
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Source field and term IDs must be non-empty.")
        self.field_name = field
        self.source = (
            source if isinstance(source, _ResolvedCoefficient) else coefficient(source)
        )
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("SourceTerm requires a cell integration domain.")
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class BoundaryLoadTerm(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    load: _ResolvedCoefficient
    term_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        field_name: str,
        load: _ResolvedCoefficient | ArrayLike,
        /,
        *,
        term_id: str = "boundary-load",
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not identifier:
            raise ValueError("Boundary-load field and term IDs must be non-empty.")
        self.field_name = field
        self.load = load if isinstance(load, _ResolvedCoefficient) else coefficient(load)
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "exterior_facet":
            raise ValueError(
                "BoundaryLoadTerm requires an exterior-facet integration domain."
            )
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class CellResidualTerm(StrictModule, NonTrainableState):
    """User-defined cell-local residual with explicit field dependencies."""

    field_name: str = eqx.field(static=True)
    input_fields: tuple[str, ...] = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        input_fields: Sequence[str],
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        term_id: str,
    ):
        output = str(field_name)
        inputs = tuple(str(value) for value in input_fields)
        identifier = str(term_id)
        if not output or not inputs or any(not value for value in inputs):
            raise ValueError("Residual output/input field names must be non-empty.")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Residual input field names must be unique.")
        if not callable(kernel) or not identifier:
            raise ValueError("Residual kernel and term_id are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellResidualTerm requires a cell integration domain.")
        self.field_name = output
        self.input_fields = inputs
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class InteriorFacetTerm(StrictModule, NonTrainableState):
    """Two-sided numerical flux density over interior facets."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        term_id: str,
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError("Interior facet field, kernel, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "interior_facet"
        ):
            raise ValueError("InteriorFacetTerm requires an interior-facet domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class CellEnergyTerm(StrictModule, NonTrainableState):
    """Cell-local scalar energy differentiated into a residual."""

    field_name: str = eqx.field(static=True)
    density: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        density: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        term_id: str,
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not callable(density) or not identifier:
            raise ValueError("Energy field, density, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellEnergyTerm requires a cell domain.")
        self.field_name = field
        self.density = density
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


class CellBilinearTerm(StrictModule, NonTrainableState):
    """User-provided cell-local matrix over one field."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        term_id: str,
    ):
        field = str(field_name)
        identifier = str(term_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError("Bilinear field, kernel, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellBilinearTerm requires a cell domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.term_id = identifier


FiniteElementTerm = (
    DiffusionTerm
    | MassTerm
    | SourceTerm
    | BoundaryLoadTerm
    | CellResidualTerm
    | InteriorFacetTerm
    | CellEnergyTerm
    | CellBilinearTerm
)


class _FiniteElementWorkBlock(StrictModule, NonTrainableState):
    block_index: int = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_dofs: Array
    basis_values: Array
    reference_gradients: Array
    cell_indices: Array
    reference_points: Array
    reference_weights: Array
    work_id: str = eqx.field(static=True)


def _term_payload(term: FiniteElementTerm, /) -> dict[str, object]:
    if isinstance(term, DiffusionTerm):
        coefficient_id = term.diffusivity.coefficient_id
        kind = "diffusion"
    elif isinstance(term, MassTerm):
        coefficient_id = term.coefficient.coefficient_id
        kind = "mass"
    elif isinstance(term, SourceTerm):
        coefficient_id = term.source.coefficient_id
        kind = "source"
    elif isinstance(term, BoundaryLoadTerm):
        coefficient_id = term.load.coefficient_id
        kind = "boundary-load"
    elif isinstance(term, CellResidualTerm):
        coefficient_id = None
        kind = "cell-residual"
    elif isinstance(term, InteriorFacetTerm):
        coefficient_id = None
        kind = "interior-facet"
    elif isinstance(term, CellEnergyTerm):
        coefficient_id = None
        kind = "cell-energy"
    elif isinstance(term, CellBilinearTerm):
        coefficient_id = None
        kind = "cell-bilinear"
    else:
        raise TypeError("Unsupported finite-element term.")
    return {
        "kind": kind,
        "term_id": term.term_id,
        "field_name": term.field_name,
        "coefficient_id": coefficient_id,
        "input_fields": (
            list(term.input_fields)
            if isinstance(term, CellResidualTerm)
            else [term.field_name]
        ),
        "domain_id": None if term.domain is None else term.domain.domain_id,
        "rules": [[block_name, _rule_id(rule)] for block_name, rule in term.rules],
    }


class WeakForm(StrictModule, NonTrainableState):
    form_id: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    terms: tuple[FiniteElementTerm, ...]

    def __init__(
        self,
        form_id: str,
        field_name: str | Sequence[str],
        terms: Sequence[FiniteElementTerm],
        /,
    ):
        identifier = str(form_id)
        fields = (
            (str(field_name),)
            if isinstance(field_name, str)
            else tuple(str(value) for value in field_name)
        )
        term_values = tuple(terms)
        if not identifier or not fields or any(not field for field in fields):
            raise ValueError("Weak-form and field IDs must be non-empty.")
        if len(set(fields)) != len(fields):
            raise ValueError("Weak-form field names must be unique.")
        if not term_values:
            raise ValueError("WeakForm requires at least one term.")
        if not all(
            isinstance(
                term,
                (
                    DiffusionTerm,
                    MassTerm,
                    SourceTerm,
                    BoundaryLoadTerm,
                    CellResidualTerm,
                    InteriorFacetTerm,
                    CellEnergyTerm,
                    CellBilinearTerm,
                ),
            )
            for term in term_values
        ):
            raise TypeError("WeakForm contains an unsupported term type.")
        if any(term.field_name not in fields for term in term_values):
            raise ValueError("Every weak-form term must target a declared field.")
        term_ids = tuple(term.term_id for term in term_values)
        if any(
            isinstance(term, CellResidualTerm)
            and any(input_field not in fields for input_field in term.input_fields)
            for term in term_values
        ):
            raise ValueError("Cell residual inputs must be declared weak-form fields.")
        if len(set(term_ids)) != len(term_ids):
            raise ValueError("Weak-form term IDs must be unique.")
        self.form_id = canonical_fingerprint(
            {
                "kind": "finite-element-weak-form",
                "declared_id": identifier,
                "field_names": list(fields),
                "terms": [_term_payload(term) for term in term_values],
            }
        )
        self.field_names = fields
        self.terms = term_values

    @property
    def field_name(self) -> str:
        if len(self.field_names) != 1:
            raise ValueError("field_name is ambiguous for a mixed weak form.")
        return self.field_names[0]


def _term_domain(
    term: FiniteElementTerm,
    discretization: FiniteElementDiscretization,
    /,
) -> IntegrationDomain:
    if term.domain is not None:
        domain = term.domain
    elif isinstance(term, BoundaryLoadTerm):
        domain = discretization.exterior_facet_domain
    elif isinstance(term, InteriorFacetTerm):
        domain = discretization.interior_facet_domain
    else:
        domain = discretization.cell_domain
    if domain.support_id != discretization.support.support_id:
        raise ValueError("Finite-element term domain belongs to another support.")
    return domain


def _term_rule(
    term: FiniteElementTerm,
    block_name: str,
    cell_kind: str,
    /,
) -> ReferenceRule:
    rules = dict(term.rules)
    rule = rules.get(block_name, _default_rule(cell_kind))
    data = _reference_rule_data(rule)
    if data.cell != cell_kind:
        raise ValueError(
            f"Reference rule {data.cell!r} does not match cell kind {cell_kind!r}."
        )
    return rule


class FiniteElementFunctional(StrictModule, NonTrainableState):
    functional_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    density: Callable[[Array, Array, Array, object], ArrayLike]
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        functional_id: str,
        field_name: str,
        density: Callable[[Array, Array, Array, object], ArrayLike],
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        identifier = str(functional_id)
        field = str(field_name)
        if not identifier or not field or not callable(density):
            raise ValueError("Functional ID, field, and callable density are required.")
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("FiniteElementFunctional currently requires a cell domain.")
        normalized_rules = _normalize_rules(rules)
        self.functional_id = canonical_fingerprint(
            {
                "kind": "finite-element-functional",
                "declared_id": identifier,
                "field_name": field,
                "domain": None if domain is None else domain.domain_id,
                "rules": [
                    [block_name, _rule_id(rule)] for block_name, rule in normalized_rules
                ],
            }
        )
        self.field_name = field
        self.density = density
        self.domain = domain
        self.rules = normalized_rules

    def evaluate(
        self,
        discretization: FiniteElementDiscretization,
        state: ArrayLike,
        args: object = None,
        /,
    ) -> Array:
        field_index = discretization._field_index(self.field_name)
        values = discretization.field_spaces[field_index].vector_space.validate(state)
        context = (
            args
            if isinstance(args, FiniteElementExecutionContext)
            else FiniteElementExecutionContext(
                discretization.default_runtime,
                user_args=args,
            )
        )
        domain = discretization.cell_domain if self.domain is None else self.domain
        if domain.support_id != discretization.support.support_id:
            raise ValueError("Functional domain belongs to another support.")
        rules = dict(self.rules)
        contributions = []
        cell_offset = 0
        for block_index, (block, dofs) in enumerate(
            zip(
                discretization.mesh.blocks,
                discretization.dof_maps[field_index].cell_dofs,
                strict=True,
            )
        ):
            block_cells = jnp.arange(
                cell_offset,
                cell_offset + block.cell_count,
                dtype=jnp.int32,
            )
            cell_offset += block.cell_count
            selected = jnp.isin(block_cells, domain.entity_indices)
            rule = rules.get(block.name, _default_rule(block.cell_kind))
            data = _reference_rule_data(rule)
            if data.cell != block.cell_kind:
                raise ValueError("Functional rule does not match its cell block.")
            geometry = discretization.evaluate_block_geometry(
                self.field_name,
                block_index,
                context.runtime.coordinates,
                data.points,
                data.weights,
            )
            local = values[dofs]
            field_values = oe.contract(
                "qi,ci...->cq...",
                geometry.basis_values,
                local,
            )
            gradients = oe.contract(
                "cqid,ci...->cqd...",
                geometry.physical_gradients,
                local,
            )
            density = jnp.asarray(
                self.density(
                    field_values,
                    gradients,
                    geometry.physical_points,
                    context,
                )
            )
            expected = geometry.physical_weights.shape
            if density.shape != expected:
                raise ValueError(
                    "Finite-element functional density must return one scalar "
                    "per selected quadrature point."
                )
            contributions.append(
                discretization.precision_policy.accumulation(
                    density * geometry.physical_weights * selected[:, None]
                ).reshape((-1,))
            )
        if not contributions:
            return discretization.precision_policy.output(jnp.asarray(0.0))
        combined = jnp.concatenate(tuple(contributions))
        if discretization.precision_policy.compensated_accumulation:
            return discretization.precision_policy.output(compensated_sum(combined))
        return discretization.precision_policy.output(jnp.sum(combined))


class CompiledFiniteElementProblem(StrictModule, NonTrainableState):
    form: WeakForm
    discretization: FiniteElementDiscretization
    constraint: FiniteElementDirichletConstraint | None
    execution_policy: FiniteElementExecutionPolicy
    _action_ir: object
    _workset_program: object
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
        execution_policy: FiniteElementExecutionPolicy | None = None,
    ):
        if not isinstance(form, WeakForm):
            raise TypeError("form must be a WeakForm.")
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        policy = (
            FiniteElementExecutionPolicy()
            if execution_policy is None
            else execution_policy
        )
        if not isinstance(policy, FiniteElementExecutionPolicy):
            raise TypeError(
                "execution_policy must be FiniteElementExecutionPolicy or None."
            )
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
        work_block_values = []
        cell_offset = 0
        for block_index, (dofs, geometry) in enumerate(
            zip(
                discretization.dof_maps[field_index].cell_dofs,
                discretization.block_geometries[field_index],
                strict=True,
            )
        ):
            cell_indices = jnp.arange(
                cell_offset,
                cell_offset + dofs.shape[0],
                dtype=jnp.int32,
            )
            cell_offset += dofs.shape[0]
            work_block_values.append(
                _FiniteElementWorkBlock(
                    block_index=block_index,
                    block_name=geometry.block_name,
                    cell_dofs=dofs,
                    basis_values=geometry.basis_values,
                    reference_gradients=geometry.reference_gradients,
                    cell_indices=cell_indices,
                    reference_points=geometry.reference_points,
                    reference_weights=geometry.reference_weights,
                    work_id=canonical_fingerprint(
                        {
                            "kind": "finite-element-work-block",
                            "form": form.form_id,
                            "discretization": discretization.prepared_id,
                            "block_index": block_index,
                            "block": geometry.block_name,
                            "cell_dofs": array_tree_fingerprint(np.asarray(dofs)),
                            "cell_indices": array_tree_fingerprint(
                                np.asarray(cell_indices)
                            ),
                            "reference_points": array_tree_fingerprint(
                                np.asarray(geometry.reference_points)
                            ),
                            "reference_weights": array_tree_fingerprint(
                                np.asarray(geometry.reference_weights)
                            ),
                        }
                    ),
                )
            )
        work_blocks = tuple(work_block_values)
        from .fem import compile_workset_program, lower_weak_form

        action_ir = lower_weak_form(form, discretization)
        workset_program = compile_workset_program(
            action_ir,
            form,
            discretization,
        )
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-finite-element-problem",
                "form": form.form_id,
                "terms": [_term_payload(term) for term in form.terms],
                "discretization": discretization.prepared_id,
                "precision": discretization.precision_policy.policy_id,
                "constraint": (None if constraint is None else constraint.constraint_id),
                "lift": array_tree_fingerprint(np.asarray(lift)),
                "work_blocks": [block.work_id for block in work_blocks],
                "action_ir": action_ir.ir_id,
                "workset_program": workset_program.program_id,
                "execution_policy": policy.policy_id,
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
        self.execution_policy = policy
        self._action_ir = action_ir
        self._workset_program = workset_program
        self.lift = lift
        self.work_blocks = work_blocks
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

    def _execution_context(
        self,
        args: object,
        /,
    ) -> FiniteElementExecutionContext:
        if isinstance(args, FiniteElementExecutionContext):
            context = args
        else:
            context = FiniteElementExecutionContext(
                self.discretization.default_runtime,
                lift=self.lift,
                user_args=args,
            )
        if (
            context.runtime.topology_id != self.discretization.mesh.topology_id
            or context.runtime.geometry_layout_id
            != self.discretization.default_runtime.geometry_layout_id
        ):
            raise ValueError(
                "Finite-element execution runtime does not match the compiled layout."
            )
        return context

    def expand(self, state: ArrayLike, args: object = None, /) -> Array:
        if self.constraint is None:
            return self.full_space.validate(state)
        context = self._execution_context(args)
        lift = self.lift if context.lift is None else context.lift
        return self.constraint.constraint_map.expand(state, lift)

    def full_residual(self, state: ArrayLike, args: object = None, /) -> Array:
        full = self.full_space.validate(state)
        context = self._execution_context(args)
        return _full_residual(
            self.form,
            self.discretization,
            self.work_blocks,
            full,
            self.execution_policy.accumulation,
            context,
        )

    def residual(self, state: ArrayLike, args: object = None, /) -> Array:
        context = self._execution_context(args)
        full_residual = self.full_residual(self.expand(state, context), context)
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

    def linearization_operator(
        self,
        state: ArrayLike,
        args: object = None,
        /,
    ) -> FunctionLinearOperator:
        state_ = self.state_space.validate(state)
        context = self._execution_context(args)
        return FunctionLinearOperator(
            lambda direction: jax.jvp(
                lambda value: self.residual(value, context),
                (state_,),
                (direction,),
            )[1],
            source=self.state_space,
            target=self.residual_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-linearization",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )

    def operator_function(self, state, args):
        if self.execution_policy.realization == "sparse":
            try_affine = all(
                (
                    (
                        isinstance(term, DiffusionTerm)
                        and term.diffusivity.constant
                        and term.diffusivity.value.shape == ()
                    )
                    or (
                        isinstance(term, MassTerm)
                        and term.coefficient.constant
                        and term.coefficient.value.shape == ()
                    )
                    or isinstance(term, (SourceTerm, BoundaryLoadTerm))
                )
                and term.domain is None
                and not term.rules
                for term in self.form.terms
            )
            if try_affine:
                return self.affine_operator(args)
        return self.linearization_operator(state, args)

    def lagged_update(
        self,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        damping: float = 1.0,
    ) -> LaggedLinearSolveUpdate:
        return LaggedLinearSolveUpdate(
            self.operator_function,
            linear_policy=linear_policy,
            damping=damping,
            update_id=canonical_fingerprint(
                {
                    "kind": "finite-element-lagged-update",
                    "compilation": self.compilation_id,
                }
            ),
        )

    def solve_adjoint(
        self,
        solution: ArrayLike,
        cotangent: ArrayLike,
        args: object = None,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        raw_jacobian = self.linearization_operator(solution, args)
        primal_jacobian = FunctionLinearOperator(
            lambda direction: self.state_space.inverse_riesz(raw_jacobian.mv(direction)),
            source=self.state_space,
            target=self.state_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-primal-jacobian",
                    "compilation": self.compilation_id,
                }
            ),
        )
        adjoint_operator = adjoint(primal_jacobian)
        return solve(
            LinearSystem(adjoint_operator),
            self.state_space.validate(cotangent),
            policy=linear_policy,
        )

    def affine_operator(self, args: object = None, /):
        field_index = self.field_index
        dof_map = self.discretization.dof_maps[field_index]
        context = self._execution_context(args)
        mass_operator, stiffness_operator = self.discretization.assemble_field_operators(
            self.form.field_name,
            context.runtime,
        )
        coefficients = None
        relation = None
        for term in self.form.terms:
            if term.domain is not None or term.rules:
                raise ValueError(
                    "Sparse affine lowering currently requires default domains and rules."
                )
            if isinstance(term, DiffusionTerm):
                if not term.diffusivity.constant or term.diffusivity.value.shape != ():
                    raise ValueError(
                        "Sparse affine diffusion requires a scalar constant."
                    )
                operator = stiffness_operator
                term_values = term.diffusivity.value * operator.coefficients
            elif isinstance(term, MassTerm):
                if not term.coefficient.constant or term.coefficient.value.shape != ():
                    raise ValueError("Sparse affine mass requires a scalar constant.")
                operator = mass_operator
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

    def to_scipy_csr(self, args: object = None, /):
        """Materialize the constrained affine operator as a host SciPy CSR matrix."""
        import scipy.sparse as sp

        system, _ = self.linear_system(args)
        dense = np.asarray(system.operator.materialize())
        return sp.csr_array(dense)

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
        selected_plan = plan_sparse_assembly(operator, policy) if plan is None else plan
        return prepare_sparse_assembly(selected_plan, operator)

    def linear_system(
        self,
        args: object = None,
        /,
        *,
        nullspace_policy: NullspacePolicy | None = None,
    ) -> tuple[LinearSystem, Array]:
        raw_operator = self.affine_operator(args)
        primal_operator = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_operator.mv(state)),
            source=self.state_space,
            target=self.state_space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
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
        return (
            LinearSystem(
                primal_operator,
                nullspace_policy=nullspace_policy,
            ),
            right_hand_side,
        )

    def _mass_operators(
        self,
        context: FiniteElementExecutionContext,
        coefficient: Array,
        /,
    ) -> tuple[SparseCoordinateOperator, object]:
        assembled, _ = self.discretization.assemble_field_operators(
            self.form.field_name,
            context.runtime,
        )
        full_mass = SparseCoordinateOperator(
            assembled.relation,
            coefficient * assembled.coefficients,
            source=self.full_space,
            target=DualSpace(self.full_space),
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-mass-operator",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )
        if self.constraint is None:
            return full_mass, full_mass
        constraint_map = self.constraint.constraint_map
        reduced_mass = FunctionLinearOperator(
            lambda reduced: constraint_map.pullback_dual(
                full_mass.mv(constraint_map.homogeneous_correction(reduced))
            ),
            source=constraint_map.reduced_space,
            target=DualSpace(constraint_map.reduced_space),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constrained-finite-element-mass",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )
        return full_mass, reduced_mass

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

        def execution_context(time, args):
            base = self._execution_context(args)
            return FiniteElementExecutionContext(
                base.runtime,
                time=time,
                lift=base.lift,
                lift_rate=base.lift_rate,
                lift_acceleration=base.lift_acceleration,
                user_args=base.user_args,
            )

        def mass_matrix(time, state, args):
            context = execution_context(time, args)
            _, reduced_mass = self._mass_operators(context, coefficient_)
            return reduced_mass

        def vector_field(time, state, args):
            context = execution_context(time, args)
            full_mass, _ = self._mass_operators(context, coefficient_)
            residual = self.residual(state, context)
            if self.constraint is not None and context.lift_rate is not None:
                lift_rate = self.full_space.validate(context.lift_rate)
                residual = residual + self.constraint.constraint_map.pullback_dual(
                    full_mass.mv(lift_rate)
                )
            return -residual

        return DifferentialAlgebraicSystem.from_mass_matrix(
            mass_matrix,
            vector_field,
            state_shape=structure.shape,
            structure=DAEStructure(("differential",), component_axis=None),
            system_id=identifier,
        )

    def as_second_order_system(
        self,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
        damping_coefficient: ArrayLike = 0.0,
        system_id: str | None = None,
    ) -> SecondOrderDifferentialSystem:
        mass_ = jnp.asarray(mass_coefficient)
        damping_ = jnp.asarray(damping_coefficient)
        if mass_.shape != () or damping_.shape != ():
            raise ValueError("Second-order mass and damping coefficients must be scalar.")
        structure = self.state_space.structure()
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-element-second-order-system",
                    "compilation": self.compilation_id,
                }
            )
            if system_id is None
            else str(system_id)
        )

        def residual(time, configuration, velocity, acceleration, args):
            base = self._execution_context(args)
            context = FiniteElementExecutionContext(
                base.runtime,
                time=time,
                lift=base.lift,
                lift_rate=base.lift_rate,
                lift_acceleration=base.lift_acceleration,
                user_args=base.user_args,
            )
            full_mass, reduced_mass = self._mass_operators(context, mass_)
            value = (
                reduced_mass.mv(acceleration)
                + damping_ * reduced_mass.mv(velocity)
                + self.residual(configuration, context)
            )
            if self.constraint is not None and context.lift_rate is not None:
                value = value + damping_ * self.constraint.constraint_map.pullback_dual(
                    full_mass.mv(self.full_space.validate(context.lift_rate))
                )
            if self.constraint is not None and context.lift_acceleration is not None:
                value = value + self.constraint.constraint_map.pullback_dual(
                    full_mass.mv(self.full_space.validate(context.lift_acceleration))
                )
            return self.state_space.inverse_riesz(value)

        return SecondOrderDifferentialSystem(
            residual,
            state_shape=structure.shape,
            system_id=identifier,
        )

    def as_generalized_eigenproblem(
        self,
        args: object = None,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
    ) -> GeneralizedEigenproblem:
        context = self._execution_context(args)
        raw_stiffness = self.affine_operator(context)
        _, raw_mass = self._mass_operators(context, jnp.asarray(mass_coefficient))
        properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        stiffness = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_stiffness.mv(state)),
            source=self.state_space,
            target=self.state_space,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-eigen-stiffness",
                    "compilation": self.compilation_id,
                }
            ),
        )
        mass = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_mass.mv(state)),
            source=self.state_space,
            target=self.state_space,
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
                {
                    "kind": "finite-element-eigen-mass",
                    "compilation": self.compilation_id,
                }
            ),
        )
        return GeneralizedEigenproblem(stiffness, mass)


class CompiledMixedFiniteElementProblem(StrictModule, NonTrainableState):
    """Ordered independent-field block compilation with native BlockSpace semantics."""

    form: WeakForm
    discretization: FiniteElementDiscretization
    subproblems: tuple[CompiledFiniteElementProblem, ...]
    state_space: BlockSpace
    residual_space: BlockSpace
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        form: WeakForm,
        discretization: FiniteElementDiscretization,
        subproblems: Sequence[CompiledFiniteElementProblem],
        /,
    ):
        problems = tuple(subproblems)
        if len(problems) != len(form.field_names):
            raise ValueError("Mixed compilation requires one subproblem per field.")
        if tuple(problem.form.field_name for problem in problems) != form.field_names:
            raise ValueError("Mixed subproblem order must match weak-form fields.")
        self.form = form
        self.discretization = discretization
        self.subproblems = problems
        self.state_space = BlockSpace(
            tuple(problem.state_space for problem in problems),
            names=form.field_names,
        )
        self.residual_space = BlockSpace(
            tuple(problem.residual_space for problem in problems),
            names=form.field_names,
        )
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-mixed-finite-element-problem",
                "form": form.form_id,
                "subproblems": [problem.compilation_id for problem in problems],
            }
        )

    def expand(self, state, args: object = None, /):
        values = self.state_space.validate(state)
        return tuple(
            problem.expand(value, args)
            for problem, value in zip(self.subproblems, values, strict=True)
        )

    def _coupled_residual(
        self,
        term: CellResidualTerm,
        full_states: tuple,
        context: FiniteElementExecutionContext,
        /,
    ) -> Array:
        output_index = self.form.field_names.index(term.field_name)
        output_problem = self.subproblems[output_index]
        output_dof_map = self.discretization.dof_maps[
            self.discretization._field_index(term.field_name)
        ]
        full_output_space = output_problem.full_space
        result = jnp.zeros(
            full_output_space.structure().shape,
            dtype=full_output_space.structure().dtype,
        )
        domain = _term_domain(term, self.discretization)
        cell_offset = 0
        for block_index, block in enumerate(self.discretization.mesh.blocks):
            block_cells = jnp.arange(
                cell_offset,
                cell_offset + block.cell_count,
                dtype=jnp.int32,
            )
            cell_offset += block.cell_count
            selected = jnp.isin(block_cells, domain.entity_indices)
            rule = _term_rule(term, block.name, block.cell_kind)
            data = _reference_rule_data(rule)
            output_geometry = self.discretization.evaluate_block_geometry(
                term.field_name,
                block_index,
                context.runtime.coordinates,
                data.points,
                data.weights,
            )
            input_values = []
            input_gradients = []
            for input_field in term.input_fields:
                input_form_index = self.form.field_names.index(input_field)
                input_field_index = self.discretization._field_index(input_field)
                input_geometry = self.discretization.evaluate_block_geometry(
                    input_field,
                    block_index,
                    context.runtime.coordinates,
                    data.points,
                    data.weights,
                )
                input_dofs = self.discretization.dof_maps[input_field_index].cell_dofs[
                    block_index
                ]
                local_input = full_states[input_form_index][input_dofs]
                input_values.append(
                    oe.contract(
                        "qi,ci...->cq...",
                        input_geometry.basis_values,
                        local_input,
                    )
                )
                input_gradients.append(
                    oe.contract(
                        "cqid,ci...->cqd...",
                        input_geometry.physical_gradients,
                        local_input,
                    )
                )
            output_dofs = output_dof_map.cell_dofs[block_index]
            local = jnp.asarray(
                term.kernel(
                    tuple(input_values),
                    tuple(input_gradients),
                    output_geometry.physical_points,
                    output_geometry.physical_weights * selected[:, None],
                    output_geometry.basis_values,
                    output_geometry.physical_gradients,
                    context,
                )
            )
            expected = full_states[output_index][output_dofs].shape
            if local.shape != expected:
                raise ValueError(
                    "Coupled residual kernel returned an incompatible local shape."
                )
            result = result.at[output_dofs].add(local)
        if output_problem.constraint is None:
            return DualSpace(full_output_space).validate(result)
        return output_problem.constraint.constraint_map.pullback_dual(result)

    def residual(self, state, args: object = None, /):
        values = self.state_space.validate(state)
        full_states = self.expand(values, args)
        context = self.subproblems[0]._execution_context(args)
        residuals = [
            problem.residual(value, args)
            for problem, value in zip(self.subproblems, values, strict=True)
        ]
        for term in self.form.terms:
            if isinstance(term, CellResidualTerm) and any(
                input_field != term.field_name for input_field in term.input_fields
            ):
                output_index = self.form.field_names.index(term.field_name)
                residuals[output_index] = residuals[
                    output_index
                ] + self._coupled_residual(term, full_states, context)
        return self.residual_space.validate(tuple(residuals))

    def as_nonlinear_problem(self) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            lambda state, args: self.residual(state, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
            problem_id=self.compilation_id,
        )

    def linear_system(self, args: object = None, /) -> tuple[LinearSystem, tuple]:
        if any(
            isinstance(term, CellResidualTerm)
            and any(input_field != term.field_name for input_field in term.input_fields)
            for term in self.form.terms
        ):
            raise ValueError(
                "Coupled CellResidualTerm forms require nonlinear operator preparation."
            )
        systems_and_rhs = tuple(
            problem.linear_system(args) for problem in self.subproblems
        )
        operators = tuple(system.operator for system, _ in systems_and_rhs)
        right_hand_side = tuple(rhs for _, rhs in systems_and_rhs)
        operator = FunctionLinearOperator(
            lambda state: tuple(
                block.mv(value) for block, value in zip(operators, state, strict=True)
            ),
            source=self.state_space,
            target=self.state_space,
            transpose_action=lambda state: tuple(
                block.transpose_mv(value)
                for block, value in zip(operators, state, strict=True)
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "mixed-finite-element-linear-operator",
                    "compilation": self.compilation_id,
                    "blocks": [block.operator_id for block in operators],
                }
            ),
        )
        return LinearSystem(operator), right_hand_side


def _coefficient_values(
    coefficient_: _ResolvedCoefficient,
    points: Array,
    context: FiniteElementExecutionContext,
    /,
    *,
    value_shape: tuple[int, ...] = (),
    entity_indices: ArrayLike | None = None,
) -> Array:
    values = coefficient_.evaluate(
        points,
        context,
        entity_indices=entity_indices,
    )
    point_shape = points.shape[:-1]
    expected = point_shape + value_shape
    if values.shape == ():
        return jnp.broadcast_to(values, expected)
    if values.shape == point_shape and value_shape:
        return jnp.broadcast_to(
            values.reshape(point_shape + (1,) * len(value_shape)),
            expected,
        )
    if values.shape != expected:
        raise ValueError(
            f"Finite-element coefficient must return shape {expected}; "
            f"got {values.shape}."
        )
    return values


def _scatter_local(
    residual: Array,
    dofs: Array,
    local: Array,
    accumulation: str,
    /,
) -> Array:
    if accumulation == "fast":
        return residual.at[dofs].add(local)
    flat_dofs = dofs.reshape((-1,))
    component_shape = residual.shape[1:]
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    flat_local = local.reshape((flat_dofs.size, component_count))
    if accumulation == "deterministic":
        grouped = jax.ops.segment_sum(
            flat_local,
            flat_dofs,
            residual.shape[0],
            indices_are_sorted=False,
            unique_indices=False,
        )
    elif accumulation == "compensated":
        grouped_components = []
        for component in range(component_count):
            grouped_components.append(
                jnp.stack(
                    tuple(
                        compensated_sum(
                            jnp.where(
                                flat_dofs == index,
                                flat_local[:, component],
                                jnp.zeros((), dtype=flat_local.dtype),
                            )
                        )
                        for index in range(residual.shape[0])
                    )
                )
            )
        grouped = jnp.stack(tuple(grouped_components), axis=-1)
    else:
        raise ValueError("Unknown finite-element accumulation policy.")
    return residual + grouped.reshape(residual.shape)


def _full_residual(
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    work_blocks: tuple[_FiniteElementWorkBlock, ...],
    state: Array,
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    field_index = discretization._field_index(form.field_name)
    residual = jnp.zeros_like(state)
    for term in form.terms:
        domain = _term_domain(term, discretization)
        if isinstance(term, BoundaryLoadTerm):
            residual = residual - _boundary_load(
                discretization,
                field_index,
                term,
                domain,
                context,
            )
            continue
        if isinstance(term, InteriorFacetTerm):
            residual = residual + _interior_facet_residual(
                discretization,
                field_index,
                state,
                term,
                domain,
                context,
                accumulation,
            )
            continue
        for work in work_blocks:
            block = discretization.mesh.blocks[work.block_index]
            rule = _term_rule(term, work.block_name, block.cell_kind)
            rule_data = _reference_rule_data(rule)
            geometry = discretization.evaluate_block_geometry(
                form.field_name,
                work.block_index,
                context.runtime.coordinates,
                rule_data.points,
                rule_data.weights,
            )
            work_cells = jnp.asarray(work.cell_indices, dtype=jnp.int32)
            selected = jnp.isin(work_cells, domain.entity_indices)
            dofs = work.cell_dofs
            local_state = state[dofs]
            orientation = discretization.dof_maps[field_index].orientations[
                work.block_index
            ]
            local_state = local_state * orientation.reshape(
                orientation.shape + (1,) * (local_state.ndim - orientation.ndim)
            )
            physical_points = geometry.physical_points
            physical_gradients = geometry.physical_gradients
            physical_weights = geometry.physical_weights * selected[:, None]
            basis_values = geometry.basis_values
            if isinstance(term, DiffusionTerm):
                field_gradient = oe.contract(
                    "cqid,ci...->cqd...",
                    physical_gradients,
                    local_state,
                )
                values = _coefficient_values(
                    term.diffusivity,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                )
                local = oe.contract(
                    "cq,cq,cqid,cqd...->ci...",
                    physical_weights,
                    values,
                    physical_gradients,
                    field_gradient,
                )
            elif isinstance(term, MassTerm):
                field_value = oe.contract(
                    "qi,ci...->cq...",
                    basis_values,
                    local_state,
                )
                values = _coefficient_values(
                    term.coefficient,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                )
                local = oe.contract(
                    "cq,cq,qi,cq...->ci...",
                    physical_weights,
                    values,
                    basis_values,
                    field_value,
                )
            elif isinstance(term, SourceTerm):
                values = _coefficient_values(
                    term.source,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                    value_shape=state.shape[1:],
                )
                local = -oe.contract(
                    "cq,cq...,qi->ci...",
                    physical_weights,
                    values,
                    basis_values,
                )
            elif isinstance(term, CellResidualTerm):
                if any(
                    input_field != form.field_name for input_field in term.input_fields
                ):
                    raise ValueError(
                        "Cross-field CellResidualTerm requires mixed compilation."
                    )
                if basis_values.ndim == 2:
                    field_value = oe.contract(
                        "qi,ci...->cq...",
                        basis_values,
                        local_state,
                    )
                    field_gradient = oe.contract(
                        "cqid,ci...->cqd...",
                        physical_gradients,
                        local_state,
                    )
                else:
                    field_value = oe.contract(
                        "cqiv,ci->cqv",
                        basis_values,
                        local_state,
                    )
                    field_gradient = oe.contract(
                        "cqivd,ci->cqvd",
                        physical_gradients,
                        local_state,
                    )
                local = jnp.asarray(
                    term.kernel(
                        (field_value,),
                        (field_gradient,),
                        physical_points,
                        physical_weights,
                        basis_values,
                        physical_gradients,
                        context,
                    )
                )
                if local.shape != local_state.shape:
                    raise ValueError(
                        "Cell residual kernel must return one local test residual "
                        "per selected cell and output-field DOF."
                    )
            elif isinstance(term, CellEnergyTerm):

                def energy(
                    local_coefficients,
                    basis_values_=basis_values,
                    physical_gradients_=physical_gradients,
                    physical_points_=physical_points,
                    physical_weights_=physical_weights,
                    term_=term,
                    context_=context,
                ):
                    if basis_values_.ndim == 2:
                        values_ = oe.contract(
                            "qi,ci...->cq...",
                            basis_values_,
                            local_coefficients,
                        )
                        gradients_ = oe.contract(
                            "cqid,ci...->cqd...",
                            physical_gradients_,
                            local_coefficients,
                        )
                    else:
                        values_ = oe.contract(
                            "cqiv,ci->cqv",
                            basis_values_,
                            local_coefficients,
                        )
                        gradients_ = oe.contract(
                            "cqivd,ci->cqvd",
                            physical_gradients_,
                            local_coefficients,
                        )
                    density = jnp.asarray(
                        term_.density(
                            values_,
                            gradients_,
                            physical_points_,
                            context_,
                        )
                    )
                    if density.shape != physical_weights_.shape:
                        raise ValueError(
                            "Cell energy density must return one scalar per "
                            "selected quadrature point."
                        )
                    return jnp.sum(density * physical_weights_)

                local = jax.grad(energy)(local_state)
            elif isinstance(term, CellBilinearTerm):
                matrix = jnp.asarray(
                    term.kernel(
                        physical_points,
                        physical_weights,
                        basis_values,
                        physical_gradients,
                        context,
                    )
                )
                expected_prefix = (
                    local_state.shape[0],
                    local_state.shape[1],
                    local_state.shape[1],
                )
                if matrix.shape != expected_prefix:
                    raise ValueError(
                        "Cell bilinear kernel must return shape "
                        "(cells, local_dofs, local_dofs)."
                    )
                local = oe.contract(
                    "cij,cj...->ci...",
                    matrix,
                    local_state,
                )
            else:
                raise TypeError("Unsupported finite-element term.")
            local = local * orientation.reshape(
                orientation.shape + (1,) * (local.ndim - orientation.ndim)
            )
            residual = _scatter_local(residual, dofs, local, accumulation)
    return DualSpace(discretization.field_spaces[field_index].vector_space).validate(
        residual
    )


def _interior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    term: InteriorFacetTerm,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    facets = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owners = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    neighbours = jnp.asarray(domain.neighbour_cells, dtype=jnp.int32)
    dof_map = discretization.dof_maps[field_index]
    result = jnp.zeros_like(state)
    if isinstance(connectivity, PolygonalConnectivity):
        rule = _interval_rule()
        if term.rules:
            rule = term.rules[0][1]
        data = _reference_rule_data(rule)
        if data.cell != "interval":
            raise ValueError("Polygon interior facets require an interval rule.")
        edge_vertices = jnp.asarray(connectivity.edges)[facets]
        edge_points = context.runtime.coordinates[edge_vertices]
        parameter = data.points[:, 0]
        physical_points = (1.0 - parameter)[None, :, None] * edge_points[
            :, None, 0, :
        ] + parameter[None, :, None] * edge_points[:, None, 1, :]
        tangent = edge_points[:, 1] - edge_points[:, 0]
        measure = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
        normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        normal = normal / measure[:, None]
        cell_centers = jnp.concatenate(
            tuple(
                jnp.mean(
                    context.runtime.coordinates[block.vertices],
                    axis=1,
                )
                for block in discretization.mesh.blocks
            ),
            axis=0,
        )
        owner_centers = cell_centers[owners]
        midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
        outward = jnp.sum(normal * (midpoint - owner_centers), axis=-1)
        normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
        weights = measure[:, None] * data.weights[None, :]
        if dof_map.association == "cell":
            plus_dofs = jnp.asarray(owners)[:, None]
            minus_dofs = jnp.asarray(neighbours)[:, None]
            trace_basis = jnp.ones((data.points.shape[0], 1))
        elif dof_map.association == "edge":
            plus_dofs = jnp.asarray(facets)[:, None]
            minus_dofs = plus_dofs
            trace_basis = jnp.ones((data.points.shape[0], 1))
        elif dof_map.association == "vertex_edge":
            edge_dofs = int(discretization.mesh.coordinates.shape[0]) + jnp.asarray(
                facets
            )
            plus_dofs = jnp.concatenate((edge_vertices, edge_dofs[:, None]), axis=1)
            minus_dofs = plus_dofs
            trace_basis = jnp.stack(
                (
                    (1.0 - parameter) * (1.0 - 2.0 * parameter),
                    parameter * (2.0 * parameter - 1.0),
                    4.0 * parameter * (1.0 - parameter),
                ),
                axis=-1,
            )
        else:
            plus_dofs = edge_vertices
            minus_dofs = edge_vertices
            trace_basis = jnp.stack((1.0 - parameter, parameter), axis=-1)
    elif isinstance(connectivity, TetrahedralConnectivity):
        data = _reference_rule_data(_triangle_rule())
        face_vertices = jnp.asarray(connectivity.faces)[facets]
        face_points = context.runtime.coordinates[face_vertices]
        first = data.points[:, 0]
        second = data.points[:, 1]
        trace_basis = jnp.stack((1.0 - first - second, first, second), axis=-1)
        physical_points = oe.contract("qi,eid->eqd", trace_basis, face_points)
        cross = jnp.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        )
        measure = jnp.sqrt(jnp.sum(cross**2, axis=-1))
        normal = cross / measure[:, None]
        weights = measure[:, None] * data.weights[None, :]
        plus_dofs = face_vertices
        minus_dofs = face_vertices
    else:
        raise TypeError("Unsupported interior-facet connectivity.")
    plus_local = state[plus_dofs]
    minus_local = state[minus_dofs]
    plus_value = oe.contract("qi,ei...->eq...", trace_basis, plus_local)
    minus_value = oe.contract("qi,ei...->eq...", trace_basis, minus_local)
    plus_flux, minus_flux = term.kernel(
        plus_value,
        minus_value,
        physical_points,
        weights,
        normal,
        context,
    )
    plus_flux = jnp.asarray(plus_flux)
    minus_flux = jnp.asarray(minus_flux)
    expected = plus_value.shape
    if plus_flux.shape != expected or minus_flux.shape != expected:
        raise ValueError(
            "Interior facet kernel must return plus/minus quadrature flux densities."
        )
    plus_residual = oe.contract(
        "eq,eq...,qi->ei...",
        weights,
        plus_flux,
        trace_basis,
    )
    minus_residual = oe.contract(
        "eq,eq...,qi->ei...",
        weights,
        minus_flux,
        trace_basis,
    )
    result = _scatter_local(result, plus_dofs, plus_residual, accumulation)
    return _scatter_local(result, minus_dofs, minus_residual, accumulation)


def _boundary_load(
    discretization: FiniteElementDiscretization,
    field_index: int,
    term: BoundaryLoadTerm,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    selected = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owner_cells = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    field_shape = discretization.field_spaces[field_index].vector_space.structure().shape
    component_shape = field_shape[1:]
    result = jnp.zeros(field_shape, dtype=context.runtime.coordinates.dtype)
    rule_bindings = dict(term.rules)
    cell_start = 0
    for block in discretization.mesh.blocks:
        cell_end = cell_start + block.cell_count
        active = (owner_cells >= cell_start) & (owner_cells < cell_end)
        cell_start = cell_end
        facet_indices = selected
        rule = rule_bindings.get(
            block.name,
            _interval_rule()
            if isinstance(connectivity, PolygonalConnectivity)
            else _triangle_rule(),
        )
        data = _reference_rule_data(rule)
        if isinstance(connectivity, PolygonalConnectivity):
            if data.cell != "interval":
                raise ValueError("Polygon boundary terms require an interval rule.")
            edge_vertices = jnp.asarray(connectivity.edges)[facet_indices]
            edge_points = context.runtime.coordinates[edge_vertices]
            parameter = data.points[:, 0]
            physical_points = (1.0 - parameter)[None, :, None] * edge_points[
                :, None, 0, :
            ] + parameter[None, :, None] * edge_points[:, None, 1, :]
            measure = jnp.sqrt(
                jnp.sum((edge_points[:, 1] - edge_points[:, 0]) ** 2, axis=-1)
            )
            physical_weights = measure[:, None] * data.weights[None, :]
            if (
                discretization.dof_maps[field_index].global_dof_count
                > discretization.mesh.coordinates.shape[0]
            ):
                basis = jnp.stack(
                    (
                        (1.0 - parameter) * (1.0 - 2.0 * parameter),
                        parameter * (2.0 * parameter - 1.0),
                        4.0 * parameter * (1.0 - parameter),
                    ),
                    axis=-1,
                )
                edge_dofs = int(discretization.mesh.coordinates.shape[0]) + jnp.asarray(
                    facet_indices
                )
                dofs = jnp.concatenate((edge_vertices, edge_dofs[:, None]), axis=1)
            else:
                basis = jnp.stack((1.0 - parameter, parameter), axis=-1)
                dofs = edge_vertices
        elif isinstance(connectivity, TetrahedralConnectivity):
            if data.cell != "triangle":
                raise ValueError("Tetrahedron boundary terms require a triangle rule.")
            face_vertices = jnp.asarray(connectivity.faces)[facet_indices]
            face_points = context.runtime.coordinates[face_vertices]
            first = data.points[:, 0]
            second = data.points[:, 1]
            basis = jnp.stack((1.0 - first - second, first, second), axis=-1)
            physical_points = oe.contract("qi,eid->eqd", basis, face_points)
            cross = jnp.cross(
                face_points[:, 1] - face_points[:, 0],
                face_points[:, 2] - face_points[:, 0],
            )
            measure_factor = jnp.sqrt(jnp.sum(cross**2, axis=-1))
            physical_weights = measure_factor[:, None] * data.weights[None, :]
            dofs = face_vertices
        else:
            raise TypeError("Unsupported finite-element boundary connectivity.")
        physical_weights = physical_weights * active[:, None]
        values = _coefficient_values(
            term.load,
            physical_points,
            context,
            entity_indices=facet_indices,
            value_shape=component_shape,
        )
        local = oe.contract(
            "eq,eq...,qi->ei...",
            physical_weights,
            values,
            basis,
        )
        result = result.at[dofs].add(local)
    return result


def compile_finite_element_problem(
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    /,
    *,
    constraint: FiniteElementDirichletConstraint | None = None,
    dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
    constraints: Mapping[str, FiniteElementDirichletConstraint] | None = None,
    dirichlet_values_by_field: Mapping[str, ArrayLike | Callable[[Array], ArrayLike]]
    | None = None,
    execution_policy: FiniteElementExecutionPolicy | None = None,
) -> CompiledFiniteElementProblem | CompiledMixedFiniteElementProblem:
    if len(form.field_names) == 1:
        if constraints is not None or dirichlet_values_by_field is not None:
            field = form.field_names[0]
            resolved_constraints = {} if constraints is None else dict(constraints)
            resolved_values = (
                {}
                if dirichlet_values_by_field is None
                else dict(dirichlet_values_by_field)
            )
            constraint = resolved_constraints.get(field, constraint)
            dirichlet_values = resolved_values.get(field, dirichlet_values)
        return CompiledFiniteElementProblem(
            form,
            discretization,
            constraint=constraint,
            dirichlet_values=dirichlet_values,
            execution_policy=execution_policy,
        )
    if constraint is not None or dirichlet_values is not None:
        raise ValueError(
            "Mixed forms require field-keyed constraints and Dirichlet values."
        )
    resolved_constraints = {} if constraints is None else dict(constraints)
    resolved_values = (
        {} if dirichlet_values_by_field is None else dict(dirichlet_values_by_field)
    )
    unknown_constraints = set(resolved_constraints) - set(form.field_names)
    unknown_values = set(resolved_values) - set(form.field_names)
    if unknown_constraints or unknown_values:
        raise ValueError("Mixed constraint/value mappings contain unknown fields.")
    subproblems = []
    for field in form.field_names:
        field_terms = tuple(
            term
            for term in form.terms
            if term.field_name == field
            and not (
                isinstance(term, CellResidualTerm)
                and any(input_field != field for input_field in term.input_fields)
            )
        )
        if not field_terms:
            field_terms = (
                SourceTerm(
                    field,
                    0.0,
                    term_id=f"__mixed_zero__:{field}",
                ),
            )
        subform = WeakForm(
            f"{form.form_id}:{field}",
            field,
            field_terms,
        )
        subproblems.append(
            CompiledFiniteElementProblem(
                subform,
                discretization,
                constraint=resolved_constraints.get(field),
                dirichlet_values=resolved_values.get(field),
                execution_policy=execution_policy,
            )
        )
    return CompiledMixedFiniteElementProblem(form, discretization, subproblems)


__all__ = [
    "BoundaryLoadTerm",
    "CellBilinearTerm",
    "CellEnergyTerm",
    "CellResidualTerm",
    "CompiledFiniteElementProblem",
    "CompiledMixedFiniteElementProblem",
    "DiffusionTerm",
    "FiniteElementFunctional",
    "FiniteElementExecutionContext",
    "FiniteElementExecutionPolicy",
    "InteriorFacetTerm",
    "MassTerm",
    "SourceTerm",
    "WeakForm",
    "coefficient",
    "compile_finite_element_problem",
]
