#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization._local_variational import (
    AbstractPreparedLocalDiscretization,
)
from ..discretization.fem import (
    finite_element_hp_constraint,
    finite_element_hp_domains,
    FiniteElementDirichletConstraint,
    FiniteElementDiscretization,
    FiniteElementHPEpoch,
    FiniteElementHPTraceConstraintPlan,
    FiniteElementLinearConstraint,
    IntegrationDomain,
)
from ..dynamics import (
    DAEStructure,
    DifferentialAlgebraicSystem,
    SecondOrderDifferentialSystem,
)
from ..linalg import (
    AbstractLinearOperator,
    adjoint,
    assemble_diagonal,
    BlockLinearOperator,
    BlockSpace,
    ConstraintMap,
    DualSpace,
    FunctionLinearOperator,
    IdentityLinearOperator,
    KernelCertificate,
    LinearSolvePolicy,
    LinearSubspace,
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
from ..optim import MinimizationProblem
from ..sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ..variational import (
    Functional,
    FunctionalEvaluation,
    LocalIntegralTerm,
)
from ._variational import (
    _normalize_rules,
    _rule_id,
    BoundaryLoadAction,
    coefficient,
    DiffusionAction,
    IntegrationRule as ReferenceRule,
    MassAction,
    SourceAction,
    VariationalCoefficient,
)


if TYPE_CHECKING:
    from .fem._ir import LocalActionIR
    from .fem._kernels import KernelTable
    from .fem._worksets import WorksetProgram


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


def _hexahedron_rule():
    from ..integration import ReferenceHexahedronRule

    return ReferenceHexahedronRule()


def _default_rule(cell_kind: str, /) -> ReferenceRule:
    if cell_kind == "triangle":
        return _triangle_rule()
    if cell_kind == "quadrilateral":
        return _quadrilateral_rule()
    if cell_kind == "tetrahedron":
        return _tetrahedron_rule()
    if cell_kind == "hexahedron":
        return _hexahedron_rule()
    raise ValueError(f"No finite-element rule exists for cell kind {cell_kind!r}.")


_UNIT_MASS_COEFFICIENT = coefficient(np.asarray(1.0))


class FiniteElementExecutionContext(StrictModule, NonTrainableState):
    """Dynamic geometry, time, lift, and user arguments for FE execution."""

    runtime: object
    time: Array
    lift: object
    lift_rate: object
    lift_acceleration: object
    metric_data: object
    user_args: object

    def __init__(
        self,
        runtime: object,
        /,
        *,
        time: ArrayLike = 0.0,
        lift: object = None,
        lift_rate: object = None,
        lift_acceleration: object = None,
        metric_data: object = None,
        user_args: object = None,
    ):
        self.runtime = runtime
        self.time = jnp.asarray(time)
        self.lift = None if lift is None else jax.tree.map(jnp.asarray, lift)
        self.lift_rate = (
            None if lift_rate is None else jax.tree.map(jnp.asarray, lift_rate)
        )
        self.lift_acceleration = (
            None
            if lift_acceleration is None
            else jax.tree.map(jnp.asarray, lift_acceleration)
        )
        self.metric_data = metric_data
        self.user_args = user_args


class FiniteElementExecutionPolicy(StrictModule, NonTrainableState):
    """Independent global realization, local-kernel, and reduction policy."""

    realization: str = eqx.field(static=True)
    local_kernel: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        realization: str = "sparse",
        local_kernel: str = "auto",
        accumulation: str = "fast",
    ):
        realization_ = str(realization)
        local_kernel_ = str(local_kernel)
        accumulation_ = str(accumulation)
        if realization_ not in ("matrix_free", "sparse"):
            raise ValueError("Unknown finite-element operator realization.")
        if local_kernel_ not in (
            "auto",
            "dense",
            "partial",
            "sum_factorized",
            "collocated",
        ):
            raise ValueError("Unknown finite-element local-kernel strategy.")
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown finite-element accumulation policy.")
        if realization_ == "sparse" and local_kernel_ in (
            "partial",
            "sum_factorized",
            "collocated",
        ):
            raise ValueError(
                "Sparse realization requires auto or dense local-kernel strategy."
            )
        self.realization = realization_
        self.local_kernel = local_kernel_
        self.accumulation = accumulation_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-execution-policy",
                "realization": realization_,
                "local_kernel": local_kernel_,
                "accumulation": accumulation_,
            }
        )


class CellResidualAction(StrictModule, NonTrainableState):
    """User-defined cell-local residual with explicit field dependencies."""

    field_name: str = eqx.field(static=True)
    input_fields: tuple[str, ...] = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        input_fields: Sequence[str],
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        output = str(field_name)
        inputs = tuple(str(value) for value in input_fields)
        identifier = str(action_id)
        if not output or not inputs or any(not value for value in inputs):
            raise ValueError("Residual output/input field names must be non-empty.")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Residual input field names must be unique.")
        if not callable(kernel) or not identifier:
            raise ValueError("Residual kernel and action_id are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellResidualAction requires a cell integration domain.")
        self.field_name = output
        self.input_fields = inputs
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class PairwiseVolumeFluxAction(StrictModule, NonTrainableState):
    """Collocated tensor-cell flux differencing from a symmetric two-point flux."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError(
                "Pairwise volume-flux field, kernel, and action_id are required."
            )
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("PairwiseVolumeFluxAction requires a cell domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class InteriorFacetAction(StrictModule, NonTrainableState):
    """Two-sided numerical flux density over interior facets."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError("Interior facet field, kernel, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "interior_facet"
        ):
            raise ValueError("InteriorFacetAction requires an interior-facet domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class ExteriorFacetAction(StrictModule, NonTrainableState):
    """One-sided state-dependent numerical flux over exterior facets."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError("Exterior facet field, kernel, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet"
        ):
            raise ValueError("ExteriorFacetAction requires an exterior-facet domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


SIPGBoundaryKind: TypeAlias = Literal["dirichlet", "neumann", "robin"]


class SIPGPenaltyPolicy(StrictModule, NonTrainableState):
    """Explicit symmetric interior-penalty scaling policy."""

    factor: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, factor: float, /):
        factor_ = float(factor)
        if not np.isfinite(factor_) or factor_ <= 0.0:
            raise ValueError("SIPG penalty factor must be positive and finite.")
        self.factor = factor_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "sipg-penalty-policy",
                "factor": factor_,
                "degree_rule": "maximum",
                "height_rule": "harmonic-normal-height",
                "coefficient_rule": "harmonic",
            }
        )

    def evaluate(
        self,
        plus_order: ArrayLike,
        minus_order: ArrayLike,
        plus_height: ArrayLike,
        minus_height: ArrayLike,
        plus_diffusivity: ArrayLike,
        minus_diffusivity: ArrayLike,
        /,
    ) -> Array:
        p_plus = jnp.asarray(plus_order)
        p_minus = jnp.asarray(minus_order)
        h_plus = jnp.asarray(plus_height)
        h_minus = jnp.asarray(minus_height)
        k_plus = jnp.asarray(plus_diffusivity)
        k_minus = jnp.asarray(minus_diffusivity)
        invalid = (
            ~jnp.isfinite(h_plus)
            | ~jnp.isfinite(h_minus)
            | ~jnp.isfinite(k_plus)
            | ~jnp.isfinite(k_minus)
            | (h_plus <= 0.0)
            | (h_minus <= 0.0)
            | (k_plus <= 0.0)
            | (k_minus <= 0.0)
        )
        h_facet = 2.0 * h_plus * h_minus / (h_plus + h_minus)
        k_hat = 2.0 * k_plus * k_minus / (k_plus + k_minus)
        order = jnp.maximum(p_plus, p_minus)
        penalty = self.factor * order**2 * k_hat / h_facet
        return eqx.error_if(
            penalty,
            jnp.any(invalid | ~jnp.isfinite(penalty) | (penalty <= 0.0)),
            "SIPG penalty evidence must be positive and finite.",
        )


class SIPGBoundaryCondition(StrictModule, NonTrainableState):
    """One explicit exterior SIPG boundary declaration."""

    kind: SIPGBoundaryKind = eqx.field(static=True)
    domain: IntegrationDomain
    value: VariationalCoefficient
    robin_coefficient: VariationalCoefficient | None
    penalty_policy: SIPGPenaltyPolicy | None
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SIPGBoundaryKind,
        domain: IntegrationDomain,
        value: VariationalCoefficient | ArrayLike | Callable,
        /,
        *,
        robin_coefficient: VariationalCoefficient | ArrayLike | Callable | None = None,
        penalty_policy: SIPGPenaltyPolicy | None = None,
    ):
        if kind not in ("dirichlet", "neumann", "robin"):
            raise ValueError("Unknown SIPG boundary kind.")
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise ValueError("SIPG boundary conditions require an exterior-facet domain.")
        if penalty_policy is not None and not isinstance(
            penalty_policy, SIPGPenaltyPolicy
        ):
            raise TypeError("penalty_policy must be SIPGPenaltyPolicy or None.")
        value_ = (
            value if isinstance(value, VariationalCoefficient) else coefficient(value)
        )
        robin = (
            robin_coefficient
            if isinstance(robin_coefficient, VariationalCoefficient)
            else (None if robin_coefficient is None else coefficient(robin_coefficient))
        )
        if kind == "robin" and robin is None:
            raise ValueError("Robin SIPG data require a boundary coefficient.")
        if kind != "robin" and robin is not None:
            raise ValueError("Only Robin SIPG data accept a boundary coefficient.")
        if kind != "dirichlet" and penalty_policy is not None:
            raise ValueError("Only Dirichlet SIPG data accept a penalty override.")
        self.kind = kind
        self.domain = domain
        self.value = value_
        self.robin_coefficient = robin
        self.penalty_policy = penalty_policy
        self.condition_id = canonical_fingerprint(
            {
                "kind": "sipg-boundary-condition",
                "boundary_kind": kind,
                "domain": domain.domain_id,
                "value": value_.coefficient_id,
                "robin": None if robin is None else robin.coefficient_id,
                "penalty": (None if penalty_policy is None else penalty_policy.policy_id),
            }
        )


class SIPGFacetAction(StrictModule, NonTrainableState):
    """Executable SIPG interior or exterior facet action."""

    field_name: str = eqx.field(static=True)
    diffusivity: VariationalCoefficient
    penalty_policy: SIPGPenaltyPolicy
    boundary: SIPGBoundaryCondition | None
    domain: IntegrationDomain
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        diffusivity: VariationalCoefficient | ArrayLike | Callable,
        penalty_policy: SIPGPenaltyPolicy,
        domain: IntegrationDomain,
        /,
        *,
        boundary: SIPGBoundaryCondition | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("SIPG field and term IDs must be non-empty.")
        if not isinstance(penalty_policy, SIPGPenaltyPolicy):
            raise TypeError("penalty_policy must be SIPGPenaltyPolicy.")
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be an IntegrationDomain.")
        expected_kind = "interior_facet" if boundary is None else "exterior_facet"
        if domain.kind != expected_kind:
            raise ValueError(f"SIPG facet term requires a {expected_kind} domain.")
        if boundary is not None and (
            not isinstance(boundary, SIPGBoundaryCondition)
            or boundary.domain.domain_id != domain.domain_id
        ):
            raise ValueError("SIPG boundary data must match the facet term domain.")
        self.field_name = field
        self.diffusivity = (
            diffusivity
            if isinstance(diffusivity, VariationalCoefficient)
            else coefficient(diffusivity)
        )
        self.penalty_policy = penalty_policy
        self.boundary = boundary
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class CellEnergyAction(StrictModule, NonTrainableState):
    """Cell-local scalar energy differentiated into one field residual."""

    field_name: str = eqx.field(static=True)
    density: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        density: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not callable(density) or not identifier:
            raise ValueError("Energy field, density, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellEnergyAction requires a cell domain.")
        self.field_name = field
        self.density = density
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class LocalFunctionalAction(StrictModule, NonTrainableState):
    """One local functional term differentiated into all active field residuals."""

    term: LocalIntegralTerm
    field_bindings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    variable_fields: tuple[str, ...] = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        term: LocalIntegralTerm,
        field_bindings: Mapping[str, str],
        variable_fields: Sequence[str],
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        if not isinstance(term, LocalIntegralTerm):
            raise TypeError("term must be a variational.LocalIntegralTerm.")
        bindings = tuple(
            (spec.field_name, str(field_bindings[spec.field_name]))
            for spec in term.fields
        )
        if any(not field for _, field in bindings):
            raise ValueError("Functional field bindings must be non-empty.")
        variables = tuple(str(field) for field in variable_fields)
        bound_fields = {field for _, field in bindings}
        if not variables or any(field not in bound_fields for field in variables):
            raise ValueError(
                "Functional variable fields must be non-empty bound input fields."
            )
        if len(set(variables)) != len(variables):
            raise ValueError("Functional variable fields must be unique.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain)
            or domain.kind not in ("cell", "exterior_facet")
        ):
            raise ValueError(
                "LocalFunctionalAction requires a cell or exterior-facet domain."
            )
        if (
            domain is not None
            and domain.kind == "exterior_facet"
            and any(spec.gradient for spec in term.fields)
        ):
            raise ValueError(
                "Exterior functional terms currently support value jets only."
            )
        if term.normal and (domain is None or domain.kind != "exterior_facet"):
            raise ValueError(
                "Functional normal requests require an exterior-facet domain."
            )
        identifier = str(action_id)
        if not identifier:
            raise ValueError("Functional action_id must be non-empty.")
        self.term = term
        self.field_bindings = bindings
        self.variable_fields = variables
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier

    @property
    def input_fields(self) -> tuple[str, ...]:
        return tuple(field for _, field in self.field_bindings)

    @property
    def output_fields(self) -> tuple[str, ...]:
        return self.variable_fields

    @property
    def semantic_to_field(self) -> dict[str, str]:
        return dict(self.field_bindings)


class CellBilinearAction(StrictModule, NonTrainableState):
    """User-provided cell-local matrix over one field."""

    field_name: str = eqx.field(static=True)
    kernel: Callable
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        kernel: Callable,
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not callable(kernel) or not identifier:
            raise ValueError("Bilinear field, kernel, and term ID are required.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("CellBilinearAction requires a cell domain.")
        self.field_name = field
        self.kernel = kernel
        self.domain = domain
        self.rules = _normalize_rules(rules)
        self.action_id = identifier


class PreparedOperatorAction(StrictModule, NonTrainableState):
    """One prepared global linear action scheduled by a finite-element form."""

    field_name: str = eqx.field(static=True)
    operator: AbstractLinearOperator
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        operator: AbstractLinearOperator,
        /,
        *,
        domain: IntegrationDomain | None = None,
        action_id: str,
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Operator action field and term IDs must be non-empty.")
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("PreparedOperatorAction requires a cell domain.")
        self.field_name = field
        self.operator = operator
        self.domain = domain
        self.rules = ()
        self.action_id = identifier


FiniteElementAction = (
    DiffusionAction
    | MassAction
    | SourceAction
    | BoundaryLoadAction
    | CellResidualAction
    | PairwiseVolumeFluxAction
    | InteriorFacetAction
    | ExteriorFacetAction
    | SIPGFacetAction
    | CellEnergyAction
    | LocalFunctionalAction
    | CellBilinearAction
    | PreparedOperatorAction
)


def _action_output_fields(term: FiniteElementAction, /) -> tuple[str, ...]:
    if isinstance(term, LocalFunctionalAction):
        return term.output_fields
    return (term.field_name,)


def _action_input_fields(term: FiniteElementAction, /) -> tuple[str, ...]:
    if isinstance(term, (CellResidualAction, LocalFunctionalAction)):
        return term.input_fields
    return (term.field_name,)


def _action_payload(term: FiniteElementAction, /) -> dict[str, object]:
    if isinstance(term, DiffusionAction):
        coefficient_id = term.diffusivity.coefficient_id
        kind = "diffusion"
    elif isinstance(term, MassAction):
        coefficient_id = term.coefficient.coefficient_id
        kind = "mass"
    elif isinstance(term, SourceAction):
        coefficient_id = term.source.coefficient_id
        kind = "source"
    elif isinstance(term, BoundaryLoadAction):
        coefficient_id = term.load.coefficient_id
        kind = "boundary-load"
    elif isinstance(term, CellResidualAction):
        coefficient_id = None
        kind = "cell-residual"
    elif isinstance(term, PairwiseVolumeFluxAction):
        coefficient_id = None
        kind = "pairwise-volume-flux"
    elif isinstance(term, InteriorFacetAction):
        coefficient_id = None
        kind = "interior-facet"
    elif isinstance(term, ExteriorFacetAction):
        coefficient_id = None
        kind = "exterior-facet"
    elif isinstance(term, SIPGFacetAction):
        coefficient_id = term.diffusivity.coefficient_id
        kind = "sipg-facet"
    elif isinstance(term, CellEnergyAction):
        coefficient_id = None
        kind = "cell-energy"
    elif isinstance(term, LocalFunctionalAction):
        coefficient_id = term.term.term_id
        kind = "functional"
    elif isinstance(term, CellBilinearAction):
        coefficient_id = None
        kind = "cell-bilinear"
    elif isinstance(term, PreparedOperatorAction):
        coefficient_id = term.operator.operator_id
        kind = "operator-action"
    else:
        raise TypeError("Unsupported finite-element action.")
    return {
        "kind": kind,
        "action_id": term.action_id,
        "output_fields": list(_action_output_fields(term)),
        "coefficient_id": coefficient_id,
        "input_fields": list(_action_input_fields(term)),
        "domain_id": None if term.domain is None else term.domain.domain_id,
        "rules": [[block_name, _rule_id(rule)] for block_name, rule in term.rules],
        "penalty_policy": (
            term.penalty_policy.policy_id if isinstance(term, SIPGFacetAction) else None
        ),
        "boundary": (
            None
            if not isinstance(term, SIPGFacetAction) or term.boundary is None
            else term.boundary.condition_id
        ),
    }


class FiniteElementForm(StrictModule, NonTrainableState):
    form_id: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    actions: tuple[FiniteElementAction, ...]
    declared_properties: OperatorProperties
    auxiliary_evaluator: Callable | None
    auxiliary_id: str | None = eqx.field(static=True)
    functional: Functional | None

    def __init__(
        self,
        form_id: str,
        field_name: str | Sequence[str],
        actions: Sequence[FiniteElementAction],
        /,
        *,
        properties: OperatorProperties | None = None,
        auxiliary_evaluator: Callable | None = None,
        auxiliary_id: str | None = None,
        functional: Functional | None = None,
    ):
        identifier = str(form_id)
        fields = (
            (str(field_name),)
            if isinstance(field_name, str)
            else tuple(str(value) for value in field_name)
        )
        action_values = tuple(actions)
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        if (auxiliary_evaluator is None) != (auxiliary_id is None):
            raise ValueError(
                "Form auxiliary evaluator and explicit identity must be supplied together."
            )
        if auxiliary_evaluator is not None and not callable(auxiliary_evaluator):
            raise TypeError("auxiliary_evaluator must be callable or None.")
        auxiliary_identifier = None if auxiliary_id is None else str(auxiliary_id)
        if auxiliary_identifier == "":
            raise ValueError("auxiliary_id must be non-empty when supplied.")
        if not identifier or not fields or any(not field for field in fields):
            raise ValueError("Finite-element form and field IDs must be non-empty.")
        if len(set(fields)) != len(fields):
            raise ValueError("Finite-element form field names must be unique.")
        if not action_values:
            raise ValueError("FiniteElementForm requires at least one action.")
        if not all(
            isinstance(
                action,
                (
                    DiffusionAction,
                    MassAction,
                    SourceAction,
                    BoundaryLoadAction,
                    CellResidualAction,
                    PairwiseVolumeFluxAction,
                    InteriorFacetAction,
                    ExteriorFacetAction,
                    SIPGFacetAction,
                    CellEnergyAction,
                    LocalFunctionalAction,
                    CellBilinearAction,
                    PreparedOperatorAction,
                ),
            )
            for action in action_values
        ):
            raise TypeError("FiniteElementForm contains an unsupported action type.")
        if any(
            output not in fields
            for action in action_values
            for output in _action_output_fields(action)
        ):
            raise ValueError("Every form action output must be a declared field.")
        if any(
            input_field not in fields
            for action in action_values
            for input_field in _action_input_fields(action)
        ):
            raise ValueError("Every form action input must be a declared field.")
        functional_ = functional
        if functional_ is not None:
            if not isinstance(functional_, Functional):
                raise TypeError("functional must be variational.Functional or None.")
            if not all(
                isinstance(action, LocalFunctionalAction) for action in action_values
            ):
                raise ValueError(
                    "A functional finite-element form may contain only "
                    "LocalFunctionalAction values."
                )
            if tuple(action.term.term_id for action in action_values) != tuple(
                term.term_id for term in functional_.terms
            ):
                raise ValueError(
                    "Functional actions must match functional terms in declared order."
                )
        action_ids = tuple(action.action_id for action in action_values)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Finite-element form action IDs must be unique.")
        self.form_id = canonical_fingerprint(
            {
                "kind": "finite-element-form",
                "declared_id": identifier,
                "field_names": list(fields),
                "actions": [_action_payload(action) for action in action_values],
                "properties": {
                    "diagonal": properties_.diagonal,
                    "triangular": properties_.triangular,
                    "self_adjoint": properties_.self_adjoint,
                    "positive_definite": properties_.positive_definite,
                    "positive_semidefinite": properties_.positive_semidefinite,
                    "block_diagonal": properties_.block_diagonal,
                    "rank": properties_.rank,
                    "evidence": [list(item) for item in properties_.evidence],
                },
                "auxiliary": auxiliary_identifier,
                "functional": (
                    None if functional_ is None else functional_.functional_id
                ),
            }
        )
        self.field_names = fields
        self.actions = action_values
        self.declared_properties = properties_
        self.auxiliary_evaluator = auxiliary_evaluator
        self.auxiliary_id = auxiliary_identifier
        self.functional = functional_

    @property
    def field_name(self) -> str:
        if len(self.field_names) != 1:
            raise ValueError("field_name is ambiguous for a mixed finite-element form.")
        return self.field_names[0]


def _action_domain(
    term: FiniteElementAction,
    discretization: AbstractPreparedLocalDiscretization,
    /,
) -> IntegrationDomain:
    if term.domain is not None:
        domain = term.domain
    elif isinstance(term, (BoundaryLoadAction, ExteriorFacetAction)):
        domain = discretization.integration_domain("exterior_facet")
    elif isinstance(term, InteriorFacetAction):
        domain = discretization.integration_domain("interior_facet")
    else:
        domain = discretization.integration_domain("cell")
    if domain.support_id != discretization.support.support_id:
        raise ValueError("Finite-element term domain belongs to another support.")
    return domain


def _action_rule(
    term: FiniteElementAction,
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


def finite_element_form_from_functional(
    functional: Functional,
    fields: Mapping[str, str],
    regions: Mapping[str, IntegrationDomain | None],
    /,
    *,
    rules: Mapping[
        str,
        Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]],
    ]
    | None = None,
    form_id: str | None = None,
) -> FiniteElementForm:
    """Bind one physical functional to finite-element local actions."""
    if not isinstance(functional, Functional):
        raise TypeError("functional must be a variational.Functional.")
    expected_fields = set(functional.field_names)
    if set(fields) != expected_fields:
        raise KeyError(
            "Finite-element functional fields must match exactly; "
            f"missing={tuple(sorted(expected_fields - set(fields)))}, "
            f"extra={tuple(sorted(set(fields) - expected_fields))}."
        )
    bound_fields = tuple(str(fields[name]) for name in functional.field_names)
    if any(not field for field in bound_fields):
        raise ValueError("Finite-element functional field bindings must be non-empty.")
    if len(set(bound_fields)) != len(bound_fields):
        raise ValueError("Finite-element functional field bindings must be one-to-one.")
    expected_regions = set(functional.region_names)
    if set(regions) != expected_regions:
        raise KeyError(
            "Finite-element functional regions must match exactly; "
            f"missing={tuple(sorted(expected_regions - set(regions)))}, "
            f"extra={tuple(sorted(set(regions) - expected_regions))}."
        )
    rule_bindings = {} if rules is None else dict(rules)
    unknown_rule_regions = tuple(sorted(set(rule_bindings).difference(expected_regions)))
    if unknown_rule_regions:
        raise KeyError(f"Unknown functional rule regions {unknown_rule_regions}.")
    field_mapping = {name: str(fields[name]) for name in functional.field_names}
    actions: list[LocalFunctionalAction] = []
    for term in functional.terms:
        term_semantic_fields = {spec.field_name for spec in term.fields}
        output_fields = tuple(
            field_mapping[name]
            for name in functional.variable_fields
            if name in term_semantic_fields
        )
        if not output_fields:
            raise ValueError(
                f"Functional term {term.identifier!r} has no active variable field."
            )
        actions.append(
            LocalFunctionalAction(
                term,
                field_mapping,
                output_fields,
                domain=regions[term.region],
                rules=rule_bindings.get(term.region, ()),
                action_id=f"{functional.identifier}:{term.identifier}",
            )
        )
    return FiniteElementForm(
        functional.identifier if form_id is None else form_id,
        bound_fields,
        actions,
        functional=functional,
    )


def _pure_neumann_sipg(form: FiniteElementForm, /) -> bool:
    facet_terms = tuple(
        term for term in form.actions if isinstance(term, SIPGFacetAction)
    )
    if not facet_terms:
        return False
    boundary_kinds = tuple(
        term.boundary.kind for term in facet_terms if term.boundary is not None
    )
    return not any(kind in ("dirichlet", "robin") for kind in boundary_kinds)


def _sipg_constant_subspace(
    form: FiniteElementForm,
    discretization: FiniteElementDiscretization,
    space,
    /,
) -> LinearSubspace:
    if len(form.field_names) != 1:
        raise ValueError("SIPG nullspace construction requires one field.")
    field_index = discretization._field_index(form.field_names[0])
    dof_map = discretization.dof_maps[field_index]
    if dof_map.component_shape:
        raise ValueError("SIPG nullspace construction requires a scalar field.")
    cell_count = sum(block.cell_count for block in discretization.mesh.blocks)
    parents = np.arange(cell_count, dtype=np.int32)

    def root(value: int) -> int:
        current = int(value)
        while parents[current] != current:
            parents[current] = parents[parents[current]]
            current = int(parents[current])
        return current

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for owner, neighbour in zip(
        np.asarray(discretization.interior_facet_domain.owner_cells),
        np.asarray(discretization.interior_facet_domain.neighbour_cells),
        strict=True,
    ):
        union(int(owner), int(neighbour))
    roots = tuple(sorted({root(cell) for cell in range(cell_count)}))
    component_by_root = {value: index for index, value in enumerate(roots)}
    dtype = np.asarray(space.zeros()).dtype
    basis = np.zeros((space.size, len(roots)), dtype=dtype)
    cell_offset = 0
    for block_dofs in dof_map.cell_dofs:
        routes = np.asarray(block_dofs, dtype=np.int32)
        for local_cell, dofs in enumerate(routes):
            component = component_by_root[root(cell_offset + local_cell)]
            basis[dofs, component] = 1.0
        cell_offset += routes.shape[0]
    return LinearSubspace(
        space,
        basis,
        subspace_id=canonical_fingerprint(
            {
                "kind": "sipg-constant-nullspace",
                "topology": discretization.mesh.topology_id,
                "field": form.field_names[0],
                "components": len(roots),
            }
        ),
    )


class CompiledFiniteElementProblem(StrictModule, NonTrainableState):
    form: FiniteElementForm
    discretization: AbstractPreparedLocalDiscretization
    constraint: ConstraintMap | None
    constraints: tuple[ConstraintMap | None, ...]
    execution_policy: FiniteElementExecutionPolicy
    _action_ir: LocalActionIR
    _workset_program: WorksetProgram
    _kernel_table: KernelTable
    lift: object
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        form: FiniteElementForm,
        discretization: AbstractPreparedLocalDiscretization,
        /,
        *,
        constraint: ConstraintMap
        | FiniteElementDirichletConstraint
        | FiniteElementLinearConstraint
        | None = None,
        dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
        constraints: Mapping[
            str,
            ConstraintMap
            | FiniteElementDirichletConstraint
            | FiniteElementLinearConstraint,
        ]
        | None = None,
        dirichlet_values_by_field: Mapping[str, ArrayLike | Callable[[Array], ArrayLike]]
        | None = None,
        execution_policy: FiniteElementExecutionPolicy | None = None,
    ):
        if not isinstance(form, FiniteElementForm):
            raise TypeError("form must be a FiniteElementForm.")
        if not isinstance(discretization, AbstractPreparedLocalDiscretization):
            raise TypeError("discretization must be AbstractPreparedLocalDiscretization.")
        policy = (
            (
                FiniteElementExecutionPolicy()
                if isinstance(discretization, FiniteElementDiscretization)
                else FiniteElementExecutionPolicy(
                    realization="matrix_free",
                    local_kernel="sum_factorized",
                )
            )
            if execution_policy is None
            else execution_policy
        )
        if not isinstance(policy, FiniteElementExecutionPolicy):
            raise TypeError(
                "execution_policy must be FiniteElementExecutionPolicy or None."
            )
        if not isinstance(discretization, FiniteElementDiscretization):
            if policy.realization != "matrix_free":
                raise ValueError(
                    "Method-neutral local discretizations require matrix-free execution."
                )
            if any(
                isinstance(
                    action,
                    (
                        ExteriorFacetAction,
                        InteriorFacetAction,
                        SIPGFacetAction,
                    ),
                )
                for action in form.actions
            ):
                raise ValueError("Non-load facet actions remain finite-element-gated.")
        if len(form.field_names) > 1 and (
            constraint is not None or dirichlet_values is not None
        ):
            raise ValueError(
                "Mixed forms require field-keyed constraints and Dirichlet values."
            )
        resolved_constraints = {} if constraints is None else dict(constraints)
        resolved_values = (
            {} if dirichlet_values_by_field is None else dict(dirichlet_values_by_field)
        )
        if len(form.field_names) == 1:
            field_name = form.field_names[0]
            if constraint is not None:
                resolved_constraints[field_name] = constraint
            if dirichlet_values is not None:
                resolved_values[field_name] = dirichlet_values
        unknown_constraints = set(resolved_constraints) - set(form.field_names)
        unknown_values = set(resolved_values) - set(form.field_names)
        if unknown_constraints or unknown_values:
            raise ValueError("Constraint/value mappings contain unknown fields.")
        if resolved_constraints and any(
            isinstance(term, SIPGFacetAction) for term in form.actions
        ):
            raise ValueError(
                "DG SIPG Dirichlet data must use Nitsche boundary terms, "
                "not strong finite-element constraints."
            )
        constraint_values = []
        lifts = []
        for field_name in form.field_names:
            field_index = discretization._field_index(field_name)
            full_space = discretization.field_spaces[field_index].vector_space
            constraint_value = resolved_constraints.get(field_name)
            boundary_values = resolved_values.get(field_name)
            if constraint_value is None:
                if boundary_values is not None:
                    raise ValueError(
                        f"Dirichlet values for {field_name!r} require a constraint."
                    )
                lift_value = full_space.zeros()
            else:
                if isinstance(constraint_value, ConstraintMap):
                    if boundary_values is not None:
                        raise ValueError(
                            "Plain ConstraintMap values are homogeneous and do not "
                            "accept Dirichlet values."
                        )
                    constraint_map = constraint_value
                    lift_value = full_space.zeros()
                elif isinstance(
                    constraint_value,
                    (FiniteElementDirichletConstraint, FiniteElementLinearConstraint),
                ):
                    if constraint_value.field_name != field_name:
                        raise ValueError("Constraint field does not match its map key.")
                    constraint_map = constraint_value.constraint_map
                    if isinstance(constraint_value, FiniteElementLinearConstraint):
                        if boundary_values is not None:
                            raise ValueError(
                                "Homogeneous hp constraints do not accept "
                                "Dirichlet values."
                            )
                        lift_value = constraint_value.lift()
                    else:
                        if boundary_values is None:
                            raise ValueError(
                                f"Constraint for {field_name!r} requires "
                                "Dirichlet values."
                            )
                        lift_value = constraint_value.lift(boundary_values)
                else:
                    raise TypeError(
                        "constraints must contain ConstraintMap or finite-element "
                        "constraint values."
                    )
                if not constraint_map.full_space.compatible(full_space):
                    raise ValueError(
                        "Constraint full space does not match its declared field."
                    )
            constraint_values.append(None if constraint_value is None else constraint_map)
            lifts.append(lift_value)
        constraints_ = tuple(constraint_values)
        lift = lifts[0] if len(lifts) == 1 else tuple(lifts)
        from .fem import (
            compile_workset_program,
            kernel_table_from_form,
            lower_finite_element_form,
        )

        action_ir = lower_finite_element_form(form, discretization)
        workset_program = compile_workset_program(
            action_ir,
            form,
            discretization,
            local_kernel=policy.local_kernel,
            realization=policy.realization,
        )
        kernel_table = kernel_table_from_form(
            form,
            action_ir,
            workset_program,
            discretization,
        )
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-finite-element-problem",
                "form": form.form_id,
                "terms": [_action_payload(term) for term in form.actions],
                "discretization": discretization.prepared_id,
                "precision": discretization.precision_policy.policy_id,
                "constraints": [
                    None if value is None else value.constraint_id
                    for value in constraints_
                ],
                "lift": array_tree_fingerprint(lift),
                "kernel_table": kernel_table.table_id,
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
        self.constraint = constraints_[0] if len(constraints_) == 1 else None
        self.constraints = constraints_
        self.execution_policy = policy
        self._action_ir = action_ir
        self._workset_program = workset_program
        self._kernel_table = kernel_table
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
        self.compilation_id = compilation_id

    @property
    def field_index(self) -> int:
        if len(self.form.field_names) != 1:
            raise ValueError("field_index is ambiguous for a mixed form.")
        return self.discretization._field_index(self.form.field_names[0])

    @property
    def full_space(self):
        spaces = tuple(
            self.discretization.field_spaces[
                self.discretization._field_index(name)
            ].vector_space
            for name in self.form.field_names
        )
        return (
            spaces[0]
            if len(spaces) == 1
            else BlockSpace(spaces, names=self.form.field_names)
        )

    @property
    def state_space(self):
        spaces = tuple(
            (
                self.discretization.field_spaces[
                    self.discretization._field_index(name)
                ].vector_space
                if constraint is None
                else constraint.reduced_space
            )
            for name, constraint in zip(
                self.form.field_names,
                self.constraints,
                strict=True,
            )
        )
        return (
            spaces[0]
            if len(spaces) == 1
            else BlockSpace(spaces, names=self.form.field_names)
        )

    @property
    def residual_space(self):
        if len(self.form.field_names) == 1:
            return DualSpace(self.state_space)
        return BlockSpace(
            tuple(DualSpace(space) for space in self.state_space.spaces),
            names=self.form.field_names,
        )

    @property
    def constraint_map(self) -> ConstraintMap | None:
        if len(self.form.field_names) == 1:
            constraint = self.constraints[0]
            return constraint
        blocks = []
        for row, (full_block, reduced_block, constraint) in enumerate(
            zip(
                self.full_space.spaces,
                self.state_space.spaces,
                self.constraints,
                strict=True,
            )
        ):
            block_row = []
            for column in range(len(self.form.field_names)):
                if row != column:
                    block_row.append(None)
                elif constraint is None:
                    block_row.append(IdentityLinearOperator(full_block))
                else:
                    block_row.append(constraint.prolongation)
            blocks.append(tuple(block_row))
        prolongation = BlockLinearOperator(
            tuple(blocks),
            source=self.state_space,
            target=self.full_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-block-constraint-prolongation",
                    "compilation": self.compilation_id,
                }
            ),
        )
        return ConstraintMap(
            self.full_space,
            self.state_space,
            prolongation,
            constraint_id=canonical_fingerprint(
                {
                    "kind": "finite-element-block-constraint-map",
                    "compilation": self.compilation_id,
                }
            ),
        )

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
        self.discretization.validate_local_runtime(context.runtime)
        return context

    def expand(self, state: object, args: object = None, /):
        values = self.state_space.validate(state)
        context = self._execution_context(args)
        lifts = self.lift if context.lift is None else context.lift
        if len(self.form.field_names) == 1:
            constraint = self.constraints[0]
            if constraint is None:
                return self.full_space.validate(values)
            return constraint.expand(values, lifts)
        if not isinstance(lifts, tuple):
            raise ValueError("Mixed finite-element lifts must be field-block tuples.")
        expanded = []
        for value, lift, constraint, full_block in zip(
            values,
            lifts,
            self.constraints,
            self.full_space.spaces,
            strict=True,
        ):
            expanded.append(
                full_block.validate(value)
                if constraint is None
                else constraint.expand(value, lift)
            )
        return self.full_space.validate(tuple(expanded))

    @property
    def potential_compatible(self) -> bool:
        """Whether the residual is generated from one declared scalar functional."""
        return (
            self.form.functional is not None
            and bool(self.form.actions)
            and all(
                isinstance(action, LocalFunctionalAction) for action in self.form.actions
            )
        )

    def _full_potential_evaluation(
        self,
        state: object,
        args: object = None,
        /,
    ) -> FunctionalEvaluation:
        from .fem._executor import execute_finite_element_potential

        if not self.potential_compatible or self.form.functional is None:
            raise ValueError(
                "Finite-element potential evaluation requires a form generated "
                "from variational.Functional; arbitrary residual, flux, source, "
                "and operator actions are not relabeled as conservative."
            )
        full = self.full_space.validate(state)
        context = self._execution_context(args)
        value, term_values = execute_finite_element_potential(
            self._workset_program,
            self.form,
            self.discretization,
            full,
            context,
        )
        return FunctionalEvaluation(
            value,
            term_values,
            functional_id=self.form.functional.functional_id,
            binding_id=self.form.form_id,
        )

    def full_potential(self, state: object, args: object = None, /) -> Array:
        """Evaluate the declared scalar functional on the full state space."""
        return self._full_potential_evaluation(state, args).value

    def potential_evaluation(
        self,
        state: object,
        args: object = None,
        /,
    ) -> FunctionalEvaluation:
        """Evaluate ordered functional terms on the constrained state space."""
        context = self._execution_context(args)
        return self._full_potential_evaluation(self.expand(state, context), context)

    def potential(self, state: object, args: object = None, /) -> Array:
        """Evaluate the scalar functional on the constrained state space."""
        return self.potential_evaluation(state, args).value

    def value_and_residual(self, state: object, args: object = None, /):
        """Return the scalar functional and reduced variation in one local pass."""
        from .fem._executor import execute_finite_element_value_and_residual

        state_ = self.state_space.validate(state)
        context = self._execution_context(args)
        full = self.expand(state_, context)
        value, _term_values, full_residual = execute_finite_element_value_and_residual(
            self._workset_program,
            self.form,
            self.discretization,
            full,
            self.execution_policy.accumulation,
            context,
        )
        if len(self.form.field_names) == 1:
            constraint = self.constraints[0]
            reduced = (
                self.residual_space.validate(full_residual)
                if constraint is None
                else constraint.constraint_map.pullback_dual(full_residual)
            )
        else:
            reduced = self.residual_space.validate(
                tuple(
                    block
                    if constraint is None
                    else constraint.constraint_map.pullback_dual(block)
                    for block, constraint in zip(
                        full_residual,
                        self.constraints,
                        strict=True,
                    )
                )
            )
        return value, reduced

    def as_minimization_problem(self) -> MinimizationProblem:
        """Expose the declared scalar functional to iterative optimization."""
        if not self.potential_compatible:
            raise ValueError("Only functional-generated forms define minimization.")
        return MinimizationProblem(lambda state, args: self.potential(state, args))

    def full_residual(self, state: object, args: object = None, /):
        from .fem._executor import execute_finite_element_residual

        full = self.full_space.validate(state)
        context = self._execution_context(args)
        return execute_finite_element_residual(
            self._action_ir,
            self._workset_program,
            self.form,
            self._kernel_table,
            self.discretization,
            full,
            self.execution_policy.accumulation,
            context,
        )

    def residual(self, state: object, args: object = None, /):
        context = self._execution_context(args)
        full_residual = self.full_residual(self.expand(state, context), context)
        if len(self.form.field_names) == 1:
            constraint = self.constraints[0]
            if constraint is None:
                return self.residual_space.validate(full_residual)
            return constraint.pullback_dual(full_residual)
        reduced = tuple(
            full if constraint is None else constraint.pullback_dual(full)
            for full, constraint in zip(
                full_residual,
                self.constraints,
                strict=True,
            )
        )
        return self.residual_space.validate(reduced)

    def weak_residual(self, state: object, args: object = None, /):
        """Return the assembled dual-valued weak residual without a mass inverse."""
        return self.residual(state, args)

    def mass_inverted_rate(
        self,
        state: object,
        args: object = None,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
        mass_policy: object = None,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        """Solve the explicitly selected mass operator for the negative residual."""
        if len(self.form.field_names) != 1:
            raise ValueError("Mass-inverted rates currently require one field.")
        coefficient_ = jnp.asarray(mass_coefficient)
        if coefficient_.shape != ():
            raise ValueError("Mass-inverted rate coefficient must be scalar.")
        context = self._execution_context(args)
        _, mass = self._mass_operators(context, coefficient_, mass_policy)
        weak = jax.tree.map(lambda value: -value, self.residual(state, context))
        primal_mass = FunctionLinearOperator(
            lambda value: self.state_space.inverse_riesz(mass.mv(value)),
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
                    "kind": "finite-element-mass-inverted-rate",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )
        right_hand_side = self.state_space.inverse_riesz(weak)
        return solve(
            LinearSystem(primal_mass),
            right_hand_side,
            policy=linear_policy,
        ).value

    def residual_with_auxiliary(
        self,
        state: object,
        args: object = None,
        /,
    ):
        from .fem import FiniteElementAuxiliaryEvaluation

        residual = self.residual(state, args)
        if self.form.auxiliary_evaluator is None:
            auxiliary = FiniteElementAuxiliaryEvaluation()
        else:
            auxiliary = self.form.auxiliary_evaluator(
                self.state_space.validate(state),
                self._execution_context(args),
            )
            if not isinstance(auxiliary, FiniteElementAuxiliaryEvaluation):
                raise TypeError(
                    "Form auxiliary evaluator must return "
                    "FiniteElementAuxiliaryEvaluation."
                )
        return residual, auxiliary

    def as_nonlinear_problem(self) -> NonlinearSystemProblem:
        if self.form.auxiliary_evaluator is None:
            return NonlinearSystemProblem(
                lambda state, args: self.residual(state, args),
                state_space=self.state_space,
                residual_space=self.residual_space,
                problem_id=self.compilation_id,
            )
        return NonlinearSystemProblem(
            lambda state, args: self.residual_with_auxiliary(state, args),
            has_aux=True,
            validity=lambda state, residual, auxiliary, args: auxiliary.valid,
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
            transpose_action=lambda cotangent: jax.vjp(
                lambda value: self.residual(value, context),
                state_,
            )[1](cotangent)[0],
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-linearization",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                }
            ),
        )

    def block_dependency_graph(self, /) -> tuple[tuple[bool, ...], ...]:
        fields = self.form.field_names
        consumed = {
            (fields.index(output_field), fields.index(input_field))
            for action in self.form.actions
            for output_field in _action_output_fields(action)
            for input_field in _action_input_fields(action)
        }
        return tuple(
            tuple((row, column) in consumed for column in range(len(fields)))
            for row in range(len(fields))
        )

    def block_linearization_operator(
        self,
        state: object,
        args: object = None,
        /,
    ) -> BlockLinearOperator:
        if not isinstance(self.state_space, BlockSpace) or not isinstance(
            self.residual_space, BlockSpace
        ):
            raise ValueError("Block linearization requires a product-space form.")
        state_ = self.state_space.validate(state)
        context = self._execution_context(args)
        graph = self.block_dependency_graph()
        rows = []
        for row_index, (target_space, dependencies) in enumerate(
            zip(self.residual_space.spaces, graph, strict=True)
        ):
            row = []
            for column_index, (source_space, present) in enumerate(
                zip(self.state_space.spaces, dependencies, strict=True)
            ):
                if not present:
                    row.append(None)
                    continue

                def block_action(
                    direction,
                    row_index_=row_index,
                    column_index_=column_index,
                ):
                    directions = list(self.state_space.zeros())
                    directions[column_index_] = direction
                    image = jax.jvp(
                        lambda value: self.residual(value, context),
                        (state_,),
                        (tuple(directions),),
                    )[1]
                    return image[row_index_]

                row.append(
                    FunctionLinearOperator(
                        block_action,
                        source=source_space,
                        target=target_space,
                        operator_id=canonical_fingerprint(
                            {
                                "kind": "finite-element-jacobian-block",
                                "compilation": self.compilation_id,
                                "row": row_index,
                                "column": column_index,
                                "runtime": context.runtime.runtime_id,
                            }
                        ),
                    )
                )
            rows.append(tuple(row))
        return BlockLinearOperator(
            tuple(rows),
            source=self.state_space,
            target=self.residual_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-block-linearization",
                    "compilation": self.compilation_id,
                    "runtime": context.runtime.runtime_id,
                    "graph": [list(row) for row in graph],
                }
            ),
        )

    def preconditioner_operator(
        self,
        preconditioner_form: FiniteElementForm,
        state: object,
        args: object = None,
        /,
    ):
        if not isinstance(preconditioner_form, FiniteElementForm):
            raise TypeError("preconditioner_form must be a FiniteElementForm.")
        if preconditioner_form.field_names != self.form.field_names:
            raise ValueError(
                "Preconditioner form fields must exactly match the problem form."
            )
        if any(constraint is not None for constraint in self.constraints):
            raise ValueError(
                "Preconditioner-form binding currently requires unconstrained spaces."
            )
        compiled = CompiledFiniteElementProblem(
            preconditioner_form,
            self.discretization,
            execution_policy=self.execution_policy,
        )
        return (
            compiled.block_linearization_operator(state, args)
            if len(self.form.field_names) > 1
            else compiled.linearization_operator(state, args)
        )

    def exact_diagonal(
        self,
        state: object | None = None,
        args: object = None,
        /,
        *,
        allow_coordinate_fallback: bool = False,
        maximum_coordinate_size: int = 4096,
    ):
        from .fem import FiniteElementDiagonalData

        raw = (
            self.affine_operator(args)
            if state is None
            else self.operator_function(self.state_space.validate(state), args)
        )
        method = "workset"
        if isinstance(raw, SparseCoordinateOperator) and isinstance(
            raw.relation, EdgeRelation
        ):
            relation = raw.relation
            on_diagonal = relation.valid & (
                relation.source_indices == relation.target_indices
            )
            diagonal = (
                jnp.zeros((relation.source_size,), dtype=raw.coefficients.dtype)
                .at[relation.source_indices]
                .add(jnp.where(on_diagonal, raw.coefficients, 0.0))
            )
            method = "sparse-coordinate"
        elif raw.capabilities.diagonal_assembly:
            diagonal = assemble_diagonal(raw)
        else:
            if not allow_coordinate_fallback:
                raise ValueError(
                    "Exact diagonal lowering is unavailable; explicitly enable the "
                    "bounded coordinate-linearization fallback."
                )
            limit = int(maximum_coordinate_size)
            if limit < 1 or self.state_space.size > limit:
                raise ValueError(
                    "Coordinate diagonal fallback exceeds its explicit dimension cap."
                )
            coordinates = jnp.eye(
                self.state_space.size,
                dtype=self.state_space.flatten(self.state_space.zeros()).dtype,
            )

            def column(direction):
                image = raw.mv(self.state_space.unflatten(direction))
                return self.residual_space.flatten(image)

            matrix_columns = jax.vmap(column)(coordinates)
            diagonal = self.state_space.unflatten(jnp.diag(matrix_columns))
            method = "coordinate-linearization"
        return FiniteElementDiagonalData(
            diagonal,
            method,
            raw.operator_id,
        )

    def preconditioner_data(
        self,
        state: object | None = None,
        args: object = None,
        /,
    ):
        from .fem import FiniteElementPreconditionerData

        diagonal_data = self.exact_diagonal(
            state,
            args,
            allow_coordinate_fallback=False,
        )
        diagonal = diagonal_data.diagonal
        graph = (
            self.block_dependency_graph()
            if len(self.form.field_names) > 1
            else ((True,),)
        )
        return FiniteElementPreconditionerData(
            diagonal,
            graph,
            tuple(workset.workset_id for workset in self._workset_program.worksets),
        )

    def operator_function(self, state, args):
        if len(self.form.field_names) > 1:
            return self.block_linearization_operator(state, args)
        if self.execution_policy.realization == "sparse":
            try_affine = all(
                (
                    (
                        isinstance(term, DiffusionAction)
                        and term.diffusivity.constant
                        and term.diffusivity.value.shape == ()
                    )
                    or (
                        isinstance(term, MassAction)
                        and term.coefficient.constant
                        and term.coefficient.value.shape == ()
                    )
                    or isinstance(term, (SourceAction, BoundaryLoadAction))
                )
                and term.domain is None
                and not term.rules
                for term in self.form.actions
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

    def _structural_affine_operator(self, args: object = None, /):
        if (
            self.execution_policy.realization != "sparse"
            or len(self.form.field_names) != 1
            or self.constraint is not None
        ):
            return None
        context = self._execution_context(args)
        if context.runtime.runtime_id != self.discretization.default_runtime.runtime_id:
            return None
        operators = []
        mass, stiffness = self.discretization.assemble_field_operators(
            self.form.field_name,
            context.runtime,
        )
        for action in self.form.actions:
            if (
                isinstance(action, DiffusionAction)
                and action.domain is None
                and not action.rules
                and action.diffusivity.constant
                and action.diffusivity.value.shape == ()
            ):
                operators.append(action.diffusivity.value * stiffness)
            elif (
                isinstance(action, MassAction)
                and action.domain is None
                and not action.rules
                and action.coefficient.constant
                and action.coefficient.value.shape == ()
            ):
                operators.append(action.coefficient.value * mass)
            elif isinstance(action, (SourceAction, BoundaryLoadAction)):
                continue
            else:
                return None
        if not operators:
            return None
        result = operators[0]
        for operator in operators[1:]:
            result = result + operator
        return result

    def affine_operator(self, args: object = None, /):
        """Return exact structural storage or linearize the authoritative program."""
        structural = self._structural_affine_operator(args)
        if structural is not None:
            return structural
        zero = self.state_space.zeros()
        return (
            self.block_linearization_operator(zero, args)
            if len(self.form.field_names) > 1
            else self.linearization_operator(zero, args)
        )

    def to_scipy_csr(self, args: object = None, /):
        """Assemble exact sparse coordinates and convert them directly to SciPy CSR."""
        import scipy.sparse as sp

        operator = self.affine_operator(args)
        plan = plan_sparse_assembly(operator)
        if plan.uses_materialization:
            raise ValueError(
                "Direct SciPy CSR conversion requires a structurally sparse lowering."
            )
        sparse = prepare_sparse_assembly(plan, operator).operator
        if not isinstance(sparse, SparseCoordinateOperator):
            raise TypeError("Sparse assembly did not produce coordinate storage.")
        relation = sparse.relation
        if isinstance(relation, RowRelation):
            relation = relation.as_edge_relation()
        elif not isinstance(relation, EdgeRelation):
            raise TypeError("Sparse coordinate storage has an unknown relation type.")
        valid = np.asarray(relation.valid, dtype=bool)
        rows = np.asarray(relation.target_indices)[valid]
        columns = np.asarray(relation.source_indices)[valid]
        values = np.asarray(sparse.coefficients)[valid]
        return sp.coo_array(
            (values, (rows, columns)),
            shape=(relation.target_size, relation.source_size),
        ).tocsr()

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

    def _default_nullspace_subspace(self, /) -> LinearSubspace | None:
        if any(constraint is not None for constraint in self.constraints):
            return None
        action_ids = {action.action_id for action in self.form.actions}
        if len(self.form.field_names) > 1 and (
            {"stokes-momentum", "stokes-incompressibility"} <= action_ids
            or {"darcy-flux", "darcy-mass-balance"} <= action_ids
        ):
            pressure_index = 1
            blocks = list(self.state_space.zeros())
            blocks[pressure_index] = jnp.ones_like(blocks[pressure_index])
            basis = self.state_space.flatten(tuple(blocks))[:, None]
            return LinearSubspace(
                self.state_space,
                basis,
                subspace_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-pressure-nullspace",
                        "compilation": self.compilation_id,
                    }
                ),
            )
        if "elasticity" in action_ids and len(self.form.field_names) == 1:
            zero = self.state_space.zeros()
            if zero.ndim != 2 or zero.shape[1] not in (2, 3):
                return None
            dimension = zero.shape[1]
            coordinates = self.discretization.dof_maps[
                self.field_index
            ].evaluate_coordinates(
                self.discretization.mesh,
                self.discretization.default_runtime.coordinates,
            )
            modes = []
            for component in range(dimension):
                mode = jnp.zeros_like(zero).at[:, component].set(1.0)
                modes.append(self.state_space.flatten(mode))
            if dimension == 2:
                rotation = jnp.stack((-coordinates[:, 1], coordinates[:, 0]), axis=-1)
                modes.append(self.state_space.flatten(rotation))
            else:
                for axis in range(3):
                    vector = jnp.zeros((3,), dtype=zero.dtype).at[axis].set(1.0)
                    rotation = jnp.cross(
                        jnp.broadcast_to(vector, coordinates.shape), coordinates
                    )
                    modes.append(self.state_space.flatten(rotation))
            return LinearSubspace(
                self.state_space,
                jnp.stack(tuple(modes), axis=1),
                subspace_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-rigid-nullspace",
                        "compilation": self.compilation_id,
                        "dimension": dimension,
                    }
                ),
            )
        scalar_diffusion = (
            len(self.form.field_names) == 1
            and any(isinstance(action, DiffusionAction) for action in self.form.actions)
            and all(
                isinstance(
                    action,
                    (
                        DiffusionAction,
                        SourceAction,
                        BoundaryLoadAction,
                    ),
                )
                for action in self.form.actions
            )
        )
        if scalar_diffusion and self.state_space.zeros().ndim == 1:
            return LinearSubspace(
                self.state_space,
                jnp.ones(
                    (self.state_space.size, 1), dtype=self.state_space.zeros().dtype
                ),
                subspace_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-constant-nullspace",
                        "compilation": self.compilation_id,
                    }
                ),
            )
        return None

    def linear_system(
        self,
        args: object = None,
        /,
        *,
        nullspace_policy: NullspacePolicy | None = None,
    ) -> tuple[LinearSystem, object]:
        raw_operator = self.affine_operator(args)
        properties = self.form.declared_properties
        primal_operator = FunctionLinearOperator(
            lambda state: self.state_space.inverse_riesz(raw_operator.mv(state)),
            source=self.state_space,
            target=self.state_space,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "riesz-finite-element-affine-operator",
                    "compilation": self.compilation_id,
                }
            ),
        )
        if nullspace_policy is None and _pure_neumann_sipg(self.form):
            constant_modes = _sipg_constant_subspace(
                self.form,
                self.discretization,
                self.state_space,
            )
            certificate = KernelCertificate(
                primal_operator,
                constant_modes,
                evidence="verified",
                scope="numerical",
                complete=True,
                tolerance=1.0e-9,
            )
            nullspace_policy = NullspacePolicy(
                right=constant_modes,
                left=constant_modes,
                certificate=certificate,
                compatibility="error",
                gauge="minimum-norm",
            )
        if nullspace_policy is None:
            default_modes = self._default_nullspace_subspace()
            if default_modes is not None:
                certificate = KernelCertificate(
                    primal_operator,
                    default_modes,
                    evidence="verified",
                    left=default_modes,
                    scope="numerical",
                    complete=True,
                    tolerance=1.0e-9,
                )
                nullspace_policy = NullspacePolicy(
                    right=default_modes,
                    left=default_modes,
                    certificate=certificate,
                    compatibility="error",
                    gauge="minimum-norm",
                )
        zero = self.state_space.zeros()
        right_hand_side = self.state_space.inverse_riesz(
            jax.tree.map(lambda value: -value, self.residual(zero, args))
        )
        return (
            LinearSystem(
                primal_operator,
                nullspace_policy=nullspace_policy,
            ),
            right_hand_side,
        )

    def _compile_unit_mass_problem(
        self,
        mass_policy: object = None,
        /,
    ) -> CompiledFiniteElementProblem:
        from .fem._execution import FiniteElementMassPolicy

        policy = FiniteElementMassPolicy() if mass_policy is None else mass_policy
        if not isinstance(policy, FiniteElementMassPolicy):
            raise TypeError("mass_policy must be FiniteElementMassPolicy or None.")
        local_kernel = "collocated" if policy.kind == "collocated_diagonal" else "auto"
        mass_rules = {}
        field_index = self.discretization._field_index(self.form.field_name)
        if policy.kind in ("exact", "lumped"):
            from ..integration import (
                GaussLegendreRule,
                ReferenceHexahedronRule,
                ReferenceQuadrilateralRule,
                ReferenceTetrahedronRule,
                ReferenceTriangleRule,
            )

            for block, element, coordinate_element in zip(
                self.discretization.mesh.blocks,
                self.discretization.elements[field_index],
                self.discretization.coordinate_elements,
                strict=True,
            ):
                exact_degree = 2 * element.degree + element.topological_dimension * max(
                    coordinate_element.degree - 1, 0
                )
                count = max(2, (exact_degree + 2) // 2)
                if block.cell_kind == "triangle":
                    count = max(count, (exact_degree + 3) // 2)
                    rule = ReferenceTriangleRule(GaussLegendreRule(count))
                elif block.cell_kind == "tetrahedron":
                    count = max(count, (exact_degree + 4) // 2)
                    rule = ReferenceTetrahedronRule(GaussLegendreRule(count))
                elif block.cell_kind == "quadrilateral":
                    tensor_degree = (
                        2 * element.degree
                        + element.topological_dimension * coordinate_element.degree
                        - 1
                    )
                    count = max(2, (tensor_degree + 2) // 2)
                    rule = ReferenceQuadrilateralRule(GaussLegendreRule(count))
                elif block.cell_kind == "hexahedron":
                    tensor_degree = (
                        2 * element.degree
                        + element.topological_dimension * coordinate_element.degree
                        - 1
                    )
                    count = max(2, (tensor_degree + 2) // 2)
                    rule = ReferenceHexahedronRule(GaussLegendreRule(count))
                else:
                    raise ValueError("Unsupported finite-element mass cell kind.")
                mass_rules[block.name] = rule
        if policy.kind == "collocated_diagonal":
            from ..integration import (
                GaussLobattoLegendreRule,
                ReferenceHexahedronRule,
                ReferenceQuadrilateralRule,
            )

            for block, element in zip(
                self.discretization.mesh.blocks,
                self.discretization.elements[field_index],
                strict=True,
            ):
                nodes = np.asarray(element.reference_nodes)
                counts = tuple(
                    np.unique(nodes[:, axis]).size for axis in range(nodes.shape[1])
                )
                if (
                    element.family != "TensorProductLagrange"
                    or element.representation != "point_value"
                    or len(set(counts)) != 1
                ):
                    raise ValueError(
                        "Collocated diagonal mass requires isotropic point-value "
                        "tensor elements."
                    )
                axis_rule = GaussLobattoLegendreRule(counts[0])
                rule = (
                    ReferenceQuadrilateralRule(axis_rule)
                    if block.cell_kind == "quadrilateral"
                    else ReferenceHexahedronRule(axis_rule)
                )
                rule_nodes = np.unique(
                    np.asarray(_reference_rule_data(rule).points)[:, 0]
                )
                if not np.allclose(
                    np.unique(nodes[:, 0]), rule_nodes, rtol=0.0, atol=2.0e-13
                ):
                    raise ValueError(
                        "Collocated diagonal mass requires Gauss--Lobatto nodes."
                    )
                mass_rules[block.name] = rule
        mass_action = MassAction(
            self.form.field_name,
            _UNIT_MASS_COEFFICIENT,
            rules=mass_rules,
            action_id="compiled-dynamics-mass",
        )
        mass_form = FiniteElementForm(
            "compiled-dynamics-mass",
            self.form.field_name,
            (mass_action,),
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
        )
        return CompiledFiniteElementProblem(
            mass_form,
            self.discretization,
            execution_policy=FiniteElementExecutionPolicy(
                realization="matrix_free",
                local_kernel=local_kernel,
                accumulation=self.execution_policy.accumulation,
            ),
        )

    def _mass_operators(
        self,
        context: FiniteElementExecutionContext,
        coefficient: Array,
        mass_policy: object = None,
        compiled_mass: CompiledFiniteElementProblem | None = None,
        /,
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
        from .fem._execution import FiniteElementMassPolicy

        policy = FiniteElementMassPolicy() if mass_policy is None else mass_policy
        if not isinstance(policy, FiniteElementMassPolicy):
            raise TypeError("mass_policy must be FiniteElementMassPolicy or None.")
        coefficient = eqx.error_if(
            coefficient,
            jnp.any(~jnp.isfinite(coefficient) | (jnp.real(coefficient) <= 0.0)),
            "Finite-element mass coefficient must be positive and finite.",
        )
        mass_problem = (
            self._compile_unit_mass_problem(policy)
            if compiled_mass is None
            else compiled_mass
        )
        if not isinstance(mass_problem, CompiledFiniteElementProblem):
            raise TypeError("compiled_mass must be CompiledFiniteElementProblem or None.")
        unit_mass = mass_problem.linearization_operator(
            mass_problem.state_space.zeros(),
            context,
        )
        full_mass = FunctionLinearOperator(
            lambda value: jax.tree.map(
                lambda image: coefficient * image,
                unit_mass.mv(value),
            ),
            source=self.full_space,
            target=DualSpace(self.full_space),
            transpose_action=lambda value: jax.tree.map(
                lambda image: coefficient * image,
                unit_mass.transpose_mv(value),
            ),
            properties=OperatorProperties(),
            operator_id=canonical_fingerprint(
                {
                    "kind": "scaled-finite-element-mass",
                    "compilation": self.compilation_id,
                    "unit_mass": unit_mass.operator_id,
                }
            ),
        )
        if policy.kind != "exact":
            ones = jax.tree.map(jnp.ones_like, self.full_space.zeros())
            diagonal = full_mass.mv(ones)
            invalid = jax.tree.reduce(
                lambda left, right: left | right,
                jax.tree.map(
                    lambda value: jnp.any(
                        ~jnp.isfinite(value) | (jnp.real(value) <= 0.0)
                    ),
                    diagonal,
                ),
                initializer=jnp.asarray(False),
            )
            diagonal = jax.tree.map(
                lambda value: eqx.error_if(
                    value,
                    invalid,
                    "Diagonal finite-element mass must be positive and finite.",
                ),
                diagonal,
            )
            full_mass = FunctionLinearOperator(
                lambda value: jax.tree.map(
                    lambda diagonal_, value_: diagonal_ * value_,
                    diagonal,
                    value,
                ),
                source=self.full_space,
                target=DualSpace(self.full_space),
                transpose_action=lambda value: jax.tree.map(
                    lambda diagonal_, value_: diagonal_ * value_,
                    diagonal,
                    value,
                ),
                properties=OperatorProperties(),
                operator_id=canonical_fingerprint(
                    {
                        "kind": "diagonal-finite-element-mass",
                        "compilation": self.compilation_id,
                        "policy": policy.policy_id,
                    }
                ),
            )
        if self.constraint is None:
            return full_mass, full_mass
        constraint_map = self.constraint
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
                    "policy": policy.policy_id,
                }
            ),
        )
        return full_mass, reduced_mass

    def as_dae_system(
        self,
        /,
        *,
        mass_coefficient: ArrayLike = 1.0,
        mass_policy: object = None,
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
                metric_data=base.metric_data,
                user_args=base.user_args,
            )

        def mass_matrix(time, state, args):
            context = execution_context(time, args)
            _, reduced_mass = self._mass_operators(context, coefficient_, mass_policy)
            return reduced_mass

        def vector_field(time, state, args):
            context = execution_context(time, args)
            full_mass, _ = self._mass_operators(context, coefficient_, mass_policy)
            residual = self.residual(state, context)
            if self.constraint is not None and context.lift_rate is not None:
                lift_rate = self.full_space.validate(context.lift_rate)
                residual = residual + self.constraint.pullback_dual(
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
        mass_policy: object = None,
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
        compiled_mass = self._compile_unit_mass_problem(mass_policy)

        def residual(time, configuration, velocity, acceleration, args):
            base = self._execution_context(args)
            context = FiniteElementExecutionContext(
                base.runtime,
                time=time,
                lift=base.lift,
                lift_rate=base.lift_rate,
                lift_acceleration=base.lift_acceleration,
                metric_data=base.metric_data,
                user_args=base.user_args,
            )
            full_mass, reduced_mass = self._mass_operators(
                context, mass_, mass_policy, compiled_mass
            )
            value = (
                reduced_mass.mv(acceleration)
                + damping_ * reduced_mass.mv(velocity)
                + self.residual(configuration, context)
            )
            if self.constraint is not None and context.lift_rate is not None:
                value = value + damping_ * self.constraint.pullback_dual(
                    full_mass.mv(self.full_space.validate(context.lift_rate))
                )
            if self.constraint is not None and context.lift_acceleration is not None:
                value = value + self.constraint.pullback_dual(
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
        mass_policy: object = None,
    ) -> GeneralizedEigenproblem:
        context = self._execution_context(args)
        raw_stiffness = self.affine_operator(context)
        _, raw_mass = self._mass_operators(
            context,
            jnp.asarray(mass_coefficient),
            mass_policy,
        )
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


def compile_finite_element_problem(
    form: FiniteElementForm,
    discretization: AbstractPreparedLocalDiscretization | FiniteElementHPEpoch,
    /,
    *,
    constraint: ConstraintMap
    | FiniteElementDirichletConstraint
    | FiniteElementLinearConstraint
    | None = None,
    dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
    constraints: Mapping[
        str,
        ConstraintMap | FiniteElementDirichletConstraint | FiniteElementLinearConstraint,
    ]
    | None = None,
    dirichlet_values_by_field: Mapping[str, ArrayLike | Callable[[Array], ArrayLike]]
    | None = None,
    execution_policy: FiniteElementExecutionPolicy | None = None,
) -> CompiledFiniteElementProblem:
    if isinstance(discretization, FiniteElementHPEpoch):
        epoch = discretization
        if epoch.discretization is None:
            raise ValueError("Compiled hp epochs require one prepared discretization.")
        resolved_constraints = {} if constraints is None else dict(constraints)
        for field_name, plan in epoch.constraints:
            if field_name not in resolved_constraints:
                resolved_constraints[field_name] = FiniteElementLinearConstraint(
                    field_name,
                    finite_element_hp_constraint(
                        epoch.discretization,
                        field_name,
                        cast(FiniteElementHPTraceConstraintPlan, plan),
                    ),
                )
        interior_domain, exterior_domain = finite_element_hp_domains(epoch)
        discretization = eqx.tree_at(
            lambda value: (
                value.interior_facet_domain,
                value.exterior_facet_domain,
            ),
            epoch.discretization,
            (interior_domain, exterior_domain),
        )
        constraints = resolved_constraints
    return CompiledFiniteElementProblem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=dirichlet_values,
        constraints=constraints,
        dirichlet_values_by_field=dirichlet_values_by_field,
        execution_policy=execution_policy,
    )


def compile_finite_element_functional(
    functional: Functional,
    discretization: AbstractPreparedLocalDiscretization | FiniteElementHPEpoch,
    /,
    *,
    fields: Mapping[str, str],
    regions: Mapping[str, IntegrationDomain | None],
    rules: Mapping[
        str,
        Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]],
    ]
    | None = None,
    constraint: ConstraintMap
    | FiniteElementDirichletConstraint
    | FiniteElementLinearConstraint
    | None = None,
    dirichlet_values: ArrayLike | Callable[[Array], ArrayLike] | None = None,
    constraints: Mapping[
        str,
        ConstraintMap | FiniteElementDirichletConstraint | FiniteElementLinearConstraint,
    ]
    | None = None,
    dirichlet_values_by_field: Mapping[str, ArrayLike | Callable[[Array], ArrayLike]]
    | None = None,
    execution_policy: FiniteElementExecutionPolicy | None = None,
) -> CompiledFiniteElementProblem:
    """Bind and compile one functional on a prepared local discretization."""
    form = finite_element_form_from_functional(
        functional,
        fields,
        regions,
        rules=rules,
    )
    return compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=dirichlet_values,
        constraints=constraints,
        dirichlet_values_by_field=dirichlet_values_by_field,
        execution_policy=execution_policy,
    )


__all__ = [
    "BoundaryLoadAction",
    "CellBilinearAction",
    "CellEnergyAction",
    "CellResidualAction",
    "CompiledFiniteElementProblem",
    "DiffusionAction",
    "ExteriorFacetAction",
    "FiniteElementAction",
    "FiniteElementExecutionContext",
    "FiniteElementExecutionPolicy",
    "FiniteElementForm",
    "InteriorFacetAction",
    "LocalFunctionalAction",
    "MassAction",
    "PairwiseVolumeFluxAction",
    "PreparedOperatorAction",
    "SIPGBoundaryCondition",
    "SIPGFacetAction",
    "SIPGPenaltyPolicy",
    "SourceAction",
    "coefficient",
    "compile_finite_element_functional",
    "compile_finite_element_problem",
    "finite_element_form_from_functional",
]
