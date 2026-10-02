"""Prepared lowering of form PDEs to native de Rham operators.

Admission binds scientific identities and coordinate layouts. Numerical field and
operator leaves remain dynamic; residual evaluation never fingerprints values or
prepares a discretization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import final, Literal

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization._cochain import CochainDiscretization
from ..discretization._structured_cochain import StructuredCochainBridge
from ..exterior._complex import AbstractDeRhamComplex, ComplexBoundary, DiscreteForm
from ..exterior._form_type import FiberProduct, FormType, FormValueSpec
from ..exterior._products import WhitneyProductPlan
from ..linalg._algebra_operators import apply_real_map_componentwise
from ..linalg._complexes import ComplexMap, HilbertComplex
from ..linalg._operators import AbstractLinearOperator
from ..linalg._spaces import AbstractVectorSpace
from ..typing import checked, parse
from ._ir import PDEExpression, PDEExpressionOp, PDEProblemIR
from ._validate import infer_expression_type, validate_pde_ir


type _LinearAction = Literal[
    "forward", "adjoint", "dual_forward", "dual_adjoint", "riesz", "inverse_riesz"
]


def _array_result(value: Array, /) -> Array:
    if not isinstance(value, (Array, jax.core.Tracer)):
        raise TypeError("Exterior realization operators must return array coordinates.")
    return jnp.asarray(value)


@final
class _LinearRoute(StrictModule):
    operator: AbstractLinearOperator | None
    space: AbstractVectorSpace | None
    source_indices: Array | None
    target_indices: Array | None
    action: _LinearAction = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    sign: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def _coordinate_action(self, coordinates: Array, /) -> Array:
        if self.action == "riesz" or self.action == "inverse_riesz":
            if self.space is None:
                raise ValueError("A Hodge route requires its prepared Hilbert space.")
            shaped = self.space.unflatten(coordinates)
            image = (
                self.space.riesz(shaped)
                if self.action == "riesz"
                else self.space.inverse_riesz(shaped)
            )
            result = _array_result(image).reshape((-1,))
        else:
            operator = self.operator
            if operator is None:
                raise ValueError("A differential route requires its prepared operator.")
            match self.action:
                case "forward":
                    image = operator.mv(operator.source.unflatten(coordinates))
                case "adjoint":
                    image = operator.adjoint_mv(operator.target.unflatten(coordinates))
                case "dual_forward":
                    shaped = operator.target.unflatten(coordinates)
                    image = operator.transpose_mv(shaped)
                case "dual_adjoint":
                    shaped = operator.source.unflatten(coordinates)
                    primal = operator.source.inverse_riesz(shaped)
                    image = operator.target.riesz(jnp.conj(operator.mv(jnp.conj(primal))))
                case _:
                    raise ValueError(f"Invalid prepared linear action {self.action!r}.")
            result = _array_result(image).reshape((-1,))
        return self.sign * result

    def apply(self, value: Array, /) -> Array:
        coordinates = value if self.source_indices is None else value[self.source_indices]
        action_space = (
            self.space
            if self.space is not None
            else self.operator.source
            if self.operator is not None
            else None
        )
        if action_space is None:
            raise ValueError("A linear route requires a prepared coordinate space.")
        complex_coordinates = jnp.issubdtype(
            action_space.structure().dtype, jnp.complexfloating
        )

        def apply_column(column: Array) -> Array:
            return (
                self._coordinate_action(column)
                if complex_coordinates
                else apply_real_map_componentwise(self._coordinate_action, column)
            )

        if value.ndim == 1:
            result = apply_column(coordinates)
        else:
            columns = coordinates.reshape((coordinates.shape[0], -1))
            image = jax.vmap(apply_column, in_axes=1, out_axes=1)(columns)
            result = image.reshape((image.shape[0], *value.shape[1:]))
        if self.target_indices is None:
            return result
        return (
            jnp.zeros((self.output_size, *value.shape[1:]), dtype=result.dtype)
            .at[self.target_indices]
            .set(result)
        )


@dataclass(frozen=True, slots=True)
class _Carrier:
    complex: HilbertComplex
    realization_id: str
    primal_twist: str
    counts: tuple[int, ...]
    indices: tuple[Array | None, ...] | None

    def coordinate_degree(self, form: FormType, /) -> int:
        return (
            form.degree
            if form.twist == self.primal_twist
            else form.dimension - form.degree
        )

    def coordinate_size(self, form: FormType, /) -> int:
        return self.counts[self.coordinate_degree(form)]

    def selected(self, degree: int, /) -> Array | None:
        return None if self.indices is None else self.indices[degree]


def _carrier(
    realization: AbstractDeRhamComplex, boundary: ComplexBoundary, /
) -> _Carrier:
    complex_ = realization.hilbert_complex(boundary=boundary)
    cochain = (
        realization.cochain
        if isinstance(realization, StructuredCochainBridge)
        else realization
    )
    if isinstance(cochain, CochainDiscretization):
        selections = tuple(
            cochain.active_indices(k, boundary=boundary)
            for k in range(realization.dimension + 1)
        )
        indices = tuple(
            None if selection.size == count else selection
            for selection, count in zip(selections, cochain.cell_counts, strict=True)
        )
        return _Carrier(
            complex_,
            realization.realization_id,
            realization.primal_twist,
            cochain.cell_counts,
            indices,
        )
    return _Carrier(
        complex_,
        realization.realization_id,
        realization.primal_twist,
        tuple(space.size for space in complex_.spaces),
        None,
    )


def _linear_route(
    operation: PDEExpressionOp, form: FormType, carrier: _Carrier, /
) -> _LinearRoute:
    degree = carrier.coordinate_degree(form)
    dual = form.twist != carrier.primal_twist
    complex_ = carrier.complex
    operator: AbstractLinearOperator | None = None
    space: AbstractVectorSpace | None = None
    sign = 1
    target_degree = degree
    if operation in ("exterior_derivative", "gradient", "curl", "divergence"):
        if dual:
            operator = complex_.differential(degree - 1)
            action: _LinearAction = "dual_forward"
            sign = (-1) ** degree
            target_degree -= 1
        else:
            operator = complex_.differential(degree)
            action = "forward"
            target_degree += 1
    elif operation == "codifferential":
        if dual:
            operator = complex_.differential(degree)
            action = "dual_adjoint"
            sign = (-1) ** (degree + 1)
            target_degree += 1
        else:
            operator = complex_.differential(degree - 1)
            action = "adjoint"
            target_degree -= 1
    elif operation == "hodge_star":
        space = complex_.space(degree)
        action = "inverse_riesz" if dual else "riesz"
        if dual:
            sign = (-1) ** (form.degree * degree)
    else:
        raise ValueError(f"Operation {operation!r} is not a linear exterior route.")
    route_id = canonical_fingerprint(
        {
            "kind": "exterior-linear-route",
            "operation": operation,
            "form": form.form_type_id,
            "complex": complex_.complex_id,
            "action": action,
            "operator": None if operator is None else operator.operator_id,
            "space": None if space is None else space.space_id,
        }
    )
    return _LinearRoute(
        operator,
        space,
        carrier.selected(degree),
        carrier.selected(target_degree),
        action,
        carrier.counts[target_degree],
        sign,
        route_id,
    )


@final
class _LaplacianRoute(StrictModule):
    paths: tuple[tuple[_LinearRoute, _LinearRoute], ...]
    route_id: str = eqx.field(static=True)

    def apply(self, value: Array, /) -> Array:
        result = jnp.zeros_like(value)
        for first, second in self.paths:
            result = result - second.apply(first.apply(value))
        return result


def _laplacian_route(form: FormType, carrier: _Carrier, /) -> _LaplacianRoute:
    paths: list[tuple[_LinearRoute, _LinearRoute]] = []
    if form.degree > 0:
        paths.append(
            (
                _linear_route("codifferential", form, carrier),
                _linear_route("exterior_derivative", form.codifferential_type(), carrier),
            )
        )
    if form.degree < form.dimension:
        paths.append(
            (
                _linear_route("exterior_derivative", form, carrier),
                _linear_route("codifferential", form.exterior_derivative_type(), carrier),
            )
        )
    identity = canonical_fingerprint(
        {
            "kind": "exterior-laplacian-route",
            "form": form.form_type_id,
            "paths": [(first.route_id, second.route_id) for first, second in paths],
        }
    )
    return _LaplacianRoute(tuple(paths), identity)


@final
class ExteriorPDERealization(StrictModule):
    """Explicit prepared geometry and named trace routes for a PDE realization."""

    complex: AbstractDeRhamComplex
    products: WhitneyProductPlan | None
    traces: tuple[ComplexMap, ...]
    trace_regions: tuple[str, ...] = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex: AbstractDeRhamComplex,
        /,
        *,
        products: WhitneyProductPlan | None = None,
        traces: Mapping[str, ComplexMap] | None = None,
    ) -> None:
        if products is not None:
            if not isinstance(products, WhitneyProductPlan):
                raise TypeError("products must be a WhitneyProductPlan.")
            if products.complex.realization_id != complex.realization_id:
                raise ValueError("Product routes must belong to the bound realization.")
        named = {} if traces is None else dict(traces)
        source_id = complex.hilbert_complex(boundary="absolute").complex_id
        for region, trace in named.items():
            if not region or not isinstance(trace, ComplexMap):
                raise TypeError(
                    "Trace bindings require nonempty region identifiers and ComplexMaps."
                )
            if (
                trace.degree_offset != 0
                or trace.target.top_degree != trace.source.top_degree - 1
            ):
                raise ValueError(
                    "Trace maps must preserve form degree and lower manifold dimension by one."
                )
        reachable = {source_id}
        pending = tuple(named.values())
        while pending:
            admitted = tuple(
                trace for trace in pending if trace.source.complex_id in reachable
            )
            if not admitted:
                raise ValueError(
                    "Trace routes must be connected to the explicitly bound realization."
                )
            pending = tuple(
                trace for trace in pending if trace.source.complex_id not in reachable
            )
            reachable.update(trace.target.complex_id for trace in admitted)
        regions = tuple(sorted(named))
        trace_values = tuple(named[name] for name in regions)
        identity = canonical_fingerprint(
            {
                "kind": "exterior-pde-realization",
                "realization": complex.realization_id,
                "products": None if products is None else products.plan_id,
                "traces": [
                    (name, trace.map_id)
                    for name, trace in zip(regions, trace_values, strict=True)
                ],
            }
        )
        self.complex = complex
        self.products = products
        self.traces = trace_values
        self.trace_regions = regions
        self.binding_id = identity


@final
class _ProductRoute(StrictModule):
    plan: WhitneyProductPlan
    operation: PDEExpressionOp = eqx.field(static=True)
    left: FormValueSpec | None = eqx.field(static=True)
    right: FormValueSpec = eqx.field(static=True)
    product: FiberProduct = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def apply(self, left: Array, right: Array, /) -> Array:
        if self.operation == "wedge":
            if self.left is None:
                raise ValueError("A wedge route requires its left form identity.")
            return self.plan.wedge(
                left,
                right,
                self.left.form_type.degree,
                self.right.form_type.degree,
                product=self.product,
                left_twist=self.left.form_type.twist,
                right_twist=self.right.form_type.twist,
            )
        if self.operation == "interior_product":
            return self.plan.interior(left, right, self.right.form_type.degree)
        if self.operation == "lie_derivative":
            return self.plan.lie(left, right, self.right.form_type.degree)
        raise ValueError(f"Invalid prepared product route {self.operation!r}.")


@final
class _TraceRoute(StrictModule):
    operator: AbstractLinearOperator
    active_indices: Array | None
    route_id: str = eqx.field(static=True)

    def apply(self, value: Array, /) -> Array:
        if self.active_indices is not None:
            value = (
                jnp.zeros_like(value)
                .at[self.active_indices]
                .set(value[self.active_indices])
            )

        def apply_real_column(column: Array) -> Array:
            return _array_result(
                self.operator.mv(self.operator.source.unflatten(column))
            ).reshape((-1,))

        def apply_column(column: Array) -> Array:
            return apply_real_map_componentwise(apply_real_column, column)

        if value.ndim == 1:
            return apply_column(value)
        columns = value.reshape((value.shape[0], -1))

        image = jax.vmap(apply_column, in_axes=1, out_axes=1)(columns)
        return image.reshape((image.shape[0], *value.shape[1:]))


@final
class _ExpressionNode(StrictModule):
    route: _LinearRoute | _LaplacianRoute | _TraceRoute | None
    product_route: _ProductRoute | None
    value: Array | None
    op: PDEExpressionOp = eqx.field(static=True)
    inputs: tuple[int, ...] = eqx.field(static=True)
    symbol: str | None = eqx.field(static=True)
    axis: int | None = eqx.field(static=True)

    def evaluate(self, values: list[Array], fields: Mapping[str, Array], /) -> Array:
        if self.op == "field":
            if self.symbol is None:
                raise ValueError("A prepared field node requires a symbol.")
            return fields[self.symbol]
        if self.value is not None:
            return self.value
        args = tuple(values[index] for index in self.inputs)
        if self.route is not None:
            return self.route.apply(args[0])
        if self.product_route is not None:
            return self.product_route.apply(args[0], args[1])
        match self.op:
            case "add":
                result = args[0]
                for argument in args[1:]:
                    result = result + argument
                return result
            case "multiply":
                result = args[0]
                for argument in args[1:]:
                    result = result * argument
                return result
            case "divide":
                return args[0] / args[1]
            case "negate":
                return -args[0]
            case "power":
                return args[0] ** args[1]
            case "sin":
                return jnp.sin(args[0])
            case "cos":
                return jnp.cos(args[0])
            case "exp":
                return jnp.exp(args[0])
            case "log":
                return jnp.log(args[0])
            case "sqrt":
                return jnp.sqrt(args[0])
            case "component":
                if self.axis is None:
                    raise ValueError("A prepared component node requires an axis.")
                return args[0][..., self.axis]
            case "dot":
                return jnp.sum(args[0] * args[1], axis=-1)
            case _:
                raise ValueError(f"No prepared route for operation {self.op!r}.")


@final
class CompiledExteriorPDE(StrictModule):
    """Reusable residual DAG with dynamic fields and native operator leaves.

    Runtime mappings must cover the admitted field names exactly and retain their
    admitted shape and dtype. Array values do not participate in binding identity.
    """

    nodes: tuple[_ExpressionNode, ...]
    fields: tuple[Array, ...]
    field_names: tuple[str, ...] = eqx.field(static=True)
    field_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    field_dtypes: tuple[str, ...] = eqx.field(static=True)
    equation_outputs: tuple[tuple[str, int], ...] = eqx.field(static=True)
    condition_outputs: tuple[tuple[str, int], ...] = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    canonical_hash: str = eqx.field(static=True)

    def _evaluate(self, fields: Mapping[str, ArrayLike] | None, /) -> list[Array]:
        if fields is None:
            bound = dict(zip(self.field_names, self.fields, strict=True))
        else:
            if set(fields) != set(self.field_names):
                raise ValueError(
                    "Runtime exterior fields must match the admitted bindings exactly."
                )
            bound = {name: jnp.asarray(fields[name]) for name in self.field_names}
        for name, shape, dtype in zip(
            self.field_names, self.field_shapes, self.field_dtypes, strict=True
        ):
            value = bound[name]
            if value.shape != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"Runtime field {name!r} must retain shape {shape} and dtype {dtype}."
                )
        values: list[Array] = []
        for node in self.nodes:
            values.append(node.evaluate(values, bound))
        return values

    def residuals(
        self, fields: Mapping[str, ArrayLike] | None = None, /
    ) -> dict[str, Array]:
        """Evaluate equation residuals with admitted or replacement numerical fields."""
        values = self._evaluate(fields)
        return {name: values[index] for name, index in self.equation_outputs}

    def condition_residuals(
        self, fields: Mapping[str, ArrayLike] | None = None, /
    ) -> dict[str, Array]:
        """Evaluate explicit initial/boundary/interface residual expressions."""
        values = self._evaluate(fields)
        return {name: values[index] for name, index in self.condition_outputs}


def _admit_fields(
    problem: PDEProblemIR,
    realization: AbstractDeRhamComplex,
    carrier: _Carrier,
    fields: Mapping[str, DiscreteForm | ArrayLike],
    /,
) -> tuple[tuple[str, ...], tuple[Array, ...]]:
    names = tuple(field.name for field in problem.fields)
    if set(fields) != set(names):
        raise ValueError("Exterior field bindings must cover the PDE fields exactly.")
    values: list[Array] = []
    for field in problem.fields:
        supplied = fields[field.name]
        if isinstance(supplied, DiscreteForm):
            if field.form is None:
                raise ValueError("A DiscreteForm binding requires PDE form metadata.")
            if (
                supplied.realization_id != realization.realization_id
                or supplied.form_type.form_type_id != field.form.form_type.form_type_id
            ):
                raise ValueError(
                    f"Field {field.name!r} has an incompatible realization or form identity."
                )
            value = supplied.values
        else:
            value = jnp.asarray(supplied)
        if field.form is not None:
            form = field.form.form_type
            if form.dimension != realization.dimension:
                raise ValueError(
                    f"Field {field.name!r} has an incompatible form dimension."
                )
            expected = (carrier.coordinate_size(form), *form.fiber_shape)
            if value.shape != expected:
                raise ValueError(
                    f"Discrete field {field.name!r} requires coordinate shape {expected}."
                )
            space_dtype = (
                carrier.complex.space(carrier.coordinate_degree(form)).structure().dtype
            )
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                raise TypeError(
                    "Discrete form fields require floating or complex coordinate values."
                )
            actual_dtype = (
                jnp.finfo(value.dtype).dtype
                if not jnp.issubdtype(space_dtype, jnp.complexfloating)
                else value.dtype
            )
            if actual_dtype != space_dtype:
                raise TypeError(
                    f"Discrete field {field.name!r} requires coordinate dtype {space_dtype}."
                )
        elif field.components > 1 and value.shape not in (
            (field.components,),
            (carrier.counts[0], field.components),
        ):
            raise ValueError(
                f"Vector field {field.name!r} requires constant or nodal tangent coordinates."
            )
        elif field.components == 1 and value.shape != ():
            raise ValueError(
                "Discrete scalar coefficient fields require form metadata; untyped coefficients must be scalar constants."
            )
        values.append(value)
    return names, tuple(values)


def _require_same_carriers(
    inputs: tuple[int, ...], carriers: Mapping[int, _Carrier | None], /
) -> None:
    identities = {
        carrier.realization_id
        for index in inputs
        if (carrier := carriers[index]) is not None
    }
    if len(identities) > 1:
        raise ValueError("Form operands must belong to the same explicit realization.")


def _require_carriers(
    inputs: tuple[int, ...],
    carriers: Mapping[int, _Carrier | None],
    expected: _Carrier,
    /,
) -> None:
    _require_same_carriers(inputs, carriers)
    for index in inputs:
        carrier = carriers[index]
        if carrier is not None and carrier.realization_id != expected.realization_id:
            raise ValueError(
                "Product operands require a geometry plan for their realization."
            )


def _prepare_product(
    expression: PDEExpression,
    problem: PDEProblemIR,
    binding: ExteriorPDERealization,
    boundary: ComplexBoundary,
    /,
) -> _ProductRoute:
    plan = binding.products
    if plan is None:
        raise ValueError(
            f"Discrete {expression.op} requires an explicitly prepared WhitneyProductPlan."
        )
    if plan.boundary != boundary:
        raise ValueError("The product plan and PDE must use the same boundary policy.")
    left = infer_expression_type(expression.args[0], problem).form
    right = infer_expression_type(expression.args[1], problem).form
    if right is None:
        raise ValueError("A product route requires its right form identity.")
    if expression.op == "wedge" and left is None:
        raise ValueError("A wedge route requires its left form identity.")
    for spec in (left, right):
        if spec is not None and spec.form_type.twist != binding.complex.primal_twist:
            raise ValueError(
                "Dual-coordinate products require a Whitney geometry plan on the dual complex."
            )
    identity = canonical_fingerprint(
        {
            "kind": "exterior-product-route",
            "operation": expression.op,
            "plan": plan.plan_id,
            "left": None if left is None else left.value_spec_id,
            "right": right.value_spec_id,
            "product": expression.product,
        }
    )
    return _ProductRoute(plan, expression.op, left, right, expression.product, identity)


def _prepare_trace(
    expression: PDEExpression,
    problem: PDEProblemIR,
    binding: ExteriorPDERealization,
    carrier: _Carrier | None,
    /,
) -> tuple[_TraceRoute, _Carrier]:
    region = expression.region
    if region not in binding.trace_regions:
        raise ValueError(
            f"Trace region {region!r} requires an explicitly prepared ComplexMap."
        )
    if carrier is None:
        raise ValueError("A trace requires a form realization binding.")
    trace = binding.traces[binding.trace_regions.index(region)]
    same_source = trace.source.complex_id == carrier.complex.complex_id
    full_root_source = (
        carrier.realization_id == binding.complex.realization_id
        and trace.source.complex_id
        == binding.complex.hilbert_complex(boundary="absolute").complex_id
    )
    if not (same_source or full_root_source):
        raise ValueError("Trace region belongs to a different source realization.")
    spec = infer_expression_type(expression.args[0], problem).form
    if spec is None:
        raise ValueError("A trace requires a form identity.")
    form = spec.form_type
    degree = carrier.coordinate_degree(form)
    if degree != form.degree:
        raise ValueError(
            "A dual-coordinate trace requires an explicit trace map on the dual complex."
        )
    operator = trace.maps[degree]
    if operator.source.size != carrier.coordinate_size(form):
        raise ValueError(
            "Trace routes must use the admitted full or compact source coordinate layout."
        )
    route_id = canonical_fingerprint(
        {
            "kind": "exterior-trace-route",
            "map": trace.map_id,
            "form": form.form_type_id,
            "operator": operator.operator_id,
        }
    )
    route = _TraceRoute(operator, carrier.selected(degree), route_id)
    target = _Carrier(
        trace.target,
        trace.target.complex_id,
        carrier.primal_twist,
        tuple(space.size for space in trace.target.spaces),
        None,
    )
    return route, target


def compile_exterior_pde(
    problem: PDEProblemIR,
    realization: AbstractDeRhamComplex | ExteriorPDERealization,
    /,
    *,
    fields: Mapping[str, DiscreteForm | ArrayLike],
    boundary: ComplexBoundary,
) -> CompiledExteriorPDE:
    """Lower forms to native operators and explicitly prepared P5 geometry routes."""
    if not isinstance(problem, PDEProblemIR):
        raise TypeError("problem must be a PDEProblemIR.")
    binding = (
        realization
        if isinstance(realization, ExteriorPDERealization)
        else ExteriorPDERealization(realization)
    )
    boundary_ = parse(boundary, ComplexBoundary, "boundary")
    validate_pde_ir(problem)
    carrier = _carrier(binding.complex, boundary_)
    field_names, field_values = _admit_fields(problem, binding.complex, carrier, fields)
    nodes: list[_ExpressionNode] = []
    admitted: dict[PDEExpression, int] = {}
    carriers: dict[int, _Carrier | None] = {}

    def lower(expression: PDEExpression) -> int:
        previous = admitted.get(expression)
        if previous is not None:
            return previous
        inputs = tuple(lower(argument) for argument in expression.args)
        typed = infer_expression_type(expression, problem)
        node_carrier = (
            carrier
            if expression.op == "field" and typed.form is not None
            else next(
                (carriers[index] for index in inputs if carriers[index] is not None), None
            )
        )
        route: _LinearRoute | _LaplacianRoute | _TraceRoute | None = None
        product_route: _ProductRoute | None = None
        value: Array | None = None
        op = expression.op
        if op == "constant":
            if expression.value is None:
                raise ValueError("A constant requires its literal value.")
            value = jnp.asarray(float(expression.value))
        elif op == "parameter":
            parameter = next(
                item for item in problem.parameters if item.name == expression.symbol
            )
            if parameter.value is None:
                raise ValueError(
                    f"Parameter {parameter.name!r} requires an admitted value."
                )
            value = jnp.asarray(parameter.value)
        elif op in (
            "exterior_derivative",
            "codifferential",
            "hodge_star",
            "gradient",
            "curl",
            "divergence",
            "laplacian",
        ):
            form = infer_expression_type(expression.args[0], problem).form
            if form is None or node_carrier is None:
                raise ValueError(
                    f"Discrete {op} requires explicit form and realization metadata."
                )
            route = (
                _laplacian_route(form.form_type, node_carrier)
                if op == "laplacian"
                else _linear_route(op, form.form_type, node_carrier)
            )
        elif op in ("wedge", "interior_product", "lie_derivative"):
            product_route = _prepare_product(expression, problem, binding, boundary_)
            _require_carriers(inputs, carriers, carrier)
        elif op == "trace":
            route, node_carrier = _prepare_trace(
                expression, problem, binding, node_carrier
            )
        elif op not in (
            "field",
            "add",
            "multiply",
            "divide",
            "negate",
            "power",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "component",
            "dot",
        ):
            raise ValueError(f"Operation {op!r} requires an exterior realization route.")
        if op == "add":
            _require_same_carriers(inputs, carriers)
        index = len(nodes)
        nodes.append(
            _ExpressionNode(
                route,
                product_route,
                value,
                op,
                inputs,
                expression.symbol,
                expression.axis,
            )
        )
        carriers[index] = node_carrier if typed.form is not None else None
        admitted[expression] = index
        return index

    equations = tuple(
        (equation.name, lower(equation.residual)) for equation in problem.equations
    )
    conditions = tuple(
        (condition.name, lower(condition.residual)) for condition in problem.conditions
    )
    canonical_hash = problem.canonical_hash
    field_layouts = tuple(
        (
            field.name,
            None if field.form is None else field.form.value_spec_id,
            value.shape,
            str(value.dtype),
        )
        for field, value in zip(problem.fields, field_values, strict=True)
    )
    compilation_id = canonical_fingerprint(
        {
            "kind": "compiled-exterior-pde",
            "problem": canonical_hash,
            "binding": binding.binding_id,
            "complex": carrier.complex.complex_id,
            "boundary": boundary_,
            "fields": field_layouts,
            "routes": [node.route.route_id for node in nodes if node.route is not None]
            + [
                node.product_route.route_id
                for node in nodes
                if node.product_route is not None
            ],
        }
    )
    return CompiledExteriorPDE(
        tuple(nodes),
        field_values,
        field_names,
        tuple(value.shape for value in field_values),
        tuple(str(value.dtype) for value in field_values),
        equations,
        conditions,
        boundary_,
        binding.complex.realization_id,
        compilation_id,
        canonical_hash,
    )


__all__ = ["CompiledExteriorPDE", "ExteriorPDERealization", "compile_exterior_pde"]
