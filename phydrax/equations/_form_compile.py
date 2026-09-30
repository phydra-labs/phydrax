"""Explicit chart, metric, and trace bindings for smooth exterior PDE lowering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, final

import equinox as eqx
from jax import Array

from .._strict import StrictModule
from ..domain import DomainFunction
from ..domain._derivative import (
    CallbackDerivativeRule,
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
)
from ..exterior._algebra import form_to_vector
from ..exterior._form_type import FormType, FormValueSpec
from ..metrix._chart import CoordinateChart
from ..metrix._metric import AbstractSemiRiemannianMetric
from ..operators.differential._form_ops import (
    domain_codifferential,
    domain_exterior_derivative,
    domain_hodge_laplacian,
    domain_hodge_star,
    domain_interior_product,
    domain_lie_derivative,
    domain_pullback_form,
    domain_wedge,
    DomainDifferentialForm,
)
from ._ir import PDEExpression


@final
class _DomainFormProxyCallable(StrictModule):
    coefficients: DomainFunction
    spec: FormValueSpec = eqx.field(static=True)

    def __call__(self, *args: Any, key: Any = None, **kwargs: Any) -> Array:
        return form_to_vector(self.coefficients.func(*args, key=key, **kwargs), self.spec)

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule | None:
        del function
        rule = self.coefficients.derivative_rule
        if rule is None:
            return None

        def derive(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode: DerivativeMode,
            backend: DerivativeBackend,
            basis: DerivativeBasis,
            periodic: bool,
        ) -> DomainFunction | None:
            derivative = rule.derive(
                var=var,
                axis=axis,
                order=order,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )
            if derivative is None:
                return None
            coordinate = self.coefficients.domain.coordinate(var)
            suffix: tuple[int, ...] = ()
            if axis is None and coordinate.kind == "array":
                shape = coordinate.event_shape
                if shape is None or len(shape) != 1:
                    raise ValueError(
                        "Proxy differentiation requires a rank-one coordinate."
                    )
                suffix = shape * order
            source = self.spec.form_type
            differentiated_type = FormType(
                source.dimension,
                source.degree,
                twist=source.twist,
                fiber_shape=(*source.fiber_shape, *suffix),
                ambient_dimension=source.ambient_dimension,
            )
            differentiated_spec = FormValueSpec(
                differentiated_type, proxy=self.spec.proxy
            )
            return domain_form_proxy(derivative, differentiated_spec)

        return CallbackDerivativeRule(derive)


def domain_form_proxy(
    coefficients: DomainFunction, spec: FormValueSpec, /
) -> DomainFunction:
    """Apply an explicit linear proxy view while preserving admitted derivative rules."""
    return DomainFunction(
        domain=coefficients.domain,
        deps=coefficients.deps,
        func=_DomainFormProxyCallable(coefficients, spec),
        metadata=coefficients.metadata,
    )


@final
@dataclass(frozen=True, slots=True)
class PDEFormTrace:
    """A declared hypersurface map; coorientation trivializes its orientation line."""

    mapping: DomainFunction
    source_chart: CoordinateChart
    source_var: str | None = None
    coorientation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, DomainFunction):
            raise TypeError("A PDE trace map must be a DomainFunction.")
        if not isinstance(self.source_chart, CoordinateChart):
            raise TypeError("A PDE trace requires a CoordinateChart.")
        if self.coorientation is not None and self.coorientation not in (-1, 1):
            raise ValueError("PDE trace coorientation must be +1 or -1.")


@final
@dataclass(frozen=True, slots=True)
class PDEFormGeometry:
    """Metric and region maps admitted explicitly at smooth lowering time."""

    metric: AbstractSemiRiemannianMetric | None = None
    traces: Mapping[str, PDEFormTrace] | None = None

    def __post_init__(self) -> None:
        if self.metric is not None and not isinstance(
            self.metric, AbstractSemiRiemannianMetric
        ):
            raise TypeError("PDE form metric must be an AbstractSemiRiemannianMetric.")
        if self.traces is not None and any(
            not isinstance(value, PDEFormTrace) for value in self.traces.values()
        ):
            raise TypeError("PDE form trace bindings must be PDEFormTrace objects.")


def lower_domain_form_operation(
    node: PDEExpression,
    forms: tuple[DomainDifferentialForm, ...],
    /,
    *,
    vector: DomainFunction | None,
    geometry: PDEFormGeometry | None,
    backend: str,
) -> DomainDifferentialForm:
    """Lower exterior operations without inventing metric or boundary geometry."""
    from ..typing import parse
    from ._compile import DifferentialBackend

    selected_backend = parse(backend, DifferentialBackend, "differential_backend")
    form = forms[-1] if node.op in ("interior_product", "lie_derivative") else forms[0]
    if node.coordinate is not None and node.coordinate != form.var:
        raise ValueError("Exterior coordinate must match the form's chart variable.")
    match node.op:
        case "exterior_derivative" | "gradient" | "curl" | "divergence":
            return domain_exterior_derivative(form, backend=selected_backend)
        case "codifferential" | "hodge_star" | "laplacian":
            if geometry is None or geometry.metric is None:
                raise ValueError(f"{node.op} requires an explicit metric binding.")
            match node.op:
                case "codifferential":
                    return domain_codifferential(
                        form, geometry.metric, backend=selected_backend
                    )
                case "hodge_star":
                    return domain_hodge_star(form, geometry.metric)
                case "laplacian":
                    result = domain_hodge_laplacian(
                        form, geometry.metric, backend=selected_backend
                    )
                    return DomainDifferentialForm(
                        -result.coefficients,
                        chart=result.chart,
                        degree=result.degree,
                        twist=result.twist,
                        fiber_shape=result.form_type.fiber_shape,
                        var=result.var,
                    )
        case "wedge":
            return domain_wedge(forms[0], forms[1], product=node.product)
        case "interior_product" | "lie_derivative":
            if vector is None:
                raise TypeError("Contraction requires a DomainFunction tangent vector.")
            if node.op == "interior_product":
                return domain_interior_product(vector, form)
            return domain_lie_derivative(vector, form, backend=selected_backend)
        case "trace":
            if (
                geometry is None
                or geometry.traces is None
                or node.region not in geometry.traces
            ):
                raise ValueError("trace requires an explicit region-map binding.")
            if node.region is None:
                raise ValueError("A compiled trace requires a declared region.")
            binding = geometry.traces[node.region]
            return domain_pullback_form(
                form,
                binding.mapping,
                source_chart=binding.source_chart,
                source_var=binding.source_var,
                coorientation=binding.coorientation,
                backend=selected_backend,
            )
        case _:
            raise ValueError(f"Unsupported smooth exterior operation {node.op!r}.")
    raise ValueError(f"Unsupported smooth exterior operation {node.op!r}.")


__all__ = ["PDEFormGeometry", "PDEFormTrace"]
