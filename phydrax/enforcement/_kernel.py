#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""RKHS minimum-norm correction providers for affine condition realization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..conditions import ArrayCodomain, BoundCondition, FieldCodomain, ProductCodomain
from ..domain import Domain, DomainFunction
from ..integration._linear import PreparedLinearReduction
from ..kernels._field_metric import (
    kernel_functional_representer,
    KernelFunctional,
    KernelFunctionalTerm,
    ProductFieldKernelMetric,
)
from ..linalg._constraint_operators import (
    ConstraintOperatorPlan,
    PreparedConstraintOperator,
)
from ..linalg._operators import DenseLinearOperator, FunctionLinearOperator
from ..linalg._policies import GMRES, LinearSolvePolicy
from ..linalg._problems import LinearSystem
from ..linalg._properties import OperatorProperties
from ..linalg._runtime import (
    prepare as prepare_linear_solve,
    solve as solve_linear_system,
)
from ..linalg._spaces import ArraySpace
from ._affine import (
    AbstractLinearCorrectionProvider,
    AffineBlockAssembly,
    AffineProjectionPolicy,
    LinearCorrectionEvidence,
    PreparedLinearCorrection,
)


KernelCorrectionRepresentation = Literal[
    "canonical", "finite-feature", "selected-section", "matrix-free"
]
KernelRepresenterExactness = Literal[
    "analytic", "finite-feature", "fixed-realization", "selected-section"
]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _functional_fingerprint(functional: KernelFunctional, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "kernel-functional-v1",
            "functional_id": functional.functional_id,
            "exactness": functional.exactness,
            "realization_id": functional.realization_id,
            "terms": tuple(
                {
                    "field": term.field_name,
                    "points": array_tree_fingerprint(term.points),
                    "orders": term.derivative_orders,
                    "coefficients": array_tree_fingerprint(term.coefficients),
                }
                for term in functional.terms
            ),
        }
    )


def _joint_exactness(
    functionals: Sequence[KernelFunctional], /
) -> KernelRepresenterExactness:
    values = {functional.exactness for functional in functionals}
    if "selected-section" in values:
        return "selected-section"
    if "fixed-realization" in values:
        return "fixed-realization"
    if "finite-feature" in values:
        return "finite-feature"
    return "analytic"


def _codomain_size(codomain: Any, /) -> int:
    if isinstance(codomain, ArrayCodomain):
        return prod(codomain.shape) if codomain.shape else 1
    if isinstance(codomain, ProductCodomain):
        return sum(_codomain_size(factor) for factor in codomain.factors)
    raise TypeError("Kernel correction residuals require finite array codomains.")


def _flatten_codomain(codomain: Any, value: Any, /) -> Array:
    if isinstance(codomain, ArrayCodomain):
        array = jnp.asarray(value)
        if array.shape != codomain.shape:
            raise ValueError(
                f"Kernel residual shape {array.shape} does not match {codomain.shape}."
            )
        return array.reshape((-1,))
    if isinstance(codomain, ProductCodomain):
        if not isinstance(value, tuple) or len(value) != len(codomain.factors):
            raise ValueError("Kernel product residual does not match its codomain.")
        return jnp.concatenate(
            tuple(
                _flatten_codomain(factor, leaf)
                for factor, leaf in zip(codomain.factors, value, strict=True)
            )
        )
    raise TypeError("Kernel correction residuals require finite array codomains.")


def _flatten_residual(codomains: Sequence[Any], value: Any, /) -> Array:
    if not isinstance(value, tuple) or len(value) != len(codomains):
        raise ValueError("Kernel residual must preserve the condition product layout.")
    blocks = tuple(
        _flatten_codomain(codomain, block)
        for codomain, block in zip(codomains, value, strict=True)
    )
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)


def _stack_functionals(
    functionals: Sequence[KernelFunctional],
    /,
    *,
    functional_id: str,
) -> KernelFunctional:
    values = tuple(functionals)
    if not values:
        raise ValueError("At least one kernel functional is required.")
    if len(values) == 1:
        return values[0]
    row_count = sum(value.row_count for value in values)
    terms: list[KernelFunctionalTerm] = []
    offset = 0
    for value in values:
        trailing = row_count - offset - value.row_count
        for term in value.terms:
            coefficients = jnp.pad(
                term.coefficients,
                ((offset, trailing), (0, 0), (0, 0), (0, 0)),
            )
            terms.append(
                KernelFunctionalTerm(
                    term.field_name,
                    term.points,
                    term.derivative_orders,
                    coefficients,
                )
            )
        offset += value.row_count
    realization_ids = {
        value.realization_id for value in values if value.realization_id is not None
    }
    exactness = _joint_exactness(values)
    if exactness == "fixed-realization" and len(realization_ids) != 1:
        raise ValueError(
            "Stacked fixed-realization functionals must share one realization_id."
        )
    return KernelFunctional(
        terms,
        functional_id=functional_id,
        exactness=exactness,
        realization_id=next(iter(realization_ids)) if realization_ids else None,
    )


class PointKernelRepresenter(StrictModule):
    """Finite point-evaluation rows and their canonical RKHS representers."""

    points: Array
    covectors: Array | None
    field_name: str = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        points: ArrayLike,
        covectors: ArrayLike | None = None,
        /,
        *,
        functional_id: str = "point",
    ):
        self.field_name = _identifier(field_name, "field_name")
        self.functional_id = _identifier(functional_id, "functional_id")
        self.points = jnp.asarray(points)
        self.covectors = None if covectors is None else jnp.asarray(covectors)

    def functional(self, metric: ProductFieldKernelMetric, /) -> KernelFunctional:
        return metric.point_functional(
            self.field_name,
            self.points,
            self.covectors,
            functional_id=self.functional_id,
        )

    def section(self, metric: ProductFieldKernelMetric, field_name: str, /):
        return metric.representer(self.functional(metric), field_name)


class JetKernelRepresenter(StrictModule):
    """Finite derivative-evaluation rows and their canonical RKHS representers."""

    points: Array
    coefficients: Array
    field_name: str = eqx.field(static=True)
    derivative_orders: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        points: ArrayLike,
        derivative_orders: Sequence[Sequence[int]],
        coefficients: ArrayLike,
        /,
        *,
        functional_id: str = "jet",
    ):
        self.field_name = _identifier(field_name, "field_name")
        self.functional_id = _identifier(functional_id, "functional_id")
        self.points = jnp.asarray(points)
        self.derivative_orders = tuple(
            tuple(int(value) for value in order) for order in derivative_orders
        )
        self.coefficients = jnp.asarray(coefficients)

    def functional(self, metric: ProductFieldKernelMetric, /) -> KernelFunctional:
        return metric.jet_functional(
            self.field_name,
            self.points,
            self.derivative_orders,
            self.coefficients,
            functional_id=self.functional_id,
        )

    def section(self, metric: ProductFieldKernelMetric, field_name: str, /):
        return metric.representer(self.functional(metric), field_name)


class IntegralKernelRepresenter(StrictModule):
    """Rows induced by one frozen ``PreparedLinearReduction``.

    The reduction's already-composed coefficient field is used directly.  This
    is exact for that immutable realization only; it is never advertised as a
    continuum integral representer.
    """

    points: Array
    covectors: Array | None
    reduction: PreparedLinearReduction
    field_name: str = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        points: ArrayLike,
        reduction: PreparedLinearReduction,
        covectors: ArrayLike | None = None,
        /,
        *,
        functional_id: str = "integral",
    ):
        if not isinstance(reduction, PreparedLinearReduction):
            raise TypeError("reduction must be a PreparedLinearReduction.")
        if len(reduction.coefficient_fields) != 1 or reduction.schema.retained_axes:
            raise ValueError(
                "Kernel integral representers require one scalar, fully reduced "
                "coefficient field."
            )
        points_ = jnp.asarray(points)
        coefficient_count = int(reduction.coefficient_fields[0].data.size)
        if points_.ndim < 2 or int(points_.shape[0]) != coefficient_count:
            raise ValueError(
                "Integral representer points must align exactly with frozen coefficients."
            )
        self.field_name = _identifier(field_name, "field_name")
        self.functional_id = _identifier(functional_id, "functional_id")
        self.points = points_
        self.covectors = None if covectors is None else jnp.asarray(covectors)
        self.reduction = reduction

    @property
    def weights(self) -> Array:
        return jnp.asarray(self.reduction.coefficient_fields[0].data).reshape((-1,))

    def functional(self, metric: ProductFieldKernelMetric, /) -> KernelFunctional:
        return metric.integral_functional(
            self.field_name,
            self.points,
            self.weights,
            self.covectors,
            functional_id=self.functional_id,
            realization_id=self.reduction.realization_id,
        )

    def section(self, metric: ProductFieldKernelMetric, field_name: str, /):
        return metric.representer(self.functional(metric), field_name)


KernelRepresenter: TypeAlias = (
    PointKernelRepresenter | JetKernelRepresenter | IntegralKernelRepresenter
)


def _resolve_functional(
    metric: ProductFieldKernelMetric,
    value: KernelFunctional | KernelRepresenter | Sequence[KernelRepresenter],
    /,
    *,
    functional_id: str,
) -> KernelFunctional:
    if isinstance(value, KernelFunctional):
        return value
    if isinstance(
        value, (PointKernelRepresenter, JetKernelRepresenter, IntegralKernelRepresenter)
    ):
        return value.functional(metric)
    representers = tuple(value)
    if not representers or any(
        not isinstance(
            item,
            (PointKernelRepresenter, JetKernelRepresenter, IntegralKernelRepresenter),
        )
        for item in representers
    ):
        raise TypeError("Expected kernel functionals or typed kernel representers.")
    return _stack_functionals(
        tuple(item.functional(metric) for item in representers),
        functional_id=functional_id,
    )


class KernelCorrectionEvidence(StrictModule):
    """Preparation, rank, exactness, and no-jitter evidence for one correction."""

    numeric_version: Array
    finite: Array
    hermitian_residual: Array
    minimum_diagonal: Array
    rank: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    representation: KernelCorrectionRepresentation = eqx.field(static=True)
    exactness_scope: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    no_jitter: bool = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        numeric_version: int,
        finite: Any,
        hermitian_residual: Any,
        minimum_diagonal: Any,
        rank: int,
        row_count: int,
        coefficient_count: int,
        representation: KernelCorrectionRepresentation,
        exactness_scope: str,
        exact: bool,
        provider_id: str,
        preparation_id: str,
        functional_id: str,
        metric_id: str,
        realization_id: str | None,
    ):
        if representation not in (
            "canonical",
            "finite-feature",
            "selected-section",
            "matrix-free",
        ):
            raise ValueError("Unknown kernel correction representation.")
        version = int(numeric_version)
        rank_ = int(rank)
        rows = int(row_count)
        coefficients = int(coefficient_count)
        if version < 0 or rank_ < 0 or rows <= 0 or coefficients <= 0:
            raise ValueError("Kernel correction dimensions and versions are invalid.")
        self.numeric_version = jnp.asarray(version, dtype=jnp.int32)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.hermitian_residual = jnp.asarray(hermitian_residual)
        self.minimum_diagonal = jnp.asarray(minimum_diagonal)
        self.rank = rank_
        self.row_count = rows
        self.coefficient_count = coefficients
        self.representation = representation
        self.exactness_scope = str(exactness_scope)
        self.exact = bool(exact)
        self.no_jitter = True
        self.provider_id = _identifier(provider_id, "provider_id")
        self.preparation_id = _identifier(preparation_id, "preparation_id")
        self.functional_id = _identifier(functional_id, "functional_id")
        self.metric_id = _identifier(metric_id, "metric_id")
        self.realization_id = realization_id


class _KernelFieldEvaluator(StrictModule):
    metric: ProductFieldKernelMetric
    functional: KernelFunctional | None
    coefficients: Array
    input_shape: tuple[int, ...] = eqx.field(static=True)
    field_shape: tuple[int, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    representation: KernelCorrectionRepresentation = eqx.field(static=True)

    def _point(self, args: tuple[Any, ...], /) -> Array:
        if any(isinstance(value, tuple) for value in args):
            raise TypeError(
                "Kernel corrections do not support coordinate-separable direct evaluation."
            )
        flattened = tuple(jnp.asarray(value).reshape((-1,)) for value in args)
        point = flattened[0] if len(flattened) == 1 else jnp.concatenate(flattened)
        if int(point.size) != prod(self.input_shape):
            raise ValueError(
                f"Kernel query has {point.size} coordinates; expected {prod(self.input_shape)}."
            )
        return point.reshape(self.input_shape)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        point = self._point(args)
        if self.representation == "finite-feature":
            basis = self.metric.field_features(self.field_name, point[None, ...])[0]
        else:
            if self.functional is None:
                raise RuntimeError("Kernel section evaluator lost its functional.")
            basis = kernel_functional_representer(
                self.metric,
                self.functional,
                self.field_name,
                point[None, ...],
            )[0]
        value = oe.contract("fr,r->f", basis, self.coefficients)
        return value.reshape(self.field_shape) if self.field_shape else value[0]


class _KernelLiftAction(StrictModule):
    metric: ProductFieldKernelMetric
    basis_functional: KernelFunctional | None
    direct_operator: PreparedConstraintOperator | None
    condition_codomains: tuple[Any, ...]
    metric_fields: tuple[str, ...] = eqx.field(static=True)
    correction_fields: tuple[str, ...] = eqx.field(static=True)
    domains: tuple[Domain, ...]
    input_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    representation: KernelCorrectionRepresentation = eqx.field(static=True)
    check_compatibility: bool = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    exactness_scope: str = eqx.field(static=True)

    def _coefficients(self, product_residual: Any, /) -> Array:
        residual = _flatten_residual(self.condition_codomains, product_residual)
        if self.direct_operator is not None:
            return jnp.asarray(
                self.direct_operator.minimum_norm_lift(
                    residual,
                    check_compatibility=self.check_compatibility,
                )
            )
        result = solve_linear_system(self.prepared_solve, residual)
        return eqx.error_if(
            jnp.asarray(result.value),
            ~result.successful,
            "Matrix-free kernel Gram solve did not converge.",
        )

    def __call__(self, product_residual: Any, /) -> tuple[DomainFunction, ...]:
        coefficients = self._coefficients(product_residual)
        corrections = []
        for field_name, metric_field, domain, input_shape in zip(
            self.correction_fields,
            self.metric_fields,
            self.domains,
            self.input_shapes,
            strict=True,
        ):
            field_index = self.metric.field_names.index(metric_field)
            field_shape = self.metric.field_shapes[field_index]
            evaluator = _KernelFieldEvaluator(
                metric=self.metric,
                functional=self.basis_functional,
                coefficients=coefficients,
                input_shape=input_shape,
                field_shape=field_shape,
                field_name=metric_field,
                representation=self.representation,
            )
            corrections.append(
                DomainFunction(
                    domain=domain,
                    deps=domain.labels,
                    func=evaluator,
                    metadata={
                        "kernel_correction": True,
                        "provider_id": self.provider_id,
                        "representation": self.representation,
                        "exactness_scope": self.exactness_scope,
                        "no_jitter": True,
                    },
                )
            )
        return tuple(corrections)


class _BaseKernelCorrectionPlan(StrictModule):
    __strict_abstract__ = True
    metric: ProductFieldKernelMetric
    functional: KernelFunctional
    provider_id: str = eqx.field(static=True)
    representation: KernelCorrectionRepresentation = eqx.field(static=True)

    def __init__(
        self,
        metric: ProductFieldKernelMetric,
        functional: KernelFunctional | KernelRepresenter | Sequence[KernelRepresenter],
        /,
        *,
        representation: KernelCorrectionRepresentation,
    ):
        if not isinstance(metric, ProductFieldKernelMetric):
            raise TypeError("metric must be a ProductFieldKernelMetric.")
        resolved = _resolve_functional(
            metric,
            functional,
            functional_id="stacked-kernel-functional",
        )
        for term in resolved.terms:
            metric._validate_term(term)
        self.metric = metric
        self.functional = resolved
        self.representation = representation
        self.provider_id = canonical_fingerprint(
            {
                "kind": "kernel-correction-plan-v1",
                "representation": representation,
                "metric": metric.metric_id,
                "functional": _functional_fingerprint(resolved),
            }
        )

    def _assembly_data(
        self,
        assembly: AffineBlockAssembly,
        correction_fields: Sequence[str],
        /,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[Domain, ...],
        tuple[tuple[int, ...], ...],
    ]:
        fields = tuple(str(name) for name in correction_fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("correction_fields must be nonempty and unique.")
        if fields != tuple(assembly.correction_fields):
            raise ValueError(
                "Kernel correction fields must preserve affine assembly order."
            )
        if (
            sum(_codomain_size(codomain) for codomain in assembly.condition_codomains)
            != self.functional.row_count
        ):
            raise ValueError(
                "Kernel functional rows must match the finite affine condition "
                "codomains exactly."
            )
        domains: list[Domain] = []
        metric_fields: list[str] = []
        for name in fields:
            matches = tuple(
                (index, field)
                for index, field in enumerate(self.metric.field_spec.fields)
                if field.source == name
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Correction source {name!r} must occur exactly once in the kernel metric."
                )
            index, field = matches[0]
            codomain = field.codomain
            if not isinstance(codomain, FieldCodomain):
                raise TypeError("Kernel correction fields require FieldCodomain entries.")
            metric_fields.append(field.name)
            domains.append(codomain.support.domain)
        shapes_by_field: dict[str, tuple[int, ...]] = {}
        for term in self.functional.terms:
            shape = tuple(int(size) for size in term.points.shape[1:])
            previous = shapes_by_field.setdefault(term.field_name, shape)
            if previous != shape:
                raise ValueError("One field cannot mix physical kernel input shapes.")
        fallback = tuple(int(size) for size in self.functional.terms[0].points.shape[1:])
        input_shapes = tuple(
            shapes_by_field.get(name, fallback) for name in metric_fields
        )
        return fields, tuple(metric_fields), tuple(domains), input_shapes

    def _prepared(
        self,
        *,
        assembly: AffineBlockAssembly,
        correction_fields: Sequence[str],
        numeric_version: int,
        representation: KernelCorrectionRepresentation,
        basis_functional: KernelFunctional | None,
        direct_operator: PreparedConstraintOperator | None,
        prepared_solve: Any = None,
        rank: int,
        coefficient_count: int,
        finite: Any,
        hermitian_residual: Any,
        minimum_diagonal: Any,
        exact: bool,
        check_compatibility: bool,
        exactness_scope: str,
        identity_defect: Any | None = None,
        range_defect: Any | None = None,
    ) -> PreparedLinearCorrection:
        fields, metric_fields, domains, input_shapes = self._assembly_data(
            assembly, correction_fields
        )
        preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-kernel-correction-v1",
                "provider": self.provider_id,
                "assembly": assembly.assembly_id,
                "representation": representation,
                "numeric_version": int(numeric_version),
            }
        )
        evidence = KernelCorrectionEvidence(
            numeric_version=numeric_version,
            finite=finite,
            hermitian_residual=hermitian_residual,
            minimum_diagonal=minimum_diagonal,
            rank=rank,
            row_count=self.functional.row_count,
            coefficient_count=coefficient_count,
            representation=representation,
            exactness_scope=exactness_scope,
            exact=exact,
            provider_id=self.provider_id,
            preparation_id=preparation_id,
            functional_id=self.functional.functional_id,
            metric_id=self.metric.metric_id,
            realization_id=self.functional.realization_id,
        )
        lift = _KernelLiftAction(
            metric=self.metric,
            basis_functional=basis_functional,
            direct_operator=direct_operator,
            condition_codomains=assembly.condition_codomains,
            metric_fields=metric_fields,
            correction_fields=fields,
            domains=domains,
            input_shapes=input_shapes,
            representation=representation,
            check_compatibility=check_compatibility,
            provider_id=self.provider_id,
            exactness_scope=exactness_scope,
        )
        linear_evidence = LinearCorrectionEvidence(
            provider_id=self.provider_id,
            preparation_id=preparation_id,
            condition_ids=assembly.condition_ids,
            field_names=fields,
            exactness_scope=exactness_scope,
            generalized=not check_compatibility,
            numeric_version=numeric_version,
            identity_defect=(
                (
                    jnp.inf
                    if direct_operator is None
                    else direct_operator.evidence.strict_right_inverse_residual_norm
                )
                if identity_defect is None
                else identity_defect
            ),
            range_defect=(
                (
                    jnp.inf
                    if direct_operator is None
                    else direct_operator.evidence.generalized_right_inverse_residual_norm
                )
                if range_defect is None
                else range_defect
            ),
            rank=rank,
            nullity=max(0, coefficient_count - rank),
            solve_evidence=evidence,
        )
        return PreparedLinearCorrection(
            lift,
            field_names=fields,
            provider_id=self.provider_id,
            preparation_id=preparation_id,
            numeric_version=numeric_version,
            exactness_scope=exactness_scope,
            generalized=not check_compatibility,
            evidence=linear_evidence,
            operator=direct_operator,
        )


class CanonicalKernelCorrectionPlan(_BaseKernelCorrectionPlan):
    """Exact canonical minimum-RKHS-norm correction with a dense Gram factor."""

    def __init__(self, metric, functional, /):
        super().__init__(metric, functional, representation="canonical")

    def prepare(
        self,
        assembly,
        correction_fields,
        /,
        *,
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ):
        gram = self.metric.functional_gram(self.functional)
        operator = DenseLinearOperator(
            gram.matrix,
            operator_id=f"kernel-gram/{self.provider_id}",
        )
        prepared = ConstraintOperatorPlan(operator, require_full_row_rank=True).prepare()
        scope = policy.exactness_scope
        return self._prepared(
            assembly=assembly,
            correction_fields=correction_fields,
            numeric_version=numeric_version,
            representation="canonical",
            basis_functional=self.functional,
            direct_operator=prepared,
            rank=prepared.rank,
            coefficient_count=self.functional.row_count,
            finite=gram.evidence.finite,
            hermitian_residual=gram.evidence.hermitian_residual,
            minimum_diagonal=gram.evidence.minimum_diagonal,
            exact=True,
            check_compatibility=policy.compatibility == "strict",
            exactness_scope=scope,
        )


class FiniteFeatureKernelCorrectionPlan(_BaseKernelCorrectionPlan):
    """Exact weight-space correction for a certified finite-feature metric."""

    def __init__(self, metric, functional, /):
        super().__init__(metric, functional, representation="finite-feature")
        if metric.feature_rank() is None:
            raise TypeError("Finite-feature correction requires an exact feature map.")

    def prepare(
        self,
        assembly,
        correction_fields,
        /,
        *,
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ):
        features = self.metric.functional_features(self.functional)
        operator = DenseLinearOperator(
            features,
            operator_id=f"kernel-features/{self.provider_id}",
        )
        prepared = ConstraintOperatorPlan(operator, require_full_row_rank=True).prepare()
        gram = features @ jnp.conj(features.T)
        skew = gram - jnp.conj(gram.T)
        scope = policy.exactness_scope
        return self._prepared(
            assembly=assembly,
            correction_fields=correction_fields,
            numeric_version=numeric_version,
            representation="finite-feature",
            basis_functional=None,
            direct_operator=prepared,
            rank=prepared.rank,
            coefficient_count=int(features.shape[1]),
            finite=jnp.all(jnp.isfinite(features)),
            hermitian_residual=jnp.max(jnp.abs(skew), initial=0.0),
            minimum_diagonal=jnp.min(jnp.real(jnp.diag(gram)), initial=jnp.inf),
            exact=True,
            check_compatibility=policy.compatibility == "strict",
            exactness_scope=scope,
        )


class SectionKernelCorrectionPlan(_BaseKernelCorrectionPlan):
    """Correction restricted to declared sections; approximate unless surjective."""

    sections: KernelFunctional
    require_exact: bool = eqx.field(static=True)

    def __init__(self, metric, functional, sections, /, *, require_exact: bool = False):
        super().__init__(metric, functional, representation="selected-section")
        self.sections = _resolve_functional(
            metric,
            sections,
            functional_id="selected-kernel-sections",
        )
        for term in self.sections.terms:
            metric._validate_term(term)
        self.require_exact = bool(require_exact)
        self.provider_id = canonical_fingerprint(
            {
                "kind": "selected-section-kernel-correction-v1",
                "metric": metric.metric_id,
                "functional": _functional_fingerprint(self.functional),
                "sections": _functional_fingerprint(self.sections),
                "require_exact": self.require_exact,
            }
        )

    def prepare(
        self,
        assembly,
        correction_fields,
        /,
        *,
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ):
        cross = self.metric.functional_gram(self.functional, self.sections).matrix
        operator = DenseLinearOperator(
            cross,
            operator_id=f"kernel-sections/{self.provider_id}",
        )
        prepared = ConstraintOperatorPlan(
            operator,
            require_full_row_rank=policy.compatibility == "strict",
        ).prepare()
        exact = self.require_exact and prepared.evidence.full_row_rank
        return self._prepared(
            assembly=assembly,
            correction_fields=correction_fields,
            numeric_version=numeric_version,
            representation="selected-section",
            basis_functional=self.sections,
            direct_operator=prepared,
            rank=prepared.rank,
            coefficient_count=self.sections.row_count,
            finite=jnp.all(jnp.isfinite(cross)),
            hermitian_residual=jnp.asarray(jnp.nan),
            minimum_diagonal=jnp.asarray(jnp.nan),
            exact=exact,
            check_compatibility=policy.compatibility == "strict",
            exactness_scope=policy.exactness_scope,
        )


class MatrixFreeKernelCorrectionPlan(_BaseKernelCorrectionPlan):
    """Iterative Gram correction with no retained dense Gram matrix.

    The operator action recomputes the exact canonical Gram action.  The returned
    correction is nevertheless approximate because a tolerance-terminated linear
    solve is part of every lift.
    """

    solve_policy: LinearSolvePolicy

    def __init__(
        self, metric, functional, /, *, solve_policy: LinearSolvePolicy | None = None
    ):
        super().__init__(metric, functional, representation="matrix-free")
        policy = LinearSolvePolicy(GMRES()) if solve_policy is None else solve_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
        self.solve_policy = policy

    def prepare(
        self,
        assembly,
        correction_fields,
        /,
        *,
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ):
        dtype = jnp.result_type(*(term.coefficients for term in self.functional.terms))
        space = ArraySpace((self.functional.row_count,), dtype=dtype)

        def action(vector):
            return self.metric.functional_gram(self.functional).matrix @ vector

        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
            operator_id=f"matrix-free-kernel-gram/{self.provider_id}",
        )
        prepared = prepare_linear_solve(LinearSystem(operator), self.solve_policy)
        identity = jnp.eye(self.functional.row_count, dtype=dtype)
        probe = solve_linear_system(prepared, identity)
        inverse = eqx.error_if(
            jnp.asarray(probe.value),
            ~jnp.all(probe.successful),
            "Matrix-free kernel Gram preparation did not converge.",
        )
        defect = jnp.max(jnp.abs(operator.mv(inverse) - identity), initial=0.0)
        diagonal = self.metric.functional_gram(self.functional).matrix.diagonal()
        return self._prepared(
            assembly=assembly,
            correction_fields=correction_fields,
            numeric_version=numeric_version,
            representation="matrix-free",
            basis_functional=self.functional,
            direct_operator=None,
            prepared_solve=prepared,
            rank=self.functional.row_count,
            coefficient_count=self.functional.row_count,
            finite=jnp.all(jnp.isfinite(diagonal)) & jnp.all(jnp.isfinite(inverse)),
            hermitian_residual=jnp.asarray(jnp.nan),
            minimum_diagonal=jnp.min(jnp.real(diagonal), initial=jnp.inf),
            exact=False,
            check_compatibility=False,
            exactness_scope=policy.exactness_scope,
            identity_defect=defect,
            range_defect=defect,
        )


KernelCorrectionPlan: TypeAlias = (
    CanonicalKernelCorrectionPlan
    | FiniteFeatureKernelCorrectionPlan
    | SectionKernelCorrectionPlan
    | MatrixFreeKernelCorrectionPlan
)


class _KernelCorrectionProvider(AbstractLinearCorrectionProvider):
    __strict_abstract__ = True
    plan: KernelCorrectionPlan

    @property
    def provider_id(self) -> str:
        return self.plan.provider_id

    def prepare(
        self,
        bound_conditions: Sequence[BoundCondition],
        assembly: AffineBlockAssembly,
        /,
        *,
        correction_fields: Sequence[str],
        realizations: Mapping[str, Any],
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ) -> PreparedLinearCorrection:
        del realizations
        if tuple(bound_conditions) != tuple(assembly.bound_conditions):
            raise ValueError("Kernel provider conditions must match the affine assembly.")
        realization_only = self.plan.functional.exactness in (
            "fixed-realization",
            "selected-section",
        ) or isinstance(
            self.plan, (SectionKernelCorrectionPlan, MatrixFreeKernelCorrectionPlan)
        )
        if realization_only and policy.exactness_scope != "realization":
            raise ValueError(
                "This kernel correction plan is exact only on a declared realization."
            )
        if (
            isinstance(self.plan, MatrixFreeKernelCorrectionPlan)
            and policy.compatibility != "generalized"
        ):
            raise ValueError(
                "Matrix-free tolerance solves require generalized affine compatibility."
            )
        return self.plan.prepare(
            assembly,
            correction_fields,
            policy=policy,
            numeric_version=int(numeric_version),
        )

    def refresh(
        self,
        previous: PreparedLinearCorrection,
        bound_conditions: Sequence[BoundCondition],
        assembly: AffineBlockAssembly,
        /,
        *,
        correction_fields: Sequence[str],
        realizations: Mapping[str, Any],
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ) -> PreparedLinearCorrection:
        del previous
        return self.prepare(
            bound_conditions,
            assembly,
            correction_fields=correction_fields,
            realizations=realizations,
            policy=policy,
            numeric_version=numeric_version,
        )


class CanonicalKernelCorrectionProvider(_KernelCorrectionProvider):
    plan: CanonicalKernelCorrectionPlan

    def __init__(self, metric, functional, /):
        self.plan = CanonicalKernelCorrectionPlan(metric, functional)


class FiniteFeatureKernelCorrectionProvider(_KernelCorrectionProvider):
    plan: FiniteFeatureKernelCorrectionPlan

    def __init__(self, metric, functional, /):
        self.plan = FiniteFeatureKernelCorrectionPlan(metric, functional)


class SectionKernelCorrectionProvider(_KernelCorrectionProvider):
    plan: SectionKernelCorrectionPlan

    def __init__(self, metric, functional, sections, /, *, require_exact: bool = False):
        self.plan = SectionKernelCorrectionPlan(
            metric,
            functional,
            sections,
            require_exact=require_exact,
        )


class MatrixFreeKernelCorrectionProvider(_KernelCorrectionProvider):
    plan: MatrixFreeKernelCorrectionPlan

    def __init__(self, metric, functional, /, *, solve_policy=None):
        self.plan = MatrixFreeKernelCorrectionPlan(
            metric,
            functional,
            solve_policy=solve_policy,
        )


__all__ = [
    "CanonicalKernelCorrectionPlan",
    "CanonicalKernelCorrectionProvider",
    "FiniteFeatureKernelCorrectionPlan",
    "FiniteFeatureKernelCorrectionProvider",
    "IntegralKernelRepresenter",
    "JetKernelRepresenter",
    "KernelCorrectionEvidence",
    "KernelCorrectionPlan",
    "KernelCorrectionRepresentation",
    "KernelRepresenter",
    "MatrixFreeKernelCorrectionPlan",
    "MatrixFreeKernelCorrectionProvider",
    "PointKernelRepresenter",
    "SectionKernelCorrectionPlan",
    "SectionKernelCorrectionProvider",
]
