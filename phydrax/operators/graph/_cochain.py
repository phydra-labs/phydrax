#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cochain GraphIR algorithms bound to ``DomainFunction`` carriers."""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx

from phydrax.domain import DomainFunction
from phydrax.domain.graph import as_cochain_field, cochain_form_type, GraphModel

from ..._strict import StrictModule
from ...exterior._complex import ComplexBoundary
from ...exterior._form_type import FormType
from ...graph._ir import GraphIR
from ...linalg import HodgeLaplacianPart


_INPUT_KEY = "_phydrax_cochain_input"
_OUTPUT_KEY = "_phydrax_cochain_output"


@final
class _TypedCochainOperator(StrictModule):
    module: Callable[[GraphIR], GraphIR]
    form_type: FormType = eqx.field(static=True)

    def __init__(
        self,
        module: Callable[[GraphIR], GraphIR],
        form_type: FormType,
        /,
    ) -> None:
        self.module = module
        self.form_type = form_type

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        from ...graph._cochain_residual import _admit_form_type

        return self.module(_admit_form_type(graph, self.form_type))


def _bind_graph_module(
    field: DomainFunction,
    module: Callable[[GraphIR], GraphIR],
    output_type: FormType,
    /,
) -> DomainFunction:
    result = DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=GraphModel(
            _TypedCochainOperator(module, cochain_form_type(field)),
            input_fn=field,
            input_key=_INPUT_KEY,
            output="nodes",
            output_key=_OUTPUT_KEY,
        ),
        metadata=field.metadata,
    )
    return as_cochain_field(result, output_type, representation="cochain")


def cochain_exterior_derivative(
    field: DomainFunction,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DomainFunction:
    """Apply the exact differential to a graph-backed domain cochain field."""
    from ...graph import CochainExteriorDerivative

    form_type = cochain_form_type(field)
    return _bind_graph_module(
        field,
        CochainExteriorDerivative(
            form_type.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary=boundary,
        ),
        form_type.exterior_derivative_type(),
    )


def cochain_codifferential(
    field: DomainFunction,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DomainFunction:
    """Apply the metric Hilbert adjoint to a domain cochain field."""
    from ...graph import CochainCodifferential

    form_type = cochain_form_type(field)
    return _bind_graph_module(
        field,
        CochainCodifferential(
            form_type.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary=boundary,
        ),
        form_type.codifferential_type(),
    )


def cochain_hodge_laplacian(
    field: DomainFunction,
    /,
    *,
    part: HodgeLaplacianPart = "complete",
    boundary: ComplexBoundary = "absolute",
) -> DomainFunction:
    """Apply the metric Laplacian to a graph-backed domain cochain field."""
    from ...graph import CochainHodgeLaplacian

    form_type = cochain_form_type(field)
    return _bind_graph_module(
        field,
        CochainHodgeLaplacian(
            form_type.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            part=part,
            boundary=boundary,
        ),
        form_type,
    )


def cochain_harmonic_projection(
    field: DomainFunction,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DomainFunction:
    """Bind the graph harmonic projection to a domain cochain field."""
    from ...graph import CochainHarmonicProjection

    form_type = cochain_form_type(field)
    return _bind_graph_module(
        field,
        CochainHarmonicProjection(
            form_type.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary=boundary,
        ),
        form_type,
    )


__all__ = [
    "cochain_codifferential",
    "cochain_exterior_derivative",
    "cochain_harmonic_projection",
    "cochain_hodge_laplacian",
]
