#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric cochain operators over graph-backed ``DomainFunction`` fields."""

from __future__ import annotations

from typing import Literal

from phydrax.domain import DomainFunction
from phydrax.domain.graph import (
    cochain_field_spec,
    with_cochain_field_spec,
)

from ...graph import (
    CochainBoundaryKind,
    CochainCodifferential,
    CochainExteriorDerivative,
    CochainFieldSpec,
    CochainHarmonicProjection,
    CochainHodgeLaplacian,
    HodgeLaplacianComponent,
)
from ...nn.models.wrappers._graph import GraphModel


_INPUT_KEY = "_phydrax_cochain_input"
_OUTPUT_KEY = "_phydrax_cochain_output"


def _validate_primal_field(field: DomainFunction, /) -> CochainFieldSpec:
    if not isinstance(field, DomainFunction):
        raise TypeError("Cochain operators require a DomainFunction.")
    spec = cochain_field_spec(field)
    if spec.complex_side != "primal":
        raise ValueError(
            "Metric cochain DomainFunction operators currently support primal cochains only."
        )
    expected_orientation = "invariant" if spec.degree == 0 else "signed"
    if spec.cell_orientation != expected_orientation:
        raise ValueError(
            f"Degree-{spec.degree} primal cochains require "
            f"{expected_orientation!r} cell orientation semantics."
        )
    return spec


def _sampling_for_degree(degree: int) -> Literal["point_value", "cell_integral"]:
    return "point_value" if int(degree) == 0 else "cell_integral"


def _bind_graph_module(
    field: DomainFunction,
    module: object,
    output_spec: CochainFieldSpec,
    /,
) -> DomainFunction:
    result = DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=GraphModel(
            module,
            input_fn=field,
            input_key=_INPUT_KEY,
            output="nodes",
            output_key=_OUTPUT_KEY,
        ),
        metadata=field.metadata,
    )
    return with_cochain_field_spec(result, output_spec)


def cochain_exterior_derivative(
    field: DomainFunction,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> DomainFunction:
    """Apply the exact sparse exterior derivative ``d_k`` to a k-cochain field."""
    spec = _validate_primal_field(field)
    target_degree = spec.degree + 1
    output_spec = CochainFieldSpec(
        target_degree,
        complex_side="primal",
        cell_orientation="signed",
        sampling=_sampling_for_degree(target_degree),
    )
    return _bind_graph_module(
        field,
        CochainExteriorDerivative(
            spec.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary_policy=boundary_policy,
        ),
        output_spec,
    )


def cochain_codifferential(
    field: DomainFunction,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> DomainFunction:
    """Apply the exact metric codifferential ``delta_k`` to a k-cochain field."""
    spec = _validate_primal_field(field)
    if spec.degree == 0:
        raise ValueError("The codifferential is undefined for degree-0 cochains.")
    target_degree = spec.degree - 1
    output_spec = CochainFieldSpec(
        target_degree,
        complex_side="primal",
        cell_orientation="invariant" if target_degree == 0 else "signed",
        sampling=_sampling_for_degree(target_degree),
    )
    return _bind_graph_module(
        field,
        CochainCodifferential(
            spec.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary_policy=boundary_policy,
        ),
        output_spec,
    )


def cochain_hodge_laplacian(
    field: DomainFunction,
    /,
    *,
    component: HodgeLaplacianComponent = "complete",
    boundary_policy: CochainBoundaryKind = "absolute",
) -> DomainFunction:
    """Apply the lower, upper, or complete metric Hodge Laplacian."""
    spec = _validate_primal_field(field)
    return _bind_graph_module(
        field,
        CochainHodgeLaplacian(
            spec.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            component=component,
            boundary_policy=boundary_policy,
        ),
        spec,
    )


def cochain_harmonic_projection(
    field: DomainFunction,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> DomainFunction:
    """Project a cochain field onto the metric harmonic subspace."""
    spec = _validate_primal_field(field)
    return _bind_graph_module(
        field,
        CochainHarmonicProjection(
            spec.degree,
            input_key=_INPUT_KEY,
            output_key=_OUTPUT_KEY,
            boundary_policy=boundary_policy,
        ),
        spec,
    )


__all__ = [
    "cochain_codifferential",
    "cochain_exterior_derivative",
    "cochain_harmonic_projection",
    "cochain_hodge_laplacian",
]
