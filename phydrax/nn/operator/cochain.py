#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adapters between metric cochain complexes and neural-operator samples."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ...discretization._cochain_hodge import DiagonalHodge
from ...exterior._complex import ComplexBoundary
from ...graph._cochain import CochainComplexIR
from ...graph._operator_topology import OperatorTopology
from ...typing import parse
from .data import FunctionSamples


def function_samples_from_cochain(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    values: Any | None,
    sample_cells: Any | None = None,
    boundary: ComplexBoundary = "absolute",
    mask: Any | None = None,
) -> FunctionSamples:
    """Represent one cochain degree as topology-aligned operator samples.

    Diagonal Hodges provide exact scalar quadrature weights. A coupled sparse
    Gram remains on the native topology owner, not approximated by quadrature.
    Relative boundary conditions retain the fixed shape and mask boundary cells.
    """
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("function_samples_from_cochain requires a CochainComplexIR.")
    resolved_degree = int(degree)
    if resolved_degree < 0 or resolved_degree > complex_ir.max_degree:
        raise ValueError(f"Cochain degree must lie in [0, {complex_ir.max_degree}].")
    realization = complex_ir.discretization
    degree_coordinates = realization.coordinates[resolved_degree]
    if degree_coordinates is None:
        raise ValueError(
            "Cochain FunctionSamples require physical coordinates at every sampled degree."
        )
    topology = OperatorTopology.from_cochain(
        complex_ir,
        resolved_degree,
        sample_cells=sample_cells,
    )
    local_cells = topology.sample_entities - complex_ir.cell_offsets[resolved_degree]
    coordinates = degree_coordinates[local_cells]
    hodge = realization.hodges[resolved_degree]
    weights = hodge.weights[local_cells] if isinstance(hodge, DiagonalHodge) else None
    resolved_boundary = parse(boundary, ComplexBoundary, "boundary")
    active_indices = realization.active_indices(
        resolved_degree, boundary=resolved_boundary
    )
    full_active = jnp.zeros((realization.cell_counts[resolved_degree],), dtype=jnp.bool_)
    active = full_active.at[active_indices].set(True)[local_cells]
    resolved_mask = (
        active if mask is None else jnp.asarray(mask, dtype=jnp.bool_) & active
    )
    return FunctionSamples(
        values=None if values is None else jnp.asarray(values),
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=resolved_mask,
        topology=topology,
    )


__all__ = ["function_samples_from_cochain"]
