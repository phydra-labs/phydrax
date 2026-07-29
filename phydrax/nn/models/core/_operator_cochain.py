#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adapters between metric cochain complexes and neural-operator samples."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ....graph._cochain import (
    CochainBoundaryKind,
    CochainBoundaryPolicy,
    CochainComplexIR,
)
from ._operator import FunctionSamples
from ._operator_topology import OperatorTopology


def function_samples_from_cochain(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    values: Any | None,
    sample_cells: Any | None = None,
    boundary_policy: CochainBoundaryKind = "absolute",
    mask: Any | None = None,
) -> FunctionSamples:
    """Represent one cochain degree as topology-aligned operator samples.

    Hodge-star entries are exposed as physical quadrature weights. Relative
    boundary conditions retain the fixed sample shape and mask boundary cells.
    """
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("function_samples_from_cochain requires a CochainComplexIR.")
    resolved_degree = int(degree)
    if resolved_degree < 0 or resolved_degree > complex_ir.max_degree:
        raise ValueError(
            f"Cochain degree must lie in [0, {complex_ir.max_degree}]."
        )
    degree_coordinates = complex_ir.coordinates[resolved_degree]
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
    weights = complex_ir.hodge_stars[resolved_degree][local_cells]
    policy = CochainBoundaryPolicy(boundary_policy)
    active = complex_ir.active_mask(resolved_degree, policy)[local_cells]
    resolved_mask = active if mask is None else jnp.asarray(mask, dtype=bool) & active
    return FunctionSamples(
        values=None if values is None else jnp.asarray(values),
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=resolved_mask,
        topology=topology,
    )


__all__ = ["function_samples_from_cochain"]
