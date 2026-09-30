#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._boundary_complex import boundary_subcomplex, BoundaryComplex
from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    ComplexMap,
    FailurePolicy,
    FunctionLinearOperator,
    HilbertComplex,
    LinearSolvePolicy,
    LinearSystem,
    OperatorPairing,
    OperatorProperties,
    PCG,
    prepare,
    TolerancePolicy,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ._complex import AbstractDeRhamComplex


@final
class TraceEvidence(StrictModule, NonTrainableState):
    """Executed commuting residual of a prepared trace, one row per degree."""

    commuting_residuals: Array
    tolerance: float = eqx.field(static=True)
    successful: Array
    map_id: str = eqx.field(static=True)


@final
class _RestrictedRiesz(StrictModule):
    parent: ArraySpace
    indices: Array
    signs: Array

    def __call__(self, values: Array, /) -> Array:
        extended = jnp.zeros(self.parent.shape, dtype=self.parent.dtype)
        extended = extended.at[self.indices].set(
            self.signs * values, indices_are_sorted=True, unique_indices=True
        )
        paired = self.parent.validate(self.parent.pairing.riesz(extended))
        return self.signs * paired[self.indices]


def _boundary_space(
    parent: ArraySpace, indices: Array, signs: Array, space_id: str, /
) -> ArraySpace:
    """Use the principal restricted Gram, never a masked full Gram inverse."""
    coordinates = ArraySpace(
        (indices.size,), dtype=parent.dtype, space_id=f"{space_id}:coordinates"
    )
    gram = FunctionLinearOperator(
        _RestrictedRiesz(parent, indices, signs),
        source=coordinates,
        target=coordinates,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"{space_id}:gram",
    )
    inverse = prepare(
        LinearSystem(gram, problem_id=f"{space_id}:gram-system"),
        LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=1.0e-12, absolute=0.0, max_steps=max(1, 4 * indices.size)
            ),
            failure=FailurePolicy("status"),
        ),
    )
    return ArraySpace(
        (indices.size,),
        dtype=parent.dtype,
        pairing=OperatorPairing(gram, prepared_inverse=inverse),
        space_id=space_id,
    )


def _cell_trace_map(
    complex: AbstractCellDeRhamComplex, boundary: BoundaryComplex, /
) -> ComplexMap:
    source = complex.hilbert_complex()
    spaces: list[ArraySpace] = []
    for degree, indices in enumerate(boundary.parent_indices):
        parent = source.space(degree)
        if not isinstance(parent, ArraySpace) or parent.shape != (
            complex.topology.entities(degree).count,
        ):
            raise ValueError(
                "Cell restriction requires one coefficient per oriented cell; use the realization's FE or spline trace map for other layouts."
            )
        signs = (
            boundary.orientation_signs[degree].astype(parent.dtype)
            if degree
            else jnp.ones((indices.size,), dtype=parent.dtype)
        )
        spaces.append(
            _boundary_space(
                parent,
                indices,
                signs,
                f"{complex.realization_id}:{boundary.boundary_id}:space:{degree}",
            )
        )
    differentials: list[AbstractLinearOperator] = []
    for degree, incidence in enumerate(boundary.topology.incidences):
        differentials.append(
            SparseCoordinateOperator(
                EdgeRelation(
                    np.asarray(incidence.relation.source_indices),
                    np.asarray(incidence.relation.target_indices),
                    source_size=spaces[degree].size,
                    target_size=spaces[degree + 1].size,
                    valid=np.asarray(incidence.relation.valid),
                ),
                incidence.signs.astype(spaces[degree].dtype),
                source=spaces[degree],
                target=spaces[degree + 1],
                operator_id=f"{boundary.boundary_id}:d:{degree}",
                accumulation_dtype=spaces[degree].dtype,
            )
        )
    target = HilbertComplex(
        tuple(spaces),
        tuple(differentials),
        complex_id=f"{complex.realization_id}:{boundary.boundary_id}:trace-complex",
    )
    return ComplexMap(
        source,
        target,
        boundary.restriction_operators(source, target),
        map_id=f"{complex.realization_id}:{boundary.boundary_id}:trace",
    )


def trace_map(
    complex: AbstractDeRhamComplex, /, *, boundary_mask: ArrayLike | None = None
) -> ComplexMap:
    """Prepare the signed outward boundary restriction of a cell realization.

    The target pairing is the true principal restricted Gram. The default uses
    the realization's degree n−1 boundary mask; an explicit mask binds a selected
    boundary patch, including its entire incidence closure. This is preparation,
    not a runtime geometry search or an orientation inferred from coordinates.
    """
    from ..discretization.fem._de_rham import FiniteElementDeRhamComplex
    from ..discretization.iga._compatible import (
        AssembledSplineDeRhamComplex,
        SplineDeRhamComplex,
    )

    if isinstance(complex, FiniteElementDeRhamComplex):
        return complex.trace_complex_map(boundary_mask=boundary_mask)
    if isinstance(complex, (SplineDeRhamComplex, AssembledSplineDeRhamComplex)):
        return complex.trace_complex_map(boundary_mask=boundary_mask)
    if not isinstance(complex, AbstractCellDeRhamComplex):
        raise TypeError("complex must implement AbstractCellDeRhamComplex.")
    mask = (
        complex.boundary_masks[complex.dimension - 1]
        if boundary_mask is None
        else boundary_mask
    )
    boundary = boundary_subcomplex(complex.topology, boundary_mask=mask)
    return _cell_trace_map(complex, boundary)


def trace_evidence(
    mapping: ComplexMap, values: tuple[ArrayLike, ...], /, *, tolerance: float = 1.0e-12
) -> TraceEvidence:
    """Measure d_boundary tr − tr d on explicitly supplied volume cochains."""
    if not isinstance(mapping, ComplexMap):
        raise TypeError("mapping must be ComplexMap.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if mapping.degree_offset != 0 or len(values) != mapping.target.top_degree:
        raise ValueError(
            "Trace evidence needs one cochain per boundary differential degree."
        )

    def subtract(left: Array, right: Array) -> Array:
        return left - right

    residuals: list[Array] = []
    for degree, value in enumerate(values):
        source_value = mapping.source.space(degree).validate(value)
        left = mapping.target.differential(degree).mv(
            mapping.maps[degree].mv(source_value)
        )
        right = mapping.maps[degree + 1].mv(
            mapping.source.differential(degree).mv(source_value)
        )
        residual = jax.tree.map(subtract, left, right)
        target_space = mapping.target.space(degree + 1)
        residuals.append(
            jnp.sqrt(jnp.maximum(jnp.real(target_space.inner(residual, residual)), 0.0))
        )
    result = jnp.stack(residuals) if residuals else jnp.empty((0,), dtype=jnp.float64)
    return TraceEvidence(
        result,
        tolerance,
        jnp.all(jnp.isfinite(result) & (result <= tolerance)),
        mapping.map_id,
    )


__all__ = ["TraceEvidence", "trace_evidence", "trace_map"]
