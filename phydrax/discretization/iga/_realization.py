#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._basis import TensorSplineBasisSpec
from ._topology import SplineSpanTopology


def _cell_gathers(basis: TensorSplineBasisSpec, /) -> np.ndarray:
    spans = tuple(np.asarray(axis.span_indices) for axis in basis.axes)
    routes: list[list[int]] = []
    for span_row in np.ndindex(basis.span_shape):
        active = tuple(int(spans[axis][row]) for axis, row in enumerate(span_row))
        local = [
            int(
                np.ravel_multi_index(
                    tuple(
                        span - basis.degree + offset
                        for span, offset in zip(active, shifts, strict=True)
                    ),
                    basis.control_shape,
                )
            )
            for shifts in np.ndindex((basis.degree + 1,) * basis.parametric_dimension)
        ]
        routes.append(local)
    return np.asarray(routes, dtype=np.int32)


@runtime_checkable
class BasisRealization(Protocol):
    """Structural contract shared by direct and extracted local realizations."""

    realization_id: str

    def gather(self, coefficients: ArrayLike, /) -> Array: ...

    def gather_transpose(self, local_values: ArrayLike, /) -> Array: ...


class DirectTensorRealization(StrictModule, NonTrainableState):
    """Exact span-to-control gather realization for a tensor spline basis."""

    basis: TensorSplineBasisSpec
    topology: SplineSpanTopology
    cell_gathers: Array
    realization_id: str = eqx.field(static=True)

    def __init__(self, basis: TensorSplineBasisSpec, topology: SplineSpanTopology, /):
        if not isinstance(basis, TensorSplineBasisSpec) or not isinstance(
            topology, SplineSpanTopology
        ):
            raise TypeError(
                "Direct realization requires TensorSplineBasisSpec and SplineSpanTopology."
            )
        if (
            basis.axis_names != topology.axis_names
            or basis.span_shape != topology.span_shape
        ):
            raise ValueError(
                "Direct realization basis and positive-span topology must match exactly."
            )
        gathers = _cell_gathers(basis)
        self.basis = basis
        self.topology = topology
        self.cell_gathers = jnp.asarray(gathers)
        self.realization_id = canonical_fingerprint(
            {
                "kind": "iga-direct-tensor-realization",
                "basis": basis.basis_id,
                "topology": topology.topology_id,
                "gathers": array_tree_fingerprint(gathers),
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.cell_gathers.shape[0])

    @property
    def local_width(self) -> int:
        return int(self.cell_gathers.shape[1])

    def gather(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[: len(self.basis.control_shape)] != self.basis.control_shape:
            raise ValueError(
                "Spline coefficients must begin with the basis control shape."
            )
        flat = values.reshape(
            (self.basis.coefficient_count,)
            + values.shape[len(self.basis.control_shape) :]
        )
        return flat[self.cell_gathers]

    def gather_transpose(self, local_values: ArrayLike, /) -> Array:
        values = jnp.asarray(local_values)
        if values.shape[:2] != tuple(self.cell_gathers.shape):
            raise ValueError("Local values must begin with (cell_count, local_width).")
        flat = jnp.zeros(
            (self.basis.coefficient_count,) + values.shape[2:], dtype=values.dtype
        )
        flat = flat.at[self.cell_gathers.reshape((-1,))].add(
            values.reshape((-1,) + values.shape[2:])
        )
        return flat.reshape(self.basis.control_shape + values.shape[2:])


class ExtractedBernsteinRealization(StrictModule, NonTrainableState):
    """Per-positive-span Bézier extraction with its exact algebraic transpose."""

    direct: DirectTensorRealization
    extraction: Array
    realization_id: str = eqx.field(static=True)

    def __init__(self, direct: DirectTensorRealization, extraction: ArrayLike, /):
        if not isinstance(direct, DirectTensorRealization):
            raise TypeError("direct must be a DirectTensorRealization.")
        matrices = np.asarray(extraction)
        width = direct.local_width
        if matrices.shape != (direct.cell_count, width, width):
            raise ValueError(
                "Extraction must have shape (cell_count, local_width, local_width)."
            )
        if not np.issubdtype(matrices.dtype, np.number) or np.any(~np.isfinite(matrices)):
            raise ValueError("Extraction matrices must be finite numeric arrays.")
        if np.any(np.linalg.matrix_rank(matrices) != width):
            raise ValueError("Every Bernstein extraction matrix must be nonsingular.")
        self.direct = direct
        self.extraction = jnp.asarray(matrices)
        self.realization_id = canonical_fingerprint(
            {
                "kind": "iga-extracted-bernstein-realization",
                "direct": direct.realization_id,
                "extraction": array_tree_fingerprint(matrices),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.direct.cell_count

    @property
    def local_width(self) -> int:
        return self.direct.local_width

    def realize(self, coefficients: ArrayLike, /) -> Array:
        local = self.direct.gather(coefficients)
        return contract(
            "eij,ej...->ei...",
            self.extraction,
            local,
            backend="jax",
        )

    def transpose(self, bernstein_values: ArrayLike, /) -> Array:
        values = jnp.asarray(bernstein_values)
        if values.shape[:2] != (self.cell_count, self.local_width):
            raise ValueError(
                "Bernstein values must begin with (cell_count, local_width)."
            )
        local = contract(
            "eji,ej...->ei...",
            self.extraction,
            values,
            backend="jax",
        )
        return self.direct.gather_transpose(local)


__all__ = [
    "BasisRealization",
    "DirectTensorRealization",
    "ExtractedBernsteinRealization",
]
