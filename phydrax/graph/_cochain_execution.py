"""Prepared native cochain owners attached to graph execution coordinates."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cochain import CochainDiscretization
from ..exterior._complex import ComplexBoundary
from ..linalg._complexes import HarmonicSubspace


@final
class CochainGraphBinding(StrictModule, NonTrainableState):
    """Native numerical preparation and its placement in a graph batch.

    Graph transformations change only placement; sparse metric preparation,
    numerical revisions, and scientific identities stay with the native owner.
    """

    discretization: CochainDiscretization
    harmonic: tuple[HarmonicSubspace | None, ...]
    boundary: ComplexBoundary = eqx.field(static=True)
    node_offset: int = eqx.field(static=True)
    graph_index: int = eqx.field(static=True)

    def __init__(
        self,
        discretization: CochainDiscretization,
        harmonic: tuple[HarmonicSubspace | None, ...],
        boundary: ComplexBoundary,
        /,
        *,
        node_offset: int = 0,
        graph_index: int = 0,
    ) -> None:
        self.discretization = discretization
        self.harmonic = harmonic
        self.boundary = boundary
        self.node_offset = node_offset
        self.graph_index = graph_index

    def shifted(self, node_offset: int, graph_index: int, /) -> CochainGraphBinding:
        return CochainGraphBinding(
            self.discretization,
            self.harmonic,
            self.boundary,
            node_offset=self.node_offset + node_offset,
            graph_index=self.graph_index + graph_index,
        )

    def degree_start(self, degree: int, /) -> int:
        return self.node_offset + sum(self.discretization.cell_counts[:degree])


def _cochain_metric_valid(bindings: tuple[CochainGraphBinding, ...], /) -> Array:
    """Compute current native evidence, including numerical metric refreshes."""
    if not bindings:
        raise ValueError("Native metric admission requires prepared cochain bindings.")
    return jnp.all(
        jnp.stack(tuple(binding.discretization.metric_valid for binding in bindings))
    )
