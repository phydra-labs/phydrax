#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._lattice_equivariant import InvariantFilterBasis


class InvariantBasisTransferReport(StrictModule):
    """Projected target coefficients and the relative discarded kernel residual."""

    coefficients: Array
    relative_residual: float
    source_fingerprint: str
    target_fingerprint: str


class InvariantBasisTransferPlan(StrictModule, NonTrainableState):
    """Host-side cross-dimensional embedding and invariant-basis projection plan."""

    source_basis: InvariantFilterBasis
    target_basis: InvariantFilterBasis
    target_axes: tuple[int, ...]
    residual_tolerance: float

    def __init__(
        self,
        source_basis: InvariantFilterBasis,
        target_basis: InvariantFilterBasis,
        /,
        *,
        target_axes: Sequence[int] | None = None,
        residual_tolerance: float = 1e-6,
    ):
        if not isinstance(source_basis, InvariantFilterBasis) or not isinstance(
            target_basis, InvariantFilterBasis
        ):
            raise TypeError(
                "source_basis and target_basis must be invariant filter bases."
            )
        source_dimension = source_basis.group.dimension
        target_dimension = target_basis.group.dimension
        if source_dimension >= target_dimension:
            raise ValueError(
                "Basis transfer requires a strictly higher target dimension."
            )
        axes = (
            tuple(range(source_dimension))
            if target_axes is None
            else tuple(int(axis) for axis in target_axes)
        )
        if (
            len(axes) != source_dimension
            or len(set(axes)) != len(axes)
            or any(not 0 <= axis < target_dimension for axis in axes)
        ):
            raise ValueError("target_axes must embed every source dimension uniquely.")
        source_input_blocks = source_basis.input_layout.blocks
        target_input_blocks = target_basis.input_layout.blocks
        source_output_blocks = source_basis.output_layout.blocks
        target_output_blocks = target_basis.output_layout.blocks
        if len(source_input_blocks) != len(target_input_blocks) or len(
            source_output_blocks
        ) != len(target_output_blocks):
            raise ValueError(
                "Source and target tensor layouts must have matching blocks."
            )
        for source_block, target_block in zip(
            source_input_blocks + source_output_blocks,
            target_input_blocks + target_output_blocks,
            strict=True,
        ):
            source_type = source_block.tensor_type
            target_type = target_block.tensor_type
            if (
                source_type.rank != 0
                or target_type.rank != 0
                or source_type.parity != target_type.parity
                or source_block.name != target_block.name
                or source_block.multiplicity != target_block.multiplicity
            ):
                raise ValueError(
                    "Initial cross-dimensional transfer supports matching scalar and "
                    "pseudoscalar tensor blocks only."
                )
        if (
            source_basis.input_layout.channel_count
            != target_basis.input_layout.channel_count
            or (
                source_basis.output_layout.channel_count
                != target_basis.output_layout.channel_count
            )
        ):
            raise ValueError("Source and target channel widths must agree.")
        for source_axis, target_axis in enumerate(axes):
            if (
                source_basis.kernel_shape[source_axis]
                != target_basis.kernel_shape[target_axis]
            ):
                raise ValueError("Embedded source and target kernel sizes must agree.")
        tolerance = float(residual_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("residual_tolerance must be finite and non-negative.")
        self.source_basis = source_basis
        self.target_basis = target_basis
        self.target_axes = axes
        self.residual_tolerance = tolerance

    def embed_kernel(self, kernel: Array, /) -> Array:
        """Embed the source kernel on the central target-dimensional slice."""
        source = np.asarray(kernel)
        expected = self.source_basis.kernel_shape + (
            self.source_basis.output_layout.channel_count,
            self.source_basis.input_layout.channel_count,
        )
        if source.shape != expected:
            raise ValueError(
                f"kernel must have source shape {expected}; got {source.shape}."
            )
        target_shape = self.target_basis.kernel_shape + expected[-2:]
        embedded = np.zeros(target_shape, dtype=source.dtype)
        target_slices: list[int | slice] = [
            size // 2 for size in self.target_basis.kernel_shape
        ]
        for source_axis, target_axis in enumerate(self.target_axes):
            target_slices[target_axis] = slice(None)
        target_slices.extend((slice(None), slice(None)))
        source_order = tuple(np.argsort(self.target_axes)) + tuple(
            range(len(self.target_axes), source.ndim)
        )
        embedded[tuple(target_slices)] = np.transpose(source, source_order)
        return jnp.asarray(embedded)

    def transfer(self, coefficients: Array, /) -> InvariantBasisTransferReport:
        """Embed, project, and reject transfers exceeding the declared residual."""
        source_kernel = self.source_basis.synthesize(coefficients)
        embedded = self.embed_kernel(source_kernel)
        projected = self.target_basis.project(embedded)
        embedded_host = np.asarray(embedded)
        projected_host = np.asarray(projected)
        denominator = max(float(np.linalg.norm(embedded_host.reshape(-1))), 1e-30)
        residual = float(
            np.linalg.norm((embedded_host - projected_host).reshape(-1)) / denominator
        )
        if residual > self.residual_tolerance:
            raise ValueError(
                "Invariant basis transfer residual exceeds tolerance: "
                f"{residual} > {self.residual_tolerance}."
            )
        flat_basis = self.target_basis.basis.reshape(self.target_basis.rank, -1)
        target_coefficients = oe.contract(
            "ri,i->r",
            flat_basis.astype(projected.dtype),
            projected.reshape(-1),
        )
        return InvariantBasisTransferReport(
            coefficients=target_coefficients,
            relative_residual=residual,
            source_fingerprint=self.source_basis.fingerprint,
            target_fingerprint=self.target_basis.fingerprint,
        )


__all__ = ["InvariantBasisTransferPlan", "InvariantBasisTransferReport"]
