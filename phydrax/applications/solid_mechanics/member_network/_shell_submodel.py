#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ShellSubmodelTransfer(StrictModule, NonTrainableState):
    """Boundary displacement/resultant transfer from a global member model."""

    boundary_node_indices: Array
    displacement_map: Array
    resultant_map: Array
    model_extent: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_node_indices: ArrayLike,
        displacement_map: ArrayLike,
        resultant_map: ArrayLike,
        model_extent: ArrayLike,
        /,
        *,
        transfer_id: str = "shell-submodel-transfer",
    ):
        nodes = jnp.asarray(boundary_node_indices, dtype=jnp.int32)
        displacement = jnp.asarray(displacement_map)
        resultant = jnp.asarray(resultant_map, dtype=displacement.dtype)
        extent = jnp.asarray(model_extent, dtype=displacement.dtype)
        if nodes.ndim != 1 or displacement.ndim != 2 or resultant.ndim != 2:
            raise ValueError("Shell transfer arrays have invalid ranks.")
        if displacement.shape[0] != nodes.size or resultant.shape[1] != nodes.size:
            raise ValueError("Shell boundary maps do not match boundary nodes.")
        if extent.shape != () or not bool(jnp.isfinite(extent) & (extent > 0.0)):
            raise ValueError("Shell submodel extent must be finite and positive.")
        self.boundary_node_indices = nodes
        self.displacement_map = displacement
        self.resultant_map = resultant
        self.model_extent = extent
        self.transfer_id = str(transfer_id)


class ShellSubmodelEvidence(StrictModule):
    beam_factor: Array
    strip_factor: Array
    shell_factor: Array
    relative_beam_error: Array
    relative_strip_error: Array
    governing_factor: Array
    agreement: Array
    boundary_transfer_residual: Array
    successful: Array


def compare_shell_submodel(
    beam_factor: ArrayLike,
    strip_factor: ArrayLike,
    shell_factor: ArrayLike,
    boundary_transfer_residual: ArrayLike,
    /,
    *,
    agreement_tolerance: float = 0.1,
    transfer_tolerance: float = 1.0e-6,
) -> ShellSubmodelEvidence:
    beam = jnp.asarray(beam_factor)
    strip = jnp.asarray(strip_factor, dtype=beam.dtype)
    shell = jnp.asarray(shell_factor, dtype=beam.dtype)
    transfer = jnp.asarray(boundary_transfer_residual, dtype=beam.dtype)
    beam_error = jnp.abs(beam - shell) / jnp.maximum(jnp.abs(shell), 1.0e-15)
    strip_error = jnp.abs(strip - shell) / jnp.maximum(jnp.abs(shell), 1.0e-15)
    agreement = jnp.maximum(beam_error, strip_error) <= agreement_tolerance
    successful = (
        jnp.all(jnp.isfinite(jnp.stack((beam, strip, shell, transfer))))
        & (shell > 0.0)
        & (transfer <= transfer_tolerance)
    )
    return ShellSubmodelEvidence(
        beam,
        strip,
        shell,
        beam_error,
        strip_error,
        jnp.minimum(jnp.minimum(beam, strip), shell),
        agreement,
        transfer,
        successful,
    )


__all__ = [
    "ShellSubmodelEvidence",
    "ShellSubmodelTransfer",
    "compare_shell_submodel",
]
