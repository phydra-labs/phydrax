#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._stencil import StencilFootprint


if TYPE_CHECKING:
    from ..amr._core import BlockLevelPlan
    from ..amr._fd_halo import FDAMRHaloWorkspace


class BlockLocalStencilResult(StrictModule):
    """Block-interior values and validity after one local stencil program."""

    values: Array
    valid: Array
    finite: Array
    successful: Array
    execution_id: str = eqx.field(static=True)


class BlockLocalStencilExecutionPlan(StrictModule, NonTrainableState):
    """Execute one footprint-qualified callable over padded AMR blocks."""

    level: BlockLevelPlan
    footprint: StencilFootprint
    required_halo: tuple[int, ...] = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: BlockLevelPlan,
        footprint: StencilFootprint,
        /,
    ) -> None:
        from ..amr._core import BlockLevelPlan

        if not isinstance(level, BlockLevelPlan):
            raise TypeError("level must be BlockLevelPlan.")
        if not isinstance(footprint, StencilFootprint):
            raise TypeError("footprint must be StencilFootprint.")
        if len(footprint.axis_names) != len(level.block_shape):
            raise ValueError("Stencil footprint and block dimensions must match.")
        required = tuple(
            max(lower, upper)
            for lower, upper in zip(footprint.lower, footprint.upper, strict=True)
        )
        if any(
            available < needed
            for available, needed in zip(level.halo_width, required, strict=True)
        ):
            raise ValueError("Block halo does not contain the stencil read footprint.")
        self.level = level
        self.footprint = footprint
        self.required_halo = required
        self.execution_id = canonical_fingerprint(
            {
                "kind": "block-local-stencil-execution",
                "level": level.plan_id,
                "footprint": footprint.footprint_id,
                "required_halo": list(required),
            }
        )

    def apply(
        self,
        workspace: FDAMRHaloWorkspace,
        block_operator: Callable[[Array], Array],
        /,
    ) -> BlockLocalStencilResult:
        """Apply ``block_operator`` independently to complete padded blocks."""
        from ..amr._fd_halo import FDAMRHaloWorkspace

        if not isinstance(workspace, FDAMRHaloWorkspace):
            raise TypeError("workspace must be FDAMRHaloWorkspace.")
        if not callable(block_operator):
            raise TypeError("block_operator must be callable.")
        expected_prefix = (self.level.maximum_blocks,) + tuple(
            size + 2 * halo
            for size, halo in zip(
                self.level.block_shape, self.level.halo_width, strict=True
            )
        )
        if workspace.values.shape[: len(expected_prefix)] != expected_prefix:
            raise ValueError("Halo workspace does not match the block execution plan.")
        output = jax.vmap(block_operator)(workspace.values)
        expected_output_prefix = (self.level.maximum_blocks,) + self.level.block_shape
        if output.shape[: len(expected_output_prefix)] != expected_output_prefix:
            raise ValueError(
                "block_operator output must begin with the unpadded block shape."
            )
        center = tuple(
            slice(halo, halo + size)
            for size, halo in zip(
                self.level.block_shape, self.level.halo_width, strict=True
            )
        )
        valid = workspace.valid[(slice(None),) + center]
        mask = valid.reshape(valid.shape + (1,) * (output.ndim - valid.ndim))
        values = jnp.where(mask, output, jnp.zeros((), dtype=output.dtype))
        finite = jnp.all(jnp.where(mask, jnp.isfinite(values), True))
        return BlockLocalStencilResult(
            values=values,
            valid=valid,
            finite=finite,
            successful=finite,
            execution_id=self.execution_id,
        )


__all__ = ["BlockLocalStencilExecutionPlan", "BlockLocalStencilResult"]
