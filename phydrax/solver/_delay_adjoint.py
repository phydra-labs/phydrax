#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import diffrax as dfx


class CheckpointedDelayAdjoint(dfx.AbstractAdjoint):
    """Exact discrete adjoint for Diffrax delay solves.

    The delay backend stores every accepted local interpolation in the wrapped solver
    state. Recursive checkpointing therefore recomputes both the numerical state and
    the causal history queried by later steps. This class makes that full-state
    checkpointing contract explicit and gives delay solves a stable adjoint identity.
    """

    checkpointing: dfx.RecursiveCheckpointAdjoint

    def __init__(self, checkpoints: int | None = None):
        if checkpoints is not None and (
            not isinstance(checkpoints, int)
            or isinstance(checkpoints, bool)
            or checkpoints <= 0
        ):
            raise ValueError("checkpoints must be a positive integer or None.")
        self.checkpointing = dfx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)

    @property
    def checkpoints(self) -> int | None:
        """Number of online checkpoints, or Diffrax's automatic schedule."""
        return self.checkpointing.checkpoints

    def loop(self, **kwargs: Any) -> Any:
        """Differentiate the complete accepted-history solver state discretely."""
        return self.checkpointing.loop(**kwargs)


class SegmentedDelayAdjoint(dfx.AbstractAdjoint):
    """Discrete adjoint for a bounded number of host execution windows.

    Ordinary calls retain host segmentation. Under JAX tracing, the segmented API
    replays the identical controller and rolling-history state in one bounded Diffrax
    loop, so recursive checkpointing can transpose the complete causal computation.
    ``max_segments`` is the static replay bound.
    """

    checkpointing: dfx.RecursiveCheckpointAdjoint
    max_segments: int

    def __init__(
        self,
        max_segments: int,
        /,
        *,
        checkpoints: int | None = None,
    ):
        if (
            not isinstance(max_segments, int)
            or isinstance(max_segments, bool)
            or max_segments <= 0
        ):
            raise ValueError("max_segments must be a positive integer.")
        if checkpoints is not None and (
            not isinstance(checkpoints, int)
            or isinstance(checkpoints, bool)
            or checkpoints <= 0
        ):
            raise ValueError("checkpoints must be a positive integer or None.")
        self.checkpointing = dfx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)
        self.max_segments = max_segments

    @property
    def checkpoints(self) -> int | None:
        return self.checkpointing.checkpoints

    @property
    def maximum_steps(self) -> int:
        """Multiplier applied to ``max_steps_per_segment`` during traced replay."""
        return self.max_segments

    def loop(self, **kwargs: Any) -> Any:
        return self.checkpointing.loop(**kwargs)


__all__ = ["CheckpointedDelayAdjoint", "SegmentedDelayAdjoint"]
