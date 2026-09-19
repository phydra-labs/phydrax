#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from ....privacy._provider import _PreparedPrivateGradient
from ._loader import OperatorBatchLoader, OperatorTrainingBatch


def prepare_private_operator_batch(
    loader: OperatorBatchLoader,
    prepared: _PreparedPrivateGradient,
    indices: np.ndarray,
    /,
    *,
    step: int,
) -> OperatorTrainingBatch:
    """Load one provider-selected case batch without exposing sampler indices."""
    if not isinstance(loader, OperatorBatchLoader):
        raise TypeError("loader must be an OperatorBatchLoader.")
    if not isinstance(prepared, _PreparedPrivateGradient):
        raise TypeError("prepared must be a private gradient preparation.")
    selected = np.asarray(indices)
    if selected.ndim != 1:
        raise ValueError("The initial private operator profile requires 1D case indices.")
    if selected.size and np.unique(selected).size != selected.size:
        raise ValueError("Poisson case sampling must not duplicate a privacy unit.")
    if selected.size:
        raw = loader.prepare_indices(
            tuple(int(value) for value in selected),
            epoch=0,
            batch_index=int(step),
        )
        is_padding = jnp.zeros((selected.size,), dtype=bool)
        return replace(
            raw,
            indices=(),
            epoch=0,
            batch_index=int(step),
            microstep=int(step),
            is_padding_example=is_padding,
        )

    # The transformed loss is vmapped over isolated singleton cases and the
    # upstream clipper replaces this lane's entire gradient with exact zero.
    # The selected source value therefore cannot influence the noised update.
    raw = loader.prepare_indices((0,), epoch=0, batch_index=int(step))
    return replace(
        raw,
        indices=(),
        epoch=0,
        batch_index=int(step),
        microstep=int(step),
        case_mask=jnp.ones((1,), dtype=bool),
        sampling_probabilities=jnp.ones(
            (1,), dtype=jnp.asarray(raw.case_log_weights).dtype
        ),
        is_padding_example=jnp.ones((1,), dtype=bool),
    )


__all__ = ["prepare_private_operator_batch"]
