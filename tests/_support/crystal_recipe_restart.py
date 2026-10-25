"""Fresh-process crystal packing continuation from a recipe and a checkpoint.

Run as ``python crystal_recipe_restart.py RECIPE CHECKPOINT OUTPUT``: the
worker builds nothing by hand.  It admits the recipe archive, rebuilds the
stream from it, restores the generation checkpoint and continues exactly as
the in-process reference does through `continuation_arrays`, writing every
pending and published proposal and every final continuation leaf to ``OUTPUT`` (``.npz``).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import jax
import numpy as np

from phydrax.atomistic._generation import (
    AtomisticGeneratedProposal,
    AtomisticGenerationState,
    AtomisticMemoryGenerationSink,
    PreparedAtomisticGenerationStream,
)
from phydrax.atomistic.crystal import read_crystal_packing_recipe


MAXIMUM_TRANSITIONS = 400


def _proposals(
    prefix: str, items: Sequence[AtomisticGeneratedProposal], /
) -> dict[str, np.ndarray]:
    cells = []
    for item in items:
        if item.cell is None:
            raise ValueError("Crystal proposals carry a periodic cell.")
        cells.append(np.asarray(item.cell))
    arrays = {
        f"{prefix}_ids": np.asarray([item.proposal_id for item in items], dtype=str),
        f"{prefix}_ordinals": np.asarray(
            [item.ordinal for item in items], dtype=np.int64
        ),
        f"{prefix}_status": np.asarray([item.status for item in items], dtype=np.int32),
        f"{prefix}_members": np.asarray([item.member for item in items], dtype=np.int32),
        f"{prefix}_atoms": np.asarray(
            [item.positions.shape[0] for item in items], dtype=np.int64
        ),
        f"{prefix}_positions": (
            np.concatenate([np.asarray(item.positions) for item in items])
            if items
            else np.zeros((0, 3), dtype=np.float64)
        ),
        f"{prefix}_cells": (
            np.stack(cells) if cells else np.zeros((0, 3, 3), dtype=np.float64)
        ),
    }
    names = sorted({name for item in items for name, _ in item.payload})
    for name in names:
        arrays[f"{prefix}_payload_{name}"] = np.concatenate(
            [np.ravel(np.asarray(item.value(name))) for item in items]
        )
    return arrays


def continuation_arrays(
    runtime: PreparedAtomisticGenerationStream, state: AtomisticGenerationState, /
) -> dict[str, np.ndarray]:
    """Pending/public proposals and every final continuation leaf, including keys."""
    pending = runtime.ready(state)
    sink = AtomisticMemoryGenerationSink("crystal-recipe-continuation")
    final = runtime.run(state, sink, maximum_transitions=MAXIMUM_TRANSITIONS)
    if not runtime.finished(final):
        raise RuntimeError("The continuation did not finish within its transition bound.")
    arrays = {**_proposals("pending", pending), **_proposals("published", sink.records)}
    paths = []
    for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(final)[0]):
        if not isinstance(leaf, jax.Array):
            raise TypeError(
                "Generation continuation leaves must be canonical JAX arrays."
            )
        value = (
            jax.random.key_data(leaf)
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        )
        arrays[f"final_state_{index:06d}"] = np.asarray(jax.device_get(value))
        paths.append(jax.tree_util.keystr(path))
    arrays["final_state_paths"] = np.asarray(paths, dtype=np.str_)
    return arrays


def main(arguments: Sequence[str], /) -> None:
    recipe_path, checkpoint_path, output_path = arguments
    recipe, receipt = read_crystal_packing_recipe(recipe_path)
    runtime = recipe.stream().prepare()
    state = runtime.read_checkpoint(checkpoint_path)
    np.savez(
        output_path,
        allow_pickle=False,
        recipe_id=np.asarray(receipt.recipe_id),
        stream_id=np.asarray(runtime.stream_id),
        **continuation_arrays(runtime, state),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
