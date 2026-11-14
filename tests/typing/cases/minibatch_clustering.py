"""Mini-batch clustering exposes typed policies and evidence."""

from jax import Array
from typing_extensions import assert_type

from phydrax.ml.clustering import (
    ClusterRevival,
    MiniBatchBatchLaw,
    MiniBatchClusterEvidence,
    MiniBatchKMeans,
)


def minibatch_types(
    evidence: MiniBatchClusterEvidence, revival: ClusterRevival, law: MiniBatchBatchLaw
) -> None:
    recipe = MiniBatchKMeans(
        4, revival=revival, batch_law=law, accumulation="deterministic"
    )
    assert_type(recipe, MiniBatchKMeans)
    assert_type(evidence.updates, Array)
    assert_type(evidence.batch_law, MiniBatchBatchLaw | None)
    MiniBatchKMeans(4, revival="invalid")  # ty: ignore[invalid-argument-type]
    MiniBatchKMeans(4, batch_law="invalid")  # ty: ignore[invalid-argument-type]
