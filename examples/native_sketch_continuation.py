#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Constrain, edit, archive, and independently restore a native sketch."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.design import (
    FixedPoint,
    ParameterId,
    PointDistance,
    Sketch,
    SketchBranchWitness,
    SketchContinuationReference,
    continue_sketch,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._model_codec import (
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)


def main() -> None:
    distance = ParameterId("triangle", "radius")
    points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
    sketch = Sketch(
        points,
        feature_id="triangle",
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 2, 2.0, parameter_id=distance),
            PointDistance(1, 2, 2.0),
            PointDistance(0, 1, 2.0),
        ),
    )
    initial = sketch.solve()
    reference = SketchContinuationReference(
        sketch, initial, (SketchBranchWitness((0, 1, 2)),)
    )
    updated = sketch.dimensions.updated({distance: 2.1})
    result = continue_sketch(sketch, reference, updated)
    if not result.accepted:
        raise RuntimeError(f"Sketch edit was rejected: {result.reason}")
    composition = Composition(
        (
            CompositionEntry(
                result.reference,
                entry_id="sketch",
                role="model-state",
                owner_id="phydrax.geometry",
                structure_id="triangle",
                revision_id="triangle-edit",
                semantics_id="continuation",
            ),
        ),
        boundary_id="sketch-workflow",
    )
    with TemporaryDirectory() as directory:
        path = write_model_artifact(
            Path(directory) / "sketch.phx", composition, limits=None
        )
        restored = read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=None,
        ).value("sketch")
        restored.solution.preparation.verify(
            restored.sketch,
            restored.solution.dimensions,
            tolerance=restored.solution.root_evidence.residual_tolerance,
        )
        np.testing.assert_allclose(restored.solution.points, result.attempt.points)
        print("restored branch:", restored.witness_signs)
        print("root residual:", float(restored.solution.residual_norm))


if __name__ == "__main__":
    main()
