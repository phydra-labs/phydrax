#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Constrain, edit, archive, and independently restore a native sketch."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.design import (
    continue_sketch,
    FixedPoint,
    ParameterId,
    PointDistance,
    Sketch,
    SketchBranchWitness,
    SketchContinuationReference,
)
from phydrax.interchange import load_composition_archive, save_composition_archive
from phydrax.lifecycle import Composition, CompositionEntry


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
        path = save_composition_archive(composition, Path(directory) / "sketch.phx")
        restored = load_composition_archive(path).value("sketch")
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
