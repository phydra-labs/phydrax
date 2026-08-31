#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


first = jr.normal(jr.key(3), (32, 32))
second = jnp.zeros_like(first).at[1:, 1:].set(first[:-1, :-1])
geometry = phx.velocimetry.imaging.ImageGeometry2D(first.shape)
plan = phx.velocimetry.piv.PIVPlan(
    (phx.velocimetry.piv.PIVPassPlan(16, 8, 3),),
    minimum_valid_fraction=0.5,
    minimum_peak_ratio=0.0,
    minimum_correlation=-1.0,
    minimum_neighbors=0,
    replacement_iterations=0,
)
result = phx.velocimetry.piv.piv(first, second, plan, geometry=geometry)

with TemporaryDirectory() as directory:
    archive_path = Path(directory) / "translation.phxv"
    phx.velocimetry.io.write_velocimetry_archive(
        archive_path,
        result,
        value_kind="piv-result",
        provenance={"experiment": "synthetic-translation"},
    )
    restored = phx.velocimetry.io.read_velocimetry_archive(
        archive_path,
        expected_kind="piv-result",
        expected_type=phx.velocimetry.piv.PIVResult,
    )

print(
    "raw vectors preserved",
    bool(jnp.array_equal(restored.value.raw.displacement_rc, result.raw.displacement_rc)),
)
print(
    "validity preserved",
    bool(jnp.array_equal(restored.value.raw.valid, result.raw.valid)),
)
print("provenance", restored.provenance)
