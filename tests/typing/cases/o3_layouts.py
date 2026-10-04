"""O(3) tensor-product plans accept both native layout kinds and nothing else."""

from __future__ import annotations

import numpy as np
from jax import Array
from typing_extensions import assert_type

from phydrax.nn.operator.layers import (
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from phydrax.nn.operator.representations import (
    o3_irrep_action,
    o3_real_coupling,
    O3IrrepBlock,
    O3IrrepLayout,
    O3Representation,
)
from phydrax.special import RealCartesianHarmonics


cartesian = O3Representation(vectors=1)
general = O3IrrepLayout([O3IrrepBlock("a", 3, -1, multiplicity=2)])
cartesian_plan = O3TensorProductPlan(cartesian, cartesian, O3Representation(scalars=1))
general_plan = O3TensorProductPlan(
    general,
    general,
    general,
    paths=[O3TensorProductPath("a", "a", "a", connection_mode="uvu")],
)
assert_type(general_plan.left_representation, O3Representation | O3IrrepLayout)
assert_type(general_plan.paths, tuple[O3TensorProductPath, ...])
assert_type(O3TensorProduct(general_plan, internal_weights=False).weight, Array | None)
assert_type(o3_real_coupling(1, 2, 3).dense(), np.ndarray)
assert_type(o3_irrep_action(np.eye(3), 2, 1), Array)
assert_type(RealCartesianHarmonics(3)(np.ones((2, 3))), Array)

O3TensorProductPlan(cartesian, cartesian, 3)  # ty: ignore[invalid-argument-type]
O3TensorProductPath("a", "a", "a", connection_mode="uuu")  # ty: ignore[invalid-argument-type]
O3IrrepBlock("a", 1, 0)  # ty: ignore[invalid-argument-type]
