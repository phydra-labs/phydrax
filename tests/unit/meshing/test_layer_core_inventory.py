from __future__ import annotations

import numpy as np

from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
    _layer_core_region_inventory,
)


def test_region_inventory_preserves_small_refined_cell_contributions() -> None:
    # Each material retains its bulk cell and splits an exactly representable
    # small content into 8192 cells. Sequential accumulation loses every child.
    source_regions = np.asarray((0, 1, 0, 1), dtype=np.int64)
    source_content = np.asarray((1.0, 2.0, 2.0**-41, 2.0**-40), dtype=np.float64)
    target_regions = np.tile(np.asarray((0, 1), dtype=np.int64), 8193)
    target_content = np.concatenate(
        (
            np.asarray((1.0, 2.0), dtype=np.float64),
            np.tile(np.asarray((2.0**-54, 2.0**-53), dtype=np.float64), 8192),
        )
    )
    expected = np.asarray((1.0 + 2.0**-41, 2.0 + 2.0**-40), dtype=np.float64)
    source_inventory = _layer_core_region_inventory(source_regions, source_content, 2)
    target_inventory = _layer_core_region_inventory(target_regions, target_content, 2)
    np.testing.assert_array_equal(source_inventory, expected)
    np.testing.assert_array_equal(target_inventory, expected)
