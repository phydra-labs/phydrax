"""White-box coverage of the BVH packing cost used by PLC row preparation."""

import numpy as np

from phydrax._bvh import _build_split_levels, _median_split, _pack, BVHBuildPolicy


def test_median_pack_accounts_for_allocated_leaf_slots() -> None:
    lower = np.arange(21, dtype=np.float64).reshape((7, 3))
    upper = lower + 0.25
    policy = BVHBuildPolicy(leaf_size=4)
    levels = _build_split_levels(7, 4, _median_split(lower, upper, 0.5 * (lower + upper)))
    events: list[int] = []
    storage: list[int] = []
    packed = _pack(
        lower,
        upper,
        levels,
        policy,
        np.dtype(np.float64),
        _charge_work=events.append,
        _reserve_storage=storage.append,
    )
    node_count = packed.left.size
    payload_slots = packed.leaf_items.size
    assert payload_slots == 8
    assert events[0] == 12 * node_count + 4 * lower.size + 3 * payload_slots
    assert events[0] != 12 * node_count + 4 * lower.size + 3 * (7 * policy.leaf_size)
    assert storage[0] >= packed.leaf_items.nbytes
