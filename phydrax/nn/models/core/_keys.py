#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.random as jr
from jaxtyping import Array, Key


EvalKey = Key[Array, ""] | None


def split_eval_key(key: EvalKey, count: int, /) -> Any:
    """Split a call-time key, or propagate deterministic keyless evaluation."""
    if key is None:
        return (None,) * int(count)
    return jr.split(key, int(count))
