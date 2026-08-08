#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.random as jr
from jaxtyping import Array, Key


EvalKey = Key[Array, ""] | None


def require_eval_key(key: EvalKey, /, *, owner: str) -> Key[Array, ""]:
    """Return an evaluation key or reject active stochastic evaluation."""
    if key is None:
        raise ValueError(f"{owner} requires an explicit evaluation key.")
    return key


def fold_in_eval_key(key: EvalKey, site: int | Array, /) -> EvalKey:
    """Derive a stable stochastic-site key without inventing a root key."""
    if key is None:
        return None
    return jr.fold_in(key, site)


def split_eval_key(key: EvalKey, count: int, /) -> Any:
    """Split a call-time key, or propagate deterministic keyless evaluation."""
    if key is None:
        return (None,) * int(count)
    return jr.split(key, int(count))
