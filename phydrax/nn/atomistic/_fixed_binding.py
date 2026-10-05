#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host comparisons binding fixed data to its owning reconstruction.

Restored or published fixed data (bases, plans, merged coefficients, folds,
tables) carries recorded identities that a constructor-bypassing restore does
not re-derive. Owners therefore rebuild the expected value from validated
sources and compare it with what executes:

* `require_identical` is for duplicated data (a copy of source structure or
  parameters): equal tree structure, including every static field, and
  bitwise-equal leaves of equal dtype;
* `require_realized` is for derived numerical realizations (a fold, a product,
  a tabulation) that a reconstruction reproduces only to rounding: equal
  structure and dtype, exact integer leaves, finite floating leaves first, and
  then agreement within a dtype-relative rounding bound. A nonfinite observed
  value never passes.

Both are host boundaries over concrete arrays and never run under tracing.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


# Relative rounding bound of a recomputed derived realization, scaled by the
# largest expected magnitude (at least one): folds and products reorder
# floating-point sums, so an exact recomputation reproduces them only to this.
REALIZATION_RELATIVE_TOLERANCE = {
    np.dtype(np.float32): 1.0e-5,
    np.dtype(np.float64): 1.0e-12,
}


def _host_leaves(tree: Any, /) -> tuple[Any, list[np.ndarray]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    return structure, [np.asarray(jax.device_get(leaf)) for leaf in leaves]


def require_identical(
    observed: Any, expected: Any, name: str, error: type[ValueError], /
) -> None:
    """Refuse ``observed`` unless it duplicates ``expected`` exactly."""
    observed_structure, observed_leaves = _host_leaves(observed)
    expected_structure, expected_leaves = _host_leaves(expected)
    if observed_structure != expected_structure:
        raise error(f"{name}: structure differs from its source.")
    for left, right in zip(observed_leaves, expected_leaves, strict=True):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise error(f"{name}: values differ from its source.")


def require_realized(
    observed: Any, expected: Any, name: str, error: type[ValueError], /
) -> None:
    """Refuse ``observed`` unless it realizes ``expected`` to rounding."""
    observed_structure, observed_leaves = _host_leaves(observed)
    expected_structure, expected_leaves = _host_leaves(expected)
    if observed_structure != expected_structure:
        raise error(f"{name}: structure differs from its source.")
    for left, right in zip(observed_leaves, expected_leaves, strict=True):
        if left.dtype != right.dtype or left.shape != right.shape:
            raise error(f"{name}: dtype or shape differs from its source.")
        if not np.issubdtype(right.dtype, np.inexact):
            if not np.array_equal(left, right):
                raise error(f"{name}: values differ from its source.")
            continue
        if not np.all(np.isfinite(left)):
            raise error(f"{name}: values must be finite.")
        if not np.all(np.isfinite(right)):
            raise error(f"{name}: the source reconstruction is nonfinite.")
        tolerance = REALIZATION_RELATIVE_TOLERANCE.get(right.dtype)
        if tolerance is None:
            raise TypeError(f"{name} must use float32 or float64.")
        bound = tolerance * max(1.0, float(np.max(np.abs(right), initial=0.0)))
        if not float(np.max(np.abs(left - right), initial=0.0)) <= bound:
            raise error(f"{name}: values differ from their source beyond rounding.")


__all__ = ["REALIZATION_RELATIVE_TOLERANCE", "require_identical", "require_realized"]
