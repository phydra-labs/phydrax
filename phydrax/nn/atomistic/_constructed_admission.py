#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host admission of restored components against their constructors.

A restored or transformed component bypasses its constructor. Its owner
re-runs the constructor abstractly (``equinox.filter_eval_shape``) from the
validated declaration and requires the stored value to be exactly what that
constructor builds: equal tree structure and every non-array field (sizes,
value shapes, activations, weight transforms, static metadata) equal, and
every array of the template's shape in the declared dtype with finite values.
No numerical array is allocated for the template. Scale and precision
contracts are rebuilt by their own constructors and must be duplicated exactly.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import numpy as np

from ..._model._fixed_binding import require_identical
from ...atomistic._types import AtomisticPrecisionPolicy, AtomisticScaleContract


def _is_template_array(value: Any, /) -> bool:
    return isinstance(value, jax.ShapeDtypeStruct)


def require_constructed(
    observed: Any, template: Any, dtype: np.dtype[Any], name: str, /
) -> None:
    """Refuse ``observed`` unless it is a finite ``dtype`` instance of ``template``.

    ``template`` is the abstract constructor value of the declaration. This is a
    host boundary over concrete arrays and never runs under tracing.
    """
    observed_arrays, observed_static = eqx.partition(observed, eqx.is_array)
    template_arrays, template_static = eqx.partition(template, _is_template_array)
    observed_fields, observed_structure = jax.tree_util.tree_flatten(observed_static)
    template_fields, template_structure = jax.tree_util.tree_flatten(template_static)
    if observed_structure != template_structure or any(
        type(left) is not type(right) or left != right
        for left, right in zip(observed_fields, template_fields, strict=True)
    ):
        raise ValueError(f"{name} differs from its constructor declaration.")
    observed_leaves, observed_layout = jax.tree_util.tree_flatten(observed_arrays)
    template_leaves, template_layout = jax.tree_util.tree_flatten(template_arrays)
    if observed_layout != template_layout:
        raise ValueError(f"{name} arrays differ from its constructor declaration.")
    for leaf, spec in zip(observed_leaves, template_leaves, strict=True):
        if tuple(leaf.shape) != tuple(spec.shape):
            raise ValueError(f"{name} array shapes differ from its declaration.")
        if np.dtype(leaf.dtype) != dtype:
            raise ValueError(f"{name} arrays are not in the declared {dtype.name} dtype.")
        if not np.all(np.isfinite(np.asarray(jax.device_get(leaf)))):
            raise ValueError(f"{name} arrays must be finite.")


def require_atomistic_contracts(
    scale: AtomisticScaleContract, precision: AtomisticPrecisionPolicy, name: str, /
) -> None:
    """Refuse scale or precision contracts their constructors do not rebuild."""
    if not isinstance(scale, AtomisticScaleContract):
        raise TypeError(f"{name} scale must be an AtomisticScaleContract.")
    if not isinstance(precision, AtomisticPrecisionPolicy):
        raise TypeError(f"{name} precision must be an AtomisticPrecisionPolicy.")
    require_identical(
        scale,
        AtomisticScaleContract(scale.length_unit, scale.energy_unit),
        f"{name} scale contract",
        ValueError,
    )
    require_identical(
        precision,
        AtomisticPrecisionPolicy(
            coordinate_dtype=precision.coordinate_dtype,
            compute_dtype=precision.compute_dtype,
            reduction_dtype=precision.reduction_dtype,
            output_dtype=precision.output_dtype,
        ),
        f"{name} precision policy",
        ValueError,
    )


__all__ = ["require_atomistic_contracts", "require_constructed"]
