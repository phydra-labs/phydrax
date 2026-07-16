#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

from jaxtyping import Array

from ._base import _AbstractScaler


def scaler_transform_fn(
    fn: Callable[[Array], Array],
    *,
    input_scaler: _AbstractScaler | None = None,
    output_scaler: _AbstractScaler | None = None,
) -> Callable[[Array], Array]:
    """Wrap a function with optional input scaling and output unscaling."""

    def transform_fn(x: Array) -> Array:
        x_scaled = input_scaler.transform(x) if input_scaler is not None else x
        output = fn(x_scaled)
        if output_scaler is not None:
            return output_scaler.inverse_transform(output)
        return output

    return transform_fn
