#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

import abc

from jaxtyping import Array, ArrayLike

from ..._strict import AbstractAttribute, StrictModule


class _AbstractScaler(StrictModule):
    """Abstract base class for data scalers."""

    @abc.abstractmethod
    def transform(self, x: ArrayLike) -> Array:
        """Transform input data."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, x: ArrayLike) -> Array:
        """Map transformed data back to the original scale."""
        raise NotImplementedError


class _AbstractScalerSpecifier(_AbstractScaler):
    """Wrapper base class for specific scalers backed by an affine scaler."""

    scaler: AbstractAttribute[_AbstractScaler]

    def transform(self, x: ArrayLike) -> Array:
        return self.scaler.transform(x)

    transform.__doc__ = _AbstractScaler.transform.__doc__

    def inverse_transform(self, x: ArrayLike) -> Array:
        return self.scaler.inverse_transform(x)

    inverse_transform.__doc__ = _AbstractScaler.inverse_transform.__doc__
