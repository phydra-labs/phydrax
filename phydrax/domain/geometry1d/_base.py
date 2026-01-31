#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from typing import Literal

from jaxtyping import Array

from .._base import _AbstractGeometry


class _AbstractGeometry1D(_AbstractGeometry):
    """Abstract 1D geometry."""

    @property
    @abstractmethod
    def length(self) -> Array:
        raise NotImplementedError

    @property
    def spatial_dim(self) -> Literal[1]:
        return 1

    @property
    def volume(self) -> Array:
        return self.length
