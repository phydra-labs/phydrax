#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

from jaxtyping import Array

from ..._strict import AbstractAttribute
from .._base import _AbstractGeometry


class _AbstractGeometry3D(_AbstractGeometry):
    """Abstract 3D geometry."""

    mesh_vertices: AbstractAttribute[Array]

    @property
    def spatial_dim(self) -> Literal[3]:
        return 3
