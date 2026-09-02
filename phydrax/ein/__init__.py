#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Einstein-style contractions and static JAX axis transformations."""

from opt_einsum import contract

from ._transform import rearrange, reduce, repeat


__all__ = ["contract", "rearrange", "reduce", "repeat"]
