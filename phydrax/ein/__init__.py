#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Einstein-style contractions and static JAX axis transformations."""

from opt_einsum import contract, get_symbol

from ._transform import rearrange, reduce, repeat


__all__ = ["contract", "get_symbol", "rearrange", "reduce", "repeat"]
