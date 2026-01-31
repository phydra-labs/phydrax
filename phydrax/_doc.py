#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.random as jr


DOC_KEY0 = eqx.internal.doc_repr(jr.key(0), "jr.key(0)")

__all__ = ["DOC_KEY0"]
