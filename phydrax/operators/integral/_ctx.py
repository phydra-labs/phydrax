#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect
from collections.abc import Callable
from typing import Any


def _compile_ctx_integrand(integrand: Callable, /) -> Callable[[dict[str, Any]], Any]:
    """Compile a ctx-dict integrand dispatcher.

    The returned callable takes a context dict and calls `integrand` with only the
    parameters it declares, in signature order.
    """
    arg_names = tuple(inspect.signature(integrand).parameters.keys())

    def call(ctx: dict[str, Any]):
        return integrand(*tuple(ctx[name] for name in arg_names))

    return call
