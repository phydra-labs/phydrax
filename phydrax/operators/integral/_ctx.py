#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect
from collections.abc import Callable
from typing import Any


def _compile_ctx_integrand(integrand: Callable, /) -> Callable[[dict[str, Any]], Any]:
    """Compile a context-aware integrand dispatcher.

    A single parameter named ``ctx`` or ``context`` receives the whole context.
    Otherwise, parameters select values by name. Legacy field-operator names are
    translated to their canonical context entries.
    """
    parameters = tuple(inspect.signature(integrand).parameters.values())
    if len(parameters) == 1 and parameters[0].name in {"ctx", "context"}:

        def call_context(ctx: dict[str, Any]):
            return integrand(ctx)

        return call_context

    aliases = {
        "value": "uy",
        "delta": "du",
        "delta_value": "du",
        "displacement": "xi",
    }
    arg_names = tuple(
        aliases.get(parameter.name, parameter.name) for parameter in parameters
    )

    def call(ctx: dict[str, Any]):
        return integrand(*tuple(ctx[name] for name in arg_names))

    return call
