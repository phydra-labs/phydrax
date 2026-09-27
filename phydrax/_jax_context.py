from __future__ import annotations

from jax.extend.core import get_opaque_trace_state


_EAGER_TRACE_STATE = get_opaque_trace_state()


def inside_jax_transformation() -> bool:
    """Return whether execution is tracing under a JAX transformation."""
    return get_opaque_trace_state() != _EAGER_TRACE_STATE
