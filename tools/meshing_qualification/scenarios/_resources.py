"""Stage resource measurement shared by the qualification scenarios.

The process high-water memory comes from its canonical owner,
`phydrax._execution_sampling.process_peak_memory`, on every operating system.
An unavailable measurement is refused with the substrate's reason.
"""

from collections.abc import Callable
from time import perf_counter

from phydrax._execution_sampling import process_peak_memory


def peak_resident_bytes() -> int:
    """The process peak resident bytes; refuses when the platform cannot observe it."""
    peak = process_peak_memory()
    if peak.value_bytes is None:
        raise RuntimeError(
            f"Meshing resource qualification has no process peak memory: {peak.unavailable_reason}"
        )
    return peak.value_bytes


def measured(call: Callable[[], object], /) -> tuple[object, dict[str, float]]:
    """Run one stage; report wall seconds and the process peak resident bytes."""
    started = perf_counter()
    value = call()
    elapsed = perf_counter() - started
    return value, {"wall_seconds": elapsed, "peak_resident_bytes": peak_resident_bytes()}
