from typing import Any

import numpy as np


def _record_array(checks: dict[str, Any], key: str, actual: Any, expected: Any) -> None:
    # Preserve rank and dtype, including empty arrays and scalar broadcasting.
    arrays = tuple(np.asarray(value) for value in (actual, expected))
    checks.setdefault(key, []).append(
        [
            {
                "values": value.tolist(),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            for value in arrays
        ]
    )


def _new_checks(keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: [] for key in keys}


def _require_present[T](value: T | None) -> T:
    assert value is not None, "Scenario prerequisite must be present"
    return value


def _require_instance[T](value: object, expected: type[T]) -> T:
    assert isinstance(value, expected), (
        f"Scenario prerequisite must be {expected.__name__}; got {type(value).__name__}"
    )
    return value
