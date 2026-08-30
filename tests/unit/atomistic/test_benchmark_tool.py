import json

import numpy as np

from tools.atomistic_rmd17_benchmarks import _json_safe


def test_benchmark_nonfinite_failures_are_structured_strict_json():
    payload = {
        "metrics": {
            "energy_mae_per_atom": float("nan"),
            "force_component_mae": float("inf"),
            "finite": np.float64(0.25),
        },
        "gates": {"passed": False},
    }
    safe = _json_safe(payload)
    assert safe["metrics"]["energy_mae_per_atom"] == {
        "kind": "nonfinite",
        "value": "nan",
    }
    assert safe["metrics"]["force_component_mae"] == {
        "kind": "nonfinite",
        "value": "positive_infinity",
    }
    encoded = json.dumps(safe, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
