"""Scientific benchmark gates reject incorrect/nonfinite evidence, not just API errors."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks._io import write_json_atomic
from benchmarks.cuest_molecular import _finite_metric, matched_df_checks


def _checks(
    component: str = "",
    value: float = 0.0,
    *,
    converged: bool = True,
    jk_successful: bool = True,
) -> dict[str, bool]:
    forces = np.zeros((2, 3), dtype=np.float64)
    coulomb = np.eye(2, dtype=np.float64)
    exchange = 0.5 * np.eye(2, dtype=np.float64)
    energy = -1.0
    if component == "energy":
        energy = value
    elif component == "forces":
        forces[0, 0] = value
    elif component == "coulomb":
        coulomb[0, 0] = value
    elif component == "exchange":
        exchange[0, 0] = value
    return matched_df_checks(
        energy=energy,
        reference_energy=-1.0,
        forces=forces,
        reference_forces=np.zeros((2, 3), dtype=np.float64),
        coulomb=coulomb,
        reference_coulomb=np.eye(2, dtype=np.float64),
        exchange=exchange,
        reference_exchange=0.5 * np.eye(2, dtype=np.float64),
        reference_converged=converged,
        native_jk_successful=jk_successful,
    )


def test_matched_benchmark_accepts_only_qualified_finite_reference() -> None:
    assert all(_checks().values())
    assert not all(_checks(converged=False).values())
    assert not all(_checks(jk_successful=False).values())


@pytest.mark.parametrize("component", ("energy", "forces", "coulomb", "exchange"))
@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_matched_benchmark_rejects_nonfinite_scientific_outputs(
    component: str, value: float
) -> None:
    assert not all(_checks(component, value).values())


@pytest.mark.parametrize(
    "component,value",
    (("energy", -0.9), ("forces", 1.0e-3), ("coulomb", 1.1), ("exchange", 0.6)),
)
def test_matched_benchmark_rejects_finite_but_incorrect_outputs(
    component: str, value: float
) -> None:
    assert not all(_checks(component, value).values())


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_nonfinite_failure_metrics_preserve_finite_json_record(
    invalid: float,
    tmp_path: Path,
) -> None:
    payload = {
        "status": "failed",
        "scientific_checks": _checks("energy", invalid),
        "independent_reference": {"df_energy_error_hartree": _finite_metric(invalid)},
    }
    path = write_json_atomic(tmp_path / "failed-benchmark.json", payload)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["status"] == "failed"
    assert restored["scientific_checks"]["energies_finite"] is False
    assert restored["independent_reference"]["df_energy_error_hartree"] is None
