"""Ensemble member lifecycle validation."""

from pathlib import Path

import jax
import pytest

from examples.nuclear_fuel_calibration import prepare_synthetic_inference_rod
from phydrax.applications.nuclear_fuel import FuelEnsembleMember, PreparedFuelRod


@pytest.fixture(scope="module")
def prepared() -> PreparedFuelRod:
    with jax.enable_x64():
        return prepare_synthetic_inference_rod()


@pytest.mark.parametrize(
    "time_step", [float("nan"), float("inf"), 0.0], ids=["nan", "infinite", "zero"]
)
def test_ensemble_member_rejects_nonfinite_or_nonpositive_time_step(
    prepared: PreparedFuelRod, tmp_path: Path, time_step: float
) -> None:
    with jax.enable_x64():
        with pytest.raises(ValueError, match="time_step must be finite and positive"):
            FuelEnsembleMember(
                prepared,
                member_id="rejected-step",
                checkpoint_directory=tmp_path,
                time_step=time_step,
                maximum_steps=1,
            )
