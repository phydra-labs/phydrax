"""Ensemble member lifecycle validation."""

from pathlib import Path

import pytest

from examples.nuclear_fuel_calibration import prepare_synthetic_inference_rod
from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel import FuelEnsembleMember, PreparedFuelRod


@pytest.fixture(scope="module")
def prepared() -> PreparedFuelRod:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        return prepare_synthetic_inference_rod()


@pytest.mark.parametrize(
    "time_step", [float("nan"), float("inf"), 0.0], ids=["nan", "infinite", "zero"]
)
def test_ensemble_member_rejects_nonfinite_or_nonpositive_time_step(
    prepared: PreparedFuelRod, tmp_path: Path, time_step: float
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        with pytest.raises(ValueError, match="time_step must be finite and positive"):
            FuelEnsembleMember(
                prepared,
                member_id="rejected-step",
                checkpoint_directory=tmp_path,
                time_step=time_step,
                maximum_steps=1,
            )
