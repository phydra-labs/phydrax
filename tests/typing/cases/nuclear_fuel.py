"""Public fuel constructors retain explicit selectors, SI metadata and result types."""

from __future__ import annotations

from typing import assert_type

from phydrax.applications.nuclear_fuel import (
    CladdingSpecification,
    FuelCalibrationPlan,
    FuelRodSpecification,
    PelletSpecification,
    prepare_fuel_least_squares,
    RodSpecification,
)
from phydrax.optim import CompositeLeastSquaresProblem, NonlinearLeastSquaresProblem
from phydrax.units import MILLIMETER


pellet = PelletSpecification(
    radius=4.0,
    height=10.0,
    density_fraction=0.95,
    uranium_235_weight_fraction=0.04,
    grain_radius=5e-6,
    resintering_density_change=0.0,
    roughness=1e-6,
    units={"radius": MILLIMETER, "height": MILLIMETER},
)
cladding = CladdingSpecification(inner_radius=0.0041, outer_radius=0.0047, roughness=1e-6)
rod = RodSpecification(stack_length=1.0, plenum_length=0.1, fill_pressure=2e6)
assert_type(pellet, PelletSpecification)
assert_type(FuelRodSpecification(pellet, cladding, rod), FuelRodSpecification)
assert_type(rod.fill_pressure, float)

PelletSpecification(
    radius=0.004,
    height=0.01,
    density_fraction=0.95,
    uranium_235_weight_fraction=0.04,
    grain_radius=5e-6,
    resintering_density_change=0.0,
    roughness=1e-6,
    form="mox",  # ty: ignore[invalid-argument-type]
)


def calibration_problems(plan: FuelCalibrationPlan) -> None:
    assert_type(prepare_fuel_least_squares(plan), CompositeLeastSquaresProblem)
    assert_type(
        prepare_fuel_least_squares(plan, include_prior=False),
        NonlinearLeastSquaresProblem,
    )
