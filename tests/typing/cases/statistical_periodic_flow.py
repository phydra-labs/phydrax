"""Statistical periodic-flow construction stays typed at the public facades."""

from typing import assert_type

from jax import Array

from phydrax.applications.incompressible_flow import (
    PeriodicFlowInitialMeasure,
    PeriodicFlowInitialState,
    PreparedPeriodicFlowInitialMeasure,
    restrict_periodic_flow_initial_state,
    ShearLayerFamily,
)
from phydrax.discretization import (
    PeriodicIncrementMomentResult,
    PeriodicRadialIncrementMomentPlan,
    PreparedSpectralVanishingViscosity,
    SpectralVanishingViscosityPlan,
    TensorSpectralDiscretization,
)
from phydrax.solver import CourantGuardedETDRKMethod, ETDRKMethod
from phydrax.stochastic import (
    PeriodicPowerLawFieldPlan,
    PeriodicPowerLawFieldRealization,
    PreparedPeriodicPowerLawField,
)
from phydrax.typing import PRNGKey


def statistical_periodic_flow_facade(
    space: TensorSpectralDiscretization,
    coarse: TensorSpectralDiscretization,
    key: PRNGKey,
    members: Array,
    coefficients: Array,
    radii: Array,
) -> None:
    viscosity = SpectralVanishingViscosityPlan(
        strength=0.05,
        order=1.0,
        cutoff_scale=1.0,
        cutoff_exponent=0.25,
        kernel="tadmor_step",
    )
    assert_type(viscosity.prepare(space), PreparedSpectralVanishingViscosity)
    assert_type(viscosity.prepare(space).dissipation(coefficients), Array)
    SpectralVanishingViscosityPlan(
        strength=0.05,
        order=1.0,
        cutoff_scale=1.0,
        cutoff_exponent=0.25,
        kernel="gaussian",  # ty: ignore[invalid-argument-type]
    )

    assert_type(
        CourantGuardedETDRKMethod(ETDRKMethod(4), maximum_courant_number=0.5),
        CourantGuardedETDRKMethod,
    )

    field = PeriodicPowerLawFieldPlan.fractional_gaussian(
        0.5, spatial_dimension=2, component_shape=(2,), solenoidal=True
    ).prepare(space)
    assert_type(field, PreparedPeriodicPowerLawField)
    realization = field.sample(key, members)
    assert_type(realization, PeriodicPowerLawFieldRealization)
    assert_type(field.restrict(realization, coarse), PeriodicPowerLawFieldRealization)

    measure = PeriodicFlowInitialMeasure(
        ShearLayerFamily(
            profile="smooth",
            thickness_range=(0.05, 0.05),
            perturbation_amplitude=0.025,
            perturbation_mode_range=(1, 10),
            interface_shift_range=(0.0, 0.0),
            oversampling=4,
        )
    ).prepare(space)
    assert_type(measure, PreparedPeriodicFlowInitialMeasure)
    state = measure.realize(key, members)
    assert_type(state, PeriodicFlowInitialState)
    assert_type(
        restrict_periodic_flow_initial_state(state, coarse), PeriodicFlowInitialState
    )

    radial = PeriodicRadialIncrementMomentPlan(
        space, radii, averaging="ball", component_shape=(2,)
    )
    assert_type(radial(state.velocity), PeriodicIncrementMomentResult)
