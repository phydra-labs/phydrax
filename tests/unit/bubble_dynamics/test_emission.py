import numpy as np
import pytest

import phydrax.bubble_dynamics as bd


AMBIENT = 101325.0
DENSITY = 998.0
SOUND_SPEED = 1481.0


def _model() -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        "keller_miksis",
        bd.PolytropicBubbleGasLaw(1.4),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        bd.CleanBubbleInterfaceLaw(0.072),
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=SOUND_SPEED,
    )


def test_far_field_pressure_matches_the_linear_monopole_amplitude() -> None:
    radius, amplitude, frequency = 5.0e-6, 50.0, 3.0e5
    omega = 2.0 * np.pi * frequency
    period = 1.0 / frequency
    distance = 2.0e-3
    delay = distance / SOUND_SPEED
    window = 10.0 * period
    end = 20.0 * period + delay
    emission_times = np.linspace(end - window, end, 801)
    early = np.array([0.5 * delay])
    plan = bd.BubbleCloudPlan(
        (
            bd.BubbleSpeciesGroup(
                _model(), np.array([radius]), np.zeros((1, 3)), bubble_ids=(7,)
            ),
        ),
        bd.HarmonicPressureDrive(amplitude, omega),
        np.linspace(0.0, end, 401)[1:],
        emission=bd.FarFieldEmissionPlan(
            np.array([[distance, 0.0, 0.0]]), np.concatenate((early, emission_times))
        ),
    )
    result = bd.solve_bubble_cloud(plan.prepare())
    assert bool(result.completed)
    emission = result.emission
    assert emission is not None
    covered = np.asarray(emission.evidence.covered)[:, 0]
    pressure = np.asarray(emission.pressure)[:, 0]
    # A retarded time before the start is uncovered, never extrapolated.
    assert not covered[0] and np.isnan(pressure[0])
    assert np.all(covered[1:])
    assert float(emission.evidence.minimum_distance_ratio) > 100.0
    # Harmonic amplitude over an integer number of periods (trapezoidal projection).
    samples = pressure[1:]
    phase = np.exp(-1j * omega * emission_times)
    weights = np.full(emission_times.shape, emission_times[1] - emission_times[0])
    weights[[0, -1]] *= 0.5
    measured = 2.0 * abs(np.sum(weights * samples * phase)) / window
    response = bd.linear_bubble_response(_model(), radius, np.array([omega]))
    expected = (
        DENSITY
        * omega**2
        * radius**2
        * abs(complex(response.radius_response[0]))
        * amplitude
        / distance
    )
    assert measured == pytest.approx(expected, rel=2.0e-2)


def test_emission_plan_refuses_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="observers"):
        bd.FarFieldEmissionPlan(np.zeros((2, 2)), np.array([1.0e-6]))
    with pytest.raises(ValueError, match="times"):
        bd.FarFieldEmissionPlan(np.zeros((1, 3)), np.array([np.nan]))
    group = bd.BubbleSpeciesGroup(
        _model(), np.array([5.0e-6]), np.zeros((1, 3)), bubble_ids=(0,)
    )
    with pytest.raises(ValueError, match="fixed bubble positions"):
        bd.BubbleCloudPlan(
            (group,),
            bd.ConstantPressureDrive(0.0),
            np.array([1.0e-6]),
            translation=bd.BubbleTranslation(1.0e-3),
            emission=bd.FarFieldEmissionPlan(np.ones((1, 3)), np.array([1.0e-6])),
        )
