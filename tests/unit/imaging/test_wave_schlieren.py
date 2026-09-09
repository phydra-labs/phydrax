import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _space(*, periodic=False, shape=(12, 12)):
    axis = (
        phx.discretization.FourierAxisSpec
        if periodic
        else phx.discretization.UniformAxisSpec
    )
    grid = phx.discretization.TensorGridPlan(
        (axis(shape[0]), axis(shape[1])), axis_names=("u", "v")
    ).prepare(jnp.asarray(((-1.0, -1.0), (1.0, 1.0))))
    return phx.optics.wave.PlaneFieldSpace(
        grid,
        phx.geometry.RigidFrame.identity(3),
        "periodic-cell" if periodic else "finite-window",
    )


def test_uniform_phase_screen_and_open_filter_preserve_uniform_intensity():
    space = _space()
    screen = phx.imaging.RefractivePhaseScreenPlan(space, 20.0)
    transmission, evidence = screen.evaluate(jnp.full(space.shape, 0.02))
    assert bool(evidence.successful)
    field = phx.optics.wave.ScalarPlaneField(space, jnp.ones(space.shape), 20.0, 0.0)
    filtered = phx.optics.wave.ScalarThinTransmission(space, jnp.ones(space.shape))
    result = phx.imaging.WaveSchlierenPlan(
        screen,
        filtered,
        object_distance=0.0,
        detector_distance=0.0,
        padding=2,
        maximum_leakage_fraction=1.0,
    ).evaluate(field, jnp.zeros(space.shape))
    np.testing.assert_allclose(result.detector_intensity.values, 1.0, atol=1e-6)
    assert bool(result.evidence.successful)
    assert transmission.transmission.shape == space.shape


def test_multislice_zero_perturbation_and_helmholtz_zero_susceptibility():
    space = _space(periodic=True)
    incident = phx.optics.wave.ScalarPlaneField(space, jnp.ones(space.shape), 10.0, 0.0)
    multislice = phx.imaging.MultisliceRefractivePlan(
        space,
        np.asarray((0.1, 0.1)),
        4.0,
        padding=None,
        maximum_phase_per_slice=1.0,
        maximum_leakage_fraction=1.0,
    )
    result = multislice.evaluate(incident, jnp.zeros((2,) + space.shape))
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(jnp.abs(result.field.values), 1.0, atol=1e-5)

    helmholtz = phx.imaging.ScalarHelmholtzContinuationPlan(
        space,
        4.0,
        damping=0.2,
        iteration_count=3,
        relative_tolerance=1e-6,
    )
    solved = helmholtz.solve(jnp.ones(space.shape), jnp.zeros(space.shape), 10.0)
    assert bool(solved.evidence.successful)
    np.testing.assert_allclose(solved.evidence.residual_norm, 0.0, atol=1e-6)
