"""Two-time infiltration-to-ERT posterior on one physical source parameter."""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _geometry():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    tetrahedra = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    mesh = phx.discretization.CellMesh.from_tetrahedra(points, tetrahedra)
    finite_volume = phx.discretization.UnstructuredFiniteVolumePlan(
        points, tetrahedra=tetrahedra
    ).prepare()
    return mesh, finite_volume


def _electrical(mesh):
    exterior = np.flatnonzero(np.asarray(mesh.connectivity.boundary_faces))
    patches = tuple(
        phx.applications.geophysics.ElectrodePatch(f"E{index}", (int(face),))
        for index, face in enumerate(exterior[:4])
    )
    survey = phx.applications.geophysics.ElectricalSurvey(
        patches,
        jnp.asarray(((1.0, -1.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, -1.0, 0.0, 0.0),)),
        jnp.asarray((0,)),
    )
    return phx.applications.geophysics.FinitePatchDCPlan(mesh, survey).prepare()


def main() -> None:
    mesh, finite_volume = _geometry()
    exterior = np.flatnonzero(np.asarray(finite_volume.neighbour_cells) < 0)
    porosity = jnp.asarray((0.3, 0.4))
    water = phx.applications.porous_media.RichardsPlan(
        finite_volume,
        phx.applications.porous_media.PorousMaterial(porosity, 1.0e-12),
        phx.applications.porous_media.VanGenuchtenMualem(1.0e-5, 2.0),
        phx.applications.porous_media.PorousBoundaryConditions(
            finite_volume,
            pressure_Pa={int(face): -2.0e4 for face in exterior},
        ),
        gravity_m_s2=(0.0, 0.0, 0.0),
        method=phx.nonlinear.NewtonKrylov(
            linear_policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
        ),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=40,
        ),
    )
    initial = water.initialize(-2.0e4)
    coordinates = phx.interchange.GeospatialContract.local_cartesian(
        phx.SpatialCoordinateContract.si(), vertical_datum="local-survey-datum"
    )
    transfer = phx.discretization.nested_cell_transfer(
        finite_volume, finite_volume, np.arange(2)
    )
    hydro = phx.applications.geophysics.HydrogeophysicalPlan(
        _electrical(mesh),
        transfer,
        finite_volume.cell_volumes,
        finite_volume.cell_volumes,
        source_geometry_id=finite_volume.geometry_id,
        target_geometry_id=finite_volume.geometry_id,
        source_coordinates=coordinates,
        target_coordinates=coordinates,
    )
    calibration = phx.applications.geophysics.ArchieSaturationConductivity(
        0.5,
        cementation_exponent=2.0,
        saturation_exponent=2.2,
    )

    def predict(log_source):
        source = jnp.exp(log_source)
        first = water.step(initial, 50.0, source_kg_s=jnp.asarray((source, 0.0)))
        second = water.step(first.state, 50.0, source_kg_s=jnp.asarray((source, 0.0)))
        first_voltage = hydro.predict(
            porosity,
            first.state.water_volume_m3,
            first.state.temperature_K,
            calibration,
        ).response
        second_voltage = hydro.predict(
            porosity,
            second.state.water_volume_m3,
            second.state.temperature_K,
            calibration,
        ).response
        return jnp.concatenate((first_voltage, second_voltage))

    truth = jnp.log(jnp.asarray(1.0e-2))
    observed = predict(truth)
    noise = jnp.maximum(1.0e-8, 1.0e-4 * jnp.max(jnp.abs(observed)))
    likelihood = phx.uq.GaussianLikelihood(noise)
    parameter_space = phx.uq.ParameterSpace(
        truth,
        log_prior=lambda value: -0.5 * ((value - truth) / 2.0) ** 2,
    )
    posterior = phx.uq.PosteriorProblem(
        parameter_space,
        lambda value: jnp.sum(likelihood.log_prob(predict(value), observed)),
        predict=predict,
    )
    candidates = truth + jnp.linspace(-0.5, 0.5, 11)
    densities = jax.vmap(posterior.log_density)(candidates)
    estimate = candidates[jnp.argmax(densities)]
    gradient = jax.grad(posterior.log_density)(truth)
    print(
        {
            "true_infiltration_kg_s": float(jnp.exp(truth)),
            "estimated_infiltration_kg_s": float(jnp.exp(estimate)),
            "time_lapse_voltage_V": np.asarray(observed).tolist(),
            "truth_log_posterior_gradient": float(gradient),
        }
    )


if __name__ == "__main__":
    main()
