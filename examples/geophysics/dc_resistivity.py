"""Finite-patch three-dimensional DC voltage and log-conductivity gradient."""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    )
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
    prepared = phx.applications.geophysics.FinitePatchDCPlan(mesh, survey).prepare()
    parameterization = phx.applications.geophysics.LogConductivity(jnp.ones(2))

    def response(log_conductivity):
        return prepared.predict(parameterization(log_conductivity))[0]

    parameters = jnp.asarray((0.1, -0.2))
    print(
        {
            "voltage_V": float(response(parameters)),
            "gradient_V": np.asarray(jax.grad(response)(parameters)).tolist(),
            "electrode_areas_m2": np.asarray(prepared.electrode_areas).tolist(),
        }
    )


if __name__ == "__main__":
    main()
