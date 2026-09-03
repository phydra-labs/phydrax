#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    coordinates = jnp.asarray(
        tuple((0.5 * i, 0.5 * j) for j in range(3) for i in range(3))
    )
    cells = (
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (3, 4, 7, 6),
        (4, 5, 8, 7),
    )
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, cells)
    field = phx.discretization.ExplicitPolygonH1FieldSpec("u", component_shape=(2,))
    discretization = phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()
    constraint = phx.discretization.explicit_polygon_h1_dirichlet_constraint(
        discretization, "u"
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters(1.0, 2.0)
    stored_energy = phx.applications.solid_mechanics.neo_hookean_functional(
        "u", parameters
    )

    def load_density(fields, geometry, context):
        del geometry, context
        return -0.05 * fields["u"].value[..., 1]

    body_load = phx.variational.LocalIntegralTerm(
        "body-load",
        region="body",
        fields=(phx.variational.FieldJetSpec("u", value=True),),
        density=load_density,
        density_id="explicit-polygon-vertical-body-load",
    )
    functional = phx.variational.Functional(
        "explicit-polygon-loaded-neo-hookean",
        stored_energy.terms + (body_load,),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"body": None},
        constraint=constraint,
        dirichlet_values=0.0,
    )
    result = phx.optim.minimize(
        compiled.as_minimization_problem(),
        compiled.state_space.zeros(),
        method=phx.optim.NewtonKrylov(),
        termination=phx.optim.OptimizationTermination(maximum_steps=20),
    )
    residual = compiled.residual(result.parameters)
    maximum_basis_error = max(
        float(jnp.max(block.evidence.affine_gradient_error))
        for block in discretization.default_runtime.bases
    )
    print("successful", bool(result.successful))
    print("residual_norm", float(jnp.sqrt(jnp.sum(residual * residual))))
    print("maximum_affine_gradient_error", maximum_basis_error)


if __name__ == "__main__":
    main()
