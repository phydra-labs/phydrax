from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.fem._generic import FiniteElementDiscretization
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from tools.meshing_qualification.scenarios import native_planar_profiles as profiles


def test_cylinder_source_metric_and_carrier_are_distinct() -> None:
    profile = "native-surface-pde:cylinder_sheet"
    domain = profiles._surface_definition(profile, "cylinder_sheet").domain
    scope = profiles.MeshingScope.whole(domain, 2)
    specification = profiles.SurfaceMeshingSpec(
        profiles.CellMeshingTarget(
            2, 3, profiles.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            profiles.UniformSizeControl(
                scope, 0.25, strength=profiles.SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=0.02,
    )
    result = (
        profiles.NativeMeshingProvider(
            profiles.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            domain,
            specification,
            coordinate_contract=profiles.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert result.certification is not None
    assert result.certification.inputs.source is not None
    source = phx.discretization.SurfaceFiniteElementGeometry.from_result(
        result, result.certification.inputs.source
    )
    for exact_block, bounds in zip(
        source.exact_parameters, source.parameter_rounding_bounds, strict=True
    ):
        for corners, bound in zip(exact_block, bounds, strict=True):
            assert all(
                abs(value - Fraction.from_float(float(value)))
                <= Fraction.from_float(bound)
                for point in corners
                for value in point
            )
    with pytest.raises(ValueError, match="H1 P1"):
        profiles.FiniteElementPlan(
            result.mesh,
            profiles.FiniteElementFieldSpec(
                "u", profiles.lagrange_element("triangle", 2)
            ),
            coordinate_spec=result.geometry,
            source_surface=source,
        )
    field = profiles.FiniteElementFieldSpec("u", profiles.lagrange_element("triangle", 1))
    source_space = profiles.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry, source_surface=source
    ).prepare()
    carrier_space = profiles.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    source_block = source_space.block_geometries[0][0]
    carrier_block = carrier_space.block_geometries[0][0]
    points = np.asarray(source_block.physical_points)
    np.testing.assert_allclose(np.linalg.norm(points[..., :2], axis=-1), 1.0, atol=2e-15)
    np.testing.assert_allclose(
        float(jnp.sum(source_block.physical_weights)), np.pi / 2, atol=2e-12
    )
    assert float(jnp.sum(carrier_block.physical_weights)) < np.pi / 2 - 1e-5

    def residual(space: FiniteElementDiscretization) -> float:
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        form = phx.equations.FiniteElementForm(
            "source-cylinder-independent-affine-oracle",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
        )
        problem = phx.equations.compile_finite_element_problem(
            form, space, constraint=constraint, dirichlet_values=lambda x: x[..., 2]
        )
        exact = space.dof_maps[0].dof_coordinates[..., 2]
        return float(jnp.linalg.norm(problem.residual(exact[constraint.free_dofs])))

    assert residual(source_space) < 1e-10
    assert residual(carrier_space) > 1e-6
    _, stiffness = source_space.assemble_field_operators(
        "u", source_space.default_runtime
    )
    free = phx.discretization.dirichlet_constraint(source_space, "u").free_dofs
    assert (
        float(
            jnp.linalg.norm(
                stiffness.mv(source_space.dof_maps[0].dof_coordinates[..., 2])[free]
            )
        )
        < 1e-10
    )
    from phydrax.discretization.fem._local_provider import SourceSurfaceGeometryActions
    from phydrax.equations.fem._executor import _restricted_workset_rows

    form = phx.equations.FiniteElementForm(
        "source-cylinder-bounded-workset", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    compiled = phx.equations.compile_finite_element_problem(form, source_space)
    workset = next(
        workset
        for workset in compiled._workset_program.worksets
        if workset.local_region is not None
        and isinstance(
            workset.local_region.geometry_actions, SourceSurfaceGeometryActions
        )
    )
    assert workset.local_region is not None
    geometry = workset.local_region.geometry_actions
    selected = jnp.asarray((0, 2, 5), dtype=jnp.int32)

    def restricted_points(rows: Array) -> Array:
        restricted = _restricted_workset_rows(workset, rows)
        assert restricted is not None
        assert restricted.local_region is not None
        return restricted.local_region.geometry_actions.realize(
            source_space.default_runtime
        ).points

    expected_points = geometry.realize(source_space.default_runtime).points[selected]
    np.testing.assert_allclose(
        jax.jit(restricted_points)(selected), expected_points, rtol=0.0, atol=1e-14
    )
    batches = jnp.asarray(((0, 2, 5), (1, 3, 6)), dtype=jnp.int32)
    scanned = jax.jit(
        lambda groups: jax.lax.scan(
            lambda carry, rows: (carry, restricted_points(rows)), None, groups
        )[1]
    )(batches)
    np.testing.assert_allclose(
        scanned,
        geometry.realize(source_space.default_runtime).points[batches],
        rtol=0.0,
        atol=1e-14,
    )
    state = jnp.linspace(0.0, 1.0, source_space.dof_maps[0].global_dof_count)
    primal, tangent = jax.jvp(compiled.residual, (state,), (jnp.ones_like(state),))
    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(tangent))
    with pytest.raises(ValueError, match="[Ff]ixed source geometry"):
        jax.jvp(
            lambda coordinates: (
                source_space.evaluate_geometry("u", coordinates)[0].physical_weights
            ),
            (result.geometry.coordinates,),
            (jnp.ones_like(result.geometry.coordinates),),
        )
    with pytest.raises(ValueError, match="[Ff]ixed source geometry"):
        jax.jvp(
            lambda coordinates: (
                source_space.evaluate_geometry("u", coordinates)[0].physical_weights
            ),
            (result.geometry.coordinates,),
            (jnp.zeros_like(result.geometry.coordinates),),
        )
    with pytest.raises(ValueError, match="[Ff]ixed source geometry"):
        jax.jvp(
            lambda parameters: eqx.tree_at(
                lambda value: value.parameters, source, (parameters,)
            ).evaluate_points(0, jnp.asarray(((0.2, 0.2),))),
            (source.parameters[0],),
            (jnp.ones_like(source.parameters[0]),),
        )

    other = profiles._surface_definition("different-source", "cylinder_sheet").domain
    wrong_query = MeshingDomainBoundarySource(other, (0,))
    with pytest.raises(ValueError, match="source scope"):
        phx.discretization.SurfaceFiniteElementGeometry.from_result(result, wrong_query)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="carrier coordinates change"
    ):
        source_space.evaluate_geometry(
            "u", result.geometry.coordinates.at[0, 0].add(0.01)
        )
    changed = eqx.tree_at(
        lambda value: value.parameters,
        source,
        (source.parameters[0].at[0, 0, 0].add(0.01),),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="Source surface geometry changed after proof",
    ):
        changed.evaluate_points(0, jnp.asarray(((0.2, 0.2),)))
    changed_carrier = eqx.tree_at(
        lambda value: value.carrier_coordinates,
        source,
        source.carrier_coordinates.at[0, 0].add(0.01),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="changed after proof"
    ):
        changed_carrier.evaluate_points(0, jnp.asarray(((0.2, 0.2),)))
    host_coordinates = np.asarray(source.carrier_coordinates)
    row, column = np.argwhere((host_coordinates == 0.0) & ~np.signbit(host_coordinates))[
        0
    ]
    signed_zero = source.carrier_coordinates.at[int(row), int(column)].set(-0.0)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="carrier coordinates change"
    ):
        source.require_carrier(signed_zero)


def test_closed_accuracy_rejects_wrong_cap_source_despite_small_residual() -> None:
    from tools.meshing_qualification.scenarios._closed_surface_accuracy import (
        closed_surface_accuracy,
    )

    profile = "native-surface-pde:capped_cylinder"
    domain = profiles._surface_definition(profile, "capped_cylinder").domain
    scope = profiles.MeshingScope.whole(domain, 2)
    specification = profiles.SurfaceMeshingSpec(
        profiles.CellMeshingTarget(
            2, 3, profiles.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            profiles.UniformSizeControl(
                scope, 0.3, strength=profiles.SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=0.02,
    )
    result = (
        profiles.NativeMeshingProvider(
            profiles.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            domain,
            specification,
            coordinate_contract=profiles.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert result.certification is not None
    assert result.certification.inputs.source is not None
    source = phx.discretization.SurfaceFiniteElementGeometry.from_result(
        result, result.certification.inputs.source
    )
    field = profiles.FiniteElementFieldSpec("u", profiles.lagrange_element("triangle", 1))
    space = profiles.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry, source_surface=source
    ).prepare()

    def wrong_force(points: Array, context: object) -> Array:
        del context
        z = points[..., 2]
        return z**2 * (1.5 - z) ** 2 - (2 * 1.5**2 - 18 * z + 12 * z**2)

    form = phx.equations.FiniteElementForm(
        "wrong-cap-source",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(wrong_force, coefficient_id="wrong-cap-source"),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(
        operator,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=1e-14)
        ),
    )
    residual = float(jnp.linalg.norm(problem.residual(solved.value))) / max(
        1.0, float(jnp.linalg.norm(rhs))
    )
    assert bool(jnp.all(solved.successful))
    assert residual < 2e-8
    z = np.asarray(space.dof_maps[0].dof_coordinates)[:, 2]
    expected = z**2 * (1.5 - z) ** 2
    evidence = closed_surface_accuracy(
        space,
        source.material_map,
        "capped_cylinder",
        np.asarray(problem.expand(solved.value)),
        expected,
    )
    assert not evidence["energy_accuracy_passed"]
