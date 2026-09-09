#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._identity import NumericRevision, SemanticProvenance
from phydrax.applications.skeletal_muscle.continuum import (
    HeidlaufRoehrle2014ActiveStressField,
    HeidlaufRoehrle2014Parameters,
    HeidlaufRoehrle2014Plan,
    HeidlaufRoehrle2014StressInput,
    UniformFiberArchitecturePlan,
)
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def _prepare():
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    parameters = HeidlaufRoehrle2014Parameters.published_table_2()
    material = HeidlaufRoehrle2014Plan("affine-tet", "affine-prescribed-gamma").prepare(
        parameters,
        UniformFiberArchitecturePlan("affine-tet-x").prepare(
            jnp.asarray((1.0, 0.0, 0.0))
        ),
        HeidlaufRoehrle2014StressInput(
            0.0, jnp.zeros(8, dtype=jnp.uint32), "affine-prescribed-gamma"
        ),
    )
    provenance = SemanticProvenance(
        {
            "kind": "prescribed-affine-gamma",
            "formula": "args[0]+args[1]*X[0]",
            "mesh": mesh.mesh_id,
            "frame": "reference",
            "unit": "1",
        }
    )

    def field(points, context):
        intercept, slope = context.user_args
        return intercept + slope * points[..., 0]

    stress_field = HeidlaufRoehrle2014ActiveStressField(
        field,
        provenance,
        NumericRevision(provenance, {}),
        source_id="affine-prescribed-gamma",
    )
    origin = 2 * parameters.c10_pa + 4 * parameters.c01_pa
    prepared = material.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero")),
        args=jnp.zeros(2),
        active_stress_field=stress_field,
        pressure_origin_pa=origin,
    )
    return material, prepared


def test_native_mixed_rest_and_spatial_active_virtual_work_are_exact():
    material, prepared = _prepare()
    state = prepared.problem.state_space.zeros()
    rest = prepared.problem.residual(state, jnp.zeros(2))
    np.testing.assert_allclose(rest[0], 0.0, atol=0.003)
    np.testing.assert_allclose(rest[1], 0.0, atol=2e-7)
    coordinates = prepared.discretization.dof_maps[0].dof_coordinates
    virtual_displacement = jnp.zeros_like(state[0]).at[:, 0].set(coordinates[:, 0])
    intercept, slope = 0.4, 0.6
    residual = prepared.problem.residual(state, jnp.asarray((intercept, slope)))
    virtual_work = jnp.sum(residual[0] * virtual_displacement)
    # Unit reference tetrahedron: volume=1/6 and integral X[0] dV=1/24.
    expected = material.parameters.maximum_active_nominal_stress_pa * (
        intercept / 6 + slope / 24
    )
    np.testing.assert_allclose(virtual_work, expected, rtol=3e-5, atol=0.1)
    np.testing.assert_allclose(jnp.sum(residual[0], axis=0), 0.0, atol=0.03)
    np.testing.assert_allclose(residual[1], 0.0, atol=2e-7)


def test_compiled_native_active_source_jvp_and_mixed_constraint_derivative():
    material, prepared = _prepare()
    state = prepared.problem.state_space.zeros()
    coordinates = prepared.discretization.dof_maps[0].dof_coordinates
    direction = jnp.zeros_like(state[0]).at[:, 0].set(coordinates[:, 0])

    def work(active_coefficients):
        residual = prepared.problem.residual(state, active_coefficients)
        return jnp.sum(residual[0] * direction)

    derivative = jax.jit(jax.grad(work))(jnp.asarray((0.4, 0.6)))
    np.testing.assert_allclose(
        derivative, 73000 * np.asarray((1 / 6, 1 / 24)), rtol=4e-5, atol=0.1
    )
    constraint_jvp = jax.jvp(
        lambda displacement: prepared.problem.residual(
            (displacement, state[1]), jnp.zeros(2)
        )[1],
        (state[0],),
        (direction,),
    )[1]
    # Partition of unity of pressure tests: d int -log(J)/d epsilon = -volume.
    np.testing.assert_allclose(jnp.sum(constraint_jvp), -1 / 6, rtol=3e-5, atol=1e-6)


def test_preparation_rejects_a_field_from_another_source():
    material, _ = _prepare()
    provenance = SemanticProvenance({"kind": "foreign-prescribed-field"})
    field = HeidlaufRoehrle2014ActiveStressField(
        lambda points, context: jnp.zeros(points.shape[:-1]),
        provenance,
        NumericRevision(provenance, {}),
        source_id="another-source",
    )
    with pytest.raises(ValueError, match="foreign active-stress source"):
        material.form(active_stress_field=field)
