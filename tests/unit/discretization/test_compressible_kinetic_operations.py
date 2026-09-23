#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.discrete_velocity import (
    CompressibleKineticRuntimePlan,
    guided_d3q39_plan,
    IntegerLatticeTransportPlan,
)
from phydrax.discretization.lattice_boltzmann._checkpoint import (
    read_kinetic_checkpoint,
    write_kinetic_checkpoint,
)
from phydrax.interchange import (
    CompressibleKineticCaseIR,
    import_packed_kinetic_boundary,
)
from phydrax.rendering import KineticVolumeRenderPlan
from phydrax.solver import write_compressible_kinetic_vti


def test_runtime_checkpoint_continuation_preserves_dual_and_stabilizer(tmp_path):
    shape = (8, 8, 8)
    model = guided_d3q39_plan()
    runtime = CompressibleKineticRuntimePlan(
        model, IntegerLatticeTransportPlan(model.rule, shape)
    )
    state = runtime.initialize(jnp.ones(shape), jnp.zeros(shape + (3,)), jnp.ones(shape))
    first = runtime.advance(state, 1.0)
    checkpoint_path = tmp_path / "kinetic.npz"
    plan = runtime.checkpoint_plan()
    write_kinetic_checkpoint(
        checkpoint_path,
        plan,
        first.accepted.time,
        first.accepted.step_index,
        first.accepted,
    )
    restored = read_kinetic_checkpoint(
        checkpoint_path,
        plan,
        first.accepted,
    )
    continued = runtime.advance(restored.state, 1.0)

    assert bool(first.successful)
    assert bool(continued.successful)
    np.testing.assert_array_equal(
        restored.state.kinetic.equilibrium_dual,
        first.accepted.kinetic.equilibrium_dual,
    )
    np.testing.assert_array_equal(
        restored.state.kinetic.stabilizer,
        first.accepted.kinetic.stabilizer,
    )


def test_case_ir_boundary_import_render_and_vtk_output(tmp_path):
    case = CompressibleKineticCaseIR(
        grid_shape=(8, 8, 8),
        model="guided-d3q39",
        collision="quasi-equilibrium",
        gamma=1.4,
        gas_constant=1.0,
        prandtl_number=0.72,
        relaxation_rate=1.0,
        time_step=1.0,
        dtype="float32",
    )
    model, runtime = case.compile()
    assert runtime is not None
    assert model.rule.population_count == 39
    assert model.rule.velocities.dtype == jnp.dtype("float32")

    packed = np.zeros((4, 4), dtype=np.uint32)
    packed[0, 0] = np.uint32(1 << 29) | np.uint32(3 << 24)
    boundary = import_packed_kinetic_boundary(packed)
    assert bool(boundary.wall_mask[0, 0])
    assert int(boundary.normal_indices[0, 0]) == 3

    field = jnp.arange(64.0).reshape((4, 4, 4))
    rendered = KineticVolumeRenderPlan("maximum").render(field)
    assert bool(rendered.evidence.successful)
    assert rendered.rgba.shape == (4, 4, 4)

    vtk = write_compressible_kinetic_vti(
        tmp_path / "field.vti",
        {"density": np.asarray(field), "velocity": np.zeros((4, 4, 4, 3))},
    )
    assert vtk.path.is_file()
    assert vtk.byte_size > 0
    assert len(vtk.sha256) == 64
