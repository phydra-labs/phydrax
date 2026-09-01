#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_coupled_field_checkpoint_roundtrip_and_identity_rejection(tmp_path):
    plan = phx.solver.CoupledFieldCheckpointPlan(
        "runtime",
        "program",
        ("concentrations", "surface_charge"),
        geometry_id="geometry",
        topology_id="topology",
    )
    state = {
        "concentrations": jnp.asarray(((1.0, 2.0), (3.0, 4.0))),
        "surface_charge": jnp.asarray((0.1, -0.1)),
    }
    args = {"temperature": jnp.asarray(300.0)}
    path = tmp_path / "coupled.phxcheckpoint"
    phx.solver.write_coupled_field_checkpoint(
        path,
        plan,
        jnp.asarray(0.25),
        jnp.asarray(5, dtype=jnp.int32),
        state,
        runtime_args=args,
    )
    restored = phx.solver.read_coupled_field_checkpoint(
        path,
        plan,
        state,
        runtime_args_template=args,
    )

    np.testing.assert_array_equal(
        restored.state["concentrations"], state["concentrations"]
    )
    np.testing.assert_array_equal(
        restored.runtime_args["temperature"], args["temperature"]
    )
    assert restored.step_index == 5

    incompatible = phx.solver.CoupledFieldCheckpointPlan(
        "other-runtime",
        "program",
        ("concentrations", "surface_charge"),
        geometry_id="geometry",
        topology_id="topology",
    )
    with pytest.raises(ValueError, match="plan_id"):
        phx.solver.read_coupled_field_checkpoint(path, incompatible, state)
