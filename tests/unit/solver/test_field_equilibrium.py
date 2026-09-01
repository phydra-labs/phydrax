#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _selected_root():
    root = {
        "u": jnp.asarray((0.4, -0.2)),
        "fixed": jnp.asarray(3.0),
    }
    subspace = phx.nn.parameters.ParameterSubspace(
        root,
        {"u": True, "fixed": False},
    )
    return root, subspace


def test_functional_stationarity_is_the_gradient_of_one_fixed_realization():
    root, subspace = _selected_root()
    target = jnp.asarray((0.1, 0.3))

    def action(functions, realization, args):
        del args
        displacement = functions["u"] - realization
        return 0.5 * jnp.vdot(displacement, displacement)

    prepared = phx.solver.prepare_functional_stationarity(
        root,
        action,
        subspace,
        realization=target,
        realization_id="fixed-target",
        provenance_id="functional-realization-0",
    )
    state = prepared.initial_state

    np.testing.assert_allclose(
        prepared.problem.residual(state),
        root["u"] - target,
        rtol=0.0,
        atol=0.0,
    )
    assert prepared.problem.state_space.size == subspace.total_dimension
    assert prepared.problem.residual_space.compatible(prepared.problem.state_space)
    assert prepared.root["fixed"] == root["fixed"]
    np.testing.assert_allclose(prepared.reconstruct(state)["u"], root["u"])


def test_virtual_work_assembles_the_jet_vjp_and_retains_its_tangent():
    root, subspace = _selected_root()
    matrix = jnp.asarray(((2.0, -1.0), (0.5, 3.0)))

    def field_jet(functions, realization, args):
        del realization, args
        return matrix @ functions["u"]

    def virtual_work(functions, jets, realization, args):
        del functions, realization, args
        return jets * jets

    prepared = phx.solver.prepare_virtual_work_equilibrium(
        root,
        field_jet,
        virtual_work,
        subspace,
        None,
        realization_id="virtual-work-points",
        provenance_id="virtual-work-realization-0",
    )
    state = prepared.initial_state
    jets = matrix @ root["u"]
    expected_residual = matrix.T @ (jets * jets)
    expected_tangent = matrix.T @ (2.0 * jets[:, None] * matrix)

    np.testing.assert_allclose(
        prepared.problem.residual(state),
        expected_residual,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        jax.jacfwd(prepared.problem.residual)(state),
        expected_tangent,
        rtol=2e-14,
        atol=2e-14,
    )
