#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_allen_cahn_accepted_step_decreases_free_energy():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(), phx.discretization.FiniteElementFieldSpec("eta", element)
    ).prepare()
    parameters = phx.applications.phase_field.AllenCahnParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    )
    result = phx.applications.phase_field.solve_allen_cahn_step(
        discretization,
        "eta",
        jnp.full((5,), 0.2),
        0.01,
        parameters,
    )

    assert bool(result.successful)
    assert result.accepted_energy < result.energy_before
    assert jnp.array_equal(result.candidate_state, result.accepted_state)


def test_cahn_hilliard_step_preserves_mass():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    parameters = phx.applications.phase_field.CahnHilliardParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    )
    result = phx.applications.phase_field.solve_cahn_hilliard_step(
        discretization,
        "c",
        "mu",
        jnp.full((5,), 0.2),
        jnp.full((5,), -0.192),
        0.01,
        parameters,
    )

    assert bool(result.successful)
    assert jnp.abs(result.accepted_mass - result.mass_before) < 1.0e-12
    assert bool(result.mass_accepted)
    assert jnp.array_equal(result.candidate_state[0], result.accepted_state[0])


class _ForcedCandidate:
    def __init__(self, candidate):
        self.candidate = candidate

    def solve(self, problem, initial, *, termination):
        del problem, initial, termination
        return SimpleNamespace(
            state=self.candidate,
            successful=jnp.asarray(True),
        )


def test_phase_field_rejection_rolls_back_every_state_leaf_and_guards_resources():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    previous = jnp.full((5,), 0.2)
    chemical = jnp.full((5,), -0.192)
    candidate = (previous + 0.4, chemical + 0.7)
    parameters = phx.applications.phase_field.CahnHilliardParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    )
    acceptance = phx.applications.phase_field.PhaseFieldAcceptancePolicy(
        relative_energy_tolerance=0.0,
        absolute_mass_tolerance=0.0,
    )

    rejected = phx.applications.phase_field.solve_cahn_hilliard_step(
        discretization,
        "c",
        "mu",
        previous,
        chemical,
        0.01,
        parameters,
        method=_ForcedCandidate(candidate),
        acceptance=acceptance,
    )

    assert not bool(rejected.successful)
    assert not bool(rejected.mass_accepted)
    assert rejected.rejection_reasons & int(
        phx.applications.phase_field.PhaseFieldRejectionReason.MASS_DRIFT
    )
    np.testing.assert_array_equal(rejected.accepted_state[0], previous)
    np.testing.assert_array_equal(rejected.accepted_state[1], chemical)
    assert rejected.accepted_mass == rejected.mass_before
    assert rejected.accepted_energy == rejected.energy_before

    with pytest.raises(ValueError, match="maximum_degrees_of_freedom"):
        phx.applications.phase_field.solve_cahn_hilliard_step(
            discretization,
            "c",
            "mu",
            previous,
            chemical,
            0.01,
            parameters,
            acceptance=phx.applications.phase_field.PhaseFieldAcceptancePolicy(
                maximum_degrees_of_freedom=9
            ),
        )
