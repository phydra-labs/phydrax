#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""An independent eigenpair remains physical through guide, archive and analysis."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import StrictModule
from phydrax.operators.quantum import LogAmplitude
from phydrax.operators.quantum.lattice import (
    QuantumAddress,
    QuantumConfigurationDomain,
    QuantumGuide,
)
from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    observe_projector_state,
    prepare_projector_monte_carlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    read_projector_monte_carlo_checkpoint,
    solve_projector_monte_carlo,
    write_projector_monte_carlo_checkpoint,
)
from phydrax.typing import as_array, Float64, Scalar
from phydrax.units import derived_unit, HARTREE
from phydrax.uq import CorrelatedRatioPolicy
from tests._support.projector_monte_carlo import (
    control_keys,
    control_operator,
    two_boson_two_site_control,
)


pytestmark = pytest.mark.strict_jax


class OccupationGuide(StrictModule):
    __strict_contract__ = True

    domain: QuantumConfigurationDomain
    slope: Float64[Scalar]

    def __init__(self, domain: QuantumConfigurationDomain, /) -> None:
        self.domain = domain
        self.slope = as_array(0.2, Float64[Scalar], "slope")

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        coordinate = self.domain.decode(address.key_words)
        return LogAmplitude(
            self.slope * coordinate[0].astype(jnp.float64),
            jnp.asarray(1.0, dtype=jnp.complex128),
        )


def test_guided_eigenpair_public_workflow_preserves_physical_energy_and_history(
    tmp_path: Path,
) -> None:
    control = two_boson_two_site_control()
    operator = control_operator(control)
    guide = QuantumGuide(
        operator.domain,
        OccupationGuide(operator.domain),
        provider_id="finite-boson-occupation-magnitude",
        mapping_id="declared-left-local-occupation",
        globally_positive=True,
    )
    keys = control_keys(operator, control)
    problem = ProjectorMonteCarloProblem(
        operator,
        keys,
        control.ground_vector,
        trial_keys=keys[1:2],
        trial_coefficients=np.asarray((1.0,), dtype=np.complex128),
        dt=0.02,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
        guide=guide,
    )
    plan = ProjectorMonteCarloPlan(
        replicas=2,
        support_capacity=3,
        group_capacity=8,
        event_capacity=2,
        attempt_capacity=16,
        source_capacity=1,
        history_capacity=24,
        maximum_retained_bytes=20_000_000,
        maximum_workspace_bytes=20_000_000,
        spawn_policy="exact",
        compression="none",
        controller="fixed-shift",
        initial_shift=control.ground_energy,
    )
    prepared = prepare_projector_monte_carlo(problem, plan)
    initial = initialize_projector_monte_carlo(prepared, jax.random.key(23))
    observation = observe_projector_state(prepared, initial)
    assert observation.finite
    np.testing.assert_allclose(
        observation.projected_numerator / observation.projected_denominator,
        control.ground_energy,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        observation.pair_numerators[:, 0] / observation.pair_denominators,
        control.ground_energy,
        rtol=2e-14,
        atol=2e-14,
    )
    first = solve_projector_monte_carlo(prepared, initial, steps=8)
    assert int(first.status) == 0
    checkpoint = write_projector_monte_carlo_checkpoint(
        tmp_path / "guided.phx", prepared, first.state
    )
    restored = read_projector_monte_carlo_checkpoint(checkpoint, prepared, initial)
    result = solve_projector_monte_carlo(prepared, restored, steps=8)
    assert int(result.status) == 0
    np.testing.assert_allclose(
        result.state.coefficients, initial.coefficients, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_array_equal(
        result.state.history.valid[:16], np.ones((16,), dtype=np.bool_)
    )
    analysis = analyze_projector_monte_carlo(
        prepared,
        result,
        policy=ProjectorEstimatorPolicy(
            ratio_policy=CorrelatedRatioPolicy(minimum_draws=2, minimum_blocks=2),
            deterministic_records=True,
            history_depths=(0, 1),
        ),
    )
    assert int(analysis.propagation_status) == 0
    assert analysis.projected.statistically_valid
    np.testing.assert_allclose(
        analysis.projected.value, control.ground_energy, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        analysis.replicas[0].value, control.ground_energy, rtol=2e-14, atol=2e-14
    )
    assert analysis.systematic.guide_id == guide.guide_id
    assert analysis.systematic.metric_id == guide.metric_id
