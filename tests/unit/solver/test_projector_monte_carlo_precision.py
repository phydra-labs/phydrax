# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Strict promotion preserves complex Euler phases and physical guide metrics."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._strict import StrictModule
from phydrax.operators.quantum._amplitude import LogAmplitude
from phydrax.operators.quantum.lattice._address import (
    QuantumAddress,
    QuantumConfigurationDomain,
)
from phydrax.operators.quantum.lattice._guide import QuantumGuide
from phydrax.solver._projector_monte_carlo import (
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    step_projector_monte_carlo,
)
from phydrax.solver._projector_monte_carlo_contracts import (
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloStatus,
)
from phydrax.solver._projector_monte_carlo_observables import observe_projector_state
from phydrax.typing import Float64, Scalar
from phydrax.units import derived_unit, HARTREE
from tests._support.projector_monte_carlo import (
    complex_flux_control,
    control_keys,
    control_operator,
)


class _CoordinateGuide(StrictModule):
    __strict_contract__ = True

    domain: QuantumConfigurationDomain
    offset: Float64[Scalar]
    slope: Float64[Scalar]

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        coordinate = self.domain.decode(address.key_words)
        return LogAmplitude(self.offset + self.slope * coordinate[0].astype(jnp.float64))


@pytest.mark.parametrize("guided", (False, True), ids=("physical", "positive-guide"))
def test_strict_complex_euler_and_physical_observation(guided: bool) -> None:
    control = complex_flux_control()
    operator = control_operator(control)
    keys = control_keys(operator, control)
    initial = np.asarray((0.4 + 0.2j, -0.3j, 0.7 - 0.1j), dtype=np.complex128)
    with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
        guide = None
        if guided:
            guide = QuantumGuide(
                operator.domain,
                _CoordinateGuide(
                    domain=operator.domain,
                    offset=jnp.log(jnp.float64(3)),
                    slope=jnp.log(jnp.float64(2)),
                ),
                provider_id="strict-precision-coordinate-guide",
                mapping_id="canonical-left-occupation",
                globally_positive=True,
            )
        problem = ProjectorMonteCarloProblem(
            operator,
            keys,
            initial,
            dt=0.05,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id=control.provenance,
            guide=guide,
        )
        prepared = prepare_projector_monte_carlo(
            problem,
            ProjectorMonteCarloPlan(
                replicas=2,
                support_capacity=3,
                group_capacity=8,
                event_capacity=2,
                attempt_capacity=8,
                source_capacity=3,
                history_capacity=2,
                maximum_retained_bytes=10_000_000,
                maximum_workspace_bytes=10_000_000,
                spawn_policy="exact",
                compression="none",
                controller="fixed-shift",
                initial_shift=0.2,
            ),
        )
        state = initialize_projector_monte_carlo(prepared, jax.random.key(42))
        result = step_projector_monte_carlo(prepared, state)
        observation = eqx.filter_jit(observe_projector_state)(prepared, result.state)
        physical = np.zeros(initial.shape, dtype=np.complex128)
        for key, coefficient, active in zip(
            np.asarray(result.state.support_keys[0]),
            np.asarray(result.state.coefficients[0]),
            np.asarray(result.state.active[0]),
            strict=True,
        ):
            if active:
                if guide is not None:
                    log_value, valid = guide.log_value(operator.domain.from_key(key))
                    assert bool(valid)
                    coefficient *= np.exp(-float(log_value))
                physical[np.all(np.asarray(keys) == key, axis=1)] = coefficient
    expected = initial + 0.05 * (0.2 * initial - control.matrix @ initial)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    np.testing.assert_allclose(physical, expected, rtol=2e-15, atol=2e-15)
    assert bool(observation.finite)
    np.testing.assert_allclose(
        observation.pair_denominators, np.vdot(expected, expected), atol=1e-12
    )
    np.testing.assert_allclose(
        observation.pair_numerators[:, 0],
        np.vdot(expected, control.matrix @ expected),
        atol=1e-12,
    )
