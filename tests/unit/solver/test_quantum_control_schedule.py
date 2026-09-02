#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


s = phx.solver
q = phx.operators.quantum


def _piecewise_constant():
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.5, 1.0]), time_id="control-grid")
    return phx.control.PiecewiseConstantControlParameterization(
        grid,
        (),
        parameterization_id="scalar-pwc",
    )


def test_quantum_control_sampling_applies_iq_carrier_delay_and_support():
    parameterization = _piecewise_constant()
    line = s.QuantumControlLine(
        parameterization,
        jnp.asarray([1.0, 2.0]),
        quadrature_coefficients=jnp.asarray([0.5, 0.25]),
        carrier=s.QuantumCarrier(angular_rate=jnp.pi, phase=0.2, delay=0.5),
        support_start=0.0,
        support_stop=1.0,
    )
    schedule = s.QuantumControlSchedule(
        (line,),
        s.LinearQuantumControlTransfer(jnp.asarray([[1.0, -0.2]])),
    )
    time_grid = jnp.asarray([0.0, 0.5, 1.0, 1.5])
    result = s.sample_quantum_control_schedule(schedule, time_grid)
    shifted = result.sample_times - 0.5
    query = jnp.clip(shifted, 0.0, 1.0)
    in_phase = parameterization.sample(jnp.asarray([1.0, 2.0]), query)
    quadrature = parameterization.sample(jnp.asarray([0.5, 0.25]), query)
    expected_line = jnp.where(
        (shifted >= 0.0) & (shifted <= 1.0),
        in_phase * jnp.cos(jnp.pi * shifted + 0.2)
        + quadrature * jnp.sin(jnp.pi * shifted + 0.2),
        0.0,
    )

    assert bool(result.diagnostics.valid)
    assert jnp.allclose(result.line_values[:, 0], expected_line)
    assert jnp.allclose(result.term_coefficients[:, 0], expected_line)
    assert jnp.allclose(result.term_coefficients[:, 1], -0.2 * expected_line)
    assert result.line_values[0, 0] == 0.0


def test_control_schedule_gradients_reach_coefficients_phase_delay_and_transfer():
    parameterization = _piecewise_constant()
    time_grid = jnp.asarray([0.5, 0.75, 1.0])

    def objective(in_phase, phase, delay, transfer):
        line = s.QuantumControlLine(
            parameterization,
            in_phase,
            carrier=s.QuantumCarrier(angular_rate=1.3, phase=phase, delay=delay),
            support_start=0.0,
            support_stop=1.0,
        )
        schedule = s.QuantumControlSchedule(
            (line,),
            s.LinearQuantumControlTransfer(transfer),
        )
        return jnp.sum(
            s.sample_quantum_control_schedule(schedule, time_grid).term_coefficients
        )

    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(
        jnp.asarray([0.7, 1.1]),
        jnp.asarray(0.2),
        jnp.asarray(0.1),
        jnp.asarray([[0.8]]),
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_sampled_controls_assemble_with_constant_drift():
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    layout = q.HilbertRegisterLayout(("q",), (2,))
    drift = s.LocalHamiltonian(
        layout,
        (s.LocalHamiltonianTerm(z, ("q",), term_id="drift"),),
    )
    parameterization = _piecewise_constant()
    controls = s.sample_quantum_control_schedule(
        s.QuantumControlSchedule(
            (
                s.QuantumControlLine(
                    parameterization,
                    jnp.asarray([0.2, 0.4]),
                    support_start=0.0,
                    support_stop=1.0,
                ),
            ),
            s.LinearQuantumControlTransfer(jnp.asarray([[1.0]])),
        ),
        jnp.asarray([0.0, 0.5, 1.0]),
    )
    assembled = s.assemble_fixed_grid_local_hamiltonian(
        drift,
        (s.LocalHamiltonianTerm(x, ("q",), term_id="drive"),),
        controls,
    )

    assert bool(assembled.valid)
    assert len(assembled.hamiltonian.terms) == 2
    assert jnp.allclose(assembled.coefficients[:, 0], 1.0)
    assert jnp.allclose(assembled.coefficients[:, 1], jnp.asarray([0.2, 0.4]))


def test_quantum_control_schedule_rejects_incompatible_shapes():
    parameterization = _piecewise_constant()
    with pytest.raises(ValueError, match="parameter_shape"):
        s.QuantumControlLine(
            parameterization,
            jnp.asarray([1.0]),
            support_start=0.0,
            support_stop=1.0,
        )
    line = s.QuantumControlLine(
        parameterization,
        jnp.asarray([1.0, 1.0]),
        support_start=1.0,
        support_stop=0.0,
    )
    assert not bool(line.valid)
    with pytest.raises(ValueError, match="line count"):
        s.QuantumControlSchedule(
            (line,),
            s.LinearQuantumControlTransfer(jnp.eye(2)),
        )
