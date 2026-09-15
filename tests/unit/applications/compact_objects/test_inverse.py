#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from phydrax.applications.compact_objects._black_hole_thermodynamics import (
    evaluate_stationary_kerr_horizon,
    KerrInput,
)
from phydrax.applications.compact_objects._inverse import (
    FixedBranchModelEvaluation,
    kerr_geometry_inverse_adapter,
    kerr_thermodynamics_model_evaluation,
    kerr_thermodynamics_inverse_adapter,
    qnm_root_model_evaluation,
    simple_qnm_root_inverse_adapter,
)


def _evaluation(values, signature=0, **flags):
    evidence = {
        "finite": True,
        "converged": True,
        "physically_valid": True,
        "qualified": True,
        "derivative_valid": True,
        "shock_free": True,
        "event_free": True,
        "topology_fixed": True,
    }
    evidence.update(flags)
    return FixedBranchModelEvaluation(
        values,
        jnp.asarray([signature]),
        **evidence,
        realization_id="kerr:analytic",
        branch_id="subextremal",
    )


def test_kerr_geometry_jvp_vjp_and_finite_difference_agree():
    def evaluate(parameters):
        mass, spin = parameters
        radius = mass + jnp.sqrt(mass**2 - spin**2)
        area = 4.0 * jnp.pi * (radius**2 + spin**2)
        return _evaluation(jnp.stack((radius, area)))

    adapter = kerr_geometry_inverse_adapter(
        evaluate,
        jnp.asarray([0]),
        parameter_count=2,
        output_count=2,
        realization_id="kerr:analytic",
        branch_id="subextremal",
        adapter_id="kerr-geometry-audit",
        evaluator_semantic_id="kerr-geometry-observable:v1",
        evaluator_numeric_id="kerr-geometry-implementation:r1",
    )
    evidence = adapter.sensitivity(
        jnp.asarray([2.0, 0.4]),
        jnp.asarray([0.25, -0.1]),
        epsilon=2.0e-4,
    )

    assert bool(evidence.derivative_valid)
    assert float(evidence.jvp_finite_difference_residual) < 2.0e-3
    assert float(evidence.vjp_pairing_residual) < 1.0e-5
    assert jnp.all(jnp.isfinite(evidence.jvp))
    assert jnp.all(jnp.isfinite(evidence.vjp))


def test_root_branch_change_and_event_do_not_expose_gradients():
    def changing_root(parameters):
        signature = jnp.where(parameters[0] < 0.0, 1, 0)
        return _evaluation(jnp.asarray([parameters[0] ** 2]), signature)

    root = simple_qnm_root_inverse_adapter(
        changing_root,
        jnp.asarray([0]),
        parameter_count=1,
        output_count=1,
        realization_id="kerr:analytic",
        branch_id="subextremal",
        adapter_id="qnm-root-audit",
        evaluator_semantic_id="qnm-root-observable:v1",
        evaluator_numeric_id="qnm-root-implementation:r1",
    )
    changed = root.sensitivity(
        jnp.asarray([0.0]), jnp.asarray([1.0]), epsilon=1.0e-3
    )
    assert not bool(changed.branch_stable)
    assert not bool(changed.derivative_valid)
    assert jnp.all(jnp.isnan(changed.jvp))
    assert jnp.all(jnp.isnan(changed.vjp))
    switched = root.evaluate(jnp.asarray([-1.0]))
    assert bool(switched.finite)
    assert bool(switched.physically_valid)
    assert not bool(switched.qualified)
    assert not bool(switched.derivative_valid)

    event = kerr_thermodynamics_inverse_adapter(
        lambda parameters: _evaluation(parameters**2, event_free=False),
        jnp.asarray([0]),
        parameter_count=1,
        output_count=1,
        realization_id="kerr:analytic",
        branch_id="subextremal",
        adapter_id="kerr-thermodynamics-audit",
        evaluator_semantic_id="kerr-thermodynamics-observable:v1",
        evaluator_numeric_id="kerr-thermodynamics-implementation:r1",
    ).sensitivity(jnp.asarray([1.0]), jnp.asarray([0.5]))
    assert not bool(event.center_eligible)
    assert not bool(event.derivative_valid)
    assert jnp.all(jnp.isnan(event.finite_difference))


def test_qnm_result_realification_retains_root_identity_and_status():
    result = SimpleNamespace(
        angular_frequency=jnp.asarray(0.4 - 0.08j),
        separation_constant=jnp.asarray(2.1 + 0.03j),
        status=jnp.asarray(0, dtype=jnp.int32),
        branch_id="l2-m2-n0",
        finite=jnp.asarray(True),
        converged=jnp.asarray(True),
        physically_valid=jnp.asarray(True),
        qualified=jnp.asarray(True),
        derivative_valid=jnp.asarray(True),
    )
    evaluation = qnm_root_model_evaluation(result, realization_id="qnm:release-1")

    assert jnp.allclose(evaluation.values, jnp.asarray([0.4, -0.08, 2.1, 0.03]))
    assert evaluation.branch_id == "l2-m2-n0"
    assert evaluation.realization_id == "qnm:release-1"
    assert bool(evaluation.sensitivity_eligible)


def test_stationary_kerr_result_adapter_retains_subextremal_branch():
    horizon = evaluate_stationary_kerr_horizon(KerrInput(2.0, 0.8))
    evaluation = kerr_thermodynamics_model_evaluation(
        horizon,
        jnp.stack((horizon.area, horizon.surface_gravity, horizon.angular_velocity)),
        realization_id=horizon.parameters.input_id,
        branch_id="subextremal",
    )

    assert bool(evaluation.sensitivity_eligible)
    assert int(evaluation.branch_signature[0]) == int(horizon.branch.code)
    assert evaluation.realization_id == horizon.parameters.input_id


def test_opaque_evaluator_requires_revision_and_body_change_changes_identity():
    def first(parameters):
        return _evaluation(parameters)

    def changed_body(parameters):
        return _evaluation(parameters + 1.0)

    kwargs = {
        "parameter_count": 1,
        "output_count": 1,
        "realization_id": "kerr:analytic",
        "branch_id": "subextremal",
        "adapter_id": "body-sensitive-adapter",
        "evaluator_semantic_id": "body-sensitive-observable:v1",
    }
    first_adapter = kerr_geometry_inverse_adapter(
        first,
        jnp.asarray([0]),
        **kwargs,
        evaluator_numeric_id="body-implementation:r1",
    )
    changed_adapter = kerr_geometry_inverse_adapter(
        changed_body,
        jnp.asarray([0]),
        **kwargs,
        evaluator_numeric_id="body-implementation:r2",
    )

    assert first_adapter.evaluator_semantic_id == changed_adapter.evaluator_semantic_id
    assert first_adapter.evaluator_numeric_id != changed_adapter.evaluator_numeric_id
    assert first_adapter.adapter_id != changed_adapter.adapter_id
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        kerr_geometry_inverse_adapter(
            first,
            jnp.asarray([0]),
            parameter_count=1,
            output_count=1,
            realization_id="kerr:analytic",
            branch_id="subextremal",
            adapter_id="missing-revision",
        )
