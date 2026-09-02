#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
from phydrax.equations.fem._ir import (
    lower_operator_program,
    OperatorNode,
    OperatorProgram,
    OperatorValue,
)
from phydrax.solver._production_resources import PreparedCompilationService


def test_operator_program_lowering_fuses_and_linearizes_static_kernel_chain():
    value = OperatorValue("state", "state", value_shape=(2,), layout_id="state-layout")
    node = OperatorNode("kernel", ("state",), ("output",), "double", ad_policy="analytic")
    program = OperatorProgram((value,), (node,), ("output",), bucket_id="one-bucket")
    lowered = lower_operator_program(program, {"double": lambda state: 2.0 * state})
    inputs = {"state": jnp.asarray((1.0, 3.0))}
    np.testing.assert_allclose(lowered(inputs)[0], (2.0, 6.0))
    output, pushforward, pullback = lowered.linearize(inputs)
    np.testing.assert_allclose(output, (2.0, 6.0))
    np.testing.assert_allclose(pushforward(jnp.ones((2,))), (2.0, 2.0))
    np.testing.assert_allclose(pullback(jnp.ones((2,)))[0], (2.0, 2.0))
    assert lowered.fusion.groups == ((0,),)


def test_compilation_service_and_precision_policy_are_operational():
    value = OperatorValue("state", "state", value_shape=(2,), layout_id="state-layout")
    node = OperatorNode("kernel", ("state",), ("output",), "shift")
    lowered = lower_operator_program(
        OperatorProgram((value,), (node,), ("output",), bucket_id="compile"),
        {"shift": lambda state: state + 1.0},
    )
    service = PreparedCompilationService()
    inputs = {"state": jnp.asarray((1.0, 2.0))}
    executable = service.compile(lowered, inputs)
    np.testing.assert_allclose(executable(inputs)[0], (2.0, 3.0))
    assert service.entry_count == 1
    assert service.compile(lowered, inputs) is executable

    precision = FiniteElementPrecisionPolicy(
        storage_dtype="float32",
        geometry_dtype="float64",
        evaluation_dtype="float64",
        accumulation_dtype="float64",
        output_dtype="float32",
    )
    assert precision.storage(inputs["state"]).dtype == jnp.float32
    assert precision.evaluation(inputs["state"]).dtype == jnp.float64
    assert precision.output(inputs["state"]).dtype == jnp.float32
    np.testing.assert_allclose(precision.sum(jnp.ones((100,))), 100.0)
    assert precision.tolerance() > 0.0
    precision.validate_backend()
