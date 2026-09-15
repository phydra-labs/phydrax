#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import importlib.util

import jax.numpy as jnp
import pytest

import phydrax.export._lattice_boltzmann_iree as lbm_iree
from phydrax.backends._types import BackendAvailability, BackendUnavailableError
from phydrax.backends.iree import IREE_CAPABILITIES
from phydrax.discretization.lattice_boltzmann._execution import (
    LatticeBoltzmannExecutionStep,
    ReferenceLatticeBoltzmannExecutionPlan,
)
from phydrax.discretization.lattice_boltzmann._lattice import D2Q9
from phydrax.export._iree import load_iree


def _step(step_index, time, populations, step_size, args):
    del step_index, time, step_size, args
    return LatticeBoltzmannExecutionStep(
        populations,
        populations,
        jnp.asarray(True),
        jnp.zeros((), dtype=populations.dtype),
        jnp.asarray(populations.size, dtype=jnp.int32),
        {"mass": jnp.sum(populations)},
    )


def _plan():
    return ReferenceLatticeBoltzmannExecutionPlan(
        D2Q9(),
        _step,
        step_id="iree-forward-test",
    )


def test_lbm_iree_contract_has_explicit_forward_and_vjp_abis():
    plan = _plan()
    initial = jnp.ones((3, 4, plan.velocity_set.population_count), dtype=jnp.float64)

    contract = lbm_iree.prepare_lattice_boltzmann_iree_contract(
        plan,
        initial,
        step_count=5,
        step_size=0.25,
        t0=1.0,
        mode="forward-vjp",
    )

    assert contract.execution_mode == "forward-vjp"
    assert contract.supports_reverse_mode
    assert contract.execution_plan_id == plan.plan_id
    assert contract.lattice_id == plan.velocity_set.lattice_id
    assert contract.input_shapes == (initial.shape,)
    assert contract.step_count == 5
    assert contract.input_names == ("populations",)
    assert contract.output_names == ("final_populations",)
    assert contract.differentiable_input_names == ("populations",)
    assert contract.pack_inputs(initial)[0] is initial
    packed_vjp = contract.pack_vjp_inputs((initial,), jnp.ones_like(initial))
    assert len(packed_vjp) == 2
    assert contract.contract_id
    with pytest.raises(ValueError, match="Unknown LBM IREE export mode"):
        lbm_iree.prepare_lattice_boltzmann_iree_contract(
            plan,
            initial,
            step_count=5,
            step_size=0.25,
            mode="reverse",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="trailing-Q"):
        lbm_iree.prepare_lattice_boltzmann_iree_contract(
            plan,
            initial[..., :-1],
            step_count=5,
            step_size=0.25,
        )


_HAS_IREE = importlib.util.find_spec("iree") is not None
if _HAS_IREE:
    _HAS_IREE = (
        importlib.util.find_spec("iree.compiler") is not None
        and importlib.util.find_spec("iree.runtime") is not None
    )


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_lbm_iree_forward_and_ordered_vjp_execute_with_named_outputs(tmp_path):
    def scaled_step(step_index, time, populations, step_size, args):
        del step_index, time, step_size
        _, runtime_values = args
        (coefficient,) = runtime_values
        candidate = coefficient * populations
        return LatticeBoltzmannExecutionStep(
            candidate,
            candidate,
            jnp.asarray(True),
            jnp.zeros((), dtype=populations.dtype),
            jnp.asarray(populations.size, dtype=jnp.int32),
            {"mass": jnp.sum(candidate)},
        )

    plan = ReferenceLatticeBoltzmannExecutionPlan(
        D2Q9(),
        scaled_step,
        step_id="iree-forward-vjp-test",
    )
    initial = jnp.ones((2, 2, plan.velocity_set.population_count), dtype=jnp.float32)
    coefficient = jnp.asarray(0.5, dtype=jnp.float32)
    bundle = lbm_iree.save_lattice_boltzmann_iree(
        plan,
        tmp_path / "forward-vjp.phxiree",
        initial_populations=initial,
        step_count=2,
        step_size=1.0,
        runtime_arrays=(coefficient,),
        runtime_array_names=("coefficient",),
        differentiable_inputs=("populations", "coefficient"),
        mode="forward-vjp",
    )

    forward = load_iree(bundle.forward.path)
    forward_output = forward(initial, coefficient)
    assert not isinstance(forward_output, tuple)
    assert bundle.forward.manifest.output_names == ("final_populations",)
    assert bundle.vjp is not None
    assert bundle.vjp.manifest.output_names == (
        "cotangent_populations",
        "cotangent_coefficient",
    )
    vjp = load_iree(bundle.vjp.path)
    cotangents = vjp(
        *bundle.contract.pack_vjp_inputs((initial, coefficient), jnp.ones_like(initial))
    )
    assert isinstance(cotangents, tuple)
    assert tuple(value.shape for value in cotangents) == (initial.shape, ())
    assert tuple(value.dtype for value in cotangents) == (
        initial.dtype,
        coefficient.dtype,
    )


def test_lbm_iree_export_fails_closed_at_existing_availability_gate(
    monkeypatch, tmp_path
):
    unavailable = BackendAvailability(
        capabilities=IREE_CAPABILITIES,
        available=False,
        requirement="install phydrax[iree]",
        reason="test compiler/runtime unavailable",
    )
    monkeypatch.setattr(
        lbm_iree,
        "lattice_boltzmann_iree_availability",
        lambda: unavailable,
    )
    plan = _plan()
    initial = jnp.ones((2, 2, plan.velocity_set.population_count), dtype=jnp.float64)

    with pytest.raises(BackendUnavailableError, match="compiled-inference"):
        lbm_iree.save_lattice_boltzmann_iree(
            plan,
            tmp_path / "forward.phxiree",
            initial_populations=initial,
            step_count=2,
            step_size=1.0,
        )
    assert not (tmp_path / "forward.phxiree").exists()
