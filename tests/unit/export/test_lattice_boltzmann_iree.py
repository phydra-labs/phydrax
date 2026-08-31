#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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


def test_lbm_iree_contract_is_static_forward_only_and_trailing_q():
    plan = _plan()
    initial = jnp.ones((3, 4, plan.velocity_set.population_count), dtype=jnp.float64)

    contract = lbm_iree.prepare_lattice_boltzmann_iree_contract(
        plan,
        initial,
        step_count=5,
        step_size=0.25,
        t0=1.0,
    )

    assert contract.execution_mode == "forward-only"
    assert not contract.supports_reverse_mode
    assert contract.execution_plan_id == plan.plan_id
    assert contract.lattice_id == plan.velocity_set.lattice_id
    assert contract.input_shape == initial.shape
    assert contract.step_count == 5
    assert contract.input_names == ("populations",)
    assert contract.output_names == ("final_populations",)
    assert contract.pack_inputs(initial)[0] is initial
    assert contract.unpack_outputs(initial)[0] is initial
    assert contract.contract_id
    with pytest.raises(ValueError, match="explicitly forward-only"):
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
