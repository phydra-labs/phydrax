#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Checkpoint restart of a coupled FE/FV solve with model, RNG, history, and observers.

The coupled solve is the FE/FV conjugate-heat composition of
`examples/adaptive_fe_fv_rebind.py` with a carried-key conductance model: every
fluid step draws a lognormal conductance factor from the checkpointed key, the
fluid counts its applied steps in its model state, and a streaming observer
accumulates both probes after each accepted window. A checkpoint is the
array-only accepted boundary written through `RuntimeCheckpointEnvelope`; a
restart rebuilds every prepared owner from the declaration (no in-memory
object crosses the file) and must continue bitwise, the declared restart class.
A topology change needs the explicit relation of a published rebind.
"""

from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples import adaptive_fe_fv_rebind as ex
from phydrax._array_archive import read_array_archive


solver = phx.solver
_WINDOW = 0.05
_CHECKPOINT_WINDOWS = 4
_TOTAL_WINDOWS = 8
_PRECISION = "float64"


def _observer() -> solver.StreamingObservablePlan:
    return solver.StreamingObservablePlan(
        "coupled-probes",
        lambda time, state, model: jnp.stack(
            (
                model.solid_probe.apply(ex.participant(state, "solid-fe").native)[0],
                ex.participant(state, "fluid-fv").native[model.fluid_probe][0],
            )
        ),
        "mean",
    )


def _drive(
    model: ex.CoupledModel,
    state: Any,
    observed: solver.StreamingObservableState,
    windows: int,
) -> tuple[Any, solver.StreamingObservableState]:
    plan = _observer()
    for _ in range(windows):
        state, _ = ex.advance(model, state, 1)
        observed = plan.update(state.time, observed, state, model)
    return state, observed


class _Identities(NamedTuple):
    mesh_id: str
    method_id: str
    topology_epoch_id: str


def _identities(model: ex.CoupledModel, state: Any) -> _Identities:
    composition = ex.compose(model, state)
    return _Identities(
        composition.structure_id,
        # The declared coupled problem, stable across topology epochs.
        composition.entry("coupling/epoch").semantics_id,
        model.epoch.epoch_id,
    )


def _checkpoint(
    path: Path,
    model: ex.CoupledModel,
    state: Any,
    observed: solver.StreamingObservableState,
) -> Path:
    identities = _identities(model, state)
    envelope = solver.RuntimeCheckpointEnvelope(
        state,
        time=state.time,
        step_index=state.window_index,
        schedule_cursor=state.window_index,
        mesh_id=identities.mesh_id,
        method_id=identities.method_id,
        precision_id=_PRECISION,
        topology_epoch_id=identities.topology_epoch_id,
        observer_states=(observed,),
    )
    return solver.write_runtime_checkpoint(path, envelope)


@pytest.fixture(scope="module")
def uninterrupted() -> tuple[Any, solver.StreamingObservableState]:
    model, state = ex.build(_WINDOW, randomness="carried-key")
    observed = _observer().initial_state((2,))
    return _drive(model, state, observed, _TOTAL_WINDOWS)


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    model, state = ex.build(_WINDOW, randomness="carried-key")
    observed = _observer().initial_state((2,))
    state, observed = _drive(model, state, observed, _CHECKPOINT_WINDOWS)
    path = _checkpoint(
        tmp_path_factory.mktemp("coupled") / "boundary.phxckpt", model, state, observed
    )
    return path, state


def test_restart_resumes_bitwise_from_rebuilt_owners(
    uninterrupted: tuple[Any, solver.StreamingObservableState],
    written: tuple[Path, Any],
) -> None:
    path, _ = written
    # A new process: every prepared owner is rebuilt from the declaration.
    model, template = ex.build(_WINDOW, randomness="carried-key")
    observer = _observer().initial_state((2,))
    identities = _identities(model, template)
    envelope = solver.read_runtime_checkpoint(
        path,
        state_template=template,
        observer_templates=(observer,),
        mesh_id=identities.mesh_id,
        method_id=identities.method_id,
        precision_id=_PRECISION,
        topology_epoch_id=identities.topology_epoch_id,
    )
    fluid = ex.participant(envelope.state, "fluid-fv")
    assert int(fluid.model_state[1]) == _CHECKPOINT_WINDOWS * ex.FLUID_SUBSTEPS
    assert int(fluid.accepted_windows) == _CHECKPOINT_WINDOWS
    resumed, observed = _drive(
        model,
        envelope.state,
        envelope.observer_states[0],
        _TOTAL_WINDOWS - _CHECKPOINT_WINDOWS,
    )
    final, expected = uninterrupted
    assert eqx.tree_equal(resumed, final)
    assert eqx.tree_equal(observed, expected)
    # The carried key really advanced: a fresh key would diverge.
    assert not np.array_equal(
        np.asarray(fluid.key_data),
        np.asarray(jax.random.key_data(jax.random.key(7, impl=ex.KEY_IMPL))),
    )


@pytest.fixture(scope="module")
def rebound_live() -> tuple[ex.CoupledModel, Any, Any, Any]:
    """Live path: rebind the solid at the checkpointed boundary and continue."""
    model, state = ex.build(_WINDOW, randomness="carried-key")
    boundary, _ = ex.advance(model, state, _CHECKPOINT_WINDOWS)
    rebound, published, receipt = ex.rebind(model, boundary, "solid", accepted=True)
    live, _ = ex.advance(rebound, published, _TOTAL_WINDOWS - _CHECKPOINT_WINDOWS)
    return rebound, published, receipt, live


def test_restart_refuses_changed_identities_without_a_relation(
    written: tuple[Path, Any],
    rebound_live: tuple[ex.CoupledModel, Any, Any, Any],
) -> None:
    path, _ = written
    model, template = ex.build(_WINDOW, randomness="carried-key")
    observer = (_observer().initial_state((2,)),)
    source = _identities(model, template)
    with pytest.raises(ValueError, match="compatibility identities changed"):
        solver.read_runtime_checkpoint(
            path,
            state_template=template,
            observer_templates=observer,
            mesh_id=source.mesh_id,
            method_id=source.method_id,
            precision_id=_PRECISION,
            topology_epoch_id=source.topology_epoch_id,
            partition_id="two-rank",
        )
    rebound, published, _, _ = rebound_live
    target = _identities(rebound, published)
    with pytest.raises(ValueError, match="compatibility identities changed"):
        solver.read_runtime_checkpoint(
            path,
            state_template=template,
            observer_templates=observer,
            mesh_id=target.mesh_id,
            method_id=target.method_id,
            precision_id=_PRECISION,
            topology_epoch_id=target.topology_epoch_id,
        )


def test_checkpoints_hold_only_portable_arrays() -> None:
    model, state = ex.build(_WINDOW, randomness="carried-key")
    fluid = ex.participant(state, "fluid-fv")
    # A typed key is an in-memory token; its portable form is the key data.
    typed = jax.random.wrap_key_data(fluid.key_data, impl=ex.KEY_IMPL)
    identities = _identities(model, state)
    with pytest.raises(TypeError):
        solver.RuntimeCheckpointEnvelope(
            typed,
            time=state.time,
            step_index=state.window_index,
            schedule_cursor=state.window_index,
            mesh_id=identities.mesh_id,
            method_id=identities.method_id,
            precision_id=_PRECISION,
            topology_epoch_id=identities.topology_epoch_id,
        )


def _restorer(template: Any, model: ex.CoupledModel) -> Any:
    """Explicit relation: archived source arrays, then the rebind's own transports."""
    treedef = jax.tree_util.tree_structure(template)

    def restore(
        arrays: Any, specification: dict[str, Any], destination: Any, encoding: Any
    ) -> Any:
        del destination, encoding
        leaves = [jnp.asarray(arrays[name]) for name in specification["arrays"]]
        source = jax.tree_util.tree_unflatten(treedef, leaves)
        _, published, receipt = ex.rebind(model, source, "solid", accepted=True)
        if not receipt.published:
            raise ValueError("The restart relation's rebind was refused.")
        return published

    return restore


def test_restart_into_a_rebound_topology_uses_the_rebind_relation(
    written: tuple[Path, Any],
    rebound_live: tuple[ex.CoupledModel, Any, Any, Any],
) -> None:
    path, _ = written
    manifest, arrays = read_array_archive(path)
    rebound, published, receipt, live = rebound_live
    target = _identities(rebound, published)
    observer = (_observer().initial_state((2,)),)
    with pytest.raises(ValueError, match="does not bind the checkpoint topology"):
        solver.restore_runtime_checkpoint_arrays(
            manifest,
            arrays,
            state_template=published,
            observer_templates=observer,
            target_mesh_id=target.mesh_id,
            target_method_id=target.method_id,
            target_precision_id=_PRECISION,
            target_topology_epoch_id=target.topology_epoch_id,
            target_runtime_id="coupled-restart",
            restart_relation=solver.RuntimeRestartRelation.identity(target.mesh_id),
        )
    # Restart process: fresh owners, then the explicit relation of the rebind.
    fresh, template = ex.build(_WINDOW, randomness="carried-key")
    relation = solver.RuntimeRestartRelation.from_composition_rebind(
        receipt, _restorer(template, fresh), classification="bitwise"
    )
    envelope, source_id = solver.restore_runtime_checkpoint_arrays(
        manifest,
        arrays,
        state_template=published,
        observer_templates=observer,
        target_mesh_id=target.mesh_id,
        target_method_id=target.method_id,
        target_precision_id=_PRECISION,
        target_topology_epoch_id=target.topology_epoch_id,
        target_runtime_id="coupled-restart",
        restart_relation=relation,
    )
    assert source_id == manifest["checkpoint_id"]
    assert eqx.tree_equal(envelope.state, published)
    resumed, _ = ex.advance(rebound, envelope.state, _TOTAL_WINDOWS - _CHECKPOINT_WINDOWS)
    assert eqx.tree_equal(resumed, live)


def test_refused_rebind_admits_no_restart_relation() -> None:
    model, state = ex.build(_WINDOW)
    boundary, _ = ex.advance(model, state, _CHECKPOINT_WINDOWS)
    _, _, receipt = ex.rebind(model, boundary, "solid", accepted=False)
    with pytest.raises(ValueError, match="refused composition rebind"):
        solver.RuntimeRestartRelation.from_composition_rebind(
            receipt, lambda *arguments: arguments, classification="bitwise"
        )
