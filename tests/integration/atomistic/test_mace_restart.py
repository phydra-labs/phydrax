"""Fresh-process continuation of imported MACE models: NVT dynamics and training."""

import copy
import errno
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax._publication as publication
import phydrax._training_checkpoint as training_checkpoint
from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax._model._structure import model_recipe_array_inventory
from phydrax._training import TrainingIterationKind
from phydrax.atomistic._checkpoint import (
    _RESTART_LIMITS,
    read_atomistic_checkpoint_resources,
    read_atomistic_restart,
    read_atomistic_restart_models,
    read_atomistic_training_restart,
    write_atomistic_restart,
    write_atomistic_training_restart,
)
from phydrax.atomistic._model_artifact import (
    atomistic_model_identity,
    AtomisticModelArtifactError,
    write_atomistic_model_artifact,
)
from phydrax.atomistic._training import (
    atomistic_irrep_linear_sites,
    AtomisticIrrepLowRankAdaptation,
)
from phydrax.execution import CallableIterationHostControl, IterationSession
from tests._support.compiler_profile import isolated_in_fixed_compiler_profile
from tests._support.mace_source import (
    admitted_source,
    committed_source_potential,
    nvt_dynamics,
    ONE_LAYER_INVARIANT_READOUT,
    provider_runtime,
    water_system,
    water_template,
    water_training_problem,
)


REPOSITORY = Path(__file__).resolve().parents[3]
SAVED_STEPS = 3
CONTINUED_STEPS = 4
HISTORIES = (
    "training_values",
    "training_objectives",
    "validation_values",
    "validation_objectives",
    "validation_steps",
)

# Every fresh process refuses to import a provider package: restoration and
# continuation consume only the declared native model and state data.
_PRELUDE = """
import importlib.abc
import sys


class RefuseProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "mace", "e3nn"}:
            raise ImportError(f"provider package {name} imported")
        return None


sys.meta_path.insert(0, RefuseProvider())
"""

_MD = """
import jax
import phydrax as phx
from phydrax.atomistic._checkpoint import read_atomistic_restart, read_atomistic_restart_models
from tests._support.mace_source import nvt_dynamics, water_system, water_template

restart, output, steps, expected_source = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
bindings = read_atomistic_restart_models(restart)
assert tuple(bindings) == ("learned",)
artifact = bindings["learned"].artifact
source = artifact.manifest.source
assert (source.conversion_id if source is not None else "-") == expected_source
dynamics, thermodynamic = nvt_dynamics(artifact.model, water_system())
plan = phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id="rank-0")
state = read_atomistic_restart(restart, plan, water_template(dynamics, thermodynamic)).state
for _ in range(steps):
    state = dynamics.step(state, thermodynamic)
phx.atomistic.write_atomistic_checkpoint(output, plan, state)
"""

_TRAINING = """
import jax
import phydrax as phx
from phydrax.atomistic._checkpoint import (
    read_atomistic_training_restart,
    write_atomistic_training_restart,
)
from tests._support.mace_source import water_training_problem

directory, output = sys.argv[1], sys.argv[2]
saved, total, payload_cases = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
problem = water_training_problem(payload_cases=payload_cases or None)
policy = lambda steps: phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=steps, learning_rate=1.0e-2, validation_every=1
)
artifact, restored = read_atomistic_training_restart(directory, problem, policy(saved))
# Iteration sessions are segment-local: the continuation binds no session.
continued = phx.atomistic.fit_atomistic_potential(
    artifact.model, problem, policy(total), key=jax.random.key(5), continuation=restored
)
write_atomistic_training_restart(output, continued, problem, policy(total), licenses=["MIT"])
"""


def _fresh(script: str, arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(script), *arguments],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _host_leaves(tree: Any) -> list[np.ndarray]:
    def host(leaf: Any) -> np.ndarray:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf)

    return [host(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _assert_identical(observed: Any, expected: Any) -> None:
    left, right = _host_leaves(observed), _host_leaves(expected)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def _assert_histories(observed: Any, expected: Any) -> None:
    for history in HISTORIES:
        np.testing.assert_array_equal(
            np.asarray(getattr(observed, history)), np.asarray(getattr(expected, history))
        )


@pytest.fixture(scope="module")
def imported() -> Any:
    return committed_source_potential("two_scale_shift")


def _nvt_restart(imported: Any, path: Path) -> dict[str, Any]:
    """Saved NVT state, its restart bundle at ``path`` and the uninterrupted reference."""

    system = water_system()
    dynamics, thermodynamic = nvt_dynamics(imported, system)
    velocity = 0.003 * np.cos(np.arange(9.0)).reshape(3, 3)
    state = dynamics.initialize_state(
        jnp.asarray(water_template(dynamics, thermodynamic).kinematics.positions),
        thermodynamic,
        velocity=jnp.asarray(velocity),
        key=jax.random.key(7),
    )
    for _ in range(SAVED_STEPS):
        state = dynamics.step(state, thermodynamic)
    plan = phx.atomistic.AtomisticCheckpointPlan(
        dynamics, thermodynamic, scope_id="rank-0"
    )
    write_atomistic_restart(path, plan, state, licenses={"learned": ["MIT"]})
    reference = state
    for _ in range(CONTINUED_STEPS):
        reference = dynamics.step(reference, thermodynamic)
    return {
        "system": system,
        "dynamics": dynamics,
        "thermodynamic": thermodynamic,
        "plan": plan,
        "saved": state,
        "reference": reference,
        "path": path,
    }


@pytest.fixture(scope="module")
def md(imported: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _nvt_restart(imported, tmp_path_factory.mktemp("md") / "nvt.restart")


def test_fresh_process_nvt_continuation_equals_uninterrupted_evolution(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    if isolated_in_fixed_compiler_profile(request):
        return
    md = _nvt_restart(
        committed_source_potential("two_scale_shift"), tmp_path / "nvt.restart"
    )
    output = tmp_path / "continued.chk"
    _fresh(_MD, [str(md["path"]), str(output), str(CONTINUED_STEPS), "-"])
    template = water_template(md["dynamics"], md["thermodynamic"])
    continued = phx.atomistic.read_atomistic_checkpoint(
        output, md["plan"], template
    ).state
    reference = md["reference"]
    assert int(continued.step_index) == SAVED_STEPS + CONTINUED_STEPS
    assert float(continued.time) == float(reference.time)
    assert int(continued.configuration_revision) == int(reference.configuration_revision)
    # Positions, momenta, forces, energies, thermostat key, typed controllers
    # and the neighborhood bundle lifecycle all equal the uninterrupted
    # accepted evolution exactly.
    _assert_identical(continued, reference)
    assert not np.array_equal(
        np.asarray(continued.kinematics.momenta),
        np.asarray(md["saved"].kinematics.momenta),
    )


def test_restart_refuses_altered_model_bundle_scope_and_thermodynamics(
    md: dict[str, Any], imported: Any
) -> None:
    path, system = md["path"], md["system"]
    bindings = read_atomistic_restart_models(path)
    assert tuple(bindings) == ("learned",)
    restored = bindings["learned"].artifact
    assert restored.manifest.source is None and restored.manifest.licenses == ("MIT",)
    assert (
        bindings["learned"].term.term_id == md["plan"].dynamics.potential.terms[0].term_id
    )
    resources = read_atomistic_checkpoint_resources(path)
    assert resources.bundle_id == md["dynamics"].bundle.prepared_id
    assert resources.plan.plan_id == md["dynamics"].bundle.plan.plan_id

    def resume(model: Any, *, scope: str | None = "rank-0", **options: Any) -> Any:
        dynamics, thermodynamic = nvt_dynamics(model, system, **options)
        plan = phx.atomistic.AtomisticCheckpointPlan(
            dynamics, thermodynamic, scope_id=scope
        )
        return read_atomistic_restart(path, plan, water_template(dynamics, thermodynamic))

    _assert_identical(resume(restored.model).state, md["saved"])
    altered = eqx.tree_at(
        lambda model: model.embedding, imported, imported.embedding * 1.001
    )
    with pytest.raises(AtomisticModelArtifactError, match="differs from the bundled"):
        resume(altered)
    with pytest.raises(ValueError, match="another neighborhood bundle plan"):
        resume(restored.model, skin=0.5)
    with pytest.raises(ValueError, match="does not match the runtime"):
        resume(restored.model, temperature=310.0)
    with pytest.raises(ValueError, match="canonical format"):
        resume(restored.model, scope=None)
    with pytest.raises(ValueError, match="does not match the runtime"):
        resume(restored.model, scope="rank-1")


def test_restart_bundle_refuses_shape_preserving_model_corruption(
    md: dict[str, Any], tmp_path: Path
) -> None:
    manifest, arrays = read_array_archive(md["path"], limits=_RESTART_LIMITS)
    manifest = copy.deepcopy(manifest)
    manifest.pop("arrays")
    arrays = {name: np.array(value) for name, value in arrays.items()}
    (section,) = manifest["models"]["artifacts"]
    inventory = model_recipe_array_inventory(
        section["model_recipe"],
        prefix="model/leaves",
        limits=_RESTART_LIMITS,
    )
    (name,) = [
        f"artifacts/0000/{entry.name}"
        for entry in inventory
        if entry.path.endswith(".embedding")
    ]
    arrays[name] = np.full_like(arrays[name], np.nan)
    corrupted = tmp_path / "corrupted.restart"
    write_array_archive(
        corrupted,
        manifest=manifest,
        limits=_RESTART_LIMITS,
        arrays=arrays,
    )
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_restart_models(corrupted)


def _policy(steps: int, **overrides: Any) -> Any:
    return phx.atomistic.AtomisticTrainingPolicy(
        maximum_steps=steps, learning_rate=1.0e-2, validation_every=1, **overrides
    )


def _training_restart(imported: Any, directory: Path) -> dict[str, Any]:
    """Partial fit published at ``directory`` and its uninterrupted counterpart."""

    problem = water_training_problem()
    partial = phx.atomistic.fit_atomistic_potential(
        imported, problem, _policy(2), key=jax.random.key(5)
    )
    assert bool(partial.successful)
    directory = write_atomistic_training_restart(
        directory, partial, problem, _policy(2), licenses=["MIT"]
    )
    uninterrupted = phx.atomistic.fit_atomistic_potential(
        imported, problem, _policy(5), key=jax.random.key(5)
    )
    return {
        "problem": problem,
        "partial": partial,
        "uninterrupted": uninterrupted,
        "directory": directory,
    }


@pytest.fixture(scope="module")
def training(imported: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _training_restart(imported, tmp_path_factory.mktemp("training") / "restart")


def _bundle_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_restores_partial(directory: Path, training: dict[str, Any]) -> None:
    artifact, restored = read_atomistic_training_restart(
        directory, training["problem"], _policy(2)
    )
    partial = training["partial"]
    assert restored.result_id == partial.result_id
    assert restored.progress == partial.progress
    assert restored.cursor == partial.cursor
    # Parameters, Adam moments, root key and cursors.
    _assert_identical(restored.training_state, partial.training_state)
    _assert_identical(restored.best_potential, partial.best_potential)
    _assert_histories(restored, partial)
    assert artifact.manifest.numeric_revision.revision_id == (
        phx.atomistic.atomistic_model_revision(partial.potential).revision_id
    )


def test_failed_training_restart_publication_keeps_resumable_bundle(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if isolated_in_fixed_compiler_profile(request):
        return
    training = _training_restart(
        committed_source_potential("two_scale_shift"),
        tmp_path_factory.mktemp("training") / "restart",
    )
    problem = training["problem"]
    bundle = tmp_path / "restart"
    shutil.copytree(training["directory"], bundle)
    published = _bundle_bytes(bundle)
    changed = training["uninterrupted"]

    # A changed valid model with a policy of other continuation semantics.
    with pytest.raises(ValueError, match="continuation semantics"):
        write_atomistic_training_restart(
            bundle, changed, problem, _policy(5, clip_global_norm=0.5), licenses=["MIT"]
        )
    assert _bundle_bytes(bundle) == published
    _assert_restores_partial(bundle, training)

    # The training component fails to publish after the model was staged.
    with monkeypatch.context() as patch:

        def disk_full(*_: Any, **__: Any) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        patch.setattr(training_checkpoint, "_publish_manifest", disk_full)
        with pytest.raises(OSError, match="No space"):
            write_atomistic_training_restart(
                bundle, changed, problem, _policy(5), licenses=["MIT"]
            )
    assert _bundle_bytes(bundle) == published
    _assert_restores_partial(bundle, training)

    # Both components staged; the process dies before the bundle commit.
    with monkeypatch.context() as patch:

        def interrupted(*_: Any, **__: Any) -> None:
            raise OSError(errno.EIO, "Interrupted before commit")

        patch.setattr(publication, "_atomic_exchange_directories", interrupted)
        with pytest.raises(OSError, match="before commit"):
            write_atomistic_training_restart(
                bundle, changed, problem, _policy(5), licenses=["MIT"]
            )
    assert _bundle_bytes(bundle) == published
    assert sorted(path.name for path in tmp_path.iterdir()) == ["restart"]
    _assert_restores_partial(bundle, training)

    # The surviving bundle resumes in a fresh process along the uninterrupted
    # accepted trajectory, and that continuation replaces it atomically.
    output = tmp_path / "continued"
    _fresh(_TRAINING, [str(bundle), str(output), "2", "5", "0"])
    artifact, continued = read_atomistic_training_restart(output, problem, _policy(5))
    expected = training["uninterrupted"]
    assert continued.progress == expected.progress
    assert continued.termination == expected.termination
    assert continued.result_id == expected.result_id
    _assert_histories(continued, expected)
    assert np.asarray(continued.training_values).size == 5
    # Parameters, Adam moments, root key and cursors.
    _assert_identical(continued.training_state, expected.training_state)
    assert artifact.manifest.numeric_revision.revision_id == (
        phx.atomistic.atomistic_model_revision(expected.potential).revision_id
    )
    write_atomistic_training_restart(
        bundle, continued, problem, _policy(5), licenses=["MIT"]
    )
    _, replaced = read_atomistic_training_restart(bundle, problem, _policy(5))
    assert replaced.result_id == continued.result_id
    _assert_identical(replaced.training_state, continued.training_state)


def test_mid_window_training_restart_resumes_in_a_fresh_process(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    if isolated_in_fixed_compiler_profile(request):
        return
    imported = committed_source_potential("two_scale_shift")
    # Two payloads per accumulation window; stop after the first payload of the
    # second window (the second MICROSTEP event).
    problem = water_training_problem(payload_cases=1)
    microsteps = []

    def stop(event: Any) -> bool:
        kind = TrainingIterationKind(int(event.record.metrics.kind))
        if kind is TrainingIterationKind.MICROSTEP:
            microsteps.append(kind)
        return len(microsteps) == 2

    session = IterationSession(
        "mid-window-training-restart",
        control=CallableIterationHostControl(stop, "stop-inside-second-window"),
    )
    stopped = phx.atomistic.fit_atomistic_potential(
        imported, problem, _policy(4), key=jax.random.key(5), session=session
    )
    assert int(stopped.status) == int(phx.atomistic.AtomisticStatus.STOPPED_EARLY)
    assert stopped.termination == "host_control_stop_inside_accumulation_window"
    assert stopped.cursor.payload_index > 0
    assert np.asarray(stopped.training_values).size == 1
    directory = write_atomistic_training_restart(
        tmp_path / "mid-window", stopped, problem, _policy(4), licenses=["MIT"]
    )
    _, restored = read_atomistic_training_restart(directory, problem, _policy(4))
    assert restored.result_id == stopped.result_id
    assert restored.cursor == stopped.cursor
    # The open accumulation window (pending gradient and model state) survives.
    _assert_identical(restored.training_state, stopped.training_state)

    uninterrupted = phx.atomistic.fit_atomistic_potential(
        imported,
        water_training_problem(payload_cases=1),
        _policy(4),
        key=jax.random.key(5),
    )
    output = tmp_path / "continued"
    _fresh(_TRAINING, [str(directory), str(output), "4", "4", "1"])
    _, continued = read_atomistic_training_restart(output, problem, _policy(4))
    assert continued.termination == uninterrupted.termination
    assert continued.progress == uninterrupted.progress
    assert continued.result_id == uninterrupted.result_id
    assert continued.cursor == uninterrupted.cursor
    _assert_histories(continued, uninterrupted)
    _assert_identical(continued.training_state, uninterrupted.training_state)


def test_adapted_training_restart_records_and_verifies_the_base_model(
    imported: Any, tmp_path: Path
) -> None:
    problem = water_training_problem()
    (site, *_) = atomistic_irrep_linear_sites(imported)

    def adapted(steps: int) -> Any:
        return _policy(
            steps, adaptation=AtomisticIrrepLowRankAdaptation((site,), rank=1, seed=3)
        )

    partial = phx.atomistic.fit_atomistic_potential(
        imported, problem, adapted(2), key=jax.random.key(5)
    )
    directory = write_atomistic_training_restart(
        tmp_path / "adapted", partial, problem, adapted(2)
    )
    assert (directory / "base.phydrax").is_file()
    artifact, restored = read_atomistic_training_restart(directory, problem, adapted(2))
    assert restored.result_id == partial.result_id
    # The artifact is the merged model; the restore template was the base.
    assert atomistic_model_identity(artifact.model) == atomistic_model_identity(
        partial.potential
    )
    assert atomistic_model_identity(
        restored.trained_tree.base_potential()
    ) == atomistic_model_identity(imported)
    _assert_identical(restored.training_state, partial.training_state)
    with pytest.raises(ValueError, match="adaptation differs"):
        read_atomistic_training_restart(directory, problem, _policy(2))
    # A base artifact from another model is a receipt mismatch.
    swapped = tmp_path / "swapped-base"
    shutil.copytree(directory, swapped)
    write_atomistic_model_artifact(swapped / "base.phydrax", partial.potential)
    with pytest.raises(ValueError, match="publication receipt"):
        read_atomistic_training_restart(swapped, problem, adapted(2))
    continued = phx.atomistic.fit_atomistic_potential(
        artifact.model, problem, adapted(3), key=jax.random.key(5), continuation=restored
    )
    uninterrupted = phx.atomistic.fit_atomistic_potential(
        imported, problem, adapted(3), key=jax.random.key(5)
    )
    assert continued.result_id == uninterrupted.result_id
    _assert_identical(continued.training_state, uninterrupted.training_state)


def test_training_restart_refuses_altered_targets_scales_and_model(
    training: dict[str, Any], tmp_path: Path
) -> None:
    directory = training["directory"]
    _, restored = read_atomistic_training_restart(
        directory, training["problem"], _policy(2)
    )
    assert restored.result_id == training["partial"].result_id
    with pytest.raises(ValueError, match="problem_id"):
        read_atomistic_training_restart(
            directory, water_training_problem(label_shift=0.01), _policy(2)
        )
    with pytest.raises(ValueError, match="continuation_id"):
        read_atomistic_training_restart(
            directory, training["problem"], _policy(2, clip_global_norm=0.5)
        )
    swapped = tmp_path / "swapped"
    shutil.copytree(directory, swapped)
    write_atomistic_model_artifact(
        swapped / "model.phydrax", training["uninterrupted"].potential, licenses=["MIT"]
    )
    with pytest.raises(ValueError, match="publication receipt"):
        read_atomistic_training_restart(swapped, training["problem"], _policy(2))
    # A receipt whose identities were edited consistently with its digests
    # (a mixed bundle claiming another cursor) refuses after restoration.
    mixed = tmp_path / "mixed"
    shutil.copytree(directory, mixed)
    receipt = (mixed / "restart.json").read_text()
    (mixed / "restart.json").write_text(
        receipt.replace(training["partial"].cursor.cursor_id, "0" * 64)
    )
    with pytest.raises(ValueError, match="receipt identities"):
        read_atomistic_training_restart(mixed, training["problem"], _policy(2))
    # A receipt-free directory layout refuses rather than loads.
    stale = tmp_path / "stale"
    shutil.copytree(directory, stale)
    (stale / "restart.json").unlink()
    with pytest.raises(ValueError, match="not a canonical atomistic training restart"):
        read_atomistic_training_restart(stale, training["problem"], _policy(2))


def test_converted_source_restart_preserves_provenance_in_a_fresh_process(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    provider = provider_runtime()
    if isolated_in_fixed_compiler_profile(request):
        return
    fixture = phx.atomistic.interchange.create_mace_provider_fixture(
        ONE_LAYER_INVARIANT_READOUT, tmp_path, provider=provider, seed=11
    )
    source = admitted_source(
        fixture.path, "torch-state-dict", architecture=fixture.declaration
    )
    conversion = phx.atomistic.interchange.convert_mace_checkpoint(
        source, provider=provider
    )
    system = water_system()
    dynamics, thermodynamic = nvt_dynamics(conversion.potential, system)
    state = dynamics.initialize_state(
        jnp.asarray(water_template(dynamics, thermodynamic).kinematics.positions),
        thermodynamic,
        velocity=jnp.asarray(0.002 * np.sin(np.arange(9.0)).reshape(3, 3)),
        key=jax.random.key(3),
    )
    state = dynamics.step(state, thermodynamic)
    plan = phx.atomistic.AtomisticCheckpointPlan(
        dynamics, thermodynamic, scope_id="rank-0"
    )
    # Source and license mappings are keyed by executed learned term name.
    with pytest.raises(ValueError, match="name no learned term"):
        write_atomistic_restart(
            tmp_path / "unbound.restart",
            plan,
            state,
            sources={"other": conversion.provenance},
        )
    # A provenance record bound to another model (cutoff 3.0 versus the
    # source's 3.2) refuses before anything is written.
    other_dynamics, other_thermodynamic = nvt_dynamics(
        committed_source_potential("two_scale_shift"), system
    )
    with pytest.raises(
        AtomisticModelArtifactError, match="differ from its recorded source"
    ):
        write_atomistic_restart(
            tmp_path / "unbound.restart",
            phx.atomistic.AtomisticCheckpointPlan(
                other_dynamics, other_thermodynamic, scope_id="rank-0"
            ),
            water_template(other_dynamics, other_thermodynamic),
            sources={"learned": conversion.provenance},
        )
    assert not (tmp_path / "unbound.restart").exists()
    path = tmp_path / "imported.restart"
    write_atomistic_restart(
        path,
        plan,
        state,
        sources={"learned": conversion.provenance},
        licenses={"learned": ["MIT"]},
    )
    reference = state
    for _ in range(2):
        reference = dynamics.step(reference, thermodynamic)
    output = tmp_path / "continued.chk"
    _fresh(_MD, [str(path), str(output), "2", conversion.provenance.conversion_id])
    continued = phx.atomistic.read_atomistic_checkpoint(
        output, plan, water_template(dynamics, thermodynamic)
    ).state
    _assert_identical(continued, reference)
