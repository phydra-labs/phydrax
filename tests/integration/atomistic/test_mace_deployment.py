import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.backends.iree import iree_availability
from tests._support.mace_deployment import (
    calculator_plan,
    electronvolt_units,
    graph_execution,
    provider_plan,
    tiny_mace,
)


REPOSITORY = Path(__file__).resolve().parents[3]
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
NUMBERS = np.array([8, 1, 1])
MASSES = np.array([15.999, 1.008, 1.008])


def _structure(*, periodic: bool) -> Any:
    return phx.atomistic.AtomicStructure(
        jnp.asarray(NUMBERS),
        jnp.asarray(WATER),
        jnp.asarray(MASSES),
        electronvolt_units().scale,
        cell=jnp.asarray(CELL) if periodic else None,
        periodic_axes=jnp.asarray([True, True, True]) if periodic else None,
    )


def _system(*, periodic: bool) -> Any:
    return phx.atomistic.AtomisticSystemPlan.from_structure(
        _structure(periodic=periodic), electronvolt_units()
    ).prepare()


def _loaded_artifact(tmp_path: Path) -> tuple[Any, Any, Path]:
    model = tiny_mace(seed=3)
    path = tmp_path / "water-mace"
    manifest = phx.atomistic.write_atomistic_model_artifact(path, model)
    artifact = phx.atomistic.read_atomistic_model_artifact(
        path, numeric_revision_id=manifest.numeric_revision.revision_id
    )
    return model, artifact, path


def _reference(model: Any) -> Any:
    """Independent batch-topology prediction route for the same scalar energy."""

    return phx.atomistic.energy_and_forces(
        model, _structure(periodic=True), graph_execution(), compute_stress=True
    )


def test_loaded_artifact_through_ase_and_ipi_matches_native_reference(
    tmp_path: Path,
) -> None:
    model, artifact, _ = _loaded_artifact(tmp_path)
    reference = _reference(model)
    assert bool(reference.valid[0])
    np.testing.assert_allclose(
        _reference(artifact.model).energy, reference.energy, rtol=0.0, atol=0.0
    )

    system = _system(periodic=True)
    provider = provider_plan(artifact.model).prepare(system)
    plan = phx.atomistic.interchange.IPITransportPlan.unix(
        str(Path("/tmp") / f"phydrax-mace-{os.getpid()}.sock"), timeout=30.0
    )
    listener = plan.listen()

    def serve() -> Any:
        with listener.accept() as session:
            return phx.atomistic.interchange.serve_ipi_once(session, provider, system)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        with plan.connect() as session:
            remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
                session, "remote-mace"
            )
            transported = remote.evaluate(system, jnp.asarray(WATER), None)
        assert future.result(timeout=60.0) is (
            phx.atomistic.interchange.IPITransportStatus.READY
        )
    listener.close()
    assert bool(transported.successful)
    assert transported.stress is not None
    assert reference.stress is not None
    np.testing.assert_allclose(transported.energy, reference.energy[0], rtol=1e-10)
    np.testing.assert_allclose(
        transported.forces, reference.forces[0], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        transported.stress, reference.stress[0], rtol=1e-8, atol=1e-10
    )

    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(numbers=NUMBERS, positions=WATER, cell=CELL, pbc=True)
    atoms.calc = phx.atomistic.interchange.NativeASECalculator(
        calculator_plan(artifact.model),
        artifact_id=artifact.manifest.artifact_id,
    )
    np.testing.assert_allclose(
        atoms.get_potential_energy(), reference.energy[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        atoms.get_forces(), reference.forces[0], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False), reference.stress[0], rtol=1e-9, atol=1e-12
    )
    assert atoms.calc.provenance["artifact"] == artifact.manifest.artifact_id


def test_native_ipi_driver_refuses_required_virial_for_finite_system() -> None:
    system = _system(periodic=False)
    provider = provider_plan(tiny_mace()).prepare(system)
    plan = phx.atomistic.interchange.IPITransportPlan.unix(
        str(Path("/tmp") / f"phydrax-mace-finite-{os.getpid()}.sock"), timeout=30.0
    )
    listener = plan.listen()

    def serve() -> Any:
        with listener.accept() as session:
            return phx.atomistic.interchange.serve_ipi_once(session, provider, system)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        with plan.connect() as session:
            remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
                session, "remote-mace"
            )
            with pytest.raises(ConnectionError):
                remote.evaluate(system, jnp.asarray(WATER), None)
        with pytest.raises(ValueError, match="virial is required"):
            future.result(timeout=60.0)
    listener.close()


_FRESH_PROCESS = textwrap.dedent(
    """
    import json
    import sys

    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from tests._support.mace_deployment import provider_plan

    artifact_path, revision, export_path, module_sha, contract_id = sys.argv[1:6]
    artifact = phx.atomistic.read_atomistic_model_artifact(
        artifact_path, numeric_revision_id=revision
    )
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray([8, 1, 1]),
        jnp.asarray(json.loads(sys.argv[6])),
        jnp.asarray([15.999, 1.008, 1.008]),
        units.scale,
        cell=jnp.asarray(json.loads(sys.argv[7])),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, units).prepare()
    provider = provider_plan(artifact.model).prepare(system)
    request = provider.prepare_request(structure.positions, None, None)
    native = provider.evaluate_request(request)
    result = {
        "native_energy": float(native.evaluation.energy),
        "native_stress": np.asarray(native.evaluation.stress).tolist(),
    }
    if export_path != "-":
        frozen = phx.export.load_atomistic_iree(
            export_path,
            trusted_module_sha256=module_sha,
            trusted_contract_id=contract_id,
        )(request)
        result["frozen_energy"] = float(frozen.energy)
        result["frozen_forces"] = np.asarray(frozen.forces).tolist()
        result["frozen_status"] = int(frozen.status)
    print(json.dumps(result))
    """
)


def test_artifact_and_frozen_export_resume_in_a_fresh_process(tmp_path: Path) -> None:
    _, artifact, artifact_path = _loaded_artifact(tmp_path)
    system = _system(periodic=True)
    provider = provider_plan(artifact.model).prepare(system)
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    native = provider.evaluate_request(request)
    arguments = [
        str(artifact_path),
        artifact.manifest.numeric_revision.revision_id,
    ]
    has_iree = iree_availability().available
    if has_iree:
        bundle = phx.export.save_atomistic_iree(
            provider_plan(artifact.model),
            _structure(periodic=True),
            electronvolt_units(),
            tmp_path / "water-mace.phxiree",
            policy=phx.export.IREEExportPolicy(executable_format="system-library"),
        )
        arguments += [str(bundle.path), bundle.module_sha256, bundle.contract.contract_id]
    else:
        arguments += ["-", "-", "-"]
    arguments += [json.dumps(WATER.tolist()), json.dumps(CELL.tolist())]

    fresh = _run_script(_FRESH_PROCESS, arguments)
    assert fresh["native_energy"] == float(native.evaluation.energy)
    np.testing.assert_array_equal(
        fresh["native_stress"], np.asarray(native.evaluation.stress)
    )
    if has_iree:
        assert fresh["frozen_status"] == int(phx.atomistic.AtomisticStatus.SUCCESS)
        np.testing.assert_allclose(
            fresh["frozen_energy"], native.evaluation.energy, rtol=1e-10
        )
        np.testing.assert_allclose(
            fresh["frozen_forces"], native.evaluation.forces, rtol=1e-8, atol=1e-10
        )


def _run_script(script: str, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *arguments],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


_MD_CONTINUATION = """
import json
import sys

import jax
import jax.numpy as jnp

import phydrax as phx
from tests._support.mace_deployment import electronvolt_units, periodic_dynamics

path, structure_json, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.loads(structure_json)
units = electronvolt_units()
structure = phx.atomistic.AtomicStructure(
    jnp.asarray(payload["numbers"]),
    jnp.asarray(payload["positions"]),
    jnp.asarray(payload["masses"]),
    units.scale,
    cell=jnp.asarray(payload["cell"]),
    periodic_axes=jnp.asarray([True, True, True]),
)
system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, units).prepare()
model = phx.atomistic.read_atomistic_restart_model(path).model
dynamics, thermodynamic = periodic_dynamics(model, system)
template = dynamics.initialize_state(
    structure.positions,
    thermodynamic,
    velocity=jnp.zeros_like(structure.positions),
    key=jax.random.key(0),
)
state = phx.atomistic.read_atomistic_restart(
    path, phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic), template
).state
for _ in range(steps):
    state = dynamics.step(state, thermodynamic)
print(json.dumps({
    "positions": state.kinematics.positions.tolist(),
    "momenta": state.kinematics.momenta.tolist(),
    "step_index": int(state.step_index),
}))
"""


def test_periodic_md_restart_continues_identically_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    from tests._support.mace_deployment import periodic_dynamics

    model = tiny_mace(seed=3)
    system = _system(periodic=True)
    dynamics, thermodynamic = periodic_dynamics(model, system)
    velocity = 0.002 * np.cos(np.arange(9.0)).reshape(3, 3)
    state = dynamics.initialize_state(
        jnp.asarray(WATER),
        thermodynamic,
        velocity=jnp.asarray(velocity),
        key=jax.random.key(0),
    )
    for _ in range(4):
        state = dynamics.step(state, thermodynamic)
    path = tmp_path / "md-restart"
    phx.atomistic.write_atomistic_restart(
        path,
        phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic),
        state,
        model=model,
    )
    reference = state
    for _ in range(4):
        reference = dynamics.step(reference, thermodynamic)
    assert bool(jnp.all(jnp.isfinite(reference.kinematics.positions)))

    fresh = _run_script(
        _MD_CONTINUATION,
        [
            str(path),
            json.dumps(
                {
                    "numbers": NUMBERS.tolist(),
                    "positions": WATER.tolist(),
                    "masses": MASSES.tolist(),
                    "cell": CELL.tolist(),
                }
            ),
            "4",
        ],
    )
    assert fresh["step_index"] == int(reference.step_index)
    np.testing.assert_array_equal(
        fresh["positions"], np.asarray(reference.kinematics.positions)
    )
    np.testing.assert_array_equal(
        fresh["momenta"], np.asarray(reference.kinematics.momenta)
    )


_TRAINING_CONTINUATION = """
import json
import sys

import jax
import numpy as np

import phydrax as phx
from tests._support.mace_deployment import strained_water_problem

directory, saved_steps, total_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
problem = strained_water_problem()
saved = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=saved_steps, learning_rate=1.0e-2)
total = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=total_steps, learning_rate=1.0e-2)
artifact, restored = phx.atomistic.read_atomistic_training_restart(directory, problem, saved)
continued = phx.atomistic.fit_atomistic_potential(
    artifact.model, problem, total, key=jax.random.key(5), continuation=restored
)
print(json.dumps({
    "loss": np.asarray(continued.training_loss_history).tolist(),
    "revision": phx.atomistic.atomistic_potential_revision(continued.potential).revision_id,
}))
"""


def test_force_stress_training_continues_identically_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    from tests._support.mace_deployment import strained_water_problem

    problem = strained_water_problem()
    saved = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1.0e-2)
    total = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=4, learning_rate=1.0e-2)
    partial = phx.atomistic.fit_atomistic_potential(
        tiny_mace(seed=4), problem, saved, key=jax.random.key(5)
    )
    assert bool(partial.successful)
    directory = phx.atomistic.write_atomistic_training_restart(
        tmp_path / "training-restart", partial, saved
    )
    uninterrupted = phx.atomistic.fit_atomistic_potential(
        tiny_mace(seed=4), problem, total, key=jax.random.key(5)
    )
    assert float(np.asarray(uninterrupted.stress_loss_history)[0]) > 0.0

    fresh = _run_script(_TRAINING_CONTINUATION, [str(directory), "2", "4"])
    np.testing.assert_array_equal(
        fresh["loss"], np.asarray(uninterrupted.training_loss_history)
    )
    assert fresh["revision"] == (
        phx.atomistic.atomistic_potential_revision(uninterrupted.potential).revision_id
    )
