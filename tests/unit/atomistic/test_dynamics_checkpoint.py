#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamics checkpoints over typed controllers and grown neighborhood bundles."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPTPlan
from phydrax.atomistic._checkpoint import (
    _CHECKPOINT_LIMITS,
    AtomisticCheckpointPlan,
    read_atomistic_checkpoint,
    read_atomistic_checkpoint_resources,
    write_atomistic_checkpoint,
)
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticStepRejectionReason,
    retry_atomistic_step_with_capacity,
)
from phydrax.atomistic._neighborhood_bundle import AtomisticNeighborhoodBundlePlan
from phydrax.atomistic._thermal import NoseHooverChainParameters, NoseHooverChainPlan
from tests._support.atomistic_bundles import role_capacity
from tests._support.atomistic_dynamics_fixtures import (
    assert_tree_equal,
    atomistic_system,
    DirectedPairTerm,
    measure,
    observed_role_storage,
    periodic_cell,
    prepare_dynamics,
    prepare_potential,
    ROLE_CAPACITIES,
    row,
    step_detailed,
    table,
)


REPOSITORY = Path(__file__).resolve().parents[3]
VECTORS = 6.0 * np.eye(3)
# The pair starts 2.2 apart, outside the 1.5 + skin search radius, and closes
# to ~1.2 in one step: the rebuilt epoch stores two directed routes where the
# initial one stores none, so the exact first ladder entry (the one-route floor
# of a positive capacity) overflows: a pure capacity failure.
POSITIONS = np.asarray([[2.0, 3.0, 3.0], [4.2, 3.0, 3.0]])
VELOCITY = np.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
# Equipartition temperature of the declared velocities (unit masses, reduced
# k_B = 1, six unrestricted momentum degrees of freedom).  The chain starts at
# its target kinetic energy, so the first thermostat half step does not quench
# the approach the way a much colder target would (a chain mass
# Q_1 = N_f kT tau^2 at kT = 0.01 damps these velocities by ~97% before the
# drift, leaving the pair outside the search radius).
TEMPERATURE = float(np.sum(VELOCITY**2)) / VELOCITY.size
SCOPE = "dynamics-checkpoint-test"
CONTINUED = 3


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _term() -> DirectedPairTerm:
    return DirectedPairTerm("smooth-repulsion", 1.5, profile="smooth", strength=10.0)


def _rows(system: Any) -> tuple[Any, ...]:
    return (row(measure(system), "nvt", temperature=TEMPERATURE),)


def _integrator() -> NoseHooverChainPlan:
    return NoseHooverChainPlan(1.0, NoseHooverChainParameters(0.5, chain_length=2))


def _declared_capacities(exact: Any) -> tuple[Any, ...]:
    return (exact, role_capacity(4096, 512))


def exact_role_storage() -> Any:
    """First ladder entry holding exactly the initial geometry's routes."""
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(VECTORS))
    dynamics = prepare_dynamics(system, (_term(),), _integrator())
    thermodynamic = table(dynamics, *_rows(system))
    state = dynamics.initialize_state(
        jnp.asarray(POSITIONS),
        thermodynamic,
        velocity=jnp.asarray(VELOCITY),
        key=jr.key(9),
    )
    return observed_role_storage(state)


def fresh_runtime(path: str | Path, exact: Any, /) -> tuple[Any, Any, Any]:
    """Rebuild the runtime from declarations plus the checkpoint resources only.

    The Hamiltonian is prepared over the re-declared bundle plan at its first
    capacity level and rebound to the recorded (grown) levels; the template is
    ``initialize_state`` of the rebuilt runtime.
    """
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(VECTORS))
    potential = prepare_potential(
        system, (_term(),), role_capacities=_declared_capacities(exact)
    )
    resources = read_atomistic_checkpoint_resources(path)
    dynamics = AtomisticDynamicsPlan(
        system, resources.bind(potential), _integrator()
    ).prepare()
    thermodynamic = table(dynamics, *_rows(system))
    template = dynamics.initialize_state(
        jnp.asarray(POSITIONS), thermodynamic, velocity=jnp.zeros((2, 3)), key=jr.key(0)
    )
    return dynamics, thermodynamic, template


def _grown_run(exact: Any) -> dict[str, Any]:
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(VECTORS))
    states = _rows(system)
    dynamics = prepare_dynamics(
        system,
        (_term(),),
        _integrator(),
        role_capacities=_declared_capacities(exact),
    )
    thermodynamic = table(dynamics, *states)
    state = dynamics.initialize_state(
        jnp.asarray(POSITIONS),
        thermodynamic,
        velocity=jnp.asarray(VELOCITY),
        key=jr.key(9),
    )
    first = step_detailed(dynamics, state, thermodynamic)
    # The declared first level genuinely overflows on the attempted step; the
    # rejected attempt leaves the accepted state in place.
    assert not bool(first.successful)
    assert int(first.rejection_reasons) & int(AtomisticStepRejectionReason.ROUTE_CAPACITY)
    assert int(first.accepted_state.step_index) == int(state.step_index)
    grown, grown_table, replayed = retry_atomistic_step_with_capacity(
        dynamics, state, thermodynamic, states
    )
    assert bool(replayed.successful)
    assert grown.bundle.role_levels == (1,)
    assert int(replayed.accepted_state.step_index) == int(state.step_index) + 1
    saved = replayed.accepted_state
    saved = step_detailed(grown, saved, grown_table).accepted_state
    reference = saved
    for _ in range(CONTINUED):
        reference = step_detailed(grown, reference, grown_table).accepted_state
    return {
        "initial_dynamics": dynamics,
        "initial_table": thermodynamic,
        "dynamics": grown,
        "thermodynamic": grown_table,
        "saved": saved,
        "reference": reference,
    }


@pytest.fixture(scope="module")
def grown(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        exact = exact_role_storage()
        run = _grown_run(exact)
        path = tmp_path_factory.mktemp("checkpoint") / "grown.chk"
        plan = AtomisticCheckpointPlan(
            run["dynamics"], run["thermodynamic"], scope_id=SCOPE
        )
        write_atomistic_checkpoint(path, plan, run["saved"])
    return {**run, "exact": exact, "path": path, "plan": plan}


def _rewrite(source: Path, target: Path, edit: Any) -> Path:
    manifest, arrays = read_array_archive(source, limits=_CHECKPOINT_LIMITS)
    manifest = copy.deepcopy(manifest)
    manifest.pop("arrays")
    arrays = {name: np.array(value) for name, value in arrays.items()}
    edit(manifest, arrays)
    write_array_archive(
        target, manifest=manifest, limits=_CHECKPOINT_LIMITS, arrays=arrays
    )
    return target


def test_grown_bundle_checkpoint_rebuilds_capacities_and_continues_exactly(
    grown: dict[str, Any],
) -> None:
    path = grown["path"]
    resources = read_atomistic_checkpoint_resources(path)
    assert resources.role_levels == (1,)
    assert resources.bundle_id == grown["dynamics"].bundle.prepared_id
    assert resources.plan.plan_id == grown["dynamics"].bundle.plan.plan_id
    assert (
        AtomisticNeighborhoodBundlePlan.from_dict(resources.plan.to_dict()).plan_id
        == resources.plan.plan_id
    )
    dynamics, thermodynamic, template = fresh_runtime(path, grown["exact"])
    assert dynamics.prepared_id == grown["dynamics"].prepared_id
    plan = AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id=SCOPE)
    assert plan.checkpoint_id == grown["plan"].checkpoint_id
    restored = read_atomistic_checkpoint(path, plan, template).state
    assert_tree_equal(restored, grown["saved"])
    # Typed controllers, configuration revision and the bundle-bound force cache.
    assert restored.controllers.particle_chain is not None
    assert int(restored.configuration_revision) == int(
        grown["saved"].configuration_revision
    )
    assert restored.force.bundle_id == dynamics.bundle.prepared_id
    for _ in range(CONTINUED):
        restored = step_detailed(dynamics, restored, thermodynamic).accepted_state
    assert_tree_equal(restored, grown["reference"])


def test_grown_bundle_checkpoint_continues_in_a_fresh_process(
    grown: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "continued.chk"
    script = textwrap.dedent(
        """
        import importlib.util
        import sys

        from phydrax import config as pxconfig
        from phydrax.atomistic._checkpoint import (
            AtomisticCheckpointPlan,
            read_atomistic_checkpoint,
            write_atomistic_checkpoint,
        )

        module_path, restart, output, steps, scope = sys.argv[1:6]
        spec = importlib.util.spec_from_file_location("dynamics_checkpoint", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
            exact = module.exact_role_storage()
            dynamics, thermodynamic, template = module.fresh_runtime(restart, exact)
            plan = AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id=scope)
            state = read_atomistic_checkpoint(restart, plan, template).state
            for _ in range(int(steps)):
                state = module.step_detailed(dynamics, state, thermodynamic).accepted_state
            write_atomistic_checkpoint(output, plan, state)
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            str(grown["path"]),
            str(output),
            str(CONTINUED),
            SCOPE,
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    template = grown["dynamics"].initialize_state(
        jnp.asarray(POSITIONS),
        grown["thermodynamic"],
        velocity=jnp.zeros((2, 3)),
        key=jr.key(0),
    )
    continued = read_atomistic_checkpoint(output, grown["plan"], template).state
    assert int(continued.step_index) == int(grown["reference"].step_index)
    assert_tree_equal(continued, grown["reference"])


def test_checkpoint_refuses_other_bundle_bindings(grown: dict[str, Any]) -> None:
    path = grown["path"]
    # The same declarations at their first capacity level: another binding.
    dynamics, thermodynamic = grown["initial_dynamics"], grown["initial_table"]
    template = dynamics.initialize_state(
        jnp.asarray(POSITIONS), thermodynamic, velocity=jnp.zeros((2, 3)), key=jr.key(0)
    )
    with pytest.raises(ValueError, match="another neighborhood bundle binding"):
        read_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id=SCOPE),
            template,
        )
    # Another bundle plan (different declared ladder) refuses before any state.
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(VECTORS))
    other = prepare_potential(system, (_term(),), role_capacities=ROLE_CAPACITIES)
    resources = read_atomistic_checkpoint_resources(path)
    with pytest.raises(ValueError, match="another neighborhood bundle plan"):
        resources.bind(other)
    other_dynamics = AtomisticDynamicsPlan(system, other, _integrator()).prepare()
    other_table = table(other_dynamics, *_rows(system))
    other_template = other_dynamics.initialize_state(
        jnp.asarray(POSITIONS), other_table, velocity=jnp.zeros((2, 3)), key=jr.key(0)
    )
    with pytest.raises(ValueError, match="another neighborhood bundle plan"):
        read_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(other_dynamics, other_table, scope_id=SCOPE),
            other_template,
        )
    # A state of the grown binding cannot be written under the initial plan.
    with pytest.raises(ValueError, match="another dynamics runtime"):
        write_atomistic_checkpoint(
            path.with_name("mixed.chk"),
            AtomisticCheckpointPlan(dynamics, thermodynamic),
            grown["saved"],
        )


def test_checkpoint_refuses_tampered_and_noncanonical_manifests(
    grown: dict[str, Any], tmp_path: Path
) -> None:
    path = grown["path"]
    dynamics, thermodynamic, template = fresh_runtime(path, grown["exact"])
    plan = AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id=SCOPE)

    def momenta(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        (index,) = [
            position
            for position, value in enumerate(manifest["state"]["paths"])
            if value == ".kinematics.momenta"
        ]
        name = manifest["state"]["arrays"][index]
        arrays[name] = arrays[name] + 1.0e-9

    with pytest.raises(ValueError, match="payload identity is corrupt"):
        read_atomistic_checkpoint(
            _rewrite(path, tmp_path / "momenta.chk", momenta), plan, template
        )

    def obsolete_state(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        # A state recorded before ``AtomisticRoleEvidence.selection_changed``
        # existed: the recipe lacks those leaves.
        state = manifest["state"]
        keep = [
            index
            for index, value in enumerate(state["paths"])
            if not value.endswith(".selection_changed")
        ]
        assert len(keep) < len(state["paths"])
        for index, name in enumerate(state["arrays"]):
            if index not in keep:
                arrays.pop(name)
        state["paths"] = [state["paths"][index] for index in keep]
        state["arrays"] = [state["arrays"][index] for index in keep]
        state["num_leaves"] = len(keep)

    with pytest.raises(ValueError, match="canonical recipe"):
        read_atomistic_checkpoint(
            _rewrite(path, tmp_path / "obsolete.chk", obsolete_state), plan, template
        )

    def renamed(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        state = manifest["state"]
        first = state["arrays"][0]
        arrays["runtime/renamed"] = arrays.pop(first)
        state["arrays"][0] = "runtime/renamed"

    with pytest.raises(ValueError, match="canonical recipe"):
        read_atomistic_checkpoint(
            _rewrite(path, tmp_path / "renamed.chk", renamed), plan, template
        )

    def levels(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        del arrays
        manifest["neighborhood_bundle"]["role_levels"] = [0]

    with pytest.raises(ValueError, match="capacity levels differ"):
        read_atomistic_checkpoint(
            _rewrite(path, tmp_path / "levels.chk", levels), plan, template
        )

    def legacy(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        del arrays
        manifest.pop("neighborhood_bundle")
        manifest["potential_id"] = manifest.pop("hamiltonian_id")

    with pytest.raises(ValueError, match="canonical current format"):
        read_atomistic_checkpoint(
            _rewrite(path, tmp_path / "legacy.chk", legacy), plan, template
        )
    with pytest.raises(ValueError, match="does not match the runtime"):
        read_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(dynamics, thermodynamic, scope_id="other-scope"),
            template,
        )


def test_mtk_controllers_round_trip_and_continue_exactly(tmp_path: Path) -> None:
    def runtime() -> tuple[Any, Any, Any]:
        system = atomistic_system(
            list(range(8)), [1.0] * 8, cell=periodic_cell(VECTORS / 2)
        )
        dynamics = prepare_dynamics(
            system,
            (DirectedPairTerm("smooth-repulsion", 2.0, profile="smooth", strength=10.0),),
            MTKNPTPlan(
                0.005,
                MTKBarostatPlan(0.5),
                NoseHooverChainParameters(0.5),
                NoseHooverChainParameters(0.5),
            ),
        )
        thermodynamic = table(
            dynamics, row(measure(system), "npt", temperature=0.25, pressure=0.2)
        )
        rng = np.random.default_rng(5)
        corners = np.asarray(list(product((0.25, 0.75), repeat=3)))
        fractional = corners + rng.uniform(-0.04, 0.04, size=corners.shape)
        state = dynamics.initialize_state(
            jnp.asarray(fractional @ (VECTORS / 2)),
            thermodynamic,
            velocity=jnp.asarray(0.5 * rng.normal(size=corners.shape)),
            key=jr.key(5),
        )
        return dynamics, thermodynamic, state

    dynamics, thermodynamic, state = runtime()
    for _ in range(2):
        state = step_detailed(dynamics, state, thermodynamic).accepted_state
    assert state.controllers.cell is not None
    assert state.controllers.particle_chain is not None
    assert state.controllers.cell_chain is not None
    path = tmp_path / "mtk.chk"
    written = write_atomistic_checkpoint(
        path, AtomisticCheckpointPlan(dynamics, thermodynamic), state
    )
    reference = state
    for _ in range(2):
        reference = step_detailed(dynamics, reference, thermodynamic).accepted_state

    rebuilt, rebuilt_table, template = runtime()
    restored = read_atomistic_checkpoint(
        path, AtomisticCheckpointPlan(rebuilt, rebuilt_table), template
    )
    assert restored.payload_id == written.payload_id
    assert_tree_equal(restored.state, state)
    continued = restored.state
    for _ in range(2):
        continued = step_detailed(rebuilt, continued, rebuilt_table).accepted_state
    assert_tree_equal(continued, reference)
    # A template of another method (other controller layout) refuses.
    nvt = prepare_dynamics(rebuilt.system, (_term(),), _integrator())
    nvt_table = table(nvt, row(measure(rebuilt.system), "nvt", temperature=0.25))
    with pytest.raises(ValueError, match="does not match the runtime|another"):
        read_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(nvt, nvt_table),
            nvt.initialize_state(
                template.kinematics.positions,
                nvt_table,
                velocity=jnp.zeros_like(template.kinematics.positions),
                key=jr.key(0),
            ),
        )
