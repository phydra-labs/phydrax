"""Two-model conditioned periodic NVT dynamics for multi-model restart scenarios.

Term ``alpha`` runs a tiny MACE with a joint embedding: its bound conditioning
carries an explicit ``total_charge`` value, a ``temperature`` supplied as a
broadcast default (``defaults_used`` evidence) and an optional
``solvent_fraction`` field that stays absent; the term also binds a typed
random key. Term ``beta`` (force group 1) runs a different unconditioned MACE
revision. Both models are random native initializations, not qualified
pretrained potentials.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._model._ports import ValuePort
from phydrax.atomistic._checkpoint import write_atomistic_restart
from phydrax.atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
)
from phydrax.nn.atomistic._mace_embedding import MACEEmbeddingFeature
from tests._support.atomistic_bundles import prepare_program
from tests._support.mace_deployment import (
    bundle_resources,
    electronvolt_units,
    graph_execution,
    tiny_mace,
)
from tests._support.mace_source import WATER, water_system


REPOSITORY = Path(__file__).resolve().parents[2]
SAVED_STEPS = 3
CONTINUED_STEPS = 4
LICENSES = {"alpha": ["MIT"], "beta": ["Apache-2.0"]}
FEATURES = (
    MACEEmbeddingFeature(
        "total_charge", "categorical", level="case", width=3, classes=5, offset=2
    ),
    MACEEmbeddingFeature("temperature", "continuous", level="case", width=2, bias=False),
)
SOLVENT_PORT = ValuePort(
    "test.solvent-volume-fraction",
    event_shape=(),
    component_ids=("fraction",),
    representation="volume-fraction",
    normalization_id="identity",
)


def conditioned_mace(*, seed: int = 7) -> Any:
    """Tiny two-layer MACE whose joint embedding consumes charge and temperature."""
    architecture = phx.nn.atomistic.MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=2.0,
        embedding_features=FEATURES,
        embedding_readout=True,
    )
    return phx.nn.atomistic.MACEPotential(
        electronvolt_units().scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(seed),
    )


def alpha_conditioning(
    model: Any,
    *,
    charge: int = 1,
    temperature_default: bool = True,
    solvent_field: bool = True,
) -> AtomisticConditioning:
    """Bound conditioning of term ``alpha`` with one default-used field."""
    fields = tuple(model.conditioning.fields)
    if solvent_field:
        fields = (
            *fields,
            AtomisticConditionField(
                "solvent_fraction",
                SOLVENT_PORT,
                required=False,
                minimum=0.0,
                maximum=1.0,
            ),
        )
    values: dict[str, Any] = {"total_charge": np.asarray([charge], dtype=np.int64)}
    defaults: dict[str, Any] = {}
    if temperature_default:
        defaults["temperature"] = np.asarray(0.3)
    else:
        values["temperature"] = np.asarray([0.3])
    return AtomisticConditioning(
        AtomisticConditioningSpec(fields),
        values,
        case_count=1,
        atom_capacity=len(WATER),
        defaults=defaults,
    )


def alpha_term(
    model: Any,
    conditioning: AtomisticConditioning | None,
    *,
    key: Any = None,
    force_group: int = 0,
) -> Any:
    return phx.atomistic.LearnedGraphPotentialTerm(
        model,
        name="alpha",
        force_group=force_group,
        allow_periodic=True,
        conditioning=conditioning,
        key=jax.random.key(11) if key is None else key,
    )


def named_term(
    model: Any,
    name: str,
    *,
    force_group: int = 1,
    conditioning: AtomisticConditioning | None = None,
) -> Any:
    return phx.atomistic.LearnedGraphPotentialTerm(
        model,
        name=name,
        force_group=force_group,
        allow_periodic=True,
        conditioning=conditioning,
    )


def models() -> tuple[Any, Any]:
    """The conditioned ``alpha`` model and a distinct ``beta`` model revision."""
    return conditioned_mace(seed=7), tiny_mace(seed=5)


def reference_terms(model_a: Any, model_b: Any) -> list[Any]:
    return [
        alpha_term(model_a, alpha_conditioning(model_a)),
        named_term(model_b, "beta"),
    ]


def nvt(terms: list[Any], system: Any) -> tuple[Any, Any]:
    """Fixed-cell BAOAB Langevin NVT of one program over ``terms``.

    Deterministic streamed accumulation makes continuation bitwise within a
    process; fresh-process GPU continuation also needs the fixed profile.
    """
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(list(terms)),
        system,
        graph_execution=graph_execution(accumulation="deterministic"),
        **bundle_resources(),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, potential, phx.atomistic.BAOABLangevinPlan(0.25, 0.02)
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=300.0,
    ).prepare(dynamics)
    return dynamics, thermodynamic


def template(dynamics: Any, thermodynamic: Any) -> Any:
    """Structural state template; restart replaces every leaf."""
    return dynamics.initialize_state(
        jnp.asarray(WATER),
        thermodynamic,
        velocity=jnp.zeros((len(WATER), 3)),
        key=jax.random.key(0),
    )


def checkpoint_plan(dynamics: Any, thermodynamic: Any) -> Any:
    return phx.atomistic.AtomisticCheckpointPlan(
        dynamics, thermodynamic, scope_id="rank-0"
    )


def write_scenario(path: Path) -> dict[str, Any]:
    """Saved two-model NVT state, its restart at ``path`` and the reference."""
    system = water_system()
    model_a, model_b = models()
    dynamics, thermodynamic = nvt(reference_terms(model_a, model_b), system)
    velocity = 0.003 * np.cos(np.arange(9.0)).reshape(3, 3)
    state = dynamics.initialize_state(
        jnp.asarray(WATER),
        thermodynamic,
        velocity=jnp.asarray(velocity),
        key=jax.random.key(7),
    )
    for _ in range(SAVED_STEPS):
        state = dynamics.step(state, thermodynamic)
    plan = checkpoint_plan(dynamics, thermodynamic)
    write_atomistic_restart(path, plan, state, licenses=LICENSES)
    reference = state
    for _ in range(CONTINUED_STEPS):
        reference = dynamics.step(reference, thermodynamic)
    return {
        "path": path,
        "system": system,
        "models": (model_a, model_b),
        "dynamics": dynamics,
        "thermodynamic": thermodynamic,
        "plan": plan,
        "saved": state,
        "reference": reference,
    }


# Every fresh process refuses provider packages: restoration and continuation
# consume only the declared native models, term bindings and state data.
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

_CONTINUATION = """
import json

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.atomistic._checkpoint import (
    read_atomistic_restart,
    read_atomistic_restart_models,
)
from tests._support.mace_source import water_system
from tests._support.multi_model_restart import checkpoint_plan, nvt, template

restart, output, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
bindings = read_atomistic_restart_models(restart)
dynamics, thermodynamic = nvt([binding.term for binding in bindings.values()], water_system())
plan = checkpoint_plan(dynamics, thermodynamic)
state = read_atomistic_restart(restart, plan, template(dynamics, thermodynamic)).state
recomputed = dynamics.initialize_state(
    state.kinematics.positions,
    thermodynamic,
    velocity=jnp.zeros_like(state.kinematics.positions),
    key=jax.random.key(0),
)
summary = {
    "terms": {
        name: {
            "term_id": binding.term.term_id,
            "artifact_id": binding.artifact.manifest.artifact_id,
            "force_group": binding.term.force_group,
            "licenses": list(binding.artifact.manifest.licenses),
            "conditioned": binding.term.conditioning is not None,
            "keyed": binding.term.key is not None,
        }
        for name, binding in bindings.items()
    },
    "potential_energy": np.asarray(state.force.potential_energy).tolist(),
    "term_energies": np.asarray(state.force.term_energies).tolist(),
    "forces": np.asarray(state.force.forces).tolist(),
    "recomputed_potential_energy": np.asarray(recomputed.force.potential_energy).tolist(),
    "recomputed_forces": np.asarray(recomputed.force.forces).tolist(),
}
for _ in range(steps):
    state = dynamics.step(state, thermodynamic)
phx.atomistic.write_atomistic_checkpoint(output, plan, state)
print(json.dumps(summary))
"""


def fresh_continuation(restart: Path, output: Path, steps: int) -> dict[str, Any]:
    """Resume ``restart`` in a fresh process from its bindings alone.

    The process rebuilds the program from the restored terms, resumes,
    reports the restored and recomputed energies/forces, continues ``steps``
    steps and writes a runtime checkpoint to ``output``.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PRELUDE + textwrap.dedent(_CONTINUATION),
            str(restart),
            str(output),
            str(steps),
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def host_leaves(tree: Any) -> list[np.ndarray]:
    def host(leaf: Any) -> np.ndarray:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf)

    return [host(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def assert_identical(observed: Any, expected: Any) -> None:
    left, right = host_leaves(observed), host_leaves(expected)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
