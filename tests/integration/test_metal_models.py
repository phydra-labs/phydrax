#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Selected-backend float32 qualification of MACE, Krylov solves and recurrences.

Every case executes its consumer program on the process's selected JAX backend
with explicit float32 roles while x64 stays enabled, and records the device
that holds its actual results. Native admission refuses unsupported floating
types; source Jaxpr weak literals are not executable precision evidence.

References never run inside the selected-backend process:

- MACE: one separate CPU process evaluates the same model, with the identical
  float32-rounded parameters and coordinates, under the float64 precision
  policy (``_write_mace_cpu_reference``).
- Krylov solves: host NumPy float64 arithmetic on the same float32 operator
  coefficients and right-hand sides.
- Recurrences: the closed-form Heun amplification of the same float32-rounded
  rate and step, evaluated in host float64.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticGraphLayout,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    energy_and_forces,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_prepare import prepare_mace_potential
from phydrax.nn.atomistic._radial_projection import RadialTableDeclaration
from phydrax.sparse import RelationAccumulation, StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.metal_qualification import record_metal_execution


pytestmark = pytest.mark.metal_native

_ROOT = Path(__file__).resolve().parents[2]
_FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _require_close_at_scale(observed: Any, expected: Any, relative: float) -> None:
    """Elementwise agreement within ``relative`` of the reference magnitude (>= 1)."""
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected_))))
    np.testing.assert_allclose(observed_, expected_, rtol=0.0, atol=relative * scale)


def _require_normwise(observed: Any, expected: Any, relative: float) -> None:
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    error = float(np.linalg.norm(observed_ - expected_))
    assert error <= relative * float(np.linalg.norm(expected_)), error


# Native-JAX MACE --------------------------------------------------------------

SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
_MANIFEST = json.loads(
    (_ROOT / "tests/fixtures/mace_torch_0_3_16/manifest.json").read_text(encoding="utf-8")
)
_ALL_FLOAT32 = AtomisticPrecisionPolicy(
    coordinate_dtype="float32",
    compute_dtype="float32",
    reduction_dtype="float32",
    output_dtype="float32",
)
_FLOAT64 = AtomisticPrecisionPolicy()
_CUTOFF = 3.0
# Rounded once to float32; the float64 reference evaluates these same values.
_COORDINATE_DIRECTION = (
    np.random.default_rng(17).normal(size=(1, 3, 3)).astype(np.float32)
)
# Tolerances are fixed before measurement from the existing float32 MACE
# precedents on identical rounded inputs: first-order outputs (energy, forces,
# energy tangents and parameter gradients) at 1e-4 of their own magnitude, as
# the periodic float32 source gate; second-order actions (force tangents and
# force-loss parameter gradients) at 2e-3 of their magnitude, as the float32
# force-loss gate. Magnitudes are floored at one so cancelling totals are
# gated absolutely at the precedent's 1e-4 eV.
_FIRST_ORDER = 1e-4
_SECOND_ORDER = 2e-3


def _mace_potential(precision: AtomisticPrecisionPolicy) -> MACEPotential:
    """Three-interaction degree-two density/residual model with ZBL repulsion."""
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=_CUTOFF,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=3,
            hidden_degree=2,
            edge_degree=3,
            interactions=(
                "real-agnostic-density",
                "real-agnostic-residual",
                "real-agnostic-density-residual",
            ),
            correlations=(3, 2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=2.0,
            pair_repulsion=True,
            energy_scaling="scale-shift",
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3], dtype=np.float64),
        energy_shift=np.asarray([0.2], dtype=np.float64),
        precision=precision,
        key=jr.key(5),
    )


def _with_parameter_values(
    model: MACEPotential, values: Sequence[np.ndarray], /
) -> MACEPotential:
    """The model with its trainable lane replaced by exactly these values."""
    parameters, state, fixed = partition_parameters(model)
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    if len(leaves) != len(values):
        raise ValueError("Parameter values do not match the model's trainable lane.")
    replaced = []
    for leaf, value in zip(leaves, values, strict=True):
        if leaf.shape != value.shape:
            raise ValueError("Parameter values do not match the model's leaf shapes.")
        replaced.append(jnp.asarray(value, dtype=leaf.dtype))
    return combine_parameters(
        jax.tree_util.tree_unflatten(structure, replaced), state, fixed
    )


def _structure(name: str, dtype: str, /) -> AtomicStructure:
    declaration = _MANIFEST["structures"][name]
    numbers = np.asarray(
        [{"O": 8, "H": 1}[symbol] for symbol in declaration["symbols"]],
        dtype=np.int32,
    )
    masses = np.where(numbers == 8, 15.999, 1.008)
    positions = np.asarray(declaration["positions"], dtype=np.float32).astype(dtype)
    if declaration["cell"] is None:
        return AtomicStructure(
            numbers,
            positions,
            masses,
            SCALE,
            coordinate_dtype=dtype,
            mass_accumulation_dtype=dtype,
        )
    return AtomicStructure(
        numbers,
        positions,
        masses,
        SCALE,
        cell=np.asarray(declaration["cell"], dtype=np.float32).astype(dtype),
        periodic_axes=np.ones((3,), dtype=np.bool_),
        name=name,
        coordinate_dtype=dtype,
        mass_accumulation_dtype=dtype,
    )


def _ordered_molecule_execution() -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        4,
        maximum_dense_atoms=4,
        streamed=StreamedRelationPlan(accumulation="deterministic"),
    )


def _molecule_execution() -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(4, maximum_dense_atoms=4)


def _crystal_execution(
    accumulation: RelationAccumulation,
) -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        256,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=4096,
            maximum_degree=256,
            maximum_images=343,
        ),
        streamed=StreamedRelationPlan(accumulation=accumulation),
    )


type _Program = Callable[[Any, jax.Array, jax.Array], dict[str, Any]]


def _molecule_programs(
    model: MACEPotential, dtype: str, /
) -> tuple[_Program, _Program, tuple[Any, jax.Array, jax.Array]]:
    """First- and second-order derivative programs of the molecule and their inputs."""
    batch = AtomisticBatch.from_structure(_structure("molecule", dtype))
    execution = _ordered_molecule_execution()
    payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=_CUTOFF).prepare(
        batch
    )
    parameters, state, fixed = partition_parameters(model)

    def derivatives(lane: Any, coordinates: jax.Array, forces: bool) -> Any:
        return atomistic_energy_derivatives(
            combine_parameters(lane, state, fixed),
            payload,
            coordinates,
            compute_forces=forces,
        )

    def energy(lane: Any, coordinates: jax.Array) -> jax.Array:
        return derivatives(lane, coordinates, False).energy[0]

    def forces(lane: Any, coordinates: jax.Array) -> jax.Array:
        return derivatives(lane, coordinates, True).forces

    def first_order(
        lane: Any, coordinates: jax.Array, direction: jax.Array
    ) -> dict[str, Any]:
        result = derivatives(lane, coordinates, True)
        _, tangent = jax.jvp(
            lambda value: energy(lane, value), (coordinates,), (direction,)
        )
        return {
            "energy": result.energy,
            "forces": result.forces,
            "energy_tangent": tangent,
            "parameter_gradient": jax.grad(energy)(lane, coordinates),
            "successful": result.successful,
        }

    def second_order(
        lane: Any, coordinates: jax.Array, direction: jax.Array
    ) -> dict[str, Any]:
        _, force_tangent = jax.jvp(
            lambda value: forces(lane, value), (coordinates,), (direction,)
        )
        force_loss_gradient = jax.grad(
            lambda current: jnp.sum(forces(current, coordinates) ** 2)
        )(lane)
        return {
            "force_tangent": force_tangent,
            "force_loss_gradient": force_loss_gradient,
        }

    direction = jnp.asarray(_COORDINATE_DIRECTION, dtype=dtype)
    return first_order, second_order, (parameters, batch.positions, direction)


def _crystal_program(
    model: MACEPotential, dtype: str, accumulation: RelationAccumulation, /
) -> tuple[Callable[[jax.Array, jax.Array], dict[str, Any]], AtomisticBatch]:
    batch = AtomisticBatch.from_structure(_structure("crystal", dtype))
    execution = _crystal_execution(accumulation)
    payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=_CUTOFF).prepare(
        batch
    )

    def evaluate(coordinates: jax.Array, cells: jax.Array) -> dict[str, Any]:
        result = atomistic_energy_derivatives(
            model,
            payload,
            coordinates,
            cell_vectors=cells,
            compute_stress=True,
        )
        return {
            "energy": result.energy,
            "forces": result.forces,
            "stress": result.stress,
            "successful": result.successful,
        }

    return evaluate, batch


def _host_outputs(prefix: str, outputs: dict[str, Any], /) -> dict[str, np.ndarray]:
    return {
        f"{prefix}/{name}/{index}": np.asarray(leaf)
        for name, value in outputs.items()
        for index, leaf in enumerate(jax.tree_util.tree_leaves(value))
    }


def _write_mace_cpu_reference(path: str, /) -> None:
    """Float64 CPU reference on the float32 parameters and coordinates (child entry)."""
    if jax.default_backend() != "cpu" or not phx.config.x64_enabled():
        raise RuntimeError("The MACE reference requires an x64 CPU-only process.")
    single = _mace_potential(_ALL_FLOAT32)
    values = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(partition_parameters(single)[0])
    ]
    if any(value.dtype != np.float32 for value in values):
        raise RuntimeError("The float32 MACE policy produced non-float32 parameters.")
    reference = _with_parameter_values(_mace_potential(_FLOAT64), values)
    reference.validate()
    arrays = {f"parameters/{index}": value for index, value in enumerate(values)}
    first_order, second_order, arguments = _molecule_programs(reference, "float64")
    arrays.update(_host_outputs("molecule", eqx.filter_jit(first_order)(*arguments)))
    arrays.update(_host_outputs("molecule", eqx.filter_jit(second_order)(*arguments)))
    evaluate, batch = _crystal_program(reference, "float64", "deterministic")
    if batch.cells is None:
        raise ValueError("The periodic reference requires declared cell geometry.")
    arrays.update(
        _host_outputs("crystal", eqx.filter_jit(evaluate)(batch.positions, batch.cells))
    )
    np.savez(path, **arrays)


_CPU_REFERENCE_PROGRAM = """
import importlib.util
import sys

specification = importlib.util.spec_from_file_location(
    "metal_models_cpu_reference", sys.argv[1]
)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
module._write_mace_cpu_reference(sys.argv[2])
"""


@pytest.fixture(scope="module")
def mace_cpu_reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    """Float64 MACE outputs from a separate x64 CPU-only process."""
    output = tmp_path_factory.mktemp("mace-cpu-reference") / "reference.npz"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("JAX_", "PHYDRAX_")) and key != "XLA_FLAGS"
    }
    environment.update({"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"})
    completed = subprocess.run(
        [sys.executable, "-c", _CPU_REFERENCE_PROGRAM, str(Path(__file__)), str(output)],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with np.load(output) as archive:
        return {name: archive[name] for name in archive.files}


def _reference_model(reference: dict[str, np.ndarray], /) -> MACEPotential:
    count = sum(1 for name in reference if name.startswith("parameters/"))
    values = [reference[f"parameters/{index}"] for index in range(count)]
    model = _with_parameter_values(_mace_potential(_ALL_FLOAT32), values)
    model.validate()
    return model


def _require_reference(
    observed: dict[str, Any],
    reference: dict[str, np.ndarray],
    prefix: str,
    name: str,
    relative: float,
) -> None:
    leaves = jax.tree_util.tree_leaves(observed[name])
    expected = sum(1 for key in reference if key.startswith(f"{prefix}/{name}/"))
    assert len(leaves) == expected, (name, len(leaves), expected)
    for index, leaf in enumerate(leaves):
        assert leaf.dtype == jnp.float32, (name, index, leaf.dtype)
        _require_close_at_scale(leaf, reference[f"{prefix}/{name}/{index}"], relative)


@pytest.mark.strict_jax
def test_float32_mace_energy_forces_and_parameter_gradient_track_cpu_float64(
    mace_cpu_reference: dict[str, np.ndarray],
    record_property: Callable[[str, object], None],
) -> None:
    model = _reference_model(mace_cpu_reference)
    first_order, _, arguments = _molecule_programs(model, "float32")

    observed = eqx.filter_jit(first_order)(*arguments)
    record_metal_execution(
        record_property,
        observed["energy"],
        observed["forces"],
        observed["energy_tangent"],
        *jax.tree_util.tree_leaves(observed["parameter_gradient"]),
    )

    assert bool(jnp.all(observed["successful"]))
    for name in ("energy", "forces", "energy_tangent", "parameter_gradient"):
        _require_reference(observed, mace_cpu_reference, "molecule", name, _FIRST_ORDER)
    # Forces are the reverse-mode energy derivative and the tangent its
    # forward-mode derivative: -F . d must equal dE[d] at float32 rounding.
    projected = -np.sum(
        np.asarray(observed["forces"], dtype=np.float64)
        * _COORDINATE_DIRECTION.astype(np.float64)
    )
    _require_close_at_scale(observed["energy_tangent"], projected, _FIRST_ORDER)


@pytest.mark.strict_jax
def test_float32_mace_second_order_actions_track_cpu_float64(
    mace_cpu_reference: dict[str, np.ndarray],
    record_property: Callable[[str, object], None],
) -> None:
    model = _reference_model(mace_cpu_reference)
    _, second_order, arguments = _molecule_programs(model, "float32")

    observed = eqx.filter_jit(second_order)(*arguments)
    record_metal_execution(
        record_property,
        observed["force_tangent"],
        *jax.tree_util.tree_leaves(observed["force_loss_gradient"]),
    )

    for name in ("force_tangent", "force_loss_gradient"):
        _require_reference(observed, mace_cpu_reference, "molecule", name, _SECOND_ORDER)
    # A second-order action that silently vanished would pass a loose gate.
    assert float(jnp.max(jnp.abs(observed["force_tangent"]))) > 0.0


@pytest.mark.strict_jax
@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
def test_float32_periodic_mace_stress_tracks_cpu_float64_and_replays_bitwise(
    accumulation: RelationAccumulation,
    mace_cpu_reference: dict[str, np.ndarray],
    record_property: Callable[[str, object], None],
) -> None:
    model = _reference_model(mace_cpu_reference)
    evaluate, batch = _crystal_program(model, "float32", accumulation)
    if batch.cells is None:
        raise ValueError("The periodic MACE case requires declared cell geometry.")

    compiled = eqx.filter_jit(evaluate)
    expected = compiled(batch.positions, batch.cells)
    record_metal_execution(
        record_property, expected["energy"], expected["forces"], expected["stress"]
    )

    assert bool(jnp.all(expected["successful"]))
    for name in ("energy", "forces", "stress"):
        _require_reference(expected, mace_cpu_reference, "crystal", name, _FIRST_ORDER)
    # Ordered accumulation fixes every position, cell and case reduction order,
    # so a repeated execution of one executable is bitwise identical.
    for _ in range(3):
        replay = compiled(batch.positions, batch.cells)
        for name in ("energy", "forces", "stress", "successful"):
            np.testing.assert_array_equal(
                np.asarray(replay[name]), np.asarray(expected[name])
            )


@pytest.mark.strict_jax
def test_float32_mace_unknown_species_fails_closed_after_an_accepted_evaluation(
    record_property: Callable[[str, object], None],
) -> None:
    model = _mace_potential(_ALL_FLOAT32)
    accepted = energy_and_forces(
        model, _structure("molecule", "float32"), _molecule_execution()
    )
    record_metal_execution(record_property, accepted.energy, accepted.forces)
    assert bool(accepted.valid[0])
    assert accepted.forces.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(accepted.forces)))

    carbon = AtomicStructure(
        np.asarray([6, 1, 1], dtype=np.int32),
        np.asarray(_MANIFEST["structures"]["molecule"]["positions"], dtype=np.float32),
        np.asarray([12.011, 1.008, 1.008], dtype=np.float64),
        SCALE,
        coordinate_dtype="float32",
        mass_accumulation_dtype="float32",
    )
    # Compiled guards surface through the JAX runtime when raised inside a
    # device program; the message identifies the species refusal either way.
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="outside the MACE model species",
    ):
        jax.block_until_ready(
            energy_and_forces(model, carbon, _molecule_execution()).energy
        )


@pytest.mark.strict_jax
def test_float32_tabulated_mace_refuses_radii_outside_its_declared_support(
    record_property: Callable[[str, object], None],
) -> None:
    model = _mace_potential(_ALL_FLOAT32)
    tabulated = prepare_mace_potential(
        model,
        radial="tabulated",
        tables=RadialTableDeclaration(0.3, 2048, layout="projected-width"),
    )
    tabulated.validate()
    batch = AtomisticBatch.from_structure(_structure("molecule", "float32"))
    execution = _molecule_execution()
    payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=_CUTOFF).prepare(
        batch
    )

    def energy(potential: Any, positions: jax.Array) -> jax.Array:
        graph = payload.graph(cutoff=_CUTOFF, positions=positions)
        result = potential.evaluate_graph(
            AtomisticGraphInputs(
                graph, batch.scale, species_kind=potential.capabilities.species_kind
            )
        )
        return result.energy[0]

    compiled = eqx.filter_jit(energy)
    accepted = compiled(tabulated, batch.positions)
    exact = compiled(model, batch.positions)
    record_metal_execution(record_property, accepted, exact)
    assert accepted.dtype == jnp.float32
    # The 2048-node table is within 1e-7 of the exact float64 radial field;
    # in float32 the first-order energy gate bounds the comparison.
    _require_close_at_scale(accepted, exact, _FIRST_ORDER)

    collapsed = batch.positions.at[0, 1].set(
        batch.positions[0, 0] + jnp.asarray([0.1, 0.0, 0.0], dtype=jnp.float32)
    )
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="outside the declared support",
    ):
        jax.block_until_ready(compiled(tabulated, collapsed))


# Native matrix-free Krylov solves ----------------------------------------------

_KRYLOV_SIZE = 16
_KRYLOV_RANDOM = np.random.default_rng(23)
_KRYLOV_RHS = _KRYLOV_RANDOM.normal(size=_KRYLOV_SIZE).astype(np.float32)
_KRYLOV_DIRECTION = _KRYLOV_RANDOM.normal(size=_KRYLOV_SIZE).astype(np.float32)
_KRYLOV_COTANGENT = _KRYLOV_RANDOM.normal(size=_KRYLOV_SIZE).astype(np.float32)
# Operator coefficients are exactly representable in float32, so the host
# float64 references solve the identical operator.
_SPD = (3.0, 1.0, 1.0)
_NONSYMMETRIC = (4.0, 1.5, 0.5)


def _tridiagonal_action(
    diagonal: jax.Array | float, lower: float, upper: float
) -> Callable[[jax.Array], jax.Array]:
    """Matrix-free ``(A v)_i = d v_i - l v_{i-1} - u v_{i+1}`` with zero ends."""

    def action(vector: jax.Array) -> jax.Array:
        zero = jnp.zeros((1,), dtype=vector.dtype)
        below = jnp.concatenate((zero, vector[:-1]))
        above = jnp.concatenate((vector[1:], zero))
        return diagonal * vector - lower * below - upper * above

    return action


def _tridiagonal_matrix(diagonal: float, lower: float, upper: float) -> np.ndarray:
    """Host float64 matrix of ``_tridiagonal_action`` for independent references."""
    return (
        diagonal * np.eye(_KRYLOV_SIZE)
        - lower * np.eye(_KRYLOV_SIZE, k=-1)
        - upper * np.eye(_KRYLOV_SIZE, k=1)
    )


def _krylov_system(
    diagonal: jax.Array | float, lower: float, upper: float, *, spd: bool
) -> Any:
    la = phx.linalg
    space = la.ArraySpace((_KRYLOV_SIZE,), dtype=jnp.float32)
    properties = (
        la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        )
        if spd
        else None
    )
    return la.LinearSystem(
        la.FunctionLinearOperator(
            _tridiagonal_action(diagonal, lower, upper),
            source=space,
            target=space,
            transpose_action=_tridiagonal_action(diagonal, upper, lower),
            properties=properties,
        )
    )


def _pcg_policy() -> Any:
    la = phx.linalg
    return la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-4, absolute=0.0, max_steps=64),
        differentiation=la.DifferentiationPolicy("none"),
        failure=la.FailurePolicy("status"),
    )


def _gmres_policy(mode: Any, *, relative: float, max_steps: int) -> Any:
    la = phx.linalg
    return la.LinearSolvePolicy(
        la.FGMRES(restart=_KRYLOV_SIZE),
        tolerance=la.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        derivative_solve=la.LinearDerivativeSolvePolicy(
            route="krylov",
            relative_tolerance=1e-5,
            absolute_tolerance=0.0,
            maximum_steps=64,
        ),
        differentiation=la.DifferentiationPolicy(mode),
        failure=la.FailurePolicy("status"),
    )


@pytest.mark.strict_jax
def test_float32_matrix_free_pcg_converges_with_consumer_true_residual(
    record_property: Callable[[str, object], None],
) -> None:
    la = phx.linalg
    problem = _krylov_system(*_SPD, spd=True)
    policy = _pcg_policy()
    assert la.plan(problem, policy).backend == "native-krylov"
    rhs = jnp.asarray(_KRYLOV_RHS)

    def solve(target: jax.Array) -> Any:
        return la.solve(problem, target, policy=policy)

    result = eqx.filter_jit(solve)(rhs)
    record_metal_execution(record_property, result.value, result.status)
    assert result.value.dtype == jnp.float32
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)

    # The consumer recomputes the true residual of the returned float32 iterate
    # in host float64. The solve confirmed ||A x - b|| <= 1e-4 ||b|| on the
    # device; host re-evaluation differs by O(n eps ||A|| ||x||) << 1e-5 ||b||.
    matrix = _tridiagonal_matrix(*_SPD)
    target = _KRYLOV_RHS.astype(np.float64)
    value = np.asarray(result.value, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    residual = float(np.linalg.norm(matrix @ value - target))
    assert residual <= 1.1e-4 * target_norm
    assert abs(float(result.diagnostics.residual_norm) - residual) <= 1e-5 * target_norm
    # A-posteriori forward error: ||x - x*|| <= ||r|| / lambda_min exactly, with
    # lambda_min = 3 - 2 cos(pi / (n + 1)) of this symmetric Toeplitz operator.
    smallest = 3.0 - 2.0 * np.cos(np.pi / (_KRYLOV_SIZE + 1))
    error = float(np.linalg.norm(value - np.linalg.solve(matrix, target)))
    assert error <= residual / smallest + 1e-12 * target_norm

    checked, evidence = la.solve_checked(
        problem,
        rhs,
        policy=policy,
        check_policy=la.LinearSolveCheckPolicy(
            relative_tolerance=2e-4, absolute_tolerance=0.0
        ),
    )
    assert int(checked.status) == int(la.LinearSolveStatus.SUCCESS)
    assert bool(evidence.finite) and bool(evidence.converged)
    assert bool(evidence.residual_ok) and bool(evidence.valid)


@pytest.mark.strict_jax
def test_float32_matrix_free_gmres_mathematical_derivatives_match_host_solves(
    record_property: Callable[[str, object], None],
) -> None:
    la = phx.linalg
    policy = _gmres_policy("mathematical", relative=1e-4, max_steps=64)
    assert (
        la.plan(_krylov_system(*_NONSYMMETRIC, spd=False), policy).backend
        == "native-krylov"
    )
    lower, upper = _NONSYMMETRIC[1:]

    def solve(shift: jax.Array, rhs: jax.Array) -> Any:
        system = _krylov_system(_NONSYMMETRIC[0] + shift, lower, upper, spd=False)
        return la.solve(system, rhs, policy=policy)

    def state(shift: jax.Array, rhs: jax.Array) -> jax.Array:
        return solve(shift, rhs).value

    def derivatives(
        shift: jax.Array, rhs: jax.Array, direction: jax.Array, cotangent: jax.Array
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
        primal = solve(shift, rhs)
        _, tangent = jax.jvp(
            state,
            (shift, rhs),
            (jnp.asarray(0.25, dtype=jnp.float32), direction),
        )
        _, pullback = jax.vjp(state, shift, rhs)
        shift_gradient, rhs_gradient = pullback(cotangent)
        return primal, tangent, shift_gradient, rhs_gradient

    arguments = (
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(_KRYLOV_RHS),
        jnp.asarray(_KRYLOV_DIRECTION),
        jnp.asarray(_KRYLOV_COTANGENT),
    )

    primal, tangent, shift_gradient, rhs_gradient = eqx.filter_jit(derivatives)(
        *arguments
    )
    record_metal_execution(
        record_property, primal.value, tangent, shift_gradient, rhs_gradient
    )
    assert int(primal.status) == int(la.LinearSolveStatus.SUCCESS)
    assert bool(primal.derivative_valid)
    for value in (primal.value, tangent, shift_gradient, rhs_gradient):
        assert value.dtype == jnp.float32

    # dA/dshift = I. Diagonal dominance by 2 in every row and column bounds
    # sigma_min >= 2 and sigma_max <= 6, so kappa_2 <= 3: the 1e-4 primal and
    # 1e-5 derivative residuals bound each forward error by 3e-4; the 1e-3 gate
    # adds float32 evaluation margin.
    matrix = _tridiagonal_matrix(*_NONSYMMETRIC)
    target = _KRYLOV_RHS.astype(np.float64)
    expected = np.linalg.solve(matrix, target)
    expected_tangent = np.linalg.solve(
        matrix, _KRYLOV_DIRECTION.astype(np.float64) - 0.25 * expected
    )
    adjoint = np.linalg.solve(matrix.T, _KRYLOV_COTANGENT.astype(np.float64))
    _require_normwise(primal.value, expected, 1e-3)
    _require_normwise(tangent, expected_tangent, 1e-3)
    _require_normwise(rhs_gradient, adjoint, 1e-3)
    _require_normwise(shift_gradient, -np.dot(adjoint, expected), 1e-3)


@pytest.mark.strict_jax
def test_float32_iteration_limit_reports_status_and_invalidates_implicit_derivative(
    record_property: Callable[[str, object], None],
) -> None:
    la = phx.linalg
    system = _krylov_system(*_NONSYMMETRIC, spd=False)
    rhs = jnp.asarray(_KRYLOV_RHS)
    direction = jnp.asarray(_KRYLOV_DIRECTION)
    cotangent = jnp.asarray(_KRYLOV_COTANGENT)

    def objective(mode: str) -> Callable[[jax.Array], jax.Array]:
        policy = _gmres_policy(mode, relative=1e-6, max_steps=2)
        return lambda target: jnp.vdot(
            cotangent, la.solve(system, target, policy=policy).value
        )

    def limited(target: jax.Array) -> tuple[Any, Any, jax.Array, jax.Array, jax.Array]:
        implicit = la.solve(
            system,
            target,
            policy=_gmres_policy("mathematical", relative=1e-6, max_steps=2),
        )
        unrolled = la.solve(
            system,
            target,
            policy=_gmres_policy("algorithmic", relative=1e-6, max_steps=2),
        )
        failed_gradient = jax.grad(objective("mathematical"))(target)
        reverse = jax.grad(objective("algorithmic"))(target)
        _, forward = jax.jvp(objective("algorithmic"), (target,), (direction,))
        return implicit, unrolled, failed_gradient, jnp.vdot(reverse, direction), forward

    implicit, unrolled, failed_gradient, reverse, forward = eqx.filter_jit(limited)(rhs)
    record_metal_execution(
        record_property, implicit.value, unrolled.value, failed_gradient, reverse
    )

    for result in (implicit, unrolled):
        assert int(result.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
        assert not bool(result.successful)
        assert int(result.diagnostics.iterations) == 2
        assert bool(result.diagnostics.finite)
    # The implicit contract needs a converged solve: its derivative is invalid
    # and poisoned, never a finite plausible value.
    assert not bool(implicit.derivative_valid)
    assert not bool(jnp.all(jnp.isfinite(failed_gradient)))
    # The unrolled contract differentiates the executed iteration: reverse and
    # forward mode of one executed float32 program agree at its rounding.
    assert bool(unrolled.derivative_valid)
    _require_close_at_scale(reverse, forward, 1e-4)


@pytest.mark.strict_jax
def test_float32_matrix_free_bicgstab_breakdown_keeps_a_finite_iterate(
    record_property: Callable[[str, object], None],
) -> None:
    la = phx.linalg
    space = la.ArraySpace((2,), dtype=jnp.float32)
    # A plane rotation: r0 is orthogonal to A r0, so the alpha denominator of
    # the first BiCGStab step vanishes exactly.
    rotation = la.FunctionLinearOperator(
        lambda vector: jnp.stack((vector[1], -vector[0])), source=space, target=space
    )
    policy = la.LinearSolvePolicy(
        la.BiCGStab(),
        tolerance=la.TolerancePolicy(relative=1e-6, max_steps=10),
        failure=la.FailurePolicy("status"),
    )

    def solve(target: jax.Array) -> Any:
        return la.solve(la.LinearSystem(rotation), target, policy=policy)

    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    result = eqx.filter_jit(solve)(rhs)
    record_metal_execution(record_property, result.value, result.status)

    assert int(result.status) == int(la.LinearSolveStatus.BREAKDOWN)
    assert not bool(result.successful)
    assert bool(result.diagnostics.finite)
    assert int(result.diagnostics.iterations) == 1
    np.testing.assert_array_equal(np.asarray(result.value), np.zeros(2, np.float32))
    np.testing.assert_allclose(float(result.diagnostics.residual_norm), 1.0)


@pytest.mark.strict_jax
def test_float32_nonfinite_rhs_reports_status_and_invalid_evidence(
    record_property: Callable[[str, object], None],
) -> None:
    la = phx.linalg
    problem = _krylov_system(*_SPD, spd=True)
    policy = _pcg_policy()
    rhs = jnp.asarray(_KRYLOV_RHS).at[3].set(jnp.asarray(jnp.inf, dtype=jnp.float32))

    def solve(target: jax.Array) -> Any:
        return la.solve(problem, target, policy=policy)

    result = eqx.filter_jit(solve)(rhs)
    record_metal_execution(record_property, result.value, result.status)
    assert int(result.status) == int(la.LinearSolveStatus.NONFINITE_INPUT)
    assert not bool(result.successful)

    _, evidence = la.solve_checked(problem, rhs, policy=policy)
    assert not bool(evidence.finite)
    assert not bool(evidence.valid)


# Explicit fixed-step temporal recurrence ----------------------------------------


def _temporal_precision() -> Any:
    return phx.solver.TemporalPrecisionPolicy(
        state_dtype="float32", output_dtype="float32"
    )


def _heun(
    drift: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    initial: np.ndarray,
    parameter: jax.Array,
    steps: int,
    *,
    max_steps: int,
) -> Any:
    """Heun's method on [0, 1] with a float32 step that lands exactly on t1."""
    problem = phx.solver.DifferentialProblem(
        drift,
        jnp.asarray(initial, dtype=jnp.float32),
        t0=0.0,
        t1=1.0,
        args=parameter,
        problem_id="metal-heun-recurrence",
    )
    return phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.5, 1.0], dtype=jnp.float32),
        solver=dfx.Heun(),
        # Heun carries an embedded error estimate, so the fixed recurrence the
        # closed-form oracle describes must be selected explicitly.
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=1.0 / steps,
        max_steps=max_steps,
        precision=_temporal_precision(),
    )


def _decay(time: jax.Array, state: jax.Array, rate: jax.Array) -> jax.Array:
    del time
    return -rate * state


def _oscillator(time: jax.Array, state: jax.Array, frequency: jax.Array) -> jax.Array:
    del time
    return jnp.stack((state[1], -(frequency**2) * state[0]))


def _heun_amplification(matrix: np.ndarray, step: float) -> np.ndarray:
    """Exact one-step map I + hA + (hA)^2 / 2 of Heun's method on y' = A y."""
    scaled = step * matrix
    return np.eye(matrix.shape[0], dtype=matrix.dtype) + scaled + 0.5 * scaled @ scaled


def _require_temporal_status(solution: Any, steps: int) -> None:
    assert solution.temporal_evidence is not None
    assert not solution.temporal_evidence.adaptive
    assert bool(solution.successful)
    assert bool(solution.terminal_successful)
    assert int(solution.stats["num_steps"]) == steps
    assert solution.times.dtype == jnp.float32
    assert solution.states.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(solution.terminal_time), np.float32(1.0))


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(2, id="two-steps"),
        pytest.param(8, id="eight-steps"),
        pytest.param(512, id="long-512-steps"),
    ],
)
def test_float32_heun_decay_matches_closed_form_value_and_rate_derivative(
    steps: int, record_property: Callable[[str, object], None]
) -> None:
    rate = jnp.asarray(1.25, dtype=jnp.float32)
    initial = np.ones((1,), dtype=np.float32)

    def run(value: jax.Array) -> tuple[Any, jax.Array]:
        solution = _heun(_decay, initial, value, steps, max_steps=steps)
        gradient = jax.grad(
            lambda current: _heun(
                _decay, initial, current, steps, max_steps=steps
            ).states[-1, 0]
        )(value)
        return solution, gradient

    solution, gradient = eqx.filter_jit(run)(rate)
    record_metal_execution(record_property, solution.states, gradient)
    _require_temporal_status(solution, steps)
    assert gradient.dtype == jnp.float32

    # Closed form of the executed recurrence on the same float32-rounded rate
    # and step: y_N = R^N with R = 1 - z + z^2/2, z = k h. Each step rounds a
    # bounded number of float32 operations, so the relative error stays below
    # 8 (N + 1) eps for the value and 16 (N + 1) eps for its reverse derivative.
    step = float(np.float32(1.0 / steps))
    z = float(np.float32(1.25)) * step
    amplification = 1.0 - z + 0.5 * z * z
    half = steps // 2
    np.testing.assert_allclose(
        np.asarray(solution.states[:, 0], dtype=np.float64),
        [amplification**half, amplification**steps],
        rtol=8.0 * (steps + 1) * _FLOAT32_EPS,
    )
    np.testing.assert_allclose(
        float(gradient),
        steps * amplification ** (steps - 1) * (z - 1.0) * step,
        rtol=16.0 * (steps + 1) * _FLOAT32_EPS,
    )


def test_float32_heun_oscillator_matches_closed_form_value_and_frequency_derivative(
    record_property: Callable[[str, object], None],
) -> None:
    steps = 64
    frequency = jnp.asarray(2.0, dtype=jnp.float32)
    initial = np.asarray([1.0, 0.0], dtype=np.float32)

    def run(value: jax.Array) -> tuple[Any, jax.Array]:
        solution = _heun(_oscillator, initial, value, steps, max_steps=steps)
        gradient = jax.grad(
            lambda current: _heun(
                _oscillator, initial, current, steps, max_steps=steps
            ).states[-1, 0]
        )(value)
        return solution, gradient

    solution, gradient = eqx.filter_jit(run)(frequency)
    record_metal_execution(record_property, solution.states, gradient)
    _require_temporal_status(solution, steps)

    step = float(np.float32(1.0 / steps))

    def terminal(omega: complex) -> np.ndarray:
        generator = np.asarray([[0.0, 1.0], [-(omega**2), 0.0]], dtype=np.complex128)
        return np.linalg.matrix_power(
            _heun_amplification(generator, step), steps
        ) @ initial.astype(np.complex128)

    expected = terminal(2.0).real
    # Complex-step derivative of the host recurrence: exact to float64 rounding.
    perturbation = 1e-30
    expected_gradient = terminal(2.0 + 1j * perturbation)[0].imag / perturbation
    tolerance = 16.0 * (steps + 1) * _FLOAT32_EPS
    _require_close_at_scale(solution.states[-1], expected, tolerance)
    _require_close_at_scale(gradient, expected_gradient, 2.0 * tolerance)


def test_float32_heun_step_budget_exhaustion_reports_failure(
    record_property: Callable[[str, object], None],
) -> None:
    rate = jnp.asarray(1.25, dtype=jnp.float32)
    initial = np.ones((1,), dtype=np.float32)

    def run(value: jax.Array) -> Any:
        return _heun(_decay, initial, value, 8, max_steps=7)

    solution = eqx.filter_jit(run)(rate)
    record_metal_execution(record_property, solution.states, solution.valid)

    # The bounded loop stops at its declared budget; the unreached terminal
    # save is reported invalid instead of a plausible state.
    assert solution.temporal_evidence is not None
    assert not solution.temporal_evidence.adaptive
    assert int(solution.stats["num_steps"]) == 7
    assert not bool(solution.backend_successful)
    assert not bool(solution.successful)
    assert bool(solution.valid[0])
    assert not bool(solution.valid[1])
