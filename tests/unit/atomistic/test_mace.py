import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

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
from phydrax.atomistic._model_artifact import (
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_source import mace_potential_from_source
from phydrax.sparse import RelationAccumulation, StreamedRelationPlan
from phydrax.special import RealCartesianHarmonics
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.mace_source import FLOAT32_GATES


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/mace_torch_0_3_16"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
INTERACTION_KINDS = {
    "RealAgnosticInteractionBlock": "real-agnostic",
    "RealAgnosticResidualInteractionBlock": "real-agnostic-residual",
    "RealAgnosticDensityInteractionBlock": "real-agnostic-density",
    "RealAgnosticDensityResidualInteractionBlock": "real-agnostic-density-residual",
}
# Source parameter names are the trainable source leaves; every other source
# tensor is a fixed buffer (reference energies, scales, radial frequencies, U).
DROPPED_SOURCE_BUFFERS = ("output_mask", "zeroed", "num_interactions", "_w3j")


def _molecule_execution() -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(4, maximum_dense_atoms=4)


def _periodic_execution(
    *, accumulation: RelationAccumulation = "fast"
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


def _structure(name: str, *, positions: Any = None) -> AtomicStructure:
    declaration = MANIFEST["structures"][name]
    numbers = np.asarray(
        [{"O": 8, "H": 1}[symbol] for symbol in declaration["symbols"]], dtype=np.int32
    )
    masses = np.where(numbers == 8, 15.999, 1.008).astype(np.float64)
    coordinates = np.asarray(
        declaration["positions"] if positions is None else positions, dtype=np.float64
    )
    if declaration["cell"] is None:
        return AtomicStructure(numbers, coordinates, masses, SCALE)
    return AtomicStructure(
        numbers,
        coordinates,
        masses,
        SCALE,
        cell=np.asarray(declaration["cell"], dtype=np.float64),
        periodic_axes=np.ones((3,), dtype=np.bool_),
    )


def _source_architecture(case: str, head: str) -> MACEArchitecture:
    declaration = MANIFEST["cases"][case]
    count = declaration["interactions"]
    kinds = (INTERACTION_KINDS[declaration["first"]],) + (
        INTERACTION_KINDS[declaration["rest"]],
    ) * (count - 1)
    gate = declaration["normalize2mom_silu"]
    agnesi = declaration["distance_transform"] == "Agnesi"
    return MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=kinds,
        correlations=tuple(declaration["correlation"]),
        radial_widths=(8,),
        readout_width=5 if count > 1 else None,
        radial_activation_scale=gate,
        readout_activation_scale=gate if count > 1 else None,
        average_neighbor_count=2.0,
        heads=tuple(declaration["heads"]),
        head=head,
        energy_scaling="scale-shift"
        if declaration["class"] == "ScaleShiftMACE"
        else "unscaled",
        pair_repulsion=declaration["pair_repulsion"],
        distance_transform="agnesi" if agnesi else "none",
        agnesi_parameters=(1.0805, 0.9183, 4.5791) if agnesi else None,
        cutoff_placement="embedding" if declaration["apply_cutoff"] else "weights",
    )


def _source_basis(
    fixture: Any,
) -> tuple[list[np.ndarray], dict[tuple[int, int, int], np.ndarray]]:
    """Per-degree maps from sampled source harmonics to native harmonics."""
    native = np.asarray(
        RealCartesianHarmonics(2, normalization="fully_normalized")(
            fixture["harmonics/vectors"]
        )
    )
    source = np.asarray(fixture["harmonics/values"], dtype=np.float64)
    transforms: list[np.ndarray] = []
    for degree in range(3):
        block = slice(degree * degree, (degree + 1) ** 2)
        solution, *_ = np.linalg.lstsq(source[:, block], native[:, block], rcond=None)
        transforms.append(np.asarray(solution, dtype=np.float64).T)
    couplings: dict[tuple[int, int, int], np.ndarray] = {}
    for name in fixture.files:
        if name.startswith("w3j/"):
            left, right, output = (int(value) for value in name.split("/")[1].split("_"))
            couplings[(left, right, output)] = np.asarray(fixture[name], dtype=np.float64)
    return transforms, couplings


def _source_tensors(
    fixture: Any, offsets: dict[str, np.ndarray] | None = None
) -> dict[str, Any]:
    tensors = {}
    for name in fixture.files:
        if not name.startswith("state/"):
            continue
        key = name[len("state/") :]
        value = fixture[name]
        if value.size == 0 or any(marker in key for marker in DROPPED_SOURCE_BUFFERS):
            continue
        if offsets is not None and key in offsets:
            value = value + offsets[key]
        tensors[key] = value
    return tensors


def _source_potential(
    case: str,
    head: str,
    offsets: dict[str, np.ndarray] | None = None,
    *,
    precision: AtomisticPrecisionPolicy | None = None,
) -> MACEPotential:
    fixture = np.load(FIXTURES / f"{case}.npz")
    transforms, couplings = _source_basis(fixture)
    return mace_potential_from_source(
        SCALE,
        _source_architecture(case, head),
        _source_tensors(fixture, offsets),
        degree_transforms=transforms,
        couplings=couplings,
        source_id=f"mace-torch-0.3.16-fixture:{case}",
        precision=precision,
    )


SOURCE_ROWS = [
    pytest.param(case, head, id=f"{case}-{head}")
    for case, declaration in MANIFEST["cases"].items()
    for head in declaration["heads"]
]


@pytest.mark.parametrize(("case", "head"), SOURCE_ROWS)
def test_source_reconstruction_matches_provider_molecule(case: str, head: str) -> None:
    fixture = np.load(FIXTURES / f"{case}.npz")
    potential = _source_potential(case, head)
    potential.validate()
    prediction = energy_and_forces(
        potential, _structure("molecule"), _molecule_execution()
    )
    assert bool(prediction.valid[0])
    np.testing.assert_allclose(
        prediction.energy, fixture[f"molecule/{head}/energy"], rtol=1e-12, atol=1e-11
    )
    np.testing.assert_allclose(
        prediction.forces[0], fixture[f"molecule/{head}/forces"], rtol=1e-10, atol=1e-11
    )


@pytest.mark.parametrize(("case", "head"), SOURCE_ROWS)
def test_source_reconstruction_matches_provider_periodic_crystal(
    case: str, head: str
) -> None:
    fixture = np.load(FIXTURES / f"{case}.npz")
    potential = _source_potential(case, head)
    prediction = energy_and_forces(
        potential, _structure("crystal"), _periodic_execution(), compute_stress=True
    )
    assert bool(prediction.valid[0]) and prediction.stress is not None
    np.testing.assert_allclose(
        prediction.energy, fixture[f"crystal/{head}/energy"], rtol=1e-12, atol=1e-11
    )
    np.testing.assert_allclose(
        prediction.forces[0], fixture[f"crystal/{head}/forces"], rtol=1e-10, atol=1e-11
    )
    # Native stress is tensile-positive dE/d(strain) / V; the provider's virial
    # is -dE/d(strain).
    volume = abs(np.linalg.det(np.asarray(MANIFEST["structures"]["crystal"]["cell"])))
    np.testing.assert_allclose(
        np.asarray(prediction.stress[0]) * volume,
        -fixture[f"crystal/{head}/virials"][0],
        rtol=1e-10,
        atol=1e-11,
    )


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
def test_ordered_periodic_derivatives_match_provider_and_replay_exactly(
    accumulation: RelationAccumulation,
) -> None:
    case, head = "two_scale_shift", "Default"
    fixture = np.load(FIXTURES / f"{case}.npz")
    potential = _source_potential(case, head)
    batch = AtomisticBatch.from_structure(_structure("crystal"))

    def compiled(execution: AtomisticGraphExecutionPlan) -> Any:
        payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=3.0).prepare(
            batch
        )

        def evaluate(positions: jax.Array, cells: jax.Array) -> tuple[jax.Array, ...]:
            result = atomistic_energy_derivatives(
                potential,
                payload,
                positions,
                cell_vectors=cells,
                compute_stress=True,
            )
            assert result.forces is not None and result.stress is not None
            return result.energy, result.forces, result.stress, result.successful

        return jax.jit(evaluate)

    ordered = compiled(_periodic_execution(accumulation=accumulation))
    expected = ordered(batch.positions, batch.cells)
    energy, forces, stress, successful = expected
    assert bool(jnp.all(successful))
    np.testing.assert_allclose(
        energy, fixture[f"crystal/{head}/energy"], rtol=1e-12, atol=1e-11
    )
    np.testing.assert_allclose(
        forces[0], fixture[f"crystal/{head}/forces"], rtol=1e-10, atol=1e-11
    )
    volume = abs(np.linalg.det(np.asarray(MANIFEST["structures"]["crystal"]["cell"])))
    np.testing.assert_allclose(
        np.asarray(stress[0]) * volume,
        -fixture[f"crystal/{head}/virials"][0],
        rtol=1e-10,
        atol=1e-11,
    )
    # Every position, cell and case reduction follows the declared order, so a
    # repeated evaluation is bitwise identical on any backend.
    for _ in range(5):
        for value, reference in zip(
            ordered(batch.positions, batch.cells), expected, strict=True
        ):
            np.testing.assert_array_equal(value, reference)
    # Sequenced route gathers keep forward-mode derivatives of the forces.
    direction = jnp.where(
        batch.atom_mask[:, :, None],
        jnp.asarray(np.random.default_rng(13).normal(size=batch.positions.shape)),
        0.0,
    )

    def force_tangent(evaluate: Any) -> jax.Array:
        return jax.jvp(
            lambda positions: evaluate(positions, batch.cells)[1],
            (batch.positions,),
            (direction,),
        )[1]

    _require_close_at_scale(
        force_tangent(ordered),
        force_tangent(compiled(_periodic_execution())),
        1e-10,
    )


def test_source_parameter_directions_match_provider_gradients() -> None:
    case, head = "three_density", "Default"
    fixture = np.load(FIXTURES / f"{case}.npz")
    prefix = f"molecule/{head}/parameter_gradient/"
    rng = np.random.default_rng(11)
    directions = {
        name[len(prefix) :]: rng.normal(size=fixture[name].shape)
        for name in fixture.files
        if name.startswith(prefix)
    }
    expected = sum(
        float(np.sum(fixture[prefix + name] * direction))
        for name, direction in directions.items()
    )
    base = _source_potential(case, head)
    shifted = _source_potential(case, head, directions)
    parameters, state, fixed = partition_parameters(base)
    # The source map is linear in every trainable tensor, so the native image of
    # a source direction is the difference of the reconstructed PARAMETER lanes;
    # fixed lanes (U, CG, E0, radial frequencies) must not move with it.
    tangent = jax.tree_util.tree_map(
        lambda after, before: after - before, partition_parameters(shifted)[0], parameters
    )
    for after, before in zip(
        jax.tree_util.tree_leaves(partition_parameters(shifted)[2]),
        jax.tree_util.tree_leaves(fixed),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    structure = _structure("molecule")
    execution = _molecule_execution()

    def energy(lane: Any) -> Any:
        model = combine_parameters(lane, state, fixed)
        return model.energy(AtomisticBatch.from_structure(structure), execution)[0]

    _, observed = jax.jvp(energy, (parameters,), (tangent,))
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def _native(**overrides: Any) -> MACEPotential:
    declaration: dict[str, Any] = {
        "species": (1, 8),
        "cutoff": 3.0,
        "radial_basis_count": 4,
        "cutoff_power": 5,
        "channel_count": 3,
        "hidden_degree": 2,
        "edge_degree": 3,
        "interactions": (
            "real-agnostic-density",
            "real-agnostic-residual",
            "real-agnostic-density-residual",
        ),
        "correlations": (3, 2, 2),
        "radial_widths": (6,),
        "readout_width": 4,
        "average_neighbor_count": 2.0,
        "pair_repulsion": True,
        "energy_scaling": "scale-shift",
    }
    energies = overrides.pop("energies", {})
    precision = overrides.pop("precision", None)
    declaration.update(overrides)
    return MACEPotential(
        SCALE,
        MACEArchitecture(**declaration),
        atomic_energies=np.asarray(
            energies.get("atomic_energies", [[-1.0, -2.0]]), dtype=np.float64
        ),
        energy_scale=np.asarray(energies.get("energy_scale", [1.3]), dtype=np.float64),
        energy_shift=np.asarray(energies.get("energy_shift", [0.2]), dtype=np.float64),
        key=jr.key(5),
        precision=precision,
    )


@pytest.mark.parametrize("construction", ["native", "native-readout", "source"])
def test_float32_coupling_bases_execute_and_persist_their_stored_representation(
    construction: str, tmp_path: Path
) -> None:
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float32",
        reduction_dtype="float32",
        output_dtype="float32",
    )
    if construction == "source":
        potential = _source_potential("two_scale_shift", "Default", precision=precision)
    elif construction == "native-readout":
        potential = _native(
            interactions=("real-agnostic",),
            correlations=(2,),
            edge_degree=1,
            hidden_degree=1,
            readout_correlation=2,
            precision=precision,
        )
    else:
        potential = _native(precision=precision)
    potential.validate()
    plans = [layer.product.contraction.plan for layer in potential.layers]
    plans.extend(
        layer.readout_product.contraction.plan
        for layer in potential.layers
        if layer.readout_product is not None
    )
    assert all(
        basis.coefficients.dtype == np.dtype(np.float32)
        and basis._host_projection().dtype == np.dtype(np.float32)
        for plan in plans
        for block in plan.bases
        for basis in block
    )
    path = tmp_path / f"{construction}.phydrax"
    write_atomistic_model_artifact(path, potential)
    restored = read_atomistic_model_artifact(path).model
    restored.validate()
    assert restored.architecture_id == potential.architecture_id
    original_leaves, original_tree = jax.tree_util.tree_flatten(potential)
    restored_leaves, restored_tree = jax.tree_util.tree_flatten(restored)
    assert original_tree == restored_tree
    for original, current in zip(original_leaves, restored_leaves, strict=True):
        if eqx.is_array(original):
            assert eqx.is_array(current) and original.dtype == current.dtype
            np.testing.assert_array_equal(original, current)
        else:
            assert original == current
    source = _structure("molecule")
    structure = AtomicStructure(
        source.atomic_numbers,
        np.asarray(source.positions, dtype=np.float32),
        np.asarray(source.masses, dtype=np.float32),
        SCALE,
    )
    before = energy_and_forces(potential, structure, _molecule_execution())
    after = energy_and_forces(restored, structure, _molecule_execution())
    assert bool(before.valid[0]) and bool(after.valid[0])
    assert before.energy.dtype == np.dtype(np.float32)
    assert before.forces.dtype == np.dtype(np.float32)
    assert np.all(np.isfinite(before.energy)) and np.all(np.isfinite(before.forces))
    np.testing.assert_allclose(after.energy, before.energy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(after.forces, before.forces, rtol=1e-6, atol=1e-6)
    if construction == "source":
        fixture = np.load(FIXTURES / "two_scale_shift.npz")
        for name, value in (("energy", before.energy), ("forces", before.forces[0])):
            absolute, relative = FLOAT32_GATES[name]
            np.testing.assert_allclose(
                value,
                fixture[f"molecule/Default/{name}"],
                rtol=relative,
                atol=absolute,
            )


@pytest.mark.parametrize(
    "orthogonal",
    [
        pytest.param([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], id="rotation"),
        pytest.param(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], id="reflection"
        ),
        pytest.param(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], id="inversion"
        ),
    ],
)
def test_degree_two_three_layer_model_is_o3_invariant(orthogonal: Any) -> None:
    potential = _native()
    transform = jnp.asarray(orthogonal)
    structure = _structure("molecule")
    reference = energy_and_forces(potential, structure, _molecule_execution())
    moved = _structure(
        "molecule",
        positions=np.asarray(structure.positions) @ np.asarray(transform).T + 0.4,
    )
    observed = energy_and_forces(potential, moved, _molecule_execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0] @ transform.T, rtol=1e-9, atol=1e-10
    )


def test_atom_permutation_and_padding_preserve_energy_and_route_forces() -> None:
    potential = _native()
    structure = _structure("molecule")
    reference = energy_and_forces(potential, structure, _molecule_execution())
    order = np.asarray([2, 0, 1], dtype=np.int64)
    permuted = AtomicStructure(
        np.asarray(structure.atomic_numbers)[order],
        np.asarray(structure.positions)[order],
        np.asarray(structure.masses)[order],
        SCALE,
    )
    observed = energy_and_forces(potential, permuted, _molecule_execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        observed.forces[0], np.asarray(reference.forces[0])[order], atol=1e-11
    )
    padded = AtomicStructure(
        np.concatenate(
            (np.asarray(structure.atomic_numbers), np.zeros((1,), dtype=np.int32))
        ),
        np.concatenate(
            (
                np.asarray(structure.positions),
                np.asarray([[0.3, 0.2, 0.1]], dtype=np.float64),
            )
        ),
        np.concatenate(
            (np.asarray(structure.masses), np.asarray([15.999], dtype=np.float64))
        ),
        SCALE,
        active_mask=np.asarray([True, True, True, False], dtype=np.bool_),
    )
    observed = energy_and_forces(potential, padded, _molecule_execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed.forces[0][:3], reference.forces[0], atol=1e-11)
    np.testing.assert_array_equal(np.asarray(observed.forces[0][3]), np.zeros(3))


def test_reference_energy_scale_and_shift_placement() -> None:
    structure = _structure("molecule")
    execution = _molecule_execution()
    base = energy_and_forces(_native(), structure, execution)
    reference = energy_and_forces(
        _native(energies={"atomic_energies": [[-0.5, -2.0]]}), structure, execution
    )
    shifted = energy_and_forces(
        _native(energies={"energy_shift": [0.7]}), structure, execution
    )
    scaled = energy_and_forces(
        _native(energies={"energy_scale": [2.6]}), structure, execution
    )
    # Two hydrogens move E0 by 2 * 0.5; E0 never enters forces.
    np.testing.assert_allclose(reference.energy - base.energy, 1.0, rtol=1e-12)
    np.testing.assert_allclose(reference.forces, base.forces, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(shifted.energy - base.energy, 3 * 0.5, rtol=1e-12)
    # Scale multiplies the ZBL-plus-readout interaction energy, not E0 or shift.
    interaction = base.energy - (-4.0) - 3 * 0.2
    np.testing.assert_allclose(scaled.energy - base.energy, interaction, rtol=1e-11)
    np.testing.assert_allclose(scaled.forces, 2.0 * base.forces, rtol=1e-11, atol=1e-12)


def test_unknown_species_fails_closed() -> None:
    potential = _native()
    structure = AtomicStructure(
        np.asarray([6, 1, 1], dtype=np.int32),
        _structure("molecule").positions,
        np.asarray([12.011, 1.008, 1.008], dtype=np.float64),
        SCALE,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside the MACE model species"):
        energy_and_forces(potential, structure, _molecule_execution())


def test_energy_is_smooth_across_the_cutoff() -> None:
    potential = _native()
    execution = _molecule_execution()

    def dimer(distance: float) -> Any:
        return energy_and_forces(
            potential,
            AtomicStructure(
                np.asarray([8, 1], dtype=np.int32),
                np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64),
                np.asarray([15.999, 1.008], dtype=np.float64),
                SCALE,
            ),
            execution,
        )

    inside, outside = dimer(3.0 - 1e-4), dimer(3.0 + 1e-4)
    # The p = 5 polynomial envelope is C2 at the cutoff: the energy gap and the
    # force across the join vanish at least cubically and quadratically.
    assert abs(float(inside.energy[0] - outside.energy[0])) < 1e-9
    assert float(jnp.max(jnp.abs(inside.forces))) < 1e-6
    np.testing.assert_array_equal(np.asarray(outside.forces), 0.0)


def test_parameter_lane_holds_source_trainables_and_fixes_buffers() -> None:
    potential = _native()
    parameters, _, fixed = partition_parameters(potential)
    parameter_paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(parameters)[0]
    }
    fixed_paths = {
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(fixed)[0]
        if eqx.is_inexact_array(leaf)
    }
    assert any(".contraction.weights" in path for path in parameter_paths)
    assert any(".radial.weights" in path for path in parameter_paths)
    assert any(".self_connection.weights" in path for path in parameter_paths)
    assert ".embedding" in parameter_paths
    for buffer in (
        ".energy_reference.atomic_energies",
        ".energy_reference.scale",
        ".energy_reference.shift",
        ".geometry.radial.basis.frequencies",
    ):
        assert buffer in fixed_paths and buffer not in parameter_paths
    assert not any("prepared.coefficients" in path for path in parameter_paths)
    assert not any(".plan.bases" in path for path in parameter_paths)
    assert not any("pair_repulsion" in path for path in parameter_paths)


def test_restored_value_with_corrupt_parameters_or_identity_refuses() -> None:
    potential = _native()
    potential.validate()
    poisoned = eqx.tree_at(
        lambda model: model.embedding,
        potential,
        potential.embedding.at[0, 0].set(jnp.nan),
    )
    with pytest.raises(ValueError, match="finite"):
        poisoned.validate()
    shifted = eqx.tree_at(
        lambda model: model.energy_reference.shift,
        potential,
        potential.energy_reference.shift + 1.0,
    )
    with pytest.raises(ValueError, match="identity"):
        shifted.validate()


@pytest.fixture(scope="module")
def zbl_agnesi_source() -> MACEPotential:
    return _source_potential("two_scale_shift", "Default")


@pytest.mark.parametrize(
    ("where", "change"),
    [
        pytest.param(
            lambda m: m.geometry.radial.basis.prefactor,
            lambda v: v * 2.0,
            id="bessel-prefactor",
        ),
        pytest.param(
            lambda m: m.geometry.radial.basis.frequencies,
            lambda v: v * 2.0,
            id="bessel-frequencies",
        ),
        pytest.param(
            lambda m: m.geometry.radial.transform.covalent_radii,
            lambda v: v * 2.0,
            id="agnesi-radii",
        ),
        pytest.param(
            lambda m: m.layers[0].pair_repulsion.exponents,
            lambda v: v * 2.0,
            id="zbl-exponents",
        ),
        pytest.param(
            lambda m: m.layers[0].pair_repulsion.atomic_numbers,
            lambda v: v[::-1],
            id="zbl-atomic-numbers",
        ),
    ],
)
def test_restored_fixed_radial_or_zbl_field_refuses_under_its_recorded_identity(
    zbl_agnesi_source: MACEPotential, where: Any, change: Any
) -> None:
    # A constructor-bypassing restore keeps every recorded identity while the
    # changed fixed array executes (and changes the energy), so validation must
    # rebuild each fixed field from what executes rather than trust its label.
    zbl_agnesi_source.validate()
    mutated = eqx.tree_at(where, zbl_agnesi_source, change(where(zbl_agnesi_source)))
    assert mutated.architecture_id == zbl_agnesi_source.architecture_id
    with pytest.raises(ValueError, match="differ"):
        mutated.validate()


FLOAT32 = AtomisticPrecisionPolicy(
    coordinate_dtype="float32",
    compute_dtype="float32",
    reduction_dtype="float64",
    output_dtype="float64",
)


def _float32_source(
    case: str, head: str, precision: AtomisticPrecisionPolicy
) -> MACEPotential:
    """Reconstruct from float32-stored source tensors, as legacy checkpoints store them."""
    fixture = np.load(FIXTURES / f"{case}.npz")
    transforms, couplings = _source_basis(fixture)
    tensors = {
        name: value.astype(np.float32) if value.dtype == np.float64 else value
        for name, value in _source_tensors(fixture).items()
    }
    return mace_potential_from_source(
        SCALE,
        _source_architecture(case, head),
        tensors,
        degree_transforms=transforms,
        couplings={
            key: np.asarray(value, dtype=np.float32) for key, value in couplings.items()
        },
        source_id=f"mace-torch-0.3.16-fixture-float32:{case}",
        precision=precision,
    )


@pytest.mark.strict_jax
def test_float32_source_and_native_models_construct_and_track_float64() -> None:
    native = MACEPotential(
        SCALE,
        _native().configuration,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3], dtype=np.float64),
        energy_shift=np.asarray([0.2], dtype=np.float64),
        precision=FLOAT32,
        key=jr.key(5),
    )
    native.validate()
    case, head = "two_scale_shift", "Default"
    single = _float32_source(case, head, FLOAT32)
    double = _float32_source(case, head, AtomisticPrecisionPolicy())
    single.validate()
    double.validate()
    reference = _structure("molecule")
    structure = AtomicStructure(
        reference.atomic_numbers,
        np.asarray(reference.positions),
        reference.masses,
        SCALE,
        coordinate_dtype="float32",
    )
    execution = _molecule_execution()
    observed = energy_and_forces(single, structure, execution)
    expected = energy_and_forces(double, reference, execution)
    assert bool(observed.valid[0])
    # The total cancels O(1) eV reference and interaction terms, so the float32
    # gate is absolute at a few float32 ulps of those terms.
    np.testing.assert_allclose(observed.energy, expected.energy, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(observed.forces, expected.forces, rtol=1e-3, atol=1e-4)
    # The float32-stored source realization stays the provider's function.
    fixture = np.load(FIXTURES / f"{case}.npz")
    np.testing.assert_allclose(
        observed.energy, fixture[f"molecule/{head}/energy"], rtol=1e-4
    )

    def force_loss(model: MACEPotential, batch: AtomisticBatch) -> Any:
        parameters, state, fixed = partition_parameters(model)

        def loss(lane: Any) -> Any:
            rebuilt = combine_parameters(lane, state, fixed)
            forces = jax.grad(lambda x: rebuilt.energy(batch, execution, positions=x)[0])(
                batch.positions
            )
            return jnp.sum(forces.astype(jnp.float64) ** 2)

        return jax.grad(loss)(parameters)

    single_gradient = force_loss(single, AtomisticBatch.from_structure(structure))
    double_gradient = force_loss(double, AtomisticBatch.from_structure(reference))
    for left, right in zip(
        jax.tree_util.tree_leaves(single_gradient),
        jax.tree_util.tree_leaves(double_gradient),
        strict=True,
    ):
        assert left.dtype == jnp.float32
        scale = max(1.0, float(jnp.max(jnp.abs(right))))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=2e-3 * scale)


PERIODIC_FLOAT32 = AtomisticPrecisionPolicy(
    coordinate_dtype="float32",
    compute_dtype="float32",
    reduction_dtype="float64",
    output_dtype="float32",
)


@pytest.mark.strict_jax
def test_float32_periodic_source_energy_forces_stress_track_float64() -> None:
    case, head = "two_scale_shift", "Default"
    single = _float32_source(case, head, PERIODIC_FLOAT32)
    double = _float32_source(case, head, AtomisticPrecisionPolicy())
    reference = _structure("crystal")
    structure = AtomicStructure(
        reference.atomic_numbers,
        np.asarray(reference.positions, dtype=np.float64),
        reference.masses,
        SCALE,
        cell=np.asarray(MANIFEST["structures"]["crystal"]["cell"], dtype=np.float64),
        periodic_axes=np.ones((3,), dtype=np.bool_),
        coordinate_dtype="float32",
    )
    # The float64 reference evaluates the same float32-rounded coordinates and
    # cell, so the comparison isolates float32 model arithmetic from input
    # rounding (which alone moves this energy by about 2e-4 eV).
    rounded = AtomicStructure(
        reference.atomic_numbers,
        np.asarray(structure.positions, dtype=np.float64),
        reference.masses,
        SCALE,
        cell=np.asarray(structure.cell, dtype=np.float64),
        periodic_axes=np.ones((3,), dtype=np.bool_),
    )
    # Ordered accumulation fixes every accelerator addition order, so the two
    # routes below can agree at float32 output rounding; fast accumulation may
    # reorder GPU additions between otherwise identical evaluations.
    execution = _periodic_execution(accumulation="deterministic")
    observed = energy_and_forces(single, structure, execution, compute_stress=True)
    expected = energy_and_forces(double, rounded, execution, compute_stress=True)
    assert bool(observed.valid[0])
    assert observed.stress is not None and expected.stress is not None
    assert observed.forces.dtype == jnp.float32
    assert observed.stress.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(observed.forces)))
    assert bool(jnp.all(jnp.isfinite(observed.stress)))
    # This gate is native float32 arithmetic tracking the native float64 model on
    # identical rounded inputs, not the source-fidelity gate of imported
    # checkpoints. The randomly perturbed fixture model has per-layer atom
    # energies of order 1e2 eV, forces of order 5e2 eV/A and stresses of order
    # 3e2 in the compact crystal. The provider's own float32 evaluation of the
    # same float32-rounded W/U and coordinates deviates from float64 by 2.6e-3
    # eV/atom and 9.4e-3 eV/A (about 2e-5 relative), the same scale as native
    # float32 here, so the gap is conditioning of this model rather than a native
    # route error. Each output is gated at 1e-4 of its own reference magnitude; a
    # dropped image, cell or sign error changes these outputs at order one.
    _require_close_at_scale(observed.energy, expected.energy, 1e-4)
    _require_close_at_scale(observed.forces, expected.forces, 1e-4)
    _require_close_at_scale(observed.stress, expected.stress, 1e-4)
    # The numerical helper canonicalizes a float64 cell override once into the
    # float32 coordinate contract before the image matmul.
    batch = AtomisticBatch.from_structure(structure)
    payload = AtomisticGraphLayout.for_batch(batch, execution, cutoff=3.0).prepare(batch)
    derivatives = atomistic_energy_derivatives(
        single,
        payload,
        batch.positions,
        cell_vectors=np.asarray(batch.cells, dtype=np.float64),
        compute_stress=True,
    )
    assert bool(jnp.all(derivatives.successful))
    assert derivatives.forces is not None and derivatives.stress is not None
    assert bool(jnp.all(jnp.isfinite(derivatives.forces)))
    assert bool(jnp.all(jnp.isfinite(derivatives.stress)))
    # Same float32 model and inputs: only the route into the same derivatives
    # differs, so agreement is at float32 output rounding of each magnitude.
    _require_close_at_scale(derivatives.energy, observed.energy, 1e-6)
    _require_close_at_scale(derivatives.forces, observed.forces, 1e-6)
    _require_close_at_scale(derivatives.stress, observed.stress, 1e-6)


def _require_close_at_scale(observed: Any, expected: Any, relative: float) -> None:
    """Elementwise agreement within ``relative`` times the reference magnitude."""
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected_)))
    np.testing.assert_allclose(observed_, expected_, rtol=0.0, atol=relative * scale)


def test_default_declaration_is_the_bessel_o3_family_without_conditioning() -> None:
    potential = _native()
    architecture = potential.configuration
    assert architecture.radial_basis == "bessel"
    assert architecture.distance_transform == "none"
    assert architecture.soft_alpha is None
    assert architecture.parity == "o3"
    assert architecture.edge_channel_count is None
    assert architecture.edge_feature_degree is None
    assert not architecture.edge_irreps_first
    assert not architecture.keep_last_layer_irreps
    assert architecture.readout_variant == "nonlinear"
    assert architecture.gate_scales is None
    assert architecture.embedding_features == ()
    assert not architecture.embedding_readout
    assert potential.conditioning.fields == ()
    assert potential.joint_embedding is None and potential.embedding_readout is None
    for index in range(architecture.interaction_count):
        assert (
            architecture.edge_layout(index).layout_id
            == architecture.input_layout(index).layout_id
        )


def test_node_states_carry_every_layer_state_and_match_node_energies() -> None:
    potential = _native()
    architecture = potential.configuration
    batch = AtomisticBatch.from_structure(_structure("molecule"))
    graph = (
        AtomisticGraphLayout.for_batch(batch, _molecule_execution(), cutoff=3.0)
        .prepare(batch)
        .graph(cutoff=3.0)
    )
    inputs = AtomisticGraphInputs(
        graph, SCALE, species_kind=potential.capabilities.species_kind
    )
    species, states, relation = potential.graph_node_states(inputs)
    ir = graph.graph
    assert ir.edge_mask is not None
    energies, successful = potential.node_energies(
        species,
        inputs.atom_mask,
        relation,
        jnp.asarray(ir.edges["displacement"]),
        edge_active=jnp.asarray(ir.edge_mask, dtype=jnp.bool_),
    )
    assert bool(successful) and bool(states.successful)
    np.testing.assert_array_equal(np.asarray(energies), np.asarray(states.atom_energy))
    assert len(states.states) == architecture.interaction_count
    for index, state in enumerate(states.states):
        assert state.shape == (
            batch.atom_capacity,
            architecture.output_layout(index).packed_size,
        )
    result = potential.evaluate_graph(inputs)
    assert result.atom_energy_complete
    np.testing.assert_allclose(
        result.energy[0], jnp.sum(states.atom_energy), rtol=1e-13, atol=1e-13
    )
