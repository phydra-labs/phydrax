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
    atomistic_potential_revision,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
    prepare_atomistic_graph_topology,
)
from phydrax.atomistic._graph import realize_atomistic_graph
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_kernels import MACEAcceleratedCoupling, MACEKernelPlan
from phydrax.nn.atomistic._mace_prepare import (
    prepare_mace_potential,
    StaleMACEPreparation,
)
from phydrax.nn.atomistic._radial_projection import RadialTableDeclaration
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
EXECUTION = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=5)
NUMBERS = [8, 1, 1, 8, 1]
# Atom 0 receives four edges, more than one two-event fragment.
POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.95, 0.1, 0.0],
        [-0.25, 0.85, 0.2],
        [0.3, -0.9, 0.6],
        [-0.7, -0.3, -0.8],
    ],
    dtype=np.float64,
)


def _potential(*, streaming: StreamedRelationPlan | None = None) -> MACEPotential:
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=2.5,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=3,
            hidden_degree=1,
            edge_degree=2,
            interactions=(
                "real-agnostic-density",
                "real-agnostic-residual",
                "real-agnostic-density-residual",
            ),
            correlations=(3, 2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=3.0,
            pair_repulsion=True,
            distance_transform="agnesi",
            agnesi_parameters=(1.0805, 0.9183, 4.5791),
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        streaming=streaming,
        key=jr.key(9),
    )


def _batch() -> AtomisticBatch:
    numbers = np.asarray(NUMBERS, dtype=np.int32)
    masses = np.where(numbers == 8, 15.999, 1.008).astype(np.float64)
    return AtomisticBatch.from_structure(
        AtomicStructure(numbers, POSITIONS, masses, SCALE)
    )


def _energy(model: Any, positions: Any) -> Any:
    batch = _batch()
    graph = realize_atomistic_graph(batch, EXECUTION, cutoff=2.5, positions=positions)
    energy, _ = model.graph_energy(
        batch.atomic_numbers,
        batch.atom_mask,
        batch.atom_cases,
        batch.case_count,
        batch.atom_capacity,
        graph,
    )
    return energy[0]


def _positions() -> Any:
    return jnp.asarray(POSITIONS)[None]


def test_fragmented_streaming_matches_single_tile_derivatives() -> None:
    whole = _potential()
    fragmented = _potential(
        streaming=StreamedRelationPlan(receiver_tile=1, edge_tile=2, channel_capacity=512)
    )
    positions = _positions()
    tangent = jnp.asarray(np.random.default_rng(3).normal(size=positions.shape))
    for model in (whole, fragmented):
        assert np.isfinite(float(_energy(model, positions)))
    np.testing.assert_allclose(
        _energy(fragmented, positions), _energy(whole, positions), rtol=1e-12, atol=1e-12
    )
    gradient = jax.grad(_energy, argnums=1)
    np.testing.assert_allclose(
        gradient(fragmented, positions),
        gradient(whole, positions),
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        jax.jvp(lambda x: _energy(fragmented, x), (positions,), (tangent,))[1],
        jax.jvp(lambda x: _energy(whole, x), (positions,), (tangent,))[1],
        rtol=1e-10,
        atol=1e-11,
    )

    def force_loss(lane: Any, model: MACEPotential) -> Any:
        _, state, fixed = partition_parameters(model)
        rebuilt = combine_parameters(lane, state, fixed)
        forces = -jax.grad(_energy, argnums=1)(rebuilt, positions)
        return jnp.sum(forces * forces)

    observed = jax.grad(force_loss)(partition_parameters(fragmented)[0], fragmented)
    expected = jax.grad(force_loss)(partition_parameters(whole)[0], whole)
    for left, right in zip(
        jax.tree_util.tree_leaves(observed),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)


def test_prepared_folded_realization_matches_exact_energy_and_forces() -> None:
    model = _potential()
    prepared = prepare_mace_potential(model)
    prepared.validate()
    positions = _positions()
    np.testing.assert_allclose(
        _energy(prepared, positions), _energy(model, positions), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.grad(_energy, argnums=1)(prepared, positions),
        jax.grad(_energy, argnums=1)(model, positions),
        rtol=1e-10,
        atol=1e-11,
    )


def test_prepared_realization_refuses_parameter_mutation() -> None:
    model = _potential()
    prepared = prepare_mace_potential(model)
    mutated = eqx.tree_at(
        lambda value: value.model.embedding, prepared, prepared.model.embedding + 0.1
    )
    with pytest.raises(StaleMACEPreparation):
        mutated.validate()
    updated = eqx.tree_at(lambda value: value.embedding, model, model.embedding + 0.1)
    assert prepare_mace_potential(updated).prepared_id != prepared.prepared_id


def test_tabulated_radial_realization_is_close_and_refuses_unsupported_radii() -> None:
    model = _potential()
    tabulated = prepare_mace_potential(
        model,
        radial="tabulated",
        tables=RadialTableDeclaration(0.3, 2048, layout="projected-width"),
    )
    tabulated.validate()
    positions = _positions()
    np.testing.assert_allclose(
        _energy(tabulated, positions), _energy(model, positions), rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        jax.grad(_energy, argnums=1)(tabulated, positions),
        jax.grad(_energy, argnums=1)(model, positions),
        rtol=1e-5,
        atol=1e-6,
    )
    collapsed = positions.at[0, 1].set(positions[0, 0] + jnp.asarray([0.1, 0.0, 0.0]))
    # Inside the streamed scan the equinox check surfaces through JAX's runtime.
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="outside the declared support",
    ):
        jax.block_until_ready(_energy(tabulated, collapsed))


def test_tabulated_publication_refuses_a_stale_fixed_radial_field() -> None:
    tabulated = prepare_mace_potential(
        _potential(),
        radial="tabulated",
        tables=RadialTableDeclaration(0.3, 256, layout="projected-width"),
    )
    tabulated.validate()
    # The prefactor changes the exact model and every table it binds, while
    # every recorded identity (model, embedding, tables, preparation) is kept.
    mutated = eqx.tree_at(
        lambda value: value.model.geometry.radial.basis.prefactor,
        tabulated,
        tabulated.model.geometry.radial.basis.prefactor * 2.0,
    )
    with pytest.raises(ValueError, match="Radial embedding"):
        mutated.validate()


def test_atom_type_id_zero_tabulated_realization_matches_exact() -> None:
    model = MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(0, 2),
            species_kind=AtomisticSpeciesKind.ATOM_TYPE_ID,
            cutoff=2.5,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=3,
            hidden_degree=1,
            edge_degree=2,
            interactions=("real-agnostic-density", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=3.0,
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jr.key(9),
    )
    tabulated = prepare_mace_potential(
        model,
        radial="tabulated",
        tables=RadialTableDeclaration(0.3, 2048, layout="projected-width"),
    )
    tabulated.validate()
    batch = _batch()
    type_ids = jnp.where(batch.atomic_numbers == 8, 0, 2)

    def energy(potential: Any, positions: Any) -> Any:
        graph = realize_atomistic_graph(batch, EXECUTION, cutoff=2.5, positions=positions)
        value, _ = potential.graph_energy(
            type_ids,
            batch.atom_mask,
            batch.atom_cases,
            batch.case_count,
            batch.atom_capacity,
            graph,
        )
        return value[0]

    positions = _positions()
    np.testing.assert_allclose(
        energy(tabulated, positions), energy(model, positions), rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        jax.grad(energy, argnums=1)(tabulated, positions),
        jax.grad(energy, argnums=1)(model, positions),
        rtol=1e-5,
        atol=1e-6,
    )


def test_prepared_active_species_bound_refuses_other_species() -> None:
    prepared = prepare_mace_potential(_potential(), active_species=(1,))
    with pytest.raises(eqx.EquinoxRuntimeError, match="prepared MACE active species"):
        jax.block_until_ready(_energy(prepared, _positions()))


def test_streamed_payload_beyond_channel_capacity_refuses() -> None:
    model = _potential(
        streaming=StreamedRelationPlan(receiver_tile=4, edge_tile=8, channel_capacity=8)
    )
    with pytest.raises(ValueError, match="channel elements"):
        _energy(model, _positions())


# Accelerated fragment execution ----------------------------------------------
# The kernels run on the Mosaic GPU interpreter: kernel semantics and model
# integration, not GPU qualification.

CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
DETERMINISTIC = StreamedRelationPlan(
    receiver_tile=2, edge_tile=8, channel_capacity=2048, accumulation="deterministic"
)


def _coupling() -> MACEAcceleratedCoupling:
    return MACEAcceleratedCoupling(
        MACEKernelPlan(
            target="cpu_interpret",
            precision="float64",
            receiver_tile=2,
            edge_tile=4,
            channel_tile=128,
            reduction_programs=2,
            fragment_budget_bytes=1 << 26,
        )
    )


def _periodic_execution(edges: int = 512) -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=edges,
            maximum_degree=64,
            maximum_images=125,
        ),
        streamed=DETERMINISTIC,
    )


def _periodic_batch(scale: float = 1.0) -> AtomisticBatch:
    # Three atoms and one masked slot; every receiver has more periodic routes
    # than one eight-lane fragment. Shrinking ``scale`` densifies the crystal.
    return AtomisticBatch(
        np.asarray([[1, 8, 1, 0]], dtype=np.int32),
        scale
        * np.asarray(
            [[[0.0, 0.0, 0.0], [1.1, 0.3, 0.2], [0.4, 1.6, 1.3], [0.0, 0.0, 0.0]]]
        ),
        np.asarray([[1.008, 15.999, 1.008, 1.008]]),
        SCALE,
        atom_mask=np.asarray([[True, True, True, False]]),
        cells=scale * CELL[None],
        periodic_axes=np.ones((1, 3), dtype=np.bool_),
    )


_VARIANTS = {
    "density-residual-zbl-scale-shift": {
        "interactions": ("real-agnostic-density", "real-agnostic-density-residual"),
        "correlations": (3, 2),
        "readout_width": 4,
        "pair_repulsion": True,
        "distance_transform": "agnesi",
        "agnesi_parameters": (1.0805, 0.9183, 4.5791),
        "energy_scaling": "scale-shift",
    },
    "one-layer-invariant-readout": {
        "interactions": ("real-agnostic-residual",),
        "correlations": (2,),
        "readout_correlation": 2,
        "readout_width": 4,
    },
    "multihead-cutoff-on-weights": {
        "interactions": ("real-agnostic", "real-agnostic-residual"),
        "correlations": (2, 2),
        "readout_width": 4,
        "heads": ("first", "second"),
        "head": "second",
        "cutoff_placement": "weights",
    },
}


def _variant(name: str, *, channels: int = 3) -> MACEPotential:
    declaration: dict[str, Any] = {
        "species": (1, 8),
        "cutoff": 3.0,
        "radial_basis_count": 4,
        "cutoff_power": 5,
        "channel_count": channels,
        "hidden_degree": 1,
        "edge_degree": 2,
        "radial_widths": (6,),
        "average_neighbor_count": 8.0,
        **_VARIANTS[name],
    }
    architecture = MACEArchitecture(**declaration)
    heads = len(architecture.heads)
    scaled = architecture.energy_scaling == "scale-shift"
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]] * heads, dtype=np.float64),
        energy_scale=np.asarray([1.3] * heads, dtype=np.float64) if scaled else None,
        energy_shift=np.asarray([0.2] * heads, dtype=np.float64) if scaled else None,
        key=jr.key(13),
    )


def _efs(model: MACEPotential, topology: Any, positions: Any) -> tuple[Any, Any, Any]:
    derivatives = atomistic_energy_derivatives(
        model,
        _periodic_batch(),
        _periodic_execution(),
        positions,
        topology=topology,
        compute_stress=True,
    )
    return derivatives.energy, derivatives.forces, derivatives.stress


@pytest.mark.parametrize("name", list(_VARIANTS))
def test_accelerated_fragments_match_exact_energy_forces_stress_and_training_gradient(
    name: str,
) -> None:
    exact = _variant(name)
    accelerated = exact.with_acceleration(_coupling())
    # An execution policy: identical scientific identity and parameter revision.
    assert accelerated.architecture_id == exact.architecture_id
    assert (
        atomistic_potential_revision(accelerated).revision_id
        == atomistic_potential_revision(exact).revision_id
    )
    batch = _periodic_batch()
    topology = prepare_atomistic_graph_topology(batch, _periodic_execution(), cutoff=3.0)
    assert int(topology.streamed.schedule.fragmented_receivers) > 0
    positions = batch.positions

    observed = _efs(accelerated, topology, positions)
    expected = _efs(exact, topology, positions)
    for left, right in zip(observed, expected, strict=True):
        assert bool(jnp.all(jnp.isfinite(left)))
        np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(observed[1][0, 3], 0.0)

    def supervision(lane: Any, model: MACEPotential) -> Any:
        _, state, fixed = partition_parameters(model)
        energy, forces, stress = _efs(
            combine_parameters(lane, state, fixed), topology, positions
        )
        return energy[0] ** 2 + jnp.sum(forces * forces) + 10.0 * jnp.sum(stress * stress)

    gradient = jax.grad(supervision)(partition_parameters(accelerated)[0], accelerated)
    reference = jax.grad(supervision)(partition_parameters(exact)[0], exact)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert leaves and any(float(jnp.max(jnp.abs(leaf))) > 0.0 for leaf in leaves)
    for left, right in zip(leaves, jax.tree_util.tree_leaves(reference), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)


def test_accelerated_raw_and_prepared_fragments_match_exact_on_a_split_receiver() -> None:
    streaming = StreamedRelationPlan(
        receiver_tile=1, edge_tile=2, channel_capacity=512, accumulation="deterministic"
    )
    exact = _potential(streaming=streaming)
    accelerated = exact.with_acceleration(_coupling())
    positions = _positions()
    gradient = jax.grad(_energy, argnums=1)
    np.testing.assert_allclose(
        _energy(accelerated, positions), _energy(exact, positions), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient(accelerated, positions),
        gradient(exact, positions),
        rtol=1e-10,
        atol=1e-11,
    )
    prepared = prepare_mace_potential(exact, edge_coupling=_coupling())
    prepared.validate()
    np.testing.assert_allclose(
        _energy(prepared, positions), _energy(exact, positions), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient(prepared, positions), gradient(exact, positions), rtol=1e-10, atol=1e-11
    )
