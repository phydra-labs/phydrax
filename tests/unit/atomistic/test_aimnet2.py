from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    energy_and_forces,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditioning,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from phydrax.atomistic._graph import AtomisticGraph, AtomisticGraphLayout
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._aimnet2 import (
    AIMNET2_CHARGE_EPSILON,
    AIMNet2Architecture,
    AIMNet2FrozenAtomicShifts,
    AIMNet2OutputKind,
    AIMNet2OutputStage,
    AIMNet2Potential,
)
from phydrax.nn.atomistic._aimnet2_features import (
    AIMNet2ConvSV,
    AIMNet2RadialBasis,
)
from phydrax.nn.atomistic._aimnet2_source import (
    aimnet2_architecture_from_source,
    aimnet2_calculator_composition,
    aimnet2_potential_from_source,
    AIMNet2SourceRefusedError,
    externalize_source_long_range,
    SOURCE_COULOMB_CONSTANT,
)
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.aimnet2_source import (
    RC_S,
    reference_forward,
    source_inventory,
)


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
NUMBERS = np.asarray([8, 1, 1, 6])
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.3, 0.9, 0.1], [0.4, -0.7, 1.2]]
)
EXECUTION = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=8)
ALL = (
    AtomisticObservable.ENERGY,
    AtomisticObservable.CHARGES,
    AtomisticObservable.CHARGE_RESIDUAL,
    AtomisticObservable.NODE_EMBEDDINGS,
)


def _reconstruct(
    declaration: dict[str, Any],
    tensors: dict[str, Any],
    *,
    neighbor_cutoff: float = RC_S,
    **options: Any,
) -> Any:
    core, state, _ = externalize_source_long_range(declaration, tensors)
    return aimnet2_potential_from_source(
        SCALE, core, state, neighbor_cutoff=neighbor_cutoff, **options
    )


def _model(**options: Any) -> tuple[Any, dict, dict]:
    declaration, tensors = source_inventory(3, **options)
    channels = options.get("channels", 1)
    reconstruction = _reconstruct(
        declaration,
        tensors,
        neighbor_cutoff=options.get("aev_cutoff", RC_S),
        ts_pair_cutoff=6.0 if options.get("dispersion") else None,
    )
    assert reconstruction.architecture.charge_channels == channels
    return reconstruction.potential, declaration, tensors


def _structure(numbers: Any = NUMBERS, positions: Any = POSITIONS) -> AtomicStructure:
    return AtomicStructure(
        np.asarray(numbers), np.asarray(positions), np.ones(len(numbers)), SCALE
    )


def _conditioning(model: Any, batch: AtomisticBatch, **values: Any) -> Any:
    return AtomisticConditioning(
        model.conditioning_spec,
        {name: np.asarray(value, dtype=np.float64) for name, value in values.items()},
        case_count=batch.case_count,
        atom_capacity=batch.atom_capacity,
    )


def _graph(model: Any, batch: AtomisticBatch, positions: Any = None) -> AtomisticGraph:
    cutoff = model.configuration.cutoff
    payload = AtomisticGraphLayout.for_batch(batch, EXECUTION, cutoff=cutoff).prepare(
        batch
    )
    return payload.graph(cutoff=cutoff, positions=positions)


def _evaluate(
    model: Any,
    batch: AtomisticBatch,
    request: Any = ALL,
    positions: Any = None,
    **values: Any,
) -> Any:
    graph = _graph(model, batch, positions)
    inputs = AtomisticGraphInputs(
        graph,
        SCALE,
        species_kind=model.capabilities.species_kind,
        conditioning=_conditioning(model, batch, **values),
    )
    return model.evaluate_graph(
        inputs, None if request is None else AtomisticObservableRequest(request)
    )


def test_feature_equations_match_the_source_scalar_vector_aev() -> None:
    shifts = np.asarray([0.8, 1.4, 2.0, 2.6])
    basis = AIMNet2RadialBasis(shifts, np.asarray(2.5), np.asarray(3.2))
    distance = jnp.asarray([0.5, 1.7, 3.2, 4.0, 1e-9])
    unit = jnp.asarray(np.eye(3)[[0, 1, 2, 0, 1]])
    observed = np.asarray(basis.scalar_vector(distance, unit))
    clamped = np.clip(np.asarray(distance), 1e-6, 3.2)
    cutoff = 0.5 * (np.cos(np.pi * clamped / 3.2) + 1.0)
    scalar = (
        np.exp(-2.5 * (np.asarray(distance)[:, None] - shifts) ** 2) * cutoff[:, None]
    )
    np.testing.assert_allclose(observed[..., 0], scalar, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        observed[..., 1:], scalar[..., None] * np.asarray(unit)[:, None, :], atol=0.0
    )
    np.testing.assert_array_equal(observed[2:4], 0.0)
    slope = jax.grad(lambda r: basis.scalar(r)[2])(jnp.asarray(3.2))
    np.testing.assert_allclose(slope, 0.0, atol=1e-14)


@pytest.mark.parametrize("d2", [True, False])
def test_convsv_contraction_and_rotation_invariance(d2: bool) -> None:
    rng = np.random.default_rng(1)
    conv = AIMNet2ConvSV(rng.normal(size=(3, 4, 2)), d2_features=d2)
    features = rng.normal(size=(5, 3, 4) if d2 else (5, 3))
    geometry = rng.normal(size=(5, 4, 4))
    aggregate = sum(
        np.asarray(conv.message(jnp.asarray(features[j]), jnp.asarray(geometry[j])))
        for j in range(5)
    )
    if d2:
        expected = np.einsum("mag,mgd->agd", features, geometry)
    else:
        expected = np.einsum("ma,mgd->agd", features, geometry)
    np.testing.assert_allclose(aggregate, expected, rtol=1e-13)
    out = np.asarray(conv.finish(jnp.asarray(aggregate)))
    vector = np.einsum("agh,agd->ahd", np.asarray(conv.agh), expected[..., 1:])
    np.testing.assert_allclose(out[:12], expected[..., 0].reshape(-1), rtol=1e-13)
    np.testing.assert_allclose(out[12:], (vector**2).sum(-1).reshape(-1), rtol=1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = expected.copy()
    rotated[..., 1:] = expected[..., 1:] @ q.T
    np.testing.assert_allclose(
        np.asarray(conv.finish(jnp.asarray(rotated))), out, rtol=1e-12
    )


@pytest.mark.parametrize(
    "options",
    [
        {"d2": True, "moments": True},
        {"d2": False, "envelope": "cosine"},
        {"channels": 2, "moments": True, "center": True},
        {"dispersion": True},
        {
            "repulsion": True,
            "head_activation": ("torch.nn.Softplus", {"beta": 2.0, "threshold": 1.0}),
        },
        {
            "frozen_shift": True,
            "head_activation": ("torch.nn.GELU", {"approximate": "tanh"}),
        },
        {"d2": False, "head_activation": ("torch.nn.SiLU", {})},
        pytest.param({"aev_cutoff": 4.0}, id="larger-aev-radius"),
        pytest.param({"aev_cutoff": 4.0, "channels": 2}, id="nse-larger-aev-radius"),
    ],
)
def test_native_forward_matches_independent_source_reference(options: Any) -> None:
    model, declaration, tensors = _model(**options)
    channels = options.get("channels", 1)
    values = {"charge": [1.0]}
    if channels == 2:
        values["multiplicity"] = [3.0]
    request = list(ALL)
    if options.get("moments"):
        request += [AtomisticObservable.DIPOLE, AtomisticObservable.QUADRUPOLE]
    if channels == 2:
        request.append(AtomisticObservable.SPIN_CHARGES)
    if options.get("dispersion"):
        request += [
            AtomisticObservable.ATOMIC_C6,
            AtomisticObservable.ATOMIC_POLARIZABILITY,
        ]
    batch = AtomisticBatch.from_structure(_structure())
    outputs = _evaluate(model, batch, request, **values)
    reference = reference_forward(
        declaration,
        tensors,
        NUMBERS,
        POSITIONS,
        1.0,
        3.0 if channels == 2 else None,
        ts_cutoff=6.0,
    )
    assert bool(outputs.successful[0])
    np.testing.assert_allclose(outputs.energy[0], reference["energy"], rtol=1e-11)
    np.testing.assert_allclose(outputs.atom_energy.sum(), reference["energy"], rtol=1e-11)
    charges = outputs.observable(AtomisticObservable.CHARGES)
    np.testing.assert_allclose(
        charges.values, reference["charges"], rtol=1e-10, atol=1e-12
    )
    residual = outputs.observable(AtomisticObservable.CHARGE_RESIDUAL)
    np.testing.assert_allclose(residual.values[0], reference["residual"], atol=1e-14)
    np.testing.assert_allclose(
        outputs.observable(AtomisticObservable.NODE_EMBEDDINGS).values,
        reference["aim"],
        rtol=1e-10,
        atol=1e-12,
    )
    if channels == 2:
        np.testing.assert_allclose(
            outputs.observable(AtomisticObservable.SPIN_CHARGES).values,
            reference["spin_charges"],
            rtol=1e-10,
            atol=1e-12,
        )
    if options.get("moments"):
        np.testing.assert_allclose(
            outputs.observable(AtomisticObservable.DIPOLE).values[0],
            reference["dipole"],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            outputs.observable(AtomisticObservable.QUADRUPOLE).values[0],
            reference["quadrupole"],
            rtol=1e-10,
            atol=1e-12,
        )
    if options.get("dispersion"):
        np.testing.assert_allclose(
            outputs.observable(AtomisticObservable.ATOMIC_C6).values,
            reference["coefficients"][:, 0],
            rtol=1e-10,
        )


def test_charge_residual_is_the_source_epsilon_residual_not_exact_neutrality() -> None:
    model, declaration, tensors = _model()
    batch = AtomisticBatch.from_structure(_structure())
    outputs = _evaluate(model, batch, ALL, charge=[2.0])
    charges = np.asarray(outputs.observable(AtomisticObservable.CHARGES).values)
    residual = np.asarray(outputs.observable(AtomisticObservable.CHARGE_RESIDUAL).values)
    # Target minus achieved, exactly what the charges sum to.
    np.testing.assert_allclose(residual[0, 0], 2.0 - charges.sum(), atol=1e-14)
    assert abs(residual[0, 0]) > 0.0
    reference = reference_forward(declaration, tensors, NUMBERS, POSITIONS, 2.0)
    np.testing.assert_allclose(residual[0], reference["residual"], atol=1e-14)


def test_equilibration_residual_algebra_uses_the_source_epsilon() -> None:
    from phydrax.nn.atomistic._aimnet2 import _equilibrate

    unconstrained = jnp.asarray([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.05]])
    flexibility = jnp.asarray([[1e-4, 2e-6], [3e-5, 1e-6], [2e-4, 0.0]])
    targets = jnp.asarray([[1.0, -0.5]])
    charges = _equilibrate(targets, unconstrained, flexibility, 1, 3)
    deficit = targets[0] - unconstrained.sum(0)
    total = flexibility.sum(0)
    expected = AIMNET2_CHARGE_EPSILON * deficit / (total + AIMNET2_CHARGE_EPSILON)
    np.testing.assert_allclose(targets[0] - charges.sum(0), expected, rtol=1e-12)
    assert float(jnp.max(jnp.abs(targets[0] - charges.sum(0)))) > 1e-3


def test_changing_charge_and_multiplicity_change_energy_and_targets() -> None:
    model, _, _ = _model(channels=2)
    batch = AtomisticBatch.from_structure(_structure())
    request = (*ALL, AtomisticObservable.SPIN_CHARGES)
    singlet = _evaluate(model, batch, request, charge=[0.0], multiplicity=[1.0])
    triplet = _evaluate(model, batch, request, charge=[0.0], multiplicity=[3.0])
    cation = _evaluate(model, batch, request, charge=[1.0], multiplicity=[1.0])
    assert abs(float(triplet.energy[0] - singlet.energy[0])) > 0.0
    assert abs(float(cation.energy[0] - singlet.energy[0])) > 0.0
    spin = np.asarray(triplet.observable(AtomisticObservable.SPIN_CHARGES).values)
    residual = np.asarray(triplet.observable(AtomisticObservable.CHARGE_RESIDUAL).values)
    # alpha - beta channels target M - 1; the source epsilon leaves a residual.
    np.testing.assert_allclose(
        spin.sum() + residual[0, 0] - residual[0, 1], 2.0, atol=1e-12
    )


def test_missing_and_invalid_conditions_refuse_or_fail_per_case() -> None:
    model, _, _ = _model(channels=2)
    batch = AtomisticBatch.from_structure(_structure())
    graph = _graph(model, batch)
    with pytest.raises(ValueError, match="total-charge conditioning"):
        model.evaluate_graph(
            AtomisticGraphInputs(
                graph, SCALE, species_kind=model.capabilities.species_kind
            )
        )
    with pytest.raises(ValueError, match="multiplicity"):
        _conditioning(model, batch, charge=[0.0])
    single, _, _ = _model()
    with pytest.raises(ValueError, match="conditioning specification"):
        model.evaluate_graph(
            AtomisticGraphInputs(
                graph,
                SCALE,
                species_kind=model.capabilities.species_kind,
                conditioning=_conditioning(single, batch, charge=[0.0]),
            )
        )
    invalid = _evaluate(model, batch, None, charge=[0.0], multiplicity=[0.5])
    assert not bool(invalid.successful[0])
    with pytest.raises(ValueError, match="does not provide"):
        _evaluate(single, batch, (AtomisticObservable.SPIN_CHARGES,), charge=[0.0])


def test_neutral_only_family_refuses_charged_cases() -> None:
    declaration, tensors = source_inventory(4)
    model = aimnet2_potential_from_source(
        SCALE, declaration, tensors, charged_systems=False
    ).potential
    batch = AtomisticBatch.from_structure(_structure())
    assert bool(_evaluate(model, batch, None, charge=[0.0]).successful[0])
    assert not bool(_evaluate(model, batch, None, charge=[1.0]).successful[0])


def test_masks_case_isolation_and_unimplemented_species() -> None:
    model, declaration, tensors = _model()
    small = _structure(NUMBERS[:3], POSITIONS[:3])
    batch = AtomisticBatch.from_structures((_structure(), small))
    outputs = _evaluate(model, batch, ALL, charge=[1.0, -1.0])
    for case, (numbers, positions, charge) in enumerate(
        ((NUMBERS, POSITIONS, 1.0), (NUMBERS[:3], POSITIONS[:3], -1.0))
    ):
        reference = reference_forward(declaration, tensors, numbers, positions, charge)
        np.testing.assert_allclose(outputs.energy[case], reference["energy"], rtol=1e-11)
    charges = np.asarray(outputs.observable(AtomisticObservable.CHARGES).values)
    np.testing.assert_array_equal(charges[7], 0.0)
    unsupported = AtomisticBatch.from_structures(
        (_structure(), _structure([8, 1, 1, 26], POSITIONS))
    )
    mixed = _evaluate(model, unsupported, None, charge=[0.0, 0.0])
    assert bool(mixed.successful[0]) and not bool(mixed.successful[1])


def test_permutation_rotation_and_translation_invariance() -> None:
    model, _, _ = _model(moments=True)
    reference = _evaluate(
        model, AtomisticBatch.from_structure(_structure()), charge=[0.0]
    )
    permutation = np.asarray([2, 0, 3, 1], dtype=np.intp)
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    transformed = POSITIONS[permutation] @ q.T + np.asarray([1.0, -2.0, 0.5])
    moved = _evaluate(
        model,
        AtomisticBatch.from_structure(_structure(NUMBERS[permutation], transformed)),
        charge=[0.0],
    )
    np.testing.assert_allclose(moved.energy, reference.energy, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(moved.observable(AtomisticObservable.CHARGES).values),
        np.asarray(reference.observable(AtomisticObservable.CHARGES).values)[permutation],
        rtol=1e-10,
        atol=1e-12,
    )


def _positions_energy(model: Any, batch: AtomisticBatch, charge: float) -> Any:
    def energy(positions: Any) -> Any:
        return _evaluate(model, batch, None, positions=positions, charge=[charge]).energy[
            0
        ]

    return energy


def test_forces_finite_differences_and_condition_response() -> None:
    model, _, _ = _model(dispersion=True)
    batch = AtomisticBatch.from_structure(_structure())
    energy = _positions_energy(model, batch, 0.5)
    positions = jnp.asarray(batch.positions)
    gradient = jax.grad(energy)(positions)
    step = 1e-6
    direction = jnp.zeros_like(positions).at[0, 3, 2].set(1.0)
    difference = (
        energy(positions + step * direction) - energy(positions - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(gradient[0, 3, 2], difference, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-10)

    def by_charge(charge: Any) -> Any:
        graph = _graph(model, batch)
        conditioning = _conditioning(model, batch, charge=[0.0])
        conditioning = eqx.tree_at(
            lambda item: item.values, conditioning, (charge[None],)
        )
        inputs = AtomisticGraphInputs(
            graph,
            SCALE,
            species_kind=model.capabilities.species_kind,
            conditioning=conditioning,
        )
        return model.evaluate_graph(inputs).energy[0]

    charge = jnp.asarray(0.3)
    response = jax.grad(by_charge)(charge)
    difference = (by_charge(charge + 1e-6) - by_charge(charge - 1e-6)) / 2e-6
    np.testing.assert_allclose(response, difference, rtol=1e-6, atol=1e-8)


def test_mixed_parameter_coordinate_derivative_matches_finite_difference() -> None:
    model, _, _ = _model()
    batch = AtomisticBatch.from_structure(_structure())
    positions = jnp.asarray(batch.positions)

    def force(weight: Any) -> Any:
        updated = eqx.tree_at(lambda item: item.passes[1].weights[0], model, weight)
        return -jax.grad(_positions_energy(updated, batch, 0.0))(positions)[0, 1, 0]

    weight = model.passes[1].weights[0]
    tangent = jnp.zeros_like(weight).at[2, 5].set(1.0)
    _, directional = jax.jvp(force, (weight,), (tangent,))
    difference = (force(weight + 1e-6 * tangent) - force(weight - 1e-6 * tangent)) / 2e-6
    np.testing.assert_allclose(directional, difference, rtol=1e-5, atol=1e-8)


PERIODIC_CELL = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])


def _periodic(numbers: Any, positions: Any, cell: Any) -> Any:
    return AtomicStructure(
        np.asarray(numbers),
        np.asarray(positions),
        np.ones(len(numbers)),
        SCALE,
        cell=cell,
        periodic_axes=np.asarray([True, True, True]),
    )


def _periodic_execution() -> Any:
    return AtomisticGraphExecutionPlan(
        256,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=8192,
            maximum_degree=256,
            maximum_images=343,
        ),
    )


def _source_cell_energy(
    declaration: dict,
    tensors: dict,
    numbers: Any,
    positions: Any,
    cell: Any,
    strain: Any = None,
    **options: Any,
) -> float:
    deformation = np.eye(3) if strain is None else np.eye(3) + strain
    return float(
        reference_forward(
            declaration,
            tensors,
            np.asarray(numbers),
            np.asarray(positions) @ deformation,
            0.0,
            cell=np.asarray(cell) @ deformation,
            **options,
        )["energy"]
    )


def test_periodic_energy_matches_source_image_sums_routes_and_stress() -> None:
    model, declaration, tensors = _model()
    unit_positions = POSITIONS[:2]
    supercell = PERIODIC_CELL * np.asarray([[2.0], [1.0], [1.0]])
    cells = (
        ([8, 1], unit_positions, PERIODIC_CELL),
        (
            [8, 1, 8, 1],
            np.concatenate((unit_positions, unit_positions + PERIODIC_CELL[0])),
            supercell,
        ),
    )
    observed, sources = [], []
    for numbers, positions, cell in cells:
        batch = AtomisticBatch.from_structure(_periodic(numbers, positions, cell))
        result = energy_and_forces(
            model,
            batch,
            _periodic_execution(),
            compute_stress=True,
            conditioning=_conditioning(model, batch, charge=[0.0]),
        )
        assert bool(result.valid[0]) and result.stress is not None
        sources.append(
            _source_cell_energy(declaration, tensors, numbers, positions, cell)
        )
        np.testing.assert_allclose(result.energy[0], sources[-1], rtol=1e-11)
        # Tensile-positive dE/d(strain) / V against the source's own strain response.
        step = 1e-5
        volume = abs(np.linalg.det(cell))
        expected = np.empty((3, 3))
        for a in range(3):
            for b in range(a, 3):
                strain = np.zeros((3, 3))
                strain[a, b] += 0.5 * step
                strain[b, a] += 0.5 * step
                difference = _source_cell_energy(
                    declaration, tensors, numbers, positions, cell, strain
                ) - _source_cell_energy(
                    declaration, tensors, numbers, positions, cell, -strain
                )
                expected[a, b] = expected[b, a] = difference / (2.0 * step * volume)
        np.testing.assert_allclose(result.stress[0], expected, rtol=1e-6, atol=1e-9)
        observed.append(result)
    unit, doubled = observed
    # Lattice translations are exact symmetries: wrapping an atom into another
    # image changes nothing, and the supercell maps each copy onto the other.
    numbers, positions, cell = cells[0]
    wrapped_batch = AtomisticBatch.from_structure(
        _periodic(
            numbers,
            positions + np.stack((np.zeros(3), cell[1] - cell[2])),
            cell,
        )
    )
    wrapped = energy_and_forces(
        model,
        wrapped_batch,
        _periodic_execution(),
        conditioning=_conditioning(model, wrapped_batch, charge=[0.0]),
    )
    np.testing.assert_allclose(wrapped.energy, unit.energy, rtol=1e-11)
    np.testing.assert_allclose(wrapped.forces, unit.forces, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        doubled.forces[0, 2:4], doubled.forces[0, :2], rtol=1e-10, atol=1e-10
    )
    # The source equilibrates each case with sum(f) + AIMNET2_CHARGE_EPSILON, so a
    # doubled cell is not exactly twice its unit cell. The native deviation is the
    # source's, and the same image sums without that epsilon are extensive.
    np.testing.assert_allclose(
        doubled.energy[0] - 2.0 * unit.energy[0],
        sources[1] - 2.0 * sources[0],
        rtol=1e-4,
    )
    exact = [
        _source_cell_energy(
            declaration, tensors, numbers, positions, cell, charge_epsilon=0.0
        )
        for numbers, positions, cell in cells
    ]
    np.testing.assert_allclose(exact[1], 2.0 * exact[0], rtol=1e-12)


def test_output_stage_order_and_refusals() -> None:
    def stages(
        *specs: Callable[[str], AIMNet2OutputStage],
    ) -> list[AIMNet2OutputStage]:
        return [spec(f"s{index}") for index, spec in enumerate(specs)]

    def architecture(outputs: Sequence[AIMNet2OutputStage]) -> AIMNet2Architecture:
        return AIMNet2Architecture(
            feature_count=2,
            d2_features=False,
            shift_count=3,
            combination_count=2,
            hidden=[[4], [4]],
            aim_size=3,
            charge_channels=1,
            aev_cutoff=3.0,
            implemented_species=(1, 8),
            outputs=outputs,
        )

    def head(name: str) -> AIMNet2OutputStage:
        return AIMNet2OutputStage(name, AIMNet2OutputKind.ENERGY_HEAD)

    def total(name: str) -> AIMNet2OutputStage:
        return AIMNet2OutputStage(name, AIMNet2OutputKind.ATOMIC_SUM)

    def coulomb(name: str) -> AIMNet2OutputStage:
        return AIMNet2OutputStage(
            name, AIMNet2OutputKind.SHORT_RANGE_COULOMB, envelope="exp", pair_cutoff=2.0
        )

    def shift(name: str) -> AIMNet2OutputStage:
        return AIMNet2OutputStage(name, AIMNet2OutputKind.ATOMIC_SHIFT)

    def summed_shift(name: str) -> AIMNet2OutputStage:
        return AIMNet2OutputStage(name, AIMNet2OutputKind.ATOMIC_SHIFT, reduce_sum=True)

    with pytest.raises(ValueError, match="after the atomic sum"):
        architecture(stages(head, coulomb, total))
    with pytest.raises(ValueError, match="atom-level energy"):
        architecture(stages(head, total, shift))
    with pytest.raises(ValueError, match="end with a case energy"):
        architecture(stages(head))
    accepted = architecture(stages(head, total, summed_shift))
    assert accepted.cutoff == 3.0


def test_reduce_sum_shift_after_sum_equals_atomic_shift_before_sum() -> None:
    declaration, tensors = source_inventory(9)
    outputs = declaration["kwargs"]["outputs"]
    reordered = dict(declaration)
    shifted = {
        "energy_mlp": outputs["energy_mlp"],
        "atomic_sum": outputs["atomic_sum"],
        "atomic_shift": {
            "class": "aimnet.modules.AtomicShift",
            "kwargs": {"key_in": "energy", "key_out": "energy", "reduce_sum": True},
        },
        "srcoulomb": outputs["srcoulomb"],
    }
    reordered = {**declaration, "kwargs": {**declaration["kwargs"], "outputs": shifted}}
    first = aimnet2_potential_from_source(SCALE, declaration, tensors).potential
    second = aimnet2_potential_from_source(SCALE, reordered, tensors).potential
    batch = AtomisticBatch.from_structure(_structure())
    np.testing.assert_allclose(
        _evaluate(second, batch, None, charge=[0.0]).energy,
        _evaluate(first, batch, None, charge=[0.0]).energy,
        rtol=1e-13,
    )


def test_source_mapping_consumes_or_refuses_every_tensor_and_key() -> None:
    declaration, tensors = source_inventory(11)
    reconstruction = aimnet2_potential_from_source(SCALE, declaration, tensors)
    assert reconstruction.inert_tensors["aev.rc_v"].startswith("registered-vector-basis")
    assert reconstruction.inert_tensors["outputs.dipole.mass"] == (
        "source-loader-expected-unexpected-key"
    )
    assert (
        2 in reconstruction.unimplemented_rows
        and 1 not in reconstruction.unimplemented_rows
    )
    with pytest.raises(AIMNet2SourceRefusedError, match="Unadmitted"):
        aimnet2_potential_from_source(
            SCALE, declaration, {**tensors, "extra.w": np.ones(2)}
        )
    missing = dict(tensors)
    missing.pop("conv_q.agh")
    with pytest.raises(AIMNet2SourceRefusedError, match="missing"):
        aimnet2_potential_from_source(SCALE, declaration, missing)
    partial = dict(tensors)
    partial["afv.weight"] = tensors["afv.weight"].copy()
    partial["afv.weight"][6, 0] = np.nan
    with pytest.raises(AIMNet2SourceRefusedError, match="entirely NaN"):
        aimnet2_potential_from_source(SCALE, declaration, partial)
    with pytest.raises(AIMNet2SourceRefusedError, match="unimplemented"):
        aimnet2_potential_from_source(
            SCALE, declaration, tensors, declared_species=[1, 2]
        )

    def variant(change: Any) -> dict:
        copy = {**declaration, "kwargs": {**declaration["kwargs"]}}
        change(copy["kwargs"])
        return copy

    def add_output(name: str, entry: Any) -> Any:
        return lambda kwargs: kwargs.update(outputs={**kwargs["outputs"], name: entry})

    # Each refusal is a declaration the source cannot execute, fails on, or an
    # untyped auxiliary output; embedded long-range modules need externalizing.
    refusals = {
        "source forward fails": variant(
            lambda k: k.update(aev={"rc_s": RC_S, "nshifts_s": 4, "nshifts_v": 3})
        ),
        "untyped auxiliary": variant(
            add_output("rep", {"class": "aimnet.modules.core.SRRep", "kwargs": {}})
        ),
        "externalized": variant(
            add_output(
                "lrcoulomb",
                {"class": "aimnet.modules.LRCoulomb", "kwargs": {"rc": 4.6}},
            )
        ),
        "unadmitted keys": variant(lambda k: k.update(unknown=1)),
        "dtype object": variant(
            add_output(
                "shift2",
                {
                    "class": "aimnet.modules.AtomicShift",
                    "kwargs": {"dtype": "torch.float64"},
                },
            )
        ),
        "typed observable": variant(
            add_output(
                "spin_dipole",
                {"class": "aimnet.modules.Dipole", "kwargs": {"key_in": "spin_charges"}},
            )
        ),
        "not admitted": variant(
            add_output(
                "head2",
                {
                    "class": "aimnet.modules.Output",
                    "kwargs": {
                        "n_in": 4,
                        "n_out": 1,
                        "key_in": "aim",
                        "key_out": "energy",
                        "mlp": {"activation_fn": "torch.nn.Mish"},
                    },
                },
            )
        ),
    }
    for reason, refused in refusals.items():
        with pytest.raises(AIMNet2SourceRefusedError, match=reason):
            aimnet2_potential_from_source(SCALE, refused, tensors)
    dispersion_declaration, dispersion_tensors = source_inventory(12, dispersion=True)
    # Every executed pair buffer is declared, so the D3TS calculator neighbor
    # radius is the only missing input: absent it refuses, declared it admits.
    interpretation: dict[str, Any] = {
        "aev_cutoff": RC_S,
        "implemented_species": (1, 6, 7, 8),
        "pair_radii": {"srcoulomb": float(dispersion_tensors["outputs.srcoulomb.rc"])},
    }
    with pytest.raises(AIMNet2SourceRefusedError):
        aimnet2_architecture_from_source(dispersion_declaration, **interpretation)
    declared = aimnet2_architecture_from_source(
        dispersion_declaration, **interpretation, ts_pair_cutoff=6.0
    )
    assert [
        stage.pair_cutoff
        for stage in declared.outputs
        if stage.kind is AIMNet2OutputKind.TS_DISPERSION
    ] == [6.0]


def test_missing_buffers_execute_constructor_values() -> None:
    declaration, tensors = source_inventory(13)
    tensors.pop("outputs.srcoulomb.rc")
    reconstruction = aimnet2_potential_from_source(SCALE, declaration, tensors)
    assert reconstruction.constructor_values["outputs.srcoulomb.rc"] == float(
        np.float32(2.4)
    )


def test_dispparam_reference_arguments_and_ptfile_semantics() -> None:
    declaration, tensors = source_inventory(14, dispersion=True)
    options = declaration["kwargs"]["outputs"]["dispparam"]["kwargs"]
    options["ptfile"] = "disp_params.pt"
    superseded = aimnet2_potential_from_source(
        SCALE, declaration, tensors, ts_pair_cutoff=6.0
    )
    assert superseded.superseded_declarations["outputs.dispparam.ptfile"] == (
        "constructor-value-superseded-by-state-buffer"
    )
    table = tensors.pop("outputs.dispparam.disp_param0")
    with pytest.raises(AIMNet2SourceRefusedError, match="ptfile"):
        aimnet2_potential_from_source(SCALE, declaration, tensors, ts_pair_cutoff=6.0)
    del options["ptfile"]
    options["ref_c6"] = {int(z): float(table[z, 0]) for z in (1, 6, 7, 8)}
    options["ref_alpha"] = {int(z): float(table[z, 1]) for z in (1, 6, 7, 8)}
    constructed = aimnet2_potential_from_source(
        SCALE, declaration, tensors, ts_pair_cutoff=6.0
    )
    assert constructed.constructor_values["outputs.dispparam.disp_param0"] == (
        "ref_c6/ref_alpha"
    )
    batch = AtomisticBatch.from_structure(_structure())
    reference = superseded.potential
    # The constructor table is float32 (torch default); energies agree to it.
    np.testing.assert_allclose(
        _evaluate(constructed.potential, batch, None, charge=[0.0]).energy,
        _evaluate(reference, batch, None, charge=[0.0]).energy,
        rtol=1e-6,
    )


def test_embedded_long_range_coulomb_is_externalized_without_double_counting() -> None:
    declaration, tensors = source_inventory(15, long_range=True)
    core, state, terms = externalize_source_long_range(declaration, tensors)
    assert terms.coulomb is not None and terms.coulomb["method"] == "simple"
    assert "srcoulomb" in core["kwargs"]["outputs"]
    model = aimnet2_potential_from_source(SCALE, core, state).potential
    batch = AtomisticBatch.from_structure(_structure())
    outputs = _evaluate(model, batch, ALL, charge=[1.0])
    charges = np.asarray(outputs.observable(AtomisticObservable.CHARGES).values)
    distance = np.linalg.norm(POSITIONS[:, None] - POSITIONS[None], axis=-1)
    upper = np.triu_indices(4, 1)
    full = SOURCE_COULOMB_CONSTANT * np.sum(
        charges[upper[0]] * charges[upper[1]] / distance[upper]
    )
    reference = reference_forward(declaration, tensors, NUMBERS, POSITIONS, 1.0)
    np.testing.assert_allclose(
        float(outputs.energy[0]) + full, reference["energy"], rtol=1e-11
    )
    composition = aimnet2_calculator_composition({"cutoff": RC_S}, externalized=terms)
    assert composition.external_coulomb and not composition.subtract_short_range
    assert composition.coulomb_mode == "sr_embedded"
    with pytest.raises(AIMNet2SourceRefusedError, match="counted twice|embeds it"):
        aimnet2_calculator_composition(
            {"cutoff": RC_S, "needs_coulomb": True}, externalized=terms
        )


def test_frozen_atomic_shifts_are_fixed_role_without_detaching_geometry() -> None:
    from phydrax._trainable import partition_parameters
    from phydrax.nn.atomistic._aimnet2 import (
        AIMNet2AtomicShifts,
        AIMNet2FrozenAtomicShifts,
    )

    frozen, _, _ = _model(frozen_shift=True)
    trainable, _, _ = _model()
    index = [stage.kind for stage in frozen.configuration.outputs].index(
        AIMNet2OutputKind.ATOMIC_SHIFT
    )
    assert isinstance(frozen.stage_values[index], AIMNet2FrozenAtomicShifts)
    assert isinstance(trainable.stage_values[index], AIMNet2AtomicShifts)
    parameters = partition_parameters(frozen)[0].stage_values[index]
    assert jax.tree_util.tree_leaves(parameters) == []
    trainable_parameters = partition_parameters(trainable)[0].stage_values[index]
    assert len(jax.tree_util.tree_leaves(trainable_parameters)) == 1
    batch = AtomisticBatch.from_structure(_structure())
    positions = jnp.asarray(batch.positions)
    np.testing.assert_allclose(
        jax.grad(_positions_energy(frozen, batch, 0.0))(positions),
        jax.grad(_positions_energy(trainable, batch, 0.0))(positions),
        rtol=1e-12,
    )


def test_calculator_composition_records_source_recipe_and_rxn_defaults() -> None:
    sr = aimnet2_calculator_composition(
        {
            "cutoff": 5.0,
            "needs_coulomb": True,
            "needs_dispersion": True,
            "coulomb_mode": "sr_embedded",
            "coulomb_sr_rc": 4.599999904632568,
            "coulomb_sr_envelope": "exp",
            "d3_params": {"s8": 0.3908, "a1": 0.566, "a2": 3.128, "s6": 1.0},
        }
    )
    assert sr.external_coulomb and not sr.subtract_short_range
    assert sr.external_coulomb_method == "simple" and sr.d3_cutoff == 15.0
    rxn = aimnet2_calculator_composition(
        {
            "cutoff": 5.0,
            "needs_coulomb": True,
            "needs_dispersion": False,
            "coulomb_mode": "sr_embedded",
            "coulomb_sr_rc": 4.6,
            "family": "rxn",
            "supports_charged_systems": False,
        }
    )
    assert rxn.external_dispersion
    assert rxn.d3_parameters is not None
    assert dict(rxn.d3_parameters) == {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}
    assert rxn.family_defaults == ("rxn-needs-dispersion", "rxn-wb97m-d3-parameters")
    with pytest.raises(AIMNet2SourceRefusedError):
        aimnet2_calculator_composition(
            {"cutoff": 5.0, "family": "rxn", "supports_charged_systems": True}
        )


def test_default_construction_runs_and_has_source_layout() -> None:
    architecture = AIMNet2Architecture(
        feature_count=2,
        d2_features=True,
        shift_count=3,
        combination_count=2,
        hidden=[[4], [4], [3]],
        aim_size=3,
        charge_channels=2,
        aev_cutoff=3.0,
        implemented_species=(1, 6, 8),
        outputs=[
            AIMNet2OutputStage("energy", AIMNet2OutputKind.ENERGY_HEAD, hidden=(3,)),
            AIMNet2OutputStage("shift", AIMNet2OutputKind.ATOMIC_SHIFT),
            AIMNet2OutputStage("sum", AIMNet2OutputKind.ATOMIC_SUM),
        ],
    )
    model = AIMNet2Potential(SCALE, architecture, key=jax.random.key(0))
    assert [mlp.out_size for mlp in model.passes] == [10, 10, 3]
    batch = AtomisticBatch.from_structure(_structure([8, 1, 1], POSITIONS[:3]))
    outputs = _evaluate(model, batch, None, charge=[0.0], multiplicity=[2.0])
    assert bool(outputs.successful[0]) and np.isfinite(float(outputs.energy[0]))


def _architecture(*, charged_systems: bool = True) -> AIMNet2Architecture:
    return AIMNet2Architecture(
        feature_count=2,
        d2_features=True,
        shift_count=3,
        combination_count=2,
        hidden=[[4], [4], [3]],
        aim_size=3,
        charge_channels=1,
        aev_cutoff=3.0,
        implemented_species=(1, 6, 8),
        charged_systems=charged_systems,
        outputs=[
            AIMNet2OutputStage("energy", AIMNet2OutputKind.ENERGY_HEAD, hidden=(3,)),
            AIMNet2OutputStage("shift", AIMNet2OutputKind.ATOMIC_SHIFT, frozen=True),
            AIMNet2OutputStage("sum", AIMNet2OutputKind.ATOMIC_SUM),
        ],
    )


def test_validate_accepts_constructed_reconstructed_and_trained_models() -> None:
    default = AIMNet2Potential(SCALE, _architecture(), key=jax.random.key(1))
    default.validate()
    for options in (
        {"dispersion": True},
        {"repulsion": True},
        {"channels": 2, "moments": True, "center": True},
    ):
        model, _, _ = _model(**options)
        model.validate()
    # Parameter updates are training, not a new scientific identity.
    trained = eqx.tree_at(
        lambda item: item.passes[0].weights[0], model, model.passes[0].weights[0] + 0.25
    )
    trained.validate()
    assert (
        trained.conditioning_spec.spec_id
        == model.configuration.conditioning_spec().spec_id
    )


def test_validate_refuses_stale_or_bypassed_components() -> None:
    model, _, _ = _model(dispersion=True)
    shift = [stage.kind for stage in model.configuration.outputs].index(
        AIMNet2OutputKind.ATOMIC_SHIFT
    )
    dispersion = [stage.kind for stage in model.configuration.outputs].index(
        AIMNet2OutputKind.DISPERSION_PARAMETERS
    )
    tampered = {
        "nonfinite parameter": eqx.tree_at(
            lambda item: item.passes[1].weights[0],
            model,
            model.passes[1].weights[0].at[0, 0].set(jnp.nan),
        ),
        "wrong dtype": eqx.tree_at(
            lambda item: item.atom_conv.agh,
            model,
            model.atom_conv.agh.astype(jnp.float32),
        ),
        "stale fixed basis": eqx.tree_at(
            lambda item: item.basis.shifts, model, model.basis.shifts + 0.1
        ),
        "stale dispersion reference": eqx.tree_at(
            lambda item: item.stage_values[dispersion].table,
            model,
            model.stage_values[dispersion].table.at[1, 0].multiply(2.0),
        ),
        "padding dummy": eqx.tree_at(
            lambda item: item.stage_values[dispersion].table,
            model,
            model.stage_values[dispersion].table.at[0, 1].set(2.0),
        ),
        "nonfinite embedding": eqx.tree_at(
            lambda item: item.embedding, model, model.embedding.at[1, 0].set(jnp.inf)
        ),
        "reduction dtype": eqx.tree_at(
            lambda item: item.stage_values[shift].shifts,
            model,
            model.stage_values[shift].shifts.astype(jnp.float32),
        ),
    }
    for reason, value in tampered.items():
        with pytest.raises((ValueError, TypeError)):
            value.validate()
        del reason
    frozen = AIMNet2Potential(SCALE, _architecture(), key=jax.random.key(2))
    frozen_shifts = frozen.stage_values[1]
    assert isinstance(frozen_shifts, AIMNet2FrozenAtomicShifts)
    stale_frozen = eqx.tree_at(
        lambda item: item.stage_values[1].shifts,
        frozen,
        frozen_shifts.shifts.at[6].add(1.0),
    )
    with pytest.raises(ValueError, match="recorded identities"):
        stale_frozen.validate()
    swapped = eqx.tree_at(
        lambda item: item.configuration, frozen, _architecture(charged_systems=False)
    )
    with pytest.raises(ValueError, match="recorded identities"):
        swapped.validate()
