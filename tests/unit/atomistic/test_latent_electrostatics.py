"""Native latent Ewald summation (LES) of learned latent charge channels.

References come from ``tests/_support/atomistic_latent_ewald_reference.py``
(independent NumPy loops and an independent Ewald split), the published Born
projection by finite differences, and finite differences of the native energy.
"""

import sys
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._differentiation import GradientLevel
from phydrax._identity import IdentityWords
from phydrax._model import model_structure_recipe
from phydrax._model._structure import model_from_array_recipe, pack_model_array_tree
from phydrax.atomistic._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
    AtomisticObservableValue,
)
from phydrax.atomistic._latent_electrostatics import (
    LATENT_CHARGE_PORT,
    LATENT_EWALD_ENERGY_PORT,
    LatentBornProjection,
    LatentEwaldDefinition,
    LatentEwaldEvaluation,
    LatentEwaldPotential,
    LatentEwaldResources,
    PreparedLatentEwald,
    SOURCE_NORM_FACTOR,
)
from phydrax.atomistic._model_artifact import ATOMISTIC_MODEL_ARTIFACT_LIMITS
from phydrax.atomistic._potential import (
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from phydrax.atomistic._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from phydrax.atomistic._reciprocal_mesh import ReciprocalBudget, ReciprocalModeSetPlan
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.atomistic._types import AtomisticScaleContract
from phydrax.units import (
    ANGSTROM,
    BOHR,
    conversion_factor,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    HARTREE,
)
from tests._support.atomistic_bundles import bundle_state, prepare_program
from tests._support.atomistic_latent_ewald_reference import (
    born_charges as reference_born_charges,
    finite_energy,
    mode_energy,
    polarization as reference_polarization,
    self_energy,
    split_energy,
)
from tests._support.atomistic_long_range_reference import central_difference
from tests._support.metal_process import json_result, run_python


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))

SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
TRICLINIC = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
POSITIONS = np.asarray(
    [[0.3, 0.4, 0.5], [2.9, 1.2, 0.8], [1.1, 3.7, 2.6], [4.2, 2.5, 5.1]]
)
CHARGES = np.asarray([0.7, -0.3, 0.45, -0.2])  # deliberately nonneutral


def _prepared(
    definition: LatentEwaldDefinition,
    count: int,
    *,
    maximum_length: float | None = None,
    scale: AtomisticScaleContract = SCALE,
    **kwargs: Any,
) -> Any:
    extents = (
        None
        if maximum_length is None and definition.mode_support == "complete-sphere"
        else definition.covering_extents(maximum_length or 1.0)
    )
    resources = LatentEwaldResources(mode_extents=extents, **kwargs)
    return definition.prepare_graph(scale, count, resources=resources)


def _evaluate(
    prepared: Any,
    charges: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> LatentEwaldEvaluation:
    count = positions.shape[0]
    if cell is None:
        vectors = jnp.zeros((1, 0, 3))
        mask = jnp.zeros((1, 0), dtype=bool)
    else:
        vectors = jnp.asarray(cell)[None]
        mask = jnp.asarray(periodic)[None]
    return prepared.energy(
        jnp.asarray(charges),
        jnp.asarray(positions),
        jnp.ones((count,), dtype=bool),
        vectors,
        mask,
    )


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# --- definition and source admission ----------------------------------------


def test_source_arguments_map_to_explicit_branches_and_supports() -> None:
    default = LatentEwaldDefinition.from_source_arguments({"use_atomwise": False})
    assert (default.sigma, default.dl, default.norm_factor) == (1.0, 2.0, 90.4756)
    assert default.branch == "per-case" and default.mode_support == "complete-sphere"
    assert default.remove_self_interaction and default.born.remove_mean
    forced = LatentEwaldDefinition.from_source_arguments(
        {"is_periodic": True, "N_max": 7}
    )
    assert forced.branch == "periodic" and forced.index_bound == 7
    assert LatentEwaldDefinition.from_source_arguments({"is_periodic": False}).branch == (
        "finite"
    )
    inert = LatentEwaldDefinition.from_source_arguments(
        {"use_atomwise": True, "compute_bec": True, "bec_output_index": 2}
    )
    assert inert.inert_source_flags == ("use_atomwise",)
    assert inert.compute_born and inert.born.output_index == 2
    assert inert.definition_id != default.definition_id
    with pytest.raises(ValueError, match="Unrecognized"):
        LatentEwaldDefinition.from_source_arguments({"sigma": 1.0, "cutoff": 5.0})
    with pytest.raises(ValueError, match="one latent channel"):
        LatentEwaldDefinition(channels=2, compute_born=True)
    assert np.isclose(default.coulomb_constant, SOURCE_NORM_FACTOR / (2 * np.pi))


def test_latent_field_is_a_distinct_scientific_kind() -> None:
    definition = LatentEwaldDefinition()
    latent = AtomisticFieldReference(
        "latent-model",
        AtomisticObservable.LATENT_CHARGES,
        LATENT_CHARGE_PORT,
        level=AtomisticConditionLevel.ATOM,
        unit=ELEMENTARY_CHARGE,
    )
    term = LatentEwaldPotential(definition, latent)
    assert term.requirements.field_dependencies == (latent,)
    assert not term.capabilities.local_energy
    physical = AtomisticFieldReference(
        "latent-model",
        AtomisticObservable.CHARGES,
        LATENT_CHARGE_PORT,
        level=AtomisticConditionLevel.ATOM,
        unit=ELEMENTARY_CHARGE,
    )
    with pytest.raises(ValueError, match="LATENT_CHARGES"):
        LatentEwaldPotential(definition, physical)
    two = LatentEwaldDefinition(channels=2)
    with pytest.raises(ValueError, match="channel count"):
        LatentEwaldPotential(two, latent)


# --- finite branch -------------------------------------------------------------


@pytest.mark.parametrize("remove_self", [True, False])
def test_finite_branch_matches_direct_reference_for_two_channels(
    remove_self: bool,
) -> None:
    channels = np.stack([CHARGES, np.asarray([-0.1, 0.5, 0.2, 0.3])], axis=1)
    definition = LatentEwaldDefinition(
        sigma=0.9, remove_self_interaction=remove_self, channels=2
    )
    prepared = _prepared(definition, 4, row_block=3)
    result = _evaluate(prepared, channels, POSITIONS, None)
    assert bool(result.successful[0]) and not bool(result.periodic[0])
    expected = finite_energy(channels, POSITIONS, 0.9, remove_self=remove_self)
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-13)
    signed = 0.0 if remove_self else self_energy(channels, 0.9)
    np.testing.assert_allclose(float(result.self_energy[0]), signed, rtol=1e-13)
    assert float(result.reciprocal_energy[0]) == 0.0


def test_finite_kernel_is_analytic_through_coincident_atoms() -> None:
    definition = LatentEwaldDefinition(sigma=1.3)
    prepared = _prepared(definition, 2)
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = _evaluate(prepared, np.asarray([0.5, -0.4]), positions, None)
    expected = finite_energy(np.asarray([0.5, -0.4]), positions, 1.3)
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-14)

    def energy(shift: Array) -> Array:
        moved = jnp.asarray(positions).at[1, 0].add(shift)
        return prepared.energy(
            jnp.asarray([0.5, -0.4]),
            moved,
            jnp.ones((2,), dtype=bool),
            jnp.zeros((1, 0, 3)),
            jnp.zeros((1, 0), dtype=bool),
        ).energy[0]

    for step in (0.0, 1.0e-4, 2.0e-3):
        gradient = float(jax.grad(energy)(jnp.asarray(step)))
        assert np.isfinite(gradient)
        numeric = (
            float(energy(jnp.asarray(step + 1e-6)))
            - float(energy(jnp.asarray(step - 1e-6)))
        ) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-9)


def test_row_blocks_and_mode_chunks_are_execution_only() -> None:
    definition = LatentEwaldDefinition()
    values = []
    for block, chunk in ((1, 7), (3, 64), (64, 500)):
        prepared = _prepared(
            definition, 4, maximum_length=8.0, row_block=block, mode_chunk=chunk
        )
        finite = _evaluate(prepared, CHARGES, POSITIONS, None)
        periodic = _evaluate(prepared, CHARGES, POSITIONS, TRICLINIC)
        values.append((float(finite.energy[0]), float(periodic.energy[0])))
    np.testing.assert_allclose(values, [values[0]] * 3, rtol=1e-13)


# --- periodic branch ---------------------------------------------------------


@pytest.mark.parametrize("remove_self", [True, False])
def test_periodic_complete_sphere_matches_full_mode_enumeration(
    remove_self: bool,
) -> None:
    definition = LatentEwaldDefinition(remove_self_interaction=remove_self)
    prepared = _prepared(definition, 4, maximum_length=8.0)
    result = _evaluate(prepared, CHARGES, POSITIONS, TRICLINIC)
    assert bool(result.successful[0]) and bool(result.periodic[0])
    assert bool(result.mode_coverage_successful[0])
    expected = mode_energy(
        CHARGES, POSITIONS, TRICLINIC, 1.0, 2.0, remove_self=remove_self
    )
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-12)
    signed = -self_energy(CHARGES, 1.0) if remove_self else 0.0
    np.testing.assert_allclose(float(result.self_energy[0]), signed, rtol=1e-13)
    assert float(result.finite_energy[0]) == 0.0
    assert int(result.active_modes[0]) > 0


def test_index_box_support_reports_its_declared_truncation() -> None:
    definition = LatentEwaldDefinition(
        dl=1.0, mode_support="index-box-sphere", index_bound=2, branch="periodic"
    )
    prepared = _prepared(definition, 4)
    result = _evaluate(prepared, CHARGES, POSITIONS, TRICLINIC)
    assert bool(result.successful[0]) and bool(result.sphere_truncated[0])
    expected = mode_energy(CHARGES, POSITIONS, TRICLINIC, 1.0, 1.0, index_bound=2)
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="N_max"):
        definition.prepare_graph(
            SCALE, 4, resources=LatentEwaldResources(mode_extents=(3, 3, 3))
        )
    # A directly bound mode set must be the definition's own N_max plan under
    # the declared resources, not merely a box of the declared extents.
    wider = LatentEwaldResources(mode_extents=(3, 3, 3))
    with pytest.raises(ValueError, match="N_max"):
        PreparedLatentEwald(
            definition,
            SCALE,
            4,
            wider,
            ReciprocalModeSetPlan((3, 3, 3)).prepare_lattice_family(4),
        )
    declared = LatentEwaldResources(mode_extents=(2, 2, 2))
    other_chunk = ReciprocalModeSetPlan((2, 2, 2), mode_chunk=7)
    with pytest.raises(ValueError, match="mode plan"):
        PreparedLatentEwald(
            definition, SCALE, 4, declared, other_chunk.prepare_lattice_family(4)
        )


def test_converged_periodic_energy_has_no_background_term() -> None:
    """Small ``dl`` converges to an independent split without any ``k = 0`` term.

    The configuration is charged; the Gaussian neutralizing-background energy
    ``-pi k_e sigma^2 Q^2 / V`` is many orders above the tolerance, so its
    silent addition would break agreement.
    """
    cell = np.diag([6.0, 6.5, 7.0]) + np.asarray(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.3, 0.6, 0.0]]
    )
    definition = LatentEwaldDefinition(dl=0.6)
    prepared = _prepared(definition, 4, maximum_length=8.0)
    result = _evaluate(prepared, CHARGES, POSITIONS, cell)
    assert abs(CHARGES.sum()) > 0.5 and bool(result.successful[0])
    expected = split_energy(
        CHARGES, POSITIONS, cell, 1.0, alpha=0.6, real_images=2, reciprocal_extent=8
    )
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-9)
    background = (
        np.pi
        * SOURCE_NORM_FACTOR
        / (2 * np.pi)
        * CHARGES.sum() ** 2
        / (np.linalg.det(cell))
    )
    assert background > 1.0e6 * 1.0e-9 * abs(expected)


def test_single_latent_charge_keeps_its_self_image_energy() -> None:
    definition = LatentEwaldDefinition(dl=1.5)
    cell = 5.0 * np.eye(3)
    prepared = _prepared(definition, 1, maximum_length=5.0)
    result = _evaluate(prepared, np.asarray([0.8]), np.zeros((1, 3)), cell)
    expected = mode_energy(np.asarray([0.8]), np.zeros((1, 3)), cell, 1.0, 1.5)
    np.testing.assert_allclose(float(result.energy[0]), expected, rtol=1e-12)


# --- invariances ---------------------------------------------------------------


def test_translation_origin_lattice_wrap_and_rotation_invariance() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=9.0)
    base_finite = float(_evaluate(prepared, CHARGES, POSITIONS, None).energy[0])
    base_periodic = float(_evaluate(prepared, CHARGES, POSITIONS, TRICLINIC).energy[0])
    shifted = POSITIONS + np.asarray([3.1, -7.4, 12.2])
    np.testing.assert_allclose(
        float(_evaluate(prepared, CHARGES, shifted, None).energy[0]),
        base_finite,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        float(_evaluate(prepared, CHARGES, shifted, TRICLINIC).energy[0]),
        base_periodic,
        rtol=1e-11,
    )
    wrapped = POSITIONS.copy()
    wrapped[2] += TRICLINIC[0] - 2 * TRICLINIC[2]
    np.testing.assert_allclose(
        float(_evaluate(prepared, CHARGES, wrapped, TRICLINIC).energy[0]),
        base_periodic,
        rtol=1e-11,
    )
    rotation = _rotation(5)
    np.testing.assert_allclose(
        float(_evaluate(prepared, CHARGES, POSITIONS @ rotation.T, None).energy[0]),
        base_finite,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        float(
            _evaluate(
                prepared, CHARGES, POSITIONS @ rotation.T, TRICLINIC @ rotation.T
            ).energy[0]
        ),
        base_periodic,
        rtol=1e-11,
    )


# --- branch semantics, refusal and staleness ---------------------------------


def test_case_batched_branches_follow_source_semantics() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=9.0)
    positions = np.concatenate([POSITIONS, POSITIONS + 0.1, POSITIONS - 0.2])
    charges = np.concatenate([CHARGES, -CHARGES, 0.5 * CHARGES])
    cells = np.stack([TRICLINIC, TRICLINIC, 1.1 * TRICLINIC])
    masks = np.asarray([[False] * 3, [True] * 3, [True, True, False]])
    active = np.ones((12,), dtype=bool)
    active[11] = False
    result = prepared.energy(
        jnp.asarray(charges), jnp.asarray(positions), jnp.asarray(active), cells, masks
    )
    assert np.asarray(result.periodic).tolist() == [False, True, True]
    assert bool(jnp.all(result.successful))
    expected = [
        finite_energy(CHARGES, POSITIONS, 1.0),
        mode_energy(-CHARGES, POSITIONS + 0.1, TRICLINIC, 1.0, 2.0),
        # A slab (mixed periodicity) runs the full 3D sum over the bound cell.
        mode_energy(0.5 * CHARGES[:3], (POSITIONS - 0.2)[:3], 1.1 * TRICLINIC, 1.0, 2.0),
    ]
    np.testing.assert_allclose(np.asarray(result.energy), expected, rtol=1e-12)


def test_invalid_periodic_cases_are_refused_not_substituted() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=8.0)
    left_handed = TRICLINIC[[1, 0, 2]]
    refused = _evaluate(prepared, CHARGES, POSITIONS, left_handed)
    assert not bool(refused.successful[0]) and np.isnan(float(refused.energy[0]))
    expanded = _evaluate(prepared, CHARGES, POSITIONS, 1.6 * TRICLINIC)
    assert not bool(expanded.mode_coverage_successful[0])
    assert not bool(expanded.successful[0])
    forced = LatentEwaldDefinition(
        branch="periodic", mode_support="index-box-sphere", index_bound=3
    )
    forced_prepared = _prepared(forced, 4)
    no_axis = _evaluate(
        forced_prepared, CHARGES, POSITIONS, TRICLINIC, (False, False, False)
    )
    assert not bool(no_axis.successful[0])
    finite_only = _prepared(definition, 4)
    with pytest.raises(ValueError, match="without reciprocal modes"):
        _evaluate(finite_only, CHARGES, POSITIONS, TRICLINIC)
    nonfinite = CHARGES.copy()
    nonfinite[1] = np.nan
    bad = _evaluate(prepared, nonfinite, POSITIONS, TRICLINIC)
    assert not bool(bad.charges_successful[0]) and not bool(bad.successful[0])


def test_stale_bindings_and_resource_admission_refuse() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=8.0)
    with pytest.raises(ValueError, match="stale"):
        _evaluate(prepared, CHARGES[:3], POSITIONS[:3], None)
    with pytest.raises(ValueError, match="shape"):
        prepared.energy(
            jnp.zeros((4, 2)),
            jnp.asarray(POSITIONS),
            jnp.ones((4,), dtype=bool),
            jnp.zeros((1, 0, 3)),
            jnp.zeros((1, 0), dtype=bool),
        )
    with pytest.raises(ValueError, match="bytes"):
        _prepared(definition, 4, maximum_finite_workspace_bytes=64)
    with pytest.raises(ValueError, match="modes"):
        _prepared(
            definition, 4, maximum_length=8.0, budget=ReciprocalBudget(maximum_modes=16)
        )
    reduced = phx.atomistic.AtomisticUnitSystem.reduced().scale
    with pytest.raises(ValueError, match="does not convert"):
        _prepared(definition, 4, scale=reduced)
    other = LatentEwaldDefinition(sigma=1.1)
    assert _prepared(other, 4).prepared_id != _prepared(definition, 4).prepared_id


def test_consumer_units_convert_by_exact_static_factors() -> None:
    definition = LatentEwaldDefinition()
    bohr_hartree = AtomisticScaleContract(BOHR, HARTREE)
    to_bohr = float(conversion_factor(ANGSTROM, BOHR))
    to_hartree = float(conversion_factor(ELECTRONVOLT, HARTREE))
    for cell in (None, TRICLINIC):
        native = _evaluate(
            _prepared(definition, 4, maximum_length=8.0), CHARGES, POSITIONS, cell
        )
        converted = _evaluate(
            _prepared(definition, 4, maximum_length=8.0, scale=bohr_hartree),
            CHARGES,
            POSITIONS * to_bohr,
            None if cell is None else cell * to_bohr,
        )
        np.testing.assert_allclose(
            float(converted.energy[0]), float(native.energy[0]) * to_hartree, rtol=1e-12
        )
        assert converted.energy_unit.unit_id == HARTREE.unit_id


# --- derivatives -----------------------------------------------------------------


@pytest.mark.parametrize("periodic", [False, True])
def test_position_cell_and_latent_responses_match_finite_differences(
    periodic: bool,
) -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=9.0)
    cell = TRICLINIC if periodic else None
    count = 4

    def energy(charges: Array, positions: Array, vectors: Array | None) -> Array:
        if vectors is None:
            lattice, mask = jnp.zeros((1, 0, 3)), jnp.zeros((1, 0), dtype=bool)
        else:
            lattice, mask = vectors[None], jnp.ones((1, 3), dtype=bool)
        return prepared.energy(
            charges, positions, jnp.ones((count,), dtype=bool), lattice, mask
        ).energy[0]

    q = jnp.asarray(CHARGES)
    r = jnp.asarray(POSITIONS)
    h = None if cell is None else jnp.asarray(cell)
    forces = -np.asarray(jax.grad(energy, argnums=1)(q, r, h))
    numeric = central_difference(
        lambda value: float(energy(q, jnp.asarray(value), h)), POSITIONS, 1e-5
    )
    np.testing.assert_allclose(-forces, numeric, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)
    latent = np.asarray(jax.grad(energy, argnums=0)(q, r, h))
    np.testing.assert_allclose(
        latent,
        central_difference(
            lambda value: float(energy(jnp.asarray(value), r, h)), CHARGES, 1e-6
        ),
        rtol=1e-7,
        atol=1e-9,
    )
    if cell is not None:

        def strained(strain: np.ndarray) -> float:
            deformation = np.eye(3) + strain
            return float(
                energy(
                    q,
                    jnp.asarray(POSITIONS @ deformation.T),
                    jnp.asarray(cell @ deformation.T),
                )
            )

        lattice = jnp.asarray(cell)

        def strained_jax(strain: Array) -> Array:
            deformation = jnp.eye(3) + strain
            return energy(q, r @ deformation.T, lattice @ deformation.T)

        np.testing.assert_allclose(
            np.asarray(jax.grad(strained_jax)(jnp.zeros((3, 3)))),
            central_difference(strained, np.zeros((3, 3)), 1e-6),
            rtol=1e-6,
            atol=1e-8,
        )


def _latent_head(xp: Any, theta: Any, positions: Any) -> Any:
    """Toy position-dependent latent head ``q_i = theta_0 + theta_1 tanh(w . r_i)``."""
    direction = xp.asarray([0.3, -0.2, 0.15])
    return theta[0] * xp.asarray([1.0, -1.0, 0.5, 0.25]) + theta[1] * xp.tanh(
        positions @ direction
    )


def test_parameter_response_and_hessian_actions_through_latent_charges() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=9.0)
    mask = jnp.ones((4,), dtype=bool)
    lattice = jnp.asarray(TRICLINIC)[None]
    periodic = jnp.ones((1, 3), dtype=bool)

    def energy(theta: Array, positions: Array) -> Array:
        charges = _latent_head(jnp, theta, positions)
        return prepared.energy(charges, positions, mask, lattice, periodic).energy[0]

    theta = jnp.asarray([0.4, 0.7])
    r = jnp.asarray(POSITIONS)
    np.testing.assert_allclose(
        np.asarray(jax.grad(energy)(theta, r)),
        central_difference(
            lambda value: float(energy(jnp.asarray(value), r)), np.asarray(theta), 1e-6
        ),
        rtol=1e-7,
        atol=1e-9,
    )
    direction = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)))
    gradient = jax.grad(energy, argnums=1)
    _, action = jax.jvp(lambda value: gradient(theta, value), (r,), (direction,))
    step = 1e-5
    expected = (
        gradient(theta, r + step * direction) - gradient(theta, r - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        np.asarray(action), np.asarray(expected), rtol=1e-6, atol=1e-8
    )


def test_derivative_contracts_declare_mode_sphere_locus() -> None:
    definition = LatentEwaldDefinition()
    finite = definition.derivative_contract(periodic=False)
    assert finite.surfaces[0].level is GradientLevel.SMOOTH
    periodic = definition.derivative_contract(periodic=True)
    assert periodic.surfaces[0].level is GradientLevel.ALMOST_EVERYWHERE
    regularity = periodic.regularity
    assert regularity is not None
    assert definition.mode_cutoff_locus() in regularity.conditions
    assert regularity.continuity == -1


# --- Born projection --------------------------------------------------------------


def test_fixed_charge_born_projection_reproduces_identity_rule() -> None:
    cell = np.asarray([[2.0, 0.3, 0.1], [0.0, 3.0, 0.2], [0.0, 0.0, 4.0]])
    positions = np.asarray([[0.2, 0.4, 0.6], [1.1, 1.2, 1.3]])
    charges = jnp.asarray([1.0, -1.0])
    for remove_mean, cell_value in ((False, cell), (False, None), (True, cell)):
        definition = LatentEwaldDefinition(
            born=LatentBornProjection(remove_mean=remove_mean), compute_born=True
        )
        prepared = _prepared(definition, 2, maximum_length=5.0)
        vectors = (
            jnp.zeros((1, 0, 3)) if cell_value is None else jnp.asarray(cell_value)[None]
        )
        mask = (
            jnp.zeros((1, 0), dtype=bool)
            if cell_value is None
            else jnp.ones((1, 3), dtype=bool)
        )
        result = prepared.born_charges(
            lambda value: charges,
            jnp.asarray(positions),
            jnp.ones((2,), dtype=bool),
            vectors,
            mask,
        )
        assert bool(result.successful[0])
        q = np.asarray(charges) - (np.mean(charges) if remove_mean else 0.0)
        np.testing.assert_allclose(
            np.asarray(result.values), q[:, None, None] * np.eye(3), atol=1e-12
        )


@pytest.mark.parametrize("periodic", [False, True])
def test_born_response_includes_latent_position_response(periodic: bool) -> None:
    theta = np.asarray([0.4, 0.7])
    cell = TRICLINIC if periodic else None
    projection = LatentBornProjection(remove_mean=True, epsilon_factor=2.5)
    definition = LatentEwaldDefinition(born=projection, compute_born=True)
    prepared = _prepared(definition, 4, maximum_length=9.0)
    vectors = jnp.zeros((1, 0, 3)) if cell is None else jnp.asarray(cell)[None]
    mask = jnp.zeros((1, 0), dtype=bool) if cell is None else jnp.ones((1, 3), dtype=bool)
    result = prepared.born_charges(
        lambda value: _latent_head(jnp, jnp.asarray(theta), value),
        jnp.asarray(POSITIONS),
        jnp.ones((4,), dtype=bool),
        vectors,
        mask,
    )
    expected = reference_born_charges(
        lambda value: _latent_head(np, theta, value),
        POSITIONS,
        cell,
        remove_mean=True,
        epsilon_factor=2.5,
    )
    np.testing.assert_allclose(np.asarray(result.values), expected, rtol=1e-7, atol=1e-9)
    polarization = prepared.polarization(
        _latent_head(jnp, jnp.asarray(theta), jnp.asarray(POSITIONS)),
        jnp.asarray(POSITIONS),
        jnp.ones((4,), dtype=bool),
        vectors,
        mask,
    )
    reference = reference_polarization(
        _latent_head(np, theta, POSITIONS),
        POSITIONS,
        cell,
        remove_mean=True,
        epsilon_factor=2.5,
    )
    if cell is None:
        np.testing.assert_allclose(
            np.asarray(polarization.finite[0]), reference, rtol=1e-12
        )
    else:
        np.testing.assert_allclose(
            np.asarray(polarization.periodic_real[0]),
            reference.real,
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(polarization.periodic_imaginary[0]),
            reference.imag,
            rtol=1e-12,
            atol=1e-14,
        )
    row = LatentEwaldDefinition(
        born=LatentBornProjection(remove_mean=True, epsilon_factor=2.5, output_index=1),
        compute_born=True,
    )
    selected = _prepared(row, 4, maximum_length=9.0).born_charges(
        lambda value: _latent_head(jnp, jnp.asarray(theta), value),
        jnp.asarray(POSITIONS),
        jnp.ones((4,), dtype=bool),
        vectors,
        mask,
    )
    np.testing.assert_allclose(
        np.asarray(selected.values), expected[:, 1, :], rtol=1e-7, atol=1e-9
    )
    value = selected.value(1, 4)
    assert value.observable is AtomisticObservable.BORN_CHARGES
    assert value.port.event_shape == (3,)


# --- scalar program composition -----------------------------------------------------


class _LatentProducerTerm(AbstractAtomisticEnergyTerm):
    theta: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, theta: Array, /) -> None:
        self.theta = theta
        self.name = "latent-model"
        self.force_group = 0
        self.term_identity = IdentityWords.of("test-latent-producer")
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
        )
        self.requirements = AtomisticPotentialRequirements()

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "_PreparedLatentProducerTerm":
        return _PreparedLatentProducerTerm(self, system)


class _PreparedLatentProducerTerm(AbstractPreparedAtomisticEnergyTerm):
    plan: _LatentProducerTerm
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: _LatentProducerTerm, system: PreparedAtomisticSystem, /
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.prepared_identity = IdentityWords.of(("prepared", plan.term_id))
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        charges = _latent_head(jnp, self.plan.theta, context.positions)
        field = AtomisticObservableValue(
            AtomisticObservable.LATENT_CHARGES,
            LATENT_CHARGE_PORT,
            charges,
            jnp.ones((1,), dtype=bool),
            level=AtomisticConditionLevel.ATOM,
            case_count=1,
            atom_capacity=charges.shape[0],
            unit=ELEMENTARY_CHARGE,
        )
        atom = 0.05 * charges * charges
        return AtomisticTermEvaluation(
            jnp.sum(atom), atom, jnp.asarray(True), fields=(field,)
        )


def _program(cell: Any, *, system_charge: float = 0.0) -> tuple[Any, Any, Array]:
    producer = _LatentProducerTerm(jnp.asarray([0.4, 0.7]))
    reference = AtomisticFieldReference(
        "latent-model",
        AtomisticObservable.LATENT_CHARGES,
        LATENT_CHARGE_PORT,
        level=AtomisticConditionLevel.ATOM,
        unit=ELEMENTARY_CHARGE,
    )
    definition = LatentEwaldDefinition()
    term = LatentEwaldPotential(
        definition,
        reference,
        resources=LatentEwaldResources(mode_extents=definition.covering_extents(9.0)),
    )
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1, 2, 3]),
        np.asarray([1, 8, 1, 8]),
        np.asarray([1.0, 16.0, 1.0, 16.0]),
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.asarray([0, 1, 0, 1]),
        charges=np.full((4,), system_charge),
        cell=cell,
    ).prepare()
    # Neither term declares a neighborhood: the explicit bundle has no sources.
    program = prepare_program(
        phx.atomistic.AtomisticPotentialProgram([term, producer]), system
    )
    position = jnp.asarray(POSITIONS)
    return program, bundle_state(program, position), position


def _latent_term_context(
    program: Any, state: Any, position: Array
) -> AtomisticPotentialContext:
    """The latent term's context carrying its producer's published latent field."""
    producer = program.terms[program.plan.term_names.index("latent-model")]
    latent = program.terms[program.plan.term_names.index("latent-ewald")]
    produced = producer.energy(program.context(position, state, term=producer))
    return replace(
        program.context(position, state, term=latent),
        field_values=produced.fields,
        field_producers=(producer.name,),
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_program_term_consumes_typed_latent_field_with_connected_forces(
    periodic: bool,
) -> None:
    cell = phx.discretization.PeriodicCell(TRICLINIC) if periodic else None
    program, state, position = _program(cell)
    evaluation = program.evaluate(position, state)
    assert bool(evaluation.successful)
    assert not evaluation.atom_energy_complete
    charges = _latent_head(np, np.asarray([0.4, 0.7]), POSITIONS)
    latent = (
        mode_energy(charges, POSITIONS, TRICLINIC, 1.0, 2.0)
        if periodic
        else finite_energy(charges, POSITIONS, 1.0)
    )
    np.testing.assert_allclose(
        float(evaluation.energy), latent + 0.05 * float(np.sum(charges**2)), rtol=1e-12
    )
    (ledger,) = evaluation.term_evidence[program.plan.term_names.index("latent-ewald")]
    np.testing.assert_allclose(float(ledger.energy[0]), latent, rtol=1e-12)

    def energy_at(value: np.ndarray) -> float:
        # Each displaced point is evaluated at its own certified bundle state.
        moved = jnp.asarray(value)
        return float(program.energy(moved, bundle_state(program, moved))[0])

    np.testing.assert_allclose(
        -np.asarray(evaluation.forces),
        central_difference(energy_at, POSITIONS, 1e-5),
        rtol=1e-7,
        atol=1e-9,
    )
    latent_term = program.terms[program.plan.term_names.index("latent-ewald")]
    own = latent_term.energy(_latent_term_context(program, state, position))
    assert bool(own.successful)
    np.testing.assert_allclose(float(own.energy), latent, rtol=1e-12)
    # A context of another material binding is refused through the term status.
    foreign, foreign_state, _ = _program(cell, system_charge=0.1)
    refused = latent_term.energy(_latent_term_context(foreign, foreign_state, position))
    assert not bool(refused.successful)
    assert np.isnan(float(refused.energy))


def test_global_energy_ledger_value_is_case_level_and_unatomized() -> None:
    definition = LatentEwaldDefinition()
    prepared = _prepared(definition, 4, maximum_length=9.0)
    result = _evaluate(prepared, CHARGES, POSITIONS, TRICLINIC)
    value = result.global_energy_value(4)
    assert value.observable is AtomisticObservable.GLOBAL_ENERGY_COMPONENT
    assert value.port.port_id == LATENT_EWALD_ENERGY_PORT.port_id
    assert value.level is AtomisticConditionLevel.CASE
    np.testing.assert_allclose(np.asarray(value.values), np.asarray(result.energy))


def _recipe_round_trip(prepared: Any) -> Any:
    """Restore through the constructor-bypassing artifact recipe path."""
    recipe = model_structure_recipe(prepared)
    limits = ATOMISTIC_MODEL_ARTIFACT_LIMITS
    arrays = pack_model_array_tree(prepared, recipe, prefix="leaves", limits=limits)
    return model_from_array_recipe(recipe, arrays, prefix="leaves", limits=limits)


@pytest.mark.parametrize("periodic", [False, True], ids=["finite", "periodic"])
def test_restored_binding_validates_and_evaluates_identically(periodic: bool) -> None:
    prepared = _prepared(
        LatentEwaldDefinition(), 4, maximum_length=9.0 if periodic else None
    )
    restored = _recipe_round_trip(prepared)
    restored.validate()
    assert restored.prepared_id == prepared.prepared_id
    assert (restored.modes is None) == (not periodic)
    cell = TRICLINIC if periodic else None
    expected = _evaluate(prepared, CHARGES, POSITIONS, cell)
    observed = _evaluate(restored, CHARGES, POSITIONS, cell)
    assert bool(expected.successful[0]) and bool(observed.successful[0])
    np.testing.assert_allclose(observed.energy, expected.energy, rtol=1e-14, atol=0.0)


# The child restores a small periodic binding whose resources, mode plan and
# support box are replaced by constructor-consistent declarations of a
# 8193^3 box under an inflated budget, while the stored mode arrays and the
# enclosing identities stay those of the small binding. It then limits its
# address space to 1 GiB above its current size: an enumeration of the
# declared box (or any full-box allocation) fails with MemoryError there,
# whereas a consumer that binds the declaration to the stored arrays first
# refuses with ValueError. The default budget must likewise refuse preparing
# the box before enumeration.
_HUGE_MODE_DECLARATIONS = """
import json
import resource
import sys
from collections.abc import Callable

from phydrax._model import model_structure_recipe
from phydrax._model._structure import model_from_array_recipe, pack_model_array_tree
from phydrax.atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from phydrax.atomistic._model_artifact import ATOMISTIC_MODEL_ARTIFACT_LIMITS
from phydrax.atomistic._reciprocal_mesh import ReciprocalBudget, ReciprocalModeSetPlan
from phydrax.atomistic._types import AtomisticScaleContract
from phydrax.units import ANGSTROM, ELECTRONVOLT

limits = ATOMISTIC_MODEL_ARTIFACT_LIMITS
small = LatentEwaldDefinition().prepare_graph(
    AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
    4,
    resources=LatentEwaldResources(mode_extents=(2, 2, 2)),
)
huge = (4096, 4096, 4096)
inflated = ReciprocalBudget(
    maximum_grid_points=1 << 62,
    maximum_modes=1 << 62,
    maximum_chunk_workspace_bytes=1 << 62,
    maximum_retained_bytes=1 << 62,
)
recipe = model_structure_recipe(small)
arrays = pack_model_array_tree(small, recipe, prefix="leaves", limits=limits)
recipe["fields"]["resources"] = model_structure_recipe(
    LatentEwaldResources(mode_extents=huge, budget=inflated)
)
modes = recipe["fields"]["modes"]["fields"]
modes["plan"] = model_structure_recipe(ReciprocalModeSetPlan(huge, budget=inflated))
modes["maximum_mode"] = model_structure_recipe(huge)
forged = model_from_array_recipe(recipe, arrays, prefix="leaves", limits=limits)

with open("/proc/self/status") as status:
    size = next(int(line.split()[1]) for line in status if line.startswith("VmSize:"))
_, hard = resource.getrlimit(resource.RLIMIT_AS)
bound = size * 1024 + (1 << 30)
resource.setrlimit(
    resource.RLIMIT_AS,
    (bound if hard == resource.RLIM_INFINITY else min(bound, hard), hard),
)


def outcome(action: Callable[[], object]) -> str:
    try:
        action()
    except ValueError:
        return "ValueError"
    except MemoryError:
        return "MemoryError"
    return "admitted"


operation = sys.argv[1]
if operation == "restore":
    result = outcome(forged.validate)
elif operation == "prepare":
    result = outcome(lambda: ReciprocalModeSetPlan(huge).prepare_lattice_family(4))
else:
    raise ValueError("Unknown bounded mode admission scenario.")
print(json.dumps({"outcome": result}))
"""


@pytest.mark.skipif(
    sys.platform != "linux", reason="The address-space bound is Linux RLIMIT_AS."
)
@pytest.mark.parametrize("operation", ["restore", "prepare"])
def test_huge_mode_declarations_refuse_before_allocation_in_a_bounded_process(
    operation: str,
) -> None:
    completed = run_python(
        _HUGE_MODE_DECLARATIONS,
        operation,
        environment={"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true"},
        timeout=600.0,
    )
    assert json_result(completed) == {"outcome": "ValueError"}
