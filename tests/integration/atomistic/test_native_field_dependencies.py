"""Typed predicted site moments drive native multipole and polarization terms.

The producer is ``BondChargeMomentModel`` (``tests/_support/atomistic_moment_model.py``):
a small analytic trainable native model whose atom-level charges, dipoles,
primitive raw second moments and physical polarizabilities are the exact
moments of a declared model charge density. It enters each program through
``LearnedGraphPotentialTerm`` and reaches the consumers only through
``MultipoleMomentFields`` and a typed ``POLARIZABILITY`` reference. No
chemistry, accuracy or transferability claim is made.

Every expectation comes from different mathematics than the native route:

* ``explicit_site_moments`` evaluates the defining law on the host from
  explicitly enumerated point charges and a Gauss-Hermite cloud quadrature;
* ``tests._support.atomistic_multipole_reference`` supplies independent host
  multipole and polarization energies and held-moment forces;
* closed-form charge-transfer dimer energies and forces, and the exact
  invisibility of isotropic second moments to point sites (Laplace);
* central differences whose every displaced geometry, cell or parameter
  recomputes the predicted moments by the defining law.

A held-prediction control (the same consumer bound to the evaluated moments
as explicit data) isolates the nonzero chain-rule force share. Polarization
uses the stationary PCG functional with a dense stability certificate; its
second-order (HVP) response is not exercised here.
"""

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
    AtomisticPotentialProgram,
    AtomisticUnitSystem,
    EwaldMultipoleRoute,
    InducedDipolePolarizationPotential,
    MultipoleMomentFields,
    MultipoleSiteKind,
    PolarizationTermEvidence,
    ReciprocalModeSetPlan,
)
from phydrax.units import (
    ANGSTROM,
    BOHR,
    conversion_factor,
    derived_unit,
    ELEMENTARY_CHARGE,
)
from tests._support.atomistic_bundles import bundle_state, prepare_program
from tests._support.atomistic_moment_model import (
    BASIS,
    bound_moment_fields,
    bound_moment_references,
    bound_polarizability,
    CLUSTER,
    CLUSTER_NUMBERS,
    cluster_system,
    consumer_evidence,
    explicit_site_moments,
    ExplicitSiteMoments,
    held_consumer,
    moment_fields,
    moment_model,
    multipole_consumer,
    PERIODIC_CELL,
    PERIODIC_OFFSET,
    polarization_consumer,
    predicted_program,
    PredictedConsumer,
    PRODUCER,
    second_moment_matrices,
    SITE_KINDS,
    SITE_WIDTHS,
    unique_second_moments,
    UNITS,
)
from tests._support.atomistic_multipole_reference import (
    multipole_reference,
    polarization_reference,
)


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
K = UNITS.coulomb_constant
SITES = SITE_KINDS
type ConsumerName = Literal["multipole", "polarization"]
CONSUMERS: tuple[ConsumerName, ...] = ("multipole", "polarization")
STEP = 1.0e-5
CHARGES = AtomisticObservable.CHARGES
DIPOLE = AtomisticObservable.DIPOLE
QUADRUPOLE = AtomisticObservable.QUADRUPOLE
POLARIZABILITY = AtomisticObservable.POLARIZABILITY
PARAMETERS = (
    "electronegativity",
    "bond_charge",
    "cloud_charge",
    "cloud_width",
    "hardness",
    "log_polarizability",
    "screening",
)


# --- native evaluation ---------------------------------------------------------


@eqx.filter_jit
def _evaluate(program: Any, positions: Any, state: Any) -> Any:
    return program.evaluate(positions, state)


@eqx.filter_jit
def _evaluate_strained(program: Any, positions: Any, state: Any) -> Any:
    return program.evaluate(positions, state, compute_stress=True)


def _at(program: Any, positions: Any) -> Any:
    """Connected evaluation with a bundle state certified at ``positions``."""
    x = jnp.asarray(positions, dtype=jnp.float64)
    return _evaluate(program, x, bundle_state(program, x))


@eqx.filter_jit
def _parameter_gradient(
    program: Any,
    parameters: Any,
    model_state: Any,
    fixed: Any,
    positions: Any,
    state: Any,
) -> tuple[Any, Any]:
    """Program-energy gradient in the producer's discovered PARAMETER leaves."""

    def energy(values: Any) -> tuple[Any, Any]:
        model = combine_parameters(values, model_state, fixed)
        bound = eqx.tree_at(lambda value: value.terms[0].plan.potential, program, model)
        total, auxiliary = bound.energy(positions, state)
        return total, auxiliary.successful

    return jax.grad(energy, has_aux=True)(parameters)


@eqx.filter_jit
def _hessian_action(program: Any, positions: Any, state: Any, direction: Any) -> Any:
    def gradient(point: Any) -> Any:
        return jax.grad(lambda value: program.energy(value, state)[0])(point)

    return jax.jvp(gradient, (positions,), (direction,))[1]


# --- independent references ----------------------------------------------------


def _sigmas(numbers: Any = CLUSTER_NUMBERS) -> np.ndarray:
    return np.asarray([SITE_WIDTHS[int(number)] for number in numbers])


def _consumer(
    name: ConsumerName, model: Any, system: Any, site: MultipoleSiteKind, /
) -> PredictedConsumer:
    if name == "multipole":
        return multipole_consumer(model, system, site)
    return polarization_consumer(model, system, site)


def _independent(
    model: Any,
    positions: Any,
    consumer: ConsumerName,
    site: MultipoleSiteKind,
    /,
    *,
    cell_vectors: Any = None,
) -> tuple[float, ExplicitSiteMoments, dict[str, Any]]:
    """Defining-law moments, consumer reference and their total energy."""
    x = np.asarray(positions, dtype=np.float64)
    moments = explicit_site_moments(model, x, CLUSTER_NUMBERS, cell_vectors=cell_vectors)
    vectors = None if cell_vectors is None else np.asarray(cell_vectors, np.float64)
    widths = _sigmas() if site == "gaussian" else None
    # A tight periodic level tolerance keeps level-to-level changes below the
    # resolution of the central differences taken of these totals.
    if consumer == "multipole":
        reference = multipole_reference(
            x,
            moments["charges"],
            moments["dipoles"],
            moments["second_moments"],
            vectors,
            widths=widths,
            coulomb_constant=K,
            tolerance=1.0e-12,
        )
    else:
        reference = polarization_reference(
            x,
            moments["charges"],
            moments["dipoles"],
            moments["second_moments"],
            moments["polarizabilities"],
            vectors,
            widths=widths,
            coulomb_constant=K,
            tolerance=1.0e-12,
        )
    assert reference["independent"] is True
    assert reference["converged"], reference["history"]
    return moments["energy"] + float(reference["energy"]), moments, reference


def _independent_forces(
    model: Any,
    positions: Any,
    consumer: ConsumerName,
    site: MultipoleSiteKind,
    /,
    *,
    cell_vectors: Any = None,
    components: Any = None,
) -> dict[tuple[int, int], float]:
    """Central differences of the independent total energy (moments recomputed)."""
    x = np.asarray(positions, dtype=np.float64)
    selected = (
        [(atom, axis) for atom in range(x.shape[0]) for axis in range(3)]
        if components is None
        else components
    )
    forces = {}
    for atom, axis in selected:
        values = []
        for sign in (1.0, -1.0):
            shifted = x.copy()
            shifted[atom, axis] += sign * STEP
            values.append(
                _independent(model, shifted, consumer, site, cell_vectors=cell_vectors)[0]
            )
        forces[(atom, axis)] = -(values[0] - values[1]) / (2.0 * STEP)
    return forces


def _dense(forces: dict[tuple[int, int], float], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape)
    for (atom, axis), value in forces.items():
        result[atom, axis] = value
    return result


# --- 1. the producer and its defining law ---------------------------------------


def test_published_fields_follow_the_defining_moment_law() -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    consumer = polarization_consumer(model, system, "point")
    program = predicted_program(model, system, (consumer,))
    evaluation = _at(program, CLUSTER)
    assert bool(evaluation.successful)
    charge_field, dipole_field, second_field = bound_moment_references(consumer)
    polarizability_field = bound_polarizability(consumer)
    published = {
        CHARGES: evaluation.field(charge_field),
        DIPOLE: evaluation.field(dipole_field),
        QUADRUPOLE: evaluation.field(second_field),
        POLARIZABILITY: evaluation.field(polarizability_field),
    }
    for observable, value in published.items():
        assert value.level is AtomisticConditionLevel.ATOM
        assert value.port.port_id == model.port(observable).port_id
        assert value.unit is not None
        assert value.unit.unit_id == model.unit(observable).unit_id
        assert bool(value.available[0]) and bool(value.successful[0])
    assert published[DIPOLE].port.normalization_id == BASIS
    assert published[QUADRUPOLE].port.normalization_id == BASIS
    assert evaluation.field_producers == (PRODUCER,) * len(evaluation.fields)

    expected = explicit_site_moments(model, CLUSTER, CLUSTER_NUMBERS)
    np.testing.assert_allclose(
        published[CHARGES].values, expected["charges"], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        published[DIPOLE].values, expected["dipoles"], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        published[QUADRUPOLE].values,
        unique_second_moments(expected["second_moments"]),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        published[POLARIZABILITY].values,
        expected["polarizabilities"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(evaluation.term_energies[0]), expected["energy"], rtol=1e-12
    )
    # Law invariants, not fixture echoes: exact pair-transfer neutrality, and
    # nonzero bond dipoles and anisotropic (traceless) second moments.
    charges = np.asarray(published[CHARGES].values)
    assert abs(float(charges.sum())) <= 1e-14
    assert float(np.max(np.abs(charges))) > 0.1
    assert float(np.max(np.abs(np.asarray(published[DIPOLE].values)))) > 1e-3
    theta = second_moment_matrices(published[QUADRUPOLE].values)
    traceless = theta - np.trace(theta, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    assert float(np.max(np.abs(traceless))) > 1e-3

    # Lab-frame covariance under a proper rotation of the whole geometry.
    rotation, upper = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    rotation = rotation * np.sign(np.diag(upper))
    rotation = rotation * np.sign(np.linalg.det(rotation))
    turned = _at(program, CLUSTER @ rotation.T)
    assert bool(turned.successful)
    np.testing.assert_allclose(
        turned.field(charge_field).values, charges, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        turned.field(dipole_field).values,
        np.asarray(published[DIPOLE].values) @ rotation.T,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        second_moment_matrices(turned.field(second_field).values),
        rotation @ theta @ rotation.T,
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        turned.field(polarizability_field).values,
        published[POLARIZABILITY].values,
        rtol=1e-12,
    )


# --- 2. analytic controls --------------------------------------------------------


DIMER_NUMBERS = (8, 1)
DIMER_DIRECTION = np.asarray([0.48, -0.6, 0.64])
DIMER_ORIGIN = np.asarray([0.1, -0.2, 0.3])
DIMER_DISTANCE = 1.3


def _dimer(model: Any, site: MultipoleSiteKind) -> tuple[Any, np.ndarray]:
    system = cluster_system(numbers=DIMER_NUMBERS)
    program = predicted_program(
        model,
        system,
        (multipole_consumer(model, system, site, numbers=DIMER_NUMBERS),),
    )
    positions = np.stack((DIMER_ORIGIN, DIMER_ORIGIN + DIMER_DISTANCE * DIMER_DIRECTION))
    return _at(program, positions), positions


def _closed_form_dimer(model: Any, site: MultipoleSiteKind) -> tuple[float, float, float]:
    """Charge-transfer O-H dimer: self energy, electrostatics and dE/dd."""
    distance = DIMER_DISTANCE
    cutoff = model.configuration.cutoff
    chi = np.asarray(model.electronegativity)
    eta = np.asarray(model.hardness)
    oxygen, hydrogen = (model.configuration.species.index(n) for n in DIMER_NUMBERS)
    remaining = 1.0 - distance**2 / cutoff**2
    envelope = remaining**3
    slope = -6.0 * distance / cutoff**2 * remaining**2
    # q_H = (chi_O - chi_H) f(d) = -q_O.
    charge = (chi[oxygen] - chi[hydrogen]) * envelope
    charge_slope = (chi[oxygen] - chi[hydrogen]) * slope
    if site == "point":
        kernel, kernel_slope = 1.0 / distance, -1.0 / distance**2
    else:
        smear = np.sqrt(2.0 * (SITE_WIDTHS[8] ** 2 + SITE_WIDTHS[1] ** 2))
        error = math.erf(distance / smear)
        kernel = error / distance
        kernel_slope = (
            2.0 / (np.sqrt(np.pi) * smear) * np.exp(-((distance / smear) ** 2))
            - error / distance
        ) / distance
    hardness = eta[oxygen] + eta[hydrogen]
    self_energy = 0.5 * hardness * charge**2
    electrostatic = -K * charge**2 * kernel
    derivative = hardness * charge * charge_slope - K * (
        2.0 * charge * charge_slope * kernel + charge**2 * kernel_slope
    )
    return float(self_energy), float(electrostatic), float(derivative)


@pytest.mark.parametrize("site", SITES)
def test_charge_transfer_dimer_matches_the_closed_form_energy_and_forces(
    site: MultipoleSiteKind,
) -> None:
    model = moment_model(UNITS, bond_charge=(0.0, 0.0, 0.0), cloud_charge=(0.0, 0.0, 0.0))
    evaluation, _ = _dimer(model, site)
    assert bool(evaluation.successful)
    self_energy, electrostatic, derivative = _closed_form_dimer(model, site)
    terms = np.asarray(evaluation.term_energies)
    np.testing.assert_allclose(terms[0], self_energy, rtol=1e-12)
    np.testing.assert_allclose(terms[1], electrostatic, rtol=1e-12)
    # E depends on d = |x_H - x_O| only: F_H = -dE/dd u, F_O = +dE/dd u.
    np.testing.assert_allclose(
        evaluation.forces,
        np.stack((derivative * DIMER_DIRECTION, -derivative * DIMER_DIRECTION)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_isotropic_second_moments_are_invisible_to_point_sites_only() -> None:
    """``1/2 c I : grad grad (1/R) = 0`` off-site; Gaussian kernels are not harmonic."""
    charges_only = moment_model(
        UNITS, bond_charge=(0.0, 0.0, 0.0), cloud_charge=(0.0, 0.0, 0.0)
    )
    clouded = moment_model(UNITS, bond_charge=(0.0, 0.0, 0.0))
    for site in SITES:
        _, electrostatic, _ = _closed_form_dimer(charges_only, site)
        evaluation, _ = _dimer(clouded, site)
        assert bool(evaluation.successful)
        theta = second_moment_matrices(
            evaluation.field(clouded.reference(QUADRUPOLE)).values
        )
        assert float(np.min(np.abs(np.diagonal(theta, axis1=1, axis2=2)))) > 0.1
        term = float(evaluation.term_energies[1])
        if site == "point":
            np.testing.assert_allclose(term, electrostatic, rtol=1e-12, atol=1e-13)
        else:
            assert abs(term - electrostatic) > 1e-3


# --- 3. connected energies, forces and the held-prediction control ---------------


def _connected_case(name: ConsumerName, site: MultipoleSiteKind) -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    consumer = _consumer(name, model, system, site)
    program = predicted_program(model, system, (consumer,))
    evaluation = _at(program, CLUSTER)
    assert bool(evaluation.successful)
    assert not evaluation.atom_energy_complete
    total, moments, reference = _independent(model, CLUSTER, name, site)
    terms = np.asarray(evaluation.term_energies)
    np.testing.assert_allclose(terms[0], moments["energy"], rtol=1e-12)
    np.testing.assert_allclose(terms[1], reference["energy"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(float(evaluation.energy), total, rtol=1e-9)
    evidence = consumer_evidence(evaluation, 1)
    assert evidence.moment_source == bound_moment_fields(consumer).fields_id
    if name == "polarization":
        assert isinstance(consumer, InducedDipolePolarizationPotential)
        assert isinstance(evidence, PolarizationTermEvidence)
        polarizability = bound_polarizability(consumer).reference_id
        assert evidence.polarizability_source == polarizability
        assert bool(evidence.inputs_successful) and bool(evidence.stationary)
        assert bool(evidence.stability_proven) and not evidence.stability_assumed
        assert bool(evidence.force_valid)

    # Independent total forces: each displaced geometry re-derives the moments.
    independent = _dense(
        _independent_forces(model, CLUSTER, name, site), np.asarray(CLUSTER).shape
    )
    np.testing.assert_allclose(evaluation.forces, independent, rtol=1e-6, atol=1e-8)

    # Held control: identical consumer bound to the evaluated predictions as
    # explicit data reproduces the energy and the held-moment reference forces,
    # so the connected surplus is exactly the predicted-field chain rule.
    alone = predicted_program(model, system, ())
    producer = _at(alone, CLUSTER)
    held = prepare_program(
        AtomisticPotentialProgram([held_consumer(consumer, evaluation)]), system
    )
    fixed = _at(held, CLUSTER)
    assert bool(producer.successful) and bool(fixed.successful)
    np.testing.assert_allclose(float(fixed.energy), terms[1], rtol=1e-11)
    np.testing.assert_allclose(fixed.forces, reference["forces"], rtol=1e-7, atol=1e-9)
    response = np.asarray(evaluation.forces) - (
        np.asarray(producer.forces) + np.asarray(fixed.forces)
    )
    assert float(np.max(np.abs(response))) > 1e-3 * max(
        1.0, float(np.max(np.abs(evaluation.forces)))
    )


@pytest.mark.parametrize("site", SITES)
def test_predicted_moments_drive_the_multipole_energy_and_forces(
    site: MultipoleSiteKind,
) -> None:
    _connected_case("multipole", site)


@pytest.mark.parametrize("site", SITES)
def test_predicted_moments_and_polarizabilities_drive_induced_polarization(
    site: MultipoleSiteKind,
) -> None:
    _connected_case("polarization", site)


# --- 4. parameter response ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "site"), [(name, site) for name in CONSUMERS for site in SITES]
)
def test_parameter_response_matches_independent_differences(
    name: ConsumerName, site: MultipoleSiteKind
) -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    program = predicted_program(model, system, (_consumer(name, model, system, site),))
    x = jnp.asarray(CLUSTER, dtype=jnp.float64)
    state = bundle_state(program, x)
    parameters, model_state, fixed = partition_parameters(model)
    discovered = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(parameters)[0]
    }
    assert discovered == {f".{field}" for field in PARAMETERS}
    gradient, successful = _parameter_gradient(
        program, parameters, model_state, fixed, x, state
    )
    assert bool(successful)
    step = 1.0e-6
    for field in PARAMETERS:
        values = np.asarray(getattr(model, field), dtype=np.float64)
        analytic = np.asarray(getattr(gradient, field), dtype=np.float64)
        for index in np.ndindex(values.shape):
            energies = []
            for sign in (1.0, -1.0):
                moved = values.copy()
                moved[index] += sign * step
                perturbed = eqx.tree_at(
                    lambda value, name=field: getattr(value, name),
                    model,
                    jnp.asarray(moved),
                )
                energies.append(_independent(perturbed, CLUSTER, name, site)[0])
            numeric = (energies[0] - energies[1]) / (2.0 * step)
            np.testing.assert_allclose(
                analytic[index], numeric, rtol=1e-6, atol=1e-8, err_msg=field
            )
    clouds = np.abs(np.asarray(gradient.cloud_charge))
    if site == "point":
        # Isotropic cloud second moments couple to nothing at point sites.
        np.testing.assert_allclose(clouds, 0.0, atol=1e-10)
    else:
        assert float(np.max(clouds)) > 1e-4
    alphas = np.abs(np.asarray(gradient.log_polarizability))
    if name == "polarization":
        assert float(np.min(alphas)) > 1e-8
    else:
        np.testing.assert_array_equal(alphas, 0.0)


# --- 5. second-order response of the connected multipole program ---------------------


@pytest.mark.parametrize("site", SITES)
def test_multipole_hessian_vector_product_matches_force_differences(
    site: MultipoleSiteKind,
) -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    program = predicted_program(model, system, (multipole_consumer(model, system, site),))
    x = jnp.asarray(CLUSTER, dtype=jnp.float64)
    direction = np.random.default_rng(5).normal(size=np.asarray(CLUSTER).shape)
    direction /= np.linalg.norm(direction)
    action = np.asarray(
        _hessian_action(program, x, bundle_state(program, x), jnp.asarray(direction))
    )
    plus = _at(program, np.asarray(CLUSTER) + STEP * direction)
    minus = _at(program, np.asarray(CLUSTER) - STEP * direction)
    assert bool(plus.successful) and bool(minus.successful)
    numeric = -(np.asarray(plus.forces) - np.asarray(minus.forces)) / (2.0 * STEP)
    assert float(np.max(np.abs(action))) > 1e-3
    np.testing.assert_allclose(action, numeric, rtol=1e-5, atol=1e-7)


# --- 6. periodic point multipoles with strain response ----------------------------------


def test_periodic_point_multipoles_consume_predicted_moments_with_strain_response() -> (
    None
):
    model = moment_model(UNITS)
    system = cluster_system(cell=PERIODIC_CELL)
    positions = np.asarray(CLUSTER) + PERIODIC_OFFSET
    consumer = multipole_consumer(
        model,
        system,
        "point",
        route=EwaldMultipoleRoute(0.6, 10.0, ReciprocalModeSetPlan((7, 7, 7))),
    )
    program = predicted_program(model, system, (consumer,))
    x = jnp.asarray(positions, dtype=jnp.float64)
    evaluation = _evaluate_strained(program, x, bundle_state(program, x))
    assert bool(evaluation.successful)
    assert evaluation.strain_derivative is not None
    total, moments, reference = _independent(
        model, positions, "multipole", "point", cell_vectors=PERIODIC_CELL
    )
    _, dipole_field, second_field = bound_moment_references(consumer)
    np.testing.assert_allclose(
        evaluation.field(dipole_field).values, moments["dipoles"], atol=1e-14
    )
    np.testing.assert_allclose(
        evaluation.field(second_field).values,
        unique_second_moments(moments["second_moments"]),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        float(evaluation.term_energies[1]), reference["energy"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(float(evaluation.energy), total, rtol=1e-8)
    for (atom, axis), value in _independent_forces(
        model,
        positions,
        "multipole",
        "point",
        cell_vectors=PERIODIC_CELL,
        components=((0, 0), (2, 1), (4, 2)),
    ).items():
        np.testing.assert_allclose(
            float(evaluation.forces[atom, axis]), value, rtol=1e-6, atol=1e-7
        )
    # Column strain F = I + e deforms positions and lattice rows together and
    # re-derives every predicted moment at the deformed geometry.
    for row, column in ((0, 0), (1, 2), (2, 1)):
        energies = []
        for sign in (1.0, -1.0):
            deformation = np.eye(3)
            deformation[row, column] += sign * STEP
            energies.append(
                _independent(
                    model,
                    positions @ deformation.T,
                    "multipole",
                    "point",
                    cell_vectors=PERIODIC_CELL @ deformation.T,
                )[0]
            )
        np.testing.assert_allclose(
            float(evaluation.strain_derivative[row, column]),
            (energies[0] - energies[1]) / (2.0 * STEP),
            rtol=1e-6,
            atol=1e-7,
        )


# --- 7. units, levels and basis witnesses ----------------------------------------


def test_declared_unit_conversions_compose_exactly() -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    atomic_dipole = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
    atomic_second = derived_unit("e*bohr^2", ((ELEMENTARY_CHARGE, 1), (BOHR, 2)))
    converted_fields = moment_fields(
        model, dipole_unit=atomic_dipole, second_moment_unit=atomic_second
    )
    native_fields = moment_fields(model)
    converted = _at(
        predicted_program(
            model,
            system,
            (multipole_consumer(model, system, "point", fields=converted_fields),),
        ),
        CLUSTER,
    )
    native = _at(
        predicted_program(model, system, (multipole_consumer(model, system, "point"),)),
        CLUSTER,
    )
    assert bool(converted.successful) and bool(native.successful)
    assert converted_fields.fields_id != native_fields.fields_id
    for degree, target in ((DIPOLE, atomic_dipole), (QUADRUPOLE, atomic_second)):
        factor = float(conversion_factor(model.unit(degree), target))
        assert abs(factor - 1.0) > 0.5
        np.testing.assert_allclose(
            converted.field(model.reference(degree, unit=target)).values,
            factor * np.asarray(native.field(model.reference(degree)).values),
            rtol=1e-15,
            atol=0.0,
        )
    # Each term keeps the original 1e-13 precision. The total cancels the
    # producer and consumer terms (each O(0.3)) to O(0.01), so its roundoff
    # bound is that same precision times the summed term magnitudes.
    native_terms = np.asarray(native.term_energies)
    np.testing.assert_allclose(
        np.asarray(converted.term_energies), native_terms, rtol=1e-13, atol=0.0
    )
    np.testing.assert_allclose(
        float(converted.energy),
        float(native.energy),
        rtol=0.0,
        atol=1e-13 * float(np.sum(np.abs(native_terms))),
    )
    np.testing.assert_allclose(converted.forces, native.forces, rtol=1e-12, atol=1e-14)


def test_case_level_moments_are_not_reinterpreted_as_atomic_fields() -> None:
    model = moment_model(UNITS)
    dipole_port = model.port(DIPOLE)
    case_dipole = AtomisticFieldReference(
        PRODUCER,
        DIPOLE,
        dipole_port,
        level=AtomisticConditionLevel.CASE,
        unit=model.unit(DIPOLE),
    )
    with pytest.raises(ValueError):
        MultipoleMomentFields(BASIS, dipoles=case_dipole)


def test_relabeling_a_moment_normalization_refuses() -> None:
    model = moment_model(UNITS)
    second_port = model.port(QUADRUPOLE)
    relabeled = AtomisticFieldReference(
        PRODUCER,
        QUADRUPOLE,
        phx.ValuePort(
            second_port.semantic_id,
            event_shape=second_port.event_shape,
            component_ids=second_port.component_ids,
            representation=second_port.representation,
            dimensions=second_port.dimensions,
            frame_id=second_port.frame_id,
            normalization_id="cartesian-traceless",
        ),
        level=AtomisticConditionLevel.ATOM,
        unit=model.unit(QUADRUPOLE),
    )
    with pytest.raises(ValueError):
        MultipoleMomentFields(BASIS, second_moments=relabeled)


def test_primitive_components_are_not_reinterpreted_in_a_spherical_basis() -> None:
    model = moment_model(UNITS)
    with pytest.raises(ValueError):
        MultipoleMomentFields(
            "native-real-orthonormal",
            second_moments=model.reference(QUADRUPOLE),
            trace="zero",
        )


def test_unpublished_moment_ports_refuse_during_field_resolution() -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    second_port = model.port(QUADRUPOLE)
    assert second_port.dimensions is not None
    spherical = AtomisticFieldReference(
        PRODUCER,
        QUADRUPOLE,
        phx.ValuePort(
            second_port.semantic_id,
            event_shape=(5,),
            component_ids=("m-2", "m-1", "m0", "m1", "m2"),
            representation="atom-traceless-second-moment",
            dimensions=(second_port.dimensions[0],) * 5,
            frame_id=second_port.frame_id,
            normalization_id="native-real-orthonormal",
        ),
        level=AtomisticConditionLevel.ATOM,
        unit=model.unit(QUADRUPOLE),
    )
    unpublished = MultipoleMomentFields(
        "native-real-orthonormal",
        charges=model.reference(CHARGES),
        second_moments=spherical,
        trace="zero",
    )
    program = predicted_program(
        model, system, (multipole_consumer(model, system, "point", fields=unpublished),)
    )
    x = jnp.asarray(CLUSTER, dtype=jnp.float64)
    with pytest.raises(ValueError):
        program.evaluate(x, bundle_state(program, x))


def test_charge_units_cannot_label_an_atomic_dipole_port() -> None:
    model = moment_model(UNITS)
    dipole_port = model.port(DIPOLE)
    with pytest.raises(ValueError):
        AtomisticFieldReference(
            PRODUCER,
            DIPOLE,
            dipole_port,
            level=AtomisticConditionLevel.ATOM,
            unit=UNITS.charge_unit,
        )


def test_si_moment_fields_refuse_in_an_uncalibrated_reduced_system() -> None:
    model = moment_model(UNITS)
    reduced = cluster_system(units=AtomisticUnitSystem.reduced())
    with pytest.raises(ValueError):
        multipole_consumer(model, reduced, "point").prepare(reduced)


def test_volume_units_are_not_reinterpreted_as_physical_polarizability() -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    volume = derived_unit("angstrom^3", ((ANGSTROM, 3),))
    volume_reference = AtomisticFieldReference(
        PRODUCER,
        POLARIZABILITY,
        phx.ValuePort(
            "test.atom-polarizability-volume",
            event_shape=(),
            component_ids=("alpha_v",),
            representation="atom-polarizability-volume",
            dimensions=(volume.dimension,),
            normalization_id="isotropic-scalar",
        ),
        level=AtomisticConditionLevel.ATOM,
        unit=volume,
    )
    with pytest.raises(ValueError):
        polarization_consumer(
            model, system, "point", polarizability=volume_reference
        ).prepare(system)


def test_polarization_refuses_missing_moment_degrees() -> None:
    model = moment_model(UNITS)
    system = cluster_system()
    x = jnp.asarray(CLUSTER, dtype=jnp.float64)
    partial = MultipoleMomentFields(
        BASIS, charges=model.reference(CHARGES), dipoles=model.reference(DIPOLE)
    )
    program = predicted_program(
        model, system, (polarization_consumer(model, system, "point", fields=partial),)
    )
    with pytest.raises(ValueError):
        program.evaluate(x, bundle_state(program, x))
