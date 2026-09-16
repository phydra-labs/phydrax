#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded dense-reference fermionic BFSS rational hybrid Monte Carlo."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._actions import BFSSConfiguration, BFSSPlan, prepare_bfss, PreparedBFSSAction


def _mul(left: Array, right: Array, /) -> Array:
    return ein.contract("...ij,...jk->...ik", left, right)


class BFSSFermionPlan(StrictModule):
    """Finite BFSS Majorana operator and rational determinant controls."""

    bosonic_plan: BFSSPlan = eqx.field(static=True)
    gamma_matrices: Array
    charge_conjugation: Array
    temporal_boundary_phase: complex = eqx.field(static=True)
    fermion_mass: float = eqx.field(static=True)
    regulator_mass: float = eqx.field(static=True)
    determinant_power: float = eqx.field(static=True)
    spectral_lower: float = eqx.field(static=True)
    spectral_upper: float = eqx.field(static=True)
    rational_poles: int = eqx.field(static=True)
    rational_verification_points: int = eqx.field(static=True)
    maximum_rational_error: float = eqx.field(static=True)
    maximum_fermion_dimension: int = eqx.field(static=True)
    antisymmetry_tolerance: float = eqx.field(static=True)
    spinor_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bosonic_plan: BFSSPlan,
        gamma_matrices: ArrayLike,
        charge_conjugation: ArrayLike,
        /,
        *,
        temporal_boundary_phase: complex = -1.0,
        fermion_mass: float = 0.0,
        regulator_mass: float = 1e-3,
        determinant_power: float = 0.25,
        spectral_lower: float,
        spectral_upper: float,
        rational_poles: int = 12,
        rational_verification_points: int = 4097,
        maximum_rational_error: float = 1e-5,
        maximum_fermion_dimension: int = 4096,
        antisymmetry_tolerance: float = 1e-9,
    ):
        if not isinstance(bosonic_plan, BFSSPlan):
            raise TypeError("bosonic_plan must be BFSSPlan.")
        gamma = np.asarray(gamma_matrices, dtype=np.complex128)
        charge = np.asarray(charge_conjugation, dtype=np.complex128)
        if gamma.ndim != 3 or gamma.shape[0] != bosonic_plan.matrix_count:
            raise ValueError(
                "gamma_matrices must have shape (matrix_count, spinor, spinor)."
            )
        spinor = int(gamma.shape[1])
        if gamma.shape[2] != spinor or charge.shape != (spinor, spinor):
            raise ValueError(
                "Gamma and charge-conjugation matrices have incompatible shapes."
            )
        clifford = 0.0
        identity = np.eye(spinor, dtype=np.complex128)
        for first in range(bosonic_plan.matrix_count):
            for second in range(bosonic_plan.matrix_count):
                expected = 2.0 * identity if first == second else 0.0
                clifford = max(
                    clifford,
                    float(
                        np.linalg.norm(
                            gamma[first] @ gamma[second]
                            + gamma[second] @ gamma[first]
                            - expected
                        )
                    ),
                )
        tolerance = float(antisymmetry_tolerance)
        if clifford > tolerance:
            raise ValueError("Gamma matrices violate the declared Clifford algebra.")
        boundary = complex(temporal_boundary_phase)
        if not np.isfinite(boundary) or abs(abs(boundary) - 1.0) > tolerance:
            raise ValueError("Temporal fermion boundary phase must have unit magnitude.")
        mass = float(fermion_mass)
        regulator = float(regulator_mass)
        power = float(determinant_power)
        lower = float(spectral_lower)
        upper = float(spectral_upper)
        poles = int(rational_poles)
        verification = int(rational_verification_points)
        maximum_error = float(maximum_rational_error)
        maximum_dimension = int(maximum_fermion_dimension)
        dimension = (
            bosonic_plan.time_slices
            * spinor
            * bosonic_plan.matrix_rank
            * bosonic_plan.matrix_rank
        )
        if any(
            not math.isfinite(value)
            for value in (mass, regulator, power, lower, upper, maximum_error, tolerance)
        ):
            raise ValueError("BFSS fermion scalar controls must be finite.")
        if (
            regulator <= 0.0
            or power <= 0.0
            or lower <= 0.0
            or upper <= lower
            or poles < 1
            or verification < max(129, 16 * (poles + 2))
            or maximum_error <= 0.0
            or dimension > maximum_dimension
        ):
            raise ValueError("BFSS fermion spectral or resource controls are invalid.")
        content = {
            "kind": "bfss-fermion-plan",
            "bosonic_plan": bosonic_plan.plan_id,
            "gamma_matrices": array_tree_fingerprint(gamma),
            "charge_conjugation": array_tree_fingerprint(charge),
            "temporal_boundary_phase": (boundary.real, boundary.imag),
            "fermion_mass": mass,
            "regulator_mass": regulator,
            "determinant_power": power,
            "spectral_lower": lower,
            "spectral_upper": upper,
            "rational_poles": poles,
            "rational_verification_points": verification,
            "maximum_rational_error": maximum_error,
            "maximum_fermion_dimension": maximum_dimension,
            "antisymmetry_tolerance": tolerance,
        }
        self.bosonic_plan = bosonic_plan
        self.gamma_matrices = jnp.asarray(gamma)
        self.charge_conjugation = jnp.asarray(charge)
        self.temporal_boundary_phase = boundary
        self.fermion_mass = mass
        self.regulator_mass = regulator
        self.determinant_power = power
        self.spectral_lower = lower
        self.spectral_upper = upper
        self.rational_poles = poles
        self.rational_verification_points = verification
        self.maximum_rational_error = maximum_error
        self.maximum_fermion_dimension = maximum_dimension
        self.antisymmetry_tolerance = tolerance
        self.spinor_dimension = spinor
        self.plan_id = canonical_fingerprint(content)

    @property
    def fermion_shape(self) -> tuple[int, int, int, int]:
        return (
            self.bosonic_plan.time_slices,
            self.spinor_dimension,
            self.bosonic_plan.matrix_rank,
            self.bosonic_plan.matrix_rank,
        )

    @property
    def fermion_dimension(self) -> int:
        return math.prod(self.fermion_shape)


class BFSSFermionOperator(StrictModule):
    plan: BFSSFermionPlan
    configuration: BFSSConfiguration
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BFSSFermionPlan,
        configuration: BFSSConfiguration,
        /,
    ):
        if not isinstance(plan, BFSSFermionPlan):
            raise TypeError("plan must be BFSSFermionPlan.")
        if not isinstance(configuration, BFSSConfiguration):
            raise TypeError("configuration must be BFSSConfiguration.")
        expected = (
            plan.bosonic_plan.time_slices,
            plan.bosonic_plan.matrix_count,
            plan.bosonic_plan.matrix_rank,
            plan.bosonic_plan.matrix_rank,
        )
        if configuration.matrices.shape != expected:
            raise ValueError("BFSS configuration does not match the fermion plan.")
        self.plan = plan
        self.configuration = configuration
        self.operator_id = canonical_fingerprint(
            {
                "kind": "bfss-fermion-operator",
                "plan": plan.plan_id,
                "shape": expected,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        fermion = jnp.asarray(value, dtype=self.configuration.matrices.dtype)
        if fermion.shape != self.plan.fermion_shape:
            raise ValueError("BFSS fermion field has the wrong shape.")
        forward_shift = jnp.roll(fermion, -1, axis=0)
        backward_shift = jnp.roll(fermion, 1, axis=0)
        forward_factor = jnp.ones(
            (self.plan.bosonic_plan.time_slices,), dtype=fermion.dtype
        )
        forward_factor = forward_factor.at[-1].set(self.plan.temporal_boundary_phase)
        backward_factor = jnp.ones_like(forward_factor)
        backward_factor = backward_factor.at[0].set(
            jnp.conj(jnp.asarray(self.plan.temporal_boundary_phase))
        )
        forward_shift = forward_shift * forward_factor[:, None, None, None]
        backward_shift = backward_shift * backward_factor[:, None, None, None]
        links = self.configuration.temporal_links[:, None]
        reverse = self.configuration.reverse_temporal_links[:, None]
        backward_links = jnp.roll(links, 1, axis=0)
        backward_reverse = jnp.roll(reverse, 1, axis=0)
        forward = _mul(_mul(links, forward_shift), reverse)
        backward = _mul(_mul(backward_reverse, backward_shift), backward_links)
        derivative = (forward - backward) / (2.0 * self.plan.bosonic_plan.time_spacing)
        interaction = jnp.zeros_like(fermion)
        averaged = 0.5 * (self.configuration.matrices + self.configuration.dual_matrices)
        for component in range(self.plan.bosonic_plan.matrix_count):
            matrix = averaged[:, component, None]
            commutator = _mul(matrix, fermion) - _mul(fermion, matrix)
            interaction = interaction + ein.contract(
                "ab,tbij->taij",
                self.plan.gamma_matrices[component],
                commutator,
            )
        raw = derivative + interaction + self.plan.fermion_mass * fermion
        return ein.contract("ab,tbij->taij", self.plan.charge_conjugation, raw)

    def materialize(self) -> Array:
        identity = jnp.eye(
            self.plan.fermion_dimension,
            dtype=self.configuration.matrices.dtype,
        )
        columns = []
        for index in range(self.plan.fermion_dimension):
            field = identity[:, index].reshape(self.plan.fermion_shape)
            columns.append(self.mv(field).reshape(-1))
        return jnp.stack(tuple(columns), axis=1)

    def normal_matrix(self) -> Array:
        matrix = self.materialize()
        identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
        return jnp.conj(matrix.T) @ matrix + self.plan.regulator_mass**2 * identity


class BFSSFermionAlgebraEvidence(StrictModule):
    antisymmetry_residual: Array
    normal_minimum_eigenvalue: Array
    normal_maximum_eigenvalue: Array
    interval_contains_spectrum: Array
    rational_maximum_relative_error: Array
    rational_positive: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class PreparedBFSSFermionRHMC(StrictModule):
    plan: BFSSFermionPlan
    bosonic_action: PreparedBFSSAction
    shifts: Array
    coefficients: Array
    evidence: BFSSFermionAlgebraEvidence
    prepared_id: str = eqx.field(static=True)


def _pack_configuration(configuration: BFSSConfiguration, /) -> Array:
    leaves = (
        configuration.matrices,
        configuration.dual_matrices,
        configuration.temporal_links,
        configuration.reverse_temporal_links,
    )
    return jnp.concatenate(
        tuple(
            part
            for leaf in leaves
            for part in (jnp.real(leaf).reshape(-1), jnp.imag(leaf).reshape(-1))
        )
    )


def _unpack_configuration(plan: BFSSPlan, coordinates: Array, /) -> BFSSConfiguration:
    matrix_shape = (
        plan.time_slices,
        plan.matrix_count,
        plan.matrix_rank,
        plan.matrix_rank,
    )
    link_shape = (plan.time_slices, plan.matrix_rank, plan.matrix_rank)
    matrix_size = math.prod(matrix_shape)
    link_size = math.prod(link_shape)
    cursor = 0
    values: list[Array] = []
    for shape, size in (
        (matrix_shape, matrix_size),
        (matrix_shape, matrix_size),
        (link_shape, link_size),
        (link_shape, link_size),
    ):
        real = coordinates[cursor : cursor + size].reshape(shape)
        cursor += size
        imaginary = coordinates[cursor : cursor + size].reshape(shape)
        cursor += size
        values.append(real + 1j * imaginary)
    if cursor != coordinates.size:
        raise ValueError("BFSS coordinate vector has the wrong size.")
    return BFSSConfiguration(*values)


def _fit_inverse_power_rational(
    plan: BFSSFermionPlan, /
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    shifts = np.geomspace(
        plan.spectral_lower / 16.0,
        plan.spectral_upper * 16.0,
        plan.rational_poles,
    )
    nodes = np.cos(
        np.pi
        * (np.arange(plan.rational_verification_points) + 0.5)
        / plan.rational_verification_points
    )
    points = plan.spectral_lower + 0.5 * (plan.spectral_upper - plan.spectral_lower) * (
        1.0 + nodes
    )
    target = points ** (-plan.determinant_power)
    basis = np.concatenate(
        (
            np.ones((points.size, 1)),
            1.0 / (points[:, None] + shifts[None, :]),
        ),
        axis=1,
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        basis / target[:, None], np.ones_like(target), rcond=None
    )
    fitted = basis @ coefficients
    relative = float(np.max(np.abs(fitted - target) / target))
    dense_points = np.geomspace(plan.spectral_lower, plan.spectral_upper, 8193)
    dense_basis = np.concatenate(
        (
            np.ones((dense_points.size, 1)),
            1.0 / (dense_points[:, None] + shifts[None, :]),
        ),
        axis=1,
    )
    positive = bool(np.all(dense_basis @ coefficients > 0.0))
    return shifts, coefficients, relative, positive


def prepare_bfss_fermion_rhmc(
    plan: BFSSFermionPlan,
    configuration: BFSSConfiguration,
    /,
) -> PreparedBFSSFermionRHMC:
    if not isinstance(plan, BFSSFermionPlan):
        raise TypeError("plan must be BFSSFermionPlan.")
    bosonic = prepare_bfss(plan.bosonic_plan)
    bosonic._validate(configuration)
    operator = BFSSFermionOperator(plan, configuration)
    matrix = np.asarray(operator.materialize())
    scale = max(1.0, float(np.max(np.abs(matrix))))
    antisymmetry = float(np.max(np.abs(matrix + matrix.T)) / scale)
    if antisymmetry > plan.antisymmetry_tolerance:
        raise ValueError("BFSS fermion matrix violates the antisymmetry tolerance.")
    eigenvalues = np.linalg.eigvalsh(np.asarray(operator.normal_matrix()))
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    contained = minimum >= plan.spectral_lower and maximum <= plan.spectral_upper
    if not contained:
        raise ValueError("BFSS normal spectrum leaves its declared interval.")
    shifts, coefficients, rational_error, positive = _fit_inverse_power_rational(plan)
    accepted = positive and rational_error <= plan.maximum_rational_error
    if not accepted:
        raise ValueError(
            "BFSS rational approximation fails its error or positivity gate."
        )
    evidence_id = canonical_fingerprint(
        {
            "kind": "bfss-fermion-algebra-evidence",
            "plan": plan.plan_id,
            "configuration": array_tree_fingerprint(_pack_configuration(configuration)),
            "antisymmetry": antisymmetry,
            "spectrum": (minimum, maximum),
            "rational_error": rational_error,
            "rational_positive": positive,
        }
    )
    evidence = BFSSFermionAlgebraEvidence(
        antisymmetry_residual=jnp.asarray(antisymmetry),
        normal_minimum_eigenvalue=jnp.asarray(minimum),
        normal_maximum_eigenvalue=jnp.asarray(maximum),
        interval_contains_spectrum=jnp.asarray(contained),
        rational_maximum_relative_error=jnp.asarray(rational_error),
        rational_positive=jnp.asarray(positive),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-bfss-fermion-rhmc",
            "plan": plan.plan_id,
            "bosonic": bosonic.prepared_id,
            "shifts": array_tree_fingerprint(shifts),
            "coefficients": array_tree_fingerprint(coefficients),
            "evidence": evidence_id,
        }
    )
    return PreparedBFSSFermionRHMC(
        plan=plan,
        bosonic_action=bosonic,
        shifts=jnp.asarray(shifts),
        coefficients=jnp.asarray(coefficients),
        evidence=evidence,
        prepared_id=prepared_id,
    )


def _rational_values(
    eigenvalues: Array,
    shifts: Array,
    coefficients: Array,
    /,
) -> Array:
    value = jnp.full_like(eigenvalues, coefficients[0])
    for index in range(shifts.shape[0]):
        value = value + coefficients[index + 1] / (eigenvalues + shifts[index])
    return value


def _pseudofermion_action(
    prepared: PreparedBFSSFermionRHMC,
    coordinates: Array,
    pseudofermion: Array,
    /,
) -> Array:
    configuration = _unpack_configuration(prepared.plan.bosonic_plan, coordinates)
    operator = BFSSFermionOperator(prepared.plan, configuration)
    normal = operator.normal_matrix()
    value = prepared.coefficients[0] * jnp.real(jnp.vdot(pseudofermion, pseudofermion))
    identity = jnp.eye(normal.shape[0], dtype=normal.dtype)
    for index in range(prepared.shifts.shape[0]):
        solution = jnp.linalg.solve(
            normal + prepared.shifts[index] * identity,
            pseudofermion,
        )
        value = value + prepared.coefficients[index + 1] * jnp.real(
            jnp.vdot(pseudofermion, solution)
        )
    return prepared.bosonic_action.action(configuration) + value


class BFSSFermionRHMCEvidence(StrictModule):
    acceptance_rate: Array
    maximum_energy_error: Array
    rational_maximum_relative_error: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class BFSSFermionRHMCRun(StrictModule):
    coordinates: Array
    accepted: Array
    energy_errors: Array
    evidence: BFSSFermionRHMCEvidence
    prepared_id: str = eqx.field(static=True)


def sample_bfss_fermion_rhmc(
    prepared: PreparedBFSSFermionRHMC,
    initial_configuration: BFSSConfiguration,
    key: Key[Array, ""],
    /,
    *,
    step_size: float,
    trajectory_steps: int,
    num_draws: int,
) -> BFSSFermionRHMCRun:
    """Run bounded exact-Metropolis HMC with a certified rational fermion action."""

    if not isinstance(prepared, PreparedBFSSFermionRHMC):
        raise TypeError("prepared must be PreparedBFSSFermionRHMC.")
    step = float(step_size)
    steps = int(trajectory_steps)
    draws = int(num_draws)
    if step <= 0.0 or steps < 1 or draws < 1:
        raise ValueError("BFSS RHMC trajectory controls must be positive.")
    coordinates = _pack_configuration(initial_configuration)
    samples: list[Array] = []
    accepted_values: list[Array] = []
    energy_errors: list[Array] = []
    for _ in range(draws):
        key, momentum_key, field_key, accept_key = jr.split(key, 4)
        configuration = _unpack_configuration(prepared.plan.bosonic_plan, coordinates)
        normal = BFSSFermionOperator(prepared.plan, configuration).normal_matrix()
        eigenvalues, eigenvectors = jnp.linalg.eigh(normal)
        rational = _rational_values(eigenvalues, prepared.shifts, prepared.coefficients)
        real = jr.normal(field_key, (normal.shape[0],), dtype=jnp.real(normal).dtype)
        field_key = jr.fold_in(field_key, 1)
        imaginary = jr.normal(field_key, (normal.shape[0],), dtype=jnp.real(normal).dtype)
        gaussian = (real + 1j * imaginary) / jnp.sqrt(2.0)
        pseudofermion = eigenvectors @ (
            (jnp.conj(eigenvectors.T) @ gaussian) / jnp.sqrt(rational)
        )
        momentum = jr.normal(momentum_key, coordinates.shape, dtype=coordinates.dtype)
        initial_energy = _pseudofermion_action(
            prepared, coordinates, pseudofermion
        ) + 0.5 * jnp.vdot(momentum, momentum)
        proposal = coordinates
        proposal_momentum = momentum - 0.5 * step * jax.grad(
            _pseudofermion_action,
            argnums=1,
        )(prepared, proposal, pseudofermion)
        for trajectory_index in range(steps):
            proposal = proposal + step * proposal_momentum
            if trajectory_index + 1 < steps:
                proposal_momentum = proposal_momentum - step * jax.grad(
                    _pseudofermion_action,
                    argnums=1,
                )(prepared, proposal, pseudofermion)
        proposal_momentum = proposal_momentum - 0.5 * step * jax.grad(
            _pseudofermion_action,
            argnums=1,
        )(prepared, proposal, pseudofermion)
        proposed_energy = _pseudofermion_action(
            prepared, proposal, pseudofermion
        ) + 0.5 * jnp.vdot(proposal_momentum, proposal_momentum)
        error = jnp.real(proposed_energy - initial_energy)
        accept = jnp.log(jr.uniform(accept_key)) < -error
        coordinates = jnp.where(accept, proposal, coordinates)
        samples.append(coordinates)
        accepted_values.append(accept)
        energy_errors.append(error)
    sample_table = jnp.stack(tuple(samples), axis=0)
    acceptance = jnp.stack(tuple(accepted_values))
    errors = jnp.stack(tuple(energy_errors))
    finite = jnp.all(jnp.isfinite(sample_table)) & jnp.all(jnp.isfinite(errors))
    evidence_id = canonical_fingerprint(
        {
            "kind": "bfss-fermion-rhmc-evidence",
            "prepared": prepared.prepared_id,
            "step_size": step,
            "trajectory_steps": steps,
            "num_draws": draws,
        }
    )
    evidence = BFSSFermionRHMCEvidence(
        acceptance_rate=jnp.mean(acceptance),
        maximum_energy_error=jnp.max(jnp.abs(errors)),
        rational_maximum_relative_error=prepared.evidence.rational_maximum_relative_error,
        finite=finite,
        successful=finite & jnp.any(acceptance),
        prepared_id=prepared.prepared_id,
        evidence_id=evidence_id,
    )
    return BFSSFermionRHMCRun(
        coordinates=sample_table,
        accepted=acceptance,
        energy_errors=errors,
        evidence=evidence,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "BFSSFermionAlgebraEvidence",
    "BFSSFermionOperator",
    "BFSSFermionPlan",
    "BFSSFermionRHMCEvidence",
    "BFSSFermionRHMCRun",
    "PreparedBFSSFermionRHMC",
    "prepare_bfss_fermion_rhmc",
    "sample_bfss_fermion_rhmc",
]
