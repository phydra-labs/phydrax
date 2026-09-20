#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fuzzy-sphere matrix algebra, Laplacian, harmonics, and scalar matrix models."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class FuzzySphereMatrixGeometryPlan(StrictModule):
    """Matrix algebra Mat(N) with one fixed irreducible SU(2) realization."""

    twice_spin: int = eqx.field(static=True)
    matrix_dimension: int = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        twice_spin: int,
        /,
        *,
        maximum_matrix_elements: int = 4_000_000,
    ):
        spin = int(twice_spin)
        dimension = spin + 1
        maximum = int(maximum_matrix_elements)
        if spin < 1 or dimension**4 > maximum:
            raise ValueError("Fuzzy matrix spin is invalid or exceeds matrix admission.")
        self.twice_spin = spin
        self.matrix_dimension = dimension
        self.maximum_matrix_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuzzy-sphere-matrix-geometry-plan",
                "twice_spin": spin,
                "maximum_matrix_elements": maximum,
            }
        )


class FuzzySphereMatrixGeometryEvidence(StrictModule):
    commutator_residual: Array
    radius_residual: Array
    laplacian_self_adjoint_residual: Array
    harmonic_eigenvalues: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class PreparedFuzzySphereMatrixGeometry(StrictModule):
    plan: FuzzySphereMatrixGeometryPlan = eqx.field(static=True)
    generators: Array
    coordinates: Array
    laplacian_matrix: Array
    harmonic_eigenvalues: Array
    harmonic_matrices: Array
    evidence: FuzzySphereMatrixGeometryEvidence
    prepared_id: str = eqx.field(static=True)

    def laplacian(self, matrix: ArrayLike, /) -> Array:
        value = jnp.asarray(matrix, dtype=self.generators.dtype)
        expected = (self.plan.matrix_dimension, self.plan.matrix_dimension)
        if value.shape != expected:
            raise ValueError("Fuzzy matrix has the wrong shape.")
        result = jnp.zeros_like(value)
        for generator in self.generators:
            first = generator @ value - value @ generator
            result = result + generator @ first - first @ generator
        return result

    def trace_integral(self, matrix: ArrayLike, /) -> Array:
        value = jnp.asarray(matrix, dtype=self.generators.dtype)
        return 4.0 * jnp.pi / self.plan.matrix_dimension * jnp.trace(value)

    def berezin_symbol(self, density_matrices: ArrayLike, matrix: ArrayLike, /) -> Array:
        densities = jnp.asarray(density_matrices, dtype=self.generators.dtype)
        value = jnp.asarray(matrix, dtype=self.generators.dtype)
        if densities.shape[-2:] != value.shape:
            raise ValueError("Berezin density matrices and observable dimensions differ.")
        return ein.contract("...ij,ji->...", densities, value)


def prepare_fuzzy_sphere_matrix_geometry(
    plan: FuzzySphereMatrixGeometryPlan,
    /,
) -> PreparedFuzzySphereMatrixGeometry:
    if not isinstance(plan, FuzzySphereMatrixGeometryPlan):
        raise TypeError("plan must be FuzzySphereMatrixGeometryPlan.")
    dimension = plan.matrix_dimension
    spin = 0.5 * plan.twice_spin
    magnetic = np.arange(dimension, dtype=np.float64) - spin
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for index, value in enumerate(magnetic[:-1]):
        raising[index + 1, index] = math.sqrt((spin - value) * (spin + value + 1.0))
    lowering = raising.conj().T
    x = 0.5 * (raising + lowering)
    y = (raising - lowering) / (2.0j)
    z = np.diag(magnetic)
    generators = np.stack((x, y, z), axis=0)
    scale = math.sqrt(spin * (spin + 1.0))
    coordinates = generators / scale
    epsilon_terms = (
        (0, 1, 2, 1.0),
        (1, 2, 0, 1.0),
        (2, 0, 1, 1.0),
        (1, 0, 2, -1.0),
        (2, 1, 0, -1.0),
        (0, 2, 1, -1.0),
    )
    commutator = 0.0
    for first, second, output, sign in epsilon_terms:
        residual = (
            coordinates[first] @ coordinates[second]
            - coordinates[second] @ coordinates[first]
            - 1j * sign / scale * coordinates[output]
        )
        commutator = max(commutator, float(np.linalg.norm(residual)))
    radius = sum(value @ value for value in coordinates)
    radius_residual = float(np.linalg.norm(radius - np.eye(dimension)))
    basis = np.eye(dimension * dimension, dtype=np.complex128)
    columns = []
    for index in range(dimension * dimension):
        matrix = basis[:, index].reshape((dimension, dimension))
        result = np.zeros_like(matrix)
        for generator in generators:
            first = generator @ matrix - matrix @ generator
            result += generator @ first - first @ generator
        columns.append(result.reshape(-1))
    laplacian = np.stack(columns, axis=1)
    laplacian = 0.5 * (laplacian + laplacian.conj().T)
    self_adjoint = float(np.linalg.norm(laplacian - laplacian.conj().T))
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    harmonics = eigenvectors.T.reshape((-1, dimension, dimension))
    accepted = max(commutator, radius_residual, self_adjoint) <= 1e-10
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-sphere-matrix-geometry-evidence",
            "plan": plan.plan_id,
            "commutator_residual": commutator,
            "radius_residual": radius_residual,
            "laplacian_self_adjoint_residual": self_adjoint,
            "harmonic_eigenvalues": array_tree_fingerprint(eigenvalues),
        }
    )
    evidence = FuzzySphereMatrixGeometryEvidence(
        commutator_residual=jnp.asarray(commutator),
        radius_residual=jnp.asarray(radius_residual),
        laplacian_self_adjoint_residual=jnp.asarray(self_adjoint),
        harmonic_eigenvalues=jnp.asarray(eigenvalues),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-fuzzy-sphere-matrix-geometry",
            "plan": plan.plan_id,
            "generators": array_tree_fingerprint(generators),
            "laplacian": array_tree_fingerprint(laplacian),
        }
    )
    return PreparedFuzzySphereMatrixGeometry(
        plan=plan,
        generators=jnp.asarray(generators),
        coordinates=jnp.asarray(coordinates),
        laplacian_matrix=jnp.asarray(laplacian),
        harmonic_eigenvalues=jnp.asarray(eigenvalues),
        harmonic_matrices=jnp.asarray(harmonics),
        evidence=evidence,
        prepared_id=prepared_id,
    )


class FuzzyScalarMatrixModelPlan(StrictModule):
    geometry: PreparedFuzzySphereMatrixGeometry
    mass_squared: float = eqx.field(static=True)
    quartic_coupling: float = eqx.field(static=True)
    kinetic_coupling: float = eqx.field(static=True)
    proposal_scale: float = eqx.field(static=True)
    draws: int = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: PreparedFuzzySphereMatrixGeometry,
        /,
        *,
        mass_squared: float,
        quartic_coupling: float,
        kinetic_coupling: float = 1.0,
        proposal_scale: float = 0.05,
        draws: int = 100,
        burn_in: int = 20,
    ):
        if not isinstance(geometry, PreparedFuzzySphereMatrixGeometry):
            raise TypeError("geometry must be PreparedFuzzySphereMatrixGeometry.")
        mass = float(mass_squared)
        quartic = float(quartic_coupling)
        kinetic = float(kinetic_coupling)
        scale = float(proposal_scale)
        draws_ = int(draws)
        burn = int(burn_in)
        if not all(math.isfinite(value) for value in (mass, quartic, kinetic, scale)):
            raise ValueError("Fuzzy matrix-model controls must be finite.")
        if quartic < 0.0 or kinetic < 0.0 or scale <= 0.0 or draws_ < 1 or burn < 0:
            raise ValueError("Fuzzy matrix-model controls are outside their support.")
        content = {
            "kind": "fuzzy-scalar-matrix-model-plan",
            "geometry": geometry.prepared_id,
            "mass_squared": mass,
            "quartic_coupling": quartic,
            "kinetic_coupling": kinetic,
            "proposal_scale": scale,
            "draws": draws_,
            "burn_in": burn,
        }
        self.geometry = geometry
        self.mass_squared = mass
        self.quartic_coupling = quartic
        self.kinetic_coupling = kinetic
        self.proposal_scale = scale
        self.draws = draws_
        self.burn_in = burn
        self.plan_id = canonical_fingerprint(content)

    def action(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field, dtype=self.geometry.generators.dtype)
        if value.shape != (
            self.geometry.plan.matrix_dimension,
            self.geometry.plan.matrix_dimension,
        ):
            raise ValueError("Fuzzy scalar field has the wrong matrix shape.")
        hermitian = 0.5 * (value + jnp.conj(value.T))
        laplacian = self.geometry.laplacian(hermitian)
        kinetic = 0.5 * self.kinetic_coupling * jnp.real(jnp.trace(hermitian @ laplacian))
        quadratic = 0.5 * self.mass_squared * jnp.real(jnp.trace(hermitian @ hermitian))
        quartic = (
            0.25
            * self.quartic_coupling
            * jnp.real(jnp.trace(hermitian @ hermitian @ hermitian @ hermitian))
        )
        return kinetic + quadratic + quartic


class FuzzyScalarMatrixModelRun(StrictModule):
    fields: Array
    actions: Array
    trace_observables: Array
    acceptance_rate: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def sample_fuzzy_scalar_matrix_model(
    plan: FuzzyScalarMatrixModelPlan,
    initial_field: ArrayLike,
    key: Key[Array, ""],
    /,
) -> FuzzyScalarMatrixModelRun:
    """Run a bounded Hermitian random-walk Metropolis matrix model."""

    if not isinstance(plan, FuzzyScalarMatrixModelPlan):
        raise TypeError("plan must be FuzzyScalarMatrixModelPlan.")
    field = jnp.asarray(initial_field, dtype=plan.geometry.generators.dtype)
    field = 0.5 * (field + jnp.conj(field.T))
    action = plan.action(field)
    samples = []
    actions = []
    traces = []
    accepts = []
    for index in range(plan.draws + plan.burn_in):
        key, real_key, imaginary_key, accept_key = jr.split(key, 4)
        real = jr.normal(real_key, field.shape, dtype=jnp.real(field).dtype)
        imaginary = jr.normal(imaginary_key, field.shape, dtype=jnp.real(field).dtype)
        noise = real + 1j * imaginary
        noise = 0.5 * (noise + jnp.conj(noise.T))
        proposal = field + plan.proposal_scale * noise
        proposal_action = plan.action(proposal)
        accepted = jnp.log(jr.uniform(accept_key)) < -(proposal_action - action)
        field = jnp.where(accepted, proposal, field)
        action = jnp.where(accepted, proposal_action, action)
        accepts.append(accepted)
        if index >= plan.burn_in:
            samples.append(field)
            actions.append(action)
            traces.append(
                jnp.asarray(
                    (
                        jnp.real(jnp.trace(field)) / field.shape[0],
                        jnp.real(jnp.trace(field @ field)) / field.shape[0],
                    )
                )
            )
    fields = jnp.stack(tuple(samples))
    action_values = jnp.stack(tuple(actions))
    trace_values = jnp.stack(tuple(traces))
    finite = (
        jnp.all(jnp.isfinite(fields))
        & jnp.all(jnp.isfinite(action_values))
        & jnp.all(jnp.isfinite(trace_values))
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-scalar-matrix-model-run",
            "plan": plan.plan_id,
            "draws": plan.draws,
            "burn_in": plan.burn_in,
        }
    )
    return FuzzyScalarMatrixModelRun(
        fields=fields,
        actions=action_values,
        trace_observables=trace_values,
        acceptance_rate=jnp.mean(jnp.asarray(accepts, dtype=jnp.float64)),
        finite=finite,
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


__all__ = [
    "FuzzyScalarMatrixModelPlan",
    "FuzzyScalarMatrixModelRun",
    "FuzzySphereMatrixGeometryEvidence",
    "FuzzySphereMatrixGeometryPlan",
    "PreparedFuzzySphereMatrixGeometry",
    "prepare_fuzzy_sphere_matrix_geometry",
    "sample_fuzzy_scalar_matrix_model",
]
