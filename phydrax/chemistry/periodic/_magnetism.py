#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.periodic import (
    PeriodicOrbitalBasisPlan,
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
    PreparedPeriodicTranslationFamily,
)
from ...sparse import EdgeRelation


class SpinorBasisConvention(StrictModule):
    """Orbital-major ``(orbital 0 up, orbital 0 down, ...)`` spin convention."""

    orbital_basis: PeriodicOrbitalBasisPlan
    spinor_labels: tuple[str, ...] = eqx.field(static=True)
    spin_eigenvalues: tuple[int, ...] = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self, orbital_basis: PeriodicOrbitalBasisPlan, /):
        if not isinstance(orbital_basis, PeriodicOrbitalBasisPlan):
            raise TypeError("orbital_basis must be PeriodicOrbitalBasisPlan.")
        if orbital_basis.spin_order != "spinless":
            raise ValueError("L·S assembly requires an undoubled spinless orbital basis.")
        labels = tuple(
            label
            for orbital in orbital_basis.labels
            for label in (f"{orbital}:up", f"{orbital}:down")
        )
        self.orbital_basis = orbital_basis
        self.spinor_labels = labels
        self.spin_eigenvalues = tuple(
            value for _ in orbital_basis.labels for value in (1, -1)
        )
        self.convention_id = canonical_fingerprint(
            {
                "kind": "spinor-basis-convention",
                "orbital_basis": orbital_basis.basis_id,
                "layout": "orbital-major-up-down",
                "spin_operator": "S=sigma/2",
            }
        )

    @property
    def orbital_count(self) -> int:
        return self.orbital_basis.orbital_count

    @property
    def spinor_count(self) -> int:
        return 2 * self.orbital_count


class SpinOrbitCouplingPlan(StrictModule):
    """Supplied dimensionless orbital L matrices for native ``lambda L·sigma/2``."""

    convention: SpinorBasisConvention
    orbital_angular_momentum: Array
    require_angular_momentum_algebra: bool = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    algebra_tolerance: float = eqx.field(static=True)
    maximum_spinor_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        convention: SpinorBasisConvention,
        orbital_angular_momentum: ArrayLike,
        /,
        *,
        source_id: str,
        maximum_spinor_count: int,
        require_angular_momentum_algebra: bool = True,
        hermiticity_tolerance: float = 1.0e-10,
        algebra_tolerance: float = 1.0e-9,
    ):
        if not isinstance(convention, SpinorBasisConvention):
            raise TypeError("convention must be SpinorBasisConvention.")
        angular = np.asarray(orbital_angular_momentum, dtype=np.complex128)
        expected = (3, convention.orbital_count, convention.orbital_count)
        if angular.shape != expected or np.any(~np.isfinite(angular)):
            raise ValueError(
                f"Orbital angular momentum must be finite with shape {expected}."
            )
        hermitian_tolerance = float(hermiticity_tolerance)
        algebra_tolerance_ = float(algebra_tolerance)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (hermitian_tolerance, algebra_tolerance_)
        ):
            raise ValueError("SOC tolerances must be finite and positive.")
        scale = max(float(np.max(np.abs(angular), initial=0.0)), 1.0)
        hermitian_defect = np.max(
            np.abs(angular - np.conj(np.swapaxes(angular, -1, -2))), initial=0.0
        )
        if hermitian_defect > hermitian_tolerance * scale:
            raise ValueError(
                "Each supplied orbital angular-momentum matrix must be Hermitian."
            )
        if require_angular_momentum_algebra:
            residuals = []
            for left, right, target in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                residuals.append(
                    np.max(
                        np.abs(
                            angular[left] @ angular[right]
                            - angular[right] @ angular[left]
                            - 1.0j * angular[target]
                        ),
                        initial=0.0,
                    )
                )
            if max(residuals) > algebra_tolerance_ * scale:
                raise ValueError(
                    "Supplied L matrices do not satisfy [Lx,Ly]=iLz cyclically."
                )
        maximum = int(maximum_spinor_count)
        if maximum < convention.spinor_count:
            raise ValueError("SOC spinor dimension exceeds the explicit resource policy.")
        source = str(source_id).strip()
        if not source:
            raise ValueError("SOC source_id must be non-empty.")
        self.convention = convention
        self.orbital_angular_momentum = jnp.asarray(angular)
        self.require_angular_momentum_algebra = bool(require_angular_momentum_algebra)
        self.hermiticity_tolerance = hermitian_tolerance
        self.algebra_tolerance = algebra_tolerance_
        self.maximum_spinor_count = maximum
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-orbit-coupling-plan",
                "convention": convention.convention_id,
                "source": source,
                "maximum_spinor_count": maximum,
                "require_algebra": bool(require_angular_momentum_algebra),
                "angular_momentum": array_tree_fingerprint(angular),
            }
        )


class PreparedSpinOrbitOperator(StrictModule):
    plan: SpinOrbitCouplingPlan
    coupling_strength: Array
    onsite_matrix: Array
    family: PreparedPeriodicTranslationFamily
    hermiticity_residual: Array
    angular_momentum_residual: Array
    prepared_id: str = eqx.field(static=True)


def _pauli_half(dtype) -> Array:
    return jnp.asarray(
        (
            ((0.0, 0.5), (0.5, 0.0)),
            ((0.0, -0.5j), (0.5j, 0.0)),
            ((0.5, 0.0), (0.0, -0.5)),
        ),
        dtype=dtype,
    )


def prepare_spin_orbit_operator(
    plan: SpinOrbitCouplingPlan,
    coupling_strength: ArrayLike,
    /,
) -> PreparedSpinOrbitOperator:
    """Assemble SOC as one onsite canonical periodic translation family."""

    if not isinstance(plan, SpinOrbitCouplingPlan):
        raise TypeError("plan must be SpinOrbitCouplingPlan.")
    coupling = np.asarray(coupling_strength)
    if coupling.shape != () or not np.isfinite(coupling) or not np.isrealobj(coupling):
        raise ValueError("SOC coupling strength lambda must be one finite real energy.")
    angular = plan.orbital_angular_momentum
    pauli = _pauli_half(angular.dtype)
    onsite = jnp.sum(
        jnp.stack(
            tuple(jnp.kron(angular[axis], pauli[axis]) for axis in range(3)), axis=0
        ),
        axis=0,
    ) * jnp.asarray(coupling)
    count = plan.convention.spinor_count
    target = np.repeat(np.arange(count, dtype=np.int32), count)
    source = np.tile(np.arange(count, dtype=np.int32), count)
    reverse = (
        np.arange(count * count, dtype=np.int32).reshape((count, count)).T.reshape((-1,))
    )
    relation = EdgeRelation(source, target, source_size=count, target_size=count)
    family_plan = PeriodicTranslationFamilyPlan(
        relation,
        np.zeros(
            (count * count, plan.convention.orbital_basis.cell.rank), dtype=np.int32
        ),
        reverse,
        hermitian=True,
        maximum_dense_entries=count * count,
    )
    values = np.asarray(onsite)[target, source].reshape((-1, 1, 1))
    family_state = PeriodicTranslationFamilyState(
        family_plan, values, hermiticity_tolerance=plan.hermiticity_tolerance
    )
    family = prepare_periodic_translation_family(family_plan, family_state)
    hermiticity = jnp.max(jnp.abs(onsite - jnp.conj(onsite.T)), initial=0.0)
    algebra_residuals = tuple(
        jnp.max(
            jnp.abs(
                angular[left] @ angular[right]
                - angular[right] @ angular[left]
                - 1.0j * angular[target_axis]
            ),
            initial=0.0,
        )
        for left, right, target_axis in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    )
    algebra_residual = jnp.max(jnp.stack(algebra_residuals), initial=0.0)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-spin-orbit-operator",
            "plan": plan.plan_id,
            "coupling": float(coupling),
            "family": family.prepared_id,
        }
    )
    return PreparedSpinOrbitOperator(
        plan,
        jnp.asarray(coupling),
        onsite,
        family,
        hermiticity,
        algebra_residual,
        prepared_id,
    )


class SpinResolvedBandObservablePlan(StrictModule):
    convention: SpinorBasisConvention
    degeneracy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        convention: SpinorBasisConvention,
        /,
        *,
        degeneracy_tolerance: float = 1.0e-8,
    ):
        if not isinstance(convention, SpinorBasisConvention):
            raise TypeError("convention must be SpinorBasisConvention.")
        tolerance = float(degeneracy_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("degeneracy_tolerance must be finite and positive.")
        self.convention = convention
        self.degeneracy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-resolved-band-observable-plan",
                "convention": convention.convention_id,
                "degeneracy_tolerance": tolerance,
            }
        )


class SpinResolvedBandObservableResult(StrictModule):
    spin_expectations: Array
    projected_spin_matrices: Array
    degenerate_pair_mask: Array
    spin_bound_residual: Array
    metric_normalization_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_spin_resolved_bands(
    plan: SpinResolvedBandObservablePlan,
    energies: ArrayLike,
    coefficients: ArrayLike,
    /,
    *,
    overlaps: ArrayLike | None = None,
) -> SpinResolvedBandObservableResult:
    """Return scalar isolated-band spins and covariant degenerate-cluster matrices."""

    if not isinstance(plan, SpinResolvedBandObservablePlan):
        raise TypeError("plan must be SpinResolvedBandObservablePlan.")
    energy = jnp.asarray(energies)
    vectors = jnp.asarray(coefficients)
    count = plan.convention.spinor_count
    if energy.ndim != 2 or vectors.shape != (energy.shape[0], count, energy.shape[1]):
        raise ValueError("Spin-resolved energies/coefficients have incompatible shapes.")
    if overlaps is None:
        metric = jnp.broadcast_to(
            jnp.eye(count, dtype=vectors.dtype), (energy.shape[0], count, count)
        )
    else:
        metric = jnp.asarray(overlaps)
        if metric.shape != (energy.shape[0], count, count):
            raise ValueError("Spin-resolved overlap matrices have the wrong shape.")
    orbital_identity = jnp.eye(plan.convention.orbital_count, dtype=vectors.dtype)
    pauli = _pauli_half(vectors.dtype)
    spin_operators = jnp.stack(
        tuple(jnp.kron(orbital_identity, pauli[axis]) for axis in range(3)), axis=0
    )
    metric_spin = ein.contract("kij,ajl->kail", metric, spin_operators)
    projected = ein.contract(
        "kib,kail,klc->kabc", jnp.conj(vectors), metric_spin, vectors
    )
    projected = jnp.moveaxis(projected, 1, -1)
    expectations = jnp.real(jnp.diagonal(projected, axis1=1, axis2=2))
    expectations = jnp.moveaxis(expectations, 1, -1)
    differences = jnp.abs(energy[:, :, None] - energy[:, None, :])
    degenerate = differences <= plan.degeneracy_tolerance
    projected = jnp.where(degenerate[..., None], projected, 0.0)
    gram = ein.contract("kib,kij,kjc->kbc", jnp.conj(vectors), metric, vectors)
    identity = jnp.eye(energy.shape[1], dtype=gram.dtype)
    metric_residual = jnp.max(jnp.abs(gram - identity), initial=0.0)
    magnitudes = jnp.sqrt(jnp.sum(expectations * expectations, axis=-1))
    bound_residual = jnp.max(jnp.maximum(magnitudes - 0.5, 0.0), initial=0.0)
    successful = (
        jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(vectors))
        & (metric_residual <= 10.0 * plan.degeneracy_tolerance)
        & (bound_residual <= 10.0 * plan.degeneracy_tolerance)
    )
    return SpinResolvedBandObservableResult(
        expectations,
        projected,
        degenerate,
        bound_residual,
        metric_residual,
        successful,
        plan.plan_id,
    )


__all__ = [
    "PreparedSpinOrbitOperator",
    "SpinOrbitCouplingPlan",
    "SpinResolvedBandObservablePlan",
    "SpinResolvedBandObservableResult",
    "SpinorBasisConvention",
    "evaluate_spin_resolved_bands",
    "prepare_spin_orbit_operator",
]
