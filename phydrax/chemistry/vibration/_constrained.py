#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Holonomic molecular constraints and Lagrangian normal modes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticSystemPlan
from ...linalg import (
    ConstraintOperatorPlan,
    DenseLinearOperator,
    OperatorProperties,
    RankPolicy,
)
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from .._optimization import _require_structure_matches_system
from .._units import angular_frequency_to_wavenumber
from ..electronic_structure._derivatives import MolecularHessianResult
from ._harmonic import StationaryPointKind, VibrationalAnalysisResult


ConstraintEvaluator = Callable[[Array], Array]


def _particle_indices(
    system: AtomisticSystemPlan, particle_ids: Sequence[int]
) -> tuple[int, ...]:
    available = {
        int(particle_id): index
        for index, particle_id in enumerate(np.asarray(system.particle_ids))
        if bool(np.asarray(system.active_mask)[index])
    }
    requested = tuple(particle_ids)
    if any(value not in available for value in requested):
        raise ValueError("Constraint references an inactive or unknown particle ID.")
    return tuple(available[value] for value in requested)


class MolecularConstraintSetPlan(StrictModule, NonTrainableState):
    """Dimensionless holonomic residuals with exact callable identity."""

    evaluator: ConstraintEvaluator
    constraint_count: int = eqx.field(static=True)
    physical: bool = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ConstraintEvaluator,
        constraint_count: int,
        definition_id: str,
        /,
        *,
        physical: bool,
    ):
        if not callable(evaluator):
            raise TypeError("constraint evaluator must be callable.")
        count = int(constraint_count)
        definition = str(definition_id).strip()
        if count <= 0 or not definition:
            raise ValueError(
                "Constraint count and definition ID must be positive/non-empty."
            )
        self.evaluator = evaluator
        self.constraint_count = count
        self.physical = bool(physical)
        self.definition_id = definition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-constraint-set",
                "definition": definition,
                "constraint_count": count,
                "physical": bool(physical),
            }
        )

    def residual(self, positions: ArrayLike, /) -> Array:
        value = jnp.asarray(self.evaluator(jnp.asarray(positions)))
        if value.shape != (self.constraint_count,):
            raise ValueError(
                f"Constraint evaluator must return shape ({self.constraint_count},)."
            )
        return value

    @classmethod
    def distances(
        cls,
        system: AtomisticSystemPlan,
        particle_id_pairs: Sequence[tuple[int, int]],
        targets: Sequence[float],
        /,
        *,
        physical: bool = True,
    ) -> MolecularConstraintSetPlan:
        pairs = tuple((int(left), int(right)) for left, right in particle_id_pairs)
        targets_ = np.asarray(tuple(float(value) for value in targets))
        if not pairs or targets_.shape != (len(pairs),):
            raise ValueError("Distance pairs and targets must be non-empty and aligned.")
        if np.any(~np.isfinite(targets_)) or np.any(targets_ <= 0.0):
            raise ValueError("Distance targets must be finite and positive.")
        flat_ids = tuple(value for pair in pairs for value in pair)
        flat_indices = _particle_indices(system, flat_ids)
        indices = np.asarray(flat_indices, dtype=np.int32).reshape((-1, 2))
        target_array = jnp.asarray(targets_, dtype=np.dtype(system.coordinate_dtype))

        def evaluate(positions: Array) -> Array:
            displacement = positions[indices[:, 1]] - positions[indices[:, 0]]
            distances = jnp.sqrt(jnp.sum(displacement * displacement, axis=1))
            return (distances - target_array) / target_array

        definition = canonical_fingerprint(
            {
                "kind": "distance-constraints",
                "system": system.system_id,
                "pairs": [list(pair) for pair in pairs],
                "targets": targets_.tolist(),
                "length_unit": system.units.scale.length_unit.unit_id,
            }
        )
        return cls(evaluate, len(pairs), definition, physical=physical)

    @classmethod
    def cartesian(
        cls,
        system: AtomisticSystemPlan,
        coordinates: Sequence[tuple[int, int]],
        targets: Sequence[float],
        /,
        *,
        scale: float = 1.0,
        physical: bool = False,
    ) -> MolecularConstraintSetPlan:
        entries = tuple(
            (int(particle_id), int(component)) for particle_id, component in coordinates
        )
        if not entries or any(component < 0 or component > 2 for _, component in entries):
            raise ValueError("Cartesian constraints require particle/component entries.")
        targets_ = np.asarray(tuple(float(value) for value in targets))
        if targets_.shape != (len(entries),) or np.any(~np.isfinite(targets_)):
            raise ValueError("Cartesian targets must be finite and aligned.")
        scale_ = float(scale)
        if not isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("Cartesian constraint scale must be finite and positive.")
        indices = np.asarray(
            _particle_indices(system, tuple(value[0] for value in entries)),
            dtype=np.int32,
        )
        components = np.asarray(tuple(value[1] for value in entries), dtype=np.int32)
        target_array = jnp.asarray(targets_, dtype=np.dtype(system.coordinate_dtype))

        def evaluate(positions: Array) -> Array:
            return (positions[indices, components] - target_array) / scale_

        definition = canonical_fingerprint(
            {
                "kind": "cartesian-constraints",
                "system": system.system_id,
                "entries": [list(value) for value in entries],
                "targets": targets_.tolist(),
                "scale": scale_,
                "length_unit": system.units.scale.length_unit.unit_id,
            }
        )
        return cls(evaluate, len(entries), definition, physical=physical)

    @classmethod
    def angles(
        cls,
        system: AtomisticSystemPlan,
        particle_id_triplets: Sequence[tuple[int, int, int]],
        targets: Sequence[float],
        /,
        *,
        physical: bool = True,
    ) -> MolecularConstraintSetPlan:
        triplets = tuple(tuple(entry) for entry in particle_id_triplets)
        targets_ = np.asarray(tuple(float(value) for value in targets))
        if not triplets or targets_.shape != (len(triplets),):
            raise ValueError("Angle triplets and targets must be non-empty and aligned.")
        if np.any(~np.isfinite(targets_)) or np.any(
            (targets_ <= 0.0) | (targets_ >= np.pi)
        ):
            raise ValueError("Angle targets must lie strictly between zero and pi.")
        flat = _particle_indices(
            system, tuple(value for entry in triplets for value in entry)
        )
        indices = np.asarray(flat, dtype=np.int32).reshape((-1, 3))
        target_array = jnp.asarray(targets_, dtype=np.dtype(system.coordinate_dtype))

        def evaluate(positions: Array) -> Array:
            left = positions[indices[:, 0]] - positions[indices[:, 1]]
            right = positions[indices[:, 2]] - positions[indices[:, 1]]
            denominator = jnp.sqrt(
                jnp.sum(left * left, axis=1) * jnp.sum(right * right, axis=1)
            )
            cosine = jnp.clip(jnp.sum(left * right, axis=1) / denominator, -1.0, 1.0)
            return jnp.arccos(cosine) - target_array

        definition = canonical_fingerprint(
            {
                "kind": "angle-constraints",
                "system": system.system_id,
                "triplets": [list(value) for value in triplets],
                "targets": targets_.tolist(),
            }
        )
        return cls(evaluate, len(triplets), definition, physical=physical)

    @classmethod
    def dihedrals(
        cls,
        system: AtomisticSystemPlan,
        particle_id_quartets: Sequence[tuple[int, int, int, int]],
        targets: Sequence[float],
        /,
        *,
        physical: bool = True,
    ) -> MolecularConstraintSetPlan:
        quartets = tuple(tuple(entry) for entry in particle_id_quartets)
        targets_ = np.asarray(tuple(float(value) for value in targets))
        if not quartets or targets_.shape != (len(quartets),):
            raise ValueError(
                "Dihedral quartets and targets must be non-empty and aligned."
            )
        if np.any(~np.isfinite(targets_)):
            raise ValueError("Dihedral targets must be finite.")
        flat = _particle_indices(
            system, tuple(value for entry in quartets for value in entry)
        )
        indices = np.asarray(flat, dtype=np.int32).reshape((-1, 4))
        target_array = jnp.asarray(targets_, dtype=np.dtype(system.coordinate_dtype))

        def evaluate(positions: Array) -> Array:
            b0 = positions[indices[:, 1]] - positions[indices[:, 0]]
            b1 = positions[indices[:, 2]] - positions[indices[:, 1]]
            b2 = positions[indices[:, 3]] - positions[indices[:, 2]]
            b1_unit = b1 / jnp.sqrt(jnp.sum(b1 * b1, axis=1))[:, None]
            v = b0 - jnp.sum(b0 * b1_unit, axis=1)[:, None] * b1_unit
            w = b2 - jnp.sum(b2 * b1_unit, axis=1)[:, None] * b1_unit
            x = jnp.sum(v * w, axis=1)
            y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=1)
            angle = jnp.arctan2(y, x)
            difference = angle - target_array
            return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))

        definition = canonical_fingerprint(
            {
                "kind": "dihedral-constraints",
                "system": system.system_id,
                "quartets": [list(value) for value in quartets],
                "targets": targets_.tolist(),
            }
        )
        return cls(evaluate, len(quartets), definition, physical=physical)

    @classmethod
    def combine(
        cls, plans: Sequence[MolecularConstraintSetPlan], /
    ) -> MolecularConstraintSetPlan:
        plans_ = tuple(plans)
        if not plans_ or any(
            not isinstance(plan, MolecularConstraintSetPlan) for plan in plans_
        ):
            raise TypeError("plans must contain MolecularConstraintSetPlan values.")

        def evaluate(positions: Array) -> Array:
            return jnp.concatenate(tuple(plan.residual(positions) for plan in plans_))

        definition = canonical_fingerprint(
            {
                "kind": "combined-molecular-constraints",
                "plans": [plan.plan_id for plan in plans_],
            }
        )
        return cls(
            evaluate,
            sum(plan.constraint_count for plan in plans_),
            definition,
            physical=all(plan.physical for plan in plans_),
        )


class ConstrainedVibrationalAnalysisResult(StrictModule, NonTrainableState):
    vibration: VibrationalAnalysisResult
    constraint_residual: Array
    tangent_gradient_residual: Array
    multiplier_residual: Array
    rigid_tangent_residual: Array
    lagrange_multipliers: Array
    constraint_rank: int = eqx.field(static=True)
    rigid_mode_rank: int = eqx.field(static=True)
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        vibration: VibrationalAnalysisResult,
        constraint_residual: ArrayLike,
        tangent_gradient_residual: ArrayLike,
        multiplier_residual: ArrayLike,
        rigid_tangent_residual: ArrayLike,
        lagrange_multipliers: ArrayLike,
        /,
        *,
        constraint_rank: int,
        rigid_mode_rank: int,
        successful: ArrayLike,
        plan_id: str,
    ):
        if not isinstance(vibration, VibrationalAnalysisResult):
            raise TypeError("vibration must be VibrationalAnalysisResult.")
        dtype = vibration.eigenvalues.dtype
        constraint_residual_ = jnp.asarray(constraint_residual, dtype=dtype).reshape(())
        tangent_residual = jnp.asarray(tangent_gradient_residual, dtype=dtype).reshape(())
        multiplier_residual_ = jnp.asarray(multiplier_residual, dtype=dtype).reshape(())
        multipliers = jnp.asarray(lagrange_multipliers, dtype=dtype)
        rigid_residual = jnp.asarray(
            rigid_tangent_residual,
            dtype=dtype,
        ).reshape(())
        successful_ = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.vibration = vibration
        self.constraint_residual = constraint_residual_
        self.tangent_gradient_residual = tangent_residual
        self.multiplier_residual = multiplier_residual_
        self.lagrange_multipliers = multipliers
        self.rigid_tangent_residual = rigid_residual
        self.constraint_rank = int(constraint_rank)
        self.rigid_mode_rank = int(rigid_mode_rank)
        self.successful = successful_
        self.result_id = canonical_fingerprint(
            {
                "kind": "constrained-vibrational-analysis-result",
                "plan": str(plan_id),
                "vibration": vibration.result_id,
                "constraint_rank": int(constraint_rank),
                "rigid_mode_rank": int(rigid_mode_rank),
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "constraint_residual": np.asarray(constraint_residual_),
                        "tangent_gradient_residual": np.asarray(tangent_residual),
                        "multiplier_residual": np.asarray(multiplier_residual_),
                        "lagrange_multipliers": np.asarray(multipliers),
                        "rigid_tangent_residual": np.asarray(rigid_residual),
                    }
                ),
            }
        )


class ConstrainedVibrationalAnalysisPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    constraints: MolecularConstraintSetPlan
    rank_tolerance: float = eqx.field(static=True)
    stationarity_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    imaginary_wavenumber_threshold: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        constraints: MolecularConstraintSetPlan,
        /,
        *,
        rank_tolerance: float = 1.0e-9,
        stationarity_tolerance: float = 1.0e-7,
        constraint_tolerance: float = 1.0e-8,
        imaginary_wavenumber_threshold: float = 10.0,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(constraints, MolecularConstraintSetPlan):
            raise TypeError("constraints must be MolecularConstraintSetPlan.")
        values = tuple(
            float(value)
            for value in (
                rank_tolerance,
                stationarity_tolerance,
                constraint_tolerance,
                imaginary_wavenumber_threshold,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Constrained vibration tolerances must be finite and positive."
            )
        self.system = system
        self.constraints = constraints
        (
            self.rank_tolerance,
            self.stationarity_tolerance,
            self.constraint_tolerance,
            self.imaginary_wavenumber_threshold,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constrained-vibrational-analysis-plan",
                "system": system.system_id,
                "constraints": constraints.plan_id,
                "tolerances": list(values),
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        hessian: MolecularHessianResult,
        forces: ArrayLike,
        /,
    ) -> ConstrainedVibrationalAnalysisResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        if not isinstance(hessian, MolecularHessianResult):
            raise TypeError("hessian must be MolecularHessianResult.")
        _require_structure_matches_system(structure, self.system)
        active = np.asarray(self.system.active_mask, dtype=np.bool_)
        indices = np.flatnonzero(active)
        positions = jnp.asarray(structure.positions)
        force = np.asarray(forces)
        if force.shape != positions.shape:
            raise ValueError("forces must align with structure positions.")
        base = positions[active].reshape(-1)

        def residual_from_active(flat: Array) -> Array:
            full = positions.at[indices].set(flat.reshape((-1, 3)))
            return self.constraints.residual(full)

        residual = np.asarray(residual_from_active(base))
        jacobian = np.asarray(jax.jacrev(residual_from_active)(base))
        constraint_hessians = np.asarray(
            jax.jacfwd(jax.jacrev(residual_from_active))(base)
        )
        masses = np.asarray(self.system.masses)[active]
        root_mass = np.sqrt(masses)
        inverse_root_mass = np.repeat(1.0 / root_mass, 3)
        constraint_matrix = jacobian * inverse_root_mass[None, :]
        constraint_operator = DenseLinearOperator(jnp.asarray(constraint_matrix))
        prepared = ConstraintOperatorPlan(
            constraint_operator,
            require_full_row_rank=True,
            rank=RankPolicy(
                relative_cutoff=self.rank_tolerance,
                absolute_cutoff=self.rank_tolerance,
                require_full_rank=True,
            ),
            factorization_kind="svd",
        ).prepare()
        tangent = np.asarray(prepared.nullspace_basis)
        gradient = -force[active].reshape(-1)
        mass_gradient = inverse_root_mass * gradient
        multipliers = np.asarray(
            prepared.right_inverse_adjoint(jnp.asarray(mass_gradient))
        )
        normal_gradient = constraint_matrix.T @ multipliers
        multiplier_residual = float(
            np.max(np.abs(mass_gradient - normal_gradient), initial=0.0)
        )
        tangent_gradient_residual = float(
            np.max(np.abs(tangent.T @ mass_gradient), initial=0.0)
        )
        lagrangian_hessian = np.asarray(hessian.hessian)[
            np.ix_(indices, np.arange(3), indices, np.arange(3))
        ].reshape((base.size, base.size)) - np.tensordot(
            multipliers, constraint_hessians, axes=(0, 0)
        )
        mass_hessian = (
            inverse_root_mass[:, None] * lagrangian_hessian * inverse_root_mass[None, :]
        )
        centered = np.array(structure.positions, copy=True)[active]
        centered -= np.sum(masses[:, None] * centered, axis=0) / np.sum(masses)
        rigid = np.zeros((base.size, 6), dtype=centered.dtype)
        axes = np.eye(3, dtype=centered.dtype)
        for axis in range(3):
            rigid[:, axis] = (root_mass[:, None] * axes[axis]).reshape(-1)
            rigid[:, 3 + axis] = (
                root_mass[:, None] * np.cross(axes[axis], centered)
            ).reshape(-1)
        rigid_left, rigid_singular, _ = np.linalg.svd(
            rigid,
            full_matrices=False,
        )
        rigid_scale = float(rigid_singular[0]) if rigid_singular.size else 1.0
        rigid_threshold = max(
            self.rank_tolerance,
            self.rank_tolerance * rigid_scale,
        )
        independent_rigid_rank = int(np.count_nonzero(rigid_singular > rigid_threshold))
        independent_rigid = rigid_left[:, :independent_rigid_rank]
        if independent_rigid_rank:
            constraint_on_rigid = constraint_matrix @ independent_rigid
            _, constrained_singular, constrained_right = np.linalg.svd(
                constraint_on_rigid,
                full_matrices=True,
            )
            constrained_scale = (
                float(constrained_singular[0]) if constrained_singular.size else 1.0
            )
            constrained_threshold = max(
                self.rank_tolerance,
                self.rank_tolerance * constrained_scale,
            )
            constrained_rigid_rank = int(
                np.count_nonzero(constrained_singular > constrained_threshold)
            )
            rigid_tangent = (
                independent_rigid @ constrained_right[constrained_rigid_rank:].T
            )
        else:
            rigid_tangent = np.zeros((base.size, 0), dtype=centered.dtype)
        rigid_coordinates = tangent.T @ rigid_tangent
        left, singular, _ = np.linalg.svd(
            rigid_coordinates,
            full_matrices=True,
        )
        scale = float(singular[0]) if singular.size else 1.0
        rigid_coordinate_threshold = max(
            self.rank_tolerance,
            self.rank_tolerance * scale,
        )
        rigid_rank = int(np.count_nonzero(singular > rigid_coordinate_threshold))
        internal_basis = tangent @ left[:, rigid_rank:]
        reduced = internal_basis.T @ mass_hessian @ internal_basis
        count = reduced.shape[0]
        if count:
            solve = eigensolve(
                Eigenproblem(
                    DenseLinearOperator(
                        jnp.asarray(0.5 * (reduced + reduced.T)),
                        properties=OperatorProperties(
                            self_adjoint=True,
                            evidence={"self_adjoint": "construction"},
                        ),
                    )
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(), count=count, which="smallest-algebraic"
                ),
            )
            eigenvalues = np.asarray(solve.eigenvalues)
            vectors = internal_basis @ np.asarray(solve.eigenvectors)
            cartesian = inverse_root_mass[:, None] * vectors
            inverse_reduced_mass = np.sum(cartesian * cartesian, axis=0)
            reduced_masses = 1.0 / inverse_reduced_mass
            normalized = cartesian * np.sqrt(reduced_masses)[None, :]
            modes = np.zeros((active.size, 3, count), dtype=normalized.dtype)
            modes[active] = normalized.reshape((indices.size, 3, count))
            omega_squared = eigenvalues * self.system.units.force_to_momentum_rate
            angular = np.sign(omega_squared) * np.sqrt(np.abs(omega_squared))
            wavenumbers = np.asarray(
                angular_frequency_to_wavenumber(angular, self.system.units)
            )
            eigen_successful = bool(solve.successful)
        else:
            dtype = np.asarray(structure.positions).dtype
            eigenvalues = np.zeros((0,), dtype=dtype)
            angular = np.zeros((0,), dtype=dtype)
            wavenumbers = np.zeros((0,), dtype=dtype)
            reduced_masses = np.zeros((0,), dtype=dtype)
            modes = np.zeros((active.size, 3, 0), dtype=dtype)
            eigen_successful = True
        imaginary = wavenumbers < -self.imaginary_wavenumber_threshold
        constraint_residual = float(np.max(np.abs(residual), initial=0.0))
        rigid_tangent_residual = float(
            np.max(
                np.abs(constraint_matrix @ rigid_tangent),
                initial=0.0,
            )
        )
        constraint_projection_residual = float(
            np.max(
                np.abs(constraint_matrix @ internal_basis),
                initial=0.0,
            )
        )
        rigid_projection_residual = float(
            np.max(
                np.abs(rigid_tangent.T @ internal_basis),
                initial=0.0,
            )
        )
        external_projection_residual = max(
            constraint_projection_residual,
            rigid_projection_residual,
        )
        successful = (
            bool(hessian.successful)
            and bool(prepared.evidence.full_row_rank)
            and eigen_successful
            and constraint_residual <= self.constraint_tolerance
            and tangent_gradient_residual <= self.stationarity_tolerance
            and multiplier_residual <= self.stationarity_tolerance
            and np.all(np.isfinite(eigenvalues))
            and rigid_tangent_residual <= self.constraint_tolerance
            and external_projection_residual <= self.constraint_tolerance
        )
        imaginary_count = int(np.count_nonzero(imaginary))
        stationary = (
            StationaryPointKind.INCONCLUSIVE
            if not successful
            else StationaryPointKind.MINIMUM
            if imaginary_count == 0
            else StationaryPointKind.FIRST_ORDER_SADDLE
            if imaginary_count == 1
            else StationaryPointKind.HIGHER_ORDER_SADDLE
        )
        vibration = VibrationalAnalysisResult(
            eigenvalues,
            angular,
            wavenumbers,
            imaginary,
            modes,
            reduced_masses,
            external_mode_count=rigid_rank,
            external_projection_residual=external_projection_residual,
            successful=successful,
            stationary_point=stationary,
            units=self.system.units,
            plan_id=self.plan_id,
        )
        return ConstrainedVibrationalAnalysisResult(
            vibration,
            constraint_residual,
            tangent_gradient_residual,
            multiplier_residual,
            rigid_tangent_residual,
            multipliers,
            constraint_rank=int(prepared.evidence.rank),
            rigid_mode_rank=rigid_rank,
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = [
    "ConstrainedVibrationalAnalysisPlan",
    "ConstrainedVibrationalAnalysisResult",
    "MolecularConstraintSetPlan",
]
