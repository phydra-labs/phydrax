#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""General contracted Gaussian basis plans with Cartesian working functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....atomistic import AtomisticSystemPlan
from ....ein import contract
from ._shell import (
    cartesian_angular_exponents,
    cartesian_primitive_normalization,
    GaussianShellPlan,
    GaussianShellRepresentation,
    odd_double_factorial,
)
from ._spherical import cartesian_to_real_spherical


def _same_center_unnormalized_overlap(
    exponent_left: float,
    exponent_right: float,
    angular: tuple[int, int, int],
    /,
) -> float:
    power = exponent_left + exponent_right
    value = 1.0
    for component in angular:
        value *= (
            odd_double_factorial(2 * component - 1)
            * np.sqrt(np.pi)
            / (2.0**component * power ** (component + 0.5))
        )
    return value


def _normalized_contraction(
    exponents: np.ndarray,
    coefficients: np.ndarray,
    mask: np.ndarray,
    angular: tuple[int, int, int],
    /,
) -> np.ndarray:
    primitive_norm = np.asarray(
        cartesian_primitive_normalization(exponents, angular), dtype=np.float64
    )
    values = coefficients * primitive_norm * mask
    overlap = 0.0
    active = np.flatnonzero(mask)
    for left in active:
        for right in active:
            overlap += (
                values[left]
                * values[right]
                * _same_center_unnormalized_overlap(
                    float(exponents[left]), float(exponents[right]), angular
                )
            )
    if not np.isfinite(overlap) or overlap <= 0.0:
        raise ValueError("Gaussian contraction has non-positive self-overlap.")
    return values / np.sqrt(overlap)


class GaussianBasisPlan(StrictModule, NonTrainableState):
    """An ordered immutable shell basis keyed by stable nuclear particle IDs."""

    shells: tuple[GaussianShellPlan, ...]
    maximum_basis_functions: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_artifact_id: str | None = eqx.field(static=True)
    role: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        shells: Sequence[GaussianShellPlan],
        /,
        *,
        maximum_basis_functions: int = 512,
        source_id: str,
        source_artifact_id: str | None = None,
        role: str = "orbital",
    ):
        shells_ = tuple(shells)
        if not shells_ or any(
            not isinstance(value, GaussianShellPlan) for value in shells_
        ):
            raise TypeError("shells must contain GaussianShellPlan values.")
        role_ = str(role).strip()
        source = str(source_id).strip()
        artifact = None if source_artifact_id is None else str(source_artifact_id).strip()
        if not role_ or not source or source_artifact_id is not None and not artifact:
            raise ValueError("Basis role and source identities must be non-empty.")
        if any(shell.role != role_ for shell in shells_):
            raise ValueError("Every Gaussian shell role must match its basis role.")
        count = sum(shell.function_count for shell in shells_)
        capacity = int(maximum_basis_functions)
        if capacity <= 0 or count > capacity:
            raise ValueError("Gaussian basis exceeds maximum_basis_functions.")
        self.shells = shells_
        self.maximum_basis_functions = capacity
        self.source_id = source
        self.source_artifact_id = artifact
        self.role = role_
        self.basis_id = canonical_fingerprint(
            {
                "kind": "gaussian-basis",
                "shells": [value.shell_id for value in shells_],
                "maximum_basis_functions": capacity,
                "source": source,
                "source_artifact": artifact,
                "role": role_,
            }
        )

    @classmethod
    def from_contracted_s(
        cls,
        center_particle_ids: Sequence[int],
        exponents: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        primitive_mask: ArrayLike | None = None,
        maximum_basis_functions: int = 16,
        source_id: str,
    ) -> GaussianBasisPlan:
        centers = tuple(center_particle_ids)
        exponent = np.asarray(exponents, dtype=np.float64)
        coefficient = np.asarray(coefficients, dtype=np.float64)
        if (
            exponent.ndim != 2
            or coefficient.shape != exponent.shape
            or exponent.shape[0] != len(centers)
        ):
            raise ValueError(
                "Contracted-s centers, exponents, and coefficients must align."
            )
        mask = (
            np.ones(exponent.shape, dtype=np.bool_)
            if primitive_mask is None
            else np.asarray(primitive_mask, dtype=np.bool_)
        )
        if mask.shape != exponent.shape:
            raise ValueError("primitive_mask must align with contracted-s exponents.")
        return cls(
            tuple(
                GaussianShellPlan(
                    center,
                    0,
                    exponent[index],
                    coefficient[index],
                    primitive_mask=mask[index],
                )
                for index, center in enumerate(centers)
            ),
            maximum_basis_functions=maximum_basis_functions,
            source_id=source_id,
        )

    @classmethod
    def from_basis_exchange_record(
        cls,
        center_particle_ids: Sequence[int],
        atomic_numbers: Sequence[int],
        record: Mapping[str, object],
        /,
        *,
        representation: GaussianShellRepresentation = GaussianShellRepresentation.REAL_SPHERICAL,
        maximum_basis_functions: int = 4096,
        source_id: str,
        source_artifact_id: str,
        role: str = "orbital",
    ) -> GaussianBasisPlan:
        centers = tuple(center_particle_ids)
        numbers = tuple(atomic_numbers)
        if len(centers) != len(numbers) or not centers:
            raise ValueError("Basis-exchange centers and atomic numbers must align.")
        elements = record.get("elements")
        if not isinstance(elements, Mapping):
            raise TypeError("Basis-exchange record requires an elements mapping.")
        shells: list[GaussianShellPlan] = []
        for center, number in zip(centers, numbers, strict=True):
            element = elements.get(str(number))
            if not isinstance(element, Mapping):
                raise ValueError(f"Basis-exchange record lacks element {number}.")
            electron_shells = element.get("electron_shells")
            if not isinstance(electron_shells, Sequence):
                raise TypeError("Basis-exchange element requires electron_shells.")
            for shell_record in electron_shells:
                if not isinstance(shell_record, Mapping):
                    raise TypeError("Basis-exchange shell records must be mappings.")
                angular_values = tuple(shell_record["angular_momentum"])
                exponents_ = np.asarray(shell_record["exponents"], dtype=np.float64)
                coefficient_rows = np.asarray(
                    shell_record["coefficients"], dtype=np.float64
                )
                if (
                    coefficient_rows.ndim != 2
                    or coefficient_rows.shape[1] != exponents_.size
                ):
                    raise ValueError("Basis-exchange shell coefficient rows are invalid.")
                if not angular_values or coefficient_rows.shape[0] % len(angular_values):
                    raise ValueError(
                        "Basis-exchange angular momenta and contractions do not align."
                    )
                contractions_per_angular = coefficient_rows.shape[0] // len(
                    angular_values
                )
                for angular_index, angular in enumerate(angular_values):
                    start = angular_index * contractions_per_angular
                    stop = start + contractions_per_angular
                    shells.append(
                        GaussianShellPlan(
                            center,
                            angular,
                            exponents_,
                            coefficient_rows[start:stop],
                            representation=representation,
                            role=role,
                        )
                    )
        return cls(
            shells,
            maximum_basis_functions=maximum_basis_functions,
            source_id=source_id,
            source_artifact_id=source_artifact_id,
            role=role,
        )

    def prepare(self, system: AtomisticSystemPlan, /) -> PreparedGaussianBasis:
        return PreparedGaussianBasis(self, system)


class PreparedGaussianBasis(StrictModule, NonTrainableState):
    """Cartesian working basis plus a fixed transform into requested AO functions."""

    plan: GaussianBasisPlan
    center_indices: Array
    shell_indices: Array
    contraction_indices: Array
    angular_exponents: Array
    angular_tuples: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    exponents: Array
    normalized_coefficients: Array
    primitive_mask: Array
    transformation: Array
    output_center_indices: Array
    system_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GaussianBasisPlan, system: AtomisticSystemPlan, /):
        if not isinstance(plan, GaussianBasisPlan):
            raise TypeError("plan must be GaussianBasisPlan.")
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        available = {
            int(particle_id): index
            for index, (particle_id, active, element) in enumerate(
                zip(
                    np.asarray(system.particle_ids),
                    np.asarray(system.active_mask),
                    np.asarray(system.element_mask),
                    strict=True,
                )
            )
            if bool(active) and bool(element)
        }
        if any(shell.center_particle_id not in available for shell in plan.shells):
            raise ValueError(
                "Gaussian basis references an inactive or non-element center."
            )
        maximum_primitives = max(shell.primitive_count for shell in plan.shells)
        centers: list[int] = []
        shell_indices: list[int] = []
        contractions: list[int] = []
        angular_rows: list[tuple[int, int, int]] = []
        exponent_rows: list[np.ndarray] = []
        coefficient_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        output_centers: list[int] = []
        cartesian_offset = 0
        output_offset = 0
        transform_blocks: list[tuple[int, int, np.ndarray]] = []
        for shell_index, shell in enumerate(plan.shells):
            center_index = available[shell.center_particle_id]
            components = cartesian_angular_exponents(shell.angular_momentum)
            exponents_ = np.asarray(shell.exponents)
            primitive_mask = np.asarray(shell.primitive_mask)
            for contraction_index, raw_coefficients in enumerate(
                np.asarray(shell.coefficients)
            ):
                block_start = cartesian_offset
                for angular in components:
                    padded_exponents = np.ones(
                        (maximum_primitives,), dtype=exponents_.dtype
                    )
                    padded_coefficients = np.zeros(
                        (maximum_primitives,), dtype=exponents_.dtype
                    )
                    padded_mask = np.zeros((maximum_primitives,), dtype=np.bool_)
                    padded_exponents[: shell.primitive_count] = exponents_
                    padded_coefficients[: shell.primitive_count] = (
                        _normalized_contraction(
                            exponents_, raw_coefficients, primitive_mask, angular
                        )
                    )
                    padded_mask[: shell.primitive_count] = primitive_mask
                    centers.append(center_index)
                    shell_indices.append(shell_index)
                    contractions.append(contraction_index)
                    angular_rows.append(angular)
                    exponent_rows.append(padded_exponents)
                    coefficient_rows.append(padded_coefficients)
                    mask_rows.append(padded_mask)
                    cartesian_offset += 1
                local = (
                    np.eye(len(components))
                    if shell.representation is GaussianShellRepresentation.CARTESIAN
                    else cartesian_to_real_spherical(shell.angular_momentum)
                )
                transform_blocks.append((block_start, output_offset, local))
                output_centers.extend([center_index] * local.shape[1])
                output_offset += local.shape[1]
        transform = np.zeros((cartesian_offset, output_offset), dtype=np.float64)
        for cartesian_start, output_start, local in transform_blocks:
            transform[
                cartesian_start : cartesian_start + local.shape[0],
                output_start : output_start + local.shape[1],
            ] = local
        self.plan = plan
        self.center_indices = jnp.asarray(centers, dtype=jnp.int32)
        self.shell_indices = jnp.asarray(shell_indices, dtype=jnp.int32)
        self.contraction_indices = jnp.asarray(contractions, dtype=jnp.int32)
        self.angular_exponents = jnp.asarray(angular_rows, dtype=jnp.int32)
        self.angular_tuples = tuple(angular_rows)
        self.exponents = jnp.asarray(np.stack(exponent_rows))
        self.normalized_coefficients = jnp.asarray(np.stack(coefficient_rows))
        self.primitive_mask = jnp.asarray(np.stack(mask_rows))
        self.transformation = jnp.asarray(transform, dtype=self.exponents.dtype)
        self.output_center_indices = jnp.asarray(output_centers, dtype=jnp.int32)
        self.system_id = system.system_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-basis",
                "basis": plan.basis_id,
                "system": system.system_id,
                "arrays": array_tree_fingerprint(
                    {
                        "center_indices": np.asarray(centers),
                        "shell_indices": np.asarray(shell_indices),
                        "contraction_indices": np.asarray(contractions),
                        "angular_exponents": np.asarray(angular_rows),
                        "exponents": np.stack(exponent_rows),
                        "normalized_coefficients": np.stack(coefficient_rows),
                        "primitive_mask": np.stack(mask_rows),
                        "transformation": transform,
                    }
                ),
            }
        )

    @property
    def cartesian_basis_function_count(self) -> int:
        return self.center_indices.size

    @property
    def basis_function_count(self) -> int:
        return self.transformation.shape[1]

    @property
    def maximum_angular_momentum(self) -> int:
        return max(shell.angular_momentum for shell in self.plan.shells)

    def transform_one_body(self, values: ArrayLike, /) -> Array:
        matrix = jnp.asarray(values)
        return contract(
            "pa,...pq,qb->...ab", self.transformation, matrix, self.transformation
        )

    def transform_two_body(self, values: ArrayLike, /) -> Array:
        tensor = jnp.asarray(values)
        transform = self.transformation
        return contract(
            "pa,qb,rc,sd,pqrs->abcd",
            transform,
            transform,
            transform,
            transform,
            tensor,
        )


__all__ = ["GaussianBasisPlan", "PreparedGaussianBasis"]
