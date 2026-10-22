#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit FP64 cuEST density-fitting execution, without vendor SCF or AD.

cuEST 0.2.2.2 contracts: https://docs.nvidia.com/cuda/cuest/ . Workspace
limits cover wrapper-owned vendor workspaces, not JAX arrays, driver/JIT memory,
or a claim about total process memory. Metric rank is not exposed by this API;
a fitting cutoff is not a rank or energy-error certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, NoReturn, TYPE_CHECKING
from weakref import ref, ReferenceType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from ...._cuda_native import cuda_partition_key, load_cuda_component
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._basis import PreparedGaussianBasis
from ._basis_layout import CuestBasisLayout
from ._integrals import nuclear_repulsion_energy


if TYPE_CHECKING:
    from ._provider import (
        MolecularDensityFactorizationEvidence,
        MolecularJKResult,
        MolecularOneElectronResult,
    )


@jax.custom_jvp
def _no_automatic_derivatives(value: Array, /) -> Array:
    return value


@_no_automatic_derivatives.defjvp
def _no_automatic_derivatives_jvp(
    primals: tuple[Array], tangents: tuple[Array], /
) -> NoReturn:
    raise NotImplementedError(
        "cuEST supplies contracted analytic first nuclear gradients only; "
        "arbitrary JVP/VJP, basis derivatives, and analytic Hessians are unsupported."
    )


def _fp64(value: Array, name: str, /) -> Array:
    if value.dtype != jnp.float64:
        raise ValueError(
            f"cuEST {name} requires real FP64 arrays and jax_enable_x64=True."
        )
    return _no_automatic_derivatives(value)


def _density_factors(
    density: Array, tolerance: float, candidate: Array | None = None, /
) -> tuple[Array, MolecularDensityFactorizationEvidence]:
    """Certify actual D under the explicit numerical PSD tolerance.

    A native eigensolve or a certified perturbation norm must establish the same
    lower bound -tolerance before admitting any factors. Reconstruction residual
    and a spectral lower bound are always reported; measured eigenvalues and
    negative magnitude remain NaN when the eigensolver was unnecessary.
    """
    from ._provider import MolecularDensityFactorizationEvidence

    tolerance_ = jnp.asarray(tolerance, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def evidence(
        factor: Array,
        residual: Array,
        minimum: Array,
        negative: Array,
        used_eigh: bool,
        lower_bound: Array,
        /,
    ) -> MolecularDensityFactorizationEvidence:
        return MolecularDensityFactorizationEvidence(
            minimum,
            residual,
            tolerance_,
            negative,
            jnp.asarray(factor.size * factor.dtype.itemsize, dtype=jnp.int64),
            jnp.asarray(used_eigh),
            lower_bound,
        )

    if not bool(jnp.all(jnp.isfinite(density))):
        raise ValueError("cuEST exchange requires a finite symmetric PSD density.")
    if not bool(jnp.max(jnp.abs(density - density.T), initial=0.0) <= tolerance_):
        raise ValueError(
            "cuEST exchange requires a symmetric density within its explicit tolerance."
        )
    if bool(jnp.all(density == 0.0)):
        factor = jnp.zeros((0, density.shape[0]), dtype=jnp.float64)
        return factor, evidence(factor, zero, zero, zero, False, zero)
    if candidate is not None:
        if candidate.ndim != 2 or candidate.shape[1] != density.shape[0]:
            raise ValueError("Density factors must have shape (rank, AO).")
        candidate = _fp64(candidate, "density factors")
        difference = candidate.T @ candidate - density
        residual = jnp.max(jnp.abs(difference), initial=0.0)
        symmetric_difference = 0.5 * (difference + difference.T)
        # Weyl's inequality: D_sym = F.T@F - E and lambda_min(F.T@F)>=0,
        # so lambda_min(D_sym)>=-||E||_2>=-||E||_F. An elementwise bound is
        # insufficient: n coherent small entries can hide a larger negative mode.
        spectral_bound = jnp.sqrt(jnp.sum(jnp.square(symmetric_difference)))
        if bool((residual <= tolerance_) & (spectral_bound <= tolerance_)):
            unknown = jnp.asarray(jnp.nan, dtype=jnp.float64)
            negative = jnp.where(spectral_bound == 0.0, zero, unknown)
            return candidate, evidence(
                candidate, residual, unknown, negative, False, -spectral_bound
            )
    result = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (density + density.T),
                properties=OperatorProperties(
                    self_adjoint=True, evidence={"self_adjoint": "construction"}
                ),
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(), count=density.shape[0], which="smallest-algebraic"
        ),
    )
    if not bool(result.successful):
        raise RuntimeError("Native actual-density PSD factorization failed.")
    minimum = jnp.min(result.eigenvalues)
    if not bool(minimum >= -tolerance_):
        raise ValueError(
            "Actual density is indefinite beyond density_factorization_tolerance; "
            "cuEST will not substitute the newest orbital density."
        )
    positive = jnp.where(result.eigenvalues >= 0.0, result.eigenvalues, 0.0)
    factor = jnp.sqrt(positive)[:, None] * result.eigenvectors.T
    residual = jnp.max(jnp.abs(factor.T @ factor - density), initial=0.0)
    if not bool(residual <= tolerance_):
        raise ValueError(
            "Native PSD factors fail the actual-density reconstruction tolerance."
        )
    return factor, evidence(
        factor, residual, minimum, jnp.maximum(-minimum, 0.0), True, minimum
    )


class CuestIntegralPlan(StrictModule, NonTrainableState):
    """Full-range DF-RHF integral policy with explicit scientific/resource choices."""

    auxiliary_basis: PreparedGaussianBasis
    fitting_cutoff: float = eqx.field(static=True)
    relative_conditioning: bool = eqx.field(static=True)
    fitting_algorithm: Literal["qr", "matrix-power"] = eqx.field(static=True)
    screening_threshold: float = eqx.field(static=True)
    variable_workspace_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_cached_geometries: int = eqx.field(static=True)
    density_factorization_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        auxiliary_basis: PreparedGaussianBasis,
        /,
        *,
        fitting_cutoff: float,
        relative_conditioning: bool,
        fitting_algorithm: Literal["qr", "matrix-power"],
        screening_threshold: float,
        variable_workspace_bytes: int,
        maximum_workspace_bytes: int,
        maximum_cached_geometries: int = 1,
        density_factorization_tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(auxiliary_basis, PreparedGaussianBasis):
            raise TypeError(
                "cuEST requires an explicit prepared auxiliary Gaussian basis."
            )
        cutoff, screening = float(fitting_cutoff), float(screening_threshold)
        if (
            not isfinite(cutoff)
            or cutoff <= 0.0
            or not isfinite(screening)
            or screening < 0.0
        ):
            raise ValueError(
                "cuEST fitting cutoff must be positive; screening nonnegative finite."
            )
        if not isinstance(relative_conditioning, bool):
            raise TypeError("relative_conditioning must be explicit bool.")
        if fitting_algorithm not in ("qr", "matrix-power"):
            raise ValueError("cuEST fitting_algorithm must be 'qr' or 'matrix-power'.")
        resources = (
            variable_workspace_bytes,
            maximum_workspace_bytes,
            maximum_cached_geometries,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resources
        ):
            raise ValueError(
                "cuEST workspace and cache bounds must be positive integer byte/count limits."
            )
        if variable_workspace_bytes > maximum_workspace_bytes:
            raise ValueError(
                "cuEST variable scratch cannot exceed owned workspace limit."
            )
        density_tolerance = float(density_factorization_tolerance)
        if not isfinite(density_tolerance) or density_tolerance <= 0.0:
            raise ValueError("density_factorization_tolerance must be positive finite.")
        self.auxiliary_basis = auxiliary_basis
        self.fitting_cutoff = cutoff
        self.relative_conditioning = relative_conditioning
        self.fitting_algorithm = fitting_algorithm
        self.screening_threshold = screening
        self.variable_workspace_bytes = variable_workspace_bytes
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.maximum_cached_geometries = maximum_cached_geometries
        self.density_factorization_tolerance = density_tolerance
        self.plan_id = self._identity()

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "cuest-integral-plan",
                "auxiliary_basis": self.auxiliary_basis.prepared_id,
                "math": "native-fp64",
                "fitting_cutoff": self.fitting_cutoff,
                "relative_conditioning": self.relative_conditioning,
                "fitting_algorithm": self.fitting_algorithm,
                "screening": self.screening_threshold,
                "variable_workspace_bytes": self.variable_workspace_bytes,
                "maximum_workspace_bytes": self.maximum_workspace_bytes,
                "maximum_cached_geometries": self.maximum_cached_geometries,
                "density_factorization_tolerance": self.density_factorization_tolerance,
                "operator": "full-range-coulomb",
            }
        )

    def validate_basis(self, basis: PreparedGaussianBasis, /) -> None:
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("cuEST requires a prepared Gaussian orbital basis.")
        if basis.system_id != self.auxiliary_basis.system_id:
            raise ValueError(
                "cuEST orbital and auxiliary bases belong to different systems."
            )
        CuestBasisLayout(basis)
        CuestBasisLayout(self.auxiliary_basis, auxiliary=True)

    def prepare(self, basis: PreparedGaussianBasis, /) -> PreparedCuestIntegrals:
        return PreparedCuestIntegrals(self, basis)

    def definition_id(self, basis: PreparedGaussianBasis, /) -> str:
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("cuEST requires a prepared Gaussian orbital basis.")
        if basis.system_id != self.auxiliary_basis.system_id:
            raise ValueError(
                "cuEST orbital and auxiliary bases belong to different systems."
            )
        return _cuest_definition_id(
            self,
            basis,
            CuestBasisLayout(basis),
            CuestBasisLayout(self.auxiliary_basis, auxiliary=True),
        )

    def one_electron(
        self, basis: PreparedGaussianBasis, positions: ArrayLike, charges: ArrayLike, /
    ) -> MolecularOneElectronResult:
        return self.prepare(basis).bind_geometry(positions, charges).one_electron()

    def jk(
        self, basis: PreparedGaussianBasis, positions: ArrayLike, density: ArrayLike, /
    ) -> MolecularJKResult:
        coordinate = jnp.asarray(positions)
        return (
            self.prepare(basis)
            .bind_geometry(
                coordinate, jnp.zeros((coordinate.shape[0],), dtype=jnp.float64)
            )
            .jk(density)
        )


def _cuest_definition_id(
    plan: CuestIntegralPlan,
    basis: PreparedGaussianBasis,
    orbital_layout: CuestBasisLayout,
    auxiliary_layout: CuestBasisLayout,
    /,
) -> str:
    """Bind the actual dynamic recipe, not stale owner labels, at preparation."""
    return canonical_fingerprint(
        {
            "kind": "cuest-molecular-integral-definition",
            "basis": basis.prepared_id,
            "policy": plan._identity(),
            "orbital_layout": orbital_layout.layout_id,
            "auxiliary_layout": auxiliary_layout.layout_id,
            "representation": (orbital_layout.pure, auxiliary_layout.pure),
            "arrays": array_tree_fingerprint(
                (plan, basis, orbital_layout, auxiliary_layout)
            ),
        }
    )


def _geometry_key(definition_id: str, positions: Array, /) -> Array:
    identifier = canonical_fingerprint(
        {
            "kind": "cuest-bound-geometry",
            "definition": definition_id,
            "positions": array_tree_fingerprint(np.asarray(positions, dtype=np.float64)),
        }
    )
    values = np.frombuffer(bytes.fromhex(identifier), dtype=np.dtype(">i8")).astype(
        np.int64
    )
    return jnp.asarray(values, dtype=jnp.int64)


def _binding_arrays(
    prepared: PreparedCuestIntegrals, positions: Array, charges: Array, key: Array, /
) -> tuple[Array, ...]:
    leaves = jax.tree_util.tree_leaves((prepared, positions, charges, key))
    arrays: list[Array] = []
    for leaf in leaves:
        if isinstance(leaf, Tracer):
            raise NotImplementedError(
                "cuEST bound geometry is host-only; transformed recipes require explicit refresh."
            )
        if not isinstance(leaf, Array):
            raise TypeError(
                "cuEST binding numerical recipes must contain only dynamic JAX arrays."
            )
        arrays.append(leaf)
    return tuple(arrays)


@dataclass(frozen=True, eq=False)
class _CuestRecipeGuard:
    """Ephemeral weak-only staleness guard, never a durable/native cache identity."""

    arrays: tuple[ReferenceType[Array], ...]
    structure: jax.tree_util.PyTreeDef

    def check(
        self,
        prepared: PreparedCuestIntegrals,
        positions: Array,
        charges: Array,
        key: Array,
        /,
    ) -> None:
        current = _binding_arrays(prepared, positions, charges, key)
        structure = jax.tree_util.tree_structure((prepared, positions, charges, key))
        if (
            structure != self.structure
            or len(current) != len(self.arrays)
            or any(
                reference() is not value
                for reference, value in zip(self.arrays, current, strict=True)
            )
        ):
            raise ValueError(
                "cuEST binding recipe changed without recertification; use refresh or restore."
            )


class PreparedCuestIntegrals(StrictModule, NonTrainableState):
    """Portable geometry-independent dynamic recipe; no native handles or guards."""

    plan: CuestIntegralPlan
    basis: PreparedGaussianBasis
    orbital_layout: CuestBasisLayout
    auxiliary_layout: CuestBasisLayout
    definition_id: str = eqx.field(static=True)
    partition: int = eqx.field(static=True)

    def __init__(self, plan: CuestIntegralPlan, basis: PreparedGaussianBasis, /) -> None:
        if not isinstance(plan, CuestIntegralPlan) or not isinstance(
            basis, PreparedGaussianBasis
        ):
            raise TypeError(
                "cuEST preparation requires its typed policy and prepared orbital basis."
            )
        if basis.system_id != plan.auxiliary_basis.system_id:
            raise ValueError(
                "cuEST orbital and auxiliary bases belong to different systems."
            )
        orbital = CuestBasisLayout(basis)
        auxiliary = CuestBasisLayout(plan.auxiliary_basis, auxiliary=True)
        if not np.array_equal(
            np.asarray(orbital.center_indices), np.asarray(auxiliary.center_indices)
        ):
            raise ValueError(
                "cuEST orbital and auxiliary bases require the same actual centers."
            )
        definition = _cuest_definition_id(plan, basis, orbital, auxiliary)
        self.plan = plan
        self.basis = basis
        self.orbital_layout = orbital
        self.auxiliary_layout = auxiliary
        self.definition_id = definition
        self.partition = cuda_partition_key(definition)

    def validate_definition(self) -> None:
        actual = _cuest_definition_id(
            self.plan, self.basis, self.orbital_layout, self.auxiliary_layout
        )
        if (
            actual != self.definition_id
            or self.partition != cuda_partition_key(actual)
            or self.plan._identity() != self.plan.plan_id
            or self.orbital_layout.basis_id != self.basis.prepared_id
            or self.basis.system_id != self.plan.auxiliary_basis.system_id
            or self.auxiliary_layout.basis_id != self.plan.auxiliary_basis.prepared_id
        ):
            raise ValueError(
                "Prepared cuEST integral definition changed; prepare the actual basis/policy again."
            )

    def bind_geometry(self, positions: ArrayLike, charges: ArrayLike, /) -> CuestGeometry:
        return CuestGeometry(self, positions, charges)

    def _arguments(self, positions: Array, charges: Array, /) -> tuple[Array, ...]:
        orbital, auxiliary = self.orbital_layout, self.auxiliary_layout
        return (
            orbital.shell_metadata,
            _fp64(orbital.exponents, "orbital exponents"),
            _fp64(orbital.coefficients, "orbital coefficients"),
            auxiliary.shell_metadata,
            _fp64(auxiliary.exponents, "auxiliary exponents"),
            _fp64(auxiliary.coefficients, "auxiliary coefficients"),
            _fp64(positions, "positions"),
            _fp64(charges, "nuclear charges"),
        )

    def _call(
        self,
        target: str,
        shapes: tuple[jax.ShapeDtypeStruct, ...],
        arguments: tuple[Array, ...],
        /,
    ) -> tuple[Array, ...]:
        # Explicit bind/restore owns admission. Warm execution must neither probe
        # the environment nor replay/rehash the immutable numerical recipe.
        plan = self.plan
        return tuple(
            jax.ffi.ffi_call(target, shapes, has_side_effect=True)(
                *arguments,
                representation=self.orbital_layout.pure,
                fitting_cutoff=plan.fitting_cutoff,
                relative_conditioning=int(plan.relative_conditioning),
                inversion=0 if plan.fitting_algorithm == "qr" else 1,
                screening=plan.screening_threshold,
                variable_workspace_bytes=plan.variable_workspace_bytes,
                workspace_limit_bytes=plan.maximum_workspace_bytes,
                cache_capacity=plan.maximum_cached_geometries,
                partition=self.partition,
            )
        )


class CuestGeometry(StrictModule, NonTrainableState):
    """Explicit host-only binding with dynamic reconstructible recipe and key.

    Native hits transfer only the 32-byte content certificate, not shell tables
    or geometry. Eviction/release reconstructs from the same dynamic recipe.
    The opaque weak guard is a host-only dynamic leaf, not static numerical
    capture, a durable fingerprint, or a compiled program identity. Generic
    transformed/restored data must be explicitly refreshed or restored before
    execution. Bound objects are ephemeral and never stored in SCF states.
    """

    prepared: PreparedCuestIntegrals
    positions: Array
    charges: Array
    key: Array
    guard: _CuestRecipeGuard

    def __init__(
        self,
        prepared: PreparedCuestIntegrals,
        positions: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        expected_key: ArrayLike | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedCuestIntegrals):
            raise TypeError(
                "cuEST geometry binding requires a prepared integral definition."
            )
        prepared.validate_definition()
        coordinate = _fp64(jnp.asarray(positions), "positions")
        nuclear_charges = _fp64(jnp.asarray(charges), "nuclear charges")
        if (
            coordinate.ndim != 2
            or coordinate.shape[1] != 3
            or nuclear_charges.shape != coordinate.shape[:1]
            or not bool(jnp.all(jnp.isfinite(coordinate)))
            or not bool(jnp.all(jnp.isfinite(nuclear_charges)))
        ):
            raise ValueError(
                "cuEST geometry/charges require finite FP64 (atoms,3)/(atoms,) arrays."
            )
        last_center = int(np.max(np.asarray(prepared.orbital_layout.center_indices)))
        if coordinate.shape[0] <= last_center:
            raise ValueError("cuEST geometry does not contain every bound basis center.")
        key = _geometry_key(prepared.definition_id, coordinate)
        if expected_key is not None:
            restored_key = np.asarray(expected_key)
            if (
                restored_key.shape != (4,)
                or restored_key.dtype != np.dtype(np.int64)
                or not np.array_equal(restored_key, np.asarray(key))
            ):
                raise ValueError(
                    "Restored cuEST geometry certificate does not match the actual recipe."
                )
        load_cuda_component("cuest")
        status = jax.ShapeDtypeStruct((), jnp.int32)
        (code,) = prepared._call(
            "phydrax_cuest_bind_geometry",
            (status,),
            (key,) + prepared._arguments(coordinate, nuclear_charges),
        )
        if not bool(code == 0):
            raise RuntimeError("cuEST geometry binding failed.")
        values = _binding_arrays(prepared, coordinate, nuclear_charges, key)
        guard = _CuestRecipeGuard(
            tuple(ref(value) for value in values),
            jax.tree_util.tree_structure((prepared, coordinate, nuclear_charges, key)),
        )
        self.prepared = prepared
        self.positions = coordinate
        self.charges = nuclear_charges
        self.key = key
        self.guard = guard

    def refresh(
        self, *, positions: ArrayLike | None = None, charges: ArrayLike | None = None
    ) -> CuestGeometry:
        return CuestGeometry(
            self.prepared,
            self.positions if positions is None else positions,
            self.charges if charges is None else charges,
        )

    def restore(self) -> CuestGeometry:
        return CuestGeometry(
            self.prepared, self.positions, self.charges, expected_key=self.key
        )

    def _validate_bound(self) -> None:
        self.guard.check(self.prepared, self.positions, self.charges, self.key)

    def _arguments(self) -> tuple[Array, ...]:
        return (self.key,) + self.prepared._arguments(self.positions, self.charges)

    def one_electron(self) -> MolecularOneElectronResult:
        from ._provider import MolecularOneElectronResult

        self._validate_bound()
        count = self.prepared.basis.basis_function_count
        matrix = jax.ShapeDtypeStruct((count, count), jnp.float64)
        status = jax.ShapeDtypeStruct((), jnp.int32)
        overlap, kinetic, attraction, code = self.prepared._call(
            "phydrax_cuest_one_electron",
            (matrix, matrix, matrix, status),
            self._arguments(),
        )
        layout = self.prepared.orbital_layout
        overlap = layout.matrix_from_vendor(overlap)
        kinetic = layout.matrix_from_vendor(kinetic)
        attraction = layout.matrix_from_vendor(attraction)
        nuclear = nuclear_repulsion_energy(self.positions, self.charges)
        return MolecularOneElectronResult(
            overlap,
            kinetic + attraction,
            nuclear,
            (code == 0)
            & jnp.all(jnp.isfinite(overlap))
            & jnp.all(jnp.isfinite(kinetic))
            & jnp.all(jnp.isfinite(attraction))
            & jnp.isfinite(nuclear),
            self.prepared.definition_id,
            self.positions,
            kinetic,
            attraction,
        )

    def jk(
        self, density: ArrayLike, /, *, factors: ArrayLike | None = None
    ) -> MolecularJKResult:
        from ._provider import MolecularJKResult

        self._validate_bound()
        density = _fp64(jnp.asarray(density), "density")
        count = self.prepared.basis.basis_function_count
        if density.shape != (count, count):
            raise ValueError("cuEST density must align with the bound Gaussian basis.")
        factor, factor_evidence = _density_factors(
            density,
            self.prepared.plan.density_factorization_tolerance,
            None if factors is None else jnp.asarray(factors),
        )
        layout = self.prepared.orbital_layout
        matrix = jax.ShapeDtypeStruct((count, count), jnp.float64)
        status = jax.ShapeDtypeStruct((), jnp.int32)
        coulomb, exchange, code = self.prepared._call(
            "phydrax_cuest_jk",
            (matrix, matrix, status),
            self._arguments()
            + (layout.density_to_vendor(density), layout.factors_to_vendor(factor)),
        )
        return MolecularJKResult(
            layout.matrix_from_vendor(coulomb),
            layout.matrix_from_vendor(exchange),
            (code == 0)
            & jnp.all(jnp.isfinite(coulomb))
            & jnp.all(jnp.isfinite(exchange)),
            self.prepared.definition_id,
            self.positions,
            factorization=factor_evidence,
        )

    def gradient(
        self,
        density: ArrayLike,
        energy_weighted_density: ArrayLike,
        /,
        *,
        factors: ArrayLike | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        self._validate_bound()
        density = _fp64(jnp.asarray(density), "density")
        weighted = _fp64(jnp.asarray(energy_weighted_density), "Pulay density")
        count = self.prepared.basis.basis_function_count
        if density.shape != (count, count) or weighted.shape != (count, count):
            raise ValueError(
                "cuEST density and Pulay density must align with the bound Gaussian basis."
            )
        layout = self.prepared.orbital_layout
        factor, _ = _density_factors(
            density,
            self.prepared.plan.density_factorization_tolerance,
            None if factors is None else jnp.asarray(factors),
        )
        gradient = jax.ShapeDtypeStruct(self.positions.shape, jnp.float64)
        status = jax.ShapeDtypeStruct((), jnp.int32)
        one, two, pulay, code = self.prepared._call(
            "phydrax_cuest_gradient",
            (gradient, gradient, gradient, status),
            self._arguments()
            + (
                layout.density_to_vendor(density),
                layout.factors_to_vendor(factor),
                layout.energy_weighted_density_to_vendor(weighted),
            ),
        )
        return (
            one,
            two,
            pulay,
            (
                (code == 0)
                & jnp.all(jnp.isfinite(one))
                & jnp.all(jnp.isfinite(two))
                & jnp.all(jnp.isfinite(pulay))
            ),
        )


__all__ = ["CuestGeometry", "CuestIntegralPlan", "PreparedCuestIntegrals"]
