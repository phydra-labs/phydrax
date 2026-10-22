#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Packed cuEST shells and a sparse, reversible AO convention conversion.

Source contracts (cuEST 0.2.2):
https://docs.nvidia.com/cuda/cuest/gaussian_basis.html
https://docs.nvidia.com/cuda/cuest/api_methods.html#primitive-basis-function-normalization
https://docs.nvidia.com/cuda/cuest/c-api/basis-and-grid/basis/index.html

``cuestAOShellCreate`` consumes finished coefficients verbatim, with one common
radial contraction for every component. CCA Cartesian order is the native order,
but only its (L,0,0), (0,L,0), (0,0,L) components are normalized. Native Cartesians
normalize *every* component. Both spherical conventions use Racah solid harmonics
without Condon--Shortley phase, in order 0,+1,-1,+2,-2,... (p = z,x,y).
Thus only a permutation of shell blocks and diagonal component scales are needed;
there is no dense global AO transform. General contractions become individual
vendor shells, with their source shell/contraction and active primitive indices
retained. No extra shells are introduced to fill angular-momentum holes.

The C basis API requires contiguous angular sets *at each center*, even though
the overview describes only the molecular set. Mixed representations are invalid;
a density-fitting auxiliary basis must be spherical. Neither the value nor the
nuclear-gradient API documents a fixed numeric angular-momentum bound. The
configurable handle harmonic table defaults to L=10, not a documented kernel
support bound; the native integration must provision it for the requested L and
report vendor failures rather than impose a guessed basis-name or L restriction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._basis import PreparedGaussianBasis
from ._shell import (
    cartesian_angular_exponents,
    GaussianShellRepresentation,
    odd_double_factorial,
)


class CuestBasisLayout(StrictModule, NonTrainableState):
    """Dynamic packed shell buffers plus permutation/diagonal AO conversion.

    ``shell_metadata`` is int32 ``(nshell, 4)``: system center index, L,
    active primitive count, flat primitive offset. Shells are grouped by ascending
    system center, then retain native shell/contraction order. ``exponents`` and
    ``coefficients`` are contiguous flat float64 buffers with the same offsets.
    ``pure`` is the single vendor representation flag (0 Cartesian, 1 spherical).
    Geometry is not captured here; metadata center indices refer to the complete
    system position array, not to a renumbered list of basis centers.

    ``vendor_to_native[v]`` gives native AO index n, and
    ``phi_native[n] = component_scales[n] * phi_vendor[v]``. This sparse mapping
    applies independently on each AO axis. Matrix quantities are covariant;
    densities, occupied factors and energy-weighted densities are contravariant.
    Gradient vectors already contracted in vendor AO space need no AO conversion.
    """

    shell_metadata: Array
    exponents: Array
    coefficients: Array
    shell_native_indices: Array
    shell_native_offsets: Array
    shell_vendor_offsets: Array
    active_primitive_indices: Array
    center_indices: Array
    center_particle_ids: Array
    vendor_to_native: Array
    native_to_vendor: Array
    component_scales: Array
    pure: int = eqx.field(static=True)
    auxiliary: bool = eqx.field(static=True)
    maximum_angular_momentum: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PreparedGaussianBasis,
        /,
        *,
        auxiliary: bool = False,
    ) -> None:
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("cuEST layout requires PreparedGaussianBasis.")
        shells = basis.plan.shells
        representations = {shell.representation for shell in shells}
        if len(representations) != 1:
            raise ValueError("cuEST does not support mixed shell representations.")
        pure = int(shells[0].representation is GaussianShellRepresentation.REAL_SPHERICAL)
        if auxiliary and not pure:
            raise ValueError("cuEST density-fitting auxiliary basis must be spherical.")

        # Prepared coefficients already include both primitive and contraction
        # normalization. The first Cartesian row is the normalized (L,0,0) row
        # required by cuEST for either representation; do not normalize twice.
        centers = np.asarray(basis.center_indices)
        prepared_coefficients = np.asarray(basis.normalized_coefficients)
        prepared_exponents = np.asarray(basis.exponents)
        angular_sets: dict[int, set[int]] = {}
        records: list[tuple[int, int, int, int, int]] = []
        native_scales: list[float] = []
        cartesian_offset = 0
        native_offset = 0
        for shell_index, shell in enumerate(shells):
            center = int(centers[cartesian_offset])
            angular = shell.angular_momentum
            angular_sets.setdefault(center, set()).add(angular)
            if pure:
                scales = [1.0] * shell.component_count
            else:
                leading = odd_double_factorial(2 * angular - 1)
                scales = [
                    np.sqrt(
                        leading
                        / (
                            odd_double_factorial(2 * x - 1)
                            * odd_double_factorial(2 * y - 1)
                            * odd_double_factorial(2 * z - 1)
                        )
                    )
                    for x, y, z in cartesian_angular_exponents(angular)
                ]
            for contraction in range(shell.contraction_count):
                records.append(
                    (center, shell_index, contraction, cartesian_offset, native_offset)
                )
                native_scales.extend(scales)
                cartesian_offset += shell.cartesian_component_count
                native_offset += shell.component_count
        for center, angulars in angular_sets.items():
            if angulars != set(range(max(angulars) + 1)):
                raise ValueError(
                    "cuEST requires contiguous angular momenta starting at L=0 "
                    f"at system center {center}; found {sorted(angulars)}."
                )

        metadata: list[tuple[int, int, int, int]] = []
        origins: list[tuple[int, int]] = []
        native_offsets: list[int] = []
        vendor_offsets: list[int] = []
        exponent_parts: list[np.ndarray] = []
        coefficient_parts: list[np.ndarray] = []
        primitive_indices: list[np.ndarray] = []
        particle_ids: list[int] = []
        permutation: list[int] = []
        primitive_offset = 0
        vendor_offset = 0
        for center, shell_index, contraction, row, output_start in sorted(
            records, key=lambda record: (record[0], record[1], record[2])
        ):
            shell = shells[shell_index]
            active = np.flatnonzero(np.asarray(shell.primitive_mask))
            if active.size == 0:
                raise ValueError("cuEST shells must have at least one active primitive.")
            metadata.append(
                (center, shell.angular_momentum, active.size, primitive_offset)
            )
            origins.append((shell_index, contraction))
            native_offsets.append(output_start)
            vendor_offsets.append(vendor_offset)
            exponent_parts.append(prepared_exponents[row, active])
            coefficient_parts.append(prepared_coefficients[row, active])
            primitive_indices.append(active)
            particle_ids.append(shell.center_particle_id)
            permutation.extend(range(output_start, output_start + shell.component_count))
            primitive_offset += active.size
            vendor_offset += shell.component_count
        if max(primitive_offset, vendor_offset, *angular_sets) > np.iinfo(np.int32).max:
            raise ValueError("cuEST packed layout exceeds int32 indexing capacity.")

        arrays = {
            "shell_metadata": np.asarray(metadata, dtype=np.int32),
            "exponents": np.asarray(np.concatenate(exponent_parts), dtype=np.float64),
            "coefficients": np.asarray(
                np.concatenate(coefficient_parts), dtype=np.float64
            ),
            "shell_native_indices": np.asarray(origins, dtype=np.int32),
            "shell_native_offsets": np.asarray(native_offsets, dtype=np.int32),
            "shell_vendor_offsets": np.asarray(vendor_offsets, dtype=np.int32),
            "active_primitive_indices": np.asarray(
                np.concatenate(primitive_indices), dtype=np.int32
            ),
            "center_indices": np.asarray(sorted(angular_sets), dtype=np.int32),
            "center_particle_ids": np.asarray(particle_ids, dtype=np.int64),
            "vendor_to_native": np.asarray(permutation, dtype=np.int32),
            "native_to_vendor": np.asarray(np.argsort(permutation), dtype=np.int32),
            "component_scales": np.asarray(native_scales, dtype=np.float64),
        }
        self.shell_metadata = jnp.asarray(arrays["shell_metadata"], dtype=jnp.int32)
        self.exponents = jnp.asarray(arrays["exponents"], dtype=jnp.float64)
        self.coefficients = jnp.asarray(arrays["coefficients"], dtype=jnp.float64)
        self.shell_native_indices = jnp.asarray(
            arrays["shell_native_indices"], dtype=jnp.int32
        )
        self.shell_native_offsets = jnp.asarray(
            arrays["shell_native_offsets"], dtype=jnp.int32
        )
        self.shell_vendor_offsets = jnp.asarray(
            arrays["shell_vendor_offsets"], dtype=jnp.int32
        )
        self.active_primitive_indices = jnp.asarray(
            arrays["active_primitive_indices"], dtype=jnp.int32
        )
        self.center_indices = jnp.asarray(arrays["center_indices"], dtype=jnp.int32)
        self.center_particle_ids = jnp.asarray(
            arrays["center_particle_ids"], dtype=jnp.int64
        )
        self.vendor_to_native = jnp.asarray(arrays["vendor_to_native"], dtype=jnp.int32)
        self.native_to_vendor = jnp.asarray(arrays["native_to_vendor"], dtype=jnp.int32)
        self.component_scales = jnp.asarray(arrays["component_scales"], dtype=jnp.float64)
        self.pure = pure
        self.auxiliary = bool(auxiliary)
        self.maximum_angular_momentum = basis.maximum_angular_momentum
        self.basis_id = basis.prepared_id
        self.layout_id = canonical_fingerprint(
            {
                "kind": "cuest-basis-layout-cca-v1",
                "basis": basis.prepared_id,
                "pure": pure,
                "auxiliary": bool(auxiliary),
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @property
    def basis_function_count(self) -> int:
        return self.vendor_to_native.size

    @property
    def shell_count(self) -> int:
        return self.shell_metadata.shape[0]

    def _matrix(self, values: ArrayLike, /) -> Array:
        matrix = jnp.asarray(values, dtype=jnp.float64)
        if matrix.ndim < 2 or matrix.shape[-2:] != (
            self.basis_function_count,
            self.basis_function_count,
        ):
            raise ValueError("cuEST AO matrix must have trailing shape (nao, nao).")
        return matrix

    def _factors(self, values: ArrayLike, /) -> Array:
        factors = jnp.asarray(values, dtype=jnp.float64)
        if factors.ndim != 2 or factors.shape[1] != self.basis_function_count:
            raise ValueError("cuEST occupied factors must have shape (rank, nao).")
        return factors

    def matrix_from_vendor(self, values: ArrayLike, /) -> Array:
        """Convert covariant S/T/V/J/K (or AO-resolved derivatives) to native AOs."""
        matrix = self._matrix(values)
        indices = self.native_to_vendor
        native = jnp.take(jnp.take(matrix, indices, axis=-2), indices, axis=-1)
        scales = self.component_scales
        return native * scales[:, None] * scales[None, :]

    def matrix_to_vendor(self, values: ArrayLike, /) -> Array:
        """Inverse covariant conversion; not the density conversion."""
        matrix = self._matrix(values)
        indices = self.vendor_to_native
        vendor = jnp.take(jnp.take(matrix, indices, axis=-2), indices, axis=-1)
        scales = self.component_scales[indices]
        return vendor / scales[:, None] / scales[None, :]

    def density_to_vendor(self, values: ArrayLike, /) -> Array:
        """Convert contravariant spin-summed D, preserving tr(D M)."""
        matrix = self._matrix(values)
        indices = self.vendor_to_native
        vendor = jnp.take(jnp.take(matrix, indices, axis=-2), indices, axis=-1)
        scales = self.component_scales[indices]
        return vendor * scales[:, None] * scales[None, :]

    def density_from_vendor(self, values: ArrayLike, /) -> Array:
        """Inverse contravariant conversion."""
        matrix = self._matrix(values)
        indices = self.native_to_vendor
        native = jnp.take(jnp.take(matrix, indices, axis=-2), indices, axis=-1)
        scales = self.component_scales
        return native / scales[:, None] / scales[None, :]

    def factors_to_vendor(self, values: ArrayLike, /) -> Array:
        """Convert rank-by-nao factors F with D = F.T @ F (including rank zero)."""
        factors = self._factors(values)
        indices = self.vendor_to_native
        return factors[:, indices] * self.component_scales[indices]

    def factors_from_vendor(self, values: ArrayLike, /) -> Array:
        """Inverse occupied-factor conversion."""
        factors = self._factors(values)
        return factors[:, self.native_to_vendor] / self.component_scales

    def energy_weighted_density_to_vendor(self, values: ArrayLike, /) -> Array:
        """Convert Pulay W exactly like D; the caller supplies its sign/scale."""
        return self.density_to_vendor(values)


__all__ = ["CuestBasisLayout"]
