#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class NematicTensorBasis(StrictModule, NonTrainableState):
    orientation_dimension: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    matrices: Array
    basis_id: str = eqx.field(static=True)

    def __init__(self, orientation_dimension: int = 3, /):
        dimension = int(orientation_dimension)
        if dimension not in (2, 3):
            raise ValueError("Nematic orientation dimension must be two or three.")
        if dimension == 2:
            matrices = np.asarray(
                (
                    ((1.0 / np.sqrt(2.0), 0.0), (0.0, -1.0 / np.sqrt(2.0))),
                    ((0.0, 1.0 / np.sqrt(2.0)), (1.0 / np.sqrt(2.0), 0.0)),
                )
            )
        else:
            matrices = np.zeros((5, 3, 3), dtype=float)
            matrices[0] = np.diag((1.0, -1.0, 0.0)) / np.sqrt(2.0)
            matrices[1] = np.diag((1.0, 1.0, -2.0)) / np.sqrt(6.0)
            matrices[2, 0, 1] = matrices[2, 1, 0] = 1.0 / np.sqrt(2.0)
            matrices[3, 0, 2] = matrices[3, 2, 0] = 1.0 / np.sqrt(2.0)
            matrices[4, 1, 2] = matrices[4, 2, 1] = 1.0 / np.sqrt(2.0)
        gram = np.einsum("aij,bij->ab", matrices, matrices)
        if not np.allclose(gram, np.eye(matrices.shape[0]), rtol=0.0, atol=1e-14):
            raise ValueError("Nematic tensor basis is not orthonormal.")
        self.orientation_dimension = dimension
        self.component_count = matrices.shape[0]
        self.matrices = jnp.asarray(matrices)
        self.basis_id = canonical_fingerprint(
            {
                "kind": "nematic-tensor-basis",
                "dimension": dimension,
                "matrices": array_tree_fingerprint(matrices),
            }
        )

    def decode(self, compact: ArrayLike, /) -> Array:
        value = jnp.asarray(compact)
        if value.ndim < 1 or value.shape[-1] != self.component_count:
            raise ValueError("compact Q must end in nematic component axis.")
        return contract("aij,...a->...ij", self.matrices, value)

    def encode(self, tensor: ArrayLike, /) -> Array:
        value = jnp.asarray(tensor)
        expected = (self.orientation_dimension, self.orientation_dimension)
        if value.shape[-2:] != expected:
            raise ValueError("Q tensor must end in orientation tensor axes.")
        return contract("aij,...ij->...a", self.matrices, value)

    def project(self, tensor: ArrayLike, /) -> Array:
        return self.decode(self.encode(tensor))


class LandauDeGennesParameters(StrictModule):
    coefficient_a: Array
    coefficient_b: Array
    coefficient_c: Array
    elastic_constant: Array
    chiral_wave_number: Array
    dielectric_anisotropy: Array

    def __init__(
        self,
        coefficient_a: ArrayLike,
        coefficient_b: ArrayLike,
        coefficient_c: ArrayLike,
        elastic_constant: ArrayLike,
        /,
        *,
        chiral_wave_number: ArrayLike = 0.0,
        dielectric_anisotropy: ArrayLike = 0.0,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                coefficient_a,
                coefficient_b,
                coefficient_c,
                elastic_constant,
                chiral_wave_number,
                dielectric_anisotropy,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Landau-de Gennes parameters must be scalar.")
        if not bool(
            jnp.all(jnp.asarray([jnp.isfinite(value) for value in values]))
            & (values[2] > 0.0)
            & (values[3] > 0.0)
        ):
            raise ValueError("Landau-de Gennes parameters are inadmissible.")
        (
            self.coefficient_a,
            self.coefficient_b,
            self.coefficient_c,
            self.elastic_constant,
            self.chiral_wave_number,
            self.dielectric_anisotropy,
        ) = values


class BerisEdwardsParameters(StrictModule):
    rotational_mobility: Array
    flow_alignment: Array
    activity: Array

    def __init__(
        self,
        rotational_mobility: ArrayLike,
        flow_alignment: ArrayLike,
        /,
        *,
        activity: ArrayLike = 0.0,
    ):
        mobility = jnp.asarray(rotational_mobility)
        alignment = jnp.asarray(flow_alignment, dtype=mobility.dtype)
        activity_ = jnp.asarray(activity, dtype=mobility.dtype)
        if mobility.shape != () or alignment.shape != () or activity_.shape != ():
            raise ValueError("Beris-Edwards parameters must be scalar.")
        if not bool(
            jnp.isfinite(mobility)
            & (mobility > 0.0)
            & jnp.isfinite(alignment)
            & jnp.isfinite(activity_)
        ):
            raise ValueError("Beris-Edwards parameters are inadmissible.")
        self.rotational_mobility = mobility
        self.flow_alignment = alignment
        self.activity = activity_


class NematicThermodynamicFields(StrictModule):
    tensor: Array
    bulk_energy_density: Array
    elastic_energy_density: Array
    electric_energy_density: Array
    total_energy_density: Array
    molecular_field: Array
    molecular_field_tensor: Array
    distortion_stress: Array
    electric_stress: Array
    scalar_order: Array
    symmetry_residual: Array
    trace_residual: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class LandauDeGennesClosure(StrictModule, NonTrainableState):
    basis: NematicTensorBasis
    closure_id: str = eqx.field(static=True)

    def __init__(self, basis: NematicTensorBasis, /):
        if not isinstance(basis, NematicTensorBasis):
            raise TypeError("basis must be NematicTensorBasis.")
        self.basis = basis
        self.closure_id = canonical_fingerprint(
            {"kind": "landau-de-gennes-closure", "basis": basis.basis_id}
        )

    def evaluate(
        self,
        compact_q: ArrayLike,
        compact_gradient: ArrayLike,
        compact_laplacian: ArrayLike,
        parameters: LandauDeGennesParameters,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> NematicThermodynamicFields:
        compact = jnp.asarray(compact_q)
        gradient = jnp.asarray(compact_gradient, dtype=compact.dtype)
        laplacian = jnp.asarray(compact_laplacian, dtype=compact.dtype)
        if compact.shape[-1] != self.basis.component_count:
            raise ValueError("compact_q has incompatible component axis.")
        if (
            gradient.shape[:-2] != compact.shape[:-1]
            or gradient.shape[-1] != self.basis.component_count
        ):
            raise ValueError("compact_gradient must add a spatial derivative axis.")
        if laplacian.shape != compact.shape:
            raise ValueError("compact_laplacian must match compact_q.")
        q_tensor = self.basis.decode(compact)
        gradient_tensor = contract("aij,...ka->...kij", self.basis.matrices, gradient)
        laplacian_tensor = self.basis.decode(laplacian)
        q_squared = contract("...ik,...kj->...ij", q_tensor, q_tensor)
        trace_q2 = jnp.trace(q_squared, axis1=-2, axis2=-1)
        trace_q3 = jnp.trace(
            contract("...ik,...kj->...ij", q_squared, q_tensor),
            axis1=-2,
            axis2=-1,
        )
        bulk = (
            0.5 * parameters.coefficient_a * trace_q2
            - parameters.coefficient_b * trace_q3 / 3.0
            + 0.25 * parameters.coefficient_c * trace_q2**2
        )
        gradient_norm = jnp.sum(gradient_tensor * gradient_tensor, axis=(-3, -2, -1))
        elastic = 0.5 * parameters.elastic_constant * gradient_norm
        identity = jnp.eye(self.basis.orientation_dimension, dtype=compact.dtype)
        bulk_field = (
            -parameters.coefficient_a * q_tensor
            + parameters.coefficient_b
            * (
                q_squared
                - trace_q2[..., None, None] * identity / self.basis.orientation_dimension
            )
            - parameters.coefficient_c * trace_q2[..., None, None] * q_tensor
        )
        molecular = bulk_field + parameters.elastic_constant * laplacian_tensor
        spatial_dimension = gradient.shape[-2]
        chiral = parameters.chiral_wave_number
        chiral_supported = jnp.asarray(True)
        if self.basis.orientation_dimension == 3 and spatial_dimension == 3:
            levi_civita = _levi_civita(compact.dtype)
            curl_q = contract("ikl,...klj->...ij", levi_civita, gradient_tensor)
            elastic = elastic + (
                2.0
                * parameters.elastic_constant
                * chiral
                * jnp.sum(q_tensor * curl_q, axis=(-2, -1))
                + 2.0 * parameters.elastic_constant * chiral**2 * trace_q2
            )
            molecular = molecular - 4.0 * parameters.elastic_constant * (
                chiral * curl_q + chiral**2 * q_tensor
            )
        else:
            chiral_supported = chiral == 0.0
        electric = (
            jnp.zeros(
                compact.shape[:-1] + (self.basis.orientation_dimension,),
                dtype=compact.dtype,
            )
            if electric_field is None
            else jnp.asarray(electric_field, dtype=compact.dtype)
        )
        if electric.shape != compact.shape[:-1] + (self.basis.orientation_dimension,):
            raise ValueError("electric_field must match orientation dimension.")
        electric_outer = contract("...i,...j->...ij", electric, electric)
        electric_projected = electric_outer - (
            jnp.sum(electric * electric, axis=-1)[..., None, None]
            * identity
            / self.basis.orientation_dimension
        )
        electric_energy = (
            -0.5
            * parameters.dielectric_anisotropy
            * jnp.sum(electric_outer * q_tensor, axis=(-2, -1))
        )
        molecular = (
            molecular + 0.5 * parameters.dielectric_anisotropy * electric_projected
        )
        molecular = self.basis.project(molecular)
        molecular_compact = self.basis.encode(molecular)
        distortion = -parameters.elastic_constant * contract(
            "...ikl,...jkl->...ij", gradient_tensor, gradient_tensor
        )
        electric_stress = parameters.dielectric_anisotropy * contract(
            "...ik,...k,...j->...ij", q_tensor, electric, electric
        )
        total = bulk + elastic + electric_energy
        symmetry_residual = jnp.max(jnp.abs(q_tensor - jnp.swapaxes(q_tensor, -1, -2)))
        trace_residual = jnp.max(jnp.abs(jnp.trace(q_tensor, axis1=-2, axis2=-1)))
        scalar_order = jnp.sqrt(
            self.basis.orientation_dimension
            * trace_q2
            / (self.basis.orientation_dimension - 1.0)
        )
        successful = (
            jnp.all(jnp.isfinite(total))
            & jnp.all(jnp.isfinite(molecular_compact))
            & jnp.all(jnp.isfinite(distortion))
            & chiral_supported
            & (symmetry_residual <= 64.0 * jnp.finfo(compact.dtype).eps)
            & (trace_residual <= 64.0 * jnp.finfo(compact.dtype).eps)
        )
        return NematicThermodynamicFields(
            q_tensor,
            bulk,
            elastic,
            electric_energy,
            total,
            molecular_compact,
            molecular,
            distortion,
            electric_stress,
            scalar_order,
            symmetry_residual,
            trace_residual,
            successful,
            self.closure_id,
        )


class BerisEdwardsConstitutiveFields(StrictModule):
    alignment_term: Array
    passive_stress: Array
    active_stress: Array
    total_stress: Array
    passive_power: Array
    active_power: Array
    successful: Array


def beris_edwards_constitutive_fields(
    basis: NematicTensorBasis,
    compact_q: ArrayLike,
    compact_molecular_field: ArrayLike,
    velocity_gradient: ArrayLike,
    distortion_stress: ArrayLike,
    parameters: BerisEdwardsParameters,
    /,
) -> BerisEdwardsConstitutiveFields:
    q_tensor = basis.decode(compact_q)
    molecular = basis.decode(compact_molecular_field)
    velocity = jnp.asarray(velocity_gradient, dtype=q_tensor.dtype)
    distortion = jnp.asarray(distortion_stress, dtype=q_tensor.dtype)
    dimension = basis.orientation_dimension
    if velocity.shape != q_tensor.shape or distortion.shape != q_tensor.shape:
        raise ValueError("Velocity gradient and distortion stress must match Q tensor.")
    symmetric = 0.5 * (velocity + jnp.swapaxes(velocity, -1, -2))
    antisymmetric = 0.5 * (velocity - jnp.swapaxes(velocity, -1, -2))
    identity = jnp.eye(dimension, dtype=q_tensor.dtype)
    shifted = q_tensor + identity / dimension
    contraction = jnp.sum(q_tensor * symmetric, axis=(-2, -1))
    alignment = (
        contract(
            "...ik,...kj->...ij",
            parameters.flow_alignment * symmetric + antisymmetric,
            shifted,
        )
        + contract(
            "...ik,...kj->...ij",
            shifted,
            parameters.flow_alignment * symmetric - antisymmetric,
        )
        - 2.0 * parameters.flow_alignment * shifted * contraction[..., None, None]
    )
    qh = jnp.sum(q_tensor * molecular, axis=(-2, -1))
    passive = (
        2.0 * parameters.flow_alignment * shifted * qh[..., None, None]
        - parameters.flow_alignment
        * (
            contract("...ik,...kj->...ij", molecular, shifted)
            + contract("...ik,...kj->...ij", shifted, molecular)
        )
        + contract("...ik,...kj->...ij", q_tensor, molecular)
        - contract("...ik,...kj->...ij", molecular, q_tensor)
        + distortion
    )
    active = -parameters.activity * q_tensor
    total = passive + active
    passive_power = jnp.sum(passive * velocity, axis=(-2, -1))
    active_power = jnp.sum(active * velocity, axis=(-2, -1))
    successful = (
        jnp.all(jnp.isfinite(alignment))
        & jnp.all(jnp.isfinite(total))
        & jnp.all(jnp.isfinite(passive_power))
        & jnp.all(jnp.isfinite(active_power))
    )
    return BerisEdwardsConstitutiveFields(
        basis.encode(alignment),
        passive,
        active,
        total,
        passive_power,
        active_power,
        successful,
    )


def _levi_civita(dtype):
    return jnp.asarray(
        (
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        dtype=dtype,
    )


__all__ = [
    "BerisEdwardsConstitutiveFields",
    "BerisEdwardsParameters",
    "LandauDeGennesClosure",
    "LandauDeGennesParameters",
    "NematicTensorBasis",
    "NematicThermodynamicFields",
    "beris_edwards_constitutive_fields",
]
