#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-cell periodic electronic Coulomb local Hamiltonian."""

from __future__ import annotations

import itertools
import math
from abc import abstractmethod
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure
from ...discretization import PeriodicCell
from ...units import BOHR, conversion_factor, HARTREE
from ._amplitude import LogAmplitude
from ._electronic import ElectronicKineticPolicy
from ._electronic_advanced import ElectronicVMCResourcePlan
from ._local import (
    AbstractLocalQuantumOperator,
    LocalOperatorEstimate,
    LocalOperatorStatus,
)


class AbstractPeriodicElectronicAmplitude(StrictModule):
    """Boundary/resource contract consumed by the periodic local Hamiltonian."""

    configuration_shape: AbstractAttribute[tuple[int, int]]
    boundary_id: AbstractAttribute[str]
    resource_plan: AbstractAttribute[ElectronicVMCResourcePlan]

    @abstractmethod
    def __call__(self, coordinates: ArrayLike, /) -> LogAmplitude:
        raise NotImplementedError


class PeriodicElectronicEwaldPolicy(StrictModule, NonTrainableState):
    """Finite Ewald resolution and caller-owned work admission.

    ``screening`` is in the inverse of the structure length unit. The two work
    limits are code admission policy, not measured capacity or release claims.
    """

    real_shifts: Array
    reciprocal_modes: Array
    real_image_radius: int = eqx.field(static=True)
    reciprocal_radius: int = eqx.field(static=True)
    screening: float = eqx.field(static=True)
    uniform_background: bool = eqx.field(static=True)
    maximum_real_pair_terms: int = eqx.field(static=True)
    maximum_reciprocal_structure_terms: int = eqx.field(static=True)
    real_image_count: int = eqx.field(static=True)
    reciprocal_mode_count: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        real_image_radius: int,
        reciprocal_radius: int,
        screening: float,
        maximum_real_pair_terms: int,
        maximum_reciprocal_structure_terms: int,
        uniform_background: bool = False,
    ):
        real_radius = int(real_image_radius)
        reciprocal_radius_ = int(reciprocal_radius)
        alpha = float(screening)
        maximum_real = int(maximum_real_pair_terms)
        maximum_reciprocal = int(maximum_reciprocal_structure_terms)
        if (
            real_radius < 0
            or reciprocal_radius_ < 0
            or not isfinite(alpha)
            or alpha <= 0.0
            or maximum_real <= 0
            or maximum_reciprocal <= 0
        ):
            raise ValueError(
                "Ewald radii, screening, and work limits must be positive and finite."
            )
        real_shifts = np.asarray(
            tuple(itertools.product(range(-real_radius, real_radius + 1), repeat=3)),
            dtype=np.int32,
        )
        reciprocal_modes = np.asarray(
            tuple(
                mode
                for mode in itertools.product(
                    range(-reciprocal_radius_, reciprocal_radius_ + 1), repeat=3
                )
                if mode != (0, 0, 0)
            ),
            dtype=np.int32,
        ).reshape((-1, 3))
        self.real_shifts = jnp.asarray(real_shifts)
        self.reciprocal_modes = jnp.asarray(reciprocal_modes)
        self.real_image_radius = real_radius
        self.reciprocal_radius = reciprocal_radius_
        self.screening = alpha
        self.uniform_background = bool(uniform_background)
        self.maximum_real_pair_terms = maximum_real
        self.maximum_reciprocal_structure_terms = maximum_reciprocal
        self.real_image_count = int(real_shifts.shape[0])
        self.reciprocal_mode_count = int(reciprocal_modes.shape[0])
        self.policy_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-ewald-policy",
                "real_image_radius": real_radius,
                "reciprocal_radius": reciprocal_radius_,
                "screening": alpha,
                "uniform_background": self.uniform_background,
                "maximum_real_pair_terms": maximum_real,
                "maximum_reciprocal_structure_terms": maximum_reciprocal,
            }
        )


class PeriodicElectronicResourceEvidence(StrictModule, NonTrainableState):
    """Exact structural work counts and asymptotic scope for one admitted case."""

    electron_count: int = eqx.field(static=True)
    nucleus_count: int = eqx.field(static=True)
    electron_pair_count: int = eqx.field(static=True)
    pair_feature_elements: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    determinant_cubic_work: int = eqx.field(static=True)
    kinetic_hessian_vector_products: int = eqx.field(static=True)
    kinetic_coordinate_chunk_size: int = eqx.field(static=True)
    ewald_real_image_count: int = eqx.field(static=True)
    ewald_reciprocal_mode_count: int = eqx.field(static=True)
    ewald_real_pair_terms: int = eqx.field(static=True)
    ewald_reciprocal_structure_terms: int = eqx.field(static=True)
    maximum_ewald_real_pair_terms: int = eqx.field(static=True)
    maximum_ewald_reciprocal_structure_terms: int = eqx.field(static=True)
    primitive_work_per_configuration: int = eqx.field(static=True)
    pair_complexity: str = eqx.field(static=True)
    determinant_complexity: str = eqx.field(static=True)
    kinetic_complexity: str = eqx.field(static=True)
    ewald_complexity: str = eqx.field(static=True)
    differentiation_scope: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        electron_count: int,
        nucleus_count: int,
        resource_plan: ElectronicVMCResourcePlan,
        kinetic: ElectronicKineticPolicy,
        ewald: PeriodicElectronicEwaldPolicy,
        /,
    ):
        electrons = int(electron_count)
        nuclei = int(nucleus_count)
        particles = electrons + nuclei
        real_terms = particles * particles * ewald.real_image_count
        reciprocal_terms = particles * ewald.reciprocal_mode_count
        if real_terms > ewald.maximum_real_pair_terms:
            raise ValueError(
                "Periodic Ewald real-space work exceeds caller admission: "
                f"{real_terms}/{ewald.maximum_real_pair_terms}."
            )
        if reciprocal_terms > ewald.maximum_reciprocal_structure_terms:
            raise ValueError(
                "Periodic Ewald reciprocal work exceeds caller admission: "
                f"{reciprocal_terms}/{ewald.maximum_reciprocal_structure_terms}."
            )
        coordinates = 3 * electrons
        primitive_work = coordinates + real_terms + reciprocal_terms
        if primitive_work > np.iinfo(np.int32).max:
            raise ValueError("Periodic local-operator work_count exceeds int32 capacity.")
        chunk = (
            coordinates
            if kinetic.coordinate_chunk_size is None
            else kinetic.coordinate_chunk_size
        )
        self.electron_count = electrons
        self.nucleus_count = nuclei
        self.electron_pair_count = electrons * (electrons - 1) // 2
        self.pair_feature_elements = resource_plan.pair_stream_elements
        self.determinant_count = resource_plan.determinant_count
        self.determinant_cubic_work = resource_plan.determinant_work
        self.kinetic_hessian_vector_products = coordinates
        self.kinetic_coordinate_chunk_size = chunk
        self.ewald_real_image_count = ewald.real_image_count
        self.ewald_reciprocal_mode_count = ewald.reciprocal_mode_count
        self.ewald_real_pair_terms = real_terms
        self.ewald_reciprocal_structure_terms = reciprocal_terms
        self.maximum_ewald_real_pair_terms = ewald.maximum_real_pair_terms
        self.maximum_ewald_reciprocal_structure_terms = (
            ewald.maximum_reciprocal_structure_terms
        )
        self.primitive_work_per_configuration = primitive_work
        self.pair_complexity = "O(Ne^2) pair features"
        self.determinant_complexity = "O(Ndet Ne^3) dense determinant factorization"
        self.kinetic_complexity = "O(3 Ne) exact Hessian-vector products"
        self.ewald_complexity = "O((Ne+Nn)^2 Nreal + (Ne+Nn) Nk) finite Ewald evaluation"
        self.differentiation_scope = (
            "fixed-cell coordinate derivatives within fixed wrap and image selections; "
            "no gradient across image selection"
        )
        self.claim = "operation-count-evidence-not-runtime-capacity-or-release-claim"
        self.valid = jnp.asarray(True)


class PeriodicElectronicPotential(StrictModule):
    """Ewald Coulomb decomposition in the structure energy unit."""

    electron_electron: Array
    electron_nucleus: Array
    nucleus_nucleus: Array
    total: Array
    singular: Array
    ewald_valid: Array


class PeriodicElectronicLocalEnergy(StrictModule):
    """Kinetic plus decomposed Ewald local energy for fixed-shape walkers."""

    kinetic: Array
    potential: PeriodicElectronicPotential
    value: Array
    amplitude_valid: Array
    valid: Array
    status: Array


class PeriodicElectronicCoulombHamiltonian(AbstractLocalQuantumOperator):
    """Born--Oppenheimer Coulomb Hamiltonian in one fixed 3D periodic cell.

    Electron configurations are Cartesian and end in ``(electron_count, 3)``.
    The amplitude must carry the same cell/twist boundary identity. Finite Ewald
    image sets are canonicalized by wrapping; derivatives are not claimed across
    the resulting discrete wrap or image-selection boundaries.
    """

    nuclei: AtomicStructure
    cell: PeriodicCell
    twist: Array
    nuclear_fractional_positions: Array
    nuclear_charges: Array
    kinetic: ElectronicKineticPolicy
    ewald: PeriodicElectronicEwaldPolicy
    resource_plan: ElectronicVMCResourcePlan
    resource_evidence: PeriodicElectronicResourceEvidence
    electron_count: int = eqx.field(static=True)
    nucleus_count: int = eqx.field(static=True)
    net_charge: int = eqx.field(static=True)
    neutral: bool = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        nuclei: AtomicStructure,
        electron_count: int,
        cell: PeriodicCell,
        /,
        *,
        twist: ArrayLike,
        ewald: PeriodicElectronicEwaldPolicy,
        kinetic: ElectronicKineticPolicy | None = None,
        resource_plan: ElectronicVMCResourcePlan | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(nuclei, AtomicStructure):
            raise TypeError("nuclei must be an AtomicStructure.")
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        if not isinstance(ewald, PeriodicElectronicEwaldPolicy):
            raise TypeError("ewald must be a PeriodicElectronicEwaldPolicy.")
        if (
            cell.rank != 3
            or cell.ambient_dimension != 3
            or not cell.fully_periodic
            or nuclei.cell is None
            or nuclei.periodic_axes is None
            or not bool(np.all(np.asarray(nuclei.periodic_axes)))
            or not np.array_equal(np.asarray(nuclei.cell), np.asarray(cell.vectors))
            or not np.array_equal(np.asarray(cell.origin), np.zeros((3,)))
        ):
            raise ValueError(
                "Periodic electronic Coulomb requires the structure's same zero-origin, fully periodic 3D cell."
            )
        length_factor = float(conversion_factor(nuclei.scale.length_unit, BOHR))
        energy_factor = float(conversion_factor(nuclei.scale.energy_unit, HARTREE))
        electrons = int(electron_count)
        policy = ElectronicKineticPolicy() if kinetic is None else kinetic
        if not isinstance(policy, ElectronicKineticPolicy):
            raise TypeError("kinetic must be an ElectronicKineticPolicy or None.")
        resource = (
            ElectronicVMCResourcePlan(electrons)
            if resource_plan is None
            else resource_plan
        )
        if not isinstance(resource, ElectronicVMCResourcePlan):
            raise TypeError("resource_plan must be ElectronicVMCResourcePlan or None.")
        if (
            resource.electron_count != electrons
            or resource.coordinate_dimension != 3 * electrons
        ):
            raise ValueError(
                "resource_plan must match the 3D periodic electron configuration."
            )
        twist_host = np.asarray(twist, dtype=np.asarray(cell.vectors).dtype)
        if twist_host.shape != (3,) or np.any(~np.isfinite(twist_host)):
            raise ValueError(
                "twist must be a finite three-vector in radians per cell translation."
            )
        active = np.asarray(nuclei.active_mask, dtype=bool)
        nuclear_positions = nuclei.positions[active]
        nuclear_charges = nuclei.atomic_numbers[active].astype(policy.compute_dtype)
        nucleus_count = int(np.count_nonzero(active))
        net_charge = int(np.sum(np.asarray(nuclei.atomic_numbers)[active])) - electrons
        neutral = net_charge == 0
        if not neutral and not ewald.uniform_background:
            raise ValueError(
                "Periodic electronic Coulomb requires charge neutrality or an explicit uniform background."
            )
        evidence = PeriodicElectronicResourceEvidence(
            electrons, nucleus_count, resource, policy, ewald
        )
        boundary_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-boundary",
                "cell": cell.cell_id,
                "twist": array_tree_fingerprint(twist_host),
            }
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "periodic-electronic-coulomb-hamiltonian",
                    "structure": nuclei.structure_id,
                    "cell": cell.cell_id,
                    "boundary": boundary_id,
                    "electrons": electrons,
                    "kinetic": policy.method_id,
                    "ewald": ewald.policy_id,
                    "length_factor": length_factor,
                    "energy_factor": energy_factor,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.nuclei = nuclei
        self.cell = cell
        self.twist = jnp.asarray(twist_host)
        self.nuclear_fractional_positions = cell.fractional(nuclear_positions)
        self.nuclear_charges = nuclear_charges
        self.kinetic = policy
        self.ewald = ewald
        self.resource_plan = resource
        self.resource_evidence = evidence
        self.electron_count = electrons
        self.nucleus_count = nucleus_count
        self.net_charge = net_charge
        self.neutral = neutral
        self.boundary_id = boundary_id
        self.configuration_shape = (electrons, 3)
        self.operator_id = identifier
        self.method_id = f"{policy.method_id}:finite-ewald={ewald.policy_id}"

    def _require_amplitude(self, model: AbstractPeriodicElectronicAmplitude, /) -> None:
        if not isinstance(model, AbstractPeriodicElectronicAmplitude):
            raise TypeError("model must implement AbstractPeriodicElectronicAmplitude.")
        if model.configuration_shape != self.configuration_shape:
            raise ValueError(
                "Periodic amplitude and Hamiltonian configuration shapes differ."
            )
        if model.boundary_id != self.boundary_id:
            raise ValueError(
                "Periodic amplitude and Hamiltonian cell/twist boundaries differ."
            )
        if (
            model.resource_plan.electron_count != self.resource_plan.electron_count
            or model.resource_plan.determinant_count
            != self.resource_plan.determinant_count
            or model.resource_plan.coordinate_dimension
            != self.resource_plan.coordinate_dimension
        ):
            raise ValueError("Periodic amplitude and Hamiltonian resource plans differ.")

    def _potential_one(self, electrons: Array, /) -> PeriodicElectronicPotential:
        dtype = jnp.dtype(self.kinetic.compute_dtype)
        electron_fractional = self.cell.fractional(jnp.asarray(electrons, dtype=dtype))
        fractional = jnp.concatenate(
            (electron_fractional, self.nuclear_fractional_positions.astype(dtype)), axis=0
        )
        raw_difference = fractional[:, None, :] - fractional[None, :, :]
        periodically_equal = jnp.all(raw_difference == jnp.rint(raw_difference), axis=-1)
        periodic_coincidence = jnp.any(
            periodically_equal & ~jnp.eye(fractional.shape[0], dtype=bool)
        )
        wrapped = fractional - jax.lax.stop_gradient(jnp.floor(fractional))
        cartesian = contract(
            "ni,ij->nj", wrapped, self.cell.vectors.astype(dtype), backend="jax"
        )
        electron_charge = jnp.concatenate(
            (
                -jnp.ones((self.electron_count,), dtype=dtype),
                jnp.zeros((self.nucleus_count,), dtype=dtype),
            )
        )
        nuclear_charge = jnp.concatenate(
            (
                jnp.zeros((self.electron_count,), dtype=dtype),
                self.nuclear_charges.astype(dtype),
            )
        )
        translations = contract(
            "si,ij->sj",
            self.ewald.real_shifts.astype(dtype),
            self.cell.vectors.astype(dtype),
            backend="jax",
        )
        displacement = (
            cartesian[:, None, None, :]
            - cartesian[None, :, None, :]
            + translations[None, None, :, :]
        )
        squared_distance = jnp.sum(displacement * displacement, axis=-1)
        self_zero = (
            jnp.eye(fractional.shape[0], dtype=bool)[:, :, None]
            & jnp.all(self.ewald.real_shifts == 0, axis=-1)[None, None, :]
        )
        real_coincidence = jnp.any((squared_distance == 0.0) & ~self_zero)
        distance = jnp.sqrt(jnp.where(squared_distance == 0.0, 1.0, squared_distance))
        alpha = jnp.asarray(self.ewald.screening, dtype=dtype)
        real_kernel = jnp.where(
            self_zero,
            0.0,
            jax.scipy.special.erfc(alpha * distance) / distance,
        )
        electron_electron = 0.5 * contract(
            "i,j,ijs->", electron_charge, electron_charge, real_kernel, backend="jax"
        )
        electron_nucleus = contract(
            "i,j,ijs->", electron_charge, nuclear_charge, real_kernel, backend="jax"
        )
        nucleus_nucleus = 0.5 * contract(
            "i,j,ijs->", nuclear_charge, nuclear_charge, real_kernel, backend="jax"
        )

        modes = self.ewald.reciprocal_modes.astype(dtype)
        wavevectors = contract(
            "ki,ij->kj",
            modes,
            self.cell.reciprocal_vectors.astype(dtype),
            backend="jax",
        )
        squared_wavevector = jnp.sum(wavevectors * wavevectors, axis=-1)
        phase = contract("ni,ki->nk", cartesian, wavevectors, backend="jax")
        phase_factor = jnp.exp(1.0j * phase)
        electron_structure = contract(
            "n,nk->k", electron_charge, phase_factor, backend="jax"
        )
        nuclear_structure = contract(
            "n,nk->k", nuclear_charge, phase_factor, backend="jax"
        )
        reciprocal_kernel = (
            jnp.exp(-squared_wavevector / (4.0 * alpha * alpha)) / squared_wavevector
        )
        volume = jnp.asarray(self.cell.volume, dtype=dtype)
        reciprocal_prefactor = 2.0 * jnp.pi / volume
        electron_electron = electron_electron + reciprocal_prefactor * jnp.sum(
            jnp.abs(electron_structure) ** 2 * reciprocal_kernel
        )
        electron_nucleus = electron_nucleus + reciprocal_prefactor * jnp.sum(
            2.0
            * jnp.real(electron_structure * jnp.conj(nuclear_structure))
            * reciprocal_kernel
        )
        nucleus_nucleus = nucleus_nucleus + reciprocal_prefactor * jnp.sum(
            jnp.abs(nuclear_structure) ** 2 * reciprocal_kernel
        )

        self_prefactor = -alpha / jnp.sqrt(jnp.pi)
        electron_electron = electron_electron + self_prefactor * jnp.sum(
            electron_charge * electron_charge
        )
        nucleus_nucleus = nucleus_nucleus + self_prefactor * jnp.sum(
            nuclear_charge * nuclear_charge
        )
        if self.ewald.uniform_background:
            background_prefactor = -jnp.pi / (2.0 * alpha * alpha * volume)
            electron_net = jnp.sum(electron_charge)
            nuclear_net = jnp.sum(nuclear_charge)
            electron_electron = electron_electron + background_prefactor * electron_net**2
            electron_nucleus = (
                electron_nucleus + 2.0 * background_prefactor * electron_net * nuclear_net
            )
            nucleus_nucleus = nucleus_nucleus + background_prefactor * nuclear_net**2

        length_factor = jnp.asarray(
            float(conversion_factor(self.nuclei.scale.length_unit, BOHR)), dtype=dtype
        )
        energy_factor = jnp.asarray(
            float(conversion_factor(self.nuclei.scale.energy_unit, HARTREE)), dtype=dtype
        )
        conversion = 1.0 / (length_factor * energy_factor)
        electron_electron = electron_electron * conversion
        electron_nucleus = electron_nucleus * conversion
        nucleus_nucleus = nucleus_nucleus * conversion
        total = electron_electron + electron_nucleus + nucleus_nucleus
        singular = periodic_coincidence | real_coincidence
        ewald_valid = (
            ~singular
            & jnp.isfinite(total)
            & jnp.isfinite(electron_electron)
            & jnp.isfinite(electron_nucleus)
            & jnp.isfinite(nucleus_nucleus)
        )
        return PeriodicElectronicPotential(
            electron_electron=electron_electron,
            electron_nucleus=electron_nucleus,
            nucleus_nucleus=nucleus_nucleus,
            total=total,
            singular=singular,
            ewald_valid=ewald_valid,
        )

    def potential(self, configurations: ArrayLike, /) -> PeriodicElectronicPotential:
        values = jnp.asarray(configurations)
        if values.ndim < 2 or tuple(values.shape[-2:]) != self.configuration_shape:
            raise ValueError(
                "Periodic electron configurations must end in "
                f"{self.configuration_shape}; got {values.shape}."
            )
        batch_shape = tuple(int(size) for size in values.shape[:-2])
        count = math.prod(batch_shape) if batch_shape else 1
        flat = values.reshape((count,) + self.configuration_shape)
        result = jax.vmap(self._potential_one)(flat)
        return PeriodicElectronicPotential(
            electron_electron=result.electron_electron.reshape(batch_shape),
            electron_nucleus=result.electron_nucleus.reshape(batch_shape),
            nucleus_nucleus=result.nucleus_nucleus.reshape(batch_shape),
            total=result.total.reshape(batch_shape),
            singular=result.singular.reshape(batch_shape),
            ewald_valid=result.ewald_valid.reshape(batch_shape),
        )

    def local_energy(
        self,
        model: AbstractPeriodicElectronicAmplitude,
        configurations: ArrayLike,
        /,
    ) -> PeriodicElectronicLocalEnergy:
        self._require_amplitude(model)
        values = jnp.asarray(configurations)
        potential = self.potential(values)
        batch_shape = tuple(int(size) for size in values.shape[:-2])
        count = math.prod(batch_shape) if batch_shape else 1
        flat = values.reshape((count,) + self.configuration_shape)
        kinetic, amplitude_valid = jax.vmap(
            lambda coordinates: self.kinetic.local_kinetic(model, coordinates)
        )(flat)
        length_factor = jnp.asarray(
            float(conversion_factor(self.nuclei.scale.length_unit, BOHR)),
            dtype=jnp.dtype(self.kinetic.compute_dtype),
        )
        energy_factor = jnp.asarray(
            float(conversion_factor(self.nuclei.scale.energy_unit, HARTREE)),
            dtype=jnp.dtype(self.kinetic.compute_dtype),
        )
        kinetic = (kinetic / (length_factor**2 * energy_factor)).reshape(batch_shape)
        amplitude_valid = amplitude_valid.reshape(batch_shape)
        raw_value = kinetic + potential.total
        finite = jnp.isfinite(raw_value)
        status = jnp.where(
            potential.singular,
            int(LocalOperatorStatus.SINGULAR_CONFIGURATION),
            jnp.where(
                ~amplitude_valid,
                int(LocalOperatorStatus.INVALID_AMPLITUDE),
                jnp.where(
                    ~(potential.ewald_valid & finite),
                    int(LocalOperatorStatus.NONFINITE),
                    int(LocalOperatorStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(LocalOperatorStatus.SUCCESS)
        return PeriodicElectronicLocalEnergy(
            kinetic=kinetic,
            potential=potential,
            value=jnp.where(
                valid, raw_value, jnp.asarray(jnp.nan, dtype=raw_value.dtype)
            ),
            amplitude_valid=amplitude_valid,
            valid=valid,
            status=status,
        )

    def estimate(
        self,
        model: AbstractPeriodicElectronicAmplitude,
        configurations: Array,
        /,
    ) -> LocalOperatorEstimate:
        local = self.local_energy(model, configurations)
        work = jnp.full(
            local.value.shape,
            self.resource_evidence.primitive_work_per_configuration,
            dtype=jnp.int32,
        )
        return LocalOperatorEstimate(
            local.value,
            local.valid,
            local.status,
            work,
            configuration_shape=self.configuration_shape,
            operator_id=self.operator_id,
            method_id=self.method_id,
            compute_dtype=self.kinetic.compute_dtype,
        )


__all__ = [
    "AbstractPeriodicElectronicAmplitude",
    "PeriodicElectronicCoulombHamiltonian",
    "PeriodicElectronicEwaldPolicy",
    "PeriodicElectronicLocalEnergy",
    "PeriodicElectronicPotential",
    "PeriodicElectronicResourceEvidence",
]
