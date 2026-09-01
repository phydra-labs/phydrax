#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparationReport
from ._classical import LennardJonesPotential
from ._constraints import DistanceConstraintPlan, PreparedDistanceConstraints
from ._electrostatics import (
    DirectCoulombPotential,
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticPotentialProgram,
    AtomisticTermEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._system import AtomisticSystemPlan, PreparedAtomisticSystem


class ForceFieldTermKind(StrEnum):
    HARMONIC_IMPROPER = "harmonic-improper"
    UREY_BRADLEY = "urey-bradley"
    TORSION_SERIES = "torsion-series"
    RYCKAERT_BELLEMANS = "ryckaert-bellemans"
    CMAP = "cmap"
    PAIR_OVERRIDE = "pair-override"
    MORSE = "morse"
    BUCKINGHAM = "buckingham"
    TABULATED_PAIR = "tabulated-pair"
    REACTION_FIELD = "reaction-field"
    DISPERSION_CORRECTION = "dispersion-correction"
    LENNARD_JONES_PME = "lennard-jones-pme"


class AtomisticForceFieldProvenance(StrictModule, NonTrainableState):
    source_format: str = eqx.field(static=True)
    source_digests: tuple[str, ...] = eqx.field(static=True)
    family: str = eqx.field(static=True)
    parameter_set: str = eqx.field(static=True)
    water_model: str | None = eqx.field(static=True)
    ion_model: str | None = eqx.field(static=True)
    typing_source: str = eqx.field(static=True)
    charge_source: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_format: str,
        source_digests: tuple[str, ...],
        family: str,
        parameter_set: str,
        /,
        *,
        water_model: str | None = None,
        ion_model: str | None = None,
        typing_source: str = "explicit",
        charge_source: str = "explicit",
        adapter_id: str = "native",
    ):
        values = tuple(
            str(value).strip()
            for value in (
                source_format,
                family,
                parameter_set,
                typing_source,
                charge_source,
                adapter_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Force-field provenance strings must be non-empty.")
        digests = tuple(str(value).strip() for value in source_digests)
        if not digests or any(not value for value in digests):
            raise ValueError("At least one non-empty source digest is required.")
        (
            self.source_format,
            self.family,
            self.parameter_set,
            self.typing_source,
            self.charge_source,
            self.adapter_id,
        ) = values
        self.source_digests = digests
        self.water_model = (
            None if water_model is None else str(water_model).strip() or None
        )
        self.ion_model = None if ion_model is None else str(ion_model).strip() or None
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "atomistic-force-field-provenance",
                "source_format": values[0],
                "source_digests": list(digests),
                "family": values[1],
                "parameter_set": values[2],
                "water_model": self.water_model,
                "ion_model": self.ion_model,
                "typing_source": values[3],
                "charge_source": values[4],
                "adapter_id": values[5],
            }
        )


class AtomisticNonbondedPolicy(StrictModule, NonTrainableState):
    cutoff: float = eqx.field(static=True)
    switch_distance: float | None = eqx.field(static=True)
    combining_rule: str = eqx.field(static=True)
    electrostatics: str = eqx.field(static=True)
    dispersion: str = eqx.field(static=True)
    charge_neutrality: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        cutoff: float,
        /,
        *,
        switch_distance: float | None = None,
        combining_rule: str = "lorentz-berthelot",
        electrostatics: str = "pme",
        dispersion: str = "cutoff",
        charge_neutrality: str = "require-neutral",
    ):
        cutoff_ = float(cutoff)
        switch = None if switch_distance is None else float(switch_distance)
        if (
            not math.isfinite(cutoff_)
            or cutoff_ <= 0.0
            or (
                switch is not None
                and (not math.isfinite(switch) or not 0.0 <= switch < cutoff_)
            )
        ):
            raise ValueError("Nonbonded cutoff or switch_distance is invalid.")
        if combining_rule not in ("lorentz-berthelot", "geometric", "explicit"):
            raise ValueError("Unknown nonbonded combining rule.")
        if electrostatics not in ("direct", "reaction-field", "ewald", "pme"):
            raise ValueError("Unknown electrostatic policy.")
        if dispersion not in ("cutoff", "tail-correction", "lj-pme"):
            raise ValueError("Unknown dispersion policy.")
        if charge_neutrality not in ("require-neutral", "uniform-background"):
            raise ValueError("Unknown charge-neutrality policy.")
        self.cutoff = cutoff_
        self.switch_distance = switch
        self.combining_rule = combining_rule
        self.electrostatics = electrostatics
        self.dispersion = dispersion
        self.charge_neutrality = charge_neutrality
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-nonbonded-policy",
                "cutoff": cutoff_,
                "switch_distance": switch,
                "combining_rule": combining_rule,
                "electrostatics": electrostatics,
                "dispersion": dispersion,
                "charge_neutrality": charge_neutrality,
            }
        )


def _validate_general_term_data(kind, arrays, routes, cutoff, /) -> None:
    host = tuple(np.asarray(value) for value in arrays)
    route_width = {
        ForceFieldTermKind.HARMONIC_IMPROPER: 4,
        ForceFieldTermKind.UREY_BRADLEY: 3,
        ForceFieldTermKind.TORSION_SERIES: 4,
        ForceFieldTermKind.RYCKAERT_BELLEMANS: 4,
        ForceFieldTermKind.CMAP: 8,
    }.get(kind)
    if route_width is not None and routes.size and routes.shape[1] != route_width:
        raise ValueError(f"{kind.value} routes must have width {route_width}.")
    if kind in (
        ForceFieldTermKind.HARMONIC_IMPROPER,
        ForceFieldTermKind.UREY_BRADLEY,
    ):
        if (
            len(host) != 2
            or host[0].ndim != 1
            or host[1].shape != host[0].shape
            or host[0].size == 0
            or np.any(host[0] <= 0.0)
        ):
            raise ValueError(f"{kind.value} requires positive aligned parameter vectors.")
    elif kind is ForceFieldTermKind.TORSION_SERIES:
        if (
            len(host) != 4
            or host[0].ndim != 2
            or any(value.shape != host[0].shape for value in host[1:])
            or host[0].shape[1] == 0
            or not np.issubdtype(host[1].dtype, np.integer)
            or np.any(host[1] <= 0)
        ):
            raise ValueError(
                "Torsion series tables must align with positive periodicities."
            )
    elif kind is ForceFieldTermKind.RYCKAERT_BELLEMANS:
        if len(host) != 1 or host[0].ndim != 2 or host[0].shape[1] != 6:
            raise ValueError(
                "Ryckaert-Bellemans requires one six-coefficient row per route."
            )
    elif kind is ForceFieldTermKind.CMAP:
        if (
            len(host) != 1
            or host[0].ndim != 2
            or host[0].shape[0] != host[0].shape[1]
            or host[0].shape[0] < 2
        ):
            raise ValueError("CMAP requires one square periodic grid.")
    elif kind in (
        ForceFieldTermKind.PAIR_OVERRIDE,
        ForceFieldTermKind.MORSE,
        ForceFieldTermKind.BUCKINGHAM,
    ):
        if (
            len(host) != 3
            or host[0].ndim != 2
            or host[0].shape[0] != host[0].shape[1]
            or any(value.shape != host[0].shape for value in host[1:])
        ):
            raise ValueError(f"{kind.value} requires aligned square pair tables.")
        if any(not np.allclose(value, value.T) for value in host):
            raise ValueError(f"{kind.value} pair tables must be symmetric.")
        if kind is ForceFieldTermKind.PAIR_OVERRIDE and (
            np.any(host[0] < 0.0) or np.any(host[1] <= 0.0)
        ):
            raise ValueError("Pair-override epsilon and sigma are invalid.")
        if kind is ForceFieldTermKind.MORSE and (
            np.any(host[0] < 0.0) or np.any(host[1] <= 0.0) or np.any(host[2] <= 0.0)
        ):
            raise ValueError("Morse pair parameters are invalid.")
        if kind is ForceFieldTermKind.BUCKINGHAM and (
            np.any(host[0] < 0.0) or np.any(host[1] <= 0.0) or np.any(host[2] < 0.0)
        ):
            raise ValueError("Buckingham pair parameters are invalid.")
    elif kind is ForceFieldTermKind.TABULATED_PAIR:
        if (
            len(host) != 2
            or host[0].ndim != 1
            or host[0].size < 2
            or np.any(np.diff(host[0]) <= 0.0)
            or host[1].ndim != 3
            or host[1].shape[0] != host[1].shape[1]
            or host[1].shape[2] != host[0].size
        ):
            raise ValueError(
                "Tabulated pairs require increasing radii and square tables."
            )
        if not np.allclose(host[1], np.swapaxes(host[1], 0, 1)):
            raise ValueError("Tabulated pair tables must be symmetric.")
    elif kind is ForceFieldTermKind.REACTION_FIELD:
        if (
            len(host) != 2
            or any(value.size != 1 for value in host)
            or float(host[0].reshape(())) <= 1.0
            or float(host[1].reshape(())) != cutoff
        ):
            raise ValueError(
                "Reaction field requires dielectric > 1 and matching cutoff."
            )
    elif kind is ForceFieldTermKind.DISPERSION_CORRECTION:
        if len(host) != 1 or host[0].size != 1 or float(host[0].reshape(())) < 0.0:
            raise ValueError("Dispersion correction coefficient must be non-negative.")
    elif kind is ForceFieldTermKind.LENNARD_JONES_PME:
        if (
            len(host) != 4
            or host[0].ndim != 2
            or host[0].shape[0] != host[0].shape[1]
            or host[1].size != 1
            or float(host[1].reshape(())) <= 0.0
            or host[2].ndim != 2
            or host[2].shape[0] != host[0].shape[0]
            or host[3].ndim != 3
            or any(size < 4 for size in host[3].shape)
        ):
            raise ValueError("Lennard-Jones PME arrays are invalid.")


class GeneralForceFieldTerm(AbstractAtomisticEnergyTerm, NonTrainableState):
    kind: ForceFieldTermKind = eqx.field(static=True)
    arrays: tuple[Array, ...]
    route_indices: Array
    cutoff: float | None = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        kind: ForceFieldTermKind,
        arrays: tuple[ArrayLike, ...],
        /,
        *,
        route_indices: ArrayLike | None = None,
        cutoff: float | None = None,
        name: str | None = None,
        force_group: int = 0,
    ):
        if not isinstance(kind, ForceFieldTermKind):
            raise TypeError("kind must be ForceFieldTermKind.")
        values = tuple(jnp.asarray(value) for value in arrays)
        if any(np.any(~np.isfinite(np.asarray(value))) for value in values):
            raise ValueError("Force-field parameter arrays must be finite.")
        routes = (
            np.zeros((0, 0), dtype=np.int32)
            if route_indices is None
            else np.asarray(route_indices)
        )
        if (
            routes.ndim != 2
            or not np.issubdtype(routes.dtype, np.integer)
            or np.any(routes < 0)
        ):
            raise TypeError("route_indices must be a non-negative rank-2 integer array.")
        cutoff_ = None if cutoff is None else float(cutoff)
        if cutoff_ is not None and (not math.isfinite(cutoff_) or cutoff_ <= 0.0):
            raise ValueError("cutoff must be finite and positive.")
        _validate_general_term_data(kind, arrays, routes, cutoff_)
        identifier = kind.value if name is None else str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError("Force-field term name or force group is invalid.")
        pair = kind in {
            ForceFieldTermKind.PAIR_OVERRIDE,
            ForceFieldTermKind.MORSE,
            ForceFieldTermKind.BUCKINGHAM,
            ForceFieldTermKind.TABULATED_PAIR,
            ForceFieldTermKind.REACTION_FIELD,
            ForceFieldTermKind.LENNARD_JONES_PME,
        }
        reciprocal = kind is ForceFieldTermKind.LENNARD_JONES_PME
        self.kind = kind
        self.arrays = values
        self.route_indices = jnp.asarray(routes, dtype=jnp.int32)
        self.cutoff = cutoff_
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            cutoff=cutoff_,
            pair_geometry=pair,
            interaction_site_geometry=pair,
            bonded_geometry=not pair
            and kind is not ForceFieldTermKind.DISPERSION_CORRECTION,
            reciprocal_grid=reciprocal,
        )
        self.term_id = canonical_fingerprint(
            {
                "kind": "general-force-field-term",
                "term_kind": kind.value,
                "arrays": array_tree_fingerprint(values),
                "routes": array_tree_fingerprint(routes),
                "cutoff": cutoff_,
                "name": identifier,
                "force_group": group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedGeneralForceFieldTerm":
        if (
            self.route_indices.size
            and int(np.max(np.asarray(self.route_indices))) >= system.capacity
        ):
            raise ValueError("Force-field route index exceeds atom capacity.")
        bonded_kinds = {
            ForceFieldTermKind.HARMONIC_IMPROPER,
            ForceFieldTermKind.UREY_BRADLEY,
            ForceFieldTermKind.TORSION_SERIES,
            ForceFieldTermKind.RYCKAERT_BELLEMANS,
            ForceFieldTermKind.CMAP,
        }
        route_count = (
            int(system.topology.improper_indices.shape[0])
            if self.kind is ForceFieldTermKind.HARMONIC_IMPROPER
            and not self.route_indices.size
            else int(self.route_indices.shape[0])
        )
        if self.kind in bonded_kinds and route_count == 0:
            raise ValueError(f"{self.kind.value} has no interaction routes.")
        if self.kind in (
            ForceFieldTermKind.HARMONIC_IMPROPER,
            ForceFieldTermKind.UREY_BRADLEY,
        ) and self.arrays[0].size not in (1, route_count):
            raise ValueError(f"{self.kind.value} parameters do not align with routes.")
        if self.kind is ForceFieldTermKind.TORSION_SERIES and self.arrays[0].shape[
            0
        ] not in (1, route_count):
            raise ValueError("Torsion-series rows do not align with routes.")
        if self.kind is ForceFieldTermKind.RYCKAERT_BELLEMANS and self.arrays[0].shape[
            0
        ] not in (1, route_count):
            raise ValueError("Ryckaert-Bellemans rows do not align with routes.")
        pair_kinds = {
            ForceFieldTermKind.PAIR_OVERRIDE,
            ForceFieldTermKind.MORSE,
            ForceFieldTermKind.BUCKINGHAM,
            ForceFieldTermKind.TABULATED_PAIR,
            ForceFieldTermKind.LENNARD_JONES_PME,
        }
        if self.kind in pair_kinds:
            maximum_type = int(
                np.max(
                    np.asarray(system.coordinate_map.plan.sites.site_type_ids)[
                        np.asarray(system.coordinate_map.plan.sites.active_mask)
                    ]
                )
            )
            table = (
                self.arrays[1]
                if self.kind is ForceFieldTermKind.TABULATED_PAIR
                else self.arrays[0]
            )
            if table.shape[0] <= maximum_type:
                raise ValueError(f"{self.kind.value} type table is too small.")
        if self.kind is ForceFieldTermKind.PAIR_OVERRIDE:
            site_capacity = system.coordinate_map.plan.sites.capacity
            if self.arrays[0].shape != (site_capacity, site_capacity):
                raise ValueError(
                    "Pair-override tables must match interaction-site capacity."
                )
        if self.kind is ForceFieldTermKind.LENNARD_JONES_PME:
            if system.cell is None or not system.cell.fully_periodic:
                raise ValueError("Lennard-Jones PME requires a fully periodic cell.")
            system.cell.require_unique_image(float(self.cutoff))
            c6, _, factors, _ = self.arrays
            type_count = (
                int(np.max(np.asarray(system.coordinate_map.plan.sites.site_type_ids)))
                + 1
            )
            if (
                c6.shape[0] < type_count
                or c6.shape[1] < type_count
                or factors.shape[0] < type_count
            ):
                raise ValueError("Lennard-Jones PME type table does not match sites.")
        return PreparedGeneralForceFieldTerm(self, system)


class PreparedGeneralForceFieldTerm(AbstractPreparedAtomisticEnergyTerm):
    plan: GeneralForceFieldTerm
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    site_anchor: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: GeneralForceFieldTerm, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.site_anchor = int(
            np.flatnonzero(np.asarray(system.coordinate_map.plan.sites.active_mask))[0]
        )
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-general-force-field-term",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    @staticmethod
    def _dihedral(points: Array) -> tuple[Array, Array]:
        b0 = points[:, 0] - points[:, 1]
        b1 = points[:, 2] - points[:, 1]
        b2 = points[:, 3] - points[:, 2]
        b1_norm = jnp.sqrt(jnp.sum(b1 * b1, axis=-1))
        axis = b1 / jnp.where(b1_norm[:, None] > 0.0, b1_norm[:, None], 1.0)
        v = b0 - jnp.sum(b0 * axis, axis=-1)[:, None] * axis
        w = b2 - jnp.sum(b2 * axis, axis=-1)[:, None] * axis
        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(axis, v) * w, axis=-1)
        valid = (
            (b1_norm > 0.0)
            & (jnp.sum(v * v, axis=-1) > 0.0)
            & (jnp.sum(w * w, axis=-1) > 0.0)
        )
        return jnp.arctan2(y, x), valid

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        dtype = context.positions.dtype
        zero_atoms = jnp.zeros((self.system.capacity,), dtype=dtype)
        kind = self.plan.kind
        routes = self.plan.route_indices
        arrays = self.plan.arrays
        success = jnp.asarray(True)
        if kind in {
            ForceFieldTermKind.HARMONIC_IMPROPER,
            ForceFieldTermKind.TORSION_SERIES,
            ForceFieldTermKind.RYCKAERT_BELLEMANS,
            ForceFieldTermKind.CMAP,
        }:
            if kind is ForceFieldTermKind.HARMONIC_IMPROPER and routes.size == 0:
                routes = self.system.topology.improper_indices
            points = context.unwrapped_positions[routes]
            phi, valid = self._dihedral(points[:, :4])
            if kind is ForceFieldTermKind.HARMONIC_IMPROPER:
                stiffness, target = arrays
                delta = jnp.arctan2(jnp.sin(phi - target), jnp.cos(phi - target))
                values = 0.5 * stiffness * delta * delta
            elif kind is ForceFieldTermKind.TORSION_SERIES:
                amplitude, periodicity, phase, mask = arrays
                values = jnp.sum(
                    mask
                    * amplitude
                    * (1.0 + jnp.cos(periodicity * phi[:, None] - phase)),
                    axis=-1,
                )
            elif kind is ForceFieldTermKind.RYCKAERT_BELLEMANS:
                coefficients = arrays[0]
                cosine = jnp.cos(phi - jnp.pi)
                powers = jnp.stack(
                    tuple(cosine**power for power in range(coefficients.shape[-1])),
                    axis=-1,
                )
                values = jnp.sum(coefficients * powers, axis=-1)
            else:
                grid = arrays[0]
                second_phi, second_valid = self._dihedral(points[:, 4:8])
                size_x, size_y = grid.shape[-2:]
                x = (phi + jnp.pi) * size_x / (2.0 * jnp.pi)
                y = (second_phi + jnp.pi) * size_y / (2.0 * jnp.pi)
                ix = jnp.floor(x).astype(jnp.int32) % size_x
                iy = jnp.floor(y).astype(jnp.int32) % size_y
                fx, fy = x - jnp.floor(x), y - jnp.floor(y)
                values = (
                    (1 - fx) * (1 - fy) * grid[ix, iy]
                    + fx * (1 - fy) * grid[(ix + 1) % size_x, iy]
                    + (1 - fx) * fy * grid[ix, (iy + 1) % size_y]
                    + fx * fy * grid[(ix + 1) % size_x, (iy + 1) % size_y]
                )
                valid = valid & second_valid
            success = jnp.all(valid & jnp.isfinite(values))
            energy = jnp.sum(values)
        elif kind is ForceFieldTermKind.UREY_BRADLEY:
            points = context.unwrapped_positions[routes]
            distance = jnp.sqrt(jnp.sum((points[:, 0] - points[:, 2]) ** 2, axis=-1))
            stiffness, target = arrays
            values = 0.5 * stiffness * (distance - target) ** 2
            success = jnp.all((distance > 0.0) & jnp.isfinite(values))
            energy = jnp.sum(values)
        elif kind is ForceFieldTermKind.DISPERSION_CORRECTION:
            coefficient = arrays[0].reshape(())
            volume = (
                jnp.abs(
                    jnp.sum(
                        context.cell_vectors[0]
                        * jnp.cross(context.cell_vectors[1], context.cell_vectors[2])
                    )
                )
                if context.cell_vectors.size
                else jnp.asarray(jnp.inf, dtype=dtype)
            )
            energy = -coefficient / volume
            success = jnp.isfinite(energy)
        else:
            left, right = context.site_pair_left, context.site_pair_right
            distance = context.site_pair_distance
            valid = context.site_pair_valid
            cutoff = jnp.asarray(
                jnp.inf if self.plan.cutoff is None else self.plan.cutoff, dtype=dtype
            )
            active = valid & (distance > 0.0) & (distance < cutoff)
            safe = jnp.where(active, distance, 1.0)
            left_type = context.site_type_ids[left]
            right_type = context.site_type_ids[right]
            if kind is ForceFieldTermKind.MORSE:
                depth, alpha, equilibrium = arrays
                d = depth[left_type, right_type]
                a = alpha[left_type, right_type]
                r0 = equilibrium[left_type, right_type]
                values = d * ((1.0 - jnp.exp(-a * (safe - r0))) ** 2 - 1.0)
            elif kind is ForceFieldTermKind.BUCKINGHAM:
                amplitude, decay, c6 = arrays
                values = (
                    amplitude[left_type, right_type]
                    * jnp.exp(-decay[left_type, right_type] * safe)
                    - c6[left_type, right_type] / safe**6
                )
            elif kind is ForceFieldTermKind.TABULATED_PAIR:
                radii, table = arrays
                x = jnp.clip(jnp.searchsorted(radii, safe) - 1, 0, radii.size - 2)
                fraction = (safe - radii[x]) / (radii[x + 1] - radii[x])
                values = (1.0 - fraction) * table[
                    left_type, right_type, x
                ] + fraction * table[left_type, right_type, x + 1]
            elif kind is ForceFieldTermKind.REACTION_FIELD:
                dielectric, cutoff_value = arrays
                charge_product = context.site_charges[left] * context.site_charges[right]
                epsilon = dielectric.reshape(())
                rc = cutoff_value.reshape(())
                krf = (epsilon - 1.0) / (2.0 * epsilon + 1.0) / rc**3
                crf = 3.0 * epsilon / (2.0 * epsilon + 1.0) / rc
                values = (
                    self.system.plan.units.coulomb_constant
                    * charge_product
                    * (1.0 / safe + krf * safe**2 - crf)
                )
            elif kind is ForceFieldTermKind.PAIR_OVERRIDE:
                epsilon, sigma, pair_mask = arrays
                pair_index = left * self.system.coordinate_map.plan.sites.capacity + right
                eps = epsilon.reshape((-1,))[pair_index]
                sig = sigma.reshape((-1,))[pair_index]
                ratio6 = (sig / safe) ** 6
                values = (
                    pair_mask.reshape((-1,))[pair_index]
                    * 4.0
                    * eps
                    * (ratio6**2 - ratio6)
                )
            elif kind is ForceFieldTermKind.LENNARD_JONES_PME:
                c6, alpha_value, factors, _ = arrays
                coefficient = c6[left_type, right_type]
                alpha = alpha_value.reshape(())
                scaled_distance = alpha * safe
                screening = jnp.exp(-(scaled_distance**2)) * (
                    1.0 + scaled_distance**2 + 0.5 * scaled_distance**4
                )
                values = -coefficient * screening / safe**6
            else:
                raise RuntimeError(f"Unhandled force-field term kind {kind.value!r}.")
            if kind is ForceFieldTermKind.REACTION_FIELD:
                values = values * context.site_electrostatic_scales
            elif kind is ForceFieldTermKind.LENNARD_JONES_PME:
                full_dispersion = -coefficient / safe**6
                values = (
                    values + (context.site_lennard_jones_scales - 1.0) * full_dispersion
                )
            else:
                values = values * context.site_lennard_jones_scales
            values = jnp.where(active, values, 0.0)
            success = jnp.all(~valid | ((distance > 0.0) & jnp.isfinite(values)))
            energy = jnp.sum(values)
            if kind is ForceFieldTermKind.LENNARD_JONES_PME:
                c6, alpha_value, factors, grid_template = arrays
                alpha = alpha_value.reshape(())
                vectors = context.cell_vectors
                determinant = jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2]))
                inverse = (
                    jnp.stack(
                        (
                            jnp.cross(vectors[1], vectors[2]),
                            jnp.cross(vectors[2], vectors[0]),
                            jnp.cross(vectors[0], vectors[1]),
                        ),
                        axis=1,
                    )
                    / determinant
                )
                grid_shape = grid_template.shape
                shape = jnp.asarray(grid_shape, dtype=context.site_positions.dtype)
                fractional = contract("nd,di->ni", context.site_positions, inverse)
                anchor_index = self.site_anchor
                relative_fractional = fractional - fractional[anchor_index]
                scaled_position = jnp.mod(relative_fractional + 0.5, 1.0) * shape
                base = jax.lax.stop_gradient(jnp.floor(scaled_position).astype(jnp.int32))
                remainder = scaled_position - base
                channel_count = factors.shape[1]
                grid = jnp.zeros(
                    grid_shape + (channel_count,), dtype=context.site_positions.dtype
                )
                site_factor = factors[context.site_type_ids]
                site_factor = site_factor * context.site_state.active_mask[:, None]
                for x_offset in (0, 1):
                    for y_offset in (0, 1):
                        for z_offset in (0, 1):
                            corner = jnp.asarray((x_offset, y_offset, z_offset))
                            axis_weight = jnp.where(
                                corner[None, :] == 1,
                                remainder,
                                1.0 - remainder,
                            )
                            weight = jnp.prod(axis_weight, axis=-1)
                            index = (base + corner[None, :]) % jnp.asarray(
                                grid_shape, dtype=jnp.int32
                            )
                            grid = grid.at[index[:, 0], index[:, 1], index[:, 2]].add(
                                weight[:, None] * site_factor
                            )
                modes_by_axis = tuple(jnp.fft.fftfreq(size) * size for size in grid_shape)
                mode_components = jnp.meshgrid(*modes_by_axis, indexing="ij")
                modes = jnp.stack(mode_components, axis=-1)
                wave = 2.0 * jnp.pi * contract("...i,ji->...j", modes, inverse)
                squared_wave = jnp.sum(wave * wave, axis=-1)
                transformed = jnp.fft.fftn(grid, axes=(0, 1, 2))
                window = jnp.prod(
                    jnp.stack(
                        tuple(
                            jnp.sinc(mode_components[axis] / grid_shape[axis]) ** 2
                            for axis in range(3)
                        ),
                        axis=-1,
                    ),
                    axis=-1,
                )
                transformed = transformed / jnp.where(
                    jnp.abs(window[..., None]) > 0.0,
                    window[..., None],
                    1.0,
                )
                structure_power = jnp.sum(
                    jnp.real(transformed * jnp.conj(transformed)), axis=-1
                )
                safe_wave = jnp.where(squared_wave > 0.0, squared_wave, 1.0)
                reciprocal_coordinate = jnp.sqrt(safe_wave) / (2.0 * alpha)
                reciprocal_kernel = (
                    (1.0 - 2.0 * reciprocal_coordinate**2)
                    * jnp.exp(-(reciprocal_coordinate**2))
                    + 2.0
                    * jnp.sqrt(jnp.pi)
                    * reciprocal_coordinate**3
                    * jsp.erfc(reciprocal_coordinate)
                ) / 3.0
                reciprocal_kernel = jnp.where(squared_wave > 0.0, reciprocal_kernel, 0.0)
                volume = jnp.abs(determinant)
                reciprocal = (
                    jnp.pi**1.5
                    * alpha**3
                    / (2.0 * volume)
                    * jnp.sum(reciprocal_kernel * structure_power)
                )
                diagonal = jnp.diag(c6)[context.site_type_ids]
                self_energy = (
                    -(alpha**6)
                    / 12.0
                    * jnp.sum(jnp.where(context.site_state.active_mask, diagonal, 0.0))
                )
                energy = energy + reciprocal + self_energy
                success = (
                    success
                    & jnp.isfinite(volume)
                    & (volume > 0.0)
                    & jnp.isfinite(reciprocal)
                    & jnp.isfinite(self_energy)
                )
        return AtomisticTermEvaluation(
            jnp.where(success, energy, jnp.nan), zero_atoms, success
        )


def _term(kind, arrays, *, routes=None, cutoff=None, name=None, force_group=0):
    return GeneralForceFieldTerm(
        kind,
        tuple(arrays),
        route_indices=routes,
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def HarmonicImproperPotential(
    stiffness, target, *, routes=None, name=None, force_group=0
):
    return _term(
        ForceFieldTermKind.HARMONIC_IMPROPER,
        (stiffness, target),
        routes=routes,
        name=name,
        force_group=force_group,
    )


def UreyBradleyPotential(stiffness, target, routes, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.UREY_BRADLEY,
        (stiffness, target),
        routes=routes,
        name=name,
        force_group=force_group,
    )


def PeriodicTorsionSeriesPotential(
    amplitude, periodicity, phase, mask, routes, *, name=None, force_group=0
):
    return _term(
        ForceFieldTermKind.TORSION_SERIES,
        (amplitude, periodicity, phase, mask),
        routes=routes,
        name=name,
        force_group=force_group,
    )


def RyckaertBellemansPotential(coefficients, routes, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.RYCKAERT_BELLEMANS,
        (coefficients,),
        routes=routes,
        name=name,
        force_group=force_group,
    )


def CMAPPotential(grid, routes, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.CMAP,
        (grid,),
        routes=routes,
        name=name,
        force_group=force_group,
    )


def PairOverrideLennardJonesPotential(
    epsilon, sigma, pair_mask, cutoff, *, name=None, force_group=0
):
    return _term(
        ForceFieldTermKind.PAIR_OVERRIDE,
        (epsilon, sigma, pair_mask),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def MorsePotential(depth, alpha, equilibrium, cutoff, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.MORSE,
        (depth, alpha, equilibrium),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def BuckinghamPotential(amplitude, decay, c6, cutoff, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.BUCKINGHAM,
        (amplitude, decay, c6),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def TabulatedPairPotential(radii, values, cutoff, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.TABULATED_PAIR,
        (radii, values),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def ReactionFieldPotential(dielectric, cutoff, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.REACTION_FIELD,
        (jnp.asarray(dielectric), jnp.asarray(cutoff)),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


def LennardJonesDispersionCorrection(coefficient, *, name=None, force_group=0):
    return _term(
        ForceFieldTermKind.DISPERSION_CORRECTION,
        (jnp.asarray(coefficient),),
        name=name,
        force_group=force_group,
    )


def LennardJonesPMEPotential(c6, alpha, cutoff, grid_shape, *, name=None, force_group=0):
    matrix = np.asarray(c6, dtype=float)
    shape = tuple(int(value) for value in grid_shape)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.allclose(matrix, matrix.T)
        or np.any(~np.isfinite(matrix))
        or len(shape) != 3
        or any(value < 4 for value in shape)
        or float(alpha) <= 0.0
    ):
        raise ValueError("Lennard-Jones PME coefficients, alpha, or grid are invalid.")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = np.finfo(float).eps * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Lennard-Jones PME C6 matrix must be positive semidefinite.")
    positive = eigenvalues > tolerance
    factors = (
        eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])[None, :]
        if np.any(positive)
        else np.zeros((matrix.shape[0], 1))
    )
    return _term(
        ForceFieldTermKind.LENNARD_JONES_PME,
        (
            matrix,
            jnp.asarray(alpha),
            factors,
            jnp.zeros(shape),
        ),
        cutoff=cutoff,
        name=name,
        force_group=force_group,
    )


class AtomisticForceFieldPlan(StrictModule):
    system: AtomisticSystemPlan
    potential: AtomisticPotentialProgram
    nonbonded: AtomisticNonbondedPolicy
    provenance: AtomisticForceFieldProvenance
    constraint_plan: DistanceConstraintPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        potential: AtomisticPotentialProgram,
        nonbonded: AtomisticNonbondedPolicy,
        provenance: AtomisticForceFieldProvenance,
        /,
        *,
        constraint_plan: DistanceConstraintPlan | None = None,
    ):
        if not isinstance(system, AtomisticSystemPlan) or not isinstance(
            potential, AtomisticPotentialProgram
        ):
            raise TypeError(
                "Force-field plan requires atomistic system and potential plans."
            )
        if not isinstance(nonbonded, AtomisticNonbondedPolicy) or not isinstance(
            provenance, AtomisticForceFieldProvenance
        ):
            raise TypeError("Invalid force-field policy or provenance.")
        if constraint_plan is not None and not isinstance(
            constraint_plan, DistanceConstraintPlan
        ):
            raise TypeError("constraint_plan must be DistanceConstraintPlan or None.")
        for term in potential.terms:
            if isinstance(term, LennardJonesPotential) and (
                term.cutoff != nonbonded.cutoff
                or term.switch_distance != nonbonded.switch_distance
                or term.combining_rule != nonbonded.combining_rule
            ):
                raise ValueError("Lennard-Jones term disagrees with nonbonded policy.")
            if isinstance(term, DirectCoulombPotential) and (
                nonbonded.electrostatics != "direct"
            ):
                raise ValueError("Direct Coulomb term disagrees with nonbonded policy.")
            if isinstance(term, EwaldReferencePotential) and (
                nonbonded.electrostatics != "ewald"
                or term.real_cutoff != nonbonded.cutoff
            ):
                raise ValueError("Ewald term disagrees with nonbonded policy.")
            if isinstance(term, ParticleMeshEwaldPotential) and (
                nonbonded.electrostatics != "pme" or term.real_cutoff != nonbonded.cutoff
            ):
                raise ValueError("PME term disagrees with nonbonded policy.")
            if isinstance(term, GeneralForceFieldTerm):
                if term.cutoff is not None and term.cutoff != nonbonded.cutoff:
                    raise ValueError(
                        f"Term {term.name!r} cutoff disagrees with nonbonded policy."
                    )
                if (
                    term.kind is ForceFieldTermKind.REACTION_FIELD
                    and nonbonded.electrostatics != "reaction-field"
                ):
                    raise ValueError(
                        "Reaction-field term disagrees with nonbonded policy."
                    )
                if (
                    term.kind is ForceFieldTermKind.DISPERSION_CORRECTION
                    and nonbonded.dispersion != "tail-correction"
                ):
                    raise ValueError(
                        "Dispersion correction disagrees with nonbonded policy."
                    )
                if (
                    term.kind is ForceFieldTermKind.LENNARD_JONES_PME
                    and nonbonded.dispersion != "lj-pme"
                ):
                    raise ValueError("LJ-PME term disagrees with nonbonded policy.")
        if np.any(np.asarray(system.charges) != 0.0):
            electrostatic_terms = tuple(
                term
                for term in potential.terms
                if isinstance(
                    term,
                    (
                        DirectCoulombPotential,
                        EwaldReferencePotential,
                        ParticleMeshEwaldPotential,
                    ),
                )
                or (
                    isinstance(term, GeneralForceFieldTerm)
                    and term.kind is ForceFieldTermKind.REACTION_FIELD
                )
            )
            if len(electrostatic_terms) != 1:
                raise ValueError(
                    "Charged force fields require exactly one electrostatic term."
                )
        dispersion_kind_present = {
            term.kind
            for term in potential.terms
            if isinstance(term, GeneralForceFieldTerm)
        }
        if (
            nonbonded.dispersion == "tail-correction"
            and ForceFieldTermKind.DISPERSION_CORRECTION not in dispersion_kind_present
        ):
            raise ValueError("Tail-correction policy requires a dispersion term.")
        if (
            nonbonded.dispersion == "lj-pme"
            and ForceFieldTermKind.LENNARD_JONES_PME not in dispersion_kind_present
        ):
            raise ValueError("LJ-PME policy requires an LJ-PME term.")
        self.system = system
        self.potential = potential
        self.nonbonded = nonbonded
        self.provenance = provenance
        self.constraint_plan = constraint_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-force-field-plan",
                "system": system.system_id,
                "potential": potential.program_id,
                "nonbonded": nonbonded.policy_id,
                "provenance": provenance.provenance_id,
                "constraints": None
                if constraint_plan is None
                else constraint_plan.plan_id,
            }
        )

    def prepare(
        self, /, *, graph_execution=None, numeric_version: str = "0"
    ) -> "PreparedAtomisticForceField":
        system = self.system.prepare(numeric_version=numeric_version)
        potential = self.potential.prepare(system, graph_execution=graph_execution)
        constraints = (
            None if self.constraint_plan is None else self.constraint_plan.prepare(system)
        )
        counts = {
            "dof_atoms": system.capacity,
            "interaction_sites": system.coordinate_map.plan.sites.capacity,
            "virtual_sites": int(
                np.count_nonzero(
                    ~np.asarray(system.coordinate_map.plan.sites.physical_mask)
                    & np.asarray(system.coordinate_map.plan.sites.active_mask)
                )
            ),
            "terms": len(potential.terms),
            "constraints": system.topology.constraint_count,
        }
        report = PreparationReport(
            diagnostics=(
                "complete force-field bundle",
                "explicit source provenance",
                "unsupported terms are rejected by adapters",
            ),
            resource_counts=counts,
        )
        return PreparedAtomisticForceField(self, system, potential, constraints, report)


class PreparedAtomisticForceField(StrictModule):
    plan: AtomisticForceFieldPlan
    system: PreparedAtomisticSystem
    potential: PreparedAtomisticPotentialProgram
    constraints: PreparedDistanceConstraints | None
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, system, potential, constraints, preparation, /):
        self.plan = plan
        self.system = system
        self.potential = potential
        self.constraints = constraints
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-force-field",
                "plan": plan.plan_id,
                "system": system.prepared_id,
                "potential": potential.prepared_id,
                "preparation": preparation.report_id,
            }
        )


class SETTLEPlan(StrictModule, NonTrainableState):
    water_groups: Array
    oxygen_hydrogen_distance: float = eqx.field(static=True)
    hydrogen_hydrogen_distance: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        water_groups: ArrayLike,
        oxygen_hydrogen_distance: float,
        hydrogen_hydrogen_distance: float,
        /,
        *,
        tolerance: float = 1e-10,
    ):
        groups = np.asarray(water_groups)
        if (
            groups.ndim != 2
            or groups.shape[1] != 3
            or not np.issubdtype(groups.dtype, np.integer)
        ):
            raise TypeError("water_groups must have shape (N,3) integer indices.")
        oh, hh, tol = (
            float(oxygen_hydrogen_distance),
            float(hydrogen_hydrogen_distance),
            float(tolerance),
        )
        if min(oh, hh, tol) <= 0.0 or hh >= 2.0 * oh:
            raise ValueError("SETTLE geometry is invalid.")
        self.water_groups = jnp.asarray(groups, dtype=jnp.int32)
        self.oxygen_hydrogen_distance = oh
        self.hydrogen_hydrogen_distance = hh
        self.tolerance = tol
        self.plan_id = canonical_fingerprint(
            {
                "kind": "settle-plan",
                "groups": array_tree_fingerprint(groups),
                "oh": oh,
                "hh": hh,
                "tolerance": tol,
            }
        )

    def distance_constraint_plan(
        self, system: PreparedAtomisticSystem, /
    ) -> DistanceConstraintPlan:
        del system
        return DistanceConstraintPlan(maximum_iterations=64, tolerance=self.tolerance)

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedSETTLE":
        if self.water_groups.size and int(jnp.max(self.water_groups)) >= system.capacity:
            raise ValueError("SETTLE water index exceeds atom capacity.")
        return PreparedSETTLE(self, system)


class SETTLEProjection(StrictModule):
    positions: Array
    momenta: Array
    position_residual: Array
    velocity_residual: Array
    successful: Array


class PreparedSETTLE(StrictModule, NonTrainableState):
    plan: SETTLEPlan
    system: PreparedAtomisticSystem
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SETTLEPlan, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-settle",
                "plan": plan.plan_id,
                "system": system.prepared_id,
            }
        )

    def project(self, positions: ArrayLike, momenta: ArrayLike, /) -> SETTLEProjection:
        q = jnp.asarray(positions)
        p = jnp.asarray(momenta, dtype=q.dtype)
        result_q = q
        result_p = p
        maximum_position = jnp.zeros((), dtype=q.dtype)
        maximum_velocity = jnp.zeros((), dtype=q.dtype)
        success = jnp.asarray(True)
        for group in np.asarray(self.plan.water_groups):
            index = jnp.asarray(group, dtype=jnp.int32)
            water = result_q[index]
            masses = self.system.plan.masses[index]
            total_mass = jnp.sum(masses)
            center = jnp.sum(masses[:, None] * water, axis=0) / total_mass
            midpoint = 0.5 * (water[1] + water[2])
            x_raw = midpoint - water[0]
            y_raw = water[1] - water[2]
            x_norm = jnp.sqrt(jnp.sum(x_raw**2))
            y_norm = jnp.sqrt(jnp.sum(y_raw**2))
            x_axis = x_raw / jnp.where(x_norm > 0.0, x_norm, 1.0)
            y_axis = y_raw - jnp.sum(y_raw * x_axis) * x_axis
            y_axis = y_axis / jnp.where(
                jnp.sqrt(jnp.sum(y_axis**2)) > 0.0,
                jnp.sqrt(jnp.sum(y_axis**2)),
                1.0,
            )
            half_hh = 0.5 * self.plan.hydrogen_hydrogen_distance
            along = jnp.sqrt(self.plan.oxygen_hydrogen_distance**2 - half_hh**2)
            reference = jnp.stack(
                (
                    jnp.zeros((3,), dtype=q.dtype),
                    along * x_axis + half_hh * y_axis,
                    along * x_axis - half_hh * y_axis,
                )
            )
            reference_center = jnp.sum(masses[:, None] * reference, axis=0) / total_mass
            projected = center + reference - reference_center
            result_q = result_q.at[index].set(projected)
            inverse_mass = self.system.inverse_masses[index]
            water_p = result_p[index]
            pairs = ((0, 1), (0, 2), (1, 2))
            directions = jnp.stack(
                tuple(projected[left] - projected[right] for left, right in pairs)
            )
            jacobian = jnp.zeros((3, 3, 3), dtype=q.dtype)
            for pair_index, (left, right) in enumerate(pairs):
                jacobian = jacobian.at[pair_index, left].set(directions[pair_index])
                jacobian = jacobian.at[pair_index, right].set(-directions[pair_index])
            gram = contract("api,p,bpi->ab", jacobian, inverse_mass, jacobian)
            determinant = jnp.sum(gram[0] * jnp.cross(gram[1], gram[2]))
            inverse_gram = jnp.stack(
                (
                    jnp.cross(gram[1], gram[2]),
                    jnp.cross(gram[2], gram[0]),
                    jnp.cross(gram[0], gram[1]),
                ),
                axis=1,
            ) / jnp.where(jnp.abs(determinant) > 0.0, determinant, 1.0)
            velocity = water_p * inverse_mass[:, None]
            residual = contract("api,pi->a", jacobian, velocity)
            multipliers = -contract("ab,b->a", inverse_gram, residual)
            water_p = water_p + contract("a,api->pi", multipliers, jacobian)
            momentum_nonsingular = jnp.abs(determinant) > jnp.finfo(q.dtype).tiny
            result_p = result_p.at[index].set(water_p)
            oh1 = jnp.sqrt(jnp.sum((projected[0] - projected[1]) ** 2))
            oh2 = jnp.sqrt(jnp.sum((projected[0] - projected[2]) ** 2))
            hh = jnp.sqrt(jnp.sum((projected[1] - projected[2]) ** 2))
            position_residual = jnp.max(
                jnp.abs(
                    jnp.asarray(
                        [
                            oh1 - self.plan.oxygen_hydrogen_distance,
                            oh2 - self.plan.oxygen_hydrogen_distance,
                            hh - self.plan.hydrogen_hydrogen_distance,
                        ]
                    )
                )
            )
            velocity_residual = jnp.zeros((), dtype=q.dtype)
            for left, right in pairs:
                displacement = projected[left] - projected[right]
                relative_velocity = (
                    water_p[left] * inverse_mass[left]
                    - water_p[right] * inverse_mass[right]
                )
                velocity_residual = jnp.maximum(
                    velocity_residual,
                    jnp.abs(jnp.sum(displacement * relative_velocity)),
                )
            maximum_position = jnp.maximum(maximum_position, position_residual)
            maximum_velocity = jnp.maximum(maximum_velocity, velocity_residual)
            success = (
                success
                & (x_norm > 0.0)
                & (y_norm > 0.0)
                & momentum_nonsingular
                & (position_residual <= self.plan.tolerance)
                & (velocity_residual <= self.plan.tolerance)
            )
        return SETTLEProjection(
            result_q,
            result_p,
            maximum_position,
            maximum_velocity,
            success & jnp.all(jnp.isfinite(result_q)),
        )


__all__ = [
    "AtomisticForceFieldPlan",
    "AtomisticForceFieldProvenance",
    "AtomisticNonbondedPolicy",
    "BuckinghamPotential",
    "CMAPPotential",
    "ForceFieldTermKind",
    "GeneralForceFieldTerm",
    "HarmonicImproperPotential",
    "LennardJonesDispersionCorrection",
    "LennardJonesPMEPotential",
    "MorsePotential",
    "PairOverrideLennardJonesPotential",
    "PeriodicTorsionSeriesPotential",
    "PreparedAtomisticForceField",
    "PreparedSETTLE",
    "ReactionFieldPotential",
    "RyckaertBellemansPotential",
    "SETTLEPlan",
    "SETTLEProjection",
    "TabulatedPairPotential",
    "UreyBradleyPotential",
]
