#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conditional single-site rotamer free energy, not a calibrated folding model.

Each caller-defined rotamer is a point in a right-handed backbone frame. The
analytical profile has unary energies and fixed-support Gaussian pair energies
A exp(-r²/(2 sigma²)). No side-chain atoms or second molecular topology are made.
All tables, geometry, chemistry applicability and their rights come from callers.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.atomistic._potential import (
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from phydrax.atomistic._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.ein import contract
from phydrax.pgm import (
    AdvancedBeliefPropagationResult,
    BeliefPropagationState,
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    ExactFactorGraphResult,
    initialize_belief_propagation,
    prepare_belief_propagation,
    prepare_exact_factor_graph,
    PreparedBeliefPropagation,
    PreparedExactFactorGraph,
    replace_belief_propagation_tables,
    run_exact_factor_graph,
    run_implicit_belief_propagation,
    SumProductBeliefPropagation,
    VariableSelection,
)
from phydrax.qualification import ReferenceArtifactManifest

from .._construct import ProteinConstruct


def _finite_array(value: ArrayLike, name: str) -> Array:
    host = np.asarray(value)
    if (
        not np.issubdtype(host.dtype, np.number)
        or np.iscomplexobj(host)
        or not np.all(np.isfinite(host))
    ):
        raise ValueError(f"{name} must contain finite real values.")
    return jnp.asarray(host, dtype=float)


def _admit(source, commercial_use, redistribution, training_use, export):
    if not isinstance(source, ReferenceArtifactManifest):
        raise TypeError("source must be ReferenceArtifactManifest.")
    return source.require_rights(
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )


class RotamerGeometryPlan(StrictModule):
    """Stable-atom backbone frames and heterogeneous permitted local rotamer sites.

    Frame IDs are (origin, positive-x atom, positive-xy atom), in construct residue
    order. All three must be distinct active atoms at preparation. Local site
    coordinates are in the declared atomistic length unit; they are not atom IDs.
    """

    construct: ProteinConstruct = eqx.field(static=True)
    frame_atom_ids: Array
    local_sites: tuple[Array, ...]
    source: ReferenceArtifactManifest
    units: AtomisticUnitSystem
    cardinalities: tuple[int, ...] = eqx.field(static=True)
    minimum_frame_length: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        construct: ProteinConstruct,
        frame_atom_ids: ArrayLike,
        local_sites: tuple[ArrayLike, ...],
        source: ReferenceArtifactManifest,
        /,
        *,
        units: AtomisticUnitSystem,
        minimum_frame_length: float = 1e-8,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        if not isinstance(construct, ProteinConstruct):
            raise TypeError("construct must be ProteinConstruct.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        ids = np.asarray(frame_atom_ids)
        if ids.shape != (construct.residue_count, 3) or not np.issubdtype(
            ids.dtype, np.integer
        ):
            raise ValueError(
                "frame_atom_ids must be integer (residue_count, 3) stable IDs."
            )
        if any(len(set(row.tolist())) != 3 for row in ids):
            raise ValueError("Every backbone frame requires three distinct atoms.")
        sites = tuple(_finite_array(value, "local_sites") for value in local_sites)
        if (
            len(sites) != construct.residue_count
            or not sites
            or any(
                value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 1
                for value in sites
            )
        ):
            raise ValueError(
                "Supply one nonempty (valid_rotamers, 3) site table per residue."
            )
        threshold = float(minimum_frame_length)
        if not isfinite(threshold) or threshold <= 0:
            raise ValueError("minimum_frame_length must be positive and finite.")
        source_id = _admit(source, commercial_use, redistribution, training_use, export)
        self.construct = construct
        self.frame_atom_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.local_sites = sites
        self.source = source
        self.units = units
        self.cardinalities = tuple(int(value.shape[0]) for value in sites)
        self.minimum_frame_length = threshold
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "protein-single-site-rotamer-geometry",
                "construct": construct.fingerprint(),
                "frame_ids": ids.tolist(),
                "sites": array_tree_fingerprint(sites),
                "minimum_frame_length": threshold,
                "source": source_id,
                "length_unit": units.scale.length_unit.unit_id,
            }
        )


class RotamerParameterPlan(StrictModule):
    """Source-pinned caller parameters for the explicitly named Gaussian profile.

    Energies are single-system energies in units.scale.energy_unit, not molar
    energies; widths and local geometry use units.scale.length_unit. Temperature
    is absolute, in units.temperature_unit. Unknown model uncertainty may remain
    None in the source manifest; this does not confer biological qualification.
    """

    units: AtomisticUnitSystem
    unary_energies: tuple[Array, ...]
    pair_amplitudes: tuple[Array, ...]
    pair_widths: tuple[Array, ...]
    source: ReferenceArtifactManifest
    pair_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    cardinalities: tuple[int, ...] = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    thermal_energy: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        units: AtomisticUnitSystem,
        temperature: float,
        unary_energies: tuple[ArrayLike, ...],
        pair_indices: tuple[tuple[int, int], ...],
        pair_amplitudes: tuple[ArrayLike, ...],
        pair_widths: tuple[ArrayLike, ...],
        source: ReferenceArtifactManifest,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        temperature_ = float(temperature)
        if not isfinite(temperature_) or temperature_ <= 0:
            raise ValueError("Effective temperature must be finite and positive.")
        unary = tuple(_finite_array(value, "unary_energies") for value in unary_energies)
        if not unary or any(value.ndim != 1 or value.size < 1 for value in unary):
            raise ValueError(
                "Unary energies must be nonempty heterogeneous state vectors."
            )
        pairs = tuple(tuple(pair) for pair in pair_indices)
        if any(
            len(pair) != 2
            or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in pair
            )
            for pair in pairs
        ):
            raise ValueError("pair_indices must contain integer residue pairs.")
        pairs = tuple((int(i), int(j)) for i, j in pairs)
        if any(not 0 <= i < j < len(unary) for i, j in pairs) or len(set(pairs)) != len(
            pairs
        ):
            raise ValueError("Pair support must be unique ordered residue pairs i < j.")
        amplitudes = tuple(
            _finite_array(value, "pair_amplitudes") for value in pair_amplitudes
        )
        widths = tuple(_finite_array(value, "pair_widths") for value in pair_widths)
        if len(amplitudes) != len(pairs) or len(widths) != len(pairs):
            raise ValueError("Every fixed pair requires amplitude and width tables.")
        cards = tuple(int(value.size) for value in unary)
        for (i, j), amplitude, width in zip(pairs, amplitudes, widths):
            if (
                amplitude.shape != (cards[i], cards[j])
                or width.shape != amplitude.shape
                or np.any(np.asarray(width) <= 0)
            ):
                raise ValueError(
                    "Pair tables must match valid cardinalities; widths must be positive."
                )
        source_id = _admit(source, commercial_use, redistribution, training_use, export)
        self.units = units
        self.temperature = temperature_
        self.thermal_energy = units.boltzmann_constant * temperature_
        self.unary_energies = unary
        self.pair_indices = pairs
        self.pair_amplitudes = amplitudes
        self.pair_widths = widths
        self.cardinalities = cards
        self.source = source
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "single-site-rotamer-gaussian-parameters",
                "units": units.unit_system_id,
                "temperature": temperature_,
                "pairs": pairs,
                "source": source_id,
                "parameters": array_tree_fingerprint((unary, amplitudes, widths)),
            }
        )


class RotamerFreeEnergyStatus(IntEnum):
    SUCCESS = 0
    INVALID_GEOMETRY = 1
    INFERENCE_FAILED = 2
    UNQUALIFIED_BRANCH = 3
    NONFINITE_ENERGY = 4


class RotamerFreeEnergyEvaluation(StrictModule):
    energy: Array
    atom_energy: Array
    variable_probabilities: Array
    status: Array
    successful: Array
    geometry_valid: Array
    derivative_qualified: Array
    contraction_bound: Array
    inference: ExactFactorGraphResult | AdvancedBeliefPropagationResult


class RotamerFreeEnergyTerm(AbstractAtomisticEnergyTerm):
    """Conservative G = -kBT log Z with fixed, deterministic branch policy.

    Exact enumeration is capped at preparation. Bethe execution uses the native
    implicit root, always initialized at zero messages. On loops a sufficient
    global contraction bound certifies uniqueness and bounds inverse sensitivity;
    candidates outside this conservative certificate are refused, even if one
    numerical root converges. There are no warm starts or graph-switch cutoffs.
    Attribution is explicitly E * weights, not a unique physical atom observable.
    """

    geometry: RotamerGeometryPlan
    parameters: RotamerParameterPlan
    attribution_atom_ids: Array
    attribution_weights: Array
    method: SumProductBeliefPropagation
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    inference_method: str = eqx.field(static=True)
    maximum_configurations: int = eqx.field(static=True)
    maximum_contraction: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        geometry: RotamerGeometryPlan,
        parameters: RotamerParameterPlan,
        attribution_atom_ids: ArrayLike,
        attribution_weights: ArrayLike,
        /,
        *,
        sampling_temperature: float,
        inference_method: str = "exact",
        maximum_configurations: int = 65_536,
        maximum_steps: int = 100,
        absolute_tolerance: float = 1e-10,
        relative_tolerance: float = 1e-10,
        maximum_contraction: float = 0.95,
        name: str = "rotamer-free-energy",
        force_group: int = 0,
    ):
        if not isinstance(geometry, RotamerGeometryPlan) or not isinstance(
            parameters, RotamerParameterPlan
        ):
            raise TypeError("Require RotamerGeometryPlan and RotamerParameterPlan.")
        if geometry.cardinalities != parameters.cardinalities:
            raise ValueError(
                "Geometry and energy tables disagree on valid cardinalities."
            )
        if float(sampling_temperature) != parameters.temperature:
            raise ValueError(
                "Sampling temperature differs from the fixed effective model; revalidate it."
            )
        ids = np.asarray(attribution_atom_ids)
        weights = np.asarray(attribution_weights, dtype=float)
        if (
            ids.ndim != 1
            or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != ids.size
        ):
            raise ValueError("Attribution requires unique stable atom IDs.")
        if (
            weights.shape != ids.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
        ):
            raise ValueError(
                "Declared attribution weights must be nonnegative and sum to one."
            )
        if inference_method not in ("exact", "bethe"):
            raise ValueError("inference_method must be exact or bethe.")
        if not 0 < maximum_contraction < 1 or maximum_configurations < 1:
            raise ValueError(
                "Require contraction in (0, 1) and positive enumeration cap."
            )
        if not name or force_group < 0:
            raise ValueError("Require nonempty name and nonnegative force_group.")
        self.geometry = geometry
        self.parameters = parameters
        if (
            geometry.units.scale.length_unit.unit_id
            != parameters.units.scale.length_unit.unit_id
        ):
            raise ValueError("Rotamer geometry and parameter length units differ.")
        self.attribution_atom_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.attribution_weights = jnp.asarray(weights)
        self.method = SumProductBeliefPropagation(
            maximum_steps=maximum_steps,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        self.name = name
        self.force_group = int(force_group)
        self.inference_method = inference_method
        self.maximum_configurations = int(maximum_configurations)
        self.maximum_contraction = float(maximum_contraction)
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True, finite_geometry=True, local_energy=True
        )
        self.requirements = AtomisticPotentialRequirements()
        self.term_id = canonical_fingerprint(
            {
                "kind": "rotamer-free-energy-term",
                "geometry": geometry.geometry_id,
                "parameters": parameters.parameter_id,
                "inference": inference_method,
                "attribution_ids": ids.tolist(),
                "weights": weights.tolist(),
                "maximum_configurations": self.maximum_configurations,
                "maximum_steps": maximum_steps,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "maximum_contraction": maximum_contraction,
                "initialization": "zero-messages-global-contraction",
                "name": name,
                "force_group": self.force_group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedRotamerFreeEnergyTerm":
        return PreparedRotamerFreeEnergyTerm(self, system)


class PreparedRotamerFreeEnergyTerm(AbstractPreparedAtomisticEnergyTerm):
    plan: RotamerFreeEnergyTerm
    frame_rows: Array
    attribution: Array
    bp: PreparedBeliefPropagation
    initial_state: BeliefPropagationState
    exact: PreparedExactFactorGraph | None
    pair_cavity_degrees: tuple[int, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: RotamerFreeEnergyTerm, system: PreparedAtomisticSystem, /):
        if system.plan.units.unit_system_id != plan.parameters.units.unit_system_id:
            raise ValueError("Rotamer model and atomistic unit systems differ.")
        if system.cell is not None:
            raise ValueError("Rotamer frames currently require nonperiodic coordinates.")
        ids = np.asarray(system.plan.particle_ids)
        active = np.asarray(system.active_mask)
        rows = {int(key): index for index, key in enumerate(ids) if active[index]}
        frame_ids = np.asarray(plan.geometry.frame_atom_ids)
        attribution_ids = np.asarray(plan.attribution_atom_ids)
        if any(int(key) not in rows for key in frame_ids.flat) or any(
            int(key) not in rows for key in attribution_ids
        ):
            raise ValueError(
                "Every frame/attribution ID must bind to an active atom; missing atoms are not padding."
            )
        frame_rows = np.asarray(
            [[rows[int(key)] for key in row] for row in frame_ids], dtype=np.int32
        )
        attribution = np.zeros((system.capacity,), dtype=float)
        attribution[[rows[int(key)] for key in attribution_ids]] = np.asarray(
            plan.attribution_weights
        )
        variables = DiscreteVariableGroup(
            "rotamers", num_states=np.asarray(plan.geometry.cardinalities)
        )
        factors = [
            DenseTableFactorGroup(
                (VariableSelection(variables, [i]),), np.zeros((1, card))
            )
            for i, card in enumerate(plan.geometry.cardinalities)
        ]
        degrees = [0] * len(plan.geometry.cardinalities)
        for i, j in plan.parameters.pair_indices:
            factors.append(
                DenseTableFactorGroup(
                    (
                        VariableSelection(variables, [i]),
                        VariableSelection(variables, [j]),
                    ),
                    np.zeros(
                        (
                            1,
                            plan.geometry.cardinalities[i],
                            plan.geometry.cardinalities[j],
                        )
                    ),
                )
            )
            degrees[i] += 1
            degrees[j] += 1
        graph = DiscreteFactorGraph((variables,), tuple(factors))
        self.plan = plan
        self.frame_rows = jnp.asarray(frame_rows)
        self.attribution = jnp.asarray(attribution)
        self.bp = prepare_belief_propagation(graph, plan.method)
        self.initial_state = initialize_belief_propagation(self.bp)
        self.exact = (
            prepare_exact_factor_graph(
                graph, max_configurations=plan.maximum_configurations
            )
            if plan.inference_method == "exact"
            else None
        )
        self.pair_cavity_degrees = tuple(
            max(degrees[i] - 1, degrees[j] - 1) for i, j in plan.parameters.pair_indices
        )
        self.capacity = system.capacity
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rotamer-free-energy",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def log_factors(self, positions: ArrayLike, /) -> tuple[tuple[Array, ...], Array]:
        """Compile differentiable numeric tables without changing permitted support."""
        positions = jnp.asarray(positions)
        if positions.shape != (self.capacity, 3):
            raise ValueError("positions must match prepared atom capacity.")
        xyz = positions[self.frame_rows]
        x = xyz[:, 1] - xyz[:, 0]
        xnorm = jnp.sqrt(jnp.sum(x * x, axis=-1))
        minimum = self.plan.geometry.minimum_frame_length
        ex = x / jnp.where(xnorm > minimum, xnorm, 1.0)[:, None]
        y = xyz[:, 2] - xyz[:, 0]
        y = y - jnp.sum(y * ex, axis=-1)[:, None] * ex
        ynorm = jnp.sqrt(jnp.sum(y * y, axis=-1))
        ey = y / jnp.where(ynorm > minimum, ynorm, 1.0)[:, None]
        ez = jnp.cross(ex, ey)
        frames = jnp.stack((ex, ey, ez), axis=1)
        valid = (
            jnp.all(jnp.isfinite(xyz))
            & jnp.all(xnorm > minimum)
            & jnp.all(ynorm > minimum)
        )
        sites = tuple(
            xyz[i, 0] + contract("sa,ab->sb", local, frames[i])
            for i, local in enumerate(self.plan.geometry.local_sites)
        )
        thermal = self.plan.parameters.thermal_energy
        tables = [
            -value[None, :] / thermal for value in self.plan.parameters.unary_energies
        ]
        for (i, j), amplitude, width in zip(
            self.plan.parameters.pair_indices,
            self.plan.parameters.pair_amplitudes,
            self.plan.parameters.pair_widths,
        ):
            delta = sites[i][:, None, :] - sites[j][None, :, :]
            squared_distance = jnp.sum(delta * delta, axis=-1)
            energy = amplitude * jnp.exp(-squared_distance / (2.0 * width * width))
            tables.append(-energy[None, :, :] / thermal)
        return tuple(tables), valid

    def evaluate(self, positions: ArrayLike, /) -> RotamerFreeEnergyEvaluation:
        """Evaluate the scalar and retained inference/branch failure evidence."""
        tables, geometry_valid = self.log_factors(positions)
        bound = jnp.asarray(0.0)
        if self.exact is not None:
            inference = run_exact_factor_graph(self.exact, tables)
            log_normalizer = inference.log_normalizer
            marginals = inference.variable_probabilities.values
            inference_valid = inference.successful
            qualified = jnp.asarray(True)
        else:
            if not self.bp.forest:
                for table, degree in zip(
                    tables[len(self.plan.geometry.cardinalities) :],
                    self.pair_cavity_degrees,
                ):
                    # Birkhoff projective contraction <= tanh(range(log psi)/2).
                    # Multiplication by cavity degree bounds each directed row sum.
                    bound = jnp.maximum(
                        bound, degree * jnp.tanh((jnp.max(table) - jnp.min(table)) / 2.0)
                    )
            qualified = jnp.isfinite(bound) & (bound <= self.plan.maximum_contraction)
            prepared = replace_belief_propagation_tables(self.bp, tables)
            inference = run_implicit_belief_propagation(prepared, self.initial_state)
            log_normalizer = inference.inference.log_normalizer
            marginals = jnp.exp(inference.inference.variable_log_probabilities.values)
            inference_valid = inference.inference.successful
        energy = -self.plan.parameters.thermal_energy * log_normalizer
        finite = jnp.isfinite(energy)
        accepted = geometry_valid & qualified & inference_valid & finite
        status = jnp.where(
            ~geometry_valid,
            int(RotamerFreeEnergyStatus.INVALID_GEOMETRY),
            jnp.where(
                ~qualified,
                int(RotamerFreeEnergyStatus.UNQUALIFIED_BRANCH),
                jnp.where(
                    ~inference_valid,
                    int(RotamerFreeEnergyStatus.INFERENCE_FAILED),
                    jnp.where(
                        ~finite,
                        int(RotamerFreeEnergyStatus.NONFINITE_ENERGY),
                        int(RotamerFreeEnergyStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        energy = jnp.where(accepted, energy, jnp.nan)
        return RotamerFreeEnergyEvaluation(
            energy,
            energy * self.attribution,
            marginals,
            status,
            accepted,
            geometry_valid,
            qualified & inference_valid & geometry_valid & finite,
            bound,
            inference,
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        result = self.evaluate(context.positions)
        return AtomisticTermEvaluation(
            result.energy, result.atom_energy, result.successful
        )


__all__ = [
    "RotamerGeometryPlan",
    "RotamerParameterPlan",
    "RotamerFreeEnergyTerm",
    "PreparedRotamerFreeEnergyTerm",
    "RotamerFreeEnergyEvaluation",
    "RotamerFreeEnergyStatus",
]
