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
from jax import Array
from jax.typing import ArrayLike

from phydrax._fingerprint import array_tree_fingerprint
from phydrax._identity import IdentityWords
from phydrax._strict import StrictModule
from phydrax._trainable import fixed_field
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

from ....typing import checked
from .._construct import ProteinConstruct


def _finite_array(value: ArrayLike, name: str) -> Array:
    host = np.asarray(value)
    if (
        not np.issubdtype(host.dtype, np.number)
        or np.iscomplexobj(host)
        or not np.all(np.isfinite(host))
    ):
        raise ValueError(f"{name} must contain finite real values.")
    return jnp.asarray(host, dtype=jnp.float64)


def _admit(
    source: object,
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
) -> str:
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

    PyTree layout: the per-residue rotamer ``cardinalities`` (site-table shapes)
    and the declared unit system are static. Frame IDs, local sites and
    ``minimum_frame_length`` are dynamic leaves. The construct, the admitted
    source manifest and the complete geometry are exact canonical digests as
    dynamic words (``construct_identity``, ``source_identity``,
    ``geometry_identity``; ``construct_id``/``source_id``/``geometry_id`` decode
    them on the host). Rights are admitted against the source manifest at
    construction.
    """

    frame_atom_ids: Array
    local_sites: tuple[Array, ...]
    units: AtomisticUnitSystem
    minimum_frame_length: Array
    construct_identity: IdentityWords
    source_identity: IdentityWords
    geometry_identity: IdentityWords
    cardinalities: tuple[int, ...] = eqx.field(static=True)

    @checked
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
    ) -> None:
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
        construct_id = construct.fingerprint()
        geometry = IdentityWords.of(
            {
                "kind": "protein-single-site-rotamer-geometry",
                "construct": construct_id,
                "frame_ids": ids.tolist(),
                "sites": array_tree_fingerprint(sites),
                "minimum_frame_length": threshold,
                "source": source_id,
                "length_unit": units.scale.length_unit.unit_id,
            }
        )
        self.frame_atom_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.local_sites = sites
        self.units = units
        self.minimum_frame_length = jnp.asarray(threshold, dtype=jnp.float64)
        self.construct_identity = IdentityWords.from_fingerprint(construct_id)
        self.source_identity = IdentityWords.from_fingerprint(source_id)
        self.geometry_identity = geometry
        self.cardinalities = tuple(value.shape[0] for value in sites)

    @property
    def construct_id(self) -> str:
        """Host decoding of the bound protein construct fingerprint."""
        return self.construct_identity.fingerprint()

    @property
    def source_id(self) -> str:
        """Host decoding of the admitted source manifest digest."""
        return self.source_identity.fingerprint()

    @property
    def geometry_id(self) -> str:
        """Host decoding of the canonical rotamer geometry digest."""
        return self.geometry_identity.fingerprint()


class RotamerParameterPlan(StrictModule):
    """Source-pinned caller parameters for the explicitly named Gaussian profile.

    Energies are single-system energies in units.scale.energy_unit, not molar
    energies; widths and local geometry use units.scale.length_unit. Temperature
    is absolute, in units.temperature_unit. Unknown model uncertainty may remain
    None in the source manifest; this does not confer biological qualification.

    PyTree layout: ``pair_indices`` (the declared fixed pair support, i.e. the
    factor-graph scopes) and ``cardinalities`` (table shapes) are static, with
    the declared unit system. Tables, ``temperature`` and ``thermal_energy``
    are dynamic leaves; the admitted source manifest and the complete
    parameter set are exact canonical digests as dynamic words
    (``source_identity``, ``parameter_identity``; ``source_id``/
    ``parameter_id`` decode them on the host).
    """

    units: AtomisticUnitSystem
    unary_energies: tuple[Array, ...]
    pair_amplitudes: tuple[Array, ...]
    pair_widths: tuple[Array, ...]
    temperature: Array
    thermal_energy: Array
    source_identity: IdentityWords
    parameter_identity: IdentityWords
    pair_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    cardinalities: tuple[int, ...] = eqx.field(static=True)

    @checked
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
    ) -> None:
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
        cards = tuple(value.size for value in unary)
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
        parameters = IdentityWords.of(
            {
                "kind": "single-site-rotamer-gaussian-parameters",
                "units": units.unit_system_id,
                "temperature": temperature_,
                "pairs": pairs,
                "source": source_id,
                "parameters": array_tree_fingerprint((unary, amplitudes, widths)),
            }
        )
        self.units = units
        self.temperature = jnp.asarray(temperature_, dtype=jnp.float64)
        self.thermal_energy = jnp.asarray(
            units.boltzmann_constant * temperature_, dtype=jnp.float64
        )
        self.unary_energies = unary
        self.pair_indices = pairs
        self.pair_amplitudes = amplitudes
        self.pair_widths = widths
        self.cardinalities = cards
        self.source_identity = IdentityWords.from_fingerprint(source_id)
        self.parameter_identity = parameters

    @property
    def source_id(self) -> str:
        """Host decoding of the admitted source manifest digest."""
        return self.source_identity.fingerprint()

    @property
    def parameter_id(self) -> str:
        """Host decoding of the canonical rotamer parameter digest."""
        return self.parameter_identity.fingerprint()


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

    PyTree layout: ``inference_method``, ``maximum_configurations`` (exact
    enumeration capacity) and ``maximum_contraction`` (the declared loop
    certification policy) are static method selectors. The exact canonical term
    digest is the dynamic ``term_identity`` (``term_id`` decodes it on the host).
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
    term_identity: IdentityWords
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
    ) -> None:
        if not isinstance(geometry, RotamerGeometryPlan) or not isinstance(
            parameters, RotamerParameterPlan
        ):
            raise TypeError("Require RotamerGeometryPlan and RotamerParameterPlan.")
        if geometry.cardinalities != parameters.cardinalities:
            raise ValueError(
                "Geometry and energy tables disagree on valid cardinalities."
            )
        if float(sampling_temperature) != float(np.asarray(parameters.temperature)):
            raise ValueError(
                "Sampling temperature differs from the fixed effective model; revalidate it."
            )
        ids = np.asarray(attribution_atom_ids)
        weights = np.asarray(attribution_weights, dtype=np.float64)
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
        self.term_identity = IdentityWords.of(
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
    """Prepared rotamer free energy bound to one atomistic system.

    PyTree layout: the native belief-propagation and exact factor-graph
    preparations carry the static host topology built from the declared
    ``pair_indices`` and ``cardinalities``; those factor scopes define message
    routing, so they are declared layout. ``capacity`` is static. The loop
    certificate's per-pair cavity degrees are a dynamic int32 array, and
    ``term_identity``/``prepared_identity`` are exact canonical digests as
    dynamic words (``term_id``/``prepared_id`` decode them on the host).
    """

    plan: RotamerFreeEnergyTerm = fixed_field()
    frame_rows: Array = fixed_field()
    attribution: Array = fixed_field()
    bp: PreparedBeliefPropagation
    initial_state: BeliefPropagationState = fixed_field()
    exact: PreparedExactFactorGraph | None = fixed_field()
    pair_cavity_degrees: Array = fixed_field()
    capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: RotamerFreeEnergyTerm, system: PreparedAtomisticSystem, /
    ) -> None:
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
        attribution = np.zeros((system.capacity,), dtype=np.float64)
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
        self.pair_cavity_degrees = jnp.asarray(
            [
                max(degrees[i] - 1, degrees[j] - 1)
                for i, j in plan.parameters.pair_indices
            ],
            dtype=jnp.int32,
        )
        self.capacity = system.capacity
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
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
                pair_tables = tables[len(self.plan.geometry.cardinalities) :]
                for index, table in enumerate(pair_tables):
                    # Birkhoff projective contraction <= tanh(range(log psi)/2).
                    # Multiplication by cavity degree bounds each directed row sum.
                    degree = self.pair_cavity_degrees[index].astype(table.dtype)
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
