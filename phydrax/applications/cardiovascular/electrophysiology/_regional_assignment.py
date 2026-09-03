#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology regional electrophysiology assignments.

Preparation resolves anatomical labels and stable node IDs on the host.  Runtime
arrays contain only the resulting fixed worksets and dimensionless physical
multipliers; tensor discretization remains owned by the generic variational
operator layer.
"""

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _nonnegative_scale(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    scale = float(value)
    if not isfinite(scale) or scale < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scale


def _positive_scale(value: float, name: str, /) -> float:
    scale = _nonnegative_scale(value, name)
    if scale == 0.0:
        raise ValueError(f"{name} must be positive.")
    return scale


def _phenotype_index(value: int, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("phenotype_index must be an integer.")
    if value < 0:
        raise ValueError("phenotype_index must be non-negative.")
    return value


class RegionalPhenotype(StrictModule, NonTrainableState):
    """Stable regional label selecting one prepared cellular reaction model."""

    phenotype_id: str = eqx.field(static=True)
    reaction_index: int = eqx.field(static=True)

    def __init__(self, phenotype_id: str, reaction_index: int, /):
        self.phenotype_id = _identifier(phenotype_id, "phenotype_id")
        if not isinstance(reaction_index, int) or isinstance(reaction_index, bool):
            raise TypeError("reaction_index must be an integer.")
        if reaction_index < 0:
            raise ValueError("reaction_index must be non-negative.")
        self.reaction_index = reaction_index


class AnatomicalRegionSelector(StrictModule, NonTrainableState):
    """Host-side selector over region codes and optional stable node IDs."""

    region_codes: tuple[int, ...] = eqx.field(static=True)
    stable_node_ids: tuple[int, ...] = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_codes: tuple[int, ...],
        /,
        *,
        stable_node_ids: tuple[int, ...] = (),
    ):
        codes = tuple(int(code) for code in region_codes)
        node_ids = tuple(int(node_id) for node_id in stable_node_ids)
        if not codes:
            raise ValueError("region_codes must contain at least one code.")
        if len(set(codes)) != len(codes):
            raise ValueError("region_codes must be unique.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("stable_node_ids must be unique.")
        self.region_codes = codes
        self.stable_node_ids = node_ids
        self.selector_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-anatomical-region-selector-v1",
                "region_codes": codes,
                "stable_node_ids": node_ids,
            }
        )


class RegionalHeterogeneity(StrictModule, NonTrainableState):
    """Ordinary regional phenotype and physical-coefficient adjustment."""

    phenotype_index: int = eqx.field(static=True)
    conductivity_scale: float = eqx.field(static=True)
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    effect_id: str = eqx.field(static=True)

    def __init__(
        self,
        phenotype_index: int,
        /,
        *,
        conductivity_scale: float = 1.0,
        capacitance_scale: float = 1.0,
        ionic_current_scale: float = 1.0,
        state_update_scale: float = 1.0,
    ):
        self.phenotype_index = _phenotype_index(phenotype_index)
        self.conductivity_scale = _nonnegative_scale(
            conductivity_scale, "conductivity_scale"
        )
        self.capacitance_scale = _positive_scale(capacitance_scale, "capacitance_scale")
        self.ionic_current_scale = _nonnegative_scale(
            ionic_current_scale, "ionic_current_scale"
        )
        self.state_update_scale = _nonnegative_scale(
            state_update_scale, "state_update_scale"
        )
        self.effect_id = _effect_fingerprint("regional-heterogeneity", self)


class ScarCore(StrictModule, NonTrainableState):
    """Electrically isolated, non-excitable dense scar core."""

    phenotype_index: int = eqx.field(static=True)
    conductivity_scale: float = eqx.field(static=True)
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    effect_id: str = eqx.field(static=True)

    def __init__(
        self,
        phenotype_index: int,
        /,
        *,
        conductivity_scale: float = 0.0,
        capacitance_scale: float = 1.0,
        ionic_current_scale: float = 0.0,
        state_update_scale: float = 0.0,
    ):
        self.phenotype_index = _phenotype_index(phenotype_index)
        self.conductivity_scale = _nonnegative_scale(
            conductivity_scale, "conductivity_scale"
        )
        self.capacitance_scale = _positive_scale(capacitance_scale, "capacitance_scale")
        self.ionic_current_scale = _nonnegative_scale(
            ionic_current_scale, "ionic_current_scale"
        )
        self.state_update_scale = _nonnegative_scale(
            state_update_scale, "state_update_scale"
        )
        self.effect_id = _effect_fingerprint("scar-core", self)


class ScarBorderZone(StrictModule, NonTrainableState):
    """Partially conducting, remodelled excitable scar border zone."""

    phenotype_index: int = eqx.field(static=True)
    conductivity_scale: float = eqx.field(static=True)
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    effect_id: str = eqx.field(static=True)

    def __init__(
        self,
        phenotype_index: int,
        /,
        *,
        conductivity_scale: float = 0.25,
        capacitance_scale: float = 1.0,
        ionic_current_scale: float = 0.5,
        state_update_scale: float = 1.0,
    ):
        self.phenotype_index = _phenotype_index(phenotype_index)
        self.conductivity_scale = _nonnegative_scale(
            conductivity_scale, "conductivity_scale"
        )
        self.capacitance_scale = _positive_scale(capacitance_scale, "capacitance_scale")
        self.ionic_current_scale = _nonnegative_scale(
            ionic_current_scale, "ionic_current_scale"
        )
        self.state_update_scale = _nonnegative_scale(
            state_update_scale, "state_update_scale"
        )
        self.effect_id = _effect_fingerprint("scar-border-zone", self)


class DiffuseFibrosis(StrictModule, NonTrainableState):
    """Explicit stable-ID realization of diffuse fibrotic remodelling."""

    phenotype_index: int = eqx.field(static=True)
    conductivity_scale: float = eqx.field(static=True)
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    effect_id: str = eqx.field(static=True)

    def __init__(
        self,
        phenotype_index: int,
        /,
        *,
        conductivity_scale: float = 0.6,
        capacitance_scale: float = 1.0,
        ionic_current_scale: float = 0.8,
        state_update_scale: float = 1.0,
    ):
        self.phenotype_index = _phenotype_index(phenotype_index)
        self.conductivity_scale = _nonnegative_scale(
            conductivity_scale, "conductivity_scale"
        )
        self.capacitance_scale = _positive_scale(capacitance_scale, "capacitance_scale")
        self.ionic_current_scale = _nonnegative_scale(
            ionic_current_scale, "ionic_current_scale"
        )
        self.state_update_scale = _nonnegative_scale(
            state_update_scale, "state_update_scale"
        )
        self.effect_id = _effect_fingerprint("diffuse-fibrosis", self)


class AblationLesion(StrictModule, NonTrainableState):
    """Fixed ablation lesion with no conduction or cellular reaction."""

    phenotype_index: int = eqx.field(static=True)
    conductivity_scale: float = eqx.field(static=True)
    capacitance_scale: float = eqx.field(static=True)
    ionic_current_scale: float = eqx.field(static=True)
    state_update_scale: float = eqx.field(static=True)
    effect_id: str = eqx.field(static=True)

    def __init__(
        self,
        phenotype_index: int,
        /,
        *,
        conductivity_scale: float = 0.0,
        capacitance_scale: float = 1.0,
        ionic_current_scale: float = 0.0,
        state_update_scale: float = 0.0,
    ):
        self.phenotype_index = _phenotype_index(phenotype_index)
        self.conductivity_scale = _nonnegative_scale(
            conductivity_scale, "conductivity_scale"
        )
        self.capacitance_scale = _positive_scale(capacitance_scale, "capacitance_scale")
        self.ionic_current_scale = _nonnegative_scale(
            ionic_current_scale, "ionic_current_scale"
        )
        self.state_update_scale = _nonnegative_scale(
            state_update_scale, "state_update_scale"
        )
        self.effect_id = _effect_fingerprint("ablation-lesion", self)


RegionalEffect: TypeAlias = (
    RegionalHeterogeneity | ScarCore | ScarBorderZone | DiffuseFibrosis | AblationLesion
)


def _effect_fingerprint(kind: str, effect: RegionalEffect, /) -> str:
    return canonical_fingerprint(
        {
            "kind": f"cardiovascular-{kind}-v1",
            "phenotype_index": effect.phenotype_index,
            "conductivity_scale": effect.conductivity_scale,
            "capacitance_scale": effect.capacitance_scale,
            "ionic_current_scale": effect.ionic_current_scale,
            "state_update_scale": effect.state_update_scale,
        }
    )


def _effect_tissue_code(effect: RegionalEffect, /) -> int:
    if isinstance(effect, ScarCore):
        return 1
    if isinstance(effect, ScarBorderZone):
        return 2
    if isinstance(effect, DiffuseFibrosis):
        return 3
    if isinstance(effect, AblationLesion):
        return 4
    return 0


class RegionalAssignmentRule(StrictModule, NonTrainableState):
    """One disjoint anatomical selection and its electrophysiology effect."""

    selector: AnatomicalRegionSelector
    effect: RegionalEffect
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        selector: AnatomicalRegionSelector,
        effect: RegionalEffect,
        /,
        *,
        rule_id: str | None = None,
    ):
        if not isinstance(selector, AnatomicalRegionSelector):
            raise TypeError("selector must be an AnatomicalRegionSelector.")
        if not isinstance(
            effect,
            (
                RegionalHeterogeneity,
                ScarCore,
                ScarBorderZone,
                DiffuseFibrosis,
                AblationLesion,
            ),
        ):
            raise TypeError("effect must be a supported regional effect.")
        self.selector = selector
        self.effect = effect
        self.rule_id = (
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-regional-ep-rule-v1",
                    "selector": selector.selector_id,
                    "effect": effect.effect_id,
                }
            )
            if rule_id is None
            else _identifier(rule_id, "rule_id")
        )


class RegionalAssignmentEvidence(StrictModule):
    """Complete host-preparation accounting for one fixed node topology."""

    node_count: Array
    workset_count: Array
    rule_node_counts: Array
    tissue_node_counts: Array
    overlap_count: Array
    unassigned_count: Array
    stable_ids_unique: Array
    complete: Array


class PreparedRegionalAssignment(StrictModule, NonTrainableState):
    """Immutable nodal coefficients and homogeneous reaction routing."""

    plan: RegionalElectrophysiologyPlan
    stable_node_ids: tuple[int, ...] = eqx.field(static=True)
    anatomical_region_codes: Array
    phenotype_indices: Array
    tissue_codes: Array
    conductivity_scale: Array
    capacitance_scale: Array
    ionic_current_scale: Array
    state_update_scale: Array
    workset_indices: tuple[Array, ...]
    workset_ids: tuple[str, ...] = eqx.field(static=True)
    workset_phenotype_indices: tuple[int, ...] = eqx.field(static=True)
    workset_reaction_indices: tuple[int, ...] = eqx.field(static=True)
    workset_capacitance_scales: tuple[float, ...] = eqx.field(static=True)
    workset_ionic_current_scales: tuple[float, ...] = eqx.field(static=True)
    workset_state_update_scales: tuple[float, ...] = eqx.field(static=True)
    evidence: RegionalAssignmentEvidence
    runtime_id: str = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return self.plan.node_count

    def scale_nodal_diffusivity(self, diffusivity: ArrayLike, /) -> Array:
        """Apply prepared dimensionless conductivity scaling without assembly."""
        value = jnp.asarray(diffusivity)
        if value.ndim == 0:
            return self.conductivity_scale * value
        if value.shape[0] != self.node_count:
            raise ValueError(
                "Nodal diffusivity must be scalar or have node_count as its first axis."
            )
        shape = (self.node_count,) + (1,) * (value.ndim - 1)
        return value * self.conductivity_scale.reshape(shape)


class RegionalElectrophysiologyPlan(StrictModule, NonTrainableState):
    """Host plan for disjoint, deterministic regional assignments."""

    node_count: int = eqx.field(static=True)
    phenotypes: tuple[RegionalPhenotype, ...]
    default_phenotype_index: int = eqx.field(static=True)
    rules: tuple[RegionalAssignmentRule, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_count: int,
        phenotypes: tuple[RegionalPhenotype, ...],
        /,
        *,
        default_phenotype_index: int = 0,
        rules: tuple[RegionalAssignmentRule, ...] = (),
    ):
        if not isinstance(node_count, int) or isinstance(node_count, bool):
            raise TypeError("node_count must be an integer.")
        if node_count <= 0:
            raise ValueError("node_count must be positive.")
        phenotypes_ = tuple(phenotypes)
        rules_ = tuple(rules)
        if not phenotypes_ or not all(
            isinstance(phenotype, RegionalPhenotype) for phenotype in phenotypes_
        ):
            raise TypeError("phenotypes must contain RegionalPhenotype values.")
        if len({phenotype.phenotype_id for phenotype in phenotypes_}) != len(phenotypes_):
            raise ValueError("phenotype_id values must be unique.")
        default = _phenotype_index(default_phenotype_index)
        if default >= len(phenotypes_):
            raise ValueError("default_phenotype_index is out of range.")
        if not all(isinstance(rule, RegionalAssignmentRule) for rule in rules_):
            raise TypeError("rules must contain RegionalAssignmentRule values.")
        if len({rule.rule_id for rule in rules_}) != len(rules_):
            raise ValueError("rule_id values must be unique.")
        if any(rule.effect.phenotype_index >= len(phenotypes_) for rule in rules_):
            raise ValueError("A regional rule phenotype_index is out of range.")
        self.node_count = node_count
        self.phenotypes = phenotypes_
        self.default_phenotype_index = default
        self.rules = rules_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-regional-ep-plan-v1",
                "node_count": node_count,
                "phenotypes": tuple(
                    (phenotype.phenotype_id, phenotype.reaction_index)
                    for phenotype in phenotypes_
                ),
                "default_phenotype_index": default,
                "rules": tuple(rule.rule_id for rule in rules_),
            }
        )

    def prepare(
        self,
        stable_node_ids: ArrayLike,
        anatomical_region_codes: ArrayLike,
        /,
    ) -> PreparedRegionalAssignment:
        return prepare_regional_assignment(self, stable_node_ids, anatomical_region_codes)


def _integer_vector(value: ArrayLike, size: int, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}.")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must have integer dtype.")
    return array.astype(np.int64, copy=False)


def prepare_regional_assignment(
    plan: RegionalElectrophysiologyPlan,
    stable_node_ids: ArrayLike,
    anatomical_region_codes: ArrayLike,
    /,
) -> PreparedRegionalAssignment:
    """Resolve every node exactly once and retain complete assignment evidence."""
    if not isinstance(plan, RegionalElectrophysiologyPlan):
        raise TypeError("plan must be a RegionalElectrophysiologyPlan.")
    node_ids = _integer_vector(stable_node_ids, plan.node_count, "stable_node_ids")
    region_codes = _integer_vector(
        anatomical_region_codes, plan.node_count, "anatomical_region_codes"
    )
    stable_unique = len(np.unique(node_ids)) == plan.node_count
    if not stable_unique:
        raise ValueError("stable_node_ids must be globally unique.")

    selected_masks: list[np.ndarray] = []
    occupancy = np.zeros((plan.node_count,), dtype=np.int32)
    for rule in plan.rules:
        mask = np.isin(region_codes, np.asarray(rule.selector.region_codes))
        if rule.selector.stable_node_ids:
            requested = np.asarray(rule.selector.stable_node_ids, dtype=np.int64)
            missing = np.setdiff1d(requested, node_ids)
            if missing.size:
                raise ValueError(
                    f"Rule {rule.rule_id!r} references unknown stable node IDs."
                )
            requested_mask = np.isin(node_ids, requested)
            if np.count_nonzero(mask & requested_mask) != requested.size:
                raise ValueError(
                    f"Rule {rule.rule_id!r} stable node IDs must belong to its "
                    "declared anatomical regions."
                )
            mask &= requested_mask
        if not np.any(mask):
            raise ValueError(f"Rule {rule.rule_id!r} selects no nodes.")
        selected_masks.append(mask)
        occupancy += mask.astype(np.int32)
    overlap_count = int(np.count_nonzero(occupancy > 1))
    if overlap_count:
        raise ValueError("Regional assignment rules must select disjoint nodes.")

    default_mask = occupancy == 0
    phenotype_indices = np.full(
        (plan.node_count,), plan.default_phenotype_index, dtype=np.int32
    )
    tissue_codes = np.zeros((plan.node_count,), dtype=np.int32)
    conductivity = np.ones((plan.node_count,), dtype=np.float64)
    capacitance = np.ones((plan.node_count,), dtype=np.float64)
    ionic_current = np.ones((plan.node_count,), dtype=np.float64)
    state_update = np.ones((plan.node_count,), dtype=np.float64)

    workset_indices: list[Array] = []
    workset_ids: list[str] = []
    workset_phenotypes: list[int] = []
    workset_reactions: list[int] = []
    workset_capacitance: list[float] = []
    workset_ionic: list[float] = []
    workset_state_update: list[float] = []
    rule_counts = [int(np.count_nonzero(default_mask))]

    if np.any(default_mask):
        default_phenotype = plan.phenotypes[plan.default_phenotype_index]
        workset_indices.append(jnp.asarray(np.flatnonzero(default_mask), dtype=jnp.int32))
        workset_ids.append(
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-regional-default-workset-v1",
                    "plan": plan.plan_id,
                    "phenotype": default_phenotype.phenotype_id,
                }
            )
        )
        workset_phenotypes.append(plan.default_phenotype_index)
        workset_reactions.append(default_phenotype.reaction_index)
        workset_capacitance.append(1.0)
        workset_ionic.append(1.0)
        workset_state_update.append(1.0)

    for rule, mask in zip(plan.rules, selected_masks, strict=True):
        effect = rule.effect
        phenotype = plan.phenotypes[effect.phenotype_index]
        count = int(np.count_nonzero(mask))
        rule_counts.append(count)
        phenotype_indices[mask] = effect.phenotype_index
        tissue_codes[mask] = _effect_tissue_code(effect)
        conductivity[mask] = effect.conductivity_scale
        capacitance[mask] = effect.capacitance_scale
        ionic_current[mask] = effect.ionic_current_scale
        state_update[mask] = effect.state_update_scale
        workset_indices.append(jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32))
        workset_ids.append(
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-regional-workset-v1",
                    "plan": plan.plan_id,
                    "rule": rule.rule_id,
                    "phenotype": phenotype.phenotype_id,
                }
            )
        )
        workset_phenotypes.append(effect.phenotype_index)
        workset_reactions.append(phenotype.reaction_index)
        workset_capacitance.append(effect.capacitance_scale)
        workset_ionic.append(effect.ionic_current_scale)
        workset_state_update.append(effect.state_update_scale)

    assigned_count = sum(rule_counts)
    unassigned_count = plan.node_count - assigned_count
    tissue_counts = np.bincount(tissue_codes, minlength=5)
    evidence = RegionalAssignmentEvidence(
        jnp.asarray(plan.node_count, dtype=jnp.int32),
        jnp.asarray(len(workset_indices), dtype=jnp.int32),
        jnp.asarray(rule_counts, dtype=jnp.int32),
        jnp.asarray(tissue_counts, dtype=jnp.int32),
        jnp.asarray(overlap_count, dtype=jnp.int32),
        jnp.asarray(unassigned_count, dtype=jnp.int32),
        jnp.asarray(stable_unique),
        jnp.asarray(stable_unique and overlap_count == 0 and unassigned_count == 0),
    )
    runtime_id = canonical_fingerprint(
        {
            "kind": "prepared-cardiovascular-regional-ep-v1",
            "plan": plan.plan_id,
            "stable_node_ids": tuple(int(value) for value in node_ids),
            "anatomical_region_codes": tuple(int(value) for value in region_codes),
            "worksets": tuple(workset_ids),
        }
    )
    return PreparedRegionalAssignment(
        plan,
        tuple(int(value) for value in node_ids),
        jnp.asarray(region_codes, dtype=jnp.int32),
        jnp.asarray(phenotype_indices, dtype=jnp.int32),
        jnp.asarray(tissue_codes, dtype=jnp.int32),
        jnp.asarray(conductivity),
        jnp.asarray(capacitance),
        jnp.asarray(ionic_current),
        jnp.asarray(state_update),
        tuple(workset_indices),
        tuple(workset_ids),
        tuple(workset_phenotypes),
        tuple(workset_reactions),
        tuple(workset_capacitance),
        tuple(workset_ionic),
        tuple(workset_state_update),
        evidence,
        runtime_id,
    )


__all__ = [
    "AblationLesion",
    "AnatomicalRegionSelector",
    "DiffuseFibrosis",
    "PreparedRegionalAssignment",
    "RegionalAssignmentEvidence",
    "RegionalAssignmentRule",
    "RegionalEffect",
    "RegionalElectrophysiologyPlan",
    "RegionalHeterogeneity",
    "RegionalPhenotype",
    "ScarBorderZone",
    "ScarCore",
    "prepare_regional_assignment",
]
