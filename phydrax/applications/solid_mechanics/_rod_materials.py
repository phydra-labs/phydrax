#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


RodMaterialKind: TypeAlias = Literal["stretch_shear", "bend_twist"]
RodMaterialOwnerKind: TypeAlias = Literal["segment", "junction"]


def _identifier(value: str, owner: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{owner} must be nonempty.")
    return identifier


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_psd(name: str, matrices: np.ndarray, /) -> None:
    if matrices.shape[0] == 0:
        return
    tolerance = 500.0 * np.finfo(matrices.dtype).eps
    scale = max(1.0, float(np.max(np.abs(matrices))))
    if not np.allclose(
        matrices,
        np.swapaxes(matrices, -1, -2),
        rtol=tolerance,
        atol=tolerance * scale,
    ):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrices) < -tolerance * scale):
        raise ValueError(f"{name} must be positive semidefinite.")


class RodMaterialSite(StrictModule, NonTrainableState):
    """Stable ownership of one native rod constitutive integration site."""

    material_kind: RodMaterialKind = eqx.field(static=True)
    owner_kind: RodMaterialOwnerKind = eqx.field(static=True)
    owner_index: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_kind: RodMaterialKind,
        owner_kind: RodMaterialOwnerKind,
        owner_index: int,
        component_count: int,
        rod_id: str,
        /,
    ):
        if material_kind not in ("stretch_shear", "bend_twist"):
            raise ValueError("Unknown rod material kind.")
        expected_owner = "segment" if material_kind == "stretch_shear" else "junction"
        if owner_kind != expected_owner:
            raise ValueError(
                f"{material_kind} sites must be owned by a {expected_owner}."
            )
        index = int(owner_index)
        components = int(component_count)
        rod = _identifier(rod_id, "rod_id")
        if index < 0 or components < 1:
            raise ValueError(
                "Rod material site indices and component counts are invalid."
            )
        self.material_kind = material_kind
        self.owner_kind = owner_kind
        self.owner_index = index
        self.component_count = components
        self.site_id = canonical_fingerprint(
            {
                "kind": "native-rod-material-site",
                "rod": rod,
                "material_kind": material_kind,
                "owner_kind": owner_kind,
                "owner_index": index,
                "component_count": components,
            }
        )


class RodMaterialWorkset(StrictModule, NonTrainableState):
    """Uniform native sites with authoritative reference strains and measures."""

    sites: tuple[RodMaterialSite, ...]
    measures: Array
    reference_strains: Array
    material_kind: RodMaterialKind = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)

    def __init__(
        self,
        sites: Sequence[RodMaterialSite],
        measures: ArrayLike,
        reference_strains: ArrayLike,
        /,
        *,
        material_kind: RodMaterialKind,
        component_count: int,
    ):
        sites_ = tuple(sites)
        components = int(component_count)
        if material_kind not in ("stretch_shear", "bend_twist") or components < 1:
            raise ValueError("Rod material workset kind or component count is invalid.")
        if any(
            not isinstance(site, RodMaterialSite)
            or site.material_kind != material_kind
            or site.component_count != components
            for site in sites_
        ):
            raise TypeError("Rod material sites do not match their workset contract.")
        owner_indices = tuple(site.owner_index for site in sites_)
        if len(set(owner_indices)) != len(owner_indices):
            raise ValueError("Rod material workset owner indices must be unique.")
        measures_ = _real_array("measures", measures, 1)
        references = _real_array("reference_strains", reference_strains, 2)
        expected = (len(sites_), components)
        if measures_.shape != (len(sites_),) or references.shape != expected:
            raise ValueError(
                "Rod material measures/reference strains have invalid shapes."
            )
        if np.any(measures_ <= 0.0):
            raise ValueError(
                "Every populated rod material site must have positive measure."
            )
        if measures_.dtype != references.dtype:
            raise TypeError(
                "Rod material measures and reference strains must share a dtype."
            )
        self.sites = sites_
        self.measures = jnp.asarray(measures_)
        self.reference_strains = jnp.asarray(references)
        self.material_kind = material_kind
        self.component_count = components
        self.site_count = len(sites_)
        self.workset_id = canonical_fingerprint(
            {
                "kind": "native-rod-material-workset",
                "material_kind": material_kind,
                "component_count": components,
                "sites": [site.site_id for site in sites_],
                "content": array_tree_fingerprint(
                    {"measures": measures_, "reference_strains": references}
                ),
            }
        )


class RodConstitutiveControl(StrictModule):
    """Typed intrinsic-strain and stiffness control at one material workset.

    ``stiffness`` is the complete instantaneous constitutive tensor, not an
    additive load. Ownership is static so orthogonal intrinsic-strain and
    stiffness actuators can be composed while duplicate owners fail closed.
    """

    intrinsic_strain: Array
    intrinsic_strain_rate: Array
    stiffness: Array
    stiffness_rate: Array
    intrinsic_owner_id: str | None = eqx.field(static=True)
    stiffness_owner_id: str | None = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        intrinsic_strain: ArrayLike,
        intrinsic_strain_rate: ArrayLike,
        stiffness: ArrayLike,
        stiffness_rate: ArrayLike,
        /,
        *,
        workset_id: str,
        material_id: str,
        control_id: str,
        intrinsic_owner_id: str | None = None,
        stiffness_owner_id: str | None = None,
    ):
        intrinsic = jnp.asarray(intrinsic_strain)
        intrinsic_rate = jnp.asarray(intrinsic_strain_rate)
        stiffness_ = jnp.asarray(stiffness)
        stiffness_rate_ = jnp.asarray(stiffness_rate)
        if intrinsic.ndim != 2 or intrinsic_rate.shape != intrinsic.shape:
            raise ValueError(
                "Intrinsic strain and its rate must be equal-shaped rank-two arrays."
            )
        expected_stiffness = intrinsic.shape + (intrinsic.shape[-1],)
        if (
            stiffness_.shape != expected_stiffness
            or stiffness_rate_.shape != expected_stiffness
        ):
            raise ValueError(
                "Controlled stiffness and its rate must have shape "
                "(sites, components, components)."
            )
        dtype = np.dtype(intrinsic.dtype)
        values = (intrinsic, intrinsic_rate, stiffness_, stiffness_rate_)
        if any(
            not jnp.issubdtype(value.dtype, jnp.inexact)
            or jnp.iscomplexobj(value)
            or np.dtype(value.dtype) != dtype
            for value in values
        ):
            raise TypeError("Rod constitutive control arrays must share one real dtype.")
        workset = _identifier(workset_id, "workset_id")
        material = _identifier(material_id, "material_id")
        identifier = _identifier(control_id, "control_id")
        intrinsic_owner = (
            None
            if intrinsic_owner_id is None
            else _identifier(intrinsic_owner_id, "intrinsic_owner_id")
        )
        stiffness_owner = (
            None
            if stiffness_owner_id is None
            else _identifier(stiffness_owner_id, "stiffness_owner_id")
        )
        self.intrinsic_strain = intrinsic
        self.intrinsic_strain_rate = intrinsic_rate
        self.stiffness = stiffness_
        self.stiffness_rate = stiffness_rate_
        self.intrinsic_owner_id = intrinsic_owner
        self.stiffness_owner_id = stiffness_owner
        self.material_id = material
        self.workset_id = workset
        self.control_id = identifier


class RodConstitutiveEvidence(StrictModule):
    """Finite-domain and thermodynamic evidence for one pure material trial."""

    maximum_rate_defect: Array
    power_residual: Array
    minimum_stiffness_eigenvalue: Array
    finite: Array
    positive_step: Array
    rate_consistent: Array
    control_compatible: Array
    stiffness_symmetric: Array
    stiffness_psd: Array
    dissipation_nonnegative: Array
    power_balanced: Array
    valid: Array
    material_id: str = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)


class RodConstitutiveResult(StrictModule):
    """Resultants, tangent, energies, powers, and uncommitted trial history."""

    resultants: Array
    elastic_resultants: Array
    viscous_resultants: Array
    elastic_strain: Array
    consistent_tangent: Array
    effective_stiffness: Array
    stored_energy_density: Array
    viscous_dissipation_density: Array
    control_source_power_density: Array
    stored_energy: Array
    stored_energy_rate: Array
    mechanical_power: Array
    viscous_dissipation_power: Array
    viscous_dissipation: Array
    control_source_power: Array
    power_residual: Array
    candidate_history: Array
    evidence: RodConstitutiveEvidence
    material_id: str = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @property
    def stress_resultants(self) -> Array:
        return self.resultants


class RodConstitutiveTrial(StrictModule):
    """Abstract pure trial evaluator; evaluation never commits material history."""

    __strict_abstract__ = True

    workset: AbstractAttribute[RodMaterialWorkset]
    plan: AbstractAttribute[Any]
    history_size: AbstractAttribute[int]
    control_size: AbstractAttribute[int]
    material_id: AbstractAttribute[str]

    def initialize_history(self, /) -> Array:
        return jnp.zeros(
            (self.workset.site_count, self.history_size),
            dtype=self.workset.reference_strains.dtype,
        )

    def initialize_control(self, /) -> RodConstitutiveControl:
        shape = (self.workset.site_count, self.workset.component_count)
        zeros = jnp.zeros(shape, dtype=self.workset.reference_strains.dtype)
        stiffness = jnp.asarray(self.plan.stiffness)
        return RodConstitutiveControl(
            zeros,
            zeros,
            stiffness,
            jnp.zeros_like(stiffness),
            material_id=self.material_id,
            workset_id=self.workset.workset_id,
            control_id=canonical_fingerprint(
                {
                    "kind": "passive-rod-constitutive-control",
                    "material": self.material_id,
                    "workset": self.workset.workset_id,
                }
            ),
        )

    @abstractmethod
    def evaluate(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        control: RodConstitutiveControl | None,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RodConstitutiveResult:
        raise NotImplementedError

    def __call__(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        control: RodConstitutiveControl | None,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RodConstitutiveResult:
        return self.evaluate(
            source_strain,
            candidate_strain,
            strain_rate,
            source_history,
            control,
            time,
            step_size,
        )


def _trial_inputs(
    trial: RodConstitutiveTrial,
    source_strain: ArrayLike,
    candidate_strain: ArrayLike,
    strain_rate: ArrayLike,
    source_history: ArrayLike,
    control: RodConstitutiveControl | None,
    time: ArrayLike,
    step_size: ArrayLike,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    RodConstitutiveControl,
    Array,
    Array,
]:
    shape = (trial.workset.site_count, trial.workset.component_count)
    dtype = np.dtype(trial.workset.reference_strains.dtype)

    def exact_array(name: str, value: ArrayLike, expected: tuple[int, ...]) -> Array:
        array = jnp.asarray(value)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}; got {array.shape}.")
        if np.dtype(array.dtype) != dtype:
            raise TypeError(f"{name} must have dtype {dtype}; got {array.dtype}.")
        return array

    source = exact_array("source_strain", source_strain, shape)
    candidate = exact_array("candidate_strain", candidate_strain, shape)
    rate = exact_array("strain_rate", strain_rate, shape)
    history = exact_array(
        "source_history",
        source_history,
        (trial.workset.site_count, trial.history_size),
    )
    control_ = trial.initialize_control() if control is None else control
    if not isinstance(control_, RodConstitutiveControl):
        raise TypeError("control must be a RodConstitutiveControl or None.")
    if (
        control_.workset_id != trial.workset.workset_id
        or control_.material_id != trial.material_id
    ):
        raise ValueError(
            "Rod constitutive control belongs to a different material/workset."
        )
    exact_array("control.intrinsic_strain", control_.intrinsic_strain, shape)
    exact_array("control.intrinsic_strain_rate", control_.intrinsic_strain_rate, shape)
    stiffness_shape = shape + (trial.workset.component_count,)
    exact_array("control.stiffness", control_.stiffness, stiffness_shape)
    exact_array("control.stiffness_rate", control_.stiffness_rate, stiffness_shape)
    time_ = jnp.asarray(time)
    step = jnp.asarray(step_size)
    if time_.shape != () or step.shape != ():
        raise ValueError("Rod material time and step_size must be scalar arrays.")
    if np.dtype(time_.dtype) != dtype or np.dtype(step.dtype) != dtype:
        raise TypeError("Rod material time and step_size must use the workset dtype.")
    return source, candidate, rate, history, control_, time_, step


def _evidence(
    trial: RodConstitutiveTrial,
    source: Array,
    candidate: Array,
    rate: Array,
    control: RodConstitutiveControl,
    time: Array,
    step: Array,
    resultants: Array,
    tangent: Array,
    stored_density: Array,
    dissipation_density: Array,
    source_power_density: Array,
    power_residual: Array,
    history: Array,
    /,
) -> RodConstitutiveEvidence:
    positive_step = jnp.isfinite(step) & (step > 0.0)
    safe_step = jnp.where(positive_step, step, jnp.ones_like(step))
    defect = rate - (candidate - source) / safe_step
    maximum_defect = (
        jnp.max(jnp.abs(defect))
        if trial.workset.site_count
        else jnp.asarray(0.0, dtype=step.dtype)
    )
    rate_scale = (
        jnp.max(jnp.abs(rate))
        if trial.workset.site_count
        else jnp.asarray(0.0, dtype=step.dtype)
    )
    tolerance = jnp.sqrt(jnp.finfo(step.dtype).eps) * jnp.maximum(1.0, rate_scale)
    rate_consistent = positive_step & (maximum_defect <= tolerance)
    stiffness_scale = (
        jnp.maximum(1.0, jnp.max(jnp.abs(control.stiffness)))
        if trial.workset.site_count
        else jnp.asarray(1.0, dtype=step.dtype)
    )
    stiffness_tolerance = 500.0 * jnp.finfo(step.dtype).eps * stiffness_scale
    symmetry_error = (
        jnp.max(jnp.abs(control.stiffness - jnp.swapaxes(control.stiffness, -1, -2)))
        if trial.workset.site_count
        else jnp.asarray(0.0, dtype=step.dtype)
    )
    stiffness_symmetric = symmetry_error <= stiffness_tolerance
    minimum_eigenvalue = (
        jnp.min(jnp.linalg.eigvalsh(control.stiffness))
        if trial.workset.site_count
        else jnp.asarray(0.0, dtype=step.dtype)
    )
    stiffness_psd = minimum_eigenvalue >= -stiffness_tolerance
    dissipation_nonnegative = jnp.all(
        dissipation_density >= -64.0 * jnp.finfo(step.dtype).eps
    )
    control_compatible = jnp.asarray(True)
    finite = (
        jnp.all(jnp.isfinite(source))
        & jnp.all(jnp.isfinite(candidate))
        & jnp.all(jnp.isfinite(rate))
        & jnp.isfinite(time)
        & jnp.all(jnp.isfinite(control.intrinsic_strain))
        & jnp.all(jnp.isfinite(control.intrinsic_strain_rate))
        & jnp.all(jnp.isfinite(control.stiffness))
        & jnp.all(jnp.isfinite(control.stiffness_rate))
        & jnp.all(jnp.isfinite(resultants))
        & jnp.all(jnp.isfinite(tangent))
        & jnp.all(jnp.isfinite(stored_density))
        & jnp.all(jnp.isfinite(dissipation_density))
        & jnp.all(jnp.isfinite(source_power_density))
        & jnp.isfinite(power_residual)
        & jnp.all(jnp.isfinite(history))
    )
    power_scale = jnp.maximum(
        1.0,
        jnp.sum(
            trial.workset.measures
            * (
                jnp.abs(stored_density) / safe_step
                + jnp.abs(dissipation_density)
                + jnp.abs(source_power_density)
            )
        ),
    )
    power_balanced = jnp.abs(power_residual) <= tolerance * power_scale
    valid = (
        finite
        & positive_step
        & rate_consistent
        & control_compatible
        & stiffness_symmetric
        & stiffness_psd
        & dissipation_nonnegative
        & power_balanced
    )
    return RodConstitutiveEvidence(
        maximum_defect,
        power_residual,
        minimum_eigenvalue,
        finite,
        positive_step,
        rate_consistent,
        control_compatible,
        stiffness_symmetric,
        stiffness_psd,
        dissipation_nonnegative,
        power_balanced,
        valid,
        trial.material_id,
        trial.workset.workset_id,
        control.control_id,
    )


class LinearElasticRodMaterialPlan(StrictModule, NonTrainableState):
    """Sitewise symmetric linear rod stiffness lowered to a pure trial law."""

    stiffness: Array
    site_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, stiffness: ArrayLike, /, *, plan_id: str | None = None):
        stiffness_ = _real_array("stiffness", stiffness, 3)
        if stiffness_.shape[1] != stiffness_.shape[2]:
            raise ValueError("Rod stiffness matrices must be square.")
        _validate_psd("stiffness", stiffness_)
        generated = canonical_fingerprint(
            {
                "kind": "linear-elastic-rod-material-plan",
                "stiffness": array_tree_fingerprint(stiffness_),
            }
        )
        identifier = generated if plan_id is None else _identifier(plan_id, "plan_id")
        self.stiffness = jnp.asarray(stiffness_)
        self.site_count = int(stiffness_.shape[0])
        self.component_count = int(stiffness_.shape[1])
        self.plan_id = identifier

    def prepare(self, workset: RodMaterialWorkset, /) -> PreparedLinearElasticRodMaterial:
        return PreparedLinearElasticRodMaterial(self, workset)


class PreparedLinearElasticRodMaterial(RodConstitutiveTrial, NonTrainableState):
    """Prepared zero-history linear elastic rod trial evaluator."""

    plan: LinearElasticRodMaterialPlan
    workset: RodMaterialWorkset
    history_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self, plan: LinearElasticRodMaterialPlan, workset: RodMaterialWorkset, /
    ):
        if not isinstance(plan, LinearElasticRodMaterialPlan):
            raise TypeError("plan must be a LinearElasticRodMaterialPlan.")
        if not isinstance(workset, RodMaterialWorkset):
            raise TypeError("workset must be a RodMaterialWorkset.")
        if (
            plan.site_count != workset.site_count
            or plan.component_count != workset.component_count
            or plan.stiffness.dtype != workset.reference_strains.dtype
        ):
            raise ValueError("Linear rod material plan and workset are incompatible.")
        self.plan = plan
        self.workset = workset
        self.history_size = 0
        self.control_size = 0
        self.material_id = canonical_fingerprint(
            {
                "kind": "prepared-linear-elastic-rod-material",
                "plan": plan.plan_id,
                "workset": workset.workset_id,
            }
        )

    def evaluate(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        control: RodConstitutiveControl | None,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RodConstitutiveResult:
        source, candidate, rate, history, control_, time_, step = _trial_inputs(
            self,
            source_strain,
            candidate_strain,
            strain_rate,
            source_history,
            control,
            time,
            step_size,
        )
        elastic = candidate - self.workset.reference_strains - control_.intrinsic_strain
        elastic_resultants = ein.contract("sij,sj->si", control_.stiffness, elastic)
        viscous_resultants = jnp.zeros_like(elastic_resultants)
        resultants = elastic_resultants
        stored_density = 0.5 * ein.contract(
            "si,sij,sj->s", elastic, control_.stiffness, elastic
        )
        stiffness_power_density = 0.5 * ein.contract(
            "si,sij,sj->s", elastic, control_.stiffness_rate, elastic
        )
        source_power_density = (
            -ein.contract("si,si->s", elastic_resultants, control_.intrinsic_strain_rate)
            + stiffness_power_density
        )
        stored_rate_density = (
            ein.contract(
                "si,si->s",
                elastic_resultants,
                rate - control_.intrinsic_strain_rate,
            )
            + stiffness_power_density
        )
        mechanical_power_density = -ein.contract("si,si->s", resultants, rate)
        dissipation_density = jnp.zeros_like(stored_density)
        measures = self.workset.measures
        stored = jnp.sum(measures * stored_density)
        stored_rate = jnp.sum(measures * stored_rate_density)
        mechanical_power = jnp.sum(measures * mechanical_power_density)
        dissipation_power = jnp.asarray(0.0, dtype=stored.dtype)
        dissipation = jnp.asarray(0.0, dtype=stored.dtype)
        source_power = jnp.sum(measures * source_power_density)
        power_residual = stored_rate + mechanical_power + dissipation_power - source_power
        evidence = _evidence(
            self,
            source,
            candidate,
            rate,
            control_,
            time_,
            step,
            resultants,
            control_.stiffness,
            stored_density,
            dissipation_density,
            source_power_density,
            power_residual,
            history,
        )
        return RodConstitutiveResult(
            resultants,
            elastic_resultants,
            viscous_resultants,
            elastic,
            control_.stiffness,
            control_.stiffness,
            stored_density,
            dissipation_density,
            source_power_density,
            stored,
            stored_rate,
            mechanical_power,
            dissipation_power,
            dissipation,
            source_power,
            power_residual,
            history,
            evidence,
            self.material_id,
            self.workset.workset_id,
            control_.control_id,
        )


class KelvinVoigtRodMaterialPlan(StrictModule, NonTrainableState):
    """Sitewise Kelvin-Voigt rod law with zero committed material history."""

    stiffness: Array
    viscosity: Array
    site_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: ArrayLike,
        viscosity: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        stiffness_ = _real_array("stiffness", stiffness, 3)
        viscosity_ = _real_array("viscosity", viscosity, 3)
        if (
            stiffness_.shape != viscosity_.shape
            or stiffness_.shape[1] != stiffness_.shape[2]
        ):
            raise ValueError(
                "Rod stiffness/viscosity must be equal-shaped square matrices."
            )
        _validate_psd("stiffness", stiffness_)
        _validate_psd("viscosity", viscosity_)
        if stiffness_.dtype != viscosity_.dtype:
            raise TypeError("Rod stiffness and viscosity must share a dtype.")
        generated = canonical_fingerprint(
            {
                "kind": "kelvin-voigt-rod-material-plan",
                "content": array_tree_fingerprint(
                    {"stiffness": stiffness_, "viscosity": viscosity_}
                ),
            }
        )
        identifier = generated if plan_id is None else _identifier(plan_id, "plan_id")
        self.stiffness = jnp.asarray(stiffness_)
        self.viscosity = jnp.asarray(viscosity_)
        self.site_count = int(stiffness_.shape[0])
        self.component_count = int(stiffness_.shape[1])
        self.plan_id = identifier

    def prepare(self, workset: RodMaterialWorkset, /) -> PreparedKelvinVoigtRodMaterial:
        return PreparedKelvinVoigtRodMaterial(self, workset)


class PreparedKelvinVoigtRodMaterial(RodConstitutiveTrial, NonTrainableState):
    """Prepared zero-history Kelvin-Voigt trial with algorithmic tangent."""

    plan: KelvinVoigtRodMaterialPlan
    workset: RodMaterialWorkset
    history_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(self, plan: KelvinVoigtRodMaterialPlan, workset: RodMaterialWorkset, /):
        if not isinstance(plan, KelvinVoigtRodMaterialPlan):
            raise TypeError("plan must be a KelvinVoigtRodMaterialPlan.")
        if not isinstance(workset, RodMaterialWorkset):
            raise TypeError("workset must be a RodMaterialWorkset.")
        if (
            plan.site_count != workset.site_count
            or plan.component_count != workset.component_count
            or plan.stiffness.dtype != workset.reference_strains.dtype
        ):
            raise ValueError(
                "Kelvin-Voigt rod material plan and workset are incompatible."
            )
        self.plan = plan
        self.workset = workset
        self.history_size = 0
        self.control_size = 0
        self.material_id = canonical_fingerprint(
            {
                "kind": "prepared-kelvin-voigt-rod-material",
                "plan": plan.plan_id,
                "workset": workset.workset_id,
            }
        )

    def evaluate(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        control: RodConstitutiveControl | None,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RodConstitutiveResult:
        source, candidate, rate, history, control_, time_, step = _trial_inputs(
            self,
            source_strain,
            candidate_strain,
            strain_rate,
            source_history,
            control,
            time,
            step_size,
        )
        elastic = candidate - self.workset.reference_strains - control_.intrinsic_strain
        elastic_resultants = ein.contract("sij,sj->si", control_.stiffness, elastic)
        viscous_resultants = ein.contract("sij,sj->si", self.plan.viscosity, rate)
        resultants = elastic_resultants + viscous_resultants
        safe_step = jnp.where(step > 0.0, step, jnp.ones_like(step))
        tangent = control_.stiffness + self.plan.viscosity / safe_step
        stored_density = 0.5 * ein.contract(
            "si,sij,sj->s", elastic, control_.stiffness, elastic
        )
        stiffness_power_density = 0.5 * ein.contract(
            "si,sij,sj->s", elastic, control_.stiffness_rate, elastic
        )
        source_power_density = (
            -ein.contract("si,si->s", elastic_resultants, control_.intrinsic_strain_rate)
            + stiffness_power_density
        )
        stored_rate_density = (
            ein.contract(
                "si,si->s",
                elastic_resultants,
                rate - control_.intrinsic_strain_rate,
            )
            + stiffness_power_density
        )
        mechanical_power_density = -ein.contract("si,si->s", resultants, rate)
        dissipation_density = ein.contract(
            "si,sij,sj->s", rate, self.plan.viscosity, rate
        )
        measures = self.workset.measures
        stored = jnp.sum(measures * stored_density)
        stored_rate = jnp.sum(measures * stored_rate_density)
        mechanical_power = jnp.sum(measures * mechanical_power_density)
        dissipation_power = jnp.sum(measures * dissipation_density)
        dissipation = jnp.maximum(step * dissipation_power, 0.0)
        source_power = jnp.sum(measures * source_power_density)
        power_residual = stored_rate + mechanical_power + dissipation_power - source_power
        evidence = _evidence(
            self,
            source,
            candidate,
            rate,
            control_,
            time_,
            step,
            resultants,
            tangent,
            stored_density,
            dissipation_density,
            source_power_density,
            power_residual,
            history,
        )
        return RodConstitutiveResult(
            resultants,
            elastic_resultants,
            viscous_resultants,
            elastic,
            tangent,
            control_.stiffness,
            stored_density,
            dissipation_density,
            source_power_density,
            stored,
            stored_rate,
            mechanical_power,
            dissipation_power,
            dissipation,
            source_power,
            power_residual,
            history,
            evidence,
            self.material_id,
            self.workset.workset_id,
            control_.control_id,
        )


__all__ = [
    "KelvinVoigtRodMaterialPlan",
    "LinearElasticRodMaterialPlan",
    "PreparedKelvinVoigtRodMaterial",
    "PreparedLinearElasticRodMaterial",
    "RodConstitutiveControl",
    "RodConstitutiveEvidence",
    "RodConstitutiveResult",
    "RodConstitutiveTrial",
    "RodMaterialKind",
    "RodMaterialOwnerKind",
    "RodMaterialSite",
    "RodMaterialWorkset",
]
