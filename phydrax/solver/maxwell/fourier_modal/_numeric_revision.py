#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....lifecycle import NumericRevision
from ._continuous import PreparedContinuousFourierModalLayer
from ._contracts import (
    ContinuousFourierModalLayer,
    FourierModalLayer,
    FourierModalSourcePlane,
    PeriodicMaxwellPort,
)
from ._factorization import AnalyticInterfaceFramePlan, VectorFourierFactorizationPlan
from ._runtime import (
    _canonical_material_samples,
    PreparedFourierModalLayer,
    PreparedFourierModalMaxwell,
)


def _factorization_arrays(factorization, /) -> object:
    if isinstance(factorization, VectorFourierFactorizationPlan) and isinstance(
        factorization.frame, AnalyticInterfaceFramePlan
    ):
        return factorization.frame.tangent_field
    return ()


def _frame_provenance_id(factorization, /) -> str | None:
    return (
        factorization.frame.frame_id
        if isinstance(factorization, VectorFourierFactorizationPlan)
        else None
    )


def _contains_tracer(value: object, /) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def fourier_modal_physical_stack_digest(
    prepared: PreparedFourierModalMaxwell,
    /,
) -> str:
    """Bind frequency-independent geometry and material-response provenance."""
    if not isinstance(prepared, PreparedFourierModalMaxwell):
        raise TypeError("prepared must be PreparedFourierModalMaxwell.")
    problem = prepared.problem
    arrays: dict[str, object] = {
        "primitive_vectors": problem.harmonics.primitive_vectors,
    }
    static_elements: list[dict[str, object]] = []
    thicknesses = []
    for index, element in enumerate(problem.elements):
        if isinstance(element, FourierModalSourcePlane):
            static_elements.append({"kind": "source", "source_id": element.source_id})
            continue
        if isinstance(element, ContinuousFourierModalLayer):
            raise ValueError(
                "A content-bound physical-stack digest is unavailable for "
                "continuous-z profile callables."
            )
        thicknesses.append(element.thickness)
        arrays[f"element_{index}_translation"] = element.translation
        static_elements.append(
            {
                "kind": "layer",
                "layer_id": element.layer_id,
                "material_id": element.material.material_id,
                "material_role": element.material.material_role,
                "origin_evidence_id": element.material.origin_evidence_id,
                "passive": element.material.passive,
                "reciprocal": element.material.reciprocal,
                "factorization_id": element.factorization.plan_id,
                "frame_id": _frame_provenance_id(element.factorization),
            }
        )
    if _contains_tracer((arrays, thicknesses)):
        raise ValueError(
            "A physical-stack digest requires host-materialized numeric inputs."
        )
    thickness_values = np.asarray([np.asarray(value).item() for value in thicknesses])
    total = np.sum(thickness_values)
    arrays["normalized_layer_thicknesses"] = (
        thickness_values / total
        if np.abs(total) > 0.0
        else np.zeros_like(thickness_values)
    )

    def port_descriptor(port) -> dict[str, object]:
        return {
            "port_id": port.port_id,
            "material_id": port.material.material_id,
            "material_role": port.material.material_role,
            "origin_evidence_id": port.material.origin_evidence_id,
            "passive": port.material.passive,
            "reciprocal": port.material.reciprocal,
            "factorization_id": port.factorization.plan_id
            if isinstance(port, PeriodicMaxwellPort)
            else None,
            "frame_id": _frame_provenance_id(port.factorization)
            if isinstance(port, PeriodicMaxwellPort)
            else None,
            "mode_policy": port.mode_policy
            if isinstance(port, PeriodicMaxwellPort)
            else None,
        }

    return canonical_fingerprint(
        {
            "kind": "fourier-modal-physical-stack-state",
            "arrays": array_tree_fingerprint(arrays),
            "superstrate": port_descriptor(problem.superstrate),
            "elements": static_elements,
            "substrate": port_descriptor(problem.substrate),
        }
    )


def fourier_modal_physical_state_digest(
    prepared: PreparedFourierModalMaxwell,
    /,
    *,
    source_parents: Sequence[NumericRevision] = (),
) -> str:
    """Digest physical numeric inputs while excluding harmonic discretization."""
    if not isinstance(prepared, PreparedFourierModalMaxwell):
        raise TypeError("prepared must be PreparedFourierModalMaxwell.")
    parents = tuple(source_parents)
    if any(not isinstance(parent, NumericRevision) for parent in parents):
        raise TypeError("source_parents must contain NumericRevision values.")
    problem = prepared.problem
    if problem.source_ids and len(parents) != len(problem.source_ids):
        raise ValueError("Every source plane requires one host NumericRevision parent.")
    arrays: dict[str, object] = {
        "angular_frequency": problem.angular_frequency,
        "bloch_wavevector": problem.bloch_wavevector,
        "primitive_vectors": problem.harmonics.primitive_vectors,
        "superstrate_reference_distance": problem.superstrate.reference_distance,
        "substrate_reference_distance": problem.substrate.reference_distance,
        "superstrate_material": (
            problem.superstrate.material.permittivity,
            problem.superstrate.material.permeability,
            problem.superstrate.material.magnetoelectric_xi,
            problem.superstrate.material.magnetoelectric_zeta,
        ),
        "substrate_material": (
            problem.substrate.material.permittivity,
            problem.substrate.material.permeability,
            problem.substrate.material.magnetoelectric_xi,
            problem.substrate.material.magnetoelectric_zeta,
        ),
        "superstrate_frame": _factorization_arrays(problem.superstrate.factorization)
        if isinstance(problem.superstrate, PeriodicMaxwellPort)
        else (),
        "substrate_frame": _factorization_arrays(problem.substrate.factorization)
        if isinstance(problem.substrate, PeriodicMaxwellPort)
        else (),
    }
    static_elements: list[dict[str, object]] = []
    for index, (element, prepared_element) in enumerate(
        zip(problem.elements, prepared.elements, strict=True)
    ):
        if isinstance(element, FourierModalSourcePlane):
            static_elements.append({"kind": "source", "source_id": element.source_id})
            continue
        arrays[f"element_{index}_thickness"] = element.thickness
        arrays[f"element_{index}_frame"] = _factorization_arrays(element.factorization)
        if isinstance(element, FourierModalLayer):
            arrays[f"element_{index}_translation"] = element.translation
            arrays[f"element_{index}_material"] = (
                element.material.permittivity,
                element.material.permeability,
                element.material.magnetoelectric_xi,
                element.material.magnetoelectric_zeta,
            )
            static_elements.append(
                {
                    "kind": "layer",
                    "layer_id": element.layer_id,
                    "material_id": element.material.material_id,
                    "material_role": element.material.material_role,
                    "origin_evidence_id": element.material.origin_evidence_id,
                    "passive": element.material.passive,
                    "reciprocal": element.material.reciprocal,
                    "factorization_id": element.factorization.plan_id,
                }
            )
        else:
            if not isinstance(prepared_element, PreparedContinuousFourierModalLayer):
                raise TypeError("Prepared continuous-layer topology does not match.")
            raise ValueError(
                "A harmonic-independent physical-state digest is unavailable for "
                "continuous-z profile callables."
            )
    if _contains_tracer(arrays):
        raise ValueError(
            "A physical-state digest requires host-materialized numeric inputs."
        )

    def port_descriptor(port) -> dict[str, object]:
        return {
            "port_id": port.port_id,
            "material_id": port.material.material_id,
            "material_role": port.material.material_role,
            "origin_evidence_id": port.material.origin_evidence_id,
            "factorization_id": port.factorization.plan_id
            if isinstance(port, PeriodicMaxwellPort)
            else None,
            "passive": port.material.passive,
            "reciprocal": port.material.reciprocal,
            "mode_policy": port.mode_policy
            if isinstance(port, PeriodicMaxwellPort)
            else None,
        }

    return canonical_fingerprint(
        {
            "kind": "fourier-modal-physical-state",
            "arrays": array_tree_fingerprint(arrays),
            "superstrate": port_descriptor(problem.superstrate),
            "elements": static_elements,
            "substrate": port_descriptor(problem.substrate),
            "source_parents": [parent.content_digest for parent in parents],
        }
    )


def fourier_modal_numeric_revision(
    prepared: PreparedFourierModalMaxwell,
    /,
    *,
    source_parents: Sequence[NumericRevision] = (),
    label: str = "fourier-modal-maxwell",
) -> NumericRevision:
    """Content-bind one host-materialized prepared Fourier-modal numeric state."""
    if not isinstance(prepared, PreparedFourierModalMaxwell):
        raise TypeError("prepared must be PreparedFourierModalMaxwell.")
    parents = tuple(source_parents)
    if any(not isinstance(parent, NumericRevision) for parent in parents):
        raise TypeError("source_parents must contain NumericRevision values.")
    problem = prepared.problem
    if problem.source_ids and len(parents) != len(problem.source_ids):
        raise ValueError("Every source plane requires one host NumericRevision parent.")
    physical_stack_digest = fourier_modal_physical_stack_digest(prepared)
    physical_state_digest = fourier_modal_physical_state_digest(
        prepared, source_parents=parents
    )
    arrays: dict[str, object] = {
        "angular_frequency": problem.angular_frequency,
        "bloch_wavevector": problem.bloch_wavevector,
        "primitive_vectors": problem.harmonics.primitive_vectors,
        "superstrate_reference_distance": problem.superstrate.reference_distance,
        "substrate_reference_distance": problem.substrate.reference_distance,
        "superstrate_material": _canonical_material_samples(
            problem.superstrate.material, problem
        ),
        "substrate_material": _canonical_material_samples(
            problem.substrate.material, problem
        ),
        "superstrate_factorization": _factorization_arrays(
            problem.superstrate.factorization
        )
        if isinstance(problem.superstrate, PeriodicMaxwellPort)
        else (),
        "substrate_factorization": _factorization_arrays(problem.substrate.factorization)
        if isinstance(problem.substrate, PeriodicMaxwellPort)
        else (),
    }
    static_elements: list[dict[str, object]] = []
    for index, (element, prepared_element) in enumerate(
        zip(problem.elements, prepared.elements, strict=True)
    ):
        if isinstance(element, FourierModalSourcePlane):
            static_elements.append({"kind": "source", "source_id": element.source_id})
            continue
        static_elements.append(
            {
                "kind": "continuous"
                if isinstance(element, ContinuousFourierModalLayer)
                else "layer",
                "layer_id": element.layer_id,
                "factorization_id": element.factorization.plan_id,
                "material_id": None
                if isinstance(element, ContinuousFourierModalLayer)
                else element.material.material_id,
                "material_role": None
                if isinstance(element, ContinuousFourierModalLayer)
                else element.material.material_role,
                "origin_evidence_id": None
                if isinstance(element, ContinuousFourierModalLayer)
                else element.material.origin_evidence_id,
            }
        )
        arrays[f"element_{index}_thickness"] = element.thickness
        arrays[f"element_{index}_factorization"] = _factorization_arrays(
            element.factorization
        )
        if isinstance(element, FourierModalLayer):
            arrays[f"element_{index}_translation"] = element.translation
            arrays[f"element_{index}_material"] = _canonical_material_samples(
                element.material, problem
            )
            if not isinstance(prepared_element, PreparedFourierModalLayer):
                raise TypeError("Prepared layer topology does not match the problem.")
        else:
            if not isinstance(prepared_element, PreparedContinuousFourierModalLayer):
                raise TypeError("Prepared continuous-layer topology does not match.")
            arrays[f"element_{index}_segment_edges"] = prepared_element.segment_edges
            arrays[f"element_{index}_segment_active"] = prepared_element.segment_active
            arrays[f"element_{index}_segment_defects"] = prepared_element.segment_defects
            arrays[f"element_{index}_segment_prefix_boundaries"] = (
                prepared_element.segment_prefix_boundaries
            )
            arrays[f"element_{index}_boundary"] = prepared_element.boundary
            arrays[f"element_{index}_maximum_defect"] = prepared_element.maximum_defect
            arrays[f"element_{index}_maximum_constitutive_residual"] = (
                prepared_element.maximum_constitutive_residual
            )
            arrays[f"element_{index}_profile_finite"] = prepared_element.profile_finite
            arrays[f"element_{index}_status"] = prepared_element.status
            arrays[f"element_{index}_successful"] = prepared_element.successful
    if _contains_tracer(arrays):
        raise ValueError(
            "A Fourier-modal NumericRevision requires host-materialized numeric inputs."
        )
    parent_digests = tuple(parent.content_digest for parent in parents)
    content_digest = canonical_fingerprint(
        {
            "kind": "fourier-modal-numeric-state",
            "arrays": array_tree_fingerprint(arrays),
            "harmonic_layout_id": problem.harmonics.plan.layout.layout_id,
            "physical_state_digest": physical_state_digest,
            "physical_stack_digest": physical_stack_digest,
            "problem_numeric_version": problem.numeric_version,
            "superstrate": {
                "port_id": problem.superstrate.port_id,
                "material_id": problem.superstrate.material.material_id,
                "material_role": problem.superstrate.material.material_role,
                "origin_evidence_id": problem.superstrate.material.origin_evidence_id,
                "factorization_id": problem.superstrate.factorization.plan_id
                if isinstance(problem.superstrate, PeriodicMaxwellPort)
                else None,
                "mode_policy": problem.superstrate.mode_policy
                if isinstance(problem.superstrate, PeriodicMaxwellPort)
                else None,
            },
            "elements": static_elements,
            "substrate": {
                "port_id": problem.substrate.port_id,
                "material_id": problem.substrate.material.material_id,
                "material_role": problem.substrate.material.material_role,
                "origin_evidence_id": problem.substrate.material.origin_evidence_id,
                "factorization_id": problem.substrate.factorization.plan_id
                if isinstance(problem.substrate, PeriodicMaxwellPort)
                else None,
                "mode_policy": problem.substrate.mode_policy
                if isinstance(problem.substrate, PeriodicMaxwellPort)
                else None,
            },
            "source_parents": list(parent_digests),
        }
    )
    return NumericRevision(
        content_digest,
        label=label,
        metadata=(
            ("problem_id", problem.problem_id),
            ("preparation_id", prepared.preparation_id),
            ("physical_state_digest", physical_state_digest),
            ("physical_stack_digest", physical_stack_digest),
        ),
    )


def require_fourier_modal_numeric_revision(
    prepared: PreparedFourierModalMaxwell,
    revision: NumericRevision,
    /,
    *,
    source_parents: Sequence[NumericRevision] = (),
) -> None:
    """Fail closed unless a revision binds the complete prepared numeric state."""
    if not isinstance(revision, NumericRevision):
        raise TypeError("numeric_revision must be NumericRevision.")
    expected = fourier_modal_numeric_revision(
        prepared, source_parents=source_parents, label=revision.label
    )
    revision_metadata = dict(revision.metadata)
    expected_metadata = dict(expected.metadata)
    if (
        revision.content_digest != expected.content_digest
        or revision_metadata.get("physical_state_digest")
        != expected_metadata["physical_state_digest"]
        or revision_metadata.get("physical_stack_digest")
        != expected_metadata["physical_stack_digest"]
    ):
        raise ValueError("numeric_revision does not bind this prepared numeric state.")


__all__ = [
    "fourier_modal_numeric_revision",
    "fourier_modal_physical_state_digest",
    "fourier_modal_physical_stack_digest",
    "require_fourier_modal_numeric_revision",
]
