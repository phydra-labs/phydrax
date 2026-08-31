#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._hybrid_sensitivity import HybridSensitivityMode
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LatticeBoltzmannGeometrySensitivityPolicy(StrictModule, NonTrainableState):
    """Branchwise, surrogate, or event-aware policy for fixed-shape geometry AD."""

    mode: HybridSensitivityMode = eqx.field(static=True)
    classification_margin: float = eqx.field(static=True)
    link_margin: float = eqx.field(static=True)
    event_time_margin: float = eqx.field(static=True)
    surrogate_width: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: HybridSensitivityMode = HybridSensitivityMode.SHARP_BRANCHWISE,
        classification_margin: float = 1.0e-8,
        link_margin: float = 1.0e-8,
        event_time_margin: float = 1.0e-8,
        surrogate_width: float = 1.0e-3,
    ):
        if not isinstance(mode, HybridSensitivityMode):
            raise TypeError("mode must be a HybridSensitivityMode.")
        values = tuple(
            float(value)
            for value in (
                classification_margin,
                link_margin,
                event_time_margin,
                surrogate_width,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Geometry sensitivity margins and width must be positive.")
        (
            self.classification_margin,
            self.link_margin,
            self.event_time_margin,
            self.surrogate_width,
        ) = values
        self.mode = mode
        self.policy_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-geometry-sensitivity-policy",
                "mode": mode.value,
                "classification_margin": values[0],
                "link_margin": values[1],
                "event_time_margin": values[2],
                "surrogate_width": values[3],
            }
        )


class LatticeBoltzmannGeometrySensitivityMargins(StrictModule, NonTrainableState):
    """Runtime distances to classification, link, and localized-event switches."""

    minimum_classification_margin: Array
    minimum_link_margin: Array
    event_time_margin: Array
    topology_unchanged: Array
    event_requested: Array
    event_localized: Array
    forward_successful: Array

    def __init__(
        self,
        minimum_classification_margin: ArrayLike,
        minimum_link_margin: ArrayLike,
        event_time_margin: ArrayLike,
        /,
        *,
        topology_unchanged: ArrayLike = True,
        event_requested: ArrayLike = False,
        event_localized: ArrayLike = False,
        forward_successful: ArrayLike = True,
    ):
        classification = jnp.asarray(minimum_classification_margin)
        link = jnp.asarray(minimum_link_margin)
        event = jnp.asarray(event_time_margin)
        booleans = tuple(
            jnp.asarray(value)
            for value in (
                topology_unchanged,
                event_requested,
                event_localized,
                forward_successful,
            )
        )
        if classification.shape != () or link.shape != () or event.shape != ():
            raise ValueError("Geometry validity margins must be scalar arrays.")
        if any(value.shape != () or value.dtype.kind != "b" for value in booleans):
            raise ValueError("Geometry validity flags must be scalar booleans.")
        classification = eqx.error_if(
            classification,
            ~jnp.isfinite(classification)
            | ~jnp.isfinite(link)
            | ~jnp.isfinite(event)
            | (classification < 0.0)
            | (link < 0.0)
            | (event < 0.0),
            "Geometry sensitivity validity margins must be finite and nonnegative.",
        )
        self.minimum_classification_margin = classification
        self.minimum_link_margin = link
        self.event_time_margin = event
        (
            self.topology_unchanged,
            self.event_requested,
            self.event_localized,
            self.forward_successful,
        ) = booleans


class LatticeBoltzmannGeometryValidityCertificate(StrictModule, NonTrainableState):
    """Mode-specific local validity with all physical switch margins retained."""

    minimum_classification_margin: Array
    minimum_link_margin: Array
    event_time_margin: Array
    topology_unchanged: Array
    event_requested: Array
    event_localized: Array
    forward_successful: Array
    locally_valid: Array
    policy_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


class LatticeBoltzmannGeometrySensitivityResult(StrictModule):
    primal: Any
    sensitivity: Any
    certificate: LatticeBoltzmannGeometryValidityCertificate
    usable: Array
    mode: HybridSensitivityMode = eqx.field(static=True)


def lattice_boltzmann_geometry_validity_certificate(
    margins: LatticeBoltzmannGeometrySensitivityMargins,
    policy: LatticeBoltzmannGeometrySensitivityPolicy,
    /,
) -> LatticeBoltzmannGeometryValidityCertificate:
    if not isinstance(margins, LatticeBoltzmannGeometrySensitivityMargins):
        raise TypeError("margins must be LatticeBoltzmannGeometrySensitivityMargins.")
    if not isinstance(policy, LatticeBoltzmannGeometrySensitivityPolicy):
        raise TypeError("policy must be LatticeBoltzmannGeometrySensitivityPolicy.")
    classification_valid = (
        margins.minimum_classification_margin >= policy.classification_margin
    )
    link_valid = margins.minimum_link_margin >= policy.link_margin
    event_valid = margins.event_time_margin >= policy.event_time_margin
    if policy.mode is HybridSensitivityMode.SHARP_BRANCHWISE:
        locally_valid = (
            margins.forward_successful
            & classification_valid
            & link_valid
            & margins.topology_unchanged
            & ~margins.event_requested
        )
    elif policy.mode is HybridSensitivityMode.SMOOTH_SURROGATE:
        locally_valid = margins.forward_successful & link_valid
    else:
        localized_if_requested = ~margins.event_requested | margins.event_localized
        locally_valid = (
            margins.forward_successful
            & classification_valid
            & link_valid
            & event_valid
            & localized_if_requested
        )
    return LatticeBoltzmannGeometryValidityCertificate(
        margins.minimum_classification_margin,
        margins.minimum_link_margin,
        margins.event_time_margin,
        margins.topology_unchanged,
        margins.event_requested,
        margins.event_localized,
        margins.forward_successful,
        locally_valid,
        policy.policy_id,
        canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-geometry-validity-certificate",
                "policy": policy.policy_id,
                "schema": "classification-link-event-margins",
            }
        ),
    )


def _selected_function(
    sharp_function: Callable[[PyTree[Any]], PyTree[Any]],
    policy: LatticeBoltzmannGeometrySensitivityPolicy,
    smooth_function: Callable[[PyTree[Any]], PyTree[Any]] | None,
    event_aware_function: Callable[[PyTree[Any]], PyTree[Any]] | None,
    /,
) -> Callable[[PyTree[Any]], PyTree[Any]]:
    if not callable(sharp_function):
        raise TypeError("sharp_function must be callable.")
    if policy.mode is HybridSensitivityMode.SHARP_BRANCHWISE:
        return sharp_function
    if policy.mode is HybridSensitivityMode.SMOOTH_SURROGATE:
        if not callable(smooth_function):
            raise TypeError("Smooth-surrogate sensitivity requires smooth_function.")
        return smooth_function
    if not callable(event_aware_function):
        raise TypeError("Event-aware sensitivity requires event_aware_function.")
    return event_aware_function


def _invalid_sensitivity(tree: PyTree[Any], /) -> PyTree[Any]:
    return jax.tree.map(
        lambda leaf: jnp.full_like(leaf, jnp.nan) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def lattice_boltzmann_geometry_jvp(
    sharp_function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    direction: PyTree[Any],
    margins: LatticeBoltzmannGeometrySensitivityMargins,
    policy: LatticeBoltzmannGeometrySensitivityPolicy,
    /,
    *,
    smooth_function: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    event_aware_function: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
) -> LatticeBoltzmannGeometrySensitivityResult:
    """Differentiate the policy-selected fixed branch, surrogate, or event map."""

    function = _selected_function(
        sharp_function,
        policy,
        smooth_function,
        event_aware_function,
    )
    primal, tangent = jax.jvp(function, (parameters,), (direction,))
    certificate = lattice_boltzmann_geometry_validity_certificate(margins, policy)
    sensitivity = jax.lax.cond(
        certificate.locally_valid,
        lambda value: value,
        _invalid_sensitivity,
        tangent,
    )
    return LatticeBoltzmannGeometrySensitivityResult(
        primal,
        sensitivity,
        certificate,
        certificate.locally_valid,
        policy.mode,
    )


def lattice_boltzmann_geometry_vjp(
    sharp_function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    cotangent: PyTree[Any],
    margins: LatticeBoltzmannGeometrySensitivityMargins,
    policy: LatticeBoltzmannGeometrySensitivityPolicy,
    /,
    *,
    smooth_function: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    event_aware_function: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
) -> LatticeBoltzmannGeometrySensitivityResult:
    """Reverse-mode companion to :func:`lattice_boltzmann_geometry_jvp`."""

    function = _selected_function(
        sharp_function,
        policy,
        smooth_function,
        event_aware_function,
    )
    primal, pullback = jax.vjp(function, parameters)
    sensitivity = pullback(cotangent)[0]
    certificate = lattice_boltzmann_geometry_validity_certificate(margins, policy)
    sensitivity = jax.lax.cond(
        certificate.locally_valid,
        lambda value: value,
        _invalid_sensitivity,
        sensitivity,
    )
    return LatticeBoltzmannGeometrySensitivityResult(
        primal,
        sensitivity,
        certificate,
        certificate.locally_valid,
        policy.mode,
    )


__all__ = [
    "HybridSensitivityMode",
    "LatticeBoltzmannGeometrySensitivityMargins",
    "LatticeBoltzmannGeometrySensitivityPolicy",
    "LatticeBoltzmannGeometrySensitivityResult",
    "LatticeBoltzmannGeometryValidityCertificate",
    "lattice_boltzmann_geometry_jvp",
    "lattice_boltzmann_geometry_validity_certificate",
    "lattice_boltzmann_geometry_vjp",
]
