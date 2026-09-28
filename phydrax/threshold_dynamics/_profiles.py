#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profiles of hard-label threshold dynamics."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def threshold_dynamics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Unreleased profiles; `tools/threshold_dynamics_qualification.py` gates them."""
    specs = (
        (
            "threshold-dynamics.periodic-multiphase",
            {
                "route": "periodic-fourier",
                "labels": "dense",
                "kernel": "gaussian",
                "uniform-coefficients": "structured-off-diagonal",
            },
        ),
        (
            "threshold-dynamics.exact-volumes",
            {"assignment": "capacitated-auction", "site-measure": "equal"},
        ),
        (
            "threshold-dynamics.sparse-labels",
            {
                "route": "periodic-box-stencil",
                "candidates": "brick-halo",
                "exactness": "distinct-periodic-residues",
            },
        ),
        (
            "threshold-dynamics.mesh-heat",
            {
                "route": "taylor-exponential-action",
                "labels": "dense",
                "evidence": "native-provenance-work-resources",
            },
        ),
        (
            "threshold-dynamics.gas-diffusion-coarsening",
            {"pressure": "auction-dual", "cells": "incompressible"},
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attributes),),
            required_gates=(
                "analytic-control",
                "reference-qualification",
                "public-workflow",
            ),
        )
        for name, attributes in specs
    )


__all__ = ["threshold_dynamics_candidate_profiles"]
