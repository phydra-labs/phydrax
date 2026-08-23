#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
from jaxtyping import Array

from .._strict import StrictModule
from ..geometry.complex import (
    ComplexProjectiveAtlas,
    fermat_hypersurface,
    ProjectiveHypersurface,
    sample_projective_hypersurface,
)
from ..nn.models import ProjectiveInvariantPotential
from ._calabi_yau import CalabiYauMetricProblem


class CalabiYauCampaign(StrictModule):
    name: str
    hypersurface: ProjectiveHypersurface
    problem: CalabiYauMetricProblem

    def __init__(
        self,
        name: str,
        hypersurface: ProjectiveHypersurface,
        problem: CalabiYauMetricProblem,
        /,
    ):
        self.name = str(name)
        self.hypersurface = hypersurface
        self.problem = problem


def cp1_calibration() -> ComplexProjectiveAtlas:
    return ComplexProjectiveAtlas(1)


def prepare_fermat_calabi_yau(
    projective_dimension: int,
    key: Array,
    /,
    *,
    line_count: int,
    width: int = 32,
    depth: int = 2,
) -> CalabiYauCampaign:
    dimension = int(projective_dimension)
    if dimension < 2:
        raise ValueError("Calabi–Yau hypersurface campaigns require CP^N with N >= 2.")
    sample_key, model_key = jax.random.split(key)
    hypersurface = fermat_hypersurface(dimension)
    samples = sample_projective_hypersurface(hypersurface, sample_key, int(line_count))
    model = ProjectiveInvariantPotential(
        dimension + 1,
        model_key,
        width=width,
        depth=depth,
        potential_id=f"fermat-{dimension + 1}-potential",
    )
    problem = CalabiYauMetricProblem(hypersurface, samples, model)
    names = {2: "fermat-elliptic", 3: "fermat-quartic-k3", 4: "fermat-quintic"}
    return CalabiYauCampaign(
        names.get(dimension, f"fermat-cy-cp{dimension}"),
        hypersurface,
        problem,
    )


def prepare_elliptic_curve(key: Array, /, *, line_count: int = 32) -> CalabiYauCampaign:
    return prepare_fermat_calabi_yau(2, key, line_count=line_count, width=16, depth=2)


def prepare_quartic_k3(key: Array, /, *, line_count: int = 64) -> CalabiYauCampaign:
    return prepare_fermat_calabi_yau(3, key, line_count=line_count, width=32, depth=2)


def prepare_fermat_quintic(key: Array, /, *, line_count: int = 128) -> CalabiYauCampaign:
    return prepare_fermat_calabi_yau(4, key, line_count=line_count, width=64, depth=3)


__all__ = [
    "CalabiYauCampaign",
    "cp1_calibration",
    "prepare_elliptic_curve",
    "prepare_fermat_calabi_yau",
    "prepare_fermat_quintic",
    "prepare_quartic_k3",
]
