#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import pi
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import PeriodicCell
from ....discretization.bem import RWGSurfaceCurrentSpace3D
from ....linalg import DenseLinearOperator, OperatorProperties
from ._maxwell3d import MaxwellEFIEPolicy3D, prepare_maxwell_efie_3d


PeriodicMaxwellFormulation3D = Literal["efie", "mfie", "cfie"]


class PeriodicMaxwellBoundaryPolicy3D(StrictModule, NonTrainableState):
    image_cutoff: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_resident_bytes: int = eqx.field(static=True)
    cfie_electric_weight: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_cutoff: int = 2,
        maximum_images: int = 4096,
        maximum_edges: int = 2048,
        maximum_resident_bytes: int = 1_000_000_000,
        cfie_electric_weight: float = 0.5,
    ):
        cutoff = int(image_cutoff)
        limits = (int(maximum_images), int(maximum_edges), int(maximum_resident_bytes))
        weight = float(cfie_electric_weight)
        if cutoff < 1 or any(value <= 0 for value in limits):
            raise ValueError("Periodic Maxwell image/resource bounds must be positive.")
        if not 0.0 < weight < 1.0:
            raise ValueError(
                "cfie_electric_weight must lie strictly between zero and one."
            )
        self.image_cutoff = cutoff
        self.maximum_images, self.maximum_edges, self.maximum_resident_bytes = limits
        self.cfie_electric_weight = weight
        self.policy_id = canonical_fingerprint(
            {
                "kind": "periodic-maxwell-boundary-policy-3d",
                "cutoff": cutoff,
                "limits": limits,
                "cfie_weight": weight,
            }
        )


class PeriodicMaxwellBoundaryEvidence3D(StrictModule, NonTrainableState):
    image_count: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    central_free_space_singular_action: bool = eqx.field(static=True)
    smooth_noncentral_images: bool = eqx.field(static=True)
    infinite_lattice_certified: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedPeriodicMaxwellBoundary3D(StrictModule, NonTrainableState):
    current_space: RWGSurfaceCurrentSpace3D
    cell: PeriodicCell
    operator: DenseLinearOperator
    bloch_wavevector: Array
    wavenumber: Array
    evidence: PeriodicMaxwellBoundaryEvidence3D
    formulation: PeriodicMaxwellFormulation3D = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def mv(self, current: ArrayLike, /) -> Array:
        return self.operator.mv(self.current_space.validate(current))

    def transpose_mv(self, current: ArrayLike, /) -> Array:
        return self.operator.transpose_mv(self.current_space.validate(current))

    def adjoint_mv(self, current: ArrayLike, /) -> Array:
        return self.operator.adjoint_mv(self.current_space.validate(current))


def _outgoing_dyadic(displacement: np.ndarray, wavenumber: float, /) -> np.ndarray:
    radius = float(np.linalg.norm(displacement))
    if radius <= 0.0:
        raise ValueError("Noncentral Maxwell images require positive separation.")
    unit = displacement / radius
    kr = wavenumber * radius
    green = np.exp(1j * kr) / (4.0 * pi * radius)
    identity = np.eye(3)
    outer = unit[:, None] * unit[None, :]
    return green * (
        (1.0 + (1j * kr - 1.0) / (kr * kr)) * identity
        + ((3.0 - 3j * kr - kr * kr) / (kr * kr)) * outer
    )


def _smooth_image_matrix(space, translations, phases, wavenumber, /):
    surface = space.surface
    centroids = np.asarray(surface.face_centroids, dtype=float)
    areas = np.asarray(surface.face_areas, dtype=float)
    basis = np.asarray(space.centroid_basis)
    face_edges = np.asarray(surface.face_edges, dtype=np.int32)
    edge_count = space.size
    matrix = np.zeros((edge_count, edge_count), dtype=np.complex128)
    magnetic = np.zeros_like(matrix)
    for translation, phase in zip(translations, phases, strict=True):
        for target in range(surface.face_count):
            for source in range(surface.face_count):
                displacement = centroids[target] - centroids[source] - translation
                dyadic = _outgoing_dyadic(displacement, wavenumber)
                radius = float(np.linalg.norm(displacement))
                unit = displacement / radius
                gradient = (
                    np.exp(1j * wavenumber * radius)
                    * (1j * wavenumber * radius - 1.0)
                    * unit
                    / (4.0 * pi * radius**2)
                )
                scale = phase * areas[target] * areas[source]
                for local_target, edge_target in enumerate(face_edges[target]):
                    test = basis[target, local_target]
                    for local_source, edge_source in enumerate(face_edges[source]):
                        trial = basis[source, local_source]
                        matrix[edge_target, edge_source] += scale * (
                            test @ dyadic @ trial
                        )
                        magnetic[edge_target, edge_source] += scale * np.dot(
                            test, np.cross(gradient, trial)
                        )
    return matrix, magnetic


def prepare_periodic_maxwell_boundary_3d(
    current_space: RWGSurfaceCurrentSpace3D,
    cell: PeriodicCell,
    /,
    *,
    wavenumber: float,
    bloch_wavevector: ArrayLike | None = None,
    formulation: PeriodicMaxwellFormulation3D = "efie",
    wave_impedance: float = 1.0,
    policy: PeriodicMaxwellBoundaryPolicy3D | None = None,
    free_space_policy: MaxwellEFIEPolicy3D | None = None,
) -> PreparedPeriodicMaxwellBoundary3D:
    """Prepare central singular free-space plus bounded smooth periodic images."""
    if not isinstance(current_space, RWGSurfaceCurrentSpace3D):
        raise TypeError("current_space must be RWGSurfaceCurrentSpace3D.")
    if (
        not isinstance(cell, PeriodicCell)
        or cell.rank != 3
        or cell.ambient_dimension != 3
        or not cell.fully_periodic
    ):
        raise ValueError("Periodic Maxwell requires a fully periodic rank-3 cell in R3.")
    if formulation not in ("efie", "mfie", "cfie"):
        raise ValueError("formulation must be efie, mfie, or cfie.")
    selected = PeriodicMaxwellBoundaryPolicy3D() if policy is None else policy
    if current_space.size > selected.maximum_edges:
        raise ValueError("Periodic Maxwell edge capacity exceeded.")
    k = float(wavenumber)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("wavenumber must be finite and positive.")
    bloch = (
        np.zeros(3)
        if bloch_wavevector is None
        else np.asarray(bloch_wavevector, dtype=float)
    )
    if bloch.shape != (3,) or np.any(~np.isfinite(bloch)):
        raise ValueError("bloch_wavevector must be finite with shape (3,).")
    indices = np.asarray(
        tuple(
            product(range(-selected.image_cutoff, selected.image_cutoff + 1), repeat=3)
        ),
        dtype=np.int32,
    )
    indices = indices[np.any(indices != 0, axis=1)]
    if indices.shape[0] > selected.maximum_images:
        raise ValueError("Periodic Maxwell image capacity exceeded.")
    translations = indices @ np.asarray(cell.vectors, dtype=float)
    phases = np.exp(1j * (translations @ bloch))
    free = prepare_maxwell_efie_3d(
        current_space, k, wave_impedance=wave_impedance, policy=free_space_policy
    )
    electric_correction, magnetic_correction = _smooth_image_matrix(
        current_space, translations, phases, k
    )
    electric = np.asarray(free.operator.matrix) + electric_correction
    mass = np.zeros_like(electric)
    surface = current_space.surface
    basis = np.asarray(current_space.centroid_basis)
    for face in range(surface.face_count):
        edges = np.asarray(surface.face_edges[face])
        mass[np.ix_(edges, edges)] += float(surface.face_areas[face]) * (
            basis[face] @ basis[face].T
        )
    magnetic = 0.5 * mass + magnetic_correction
    matrix = electric if formulation == "efie" else magnetic
    if formulation == "cfie":
        alpha = selected.cfie_electric_weight
        matrix = alpha * electric + (1.0 - alpha) * float(wave_impedance) * magnetic
    resident = int(
        matrix.nbytes + electric_correction.nbytes + magnetic_correction.nbytes
    )
    if resident > selected.maximum_resident_bytes:
        raise ValueError("Periodic Maxwell resident-byte capacity exceeded.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "periodic-maxwell-boundary-evidence-3d",
            "cell": cell.cell_id,
            "indices": array_tree_fingerprint(indices),
            "formulation": formulation,
            "resident": resident,
            "infinite_lattice_certified": False,
        }
    )
    evidence = PeriodicMaxwellBoundaryEvidence3D(
        image_count=int(indices.shape[0]),
        resident_bytes=resident,
        central_free_space_singular_action=True,
        smooth_noncentral_images=True,
        infinite_lattice_certified=False,
        evidence_id=evidence_id,
    )
    operator = DenseLinearOperator(
        jnp.asarray(matrix),
        properties=OperatorProperties(evidence={}),
        operator_id=canonical_fingerprint(
            {
                "kind": "periodic-maxwell-boundary-operator-3d",
                "evidence": evidence_id,
                "matrix": array_tree_fingerprint(matrix),
            }
        ),
    )
    return PreparedPeriodicMaxwellBoundary3D(
        current_space=current_space,
        cell=cell,
        operator=operator,
        bloch_wavevector=jnp.asarray(bloch),
        wavenumber=jnp.asarray(k),
        evidence=evidence,
        formulation=formulation,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-periodic-maxwell-boundary-3d",
                "operator": operator.operator_id,
                "policy": selected.policy_id,
            }
        ),
    )


__all__ = [
    "PeriodicMaxwellBoundaryEvidence3D",
    "PeriodicMaxwellBoundaryPolicy3D",
    "PreparedPeriodicMaxwellBoundary3D",
    "prepare_periodic_maxwell_boundary_3d",
]
