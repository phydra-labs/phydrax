#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-k overlap bundles and evidence-bearing Wilson, Zak, and Chern invariants."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import PreparedReciprocalConnectivity
from ...operators.periodic._family import PeriodicResourceError
from ._orbital_model import PreparedPeriodicOrbitalPencil
from ._spectrum import PeriodicSpectrumResult


class PeriodicBandManifold(StrictModule, NonTrainableState):
    """A fixed isolated group of bands with raw direct-gap evidence."""

    spectrum: PeriodicSpectrumResult
    band_indices: Array
    minimum_direct_gap: Array
    direct_gaps: Array
    gap_tolerance: float = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)

    def __init__(
        self,
        spectrum: PeriodicSpectrumResult,
        band_indices: ArrayLike,
        /,
        *,
        gap_tolerance: float = 1.0e-8,
    ):
        if not isinstance(spectrum, PeriodicSpectrumResult):
            raise TypeError("spectrum must be PeriodicSpectrumResult.")
        indices = np.asarray(band_indices)
        tolerance = float(gap_tolerance)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= spectrum.energies.shape[1])
            or np.unique(indices).size != indices.size
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Band manifold indices or gap tolerance are invalid.")
        indices = np.sort(indices.astype(np.int32))
        energies = np.asarray(spectrum.energies)
        complement = np.setdiff1d(np.arange(energies.shape[1]), indices)
        if complement.size:
            gaps = np.min(
                np.abs(energies[:, indices, None] - energies[:, None, complement]),
                axis=(1, 2),
            )
        else:
            gaps = np.full((energies.shape[0],), np.inf, dtype=energies.dtype)
        minimum = float(np.min(gaps))
        resolved = bool(np.all(np.isfinite(energies)) and minimum > tolerance)
        if not resolved:
            raise ValueError(
                "Selected periodic band manifold is not isolated at every k point."
            )
        self.spectrum = spectrum
        self.band_indices = jnp.asarray(indices)
        self.minimum_direct_gap = jnp.asarray(minimum)
        self.direct_gaps = jnp.asarray(gaps)
        self.gap_tolerance = tolerance
        self.resolved = resolved
        self.manifold_id = canonical_fingerprint(
            {
                "kind": "periodic-band-manifold",
                "spectrum": spectrum.result_id,
                "band_indices": indices.tolist(),
                "gap_tolerance": tolerance,
                "direct_gaps": array_tree_fingerprint(gaps),
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.band_indices.size)


class PeriodicCrossKConnection(StrictModule, NonTrainableState):
    """Explicit basis overlaps for every directed reciprocal connectivity edge."""

    connectivity: PreparedReciprocalConnectivity
    matrices: Array
    singular_values: Array
    basis_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        connectivity: PreparedReciprocalConnectivity,
        matrices: ArrayLike,
        basis_id: str,
        source_id: str,
        /,
        *,
        reverse_tolerance: float = 1.0e-10,
        maximum_matrix_entries: int = 8_000_000,
    ):
        if not isinstance(connectivity, PreparedReciprocalConnectivity):
            raise TypeError("connectivity must be PreparedReciprocalConnectivity.")
        matrix = np.asarray(matrices)
        basis = str(basis_id).strip()
        source = str(source_id).strip()
        if (
            matrix.ndim != 3
            or matrix.shape[0] != connectivity.edge_count
            or matrix.shape[1] != matrix.shape[2]
            or matrix.shape[1] <= 0
            or np.any(~np.isfinite(matrix))
            or not basis
            or not source
        ):
            raise ValueError("Cross-k connection matrices, basis, or source are invalid.")
        if matrix.size > int(maximum_matrix_entries):
            raise PeriodicResourceError(
                "Cross-k connection exceeds maximum_matrix_entries."
            )
        reverse = np.asarray(connectivity.plan.reverse_indices)
        defect = np.max(
            np.abs(matrix[reverse] - np.conj(np.swapaxes(matrix, -1, -2))), initial=0.0
        )
        scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0)
        if defect > float(reverse_tolerance) * scale:
            raise ValueError("Cross-k connection reverse matrices are not adjoints.")
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        self.connectivity = connectivity
        self.matrices = jnp.asarray(matrix)
        self.singular_values = jnp.asarray(singular_values)
        self.basis_id = basis
        self.source_id = source
        self.connection_id = canonical_fingerprint(
            {
                "kind": "periodic-cross-k-connection",
                "connectivity": connectivity.prepared_id,
                "basis": basis,
                "source": source,
                "arrays": array_tree_fingerprint(
                    {"matrices": matrix, "singular_values": singular_values}
                ),
            }
        )


class PeriodicOverlapBundle(StrictModule, NonTrainableState):
    """Raw manifold overlaps plus separately normalized unitary link evidence."""

    manifold: PeriodicBandManifold
    connection: PeriodicCrossKConnection | None
    connectivity: PreparedReciprocalConnectivity
    raw_overlaps: Array
    normalized_links: Array
    singular_values: Array
    link_unitarity_residuals: Array
    minimum_singular_value: Array
    source_id: str = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifold: PeriodicBandManifold,
        connection_or_connectivity: PeriodicCrossKConnection
        | PreparedReciprocalConnectivity,
        /,
        *,
        raw_overlaps: ArrayLike | None = None,
        source_id: str | None = None,
        link_singular_value_floor: float = 1.0e-8,
    ):
        if not isinstance(manifold, PeriodicBandManifold):
            raise TypeError("Overlap bundle requires PeriodicBandManifold.")
        connection: PeriodicCrossKConnection | None
        if isinstance(connection_or_connectivity, PeriodicCrossKConnection):
            if raw_overlaps is not None or source_id is not None:
                raise ValueError(
                    "Basis-connection overlap construction does not accept raw link input."
                )
            connection = connection_or_connectivity
            connectivity = connection.connectivity
            plan = connectivity.plan
            if (
                manifold.spectrum.support_id != plan.mesh.mesh_id
                or manifold.spectrum.cell_id != plan.cell_id
                or connection.matrices.shape[1] != manifold.spectrum.coefficients.shape[1]
                or connection.basis_id != manifold.spectrum.basis_id
            ):
                raise ValueError("Manifold spectrum and cross-k connection do not align.")
            coefficients = np.asarray(manifold.spectrum.coefficients)[
                :, :, np.asarray(manifold.band_indices)
            ]
            source = np.asarray(plan.source_indices)
            target = np.asarray(plan.target_indices)
            raw = np.einsum(
                "eai,eab,ebj->eij",
                np.conj(coefficients[source]),
                np.asarray(connection.matrices),
                coefficients[target],
            )
            source_identifier = connection.connection_id
        elif isinstance(connection_or_connectivity, PreparedReciprocalConnectivity):
            connection = None
            connectivity = connection_or_connectivity
            plan = connectivity.plan
            source_identifier = "" if source_id is None else str(source_id).strip()
            if raw_overlaps is None or not source_identifier:
                raise ValueError(
                    "Raw overlap construction requires matrices and source identity."
                )
            raw = np.asarray(raw_overlaps)
            if (
                manifold.spectrum.support_id != plan.mesh.mesh_id
                or manifold.spectrum.cell_id != plan.cell_id
                or raw.shape
                != (connectivity.edge_count, manifold.dimension, manifold.dimension)
                or np.any(~np.isfinite(raw))
            ):
                raise ValueError(
                    "Raw manifold overlaps do not align with prepared connectivity."
                )
            reverse = np.asarray(plan.reverse_indices)
            defect = np.max(
                np.abs(raw[reverse] - np.conj(np.swapaxes(raw, -1, -2))),
                initial=0.0,
            )
            if defect > 1.0e-10 * max(float(np.max(np.abs(raw), initial=0.0)), 1.0):
                raise ValueError("Raw manifold reverse overlaps are not adjoints.")
        else:
            raise TypeError(
                "Overlap bundle requires a basis connection or prepared connectivity."
            )
        left, singular, right_h = np.linalg.svd(raw)
        links = left @ right_h
        identity = np.eye(manifold.dimension, dtype=links.dtype)
        residuals = np.max(
            np.abs(np.conj(np.swapaxes(links, -1, -2)) @ links - identity),
            axis=(-2, -1),
        )
        minimum = float(np.min(singular))
        floor = float(link_singular_value_floor)
        resolved = bool(
            manifold.resolved
            and isfinite(floor)
            and floor > 0.0
            and minimum > floor
            and np.all(np.isfinite(raw))
        )
        if not resolved:
            raise ValueError("Manifold link overlap is rank deficient or unresolved.")
        self.manifold = manifold
        self.connection = connection
        self.connectivity = connectivity
        self.raw_overlaps = jnp.asarray(raw)
        self.normalized_links = jnp.asarray(links)
        self.singular_values = jnp.asarray(singular)
        self.link_unitarity_residuals = jnp.asarray(residuals)
        self.minimum_singular_value = jnp.asarray(minimum)
        self.source_id = source_identifier
        self.resolved = resolved
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "periodic-overlap-bundle",
                "manifold": manifold.manifold_id,
                "connectivity": connectivity.prepared_id,
                "source": source_identifier,
                "link_singular_value_floor": floor,
                "arrays": array_tree_fingerprint(
                    {
                        "raw_overlaps": raw,
                        "normalized_links": links,
                        "singular_values": singular,
                    }
                ),
            }
        )


class PeriodicWilsonResult(StrictModule, NonTrainableState):
    ordered_edges: Array
    wilson_matrix: Array
    raw_link_determinant_phases: Array
    eigenphases: Array
    zak_phase: Array
    link_singular_values: Array
    link_unitarity_residuals: Array
    successful: Array
    bundle_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PeriodicWilsonPlan(StrictModule, NonTrainableState):
    bundle: PeriodicOverlapBundle
    ordered_edges: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bundle: PeriodicOverlapBundle,
        ordered_edges: ArrayLike,
        /,
        *,
        maximum_links: int = 1_000_000,
    ):
        if not isinstance(bundle, PeriodicOverlapBundle):
            raise TypeError("bundle must be PeriodicOverlapBundle.")
        edges = np.asarray(ordered_edges)
        connectivity = bundle.connectivity.plan
        if (
            edges.ndim != 1
            or edges.size == 0
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0)
            or np.any(edges >= connectivity.source_indices.size)
        ):
            raise ValueError("Wilson loop edges must be a nonempty integer sequence.")
        if edges.size > int(maximum_links):
            raise PeriodicResourceError("Wilson loop exceeds maximum_links.")
        source = np.asarray(connectivity.source_indices)[edges]
        target = np.asarray(connectivity.target_indices)[edges]
        if not np.array_equal(target, np.roll(source, -1)):
            raise ValueError(
                "Wilson edges must form one contiguous closed oriented loop."
            )
        self.bundle = bundle
        self.ordered_edges = jnp.asarray(edges, dtype=jnp.int32)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-wilson-plan",
                "bundle": bundle.bundle_id,
                "ordered_edges": edges.tolist(),
            }
        )

    def evaluate(self, /) -> PeriodicWilsonResult:
        edges = np.asarray(self.ordered_edges)
        links = np.asarray(self.bundle.normalized_links)[edges]
        loop = np.eye(self.bundle.manifold.dimension, dtype=links.dtype)
        for link in links:
            loop = loop @ link
        eigenphases = np.sort(np.angle(np.linalg.eigvals(loop)))
        raw_phases = np.angle(np.linalg.det(np.asarray(self.bundle.raw_overlaps)[edges]))
        zak = float(np.angle(np.linalg.det(loop)))
        successful = bool(np.all(np.isfinite(loop)) and self.bundle.resolved)
        return PeriodicWilsonResult(
            self.ordered_edges,
            jnp.asarray(loop),
            jnp.asarray(raw_phases),
            jnp.asarray(eigenphases),
            jnp.asarray(zak),
            self.bundle.singular_values[self.ordered_edges],
            self.bundle.link_unitarity_residuals[self.ordered_edges],
            jnp.asarray(successful),
            self.bundle.bundle_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-wilson-result",
                    "plan": self.plan_id,
                    "arrays": array_tree_fingerprint(
                        {
                            "loop": loop,
                            "eigenphases": eigenphases,
                            "raw_phases": raw_phases,
                        }
                    ),
                }
            ),
        )


class PeriodicChernRefinementEvidence(StrictModule, NonTrainableState):
    coarse_mesh_shape: tuple[int, ...] = eqx.field(static=True)
    fine_mesh_shape: tuple[int, ...] = eqx.field(static=True)
    coarse_chern: Array
    fine_chern: Array
    residual: Array
    tolerance: float = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_mesh_shape: tuple[int, ...],
        fine_mesh_shape: tuple[int, ...],
        coarse_chern: float,
        fine_chern: float,
        tolerance: float,
        /,
    ):
        coarse_shape = tuple(int(value) for value in coarse_mesh_shape)
        fine_shape = tuple(int(value) for value in fine_mesh_shape)
        coarse = float(coarse_chern)
        fine = float(fine_chern)
        tolerance_ = float(tolerance)
        if (
            len(coarse_shape) != len(fine_shape)
            or any(
                fine_value < coarse_value
                for coarse_value, fine_value in zip(coarse_shape, fine_shape, strict=True)
            )
            or any(value <= 0 for value in coarse_shape + fine_shape)
            or not all(isfinite(value) for value in (coarse, fine, tolerance_))
            or tolerance_ < 0.0
        ):
            raise ValueError("Chern refinement shapes, values, or tolerance are invalid.")
        residual = abs(fine - coarse)
        successful = residual <= tolerance_
        self.coarse_mesh_shape = coarse_shape
        self.fine_mesh_shape = fine_shape
        self.coarse_chern = jnp.asarray(coarse)
        self.fine_chern = jnp.asarray(fine)
        self.residual = jnp.asarray(residual)
        self.tolerance = tolerance_
        self.successful = successful
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-chern-refinement-evidence",
                "coarse_shape": list(coarse_shape),
                "fine_shape": list(fine_shape),
                "coarse_chern": coarse,
                "fine_chern": fine,
                "tolerance": tolerance_,
            }
        )


class PeriodicChernResult(StrictModule, NonTrainableState):
    plaquette_phases: Array
    raw_chern: Array
    nearest_integer: Array
    quantization_residual: Array
    minimum_link_singular_value: Array
    minimum_direct_gap: Array
    refinement_residual: Array
    successful: Array
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PeriodicChernPlan(StrictModule, NonTrainableState):
    bundle: PeriodicOverlapBundle
    refinement: PeriodicChernRefinementEvidence | None
    quantization_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bundle: PeriodicOverlapBundle,
        /,
        *,
        refinement: PeriodicChernRefinementEvidence | None = None,
        require_refinement: bool = False,
        quantization_tolerance: float = 1.0e-6,
    ):
        if not isinstance(bundle, PeriodicOverlapBundle):
            raise TypeError("bundle must be PeriodicOverlapBundle.")
        if bundle.connectivity.plaquette_count == 0:
            raise ValueError(
                "First Chern evaluation requires oriented plaquette connectivity."
            )
        if refinement is not None and not isinstance(
            refinement, PeriodicChernRefinementEvidence
        ):
            raise TypeError("refinement must be PeriodicChernRefinementEvidence or None.")
        if require_refinement and (refinement is None or not refinement.successful):
            raise ValueError(
                "Required Chern mesh-refinement evidence is missing or unresolved."
            )
        tolerance = float(quantization_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("quantization_tolerance must be finite and non-negative.")
        self.bundle = bundle
        self.refinement = refinement
        self.quantization_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-chern-plan",
                "bundle": bundle.bundle_id,
                "refinement": None if refinement is None else refinement.evidence_id,
                "quantization_tolerance": tolerance,
            }
        )

    def evaluate(self, /) -> PeriodicChernResult:
        connectivity = self.bundle.connectivity.plan
        edges = np.asarray(connectivity.plaquette_edges)
        orientations = np.asarray(connectivity.plaquette_orientations)
        links = np.asarray(self.bundle.normalized_links)
        phases = []
        for plaquette, signs in zip(edges, orientations, strict=True):
            loop = np.eye(self.bundle.manifold.dimension, dtype=links.dtype)
            for edge, sign in zip(plaquette, signs, strict=True):
                link = links[edge] if sign > 0 else np.conj(links[edge].T)
                loop = loop @ link
            phases.append(float(np.angle(np.linalg.det(loop))))
        phase_array = np.asarray(phases)
        raw_chern = float(np.sum(phase_array) / (2.0 * np.pi))
        nearest = int(np.rint(raw_chern))
        quantization = abs(raw_chern - nearest)
        refinement_residual = (
            np.nan if self.refinement is None else float(self.refinement.residual)
        )
        successful = bool(
            self.bundle.resolved
            and quantization <= self.quantization_tolerance
            and (self.refinement is None or self.refinement.successful)
        )
        mesh_shape = connectivity.mesh.mesh_shape
        return PeriodicChernResult(
            jnp.asarray(phase_array),
            jnp.asarray(raw_chern),
            jnp.asarray(nearest, dtype=jnp.int32),
            jnp.asarray(quantization),
            self.bundle.minimum_singular_value,
            self.bundle.manifold.minimum_direct_gap,
            jnp.asarray(refinement_residual),
            jnp.asarray(successful),
            mesh_shape,
            self.bundle.bundle_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-chern-result",
                    "plan": self.plan_id,
                    "arrays": array_tree_fingerprint(
                        {
                            "phases": phase_array,
                            "raw_chern": np.asarray(raw_chern),
                            "nearest": np.asarray(nearest),
                        }
                    ),
                }
            ),
        )


def identity_cross_k_connection(
    pencil: PreparedPeriodicOrbitalPencil,
    connectivity: PreparedReciprocalConnectivity,
    /,
) -> PeriodicCrossKConnection:
    """Construct the explicit connection only for a structurally orthonormal basis."""

    if not isinstance(pencil, PreparedPeriodicOrbitalPencil):
        raise TypeError("pencil must be PreparedPeriodicOrbitalPencil.")
    connectivity.plan.mesh.require_cell(pencil.plan.basis.cell)
    relation = pencil.overlap.plan.relation
    onsite_identity = (
        relation.capacity == pencil.plan.basis.orbital_count
        and np.all(np.asarray(pencil.overlap.plan.translations) == 0)
        and np.array_equal(
            np.asarray(relation.source_indices), np.arange(relation.capacity)
        )
        and np.array_equal(
            np.asarray(relation.target_indices), np.arange(relation.capacity)
        )
        and np.allclose(np.asarray(pencil.overlap.state.values), 1.0)
    )
    if not onsite_identity:
        raise ValueError(
            "Generalized S requires an externally supplied cross-k connection."
        )
    edge_count = connectivity.edge_count
    orbital_count = pencil.plan.basis.orbital_count
    matrices = np.broadcast_to(
        np.eye(orbital_count), (edge_count, orbital_count, orbital_count)
    ).astype(complex)
    if pencil.plan.basis.gauge.kind == "atomic":
        points = np.asarray(connectivity.plan.mesh.fractional_points)
        source = np.asarray(connectivity.plan.source_indices)
        target_points = points[np.asarray(connectivity.plan.target_indices)] + np.asarray(
            connectivity.plan.reciprocal_shifts
        )
        centers = np.asarray(pencil.plan.basis.centers_fractional)
        scale = (
            pencil.hamiltonian.plan.convention.sign
            * pencil.hamiltonian.plan.convention.phase_scale
        )
        source_phase = np.exp(1.0j * scale * (points[source] @ centers.T))
        target_phase = np.exp(1.0j * scale * (target_points @ centers.T))
        diagonal = np.conj(source_phase) * target_phase
        matrices = np.eye(orbital_count)[None, :, :] * diagonal[:, None, :]
    return PeriodicCrossKConnection(
        connectivity,
        matrices,
        pencil.plan.basis.basis_id,
        f"analytic-orthonormal:{pencil.prepared_id}",
    )


__all__ = [
    "PeriodicBandManifold",
    "PeriodicChernPlan",
    "PeriodicChernRefinementEvidence",
    "PeriodicChernResult",
    "PeriodicCrossKConnection",
    "PeriodicOverlapBundle",
    "PeriodicWilsonPlan",
    "PeriodicWilsonResult",
    "identity_cross_k_connection",
]
