#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, DenseLinearOperator, DualSpace


if TYPE_CHECKING:
    from ._de_rham import FiniteElementDeRhamComplex


class MortarInterfaceEvidence3D(StrictModule, NonTrainableState):
    coverage_fraction: float = eqx.field(static=True)
    orientation_margin: float = eqx.field(static=True)
    geometric_residual: float = eqx.field(static=True)
    inf_sup_margin: float = eqx.field(static=True)
    commuting_defect: float = eqx.field(static=True)
    exact_load_transpose: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedScalarMortarInterfaceTrace3D(StrictModule, NonTrainableState):
    trace: DenseLinearOperator
    load: DenseLinearOperator
    weak_conormal: DenseLinearOperator
    evidence: MortarInterfaceEvidence3D
    fem_trace_size: int = eqx.field(static=True)
    bem_trace_size: int = eqx.field(static=True)
    overlay_id: str = eqx.field(static=True)


class PreparedMaxwellMortarInterfaceTrace3D(StrictModule, NonTrainableState):
    tangential_trace: DenseLinearOperator
    boundary_load: DenseLinearOperator
    magnetic_conormal: DenseLinearOperator
    evidence: MortarInterfaceEvidence3D
    nedelec_trace_size: int = eqx.field(static=True)
    rwg_dual_size: int = eqx.field(static=True)
    overlay_id: str = eqx.field(static=True)
    volume_complex: FiniteElementDeRhamComplex


def _evidence(
    matrix: np.ndarray,
    *,
    coverage_fraction: float,
    orientation_margin: float,
    geometric_residual: float,
    commuting_defect: float,
    minimum_inf_sup: float,
    kind: str,
) -> MortarInterfaceEvidence3D:
    coverage = float(coverage_fraction)
    orientation = float(orientation_margin)
    residual = float(geometric_residual)
    commuting = float(commuting_defect)
    if not 0.0 < coverage <= 1.0 or coverage < 1.0 - 1e-12:
        raise ValueError("Mortar common refinement must cover the complete interface.")
    if (
        orientation <= 0.0
        or residual < 0.0
        or commuting < 0.0
        or not np.all(np.isfinite((orientation, residual, commuting)))
    ):
        raise ValueError("Mortar geometry/orientation evidence is invalid.")
    singular = np.linalg.svd(matrix, compute_uv=False)
    margin = float(singular[-1])
    if margin < float(minimum_inf_sup):
        raise ValueError("Mortar cross mass violates the declared inf-sup margin.")
    evidence_id = canonical_fingerprint(
        {
            "kind": kind,
            "matrix": array_tree_fingerprint(matrix),
            "coverage": coverage,
            "orientation": orientation,
            "residual": residual,
            "inf_sup": margin,
            "commuting": commuting,
        }
    )
    return MortarInterfaceEvidence3D(
        coverage, orientation, residual, margin, commuting, True, evidence_id
    )


def prepare_scalar_mortar_interface_trace_3d(
    cross_mass: ArrayLike,
    weak_conormal: ArrayLike,
    /,
    *,
    coverage_fraction: float,
    orientation_margin: float,
    geometric_residual: float,
    minimum_inf_sup: float = 1e-10,
) -> PreparedScalarMortarInterfaceTrace3D:
    matrix = np.asarray(cross_mass)
    conormal = np.asarray(weak_conormal)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or conormal.shape != matrix.shape
    ):
        raise ValueError(
            "Scalar mortar matrices must be aligned nonempty rank-two arrays."
        )
    evidence = _evidence(
        matrix,
        coverage_fraction=coverage_fraction,
        orientation_margin=orientation_margin,
        geometric_residual=geometric_residual,
        commuting_defect=0.0,
        minimum_inf_sup=minimum_inf_sup,
        kind="scalar-mortar-interface-evidence-3d",
    )
    trace = DenseLinearOperator(
        jnp.asarray(matrix), operator_id=f"{evidence.evidence_id}:trace"
    )
    load = DenseLinearOperator(
        jnp.asarray(matrix.T.conj()), operator_id=f"{evidence.evidence_id}:load"
    )
    conormal_op = DenseLinearOperator(
        jnp.asarray(conormal), operator_id=f"{evidence.evidence_id}:conormal"
    )
    return PreparedScalarMortarInterfaceTrace3D(
        trace,
        load,
        conormal_op,
        evidence,
        matrix.shape[1],
        matrix.shape[0],
        evidence.evidence_id,
    )


def prepare_maxwell_mortar_interface_trace_3d(
    cross_mass: ArrayLike,
    magnetic_conormal: ArrayLike,
    /,
    *,
    coverage_fraction: float,
    volume_complex: FiniteElementDeRhamComplex,
    boundary_space: ArraySpace,
    orientation_margin: float,
    geometric_residual: float,
    commuting_defect: float,
    maximum_commuting_defect: float = 1e-8,
    minimum_inf_sup: float = 1e-10,
) -> PreparedMaxwellMortarInterfaceTrace3D:
    from ._de_rham import FiniteElementDeRhamComplex

    if not isinstance(volume_complex, FiniteElementDeRhamComplex):
        raise TypeError("volume_complex must be FiniteElementDeRhamComplex.")
    if volume_complex.dimension != 3:
        raise ValueError("Maxwell mortar requires a three-dimensional FE complex.")
    matrix = np.asarray(cross_mass)
    conormal = np.asarray(magnetic_conormal)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or conormal.shape != matrix.shape
    ):
        raise ValueError(
            "Maxwell mortar matrices must be aligned nonempty rank-two arrays."
        )
    if matrix.shape[1] != volume_complex.hilbert_complex().space(1).size:
        raise ValueError("Maxwell mortar columns must span the complete FE edge space.")
    if not isinstance(boundary_space, ArraySpace):
        raise TypeError(
            "boundary_space must be the declared boundary coefficient ArraySpace."
        )
    if boundary_space.size != matrix.shape[0]:
        raise ValueError(
            "Maxwell mortar rows must span the declared boundary coefficient space."
        )
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(conormal)):
        raise ValueError("Maxwell mortar trace and conormal must be finite.")
    if float(commuting_defect) > float(maximum_commuting_defect):
        raise ValueError("Maxwell mortar commuting defect exceeds its envelope.")
    evidence = _evidence(
        matrix,
        coverage_fraction=coverage_fraction,
        orientation_margin=orientation_margin,
        geometric_residual=geometric_residual,
        commuting_defect=commuting_defect,
        minimum_inf_sup=minimum_inf_sup,
        kind="maxwell-mortar-interface-evidence-3d",
    )
    volume_space = volume_complex.hilbert_complex().space(1)
    if not isinstance(volume_space, ArraySpace):
        raise TypeError("Maxwell mortar requires an array-valued FE edge space.")
    dtype = volume_space.dtype
    if boundary_space.dtype != dtype:
        raise TypeError("FE and boundary coefficient spaces must share one dtype.")
    binding = canonical_fingerprint(
        {
            "kind": "maxwell-mortar-binding-3d",
            "evidence": evidence.evidence_id,
            "complex": volume_complex.realization_id,
            "conormal": array_tree_fingerprint(conormal),
            "boundary_space": boundary_space.space_id,
        }
    )
    trace = DenseLinearOperator(
        jnp.asarray(matrix, dtype=dtype),
        source=volume_space,
        target=DualSpace(boundary_space),
        operator_id=f"{binding}:trace",
    )
    load = DenseLinearOperator(
        jnp.asarray(matrix.T.conj(), dtype=dtype),
        source=boundary_space,
        target=DualSpace(volume_space),
        operator_id=f"{binding}:load",
    )
    conormal_op = DenseLinearOperator(
        jnp.asarray(conormal, dtype=dtype),
        source=volume_space,
        target=DualSpace(boundary_space),
        operator_id=f"{binding}:conormal",
    )
    return PreparedMaxwellMortarInterfaceTrace3D(
        trace,
        load,
        conormal_op,
        evidence,
        matrix.shape[1],
        matrix.shape[0],
        binding,
        volume_complex,
    )


__all__ = [
    "MortarInterfaceEvidence3D",
    "PreparedMaxwellMortarInterfaceTrace3D",
    "PreparedScalarMortarInterfaceTrace3D",
    "prepare_maxwell_mortar_interface_trace_3d",
    "prepare_scalar_mortar_interface_trace_3d",
]
