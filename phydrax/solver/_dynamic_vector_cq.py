#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
from jaxtyping import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import PreparedLinearSolve
from ..operators.integral._convolution_quadrature import (
    ConvolutionQuadratureContourPolicy,
    ConvolutionQuadratureMethod,
)
from ._convolution_quadrature import (
    ConvolutionQuadratureAction,
    ConvolutionQuadratureDeclaration,
    ConvolutionQuadratureResult,
    prepare_convolution_quadrature,
    PreparedConvolutionQuadrature,
)


DynamicVectorFamily3D = Literal["elasticity", "maxwell"]
NodeFamilyPreparation3D = Callable[
    [object, ConvolutionQuadratureAction], PreparedLinearSolve
]


class PreparedDynamicVectorFEMBEM3D(StrictModule, NonTrainableState):
    """Fixed-history vector FEM-BEM convolution-quadrature product."""

    __strict_abstract__ = True

    cq: PreparedConvolutionQuadrature
    family: DynamicVectorFamily3D = eqx.field(static=True)
    node_family_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def apply(self, history: ArrayLike, /) -> ConvolutionQuadratureResult:
        return self.cq.apply(history)

    def transpose_apply(self, history: ArrayLike, /) -> ConvolutionQuadratureResult:
        return self.cq.transpose_apply(history)

    def adjoint_apply(self, history: ArrayLike, /) -> ConvolutionQuadratureResult:
        return self.cq.adjoint_apply(history)


class PreparedDynamicElasticityFEMBEM3D(PreparedDynamicVectorFEMBEM3D):
    """Prepared fixed-history isotropic elastodynamic FEM-BEM product."""


class PreparedDynamicMaxwellFEMBEM3D(PreparedDynamicVectorFEMBEM3D):
    """Prepared fixed-history isotropic Maxwell FEM-BEM product."""


def _prepare_dynamic_vector(
    family: DynamicVectorFamily3D,
    prepare_node: NodeFamilyPreparation3D,
    dimension: int,
    step_size: ArrayLike,
    history_length: int,
    /,
    *,
    node_family_id: str,
    method: ConvolutionQuadratureMethod,
    fft_length: int | None,
    contour_policy: ConvolutionQuadratureContourPolicy | None,
    conjugate_symmetric: bool,
    precision: str,
) -> PreparedDynamicVectorFEMBEM3D:
    family_id = str(node_family_id)
    if not family_id:
        raise ValueError("node_family_id must bind the exact FEM-BEM resolvent family.")
    pde = (
        "linear-isotropic-elastodynamics"
        if family == "elasticity"
        else "linear-isotropic-Maxwell"
    )
    formulation = (
        "BDF convolution quadrature of K+sC+s^2M with Navier Calderon coupling"
        if family == "elasticity"
        else "BDF convolution quadrature of curl-curl material resolvents with RWG/BC Calderon coupling"
    )
    declaration = ConvolutionQuadratureDeclaration(
        int(dimension),
        family_id=family_id,
        pde=pde,
        geometry="fixed matching or qualified nonmatching three-dimensional FEM-BEM interface",
        formulation=formulation,
        provider="phydrax-native prepared complex node solves",
        precision=str(precision),
        non_goals=(
            "continuum certification",
            "moving interfaces",
            "nonlinear bulk materials",
            "contact events or topology changes",
        ),
    )
    cq = prepare_convolution_quadrature(
        prepare_node,
        step_size,
        history_length,
        declaration,
        method=method,
        fft_length=fft_length,
        contour_policy=contour_policy,
        conjugate_symmetric=conjugate_symmetric,
    )
    prepared_type = (
        PreparedDynamicElasticityFEMBEM3D
        if family == "elasticity"
        else PreparedDynamicMaxwellFEMBEM3D
    )
    return prepared_type(
        cq=cq,
        family=family,
        node_family_id=family_id,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-dynamic-vector-fem-bem-3d",
                "family": family,
                "node_family": family_id,
                "contour": array_tree_fingerprint(cq.contour.parameters),
                "declaration": declaration.family_id,
            }
        ),
    )


def prepare_dynamic_elasticity_fem_bem_cq_3d(
    prepare_node: NodeFamilyPreparation3D,
    dimension: int,
    step_size: ArrayLike,
    history_length: int,
    /,
    *,
    node_family_id: str,
    method: ConvolutionQuadratureMethod = "bdf2",
    fft_length: int | None = None,
    contour_policy: ConvolutionQuadratureContourPolicy | None = None,
    conjugate_symmetric: bool = False,
    precision: str = "complex128",
) -> PreparedDynamicElasticityFEMBEM3D:
    """Prepare bounded elastodynamic FEM-BEM CQ from exact complex node products."""
    return _prepare_dynamic_vector(
        "elasticity",
        prepare_node,
        dimension,
        step_size,
        history_length,
        node_family_id=node_family_id,
        method=method,
        fft_length=fft_length,
        contour_policy=contour_policy,
        conjugate_symmetric=conjugate_symmetric,
        precision=precision,
    )


def prepare_dynamic_maxwell_fem_bem_cq_3d(
    prepare_node: NodeFamilyPreparation3D,
    dimension: int,
    step_size: ArrayLike,
    history_length: int,
    /,
    *,
    node_family_id: str,
    method: ConvolutionQuadratureMethod = "bdf2",
    fft_length: int | None = None,
    contour_policy: ConvolutionQuadratureContourPolicy | None = None,
    conjugate_symmetric: bool = False,
    precision: str = "complex128",
) -> PreparedDynamicMaxwellFEMBEM3D:
    """Prepare bounded Maxwell FEM-BEM CQ from exact complex node products."""
    return _prepare_dynamic_vector(
        "maxwell",
        prepare_node,
        dimension,
        step_size,
        history_length,
        node_family_id=node_family_id,
        method=method,
        fft_length=fft_length,
        contour_policy=contour_policy,
        conjugate_symmetric=conjugate_symmetric,
        precision=precision,
    )


__all__ = [
    "PreparedDynamicElasticityFEMBEM3D",
    "PreparedDynamicMaxwellFEMBEM3D",
    "PreparedDynamicVectorFEMBEM3D",
    "prepare_dynamic_elasticity_fem_bem_cq_3d",
    "prepare_dynamic_maxwell_fem_bem_cq_3d",
]
