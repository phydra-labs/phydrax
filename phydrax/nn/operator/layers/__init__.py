"""Operator-aware attention, spectral, graph, and geometry layers."""

from ._attention import (
    AxialOperatorAttention,
    CodomainAttention,
    OperatorAttention,
    SliceAttention,
)
from ._basis_transfer import InvariantBasisTransferPlan, InvariantBasisTransferReport
from ._clifford import (
    audit_clifford_equivariance,
    clifford_gated_activation,
    CliffordEquivarianceAuditReport,
    CliffordEquivarianceCertificate,
    CliffordGeometricProductLayer,
    CliffordGradeLinear,
)
from ._graph_transfer import (
    GeometryMomentEmbedding,
    GraphAttentionTransfer,
    GraphKernelTransfer,
    MultiscaleGraphTransfer,
)
from ._lattice_equivariant import (
    InvariantFilterBasis,
    LatticeEquivariantConvND,
    TensorNormActivation,
    TensorPointwiseLinear,
    TensorRMSNorm,
)
from ._o3 import (
    EquivariantIntegralLayer,
    o3_gated_activation,
    O3PointwiseLinear,
    RadialBasis,
    RadialMap,
)
from ._o3_tensor_product import O3TensorProduct, O3TensorProductPlan
from ._regional_processor import RegionalGraphProcessor
from ._spectral import BasisSpectralConvND, BasisTransformPlan, ModalTransformKind
from ._transformer import OperatorTransformerProcessor


__all__ = [
    "audit_clifford_equivariance",
    "AxialOperatorAttention",
    "BasisSpectralConvND",
    "BasisTransformPlan",
    "clifford_gated_activation",
    "CliffordEquivarianceAuditReport",
    "CliffordEquivarianceCertificate",
    "CliffordGeometricProductLayer",
    "CliffordGradeLinear",
    "EquivariantIntegralLayer",
    "InvariantBasisTransferPlan",
    "InvariantBasisTransferReport",
    "InvariantFilterBasis",
    "LatticeEquivariantConvND",
    "CodomainAttention",
    "OperatorAttention",
    "OperatorTransformerProcessor",
    "SliceAttention",
    "ModalTransformKind",
    "GeometryMomentEmbedding",
    "O3PointwiseLinear",
    "O3TensorProduct",
    "O3TensorProductPlan",
    "o3_gated_activation",
    "RadialBasis",
    "TensorNormActivation",
    "TensorPointwiseLinear",
    "TensorRMSNorm",
    "RadialMap",
    "GraphAttentionTransfer",
    "GraphKernelTransfer",
    "MultiscaleGraphTransfer",
    "RegionalGraphProcessor",
]
