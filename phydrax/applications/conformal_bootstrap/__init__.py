#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite scalar-block and crossing-cone reference calculations."""

from ._blocks import (
    prepare_scalar_blocks,
    PreparedScalarBlocks,
    ScalarBlockEvidence,
    ScalarBlockPlan,
)
from ._cone import (
    assemble_scalar_crossing_cone,
    compare_known_gap_bound,
    CrossingConePlan,
    CrossingConicEvidence,
    CrossingExclusionEvidence,
    exclude_scalar_gap,
    KnownBoundEvidence,
    prepare_crossing_cone,
    PreparedCrossingCone,
    solve_crossing_cone,
)
from ._crossing import (
    CrossingSectorEvidence,
    CrossingSectorPlan,
    prepare_crossing_sectors,
    PreparedCrossingSectors,
)
from ._data import (
    ConformalDataPlan,
    CrossingChannel,
    ExchangedOperatorSector,
    ExternalScalarOperator,
)
from ._global_blocks import (
    GlobalBlockEvidence,
    GlobalScalarBlockPlan,
    prepare_global_scalar_blocks,
    PreparedGlobalScalarBlocks,
)
from ._pmp import (
    audit_pmp_samples,
    ConformalPolynomialMatrixProgram,
    DampedRationalPrefactor,
    decimal_coefficients,
    PMPSampledAuditEvidence,
    PolynomialMatrixBlock,
)
from ._qualification import (
    conformal_bootstrap_candidate_profiles,
    conformal_bootstrap_candidate_support_tuples,
)
from ._sdpb import (
    execute_sdpb,
    parse_sdpb_output,
    parse_sdpb_vector,
    reconstruct_pmp_functional,
    SDPBExecutionResult,
    SDPBExecutionStatus,
    SDPBJobPlan,
    SDPBNumericalSummary,
    SDPBProvider,
)
from ._virasoro import (
    BPZVirasoroBlockPlan,
    elliptic_nome,
    ising_sigma_crossing_evidence,
    IsingSigmaChannel,
    IsingSigmaVirasoroPlan,
    prepare_bpz_virasoro_blocks,
    prepare_ising_sigma_virasoro_blocks,
    PreparedBPZVirasoroBlocks,
    PreparedIsingSigmaVirasoroBlocks,
    VirasoroBlockEvidence,
    VirasoroCrossingEvidence,
)


__all__ = [
    "BPZVirasoroBlockPlan",
    "ConformalPolynomialMatrixProgram",
    "ConformalDataPlan",
    "CrossingChannel",
    "CrossingConePlan",
    "CrossingConicEvidence",
    "CrossingExclusionEvidence",
    "CrossingSectorEvidence",
    "CrossingSectorPlan",
    "ExchangedOperatorSector",
    "DampedRationalPrefactor",
    "ExternalScalarOperator",
    "GlobalBlockEvidence",
    "GlobalScalarBlockPlan",
    "IsingSigmaChannel",
    "IsingSigmaVirasoroPlan",
    "KnownBoundEvidence",
    "PMPSampledAuditEvidence",
    "PolynomialMatrixBlock",
    "PreparedCrossingCone",
    "PreparedCrossingSectors",
    "PreparedGlobalScalarBlocks",
    "PreparedBPZVirasoroBlocks",
    "PreparedIsingSigmaVirasoroBlocks",
    "PreparedScalarBlocks",
    "ScalarBlockEvidence",
    "ScalarBlockPlan",
    "VirasoroBlockEvidence",
    "VirasoroCrossingEvidence",
    "assemble_scalar_crossing_cone",
    "SDPBExecutionResult",
    "SDPBExecutionStatus",
    "SDPBJobPlan",
    "SDPBNumericalSummary",
    "SDPBProvider",
    "conformal_bootstrap_candidate_profiles",
    "conformal_bootstrap_candidate_support_tuples",
    "compare_known_gap_bound",
    "audit_pmp_samples",
    "exclude_scalar_gap",
    "elliptic_nome",
    "decimal_coefficients",
    "execute_sdpb",
    "prepare_crossing_cone",
    "prepare_crossing_sectors",
    "prepare_global_scalar_blocks",
    "parse_sdpb_output",
    "parse_sdpb_vector",
    "ising_sigma_crossing_evidence",
    "prepare_bpz_virasoro_blocks",
    "prepare_ising_sigma_virasoro_blocks",
    "prepare_scalar_blocks",
    "reconstruct_pmp_functional",
    "solve_crossing_cone",
]
