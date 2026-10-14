"""Native equivariant atomistic models, preparations, and numerical substrates."""

from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "MACEArchitecture": "._mace",
    "MACEDistanceTransform": "._mace",
    "MACELayerUpdate": "._mace",
    "MACEPotential": "._mace",
    "MACECutoffPlacement": "._mace_interaction",
    "MACEInteractionKind": "._mace_interaction",
    "MACESourceRows": "._mace_interaction",
    "MACECouplingDerivativeError": "._mace_kernel_derivatives",
    "fragment_extent": "._mace_kernels",
    "MACEAcceleratedCoupling": "._mace_kernels",
    "MACECouplingRow": "._mace_kernels",
    "MACEEdgeCouplingSpec": "._mace_kernels",
    "MACEFragmentExtent": "._mace_kernels",
    "MACEFragmentResources": "._mace_kernels",
    "MACEKernelEvidence": "._mace_kernels",
    "MACEKernelPlan": "._mace_kernels",
    "MACELayoutBlock": "._mace_kernels",
    "require_fragment_routing": "._mace_kernels",
    "MACERadialRealization": "._mace_prepare",
    "prepare_mace_potential": "._mace_prepare",
    "PreparedMACEPotential": "._mace_prepare",
    "StaleMACEPreparation": "._mace_prepare",
    "MACEEnergyScaling": "._mace_readout",
    "mace_potential_from_source": "._mace_source",
    "NequIPPotential": "._nequip",
    "PaiNNPotential": "._painn",
    "AgnesiTransform": "._radial",
    "BesselRadialBasis": "._radial",
    "normalized_silu_scale": "._radial",
    "PolynomialCutoff": "._radial",
    "RadialEmbedding": "._radial",
    "RadialMLP": "._radial",
    "RadialPostprocess": "._radial",
    "prepare_radial_tables": "._radial_projection",
    "PreparedRadialTables": "._radial_projection",
    "qualify_radial_tables": "._radial_projection",
    "radial_source_revision": "._radial_projection",
    "radial_table_resources": "._radial_projection",
    "RadialSpeciesBinding": "._radial_projection",
    "RadialTableDeclaration": "._radial_projection",
    "RadialTableLayout": "._radial_projection",
    "RadialTableLayoutChoice": "._radial_projection",
    "RadialTableObjective": "._radial_projection",
    "RadialTableQualification": "._radial_projection",
    "RadialTableQualificationError": "._radial_projection",
    "RadialTableQualificationPolicy": "._radial_projection",
    "RadialTableResources": "._radial_projection",
    "select_radial_table_layout": "._radial_projection",
    "StaleRadialTableBinding": "._radial_projection",
    "MergedSymmetricContraction": "._symmetric_contraction",
    "StaleSymmetricContractionBinding": "._symmetric_contraction",
    "symmetric_contraction_revision": "._symmetric_contraction",
    "SymmetricBasisOrigin": "._symmetric_contraction",
    "SymmetricContraction": "._symmetric_contraction",
    "SymmetricContractionBasis": "._symmetric_contraction",
    "SymmetricContractionPlan": "._symmetric_contraction",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._mace import (
        MACEArchitecture,
        MACEDistanceTransform,
        MACELayerUpdate,
        MACEPotential,
    )
    from ._mace_interaction import (
        MACECutoffPlacement,
        MACEInteractionKind,
        MACESourceRows,
    )
    from ._mace_kernel_derivatives import MACECouplingDerivativeError
    from ._mace_kernels import (
        fragment_extent,
        MACEAcceleratedCoupling,
        MACECouplingRow,
        MACEEdgeCouplingSpec,
        MACEFragmentExtent,
        MACEFragmentResources,
        MACEKernelEvidence,
        MACEKernelPlan,
        MACELayoutBlock,
        require_fragment_routing,
    )
    from ._mace_prepare import (
        MACERadialRealization,
        prepare_mace_potential,
        PreparedMACEPotential,
        StaleMACEPreparation,
    )
    from ._mace_readout import MACEEnergyScaling
    from ._mace_source import mace_potential_from_source
    from ._nequip import NequIPPotential
    from ._painn import PaiNNPotential
    from ._radial import (
        AgnesiTransform,
        BesselRadialBasis,
        normalized_silu_scale,
        PolynomialCutoff,
        RadialEmbedding,
        RadialMLP,
        RadialPostprocess,
    )
    from ._radial_projection import (
        prepare_radial_tables,
        PreparedRadialTables,
        qualify_radial_tables,
        radial_source_revision,
        radial_table_resources,
        RadialSpeciesBinding,
        RadialTableDeclaration,
        RadialTableLayout,
        RadialTableLayoutChoice,
        RadialTableObjective,
        RadialTableQualification,
        RadialTableQualificationError,
        RadialTableQualificationPolicy,
        RadialTableResources,
        select_radial_table_layout,
        StaleRadialTableBinding,
    )
    from ._symmetric_contraction import (
        MergedSymmetricContraction,
        StaleSymmetricContractionBinding,
        symmetric_contraction_revision,
        SymmetricBasisOrigin,
        SymmetricContraction,
        SymmetricContractionBasis,
        SymmetricContractionPlan,
    )


__all__ = [
    "AgnesiTransform",
    "BesselRadialBasis",
    "MACEAcceleratedCoupling",
    "MACEArchitecture",
    "MACECouplingDerivativeError",
    "MACECouplingRow",
    "MACECutoffPlacement",
    "MACEDistanceTransform",
    "MACEEdgeCouplingSpec",
    "MACEEnergyScaling",
    "MACEFragmentExtent",
    "MACEFragmentResources",
    "MACEInteractionKind",
    "MACEKernelEvidence",
    "MACEKernelPlan",
    "MACELayerUpdate",
    "MACELayoutBlock",
    "MACEPotential",
    "MACERadialRealization",
    "MACESourceRows",
    "MergedSymmetricContraction",
    "NequIPPotential",
    "PaiNNPotential",
    "PolynomialCutoff",
    "PreparedMACEPotential",
    "PreparedRadialTables",
    "RadialEmbedding",
    "RadialMLP",
    "RadialPostprocess",
    "RadialSpeciesBinding",
    "RadialTableDeclaration",
    "RadialTableLayout",
    "RadialTableLayoutChoice",
    "RadialTableObjective",
    "RadialTableQualification",
    "RadialTableQualificationError",
    "RadialTableQualificationPolicy",
    "RadialTableResources",
    "StaleMACEPreparation",
    "StaleRadialTableBinding",
    "StaleSymmetricContractionBinding",
    "SymmetricBasisOrigin",
    "SymmetricContraction",
    "SymmetricContractionBasis",
    "SymmetricContractionPlan",
    "fragment_extent",
    "mace_potential_from_source",
    "normalized_silu_scale",
    "prepare_mace_potential",
    "prepare_radial_tables",
    "qualify_radial_tables",
    "radial_source_revision",
    "radial_table_resources",
    "require_fragment_routing",
    "select_radial_table_layout",
    "symmetric_contraction_revision",
]
