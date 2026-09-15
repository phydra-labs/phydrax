#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native general-contraction Gaussian atomic-orbital kernels."""

from ._ao import (
    ao_gradients,
    ao_hessians,
    ao_values,
    AOEvaluation,
    cartesian_ao_values,
    evaluate_ao,
)
from ._basis import GaussianBasisPlan, PreparedGaussianBasis
from ._boys import boys0, boys_values
from ._derivatives import GaussianIntegralDerivativePlan, GaussianIntegralDerivativeResult
from ._direct import DirectJKPlan, DirectJKResult, PreparedDirectJK
from ._ecp import (
    AbstractECPIntegralProvider,
    CallableECPIntegralProvider,
    ECPChannelPlan,
    ECPGaussianTerm,
    ECPIntegralEvaluation,
    EffectiveCorePotentialPlan,
    PreparedECP,
)
from ._factorization import (
    DensityFittingPlan,
    FactorizedERITensor,
    PivotedCholeskyERIPlan,
)
from ._integrals import (
    contracted_electron_repulsion_element,
    dipole_integrals,
    electron_repulsion_tensor,
    kinetic_matrix,
    molecular_integrals,
    MolecularIntegralKernel,
    multipole_integrals,
    nuclear_attraction_matrix,
    nuclear_point_charge_energy,
    nuclear_repulsion_energy,
    overlap_matrix,
    point_charge_potential_matrix,
    range_separated_electron_repulsion_tensor,
)
from ._screening import GaussianScreeningPlan, PreparedGaussianScreening
from ._shell import (
    cartesian_angular_exponents,
    cartesian_primitive_normalization,
    GaussianShellPlan,
    GaussianShellRepresentation,
)
from ._spherical import cartesian_to_real_spherical, real_spherical_orders


__all__ = [
    "AOEvaluation",
    "AbstractECPIntegralProvider",
    "CallableECPIntegralProvider",
    "DensityFittingPlan",
    "DirectJKPlan",
    "DirectJKResult",
    "ECPChannelPlan",
    "ECPGaussianTerm",
    "ECPIntegralEvaluation",
    "EffectiveCorePotentialPlan",
    "FactorizedERITensor",
    "GaussianBasisPlan",
    "GaussianIntegralDerivativePlan",
    "GaussianIntegralDerivativeResult",
    "GaussianScreeningPlan",
    "GaussianShellPlan",
    "GaussianShellRepresentation",
    "MolecularIntegralKernel",
    "PivotedCholeskyERIPlan",
    "PreparedDirectJK",
    "PreparedECP",
    "PreparedGaussianBasis",
    "PreparedGaussianScreening",
    "ao_gradients",
    "ao_hessians",
    "ao_values",
    "boys0",
    "boys_values",
    "cartesian_angular_exponents",
    "cartesian_ao_values",
    "cartesian_primitive_normalization",
    "cartesian_to_real_spherical",
    "contracted_electron_repulsion_element",
    "dipole_integrals",
    "electron_repulsion_tensor",
    "range_separated_electron_repulsion_tensor",
    "evaluate_ao",
    "kinetic_matrix",
    "molecular_integrals",
    "multipole_integrals",
    "nuclear_attraction_matrix",
    "nuclear_point_charge_energy",
    "nuclear_repulsion_energy",
    "overlap_matrix",
    "point_charge_potential_matrix",
    "real_spherical_orders",
]
