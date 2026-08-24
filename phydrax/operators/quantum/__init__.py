#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quantum operator algebra and closed/open-system evolution residuals."""

from ._algebra import (
    anticommutator,
    commutator,
    hermiticity_residual,
    quantum_bracket,
    unit_trace_residual,
)
from ._amplitude import (
    amplitude_ratio,
    AmplitudeRatio,
    ComplexParameterMode,
    LogAmplitude,
    sampling_log_weight,
)
from ._bath_decomposition import (
    drude_lorentz_matsubara,
    drude_lorentz_pade,
    drude_lorentz_pade_from_poles,
    fit_bath_exponentials,
    underdamped_brownian_two_pole,
)
from ._berry import (
    berry_link,
    berry_loop_phase,
    quantum_geometric_tensor,
    QuantumGeometricTensorResult,
)
from ._composite import embed_operator, partial_trace, tensor_product
from ._discrete import (
    AbstractDiscreteQuantumOperator,
    CallableDiscreteQuantumOperator,
    ConnectedConfigurations,
    local_estimate,
    LocalEstimate,
)
from ._dynamics import (
    HamiltonianAction,
    heisenberg_residual,
    schrodinger_residual,
    von_neumann_residual,
)
from ._fock import (
    BosonicFockSpace,
    FockCutoffEvidence,
    jaynes_cummings_hamiltonian,
    kerr_hamiltonian,
)
from ._information import (
    density_fidelity,
    purity,
    state_fidelity,
    trace_distance,
    von_neumann_entropy,
)
from ._nonmarkovianity import (
    analyze_dynamical_map_series,
    blp_information_backflow,
    DynamicalMapSeriesPhysicality,
)
from ._open_contracts import (
    ApproximationAxis,
    ApproximationQuantity,
    evaluate_open_system_promotion,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
    OpenSystemPromotionDecision,
    OpenSystemPromotionPolicy,
    OpenSystemRefinement,
    PhysicalityStatus,
    QuantumGeneratorAction,
    QuantumObservablePlan,
)
from ._open_system import lindblad_dissipator, lindblad_residual
from ._propagation import (
    apply_unitary_to_state,
    conjugate_density,
    density_invariant_residuals,
    unitarity_residual,
)
from ._pseudomode import (
    BathCorrelationExpansion,
    lorentzian_pseudomode,
    Pseudomode,
    ReactionCoordinateMapping,
)
from ._states import (
    density_expectation,
    density_from_factor,
    observable_variance,
    state_expectation,
    state_norm_residual,
)
from ._symmetry import FiniteSignedPermutationSymmetry, SymmetryProjectedAmplitude


__all__ = [
    "AbstractDiscreteQuantumOperator",
    "AmplitudeRatio",
    "HamiltonianAction",
    "anticommutator",
    "CallableDiscreteQuantumOperator",
    "ComplexParameterMode",
    "ConnectedConfigurations",
    "commutator",
    "density_expectation",
    "density_fidelity",
    "density_from_factor",
    "FiniteSignedPermutationSymmetry",
    "embed_operator",
    "heisenberg_residual",
    "hermiticity_residual",
    "lindblad_dissipator",
    "LocalEstimate",
    "LogAmplitude",
    "lindblad_residual",
    "amplitude_ratio",
    "local_estimate",
    "quantum_bracket",
    "observable_variance",
    "purity",
    "partial_trace",
    "schrodinger_residual",
    "state_expectation",
    "state_fidelity",
    "state_norm_residual",
    "sampling_log_weight",
    "SymmetryProjectedAmplitude",
    "tensor_product",
    "trace_distance",
    "unit_trace_residual",
    "von_neumann_residual",
    "von_neumann_entropy",
    "QuantumGeometricTensorResult",
    "apply_unitary_to_state",
    "berry_link",
    "berry_loop_phase",
    "conjugate_density",
    "density_invariant_residuals",
    "quantum_geometric_tensor",
    "unitarity_residual",
    "ApproximationAxis",
    "ApproximationQuantity",
    "BathCorrelationExpansion",
    "BosonicFockSpace",
    "FockCutoffEvidence",
    "OpenSystemApproximationEvidence",
    "evaluate_open_system_promotion",
    "OpenSystemPromotionDecision",
    "OpenSystemPromotionPolicy",
    "OpenSystemPhysicalityEvidence",
    "OpenSystemRefinement",
    "PhysicalityStatus",
    "Pseudomode",
    "QuantumGeneratorAction",
    "QuantumObservablePlan",
    "ReactionCoordinateMapping",
    "jaynes_cummings_hamiltonian",
    "drude_lorentz_pade",
    "kerr_hamiltonian",
    "lorentzian_pseudomode",
    "DynamicalMapSeriesPhysicality",
    "analyze_dynamical_map_series",
    "blp_information_backflow",
    "drude_lorentz_matsubara",
    "drude_lorentz_pade_from_poles",
    "fit_bath_exponentials",
    "underdamped_brownian_two_pole",
]
