# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded 1D quantum electrostatics and coherent stationary open transport.

SI orthonormal cell basis; explicit finite transverse modes and semi-infinite
leads. Numerical success is not empirical/foundry qualification. Scattering
and transient lead memory are separate physical models, not eta parameters.
"""

from ._basis import (
    ChainHamiltonian,
    EffectiveMass1D,
    QuantumResources,
    SchrodingerResult,
    selected_eigenpairs,
    solve_schrodinger,
    TransverseModes,
)
from ._coherent import (
    bound_states,
    BoundStates,
    CoherentDevice,
    CoherentEvidence,
    CoherentResult,
    integrate_coherent,
    SpectralPoint,
)
from ._density_gradient import DensityGradient1D, DensityGradientResult
from ._dynamic import (
    lead_memory_kernel,
    QuantumInitialState,
    QuantumMemoryEvidence,
    QuantumPulse,
    QuantumTransientResult,
    solve_quantum_transient,
)
from ._hybrid import (
    HybridCouplingEvidence,
    HybridInterfaceEvaluation,
    HybridOperatingPoint,
    QuantumClassicalInterface,
    solve_quantum_classical_interface,
)
from ._leads import BoundStateOccupation, scalar_embedding, SemiInfiniteLead
from ._poisson import (
    QuantumPoisson1D,
    QuantumPoissonEvidence,
    QuantumPoissonResult,
    solve_coherent_poisson,
    solve_schrodinger_poisson,
)
from ._response import (
    coherent_low_frequency_noise,
    CoherentNoiseResult,
    finite_frequency_quantum_response,
    QuantumCapacitance,
    QuantumFrequencyEvidence,
    QuantumFrequencyResponse,
    quasistatic_quantum_response,
    QuasistaticQuantumResponse,
)
from ._scattering import (
    OpticalPhononBath,
    PhononEnergyGrid,
    ScatteringEvidence,
    ScatteringResult,
    solve_phonon_transport,
)


__all__ = [
    "BoundStateOccupation",
    "BoundStates",
    "ChainHamiltonian",
    "CoherentDevice",
    "CoherentEvidence",
    "CoherentNoiseResult",
    "HybridCouplingEvidence",
    "HybridInterfaceEvaluation",
    "HybridOperatingPoint",
    "CoherentResult",
    "DensityGradient1D",
    "DensityGradientResult",
    "EffectiveMass1D",
    "OpticalPhononBath",
    "PhononEnergyGrid",
    "QuantumCapacitance",
    "QuantumFrequencyEvidence",
    "QuantumFrequencyResponse",
    "QuantumInitialState",
    "QuantumMemoryEvidence",
    "QuantumPoisson1D",
    "QuantumPoissonEvidence",
    "QuantumPoissonResult",
    "QuantumClassicalInterface",
    "QuantumPulse",
    "QuantumResources",
    "QuantumTransientResult",
    "QuasistaticQuantumResponse",
    "ScatteringEvidence",
    "ScatteringResult",
    "SchrodingerResult",
    "SemiInfiniteLead",
    "SpectralPoint",
    "TransverseModes",
    "bound_states",
    "coherent_low_frequency_noise",
    "finite_frequency_quantum_response",
    "integrate_coherent",
    "lead_memory_kernel",
    "quasistatic_quantum_response",
    "scalar_embedding",
    "selected_eigenpairs",
    "solve_coherent_poisson",
    "solve_phonon_transport",
    "solve_quantum_transient",
    "solve_quantum_classical_interface",
    "solve_schrodinger",
    "solve_schrodinger_poisson",
]
