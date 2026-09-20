#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ._convention import AmplitudeConvention, HarmonicConvention, PhasorConvention
from ._conversion import s_to_z, z_to_s
from ._cyclic import cyclic_phase_shift
from ._frequency import FrequencyAxis
from ._harmonic_balance import harmonic_balance_residual
from ._mode import ComplexMode, modal_overlap
from ._network import Port, ScatteringMatrix
from ._passivity import check_scattering_passivity, PassivityEvidence
from ._profiles import frequency_candidate_profiles
from ._solver import CompiledFrequencySystem, FrequencyDomainResult
from ._sweep import evaluate_frequency_sweep
from ._vector_fit import fit_fixed_poles, PoleResidueModel


__all__ = [
    "AmplitudeConvention",
    "CompiledFrequencySystem",
    "ComplexMode",
    "FrequencyAxis",
    "FrequencyDomainResult",
    "HarmonicConvention",
    "PassivityEvidence",
    "PhasorConvention",
    "PoleResidueModel",
    "Port",
    "ScatteringMatrix",
    "check_scattering_passivity",
    "cyclic_phase_shift",
    "evaluate_frequency_sweep",
    "fit_fixed_poles",
    "frequency_candidate_profiles",
    "harmonic_balance_residual",
    "modal_overlap",
    "s_to_z",
    "z_to_s",
]
