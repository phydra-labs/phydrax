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
from ._composite import embed_operator, partial_trace, tensor_product
from ._dynamics import (
    HamiltonianAction,
    heisenberg_residual,
    schrodinger_residual,
    von_neumann_residual,
)
from ._information import (
    density_fidelity,
    purity,
    state_fidelity,
    trace_distance,
    von_neumann_entropy,
)
from ._open_system import lindblad_dissipator, lindblad_residual
from ._states import (
    density_expectation,
    density_from_factor,
    observable_variance,
    state_expectation,
    state_norm_residual,
)


__all__ = [
    "HamiltonianAction",
    "anticommutator",
    "commutator",
    "density_expectation",
    "density_fidelity",
    "density_from_factor",
    "embed_operator",
    "heisenberg_residual",
    "hermiticity_residual",
    "lindblad_dissipator",
    "lindblad_residual",
    "quantum_bracket",
    "observable_variance",
    "purity",
    "partial_trace",
    "schrodinger_residual",
    "state_expectation",
    "state_fidelity",
    "state_norm_residual",
    "tensor_product",
    "trace_distance",
    "unit_trace_residual",
    "von_neumann_residual",
    "von_neumann_entropy",
]
