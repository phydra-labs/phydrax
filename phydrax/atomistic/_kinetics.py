#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..dynamics import StateLayout
from ..dynamics.identification import AbstractFeatureLibrary, FeatureEvaluation
from ._dynamics import PreparedAtomisticDynamics
from .sampling._collective_variable import AbstractCollectiveVariableProgram


class CollectiveVariableFeatureLibrary(AbstractFeatureLibrary):
    """Position-only atomistic CV features over the canonical trajectory layout."""

    variables: AbstractCollectiveVariableProgram
    dynamics: PreparedAtomisticDynamics
    state_layout: StateLayout
    input_layout: None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        variables: AbstractCollectiveVariableProgram,
        /,
    ):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(variables, AbstractCollectiveVariableProgram):
            raise TypeError("variables must implement AbstractCollectiveVariableProgram.")
        layout = StateLayout(
            (2, dynamics.system.capacity, 3),
            axes=("kinematic", "atom", "cartesian"),
            layout_id=canonical_fingerprint(
                {
                    "kind": "atomistic-trajectory-state-layout",
                    "system": dynamics.system.prepared_id,
                }
            ),
        )
        self.variables = variables
        self.dynamics = dynamics
        self.state_layout = layout
        self.input_layout = None
        self.feature_names = variables.names
        self.library_id = canonical_fingerprint(
            {
                "kind": "atomistic-cv-feature-library",
                "dynamics": dynamics.prepared_id,
                "variables": variables.program_id,
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        if inputs is not None:
            raise ValueError("CollectiveVariableFeatureLibrary is state-only.")
        values = jnp.asarray(states)
        if values.ndim < 3 or tuple(values.shape[-3:]) != self.state_layout.shape:
            raise ValueError(f"states must end in {self.state_layout.shape}.")
        leading = values.shape[:-3]
        positions = values[..., 0, :, :].reshape((-1, self.dynamics.system.capacity, 3))
        evaluated, successful = jax.vmap(self.variables.evaluate)(positions)
        evaluated = evaluated.reshape(leading + (self.variables.output_size,))
        successful = successful.reshape(leading)
        finite = jnp.all(jnp.isfinite(evaluated), axis=-1)
        valid = successful & finite
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], evaluated, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


__all__ = ["CollectiveVariableFeatureLibrary"]
