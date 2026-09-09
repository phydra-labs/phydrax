#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....observation import (
    CholeskyCovarianceAction,
    CovarianceAction,
    PrecisionCovarianceAction,
)
from ....solver import FieldObservationPlan
from ....units import conversion_factor, convert_value, UnitDefinition, VOLT
from ....uq import ParameterSpace, PosteriorProblem
from ._finite_patch import DC_CONDUCTIVITY_UNIT, PreparedDC


class LogConductivity(StrictModule, NonTrainableState):
    """Positive scalar cell conductivity from dimensionless logarithmic increments.

    sigma[c] = reference[c] * exp(parameters[cell_parameter_indices[c]]).
    Reference data have one positive entry per cell in native mesh block order.
    Default indexing assigns one parameter per cell. Explicit consecutive group
    indices allow, for example, a homogeneous one-parameter inversion without an
    artificial dense cell/parameter design matrix. Priors belong to ParameterSpace
    and are explicitly priors on these log increments, not on conductivity.
    """

    reference: Array
    cell_parameter_indices: Array
    parameter_count: int = eqx.field(static=True)

    def __init__(
        self,
        reference_cell_conductivity: ArrayLike,
        /,
        *,
        cell_parameter_indices: ArrayLike | None = None,
        unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ):
        raw = jnp.asarray(reference_cell_conductivity)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Reference conductivity must be real.")
        reference = np.asarray(
            convert_value(raw, source=unit, target=DC_CONDUCTIVITY_UNIT), dtype=float
        )
        if (
            reference.ndim != 1
            or reference.size == 0
            or np.any(~np.isfinite(reference))
            or np.any(reference <= 0)
        ):
            raise ValueError(
                "Reference conductivity requires one finite positive scalar per cell."
            )
        indices = (
            np.arange(reference.size, dtype=np.int32)
            if cell_parameter_indices is None
            else np.asarray(cell_parameter_indices)
        )
        if (
            indices.shape != reference.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
        ):
            raise ValueError(
                "Cell parameter indices must be nonnegative integers, one per cell."
            )
        unique = np.unique(indices)
        if not np.array_equal(unique, np.arange(unique.size)):
            raise ValueError(
                "Cell parameter groups must use consecutive indices starting at zero."
            )
        self.reference = jnp.asarray(reference)
        self.cell_parameter_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.parameter_count = unique.size

    def __call__(self, parameters: ArrayLike, /) -> Array:
        raw = jnp.asarray(parameters)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Log conductivity parameters must be real.")
        values = jnp.asarray(raw, dtype=self.reference.dtype)
        if values.shape != (self.parameter_count,):
            raise ValueError("Log conductivity parameters must match parameter_count.")
        conductivity = self.reference * jnp.exp(values[self.cell_parameter_indices])
        return eqx.error_if(
            conductivity,
            jnp.any(~jnp.isfinite(conductivity)) | jnp.any(conductivity <= 0),
            "Log conductivity must produce finite strictly positive values.",
        )


def _voltage_identity(voltages, args):
    del args
    return voltages


class DCElectricalObservationPlan(StrictModule, NonTrainableState):
    """Native signed Gaussian DC likelihood and posterior composition.

    ``observed`` and ``covariance`` use the same declared voltage unit (covariance
    in its square). Both are converted to SI at preparation. Negative and zero
    voltage observations are valid; there is no data logarithm, magnitude-taking,
    implicit geometric factor, or apparent-resistivity conversion.
    """

    dc: PreparedDC
    parameterization: LogConductivity
    field_observation: FieldObservationPlan

    def __init__(
        self,
        dc: PreparedDC,
        parameterization: LogConductivity,
        observed: ArrayLike,
        covariance: CovarianceAction,
        /,
        *,
        observation_id: str = "dc-voltage-observations",
        voltage_unit: UnitDefinition = VOLT,
    ):
        if not isinstance(dc, PreparedDC) or not isinstance(
            parameterization, LogConductivity
        ):
            raise TypeError("DC observations require PreparedDC and LogConductivity.")
        if parameterization.reference.shape != (dc.cell_count,):
            raise ValueError("Log conductivity cell layout must match the DC mesh.")
        if not isinstance(
            covariance, (CholeskyCovarianceAction, PrecisionCovarianceAction)
        ):
            raise TypeError("covariance must be a native observation CovarianceAction.")
        raw = jnp.asarray(observed)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("DC voltage observations must be real.")
        if raw.shape != (dc.plan.survey.measurement_count,):
            raise ValueError(
                "Observed voltages must have one entry per survey measurement."
            )
        values = convert_value(raw, source=voltage_unit, target=VOLT)
        factor = float(conversion_factor(voltage_unit, VOLT))
        if factor != 1.0:
            if isinstance(covariance, CholeskyCovarianceAction):
                covariance = CholeskyCovarianceAction(
                    factor * covariance.lower_cholesky, covariance.layout
                )
            else:
                covariance = PrecisionCovarianceAction(
                    covariance.precision / factor**2,
                    covariance.logdet_covariance
                    + 2 * covariance.layout.size * jnp.log(factor),
                    covariance.layout,
                )
        self.dc = dc
        self.parameterization = parameterization
        self.field_observation = FieldObservationPlan(
            _voltage_identity, values, covariance, observation_id=observation_id
        )

    def predict(self, parameters: ArrayLike, args=None, /) -> Array:
        del args
        return self.dc.predict(self.parameterization(parameters))

    def log_likelihood(self, parameters: ArrayLike, /) -> Array:
        return self.field_observation.log_likelihood(self.predict(parameters))

    def posterior(self, parameter_space: ParameterSpace, /) -> PosteriorProblem:
        """Compose real native posterior inference on one log-increment array.

        ``ParameterSpace`` must constrain to shape ``(parameter_count,)``. Its
        supplied prior and any bijector/Jacobian semantics remain authoritative.
        The observation term only evaluates the physical PDE likelihood.
        """
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be ParameterSpace.")
        if jax.tree_util.tree_structure(
            parameter_space.initial
        ) != jax.tree_util.tree_structure(
            jnp.zeros(1)
        ) or parameter_space.physical_shapes != (
            (self.parameterization.parameter_count,),
        ):
            raise ValueError(
                "DC posterior requires one log-increment array matching parameter_count."
            )
        dc = self.dc
        parameterization = self.parameterization
        field_observation = self.field_observation

        def predict(parameters):
            return dc.predict(parameterization(parameters))

        def log_likelihood(parameters):
            return field_observation.log_likelihood(predict(parameters))

        covariance = field_observation.covariance
        residual = None
        if isinstance(covariance, CholeskyCovarianceAction):
            observed = field_observation.observed

            def residual(parameters):
                return covariance.whiten(predict(parameters) - observed)

        return PosteriorProblem(
            parameter_space,
            log_likelihood,
            predict=predict,
            gauss_newton_residual=residual,
        )


__all__ = ["DCElectricalObservationPlan", "LogConductivity"]
