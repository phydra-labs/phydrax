#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BMSStatus(IntFlag):
    """Status bits for null-infinity frame maps and charge products."""

    SUCCESS = 0
    NONFINITE = 1
    QUADRATURE_INVALID = 2
    ACAUSAL_FOUR_MOMENTUM = 4
    FRAME_INVALID = 8
    DERIVATIVE_INVALID = 16


class BMSQuadraturePlan(StrictModule, NonTrainableState):
    """Fixed angular support with declared harmonic exactness evidence."""

    directions: Array
    weights: Array
    charge_basis: Array
    charge_basis_gram: Array
    direction_capacity: int = eqx.field(static=True)
    charge_capacity: int = eqx.field(static=True)
    supported_bandlimit: int = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        weights: ArrayLike,
        charge_basis: ArrayLike,
        charge_basis_gram: ArrayLike,
        /,
        *,
        supported_bandlimit: int,
        quadrature_tolerance: float = 1.0e-10,
        plan_name: str = "bms-scri-quadrature",
    ):
        directions_host = np.asarray(directions, dtype=float)
        weights_host = np.asarray(weights, dtype=float)
        basis_host = np.asarray(charge_basis, dtype=float)
        gram_host = np.asarray(charge_basis_gram, dtype=float)
        if (
            directions_host.ndim != 2
            or directions_host.shape[1] != 3
            or directions_host.shape[0] < 4
        ):
            raise ValueError("directions must contain at least four unit vectors.")
        direction_count = int(directions_host.shape[0])
        if weights_host.shape != (direction_count,):
            raise ValueError("weights must contain one value per direction.")
        if basis_host.ndim != 2 or basis_host.shape[1] != direction_count:
            raise ValueError("charge_basis must have shape (charges, directions).")
        charge_count = int(basis_host.shape[0])
        if gram_host.shape != (charge_count, charge_count):
            raise ValueError(
                "charge_basis_gram must declare the analytic Gram matrix."
            )
        raw_bandlimit = np.asarray(supported_bandlimit)
        if (
            raw_bandlimit.shape != ()
            or not np.issubdtype(raw_bandlimit.dtype, np.integer)
            or int(raw_bandlimit) < 1
        ):
            raise ValueError("supported_bandlimit must be an integer of at least one.")
        bandlimit = int(raw_bandlimit)
        if charge_count < (bandlimit + 1) ** 2:
            raise ValueError(
                "charge_basis capacity cannot represent its supported bandlimit."
            )
        tolerance = float(quadrature_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("quadrature_tolerance must be finite and positive.")
        direction_norm = np.sqrt(np.sum(directions_host * directions_host, axis=-1))
        if (
            np.any(~np.isfinite(directions_host))
            or np.any(~np.isfinite(weights_host))
            or np.any(~np.isfinite(basis_host))
            or np.any(~np.isfinite(gram_host))
            or np.any(weights_host <= 0.0)
            or np.max(np.abs(direction_norm - 1.0)) > tolerance
            or abs(float(np.sum(weights_host)) - 4.0 * np.pi) > tolerance
        ):
            raise ValueError(
                "BMS quadrature requires finite unit directions and positive "
                "weights summing to 4π."
            )
        canonical_l_one = np.concatenate(
            (np.ones((1, direction_count)), directions_host.T), axis=0
        )
        if charge_count < 4 or np.max(
            np.abs(basis_host[:4] - canonical_l_one)
        ) > tolerance:
            raise ValueError(
                "The first four charge-basis rows must be (1, nx, ny, nz)."
            )
        first_moment = np.einsum("n,ni->i", weights_host, directions_host)
        second_moment = np.einsum(
            "n,ni,nj->ij", weights_host, directions_host, directions_host
        )
        expected_second_moment = (4.0 * np.pi / 3.0) * np.eye(3)
        sampled_gram = np.einsum(
            "qn,rn,n->qr", basis_host, basis_host, weights_host
        )
        first_error = float(np.max(np.abs(first_moment)))
        second_error = float(
            np.max(np.abs(second_moment - expected_second_moment))
        )
        gram_error = float(np.max(np.abs(sampled_gram - gram_host)))
        if max(first_error, second_error, gram_error) > tolerance:
            raise ValueError(
                "BMS quadrature does not satisfy its weighted l=0,1 moments "
                "and declared charge-basis Gram exactness."
            )
        if not isinstance(plan_name, str) or not plan_name:
            raise ValueError("plan_name must be a nonempty string.")

        self.directions = jnp.asarray(directions_host)
        self.weights = jnp.asarray(weights_host)
        self.charge_basis = jnp.asarray(basis_host)
        self.charge_basis_gram = jnp.asarray(gram_host)
        self.direction_capacity = direction_count
        self.charge_capacity = charge_count
        self.supported_bandlimit = bandlimit
        self.quadrature_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-bms-quadrature-plan",
                "name": plan_name,
                "directions": directions_host,
                "weights": weights_host,
                "charge_basis": basis_host,
                "charge_basis_gram": gram_host,
                "supported_bandlimit": bandlimit,
                "quadrature_tolerance": tolerance,
            }
        )

    def charges(self, data: BMSScriData, /) -> BMSChargeProduct:
        if data.direction_capacity != self.direction_capacity:
            raise ValueError("Scri data does not match BMS direction capacity.")
        normalization_error = jnp.abs(jnp.sum(self.weights) - 4.0 * jnp.pi)
        unit_error = jnp.max(
            jnp.abs(
                ein.contract("ni,ni->n", self.directions, self.directions) - 1.0
            )
        )
        first_moment_error = jnp.max(
            jnp.abs(ein.contract("n,ni->i", self.weights, self.directions))
        )
        second_moment = ein.contract(
            "n,ni,nj->ij", self.weights, self.directions, self.directions
        )
        second_moment_error = jnp.max(
            jnp.abs(second_moment - (4.0 * jnp.pi / 3.0) * jnp.eye(3))
        )
        sampled_gram = ein.contract(
            "qn,rn,n->qr", self.charge_basis, self.charge_basis, self.weights
        )
        charge_basis_gram_error = jnp.max(
            jnp.abs(sampled_gram - self.charge_basis_gram)
        )
        roundoff = (
            64.0
            * jnp.finfo(self.weights.dtype).eps
            * jnp.asarray(4.0 * jnp.pi, dtype=self.weights.dtype)
        )
        effective_tolerance = self.quadrature_tolerance + roundoff
        converged = (
            (normalization_error <= effective_tolerance)
            & (unit_error <= effective_tolerance)
            & (first_moment_error <= effective_tolerance)
            & (second_moment_error <= effective_tolerance)
            & (charge_basis_gram_error <= effective_tolerance)
        )

        normalization = 4.0 * jnp.pi
        energy = ein.contract(
            "tn,n->t", data.mass_aspect, self.weights
        ) / normalization
        momentum = ein.contract(
            "tn,ni,n->ti", data.mass_aspect, self.directions, self.weights
        ) / normalization
        four_momentum = jnp.concatenate((energy[:, None], momentum), axis=-1)
        supermomentum = ein.contract(
            "qn,tn,n->tq", self.charge_basis, data.mass_aspect, self.weights
        ) / normalization
        lorentz_charges = ein.contract(
            "tnab,n->tab", data.lorentz_charge_aspect, self.weights
        ) / normalization
        news_norm = jnp.real(data.news * jnp.conj(data.news))
        news_energy_flux = ein.contract(
            "tn,n->t", news_norm, self.weights
        ) / (16.0 * jnp.pi)

        finite = (
            jnp.all(jnp.isfinite(four_momentum))
            & jnp.all(jnp.isfinite(supermomentum))
            & jnp.all(jnp.isfinite(lorentz_charges))
            & jnp.all(jnp.isfinite(news_energy_flux))
        )
        momentum_squared = ein.contract("ti,ti->t", momentum, momentum)
        causal = jnp.all(energy >= 0.0) & jnp.all(
            energy * energy + self.quadrature_tolerance >= momentum_squared
        )
        physically_valid = causal & jnp.all(news_energy_flux >= 0.0)
        derivative_valid = finite
        qualified = finite & converged & physically_valid & derivative_valid
        status = jnp.asarray(int(BMSStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(finite, 0, int(BMSStatus.NONFINITE)).astype(
            jnp.int32
        )
        status = status | jnp.where(
            converged, 0, int(BMSStatus.QUADRATURE_INVALID)
        ).astype(jnp.int32)
        status = status | jnp.where(
            causal, 0, int(BMSStatus.ACAUSAL_FOUR_MOMENTUM)
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_valid, 0, int(BMSStatus.DERIVATIVE_INVALID)
        ).astype(jnp.int32)
        product_id = canonical_fingerprint(
            {
                "kind": "bms-charge-product",
                "data": data.data_id,
                "plan": self.plan_id,
            }
        )
        return BMSChargeProduct(
            data.retarded_times,
            four_momentum,
            lorentz_charges,
            supermomentum,
            news_energy_flux,
            normalization_error,
            unit_error,
            first_moment_error,
            second_moment_error,
            charge_basis_gram_error,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            data.data_id,
            self.plan_id,
            product_id,
        )


class BMSScriData(StrictModule, NonTrainableState):
    """Completed sampled Bondi data on a fixed angular support.

    The antisymmetric ``lorentz_charge_aspect`` carries rotation and boost
    charges explicitly.  It is not synthesized from the mass aspect because
    doing so would silently discard center-of-mass information.
    """

    retarded_times: Array
    mass_aspect: Array
    news: Array
    lorentz_charge_aspect: Array
    time_capacity: int = eqx.field(static=True)
    direction_capacity: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        retarded_times: ArrayLike,
        mass_aspect: ArrayLike,
        news: ArrayLike,
        lorentz_charge_aspect: ArrayLike,
        /,
        *,
        data_name: str = "bondi-scri-data",
    ):
        times = np.asarray(retarded_times)
        mass = np.asarray(mass_aspect)
        news_host = np.asarray(news)
        lorentz = np.asarray(lorentz_charge_aspect)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("retarded_times must be a nonempty vector.")
        if (
            not np.issubdtype(times.dtype, np.number)
            or np.iscomplexobj(times)
            or np.any(~np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))
        ):
            raise ValueError("retarded_times must be finite real increasing values.")
        if mass.ndim != 2:
            raise ValueError("mass_aspect must have shape (times, directions).")
        expected = mass.shape
        if expected[0] != times.size or news_host.shape != expected:
            raise ValueError("Mass aspect and news must share time/angular support.")
        if lorentz.shape != expected + (4, 4):
            raise ValueError(
                "lorentz_charge_aspect must have shape (times, directions, 4, 4)."
            )
        if np.iscomplexobj(mass) or np.iscomplexobj(lorentz):
            raise TypeError("Mass and Lorentz charge aspects must be real.")
        if (
            np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(news_host))
            or np.any(~np.isfinite(lorentz))
        ):
            raise ValueError("BMS scri data must be finite.")
        antisymmetry_error = np.max(
            np.abs(lorentz + np.swapaxes(lorentz, -1, -2)), initial=0.0
        )
        scale = np.max(np.abs(lorentz), initial=0.0)
        if antisymmetry_error > 1.0e-12 * max(1.0, float(scale)):
            raise ValueError("lorentz_charge_aspect must be antisymmetric.")
        if not isinstance(data_name, str) or not data_name:
            raise ValueError("data_name must be a nonempty string.")

        self.retarded_times = jnp.asarray(times)
        self.mass_aspect = jnp.asarray(mass)
        self.news = jnp.asarray(news_host)
        self.lorentz_charge_aspect = jnp.asarray(lorentz)
        self.time_capacity = int(times.size)
        self.direction_capacity = int(mass.shape[1])
        self.data_id = canonical_fingerprint(
            {
                "kind": "completed-bondi-scri-data",
                "name": data_name,
                "retarded_times": times,
                "mass_aspect": mass,
                "news": news_host,
                "lorentz_charge_aspect": lorentz,
            }
        )


class BMSChargeProduct(StrictModule):
    """Bondi four-momentum, Lorentz charges, and sampled supermomenta."""

    retarded_times: Array
    four_momentum: Array
    lorentz_charges: Array
    supermomentum: Array
    news_energy_flux: Array
    quadrature_normalization_error: Array
    direction_unit_error: Array
    first_moment_error: Array
    second_moment_error: Array
    charge_basis_gram_error: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    data_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class BMSFrameMap(StrictModule):
    """Poincaré BMS map of fixed scri directions and retarded times."""

    retarded_times: Array
    directions: Array
    conformal_factor: Array
    translation_cut: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    frame_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)


class TransformedBMSCharges(StrictModule):
    """Poincaré transformation of momentum and Lorentz-charge tensors."""

    four_momentum: Array
    lorentz_charges: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    source_product_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class BMSFrameTransformation(StrictModule, NonTrainableState):
    """Proper, orthochronous Lorentz boost followed by an origin translation.

    With boost velocity ``β`` the convention is ``P' = Λ(β) P`` and a rest
    momentum therefore has spatial momentum ``-γ M β``.  ``translation`` is
    ``(a⁰, a⃗)`` in the source frame, so the translated cut is
    ``α(n) = a⁰ - a⃗·n`` and ``u' = K (u - α)``.
    """

    boost_velocity: Array
    translation: Array
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        boost_velocity: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        frame_name: str = "bms-poincare-frame",
    ):
        boost = np.asarray(boost_velocity, dtype=float)
        shift = np.asarray(translation, dtype=float)
        if boost.shape != (3,) or shift.shape != (4,):
            raise ValueError("boost_velocity and translation require shapes (3,) and (4,).")
        speed_squared = float(np.dot(boost, boost))
        if (
            np.any(~np.isfinite(boost))
            or np.any(~np.isfinite(shift))
            or speed_squared >= 1.0
        ):
            raise ValueError("BMS boosts must be finite and strictly subluminal.")
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError("frame_name must be a nonempty string.")
        self.boost_velocity = jnp.asarray(boost)
        self.translation = jnp.asarray(shift)
        self.frame_id = canonical_fingerprint(
            {
                "kind": "bms-poincare-frame-transformation",
                "name": frame_name,
                "boost_velocity": boost,
                "translation": shift,
            }
        )

    def lorentz_matrix(self) -> Array:
        beta = self.boost_velocity
        speed_squared = ein.contract("i,i->", beta, beta)
        gamma = 1.0 / jnp.sqrt(1.0 - speed_squared)
        denominator = jnp.where(speed_squared > 0.0, speed_squared, 1.0)
        spatial = jnp.eye(3, dtype=beta.dtype) + (
            (gamma - 1.0) / denominator
        ) * ein.contract("i,j->ij", beta, beta)
        first_row = jnp.concatenate((gamma[None], -gamma * beta))
        remaining = jnp.concatenate((-gamma * beta[:, None], spatial), axis=1)
        return jnp.concatenate((first_row[None], remaining), axis=0)

    def map_null_infinity(
        self, plan: BMSQuadraturePlan, retarded_times: ArrayLike, /
    ) -> BMSFrameMap:
        times_host = np.asarray(retarded_times)
        if (
            times_host.ndim != 1
            or times_host.size == 0
            or not np.issubdtype(times_host.dtype, np.number)
            or np.iscomplexobj(times_host)
            or np.any(~np.isfinite(times_host))
            or (times_host.size > 1 and np.any(np.diff(times_host) <= 0.0))
        ):
            raise ValueError(
                "retarded_times must be finite real increasing values."
            )
        times = jnp.asarray(times_host)
        beta = self.boost_velocity
        speed_squared = ein.contract("i,i->", beta, beta)
        gamma = 1.0 / jnp.sqrt(1.0 - speed_squared)
        direction_dot = ein.contract("ni,i->n", plan.directions, beta)
        denominator = gamma * (1.0 - direction_dot)
        conformal_factor = 1.0 / denominator
        safe_speed_squared = jnp.where(speed_squared > 0.0, speed_squared, 1.0)
        aberration_factor = (
            (gamma - 1.0) * direction_dot / safe_speed_squared - gamma
        )
        transformed_directions = (
            plan.directions + aberration_factor[:, None] * beta
        ) / denominator[:, None]
        translation_cut = self.translation[0] - ein.contract(
            "ni,i->n", plan.directions, self.translation[1:]
        )
        transformed_times = conformal_factor[None, :] * (
            times[:, None] - translation_cut[None, :]
        )

        finite = (
            jnp.all(jnp.isfinite(transformed_times))
            & jnp.all(jnp.isfinite(transformed_directions))
            & jnp.all(jnp.isfinite(conformal_factor))
        )
        norm_error = jnp.max(
            jnp.abs(
                ein.contract(
                    "ni,ni->n", transformed_directions, transformed_directions
                )
                - 1.0
            )
        )
        converged = norm_error <= 10.0 * plan.quadrature_tolerance
        physically_valid = jnp.all(conformal_factor > 0.0)
        derivative_valid = finite & physically_valid
        qualified = finite & converged & physically_valid & derivative_valid
        status = jnp.asarray(int(BMSStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(finite, 0, int(BMSStatus.NONFINITE)).astype(
            jnp.int32
        )
        status = status | jnp.where(
            converged & physically_valid, 0, int(BMSStatus.FRAME_INVALID)
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_valid, 0, int(BMSStatus.DERIVATIVE_INVALID)
        ).astype(jnp.int32)
        map_id = canonical_fingerprint(
            {
                "kind": "bms-frame-map",
                "frame": self.frame_id,
                "plan": plan.plan_id,
                "retarded_times": times_host,
            }
        )
        return BMSFrameMap(
            transformed_times,
            transformed_directions,
            conformal_factor,
            translation_cut,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            self.frame_id,
            plan.plan_id,
            map_id,
        )

    def transform_charges(
        self, charges: BMSChargeProduct, /
    ) -> TransformedBMSCharges:
        transform = self.lorentz_matrix()
        four_momentum = ein.contract(
            "ab,tb->ta", transform, charges.four_momentum
        )
        translation_wedge = ein.contract(
            "a,tb->tab", self.translation, charges.four_momentum
        ) - ein.contract("b,ta->tab", self.translation, charges.four_momentum)
        shifted_lorentz = charges.lorentz_charges - translation_wedge
        lorentz_charges = ein.contract(
            "ac,tcd,bd->tab", transform, shifted_lorentz, transform
        )
        finite = jnp.all(jnp.isfinite(four_momentum)) & jnp.all(
            jnp.isfinite(lorentz_charges)
        )
        momentum_squared = ein.contract(
            "ti,ti->t", four_momentum[:, 1:], four_momentum[:, 1:]
        )
        physically_valid = jnp.all(four_momentum[:, 0] >= 0.0) & jnp.all(
            four_momentum[:, 0] ** 2 + 1.0e-12 >= momentum_squared
        )
        converged = charges.converged
        derivative_valid = charges.derivative_valid & finite
        qualified = (
            charges.qualified
            & finite
            & converged
            & physically_valid
            & derivative_valid
        )
        status = charges.status | jnp.where(
            finite, 0, int(BMSStatus.NONFINITE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            physically_valid, 0, int(BMSStatus.ACAUSAL_FOUR_MOMENTUM)
        ).astype(jnp.int32)
        product_id = canonical_fingerprint(
            {
                "kind": "transformed-bms-charges",
                "source": charges.product_id,
                "frame": self.frame_id,
            }
        )
        return TransformedBMSCharges(
            four_momentum,
            lorentz_charges,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            charges.product_id,
            self.frame_id,
            product_id,
        )
