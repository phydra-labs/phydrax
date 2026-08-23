#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import ClassVar

import diffrax as dfx
import equinox as eqx

from .._numerics._ssp_runge_kutta import ssprk33_step, ssprk54_step
from ._temporal_method import TemporalMethodCapabilities


class SSPRK33(dfx.AbstractSolver):
    """Three-stage, third-order strong-stability-preserving Runge--Kutta method."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = dfx.LocalLinearInterpolation
    capabilities: TemporalMethodCapabilities = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(self):
        self.solver_id = "temporal:ssprk:3:3"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="ssp-rk",
            order=3,
            dense_order=1,
            adaptive=False,
            stage_abscissae=(0.0, 1.0, 0.5),
            causal_stage_extent=1.0,
            ssp_coefficient=1.0,
            verified=True,
            method_id=self.solver_id,
        )

    def order(self, terms):
        del terms
        return 3

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = ssprk33_step(terms.vf, t0, y0, t1 - t0, args)
        return y1, None, {"y0": y0, "y1": y1}, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class SSPRK54(dfx.AbstractSolver):
    """Five-stage, fourth-order optimal SSP Runge--Kutta method."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = dfx.LocalLinearInterpolation
    capabilities: TemporalMethodCapabilities = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(self):
        self.solver_id = "temporal:ssprk:5:4"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="ssp-rk",
            order=4,
            dense_order=1,
            adaptive=False,
            stage_abscissae=(
                0.0,
                0.391752226571890,
                0.586079689311540,
                0.474542363026870,
                0.935010631009240,
            ),
            causal_stage_extent=1.0,
            ssp_coefficient=1.50818004975927,
            verified=True,
            method_id=self.solver_id,
        )

    def order(self, terms):
        del terms
        return 4

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = ssprk54_step(terms.vf, t0, y0, t1 - t0, args)
        return y1, None, {"y0": y0, "y1": y1}, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["SSPRK33", "SSPRK54", "ssprk33_step", "ssprk54_step"]
