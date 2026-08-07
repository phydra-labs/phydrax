#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import phydrax as phx


def test_delay_solver_families_are_public_and_declared_once():
    expected = {
        "CaputoFractionalProblem",
        "CheckpointedDelayAdjoint",
        "ConstantDelay",
        "ConvolutionKernel",
        "ConvolutionVolterraProblem",
        "DelayDifferentialProblem",
        "DelayHistoryWindow",
        "DelayJumpMap",
        "DistributedDelay",
        "FunctionalDelay",
        "JumpDelayBackendResult",
        "JumpDelayProblem",
        "NeutralDelayProblem",
        "RoughDelayDifferentialProblem",
        "RoughDelayDrift",
        "RoughDelayVectorFields",
        "SegmentedDelayAdjoint",
        "StateDependentDelay",
        "solve_caputo_fractional",
        "solve_convolution_volterra",
        "solve_diffrax_delay",
        "solve_diffrax_delay_segmented",
        "solve_jump_delay",
        "solve_rough_delay",
    }

    assert expected <= set(phx.solver.__all__)
    assert all(hasattr(phx.solver, name) for name in expected)
    assert len(phx.solver.__all__) == len(set(phx.solver.__all__))


def test_obsolete_fixed_grid_delay_surface_remains_removed():
    assert not hasattr(phx.solver, "StochasticDelayProblem")
    assert not hasattr(phx.solver, "solve_stochastic_delay")
