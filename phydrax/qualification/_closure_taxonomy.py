#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Orthogonal taxonomy for capability depth, physics, coupling, and workflows."""

from enum import IntEnum, StrEnum


class CapabilityDepth(IntEnum):
    SEMANTIC = 0
    ANALYTIC_CONTROL = 1
    LOCAL_CONSTITUTIVE = 2
    REDUCED_SYSTEM = 3
    SPATIAL_SINGLE_PHYSICS = 4
    SPATIAL_COUPLED = 5
    ADAPTIVE_DISTRIBUTED = 6
    SCIENTIFICALLY_VALIDATED = 7


class ClosureState(StrEnum):
    UNCLASSIFIED = "unclassified"
    CLASSIFIED = "classified"
    IMPLEMENTATION_CLOSED = "implementation-closed"
    RELEASE_CLOSED = "release-closed"


class PhysicsField(StrEnum):
    SOLID_MECHANICS = "solid-mechanics"
    FLUID_MECHANICS = "fluid-mechanics"
    THERMAL = "thermal"
    CHEMICAL_SPECIES = "chemical-species"
    ELECTRIC = "electric"
    MAGNETIC = "magnetic"
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"
    IONIZING_RADIATION = "ionizing-radiation"
    ELECTRONIC_QUANTUM = "electronic-quantum"
    GRAVITATIONAL = "gravitational"
    BIOLOGICAL_ACTIVE = "biological-active"


class CarrierRepresentation(StrEnum):
    CONTINUUM_VOLUME = "continuum-volume"
    INTERFACE_SURFACE = "interface-surface"
    PARTICLE = "particle"
    NETWORK = "network"
    KINETIC_DISTRIBUTION = "kinetic-distribution"
    ATOMISTIC = "atomistic"
    REDUCED_SYSTEM = "reduced-system"


class CouplingLocation(StrEnum):
    BULK = "bulk"
    BOUNDARY = "boundary"
    MOVING_INTERFACE = "moving-interface"
    CONTACT = "contact"
    PARTICLE_FIELD = "particle-field"
    NETWORK_PORT = "network-port"
    CROSS_SCALE_TRANSFER = "cross-scale-transfer"


class ExecutionRegime(StrEnum):
    STATIC = "static"
    TRANSIENT = "transient"
    FREQUENCY_DOMAIN = "frequency-domain"
    EIGENVALUE_STABILITY = "eigenvalue-stability"
    PERIODIC_STEADY_STATE = "periodic-steady-state"
    STOCHASTIC = "stochastic"
    MULTIRATE = "multirate"
    MULTISCALE = "multiscale"


class TopologyRegime(StrEnum):
    FIXED = "fixed"
    MOVING_MESH = "moving-mesh"
    FREE_SURFACE = "free-surface"
    PHASE_CHANGE = "phase-change"
    FRACTURE = "fracture"
    CONTACT = "contact"
    ACTIVATION_REMOVAL = "activation-removal"
    REMESHING = "remeshing"
    TOPOLOGY_TRANSITION = "topology-transition"


class WorkflowClass(StrEnum):
    FORWARD = "forward"
    INVERSE = "inverse"
    CALIBRATION = "calibration"
    OPTIMIZATION = "optimization"
    CONTROL = "control"
    UNCERTAINTY_QUANTIFICATION = "uncertainty-quantification"
    EXPERIMENT_CORRELATION = "experiment-correlation"
    RESTART_REPLAY = "restart-replay"
    QUALIFICATION = "qualification"
    DEPLOYMENT = "deployment"


class ImplementationOwnership(StrEnum):
    NATIVE = "native"
    PROVIDER = "provider"
    RESEARCH = "research"
    REJECTED = "rejected"


class SourceReuseClass(StrEnum):
    PERMISSIVE = "permissive"
    WEAK_COPYLEFT = "weak-copyleft"
    STRONG_COPYLEFT = "strong-copyleft"
    SOURCE_AVAILABLE = "source-available"
    PROPRIETARY = "proprietary"
    PUBLIC_DOMAIN = "public-domain"
    UNKNOWN = "unknown"


class ClosureDisposition(StrEnum):
    IMPLEMENTED = "implemented"
    CANDIDATE = "candidate"
    PROVIDER = "provider"
    RESEARCH = "research"
    REJECTED = "rejected"
    MISSING = "missing"


__all__ = [
    "CapabilityDepth",
    "CarrierRepresentation",
    "ClosureDisposition",
    "ClosureState",
    "CouplingLocation",
    "ExecutionRegime",
    "ImplementationOwnership",
    "PhysicsField",
    "SourceReuseClass",
    "TopologyRegime",
    "WorkflowClass",
]
