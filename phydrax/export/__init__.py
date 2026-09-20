from importlib import import_module

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ._array_archive import __all__ as _array_archive_all
from ._complex_parameters import (
    complex_coefficients_to_frame,
    ComplexImportPolicy,
    ComplexInterchangeEntry,
    ComplexInterchangeSemantics,
    ComplexInterchangeState,
    export_complex_parameters,
    frame_coefficients_to_complex,
    import_complex_parameters,
)
from ._complex_training import (
    ComplexOptimizerInterchangeState,
    ComplexOptimizerRouteKind,
    ComplexOptimizerStateGroup,
    ComplexOptimizerStateLayout,
    ComplexTrainingInterchangeState,
    export_complex_training_state,
    import_complex_training_state,
    ImportedComplexTrainingState,
    prepare_complex_optimizer_state_layout,
    prepare_complex_training_interchange,
    PreparedComplexTrainingInterchange,
    read_complex_training_checkpoint,
    RNGInterchangeState,
    write_complex_training_checkpoint,
)
from ._discrete_velocity_iree import (
    discrete_velocity_iree_availability,
    DiscreteVelocityIREEContract,
    DiscreteVelocityIREEExportBundle,
    DiscreteVelocityIREEExportMode,
    prepare_discrete_velocity_iree_contract,
    save_discrete_velocity_iree,
)
from ._iree import (
    IREEArtifactManifest,
    IREEExecutable,
    IREEExportPolicy,
    IREEExportResult,
    load_iree,
    save_iree,
)
from ._lattice_boltzmann_iree import (
    lattice_boltzmann_iree_availability,
    LatticeBoltzmannIREEContract,
    LatticeBoltzmannIREEExportBundle,
    LatticeBoltzmannIREEExportMode,
    prepare_lattice_boltzmann_iree_contract,
    save_lattice_boltzmann_iree,
)
from ._neutral import (
    NeutralAdapterBoundary,
    NeutralFieldSchema,
    NeutralGeometrySchema,
    NeutralMaterialSchema,
    NeutralPointCloudSchema,
    NeutralSchemaKind,
)
from ._onnx import OnnxExportResult, save_onnx


_FACADE_EXPORT_MODULES = ("._array_archive",)


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ComplexImportPolicy",
    "ComplexInterchangeEntry",
    "ComplexInterchangeSemantics",
    "ComplexInterchangeState",
    "ComplexOptimizerInterchangeState",
    "ComplexOptimizerRouteKind",
    "ComplexOptimizerStateGroup",
    "ComplexOptimizerStateLayout",
    "ComplexTrainingInterchangeState",
    "DiscreteVelocityIREEContract",
    "DiscreteVelocityIREEExportBundle",
    "DiscreteVelocityIREEExportMode",
    "ImportedComplexTrainingState",
    "PreparedComplexTrainingInterchange",
    "RNGInterchangeState",
    "IREEArtifactManifest",
    "IREEExecutable",
    "IREEExportPolicy",
    "IREEExportResult",
    "LatticeBoltzmannIREEExportMode",
    "LatticeBoltzmannIREEContract",
    "LatticeBoltzmannIREEExportBundle",
    "NeutralAdapterBoundary",
    "NeutralFieldSchema",
    "NeutralGeometrySchema",
    "NeutralMaterialSchema",
    "NeutralPointCloudSchema",
    "NeutralSchemaKind",
    "discrete_velocity_iree_availability",
    "complex_coefficients_to_frame",
    "export_complex_parameters",
    "export_complex_training_state",
    "import_complex_training_state",
    "prepare_complex_optimizer_state_layout",
    "prepare_complex_training_interchange",
    "read_complex_training_checkpoint",
    "write_complex_training_checkpoint",
    "frame_coefficients_to_complex",
    "load_iree",
    "lattice_boltzmann_iree_availability",
    "prepare_lattice_boltzmann_iree_contract",
    "prepare_discrete_velocity_iree_contract",
    "save_discrete_velocity_iree",
    "save_lattice_boltzmann_iree",
    "import_complex_parameters",
    "OnnxExportResult",
    "save_onnx",
    "save_iree",
]
__all__ += [name for name in _array_archive_all if name not in __all__]
