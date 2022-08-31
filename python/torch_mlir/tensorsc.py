from torch_mlir import run_pipeline_with_repro_report


def compile(module,
            verbose: bool = False):
    """Convert a PyTorch model to MLIR.

    Args:
        model: An MLIR module that contains the converted model in the specified
        output type.
        verbose: If true, print extra information about the conversion.

    Returns:
        An TensorSC MLIR module.
    """
    run_pipeline_with_repro_report(
        module,
        "linalg-on-tensors-backend-to-tensorsc-backend-pipeline",
        "Linalg-on-Tensors Backend IR -> TensorSC Backend IR")
    if verbose:
        print("\n====================")
        print("TensorSC Backend IR")
        print(module)
    return module