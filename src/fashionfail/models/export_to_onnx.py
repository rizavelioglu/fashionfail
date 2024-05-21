from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from facere import model_classes
from loguru import logger


def main(args):
    # Load PyTorch checkpoint and put it on 'eval' mode.
    model_class = model_classes.get(args.model_class)
    model = model_class.load_from_checkpoint(args.ckpt_path)
    model.eval()

    # Input to the model: (batch_size,C,H,W)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model to ONNX.
    model.to_onnx(
        args.onnx_path,
        x,  # model input (or a tuple for multiple inputs)
        export_params=True,  # store the trained parameter weights inside the model file
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    # Load & verify ONNX model.
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create an inference session.
    ort_session = onnxruntime.InferenceSession(
        str(args.onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results (compare only the 'scores')
    np.testing.assert_allclose(
        to_numpy(torch_out[0]["scores"]), ort_outs[2], rtol=1e-03, atol=1e-05
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="The path to PyTorch checkpoint, i.e. `.pth` file.",
    )
    parser.add_argument(
        "--onnx_path",
        type=Path,
        required=True,
        help="The full path where the `.onnx` file will be saved.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        choices=model_classes.keys(),
        required=True,
        help="The exact name of the class that implemented the model.",
    )

    # Parse cli arguments.
    cli_args = parser.parse_args()
    # Execute the main function with args.
    main(cli_args)
    logger.info(
        "Exported model has been tested with ONNXRuntime, and the result looks good!"
    )
    logger.info(f"Model is saved at: {cli_args.onnx_path}")
