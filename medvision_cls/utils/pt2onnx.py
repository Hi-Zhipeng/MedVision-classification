"""
Convert backbone .pt weights to ONNX format.

Usage (CLI):
    python -m medvision_cls.utils.pt2onnx \
        --pt outputs/checkpoints/exp.pt \
        --model resnet50 \
        --num_classes 2 \
        --input_shape 3 224 224
"""

import os
import torch
from typing import List, Optional


def pt_to_onnx(
    pt_path: str,
    model_name: str,
    num_classes: int,
    input_shape: List[int],
    output_path: Optional[str] = None,
    opset_version: int = 11,
) -> str:
    """
    Convert a backbone .pt file to ONNX.

    Args:
        pt_path: Path to the .pt backbone weights file
        model_name: Model architecture name, e.g. "resnet50"
        num_classes: Number of output classes
        input_shape: Input shape without batch dim, e.g. [3, 224, 224] or [1, 64, 64, 64]
        output_path: Output .onnx path; defaults to same directory as pt_path
        opset_version: ONNX opset version

    Returns:
        str: Path to the exported .onnx file
    """
    from .pt_checkpoint import load_from_pt

    if output_path is None:
        output_path = os.path.splitext(pt_path)[0] + ".onnx"

    model = load_from_pt(pt_path, model_name=model_name, num_classes=num_classes)
    sample_input = torch.zeros(1, *input_shape)

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            external_data=False
        )

    print(f"Exported: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert backbone .pt to ONNX")
    parser.add_argument("--pt", required=True, help="Path to .pt file")
    parser.add_argument("--model", required=True, help="Model name, e.g. resnet50")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--input_shape", type=int, nargs="+", required=True,
                        help="Input shape without batch dim, e.g. 3 224 224")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    pt_to_onnx(
        pt_path=args.pt,
        model_name=args.model,
        num_classes=args.num_classes,
        input_shape=args.input_shape,
        output_path=args.output,
        opset_version=args.opset,
    )
