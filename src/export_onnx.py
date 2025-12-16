from pathlib import Path
from typing import Literal

import onnx
import typer
from nemo.collections.tts.models import HifiGanModel
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer

from models.fastspeech import FastSpeechModel


def export_tts_model(
    pretrained_path: str,
    target_path: str,
    quantization: QuantType | Literal["fp16"] | None = QuantType.QInt8,
    dynamic: bool = True,
):
    """Export the TTS model to ONNX format with optional quantization."""
    model = FastSpeechModel.restore_from(pretrained_path).eval()
    # make sure target path ends with .onnx
    if not target_path.endswith(".onnx"):
        raise ValueError("Target path must end with .onnx")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    model.export(target_path)
    if isinstance(quantization, QuantType):
        print(f"Applying {quantization} quantization...")
        if dynamic:
            quantize_dynamic(
                target_path,
                target_path,
                weight_type=quantization,
                op_types_to_quantize=[
                    "MatMul",
                    "Attention",
                    "LSTM",
                    "Gather",
                    "Transpose",
                    "EmbedLayerNormalization",
                ],
            )
        else:
            raise NotImplementedError("Static quantization is not implemented in this script.")

    elif quantization == "fp16":
        # Apply float16 conversion
        print("Applying float16 conversion...")
        optimized_model = optimizer.optimize_model(target_path, model_type="bert")
        optimized_model.convert_float_to_float16(keep_io_types=True)
        optimized_model.save_model_to_file(target_path)
    else:
        print("No quantization applied.")
        pass


def export_hifigan_model(
    pretrained_path: str,
    target_path: str,
    quantization: QuantType | Literal["fp16"] = "fp16",
    dynamic: bool = True,
):
    """Export the HiFi-GAN model to ONNX format with optional quantization."""
    model = HifiGanModel.restore_from(pretrained_path).eval()
    # make sure target path ends with .onnx
    if not target_path.endswith(".onnx"):
        raise ValueError("Target path must end with .onnx")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    model.export(target_path)
    if not dynamic:
        raise ValueError("Static export for HiFi-GAN is not supported.")
    if isinstance(quantization, QuantType):
        print(f"Applying {quantization} quantization...")
        quantize_dynamic(
            target_path,
            target_path,
            weight_type=quantization,
            op_types_to_quantize=[
                "MatMul",
                "Attention",
                "LSTM",
                "Gather",
                "Transpose",
                "EmbedLayerNormalization",
            ],
        )
    elif quantization == "fp16":
        # Apply float16 conversion
        print("Applying float16 conversion...")
        model_fp16 = float16.convert_float_to_float16(onnx.load(target_path), keep_io_types=True)
        onnx.save(model_fp16, target_path)
    else:
        print("No quantization applied.")

def main(
    model_type: str = typer.Option(..., help="Type of model to export: 'tts' or 'hifigan'"),
    pretrained_path: str = typer.Option(..., help="Path to the pretrained .nemo model file"),
    target_path: str = typer.Option(..., help="Path to save the exported ONNX model"),
    quantization: str | None = typer.Option(
        "QInt8", help="Quantization type: 'QInt8', 'QUInt8', 'fp16' or 'None'"
    ),
    dynamic: bool = typer.Option(True, help="Whether to perform dynamic quantization"),
):
    """Main function to export TTS or HiFi-GAN model to ONNX format with optional quantization."""
    if isinstance(quantization, str) and quantization.lower() == "none":
        quant_type = None
    elif quantization == "QInt8":
        quant_type = QuantType.QInt8
    elif quantization == "QUInt8":
        quant_type = QuantType.QUInt8
    elif quantization == "fp16":
        quant_type = quantization
    else:
        raise ValueError(
            f"Invalid quantization type. Choose 'QInt8' or 'QUInt8' or 'fp16'. Received: {quantization}"  # noqa: E501
        )

    if model_type == "tts":
        export_tts_model(pretrained_path, target_path, quant_type)
    elif model_type == "hifigan":
        export_hifigan_model(pretrained_path, target_path, quant_type)
    else:
        raise ValueError("Invalid model type. Choose 'tts' or 'hifigan'.")


if __name__ == "__main__":
    typer.run(main)
