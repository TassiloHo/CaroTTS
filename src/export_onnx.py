from pathlib import Path

import typer
from nemo.collections.tts.models import HifiGanModel
from onnxruntime.quantization import QuantType, quantize_dynamic

from models.fastspeech import FastSpeechModel


def export_tts_model(
    pretrained_path: str, target_path: str, quantization: QuantType = QuantType.QInt8
):
    """Export the TTS model to ONNX format with optional quantization."""
    model = FastSpeechModel.restore_from(pretrained_path).eval()
    # make sure target path ends with .onnx
    if not target_path.endswith(".onnx"):
        raise ValueError("Target path must end with .onnx")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    model.export(target_path)
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


def export_hifigan_model(
    pretrained_path: str, target_path: str, quantization: QuantType = QuantType.QInt8
):
    """Export the HiFi-GAN model to ONNX format with optional quantization."""
    model = HifiGanModel.restore_from(pretrained_path).eval()
    # make sure target path ends with .onnx
    if not target_path.endswith(".onnx"):
        raise ValueError("Target path must end with .onnx")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    model.export(target_path)
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


def main(
    model_type: str = typer.Option(..., help="Type of model to export: 'tts' or 'hifigan'"),
    pretrained_path: str = typer.Option(..., help="Path to the pretrained .nemo model file"),
    target_path: str = typer.Option(..., help="Path to save the exported ONNX model"),
    quantization: str = typer.Option("QInt8", help="Quantization type: 'QInt8' or 'QUInt8'"),
):
    """Main function to export TTS or HiFi-GAN model to ONNX format with optional quantization."""
    if quantization == "QInt8":
        quant_type = QuantType.QInt8
    elif quantization == "QUInt8":
        quant_type = QuantType.QUInt8
    else:
        raise ValueError("Invalid quantization type. Choose 'QInt8' or 'QUInt8'.")

    if model_type == "tts":
        export_tts_model(pretrained_path, target_path, quant_type)
    elif model_type == "hifigan":
        export_hifigan_model(pretrained_path, target_path, quant_type)
    else:
        raise ValueError("Invalid model type. Choose 'tts' or 'hifigan'.")


if __name__ == "__main__":
    typer.run(main)
