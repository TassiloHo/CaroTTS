import json
from pathlib import Path

import torch
import typer
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models.fastpitch import FastPitchModel
from nemo.collections.tts.modules.fastpitch import log_to_duration, regulate_len
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from tqdm import tqdm


class FastPitchEncoderWrapper(torch.nn.Module):
    def __init__(self, fastpitch_module):
        super().__init__()
        self.encoder = fastpitch_module.encoder
        self.duration_predictor = fastpitch_module.duration_predictor
        self.pitch_predictor = fastpitch_module.pitch_predictor
        self.pitch_emb = fastpitch_module.pitch_emb
        self.speaker_emb = (
            fastpitch_module.speaker_emb if hasattr(fastpitch_module, "speaker_emb") else None
        )
        self.min_token_duration = fastpitch_module.min_token_duration
        self.max_token_duration = fastpitch_module.max_token_duration

    def forward(self, text, pitch, pace, speaker=None):
        spk_emb = None
        if self.speaker_emb is not None and speaker is not None:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)

        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)

        log_durs_predicted = self.duration_predictor(enc_out, enc_mask, conditioning=spk_emb)
        durs_predicted = log_to_duration(
            log_dur=log_durs_predicted,
            min_dur=self.min_token_duration,
            max_dur=self.max_token_duration,
            mask=enc_mask,
        )

        pitch_predicted = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb) + pitch
        pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)

        return len_regulated, dec_lens, spk_emb


class FastPitchDecoderWrapper(torch.nn.Module):
    def __init__(self, fastpitch_module):
        super().__init__()
        self.decoder = fastpitch_module.decoder
        self.proj = fastpitch_module.proj

    def forward(self, len_regulated, dec_lens, spk_emb=None):
        dec_out, _ = self.decoder(input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb)
        spect = self.proj(dec_out).transpose(1, 2)
        return spect


class HiFiGANWrapper(torch.nn.Module):
    def __init__(self, hifigan_model):
        super().__init__()
        self.generator = hifigan_model.generator

    def forward(self, spec):
        return self.generator(x=spec)


class FastPitchEncoderDataReader(CalibrationDataReader):
    def __init__(self, model_instance, manifest_path, device="cpu", max_rows=100):
        self.data = []
        self.input_names = ["text", "pitch", "pace"]
        if model_instance.fastpitch.speaker_emb is not None:
            self.input_names.append("speaker")

        print(f"Preparing Encoder Calibration Data from {manifest_path}...")
        with Path(manifest_path).open("r") as f:
            lines = f.readlines()[:max_rows]

        for _, line in enumerate(tqdm(lines)):
            try:
                entry = json.loads(line)
                text_raw = entry.get("text", entry.get("normalized_text", ""))

                parsed = model_instance.parse(text_raw, normalize=False)
                if not isinstance(parsed, torch.Tensor):
                    parsed = torch.tensor(parsed)

                # FORCE 2D SHAPE [1, Time]
                tokens = parsed.long().to(device).reshape(1, -1)

                pitch = torch.zeros_like(tokens, dtype=torch.float32)
                pace = torch.ones_like(tokens, dtype=torch.float32)

                inputs = {
                    "text": tokens.cpu().numpy(),
                    "pitch": pitch.cpu().numpy(),
                    "pace": pace.cpu().numpy(),
                }

                if "speaker" in self.input_names:
                    spk_idx = entry.get("speaker", 0)
                    spk = torch.tensor([spk_idx]).long().to(device)
                    inputs["speaker"] = spk.cpu().numpy()

                self.data.append(inputs)
            except Exception:
                # print(f"Skipping line {i}: {e}")
                continue

        self.iterator = iter(self.data)

    def get_next(self) -> dict:
        return next(self.iterator, None)


class FastPitchDecoderDataReader(CalibrationDataReader):
    def __init__(self, encoder_wrapper, model_instance, manifest_path, device="cpu", max_rows=100):
        self.data = []
        print("Preparing Decoder Calibration Data (running encoder)...")

        encoder_wrapper.eval()

        with Path(manifest_path).open("r") as f:
            lines = f.readlines()[:max_rows]

        with torch.no_grad():
            for _, line in enumerate(tqdm(lines)):
                try:
                    entry = json.loads(line)
                    text_raw = entry.get("text", entry.get("normalized_text", ""))
                    parsed = model_instance.parse(text_raw, normalize=False)
                    if not isinstance(parsed, torch.Tensor):
                        parsed = torch.tensor(parsed)

                    tokens = parsed.long().to(device).reshape(1, -1)
                    pitch = torch.zeros_like(tokens, dtype=torch.float32)
                    pace = torch.ones_like(tokens, dtype=torch.float32)

                    spk = None
                    if model_instance.fastpitch.speaker_emb is not None:
                        spk_idx = entry.get("speaker", 0)
                        spk = torch.tensor([spk_idx]).long().to(device)

                    # Run Encoder
                    len_reg, dec_lens, spk_emb = encoder_wrapper(tokens, pitch, pace, spk)

                    inputs = {
                        "len_regulated": len_reg.cpu().numpy(),
                        "dec_lens": dec_lens.cpu().numpy(),
                    }

                    if spk_emb is not None:
                        inputs["spk_emb"] = spk_emb.cpu().numpy()

                    self.data.append(inputs)
                except Exception:
                    continue

        self.iterator = iter(self.data)
        self.counter = 0

    def get_next(self) -> dict:
        sample = next(self.iterator, None)
        if sample is not None:
            self.counter += 1
            if self.counter % 5 == 0:
                print(f"Calibrating decoder sample {self.counter}...", end="\r")
        return sample


class HifiGanDataReader(CalibrationDataReader):
    def __init__(self, fastpitch_model, manifest_path, device="cpu", max_rows=100):
        self.data = []
        print("Preparing HiFiGAN Calibration Data (running fastpitch)...")

        with Path(manifest_path).open("r") as f:
            lines = f.readlines()[:max_rows]

        fastpitch_model.eval()

        with torch.no_grad():
            for _, line in enumerate(tqdm(lines)):
                try:
                    entry = json.loads(line)
                    text_raw = entry.get("text", entry.get("normalized_text", ""))

                    if "normalized_text" in entry:
                        parsed_text = fastpitch_model.parse(
                            entry["normalized_text"], normalize=False
                        )
                    else:
                        parsed_text = fastpitch_model.parse(text_raw)

                    speaker = entry.get("speaker", None)

                    predicted_mel = fastpitch_model.generate_spectrogram(
                        tokens=parsed_text, speaker=speaker
                    )

                    if predicted_mel.dim() == 2:
                        predicted_mel = predicted_mel.unsqueeze(0)

                    self.data.append({"spec": predicted_mel.cpu().numpy()})
                except Exception:
                    continue

        self.iterator = iter(self.data)
        self.counter = 0

    def get_next(self) -> dict:
        sample = next(self.iterator, None)
        if sample is not None:
            self.counter += 1
            if self.counter % 5 == 0:
                print(f"Calibrating hifigan sample {self.counter}...", end="\r")
        return sample


def detect_dynamic_axes(tensor, name, batch_dim=0):
    """
    Heuristic to detect which dimension is 'Time' based on a dummy tensor.
    Assumes batch is dim 0.
    The dimension that is NOT batch and matches the hidden size is static.
    The other dimension is Time.
    """
    shape = tensor.shape
    dims = len(shape)

    # Common hidden sizes in TTS
    common_hidden_sizes = [256, 384, 512, 1024, 80]

    axes = {0: "batch"}

    for i in range(1, dims):
        val = shape[i]
        if val in common_hidden_sizes:
            pass
        else:
            axes[i] = "time"

    if len(axes) == 1 and dims == 3:
        pass

    return axes


def export_fastpitch_static(
    pretrained_path: str,
    target_dir: str,
    manifest_path: str,
    quantization_type: QuantType,
    device: str = "cpu",
):
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    print("Loading FastPitch model...")
    model = FastPitchModel.restore_from(pretrained_path, map_location=device).eval()

    print("--- Processing Encoder ---")
    encoder_wrapper = FastPitchEncoderWrapper(model.fastpitch).to(device).eval()

    # Dummy inputs for trace
    dummy_text = torch.randint(0, 10, (1, 10), device=device)
    dummy_pitch = torch.randn(1, 10, device=device)
    dummy_pace = torch.ones(1, 10, device=device)

    has_speaker = model.fastpitch.speaker_emb is not None
    dummy_spk = torch.tensor([0], device=device) if has_speaker else None

    enc_args = (dummy_text, dummy_pitch, dummy_pace)
    input_names = ["text", "pitch", "pace"]
    dynamic_axes = {
        "text": {0: "batch", 1: "time"},
        "pitch": {0: "batch", 1: "time"},
        "pace": {0: "batch", 1: "time"},
    }

    if dummy_spk is not None:
        enc_args += (dummy_spk,)
        input_names.append("speaker")
        dynamic_axes["speaker"] = {0: "batch"}

    with torch.no_grad():
        len_reg, dec_lens, spk_emb_out = encoder_wrapper(*enc_args)

    enc_output_names = ["len_regulated", "dec_lens"]
    enc_dynamic_axes_outputs = {
        "len_regulated": detect_dynamic_axes(len_reg, "len_regulated"),
        "dec_lens": {0: "batch"},
    }

    if spk_emb_out is not None:
        enc_output_names.append("spk_emb")
        enc_dynamic_axes_outputs["spk_emb"] = {0: "batch"}

    enc_fp32_path = Path(target_dir).joinpath("fastpitch_encoder_fp32.onnx")
    enc_quant_path = Path(target_dir).joinpath("fastpitch_encoder_quant.onnx")

    print(f"Tracing Encoder... detected output shapes: {len_reg.shape}")
    torch.onnx.export(
        encoder_wrapper,
        enc_args,
        enc_fp32_path,
        input_names=input_names,
        output_names=enc_output_names,
        dynamic_axes=dynamic_axes | enc_dynamic_axes_outputs,
        opset_version=17,
    )

    print("Quantizing Encoder...")
    quant_pre_process(enc_fp32_path, enc_fp32_path, skip_symbolic_shape=True)
    encoder_dr = FastPitchEncoderDataReader(model, manifest_path, device=device)
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    quantize_static(
        enc_fp32_path,
        enc_quant_path,
        calibration_data_reader=encoder_dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=quantization_type,
        activation_type=QuantType.QUInt8,
        # calibrate_method=CalibrationMethod.Entropy,
        extra_options=extra_options,
    )
    print("Encoder Quantization Done.")

    print("--- Processing Decoder ---")
    decoder_wrapper = FastPitchDecoderWrapper(model.fastpitch).to(device).eval()

    dec_args = (len_reg, dec_lens)
    dec_input_names = ["len_regulated", "dec_lens"]

    dec_dynamic = {
        "len_regulated": enc_dynamic_axes_outputs["len_regulated"],
        "dec_lens": {0: "batch"},
    }

    if spk_emb_out is not None:
        dec_args += (spk_emb_out,)
        dec_input_names.append("spk_emb")
        dec_dynamic["spk_emb"] = {0: "batch"}

    dec_fp32_path = Path(target_dir).joinpath("fastpitch_decoder_fp32.onnx")
    dec_quant_path = Path(target_dir).joinpath("fastpitch_decoder_quant.onnx")

    print(f"Tracing Decoder... input shape detected as: {len_reg.shape}")
    print(f"Using dynamic axes: {dec_dynamic['len_regulated']}")

    torch.onnx.export(
        decoder_wrapper,
        dec_args,
        dec_fp32_path,
        input_names=dec_input_names,
        output_names=["spectrogram"],
        dynamic_axes=dec_dynamic | {"spectrogram": {0: "batch", 2: "time"}},
        opset_version=17,
    )

    print("Quantizing Decoder...")
    quant_pre_process(dec_fp32_path, dec_fp32_path, skip_symbolic_shape=True)
    decoder_dr = FastPitchDecoderDataReader(encoder_wrapper, model, manifest_path, device=device)
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    quantize_static(
        dec_fp32_path,
        dec_quant_path,
        calibration_data_reader=decoder_dr,
        per_channel=True,
        quant_format=QuantFormat.QDQ,
        weight_type=quantization_type,
        extra_options=extra_options,
        # calibrate_method=CalibrationMethod.Entropy,
    )

    print(f"\nFastPitch Export Complete. Models saved to {target_dir}")


def export_hifigan_static(
    pretrained_path: str,
    fastpitch_path: str,
    target_dir: str,
    manifest_path: str,
    quantization_type: QuantType,
    device: str = "cpu",
):
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    print("Loading HiFiGAN model...")
    hifi_model = HifiGanModel.restore_from(pretrained_path, map_location=device).eval()
    wrapper = HiFiGANWrapper(hifi_model).to(device).eval()

    dummy_spec = torch.randn(1, 80, 100, device=device)

    fp32_path = Path(target_dir).joinpath("hifigan_fp32.onnx")
    quant_path = Path(target_dir).joinpath("hifigan_quant.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_spec,),
        fp32_path,
        input_names=["spec"],
        output_names=["audio"],
        dynamic_axes={"spec": {0: "batch", 2: "time"}, "audio": {0: "batch", 2: "time"}},
        opset_version=17,
    )

    print("Preparing Calibration Data (Loading FastPitch Helper)...")
    fp_model = FastPitchModel.restore_from(fastpitch_path, map_location=device).eval()

    print("Quantizing HiFiGAN...")
    quant_pre_process(fp32_path, fp32_path, skip_symbolic_shape=True)
    dr = HifiGanDataReader(fp_model, manifest_path, device=device)
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    quantize_static(
        fp32_path,
        quant_path,
        calibration_data_reader=dr,
        per_channel=True,
        quant_format=QuantFormat.QDQ,
        weight_type=quantization_type,
        extra_options=extra_options,
        # calibrate_method=CalibrationMethod.Entropy,
    )
    print(f"\nHiFiGAN Export Complete. Model saved to {quant_path}")


def main(
    model_type: str = typer.Option(..., help="Type of model to export: 'tts' or 'hifigan'"),
    pretrained_path: str = typer.Option(..., help="Path to the pretrained .nemo model file"),
    target_dir: str = typer.Option(..., help="Directory to save the exported ONNX models"),
    manifest_path: str = typer.Option(..., help="Path to manifest.json for calibration data"),
    helper_fastpitch_path: str = typer.Option(
        None, help="Path to FastPitch .nemo (required for HiFiGAN export)"
    ),
    quantization: str = typer.Option("QInt8", help="Quantization type: 'QInt8' or 'QUInt8'"),
    device: str = typer.Option("cpu", help="Device to use for export trace: 'cpu' or 'cuda'"),
):
    if quantization == "QInt8":
        quant_type = QuantType.QInt8
    elif quantization == "QUInt8":
        quant_type = QuantType.QUInt8
    else:
        raise ValueError("Invalid quantization type. Choose 'QInt8' or 'QUInt8'.")

    if model_type == "tts":
        export_fastpitch_static(pretrained_path, target_dir, manifest_path, quant_type, device)
    elif model_type == "hifigan":
        if not helper_fastpitch_path:
            raise ValueError(
                "Exporting HiFiGAN requires --helper-fastpitch-path to generate calibration mels."
            )

        export_hifigan_static(
            pretrained_path, helper_fastpitch_path, target_dir, manifest_path, quant_type, device
        )
    else:
        raise ValueError("Invalid model type. Choose 'tts' or 'hifigan'.")


if __name__ == "__main__":
    typer.run(main)
