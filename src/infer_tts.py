import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
import typer
from nemo.collections.tts.models.fastpitch import FastPitchModel
from nemo.collections.tts.models.hifigan import HifiGanModel
from tqdm import tqdm


def generate_comparison_audios(
    fastpitch_ckpt: str,
    hifigan_ckpt: str,
    manifest: str,
    output_dir: str,
    device: str = "cpu",
    nrows: int = -1,
):
    """Generate audios from predicted mels and true mels for comparison."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "from_predicted_mels").mkdir(exist_ok=True)
    (output_path / "from_true_mels").mkdir(exist_ok=True)
    (output_path / "original").mkdir(exist_ok=True)

    print(f"Loading FastPitch from {fastpitch_ckpt}...")
    fastpitch = FastPitchModel.restore_from(fastpitch_ckpt, map_location=device).eval()

    print(f"Loading HiFiGAN from {hifigan_ckpt}...")
    hifigan = HifiGanModel.restore_from(hifigan_ckpt, map_location=device).eval()

    with Path(manifest).open("r") as f:
        for line in tqdm(f.readlines()[:nrows] if nrows > 0 else f.readlines()):
            entry = json.loads(line)
            audio_filepath = entry["audio_filepath"]
            text = entry.get("text", entry.get("normalized_text", ""))

            audio_name = Path(audio_filepath).stem

            y, sr = librosa.load(audio_filepath, sr=44100)
            sf.write(output_path / "original" / f"{audio_name}.wav", y, sr)
            speaker = None
            if fastpitch.fastpitch.speaker_emb is not None and "speaker" in entry:
                speaker = entry["speaker"]

            with torch.inference_mode():
                if "normalized_text" in entry:
                    parsed_text = fastpitch.parse(entry["normalized_text"], normalize=False)
                else:
                    parsed_text = fastpitch.parse(text)

                predicted_mel = fastpitch.generate_spectrogram(tokens=parsed_text, speaker=speaker)
                audio_from_pred_mel = hifigan.convert_spectrogram_to_audio(spec=predicted_mel)
                sf.write(
                    output_path / "from_predicted_mels" / f"{audio_name}.wav",
                    audio_from_pred_mel.squeeze().cpu().numpy(),
                    44100,
                )
                mels, spec_len = fastpitch.preprocessor(
                    input_signal=torch.tensor(y).unsqueeze(0).to(device),
                    length=torch.tensor(y.shape[0]).unsqueeze(0).to(device),
                )
                audio_from_true_mel = hifigan.convert_spectrogram_to_audio(spec=mels)
                sf.write(
                    output_path / "from_true_mels" / f"{audio_name}.wav",
                    audio_from_true_mel.squeeze().cpu().numpy(),
                    44100,
                )

    print(f"Comparison audios saved to {output_dir}")


def main(
    fastpitch_model_ckpt: str = typer.Option(..., help="Path to FastPitch checkpoint"),
    hifigan_model_ckpt: str = typer.Option(..., help="Path to HiFiGAN checkpoint"),
    manifests: list[str] = typer.Option(None, "--manifests", help="Paths to manifest files"),  # noqa: B008
    output_dir: str = typer.Option(..., help="Output directory for comparison audios"),
    device: str = typer.Option("cpu", help="Device to use (cpu or cuda)"),
    nrows: int = typer.Option(None, help="Number of rows to process from each manifest"),
):
    """Generate comparison audios for TTS inference."""
    for manifest in manifests:
        manifest_name = Path(manifest).stem
        output_subdir = Path(output_dir) / manifest_name

        print(f"\nProcessing {manifest}...")
        generate_comparison_audios(
            fastpitch_ckpt=fastpitch_model_ckpt,
            hifigan_ckpt=hifigan_model_ckpt,
            manifest=manifest,
            output_dir=str(output_subdir),
            device=device,
            nrows=nrows,
        )


if __name__ == "__main__":
    typer.run(main)
