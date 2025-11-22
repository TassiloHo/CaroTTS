import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import typer
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


def generate_onnx_audios(
    fastpitch_onnx: str,
    hifigan_onnx: str,
    tokenizer_config: str,
    manifest: str,
    output_dir: str,
    nrows: int = -1,
):
    """Generate audios using ONNX models for comparison."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "onnx_generated").mkdir(exist_ok=True)
    
    print(f"Loading FastPitch ONNX model from {fastpitch_onnx}...")
    fastpitch_session = ort.InferenceSession(fastpitch_onnx)
    
    print(f"Loading HiFiGAN ONNX model from {hifigan_onnx}...")
    hifigan_session = ort.InferenceSession(hifigan_onnx)
    
    # Load tokenizer configuration
    print(f"Loading tokenizer configuration from {tokenizer_config}...")
    tokenizer_cfg = OmegaConf.load(tokenizer_config).model.text_tokenizer
    tokenizer = instantiate(tokenizer_cfg)
    
    with open(manifest, 'r') as f:
        lines = f.readlines()
        lines_to_process = lines[:nrows] if nrows > 0 else lines
        
        for line in tqdm(lines_to_process, desc="Generating ONNX audios"):
            entry = json.loads(line)
            audio_filepath = entry['audio_filepath']
            text = entry.get('normalized_text', entry.get('text', ''))
            
            audio_name = Path(audio_filepath).stem
            
            # Tokenize text
            tokens = tokenizer.encode(text)
            
            # Prepare inputs for FastPitch
            paces = np.zeros(len(tokens), dtype=np.float32) + 1.0
            pitches = np.zeros(len(tokens), dtype=np.float32)
            
            inputs = {
                "text": np.array([tokens], dtype=np.int64),
                "pace": np.array([paces], dtype=np.float32),
                "pitch": np.array([pitches], dtype=np.float32),
            }
            
            # Generate spectrogram with FastPitch
            spec = fastpitch_session.run(None, inputs)[0]
            
            # Generate audio with HiFiGAN
            gan_inputs = {"spec": spec}
            audio = hifigan_session.run(None, gan_inputs)[0]
            
            # Save audio
            sf.write(
                output_path / "onnx_generated" / f"{audio_name}.wav",
                audio.squeeze(),
                44100
            )
    
    print(f"ONNX generated audios saved to {output_dir}")


def main(
    fastpitch_onnx: str = typer.Option(..., help="Path to FastPitch ONNX model"),
    hifigan_onnx: str = typer.Option(..., help="Path to HiFiGAN ONNX model"),
    tokenizer_config: str = typer.Option(..., help="Path to tokenizer config file"),
    manifests: list[str] = typer.Option(None, "--manifests", help="Paths to manifest files"),
    output_dir: str = typer.Option(..., help="Output directory for ONNX audios"),
    nrows: int = typer.Option(None, help="Number of rows to process from each manifest")
):
    """Generate audios using ONNX models for TTS inference."""
    for manifest in manifests:
        manifest_name = Path(manifest).stem
        output_subdir = Path(output_dir) / manifest_name
        
        print(f"\nProcessing {manifest}...")
        generate_onnx_audios(
            fastpitch_onnx=fastpitch_onnx,
            hifigan_onnx=hifigan_onnx,
            tokenizer_config=tokenizer_config,
            manifest=manifest,
            output_dir=str(output_subdir),
            nrows=nrows
        )


if __name__ == "__main__":
    typer.run(main)