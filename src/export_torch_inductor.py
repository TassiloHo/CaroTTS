from pathlib import Path
import os
import types
from omegaconf import OmegaConf
from models.fastspeech import FastSpeechModel
from nemo.collections.tts.models import HifiGanModel
import typer
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

# Import the specific classes to patch
from nemo.collections.tts.modules.fastpitch import regulate_len, log_to_duration
TOKENIZER_CFG = "/home/tassilo-holtzwart/Projects/Sotalis/CaroTTS/src/configs/train_config_fastpitch.yaml"

def export_tts_model(pretrained_path: str, target_dir: str, device: str = "cpu"):
    """Export the TTS model to TorchInductor AOT."""
    def infer_encoder(
        self,
        text,
        pitch,
        pace,
        speaker=None,
        energy=None,
        volume=None,
        reference_spec=None,
        reference_spec_lens=None,
    ):
        # Calculate speaker embedding
        spk_emb = self.get_speaker_embedding(
            batch_size=text.shape[0],
            speaker=speaker,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_lens,
        )

        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)

        # Predict duration and pitch
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask, conditioning=spk_emb)
        durs_predicted = log_to_duration(
            log_dur=log_durs_predicted, min_dur=self.min_token_duration, max_dur=self.max_token_duration, mask=enc_mask
        )
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb) + pitch
        pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.energy_predictor is not None:
            if energy is not None:
                assert energy.shape[-1] == text.shape[-1], f"energy.shape[-1]: {energy.shape[-1]} != len(text)"
                energy_emb = self.energy_emb(energy)
            else:
                energy_pred = self.energy_predictor(enc_out, enc_mask, conditioning=spk_emb).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1))
            enc_out = enc_out + energy_emb.transpose(1, 2)

        # Expand to decoder time dimension
        len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        volume_extended = None
        if volume is not None:
            volume_extended, _ = regulate_len(durs_predicted, volume.unsqueeze(-1), pace)
            volume_extended = volume_extended.squeeze(-1).float()

        return len_regulated, dec_lens, spk_emb


    def infer_decoder(self, len_regulated, dec_lens, spk_emb):
        # Output FFT
        dec_out, _ = self.decoder(input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb)
        spect = self.proj(dec_out).transpose(1, 2)
        return spect.to(torch.float)

    os.environ['TORCH_LOGS'] = 'dynamic'
    os.environ['TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL'] = 'u0'

    model: FastSpeechModel = FastSpeechModel.restore_from(pretrained_path, map_location=device).eval()
    tokenizer_cfg = OmegaConf.load(TOKENIZER_CFG).model.text_tokenizer
    tokenizer = instantiate(tokenizer_cfg)
    
    test_text = "Hallo. Das ist ein Testsatz."
    tokens = torch.tensor(tokenizer.encode(test_text), dtype=torch.int64).unsqueeze(0).to(device)
    paces = torch.randn_like(tokens, dtype=torch.float32).clamp(-0.2, 2) + 1.0
    pitches = torch.randn_like(tokens, dtype=torch.float32)
    
    exported_module = model.fastpitch

    # export the encoder part
    exported_module.forward = types.MethodType(infer_encoder, exported_module)
    sequence_dim = torch.export.Dim("sequence", min=1, max=1024)
    example_decoder_inputs = exported_module(tokens, pitches, paces)
    print("Tracing model with torch.export...")
    exported = torch.export.export(
        exported_module,
        (tokens, pitches, paces), 
        dynamic_shapes={
            "text": {1: sequence_dim}, 
            "pitch": {1: sequence_dim}, 
            "pace": {1: sequence_dim}
        }
    )
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    target_path = str(Path(target_dir)/"fastpitch_encoder.pt2")
    print("Compiling with TorchInductor...")
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=target_path
    )
    # export the decoder part
    exported_module.forward = types.MethodType(infer_decoder, exported_module)
    sequence_dim = torch.export.Dim("sequence", min=1, max=32768)
    print(example_decoder_inputs[0].shape)
    print("Tracing model with torch.export...")
    exported = torch.export.export(
        exported_module,
        (*example_decoder_inputs,),
        dynamic_shapes=(
            {1: sequence_dim},
            None,
            None
        )
    )
    target_path = str(Path(target_dir)/"fastpitch_decoder.pt2")
    print("Compiling with TorchInductor...")
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=target_path
    )

    return output_path


def export_hifigan_model(pretrained_path: str, target_dir: str, device: str = "cpu"):
    """Export the HiFi-GAN model to TorchInductor AOT."""
    model: HifiGanModel = HifiGanModel.restore_from(pretrained_path, map_location=device).eval()
    model.forward = model.forward_for_export
    example_spectrogram = model.input_example()[0]["spec"]
    time_dim = torch.export.Dim("time", min=1, max=256)
    exported = torch.export.export(model, (example_spectrogram,), dynamic_shapes={"spec": {2: time_dim}})
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(Path(target_dir)/"hifigan.pt2")
    )
    return output_path


def main(
    model_type: str = typer.Option(..., help="Type of model to export: 'tts' or 'hifigan'"),
    pretrained_path: str = typer.Option(..., help="Path to the pretrained .nemo model file"),
    target_dir: str = typer.Option(..., help="Directory to save the exported AOT package"),
    device: str = typer.Option("cuda", help="Device to use for export: 'cpu' or 'cuda'"),
):
    """Main function to export TTS or HiFi-GAN model to TorchInductor AOT format."""
    if model_type == "tts":
        output_path = export_tts_model(pretrained_path, target_dir, device)
        print(f"TTS model exported to: {output_path}")
    elif model_type == "hifigan":
        output_path = export_hifigan_model(pretrained_path, target_dir, device)
        print(f"HiFiGAN model exported to: {output_path}")
    else:
        raise ValueError("Invalid model type. Choose 'tts' or 'hifigan'.")


if __name__ == "__main__":
    typer.run(main)
