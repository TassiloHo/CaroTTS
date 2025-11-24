from omegaconf import OmegaConf
import onnxruntime as ort
import torch
import numpy as np
from nemo.collections.tts.models import HifiGanModel
from hydra.utils import instantiate

test_mel = "/home/tassilo-holtzwart/Projects/Sotalis/CaroTTS/data/caromopfen/mel_manifests/mels/jane_eyre_die_waise_von_lowood_43_f000003.npy"

mel_array = np.load(test_mel)

def profile_onnx(mel):
    hifigan_onnx =   "/home/tassilo-holtzwart/Projects/Sotalis/CaroTTS/trained_pipelines/caro/onnx/hifigan.onnx"      
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session = ort.InferenceSession(hifigan_onnx, sess_options=session_options)
    gan_inputs = {"spec": mel.reshape(1, mel.shape[0], mel.shape[1])}
    audio = session.run(None, gan_inputs)[0]
    session.end_profiling()
    

def profile_torch(mel):
    model = HifiGanModel.restore_from(
        "/home/tassilo-holtzwart/Projects/Sotalis/CaroTTS/trained_pipelines/caro/hifigan/checkpoints/default.nemo",
        map_location="cpu",
    ).eval()
    mel_tensor = torch.tensor(mel).unsqueeze(0)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof, torch.inference_mode():
        audio = model(spec=mel_tensor)
    prof.export_chrome_trace("hifigan_torch_profile.json")


def profile_torch_compile(mel):
    model = HifiGanModel.restore_from(
        "/home/tassilo-holtzwart/Projects/Sotalis/CaroTTS/trained_pipelines/caro/hifigan/checkpoints/default.nemo",
        map_location="cpu",
    ).eval()
    
    # Compile the model
    compiled_model = torch.compile(model)
    
    mel_tensor = torch.tensor(mel).unsqueeze(0)
    
    # Warmup run to trigger compilation
    with torch.inference_mode():
        _ = compiled_model(spec=mel_tensor)
    
    # Profile the compiled model
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof, torch.inference_mode():
        audio = compiled_model(spec=mel_tensor)
    prof.export_chrome_trace("hifigan_torch_compile_profile.json")



if __name__ == "__main__":
    profile_onnx(mel_array)
    profile_torch(mel_array) 
    profile_torch_compile(mel_array)