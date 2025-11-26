# CaroTTS 🥕

<!-- Space for carrot image -->
<p align="center">
  <img src="carotts_logo.png" alt="CaroTTS Logo" width="250"/>
</p>

**Fast, Lightweight Text-to-Speech for German** 

[![Try it on HuggingFace](https://img.shields.io/badge/🤗-Try%20on%20HuggingFace-yellow)](https://huggingface.co/spaces/Warholt/CaroTTS-DE)

---

## 🎯 Problem & Solution

Training and deploying high-quality Text-to-Speech (TTS) models has traditionally required significant computational resources and expertise. CaroTTS addresses this by providing:

- **Lightweight Models**: Non-autoregressive TTS models under 60M parameters
- **CPU-Friendly**: Fast inference on CPUs and even mobile devices
- **Simplified Training**: No pitch information required, reducing data preparation time
- **Full Pipeline**: Complete workflow from data collection to deployment
- **Reproducible**: Automated training pipeline using DVC (Data Version Control)

### How It Works

CaroTTS uses a **FastPitch + HiFi-GAN** architecture:

1. **FastPitch** (Duration Predictor): Predicts the duration for each phoneme/character, enabling parallel mel-spectrogram generation instead of sequential autoregressive prediction. **Crucially, it learns pitch implicitly during training**, eliminating the need for pitch extraction during data preparation.
2. **HiFi-GAN** (Vocoder): Converts mel-spectrograms into high-quality audio waveforms

This non-autoregressive approach allows for **extremely fast inference** while maintaining high audio quality, making it practical for real-time applications on resource-constrained devices.

## ✨ Features

- 🚀 **Fast Inference**: Non-autoregressive architecture for CPU/mobile deployment
- ⚡ **Quick Setup**: No pitch extraction required during data preparation
- 📦 **Export Options**: ONNX and PyTorch Inductor (`.pt2`) export capabilities
- 🔊 **High Quality**: German language TTS with natural-sounding voices
- 🛠️ **Complete Pipeline**: Data preparation → Training → Export → Inference
- 📊 **Reproducible**: DVC-managed experiments with parameter tracking
- 🎓 **Educational**: Learn the full TTS training workflow


## 🚀 Try It Out

Try the trained models online without any setup:

👉 **[CaroTTS Demo on HuggingFace Spaces](https://huggingface.co/spaces/Warholt/CaroTTS-DE)**

## 📋 Prerequisites

### System Requirements

- **OS**: Linux (Windows users should use WSL)
- **RAM**: At least 32 GB
- **GPU**: NVIDIA GPU with at least 8 GB VRAM (current configuration optimized for 12 GB)
- **Python**: 3.11+

### Software Dependencies

- [uv](https://github.com/astral-sh/uv) (Python package manager)
- CUDA-capable GPU drivers

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TassiloHo/CaroTTS.git
   cd CaroTTS
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

## 🎓 Reproducing Training Results

The entire training pipeline is managed with DVC, making it simple to reproduce the results:

```bash
dvc repro
```

This single command will execute the complete pipeline:

1. **Download HUI Dataset**: Fetches and prepares the German speech data
2. **Download Pretrained Models**: Gets base models for transfer learning (Optional)
3. **Train FastPitch**: Trains the duration predictor and spectrogram generator
4. **Generate Mel-Spectrograms**: Creates training data for the vocoder
5. **Train HiFi-GAN**: Trains the neural vocoder
6. **Export to ONNX**: Exports trained models to onnx format for deployment
7. **Export to PT2-Archive**: Uses torch AOTInductor to compile and package for cuda deployments 

### Training Specific Stages

You can also run individual stages:

```bash
# Train only FastPitch for a specific speaker
dvc repro train_fastpitch@caromopfen

# Train only HiFi-GAN for a specific speaker
dvc repro train_hifigan@Karlsson
```

### Customizing Training Parameters

Edit `params.yaml` to adjust training settings:

```yaml
speaker:
  your_speaker:
    fastpitch:
      batch_size: 16          # Adjust for your GPU memory
      max_dur: 13             # Maximum audio duration in seconds, adjust for your GPU memory
      epochs: 50
      peak_lr: 0.001
    hifigan:
      batch_size: 4           # Adjust for your GPU memory
      n_train_segments: 88200 # Length of audio segments in Hz, adjust for your GPU memory 
      epochs: 5
```

**Note**: Current parameters are optimized for a 12 GB GPU. Reduce `batch_size`, `max_dur`, or `n_train_segments` if you encounter out-of-memory errors.

## 📁 Project Structure

```
CaroTTS/
├── src/                          # Source code
│   ├── prepare_hui_data.py      # Data preparation
│   ├── train_fastpitch.py       # FastPitch training
│   ├── train_hifigan.py         # HiFi-GAN training
│   ├── generate_mels.py         # Mel-spectrogram generation
│   ├── export_onnx.py           # ONNX export
│   ├── export_torch_inductor.py # PyTorch Inductor export
│   ├── infer_tts.py             # Inference script
│   └── configs/                 # Training configurations
├── data/                         # Training data
├── trained_pipelines/           # Model checkpoints
├── pretrained_models/           # Base models
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Training parameters
└── pyproject.toml              # Project dependencies
```

## 🔄 Inference

### Using Trained Models

```python
from src.infer_tts import infer_tts

# Generate speech from text
audio = infer_tts(
    text="Hallo, ich bin CaroTTS!",
    fastpitch_model="trained_pipelines/caro/fastpitch/checkpoints/default.nemo",
    hifigan_model="trained_pipelines/caro/hifigan/checkpoints/default.nemo"
)
```

### Exported Models

For production deployment, export to ONNX or PyTorch Inductor:

```bash
# Export to ONNX
python src/export_onnx.py --speaker caromopfen

# Export to PyTorch Inductor (.pt2)
python src/export_torch_inductor.py --speaker caromopfen
```

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, feel free to open an issue or pull request.

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- Built with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- FastPitch architecture from ["FastPitch: Parallel Text-to-speech with Pitch Prediction"](https://arxiv.org/abs/2006.06873)
- HiFi-GAN from ["HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"](https://arxiv.org/abs/2010.05646)
- Data preparation uses the HUI-Audio-Corpus-German dataset

## 📧 Contact

For questions or feedback, please open an issue on GitHub or visit the [HuggingFace Space](https://huggingface.co/spaces/Warholt/CaroTTS-DE).

---

*CaroTTS - Making high-quality German TTS accessible to everyone* 🥕
