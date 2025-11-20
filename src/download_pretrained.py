import os
import re
import urllib.request
from pathlib import Path
from typing import List
import typer

app = typer.Typer()

def extract_nemo_model_path(arg_string):
    """Extract init_from_nemo_model path from argument string."""
    if not arg_string or "init_from_nemo_model=" not in arg_string:
        return None
    
    match = re.search(r'init_from_nemo_model=([^\s]+)', arg_string)
    if match:
        return match.group(1)
    return None

def download_if_not_exists(model_path, pretrained_dir, base_url):
    """Download model if it doesn't exist."""
    full_path = Path(model_path)
    filename = full_path.name
    
    if not filename.endswith('.nemo'):
        print(f"Skipping non-.nemo file: {filename}")
        return
    
    destination = Path(pretrained_dir) / filename
    
    if destination.exists():
        print(f"✓ Model already exists: {destination}")
        return
    
    # Construct download URL
    download_url = f"{base_url}/{filename}?download=true"
    
    print(f"Downloading {filename} from HuggingFace...")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(download_url, destination)
        print(f"✓ Successfully downloaded: {destination}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

@app.command()
def main(
    arg_strings: List[str] = typer.Argument(
        ..., help="Model initialization argument strings"
    ),
    pretrained_dir: str = typer.Option(
        "../pretrained_models", help="Directory to store pretrained models"
    ),
    base_url: str = typer.Option(
        "https://huggingface.co/Warholt/Pretrained-TTS-Modules/resolve/main",
        help="Base URL for downloading models",
    ),
):
    Path(pretrained_dir).mkdir(parents=True, exist_ok=True)
    """Download pretrained NEMO models if they don't exist."""
    if not arg_strings:
        print("No arguments provided. Nothing to download.")
        return
    
    # Extract model paths from arguments
    model_paths = []
    for arg_string in arg_strings:
        model_path = extract_nemo_model_path(arg_string)
        if model_path:
            model_paths.append(model_path)
    
    if not model_paths:
        print("No pretrained models found in arguments")
        return
    
    print(f"Found {len(model_paths)} pretrained model(s) to check:")
    for path in model_paths:
        print(f"  - {path}")
    print()
    
    # Download each model if needed
    for model_path in model_paths:
        download_if_not_exists(model_path, pretrained_dir, base_url)

if __name__ == "__main__":
    app()
