import urllib.request
from pathlib import Path

import typer
import yaml

app = typer.Typer()


def download_if_not_exists(filename, pretrained_dir, base_url):
    """Download model if it doesn't exist."""
    if not filename or not filename.endswith(".nemo"):
        print(f"Skipping invalid filename: {filename}")
        return

    destination = Path(pretrained_dir) / filename

    if destination.exists():
        print(f"✓ Model already exists: {destination}")
        return

    download_url = f"{base_url}/{filename}?download=true"

    print(f"Downloading {filename} from HuggingFace...")
    Path(pretrained_dir).mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(download_url, destination)
        print(f"✓ Successfully downloaded: {destination}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

@app.command()
def main(
    pretrained_dir: str = typer.Option(
        "pretrained_models", help="Directory to store pretrained models"
    ),
    base_url: str = typer.Option(
        "https://huggingface.co/Warholt/Pretrained-TTS-Modules/resolve/main",
        help="Base URL for downloading models",
    ),
    params_file: str = typer.Option("params.yaml", help="Path to params.yaml file"),
):
    """Download pretrained NEMO models if they don't exist."""
    Path(pretrained_dir).mkdir(parents=True, exist_ok=True)

    # Read model filenames from params.yaml
    with Path(params_file).open("r") as f:
        params = yaml.safe_load(f)

    model_filenames = params.get("pretrained_downloads", [])

    if not model_filenames:
        print("No models specified for download. Nothing to do.")
        return

    print(f"Found {len(model_filenames)} model(s) to check:")
    for filename in model_filenames:
        print(f"  - {filename}")
    print()

    for filename in model_filenames:
        download_if_not_exists(filename, pretrained_dir, base_url)


if __name__ == "__main__":
    app()
