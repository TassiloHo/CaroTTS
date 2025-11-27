import json
import zipfile
from pathlib import Path

import librosa
import pandas as pd
import requests
import typer
from remotezip import RemoteZip
from tqdm import tqdm

HUI_DOWNLOAD_URL = "https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German/dataset{clean1}/{speaker}{clean2}.zip"
AVAILABLE_SPEAKERS = ["Bernd_Ungerer", "Hokuspokus", "Karlsson", "Eva_K", "Friedrich"]


def download_audio(speaker: str, clean: bool = False, data_folder: str = "data"):
    original_speaker = speaker
    if speaker not in AVAILABLE_SPEAKERS:
        speaker = "others"

    if clean:
        clean_str_1 = "_clean"
        clean_str_2 = "_Clean"
    else:
        clean_str_1 = "_full"
        clean_str_2 = ""
    url = HUI_DOWNLOAD_URL.format(speaker=speaker, clean1=clean_str_1, clean2=clean_str_2)

    # Create speaker-specific folder
    data_folder = Path(data_folder)
    speaker_data_folder = data_folder / original_speaker
    speaker_data_folder.mkdir(parents=True, exist_ok=True)

    if speaker == "others":
        print(f"Downloading files for {original_speaker} from {url} using RemoteZip...")
        with RemoteZip(url) as remote_zip:
            speaker_files = [
                f for f in remote_zip.namelist() if f.startswith(f"{original_speaker}/")
            ]

            if not speaker_files:
                raise ValueError(f"No files found for speaker '{original_speaker}' in the archive")

            print(f"Found {len(speaker_files)} files for {original_speaker}")
            for file in tqdm(speaker_files, desc="Downloading files"):
                remote_zip.extract(file, speaker_data_folder)

        print(f"Downloaded {len(speaker_files)} files to {speaker_data_folder}")
        return None, speaker_data_folder
    else:
        zip_filename = speaker_data_folder / f"{speaker}{clean_str_2}.zip"

        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with Path(zip_filename).open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded to {zip_filename}")
        return zip_filename, speaker_data_folder


def extract_speaker_files(zip_path: Path, speaker_name: str, extract_to: Path):
    """Extract only files for the specified speaker from the zip"""
    print(f"Extracting files for {speaker_name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        speaker_files = [f for f in zip_ref.namelist() if f.startswith(f"{speaker_name}/")]
        for file in tqdm(speaker_files):
            zip_ref.extract(file, extract_to)
    print(f"Extracted {len(speaker_files)} files to {extract_to}")


def create_manifest(speaker_data_folder: Path, speaker_name: str, speaker_prefix: str = "hui"):
    """Create manifest entries from extracted audio files"""
    manifest = []

    # Look for the speaker folder inside the speaker_data_folder
    speaker_folder = speaker_data_folder / speaker_name
    if not speaker_folder.exists():
        raise ValueError(f"Speaker folder {speaker_folder} does not exist")

    content_names = [f for f in speaker_folder.iterdir() if f.is_dir()]
    for content_folder in tqdm(content_names, desc=f"Processing {speaker_name}"):
        metadata_file = content_folder / "metadata.csv"
        if not metadata_file.exists():
            continue

        metadata_df = pd.read_csv(
            metadata_file, header=None, sep="|", names=["filename", "transcript"]
        )
        for _, row in metadata_df.iterrows():
            audio_filepath = (content_folder / "wavs" / (row["filename"] + ".wav")).resolve()
            transcript = row["transcript"]

            if not audio_filepath.exists():
                print(f"File {audio_filepath} does not exist, skipping.")
                continue

            item = {
                "audio_filepath": str(audio_filepath),
                "text": transcript,
                "speaker": f"{speaker_prefix}_{speaker_name}",
                "duration": float(librosa.get_duration(path=audio_filepath)),
                "sr": int(librosa.get_samplerate(audio_filepath)),
                "normalized_text": transcript,
            }
            manifest.append(item)

    return manifest


def split_manifest(manifest: list, train_ratio: float = 0.9):
    """Split manifest into train and validation sets"""
    import random

    random.shuffle(manifest)

    split_idx = int(len(manifest) * train_ratio)
    train_manifest = manifest[:split_idx]
    val_manifest = manifest[split_idx:]

    return train_manifest, val_manifest


def save_manifest(manifest: list, filepath: Path):
    """Save manifest to JSONL file"""
    with Path(filepath).open("w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(manifest)} entries to {filepath}")


if __name__ == "__main__":
    app = typer.Typer()

    @app.command()
    def main(
        speaker: str = typer.Option(..., help="Speaker name to download"),
        data_folder: str = typer.Option("data", help="Data folder path"),
        clean: bool = typer.Option(False, help="Download clean version"),
        train_ratio: float = typer.Option(0.9, help="Train/val split ratio"),
    ):
        zip_path, speaker_data_folder = download_audio(speaker, clean, data_folder)

        if zip_path is not None:
            extract_speaker_files(zip_path, speaker, speaker_data_folder)

        manifest = create_manifest(speaker_data_folder, speaker)

        train_manifest, val_manifest = split_manifest(manifest, train_ratio)

        # Save manifests at the top level of the speaker folder
        save_manifest(train_manifest, speaker_data_folder / "train_manifest.jsonl")
        save_manifest(val_manifest, speaker_data_folder / "val_manifest.jsonl")

        print(f"Done! Train: {len(train_manifest)}, Val: {len(val_manifest)}")

    app()
