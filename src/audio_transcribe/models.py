"""Whisper models: persistent registrations and shared Hugging Face cache."""

import json
import os
from pathlib import Path

DEFAULT_MODEL = "large-v3"


def cache_dir() -> Path:
    return Path(
        os.environ.get(
            "AUDIO_TRANSCRIBE_CACHE_DIR",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "faster-whisper"
            / "models",
        )
    ).expanduser()


def registry_path() -> Path:
    return (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "audio-transcribe"
        / "models.json"
    )


def registered_models() -> dict[str, str]:
    path = registry_path()
    return json.loads(path.read_text()) if path.exists() else {}


def register(name: str, path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "model.bin").is_file() or not (path / "config.json").is_file():
        raise ValueError(f"not a CTranslate2 Whisper model directory: {path}")
    models = registered_models()
    models[name] = str(path)
    target = registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(models, indent=2) + "\n")
    temporary.replace(target)
    return path


def resolve(
    model: str = DEFAULT_MODEL,
    *,
    root: Path | None = None,
    download: bool = True,
    force: bool = False,
) -> Path:
    source = registered_models().get(model, model)
    path = Path(source).expanduser()
    if path.is_dir():
        if not (path / "model.bin").is_file() or not (path / "config.json").is_file():
            raise ValueError(f"incomplete Whisper model: {path}")
        return path.resolve()
    if source.startswith(("/", "./", "../", "~")):
        raise ValueError(f"Whisper model directory does not exist: {source}")
    from faster_whisper.utils import download_model

    resolved = Path(
        download_model(source, cache_dir=str(root or cache_dir()), local_files_only=not download)
    )
    if force:
        from huggingface_hub import snapshot_download

        repo_id = (
            source
            if "/" in source
            else resolved.parents[1].name.removeprefix("models--").replace("--", "/")
        )
        resolved = Path(
            snapshot_download(
                repo_id,
                cache_dir=str(root or cache_dir()),
                force_download=True,
                allow_patterns=[
                    "config.json",
                    "preprocessor_config.json",
                    "model.bin",
                    "tokenizer.json",
                    "vocabulary.*",
                ],
            )
        )
    return resolved
