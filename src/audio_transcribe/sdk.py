"""Reusable transcription API; runtime imports stay lazy."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DEFAULT_MODEL, resolve


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: tuple[Segment, ...]
    language: str


def compute_type_for(device: str = "cuda", compute_type: str | None = None) -> str:
    if device not in ("cuda", "cpu"):
        raise ValueError(f"unsupported device: {device}; use cuda or cpu")
    if device == "cpu":
        return compute_type or "int8"
    import ctranslate2

    try:
        if ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("no CUDA device")
        supported = ctranslate2.get_supported_compute_types("cuda")
        selected = compute_type or "float16"
        if selected not in supported:
            raise RuntimeError(f"unsupported CUDA compute type: {selected}")
    except (RuntimeError, OSError) as error:
        raise RuntimeError(
            f"no usable acceleration found (CUDA): {error}; explicitly use --device cpu to allow CPU"
        ) from error
    return selected


class Transcriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        compute_type: str | None = None,
        model_cache: Path | None = None,
    ):
        compute = compute_type_for(device, compute_type)
        from faster_whisper import WhisperModel

        path = resolve(model, root=model_cache)
        self._model: Any = WhisperModel(str(path), device=device, compute_type=compute)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        self._model = None
        gc.collect()

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        task: str = "transcribe",
        vad_filter: bool = False,
        beam_size: int = 5,
        condition_on_previous_text: bool = True,
    ) -> Transcript:
        path = Path(audio)
        if not path.is_file():
            raise ValueError(f"audio file does not exist: {path}")
        if self._model is None:
            raise RuntimeError("transcriber has been closed")
        segments, info = self._model.transcribe(
            str(path),
            language=language,
            task=task,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
        )
        return Transcript(
            tuple(Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()),
            info.language,
        )


def transcribe(
    audio: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "cuda",
    compute_type: str | None = None,
    model_cache: Path | None = None,
    **options,
) -> Transcript:
    if not Path(audio).is_file():
        raise ValueError(f"audio file does not exist: {audio}")
    with Transcriber(
        model, device=device, compute_type=compute_type, model_cache=model_cache
    ) as engine:
        return engine.transcribe(audio, **options)
