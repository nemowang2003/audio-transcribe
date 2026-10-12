# audio-transcribe

Whisper model management, transcription CLI and Python SDK. Defaults to CUDA;
CPU is allowed only with `--device cpu` (SDK: `device="cpu"`). Missing CUDA fails
before downloading a model. CUDA defaults to float16; CPU defaults to int8.

```sh
audio-transcribe input.m4a --language ja --vad --format srt -o output.srt
audio-transcribe models download large-v3
audio-transcribe models import large-v3 /path/to/ctranslate2/snapshot
audio-transcribe models check large-v3
audio-transcribe models list
audio-transcribe check
```

Models use `${XDG_CACHE_HOME:-~/.cache}/faster-whisper/models`, overridden by
`AUDIO_TRANSCRIBE_CACHE_DIR` or `--model-cache`. Imported model aliases are saved
in `${XDG_CONFIG_HOME:-~/.config}/audio-transcribe/models.json`; files remain in
place. `--model` also accepts a local directory or a Hugging Face model ID.
`models list` lists registered aliases; `models check` also resolves downloaded
models offline. Downloads use the Hugging Face cache, including snapshot revisions.

```python
from audio_transcribe import Transcriber

with Transcriber("large-v3") as engine:
    result = engine.transcribe(
        "input.mp3", language="ja", vad_filter=True, condition_on_previous_text=False
    )
    for segment in result.segments:
        print(segment.start, segment.end, segment.text)
```

`transcribe()` is a convenience function for one file. Reuse a `Transcriber` for a
batch. `close()` releases the model reference; this SDK does not reserve the GPU
or coordinate with other applications. Model runtime imports are lazy.

Use `uv sync` / `uv run audio-transcribe`. On Nix, `nix run .#cuda` includes CUDA
runtime libraries; `nix run . -- … --device cpu` uses the portable CPU build.
CUDA packaging currently supports x86_64 Linux, including NixOS WSL. The flake
exports `lib.mkPackage { pkgs; python; cudaSupport; }` for building the CLI with
the caller's Python version. Other systems can use explicit CPU; other acceleration
backends are not implemented yet.

Run `nix flake check`, `nix build .#cuda`, or source unit tests:
`PYTHONPATH=src uv run --no-project python -m unittest discover -s tests`.

## Locked Nix builds

Nix builds resolve the project's `uv.lock` with uv2nix. Python dependencies,
including optional extras, come from that lock; nixpkgs supplies the interpreter,
system libraries and llama.cpp where applicable. Build backend versions are
pinned by the `pyproject-build-systems` flake input because uv does not lock them.

All three build inputs (`pyproject-nix`, `uv2nix`, `pyproject-build-systems`) may
follow a consuming flake's shared inputs. The standalone flake pins them itself.
`lib.mkPythonSet { pkgs; python; }` exposes the locked Python package scope for
SDK consumers; `lib.mkPackage { pkgs; python; cudaSupport; }` ships the CLI with
its isolated environment. The CLI's `python-env` passthru exposes that environment
for installation and runtime checks. Development shells supply Python and uv;
use `uv sync` / `uv run` for development, and `nix build` for deployment.

`lib.runtimeOverlay { pkgs; }` patches locked CTranslate2 binary linkage and is
reused by ja2zh-srt. `lib.cudaLibraryPath pkgs` supplies the CUDA runtime search path.

PyAV is constrained to `<18` because faster-whisper 1.2.x uses its legacy
`metadata_errors` decoding argument. The installed-environment check decodes a
real WAV sample to catch decoder API incompatibility.
