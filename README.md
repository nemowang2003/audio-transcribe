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

Use `uv sync` / `uv run audio-transcribe`. On NixOS, enter `nix develop` first
to provide the native libraries required by Python wheels.
On Nix, `nix run .#cuda` includes CUDA
runtime libraries; `nix run . -- … --device cpu` uses the portable CPU build.
CUDA packaging currently supports x86_64 Linux, including NixOS WSL. The flake
exports `lib.mkPackage { pkgs; python; cudaSupport; }` using the caller's Python
version (Python 3.14 by default). Its `virtualenv` attribute provides a Python
interpreter with the SDK and all runtime dependencies; `sdk` and `pythonSet`
expose the underlying pyproject.nix package and package set.
Nix packaging uses uv2nix to read `pyproject.toml` and `uv.lock`, preferring the
locked wheels. The `build` dependency group also locks `uv_build`; keep its
version range aligned with `[build-system].requires`. Both uv and Nix therefore use the same Python dependency versions
and source hashes. `flake.lock` pins nixpkgs, uv2nix, pyproject.nix, and the build
system packages; these inputs share nixpkgs and pyproject.nix through `follows`.

After changing Python dependencies, run `uv lock` and `nix flake check`.
Update Nix tooling with `nix flake update`, then run `uv lock --check` and
`nix flake check` again. Commit the relevant lockfile changes together.
`nix flake check` checks that `uv.lock` matches the project metadata, runs unit
tests with the locked Nix environment, and imports the native runtime modules.
Other systems can use explicit CPU; other acceleration
backends are not implemented yet.

Run `nix flake check`, `nix build .#cuda`, or source unit tests:
`uv run python -m unittest discover -s tests`.

## Development checks

Development uses Python 3.14 (see `.python-version`) and a uv-managed `.venv`.
Ruff, ty, pre-commit and the build backend are pinned in `uv.lock`. The default
`dev` and `build` groups are installed by both `uv sync` and `uv run`:

```sh
uv sync --locked
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

`pre-commit install` is needed once per clone; subsequent commits run the hooks
automatically. All hooks use the project's locked tools through uv, including
ty. `--all-extras` selects optional runtime features, not dependency groups;
these projects do not need it for development checks.

`.envrc` and direnv are optional conveniences. If Python 3.14 and uv are already
available, the commands above work directly. On NixOS, enter `nix develop`
manually when the interpreter or native libraries for audio wheels are needed.
The shell supplies tools and libraries; it does not create or sync `.venv`.
If Nix garbage collection removes the interpreter behind an old `.venv`, rerun
`uv sync --locked` with the current Python available.

To build with the exact backend installed from the lockfile, use:

```sh
uv build --python .venv/bin/python --no-build-isolation --force-pep517
```

The backend in the `build` group is not automatically the backend used by an
isolated build or uv's bundled fast path. Nix builds independently provide the
locked backend and force PEP 517. Development tools are excluded from production
Nix environments.
