"""Whisper transcription CLI and Python SDK."""

import argparse
import json
import sys
from pathlib import Path

from . import models
from .sdk import Segment, Transcriber, Transcript, compute_type_for, transcribe

__all__ = ["Segment", "Transcript", "Transcriber", "transcribe"]


def format_timestamp(seconds: float, separator: str = ".") -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def render(result: Transcript, output_format: str = "txt", timestamps: bool = False) -> str:
    if output_format == "json":
        return (
            json.dumps(
                {"language": result.language, "segments": [vars(s) for s in result.segments]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    if output_format == "srt":
        return "\n".join(
            f"{i}\n{format_timestamp(s.start, ',')} --> {format_timestamp(s.end, ',')}\n{s.text}\n"
            for i, s in enumerate(result.segments, 1)
        )
    lines = [
        f"[{format_timestamp(s.start)} -> {format_timestamp(s.end)}] {s.text}"
        if timestamps
        else s.text
        for s in result.segments
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper; CUDA required unless --device cpu."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audio = commands.add_parser("transcribe")
    audio.add_argument("audio", type=Path)
    audio.add_argument("-o", "--output", type=Path)
    audio.add_argument("--model", default=models.DEFAULT_MODEL)
    audio.add_argument("--model-cache", type=Path, default=None)
    audio.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    audio.add_argument("--compute-type")
    audio.add_argument("--language")
    audio.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    audio.add_argument("--format", choices=("txt", "srt", "json"), default="txt")
    audio.add_argument("--timestamps", action="store_true")
    audio.add_argument("--vad", action="store_true")
    audio.add_argument("--beam-size", type=int, default=5)
    catalog = commands.add_parser("models").add_subparsers(dest="model_command", required=True)
    catalog.add_parser("list")
    for name in ("download", "check"):
        cmd = catalog.add_parser(name)
        cmd.add_argument("model", nargs="?", default=models.DEFAULT_MODEL)
        if name == "download":
            cmd.add_argument("--force", action="store_true")
    cmd = catalog.add_parser("import")
    cmd.add_argument("name")
    cmd.add_argument("path", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "models":
        if args.model_command == "list":
            print(json.dumps(models.registered_models(), indent=2))
        elif args.model_command == "import":
            print(models.register(args.name, args.path))
        else:
            print(
                models.resolve(
                    args.model,
                    download=args.model_command == "download",
                    force=getattr(args, "force", False),
                )
            )
        return 0
    if args.command == "check":
        print(
            f"device={args.device}; compute={compute_type_for(args.device)}; cache={models.cache_dir()}"
        )
        return 0
    result = transcribe(
        args.audio,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        model_cache=args.model_cache,
        language=args.language,
        task=args.task,
        vad_filter=args.vad,
        beam_size=args.beam_size,
    )
    print(
        f"Model: {args.model}; device: {args.device}; detected language: {result.language}",
        file=sys.stderr,
    )
    output = render(result, args.format, args.timestamps)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] not in ("models", "check", "transcribe", "-h", "--help"):
        argv.insert(0, "transcribe")
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"audio-transcribe: {error}", file=sys.stderr)
        return 1
