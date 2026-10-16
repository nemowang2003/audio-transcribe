import contextlib
import io
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import audio_transcribe as cli
from audio_transcribe.sdk import Transcriber, compute_type_for


class TranscribeTests(unittest.TestCase):
    def test_runtime_decodes_and_resamples_audio_without_loading_model(self):
        from faster_whisper.audio import decode_audio

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "录音.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(array("h", [1000] * 8000).tobytes())
            samples = decode_audio(str(path), sampling_rate=16000)

        self.assertEqual(samples.shape, (16000,))
        self.assertGreater(float(samples.max()), 0)

    def test_default_requires_cuda_and_never_falls_back(self):
        args = cli.build_parser().parse_args(["transcribe", "a.wav"])
        self.assertEqual(args.device, "cuda")
        with patch.dict(
            "sys.modules", {"ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 0)}
        ):
            with self.assertRaisesRegex(RuntimeError, "explicitly use --device cpu"):
                compute_type_for()
        self.assertEqual(compute_type_for("cpu"), "int8")

    def test_explicit_compute_type_is_checked(self):
        runtime = SimpleNamespace(
            get_cuda_device_count=lambda: 1, get_supported_compute_types=lambda _: {"float16"}
        )
        with patch.dict("sys.modules", {"ctranslate2": runtime}):
            self.assertEqual(compute_type_for(), "float16")
            with self.assertRaisesRegex(RuntimeError, "unsupported CUDA compute type"):
                compute_type_for("cuda", "int8")

    def test_sdk_consumes_segments_and_releases_model(self):
        calls = {}

        class Model:
            def __init__(self, model, **kwargs):
                calls["init"] = (model, kwargs)

            def transcribe(self, path, **kwargs):
                calls["options"] = kwargs
                return iter([SimpleNamespace(start=0, end=1.25, text=" 日本語 ")]), SimpleNamespace(
                    language="ja"
                )

        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "录音.wav"
            audio.touch()
            with (
                patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=Model)}),
                patch("audio_transcribe.sdk.resolve", return_value=Path("/model")),
            ):
                with Transcriber(device="cpu") as engine:
                    result = engine.transcribe(
                        audio, language="ja", condition_on_previous_text=False
                    )
                self.assertIsNone(engine._model)
                self.assertEqual(calls["init"][1]["compute_type"], "int8")
                self.assertFalse(calls["options"]["condition_on_previous_text"])
                self.assertEqual(result.segments[0].text, "日本語")
                self.assertIn("00:00:01,250", cli.render(result, "srt"))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    engine.transcribe(audio)

    def test_missing_file_does_not_load_runtime(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            cli.transcribe("missing.wav")

    def test_cli_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "text.txt"
            args = cli.build_parser().parse_args(
                ["transcribe", "a.wav", "-o", str(output), "--timestamps"]
            )
            result = cli.Transcript((cli.Segment(0, 1.25, "你好"),), "zh")
            with (
                patch("audio_transcribe.transcribe", return_value=result),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(cli.run(args), 0)
            self.assertEqual(output.read_text(), "[00:00:00.000 -> 00:00:01.250] 你好\n")
