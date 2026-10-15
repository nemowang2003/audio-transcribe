import multiprocessing
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audio_transcribe import models


def register_in_process(target: Path, model: Path, name: str) -> None:
    read = models.registered_models

    def slow_read():
        result = read()
        time.sleep(0.03)
        return result

    with (
        patch.object(models, "registry_path", return_value=target),
        patch.object(models, "registered_models", side_effect=slow_read),
    ):
        models.register(name, model)


class ModelTests(unittest.TestCase):
    def test_import_reuses_files_without_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "existing"
            model.mkdir()
            (model / "model.bin").touch()
            (model / "config.json").write_text("{}")
            with patch(
                "audio_transcribe.models.registry_path", return_value=root / "registry.json"
            ):
                models.register("large-v3", model)
                self.assertEqual(models.resolve("large-v3", download=False), model)
                self.assertEqual(models.registered_models()["large-v3"], str(model))

    def test_reject_incomplete_import(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "not a CTranslate2"):
                models.register("bad", Path(temp))

    def test_cached_models_require_all_model_files(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            (snapshot / "config.json").write_text("{}")
            for download in (False, True):
                with (
                    self.subTest(download=download),
                    patch.object(models, "registered_models", return_value={}),
                    patch.dict(
                        "sys.modules",
                        {
                            "faster_whisper.utils": SimpleNamespace(
                                download_model=lambda *args, **kwargs: str(snapshot)
                            ),
                        },
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "incomplete Whisper model"):
                        models.resolve("large-v3", download=download)
                    (snapshot / "model.bin").touch()
                    self.assertEqual(models.resolve("large-v3", download=download), snapshot)
                    (snapshot / "model.bin").unlink()

    def test_concurrent_imports_preserve_all_registrations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "model.bin").touch()
            (model / "config.json").write_text("{}")
            target = root / "registry.json"
            names = [f"model-{i}" for i in range(12)]
            with ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(register_in_process, target, model, name) for name in names]
                for future in futures:
                    future.result(timeout=30)
            with patch.object(models, "registry_path", return_value=target):
                self.assertEqual(models.registered_models(), dict.fromkeys(names, str(model)))
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_failed_replace_keeps_registry_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "model.bin").touch()
            (model / "config.json").write_text("{}")
            target = root / "registry.json"
            with patch.object(models, "registry_path", return_value=target):
                models.register("existing", model)
                with patch.object(Path, "replace", side_effect=OSError("write failed")):
                    with self.assertRaisesRegex(OSError, "write failed"):
                        models.register("new", model)
                self.assertEqual(models.registered_models(), {"existing": str(model)})
            self.assertEqual(list(root.glob("*.tmp")), [])
