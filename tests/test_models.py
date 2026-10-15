import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcribe import models


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
