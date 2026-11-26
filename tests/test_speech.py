"""Offline checks for the optional speech packaging contract."""

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SpeechPackagingTests(unittest.TestCase):
    def test_speech_extra_contains_both_runtimes(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        speech = project["optional-dependencies"]["speech"]
        self.assertTrue(any(dependency.startswith("sherpa-onnx>=") for dependency in speech))
        self.assertTrue(any(dependency.startswith("sherpa-onnx-core>=") for dependency in speech))
        self.assertTrue(any(dependency.startswith("piper-tts") for dependency in speech))

    def test_transcription_defaults_to_vietnamese(self):
        import inspect
        from src.speech import SpeechService
        self.assertEqual(inspect.signature(SpeechService.transcribe).parameters["language"].default, "vi")

    def test_gipformer_rejects_non_vietnamese_stt(self):
        from src.speech import transcribe_audio
        with self.assertRaisesRegex(ValueError, "recognizes Vietnamese only"):
            transcribe_audio(b"audio", "en")

    def test_pyav_resamples_real_wav(self):
        import wave
        import numpy as np
        from src.speech import _decode_audio_compat

        path = ROOT / "tests" / "speech-compatibility.wav"
        samples = (np.sin(np.linspace(0, 40, 3200)) * 8000).astype("<i2")
        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples.tobytes())
            decoded = _decode_audio_compat(path)
            self.assertEqual(decoded.dtype, np.float32)
            self.assertEqual(len(decoded), 3200)
            self.assertGreater(float(np.max(np.abs(decoded))), 0.1)
        finally:
            path.unlink(missing_ok=True)

    def test_documented_model_selection(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        roadmap = (ROOT / "docs/application-roadmap.md").read_text(encoding="utf-8")
        for document in (readme, roadmap):
            with self.subTest(document=document[:20]):
                self.assertIn("Gipformer 1.5", document)
                self.assertIn("vi_VN-vais1000-medium", document)
                self.assertIn("en_US-amy-low", document)


if __name__ == "__main__":
    unittest.main()
