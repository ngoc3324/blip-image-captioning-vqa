import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
import numpy as np

from src.speech import SpeechService, decode_audio_compat, normalize_speech_audio


class FakeSegment:
    text = " xin chào "


class FakeInfo:
    language = "vi"


class FakeWhisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, *_args, **_kwargs):
        self.kwargs = _kwargs
        return iter([FakeSegment()]), FakeInfo()


class SpeechTests(unittest.TestCase):
    def test_transcription_contract(self):
        service = SpeechService()
        model = FakeWhisper()
        samples = np.sin(np.linspace(0, 20, 16000)).astype(np.float32) * 0.05
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with patch("src.speech._whisper_model", return_value=model), patch(
                "src.speech.decode_audio_compat", return_value=samples
            ):
                result = service.transcribe(audio.name, "auto")
        self.assertEqual(result.text, "xin chào")
        self.assertEqual(result.language, "vi")
        self.assertEqual(result.model_size, "large-v3")
        self.assertFalse(model.kwargs["vad_filter"])
        self.assertEqual(model.kwargs["beam_size"], 8)
        service.audio.cleanup()

    def test_cleanup_replaces_temporary_store(self):
        service = SpeechService()
        old_directory = service.audio.directory
        Path(service.audio.new_path(".mp3")).write_bytes(b"audio")
        service.cleanup()
        self.assertFalse(old_directory.exists())
        self.assertTrue(service.audio.directory.exists())
        service.audio.cleanup()

    def test_pyav_19_metadata_errors_compatibility(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("speech extra is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * 1600)
            audio = decode_audio_compat(path)
        self.assertEqual(audio.dtype.name, "float32")
        self.assertEqual(len(audio), 1600)

    def test_quiet_speech_is_normalized(self):
        samples = np.sin(np.linspace(0, 20, 1600)).astype(np.float32) * 0.02
        normalized = normalize_speech_audio(samples)
        self.assertGreater(float(np.max(np.abs(normalized))), 0.05)

    def test_silence_is_rejected(self):
        with self.assertRaises(RuntimeError):
            normalize_speech_audio(np.zeros(1600, dtype=np.float32))

    def test_edge_no_audio_error_is_reported(self):
        service = SpeechService()
        with patch.object(service, "_edge_tts", side_effect=Exception("No audio was received")):
            with self.assertRaisesRegex(Exception, "No audio was received"):
                service.synthesize("xin chào", "vi", "edge")
        service.audio.cleanup()
