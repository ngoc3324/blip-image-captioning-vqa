"""Lazy, local speech recognition and synthesis.

Install the ``speech`` extra and download the Piper voice ``.onnx`` files and
their matching ``.onnx.json`` files into ``voices/`` before using TTS.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import atexit
import os
import io
import shutil
import tempfile
import threading
import time
import wave


VOICE_NAMES = {
    # VAIS-1000 is a studio-recorded female Northern Vietnamese voice.
    "vi": "vi_VN-vais1000-medium",
    # The official Piper catalog currently has no English x_low voice.
    "en": "en_US-amy-low",
}
DEFAULT_VOICE_DIR = Path(__file__).resolve().parent.parent / "voices"
GIPFORMER_REPO_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
GIPFORMER_REVISION = "dd9227dcd8705c13f33bdbe59728d546ab94480f"
GIPFORMER_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
_GIPFORMER_LOCK = threading.Lock()


class SpeechError(RuntimeError):
    """A speech dependency, model, audio input, or synthesis failed."""


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str
    latency_ms: float
    audio_seconds: float = 0.0
    model_size: str = "Gipformer 1.5 INT8"


def _language(language: str | None) -> str | None:
    if language is None or language == "auto":
        return None
    code = language.lower().replace("-", "_").split("_")[0]
    if code not in VOICE_NAMES:
        raise ValueError("language must be 'vi', 'en', or 'auto'")
    return code


@lru_cache(maxsize=1)
def _gipformer_recognizer():
    """Load the quantized Vietnamese model once; download its small ONNX files on first use."""
    try:
        import sherpa_onnx
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SpeechError("Gipformer STT requires: python -m pip install -e '.[speech]'") from exc

    try:
        model_paths = {
            name: hf_hub_download(
                repo_id=GIPFORMER_REPO_ID,
                filename=name,
                revision=GIPFORMER_REVISION,
            )
            for name in GIPFORMER_FILES
        }
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model_paths["encoder.int8.onnx"],
            decoder=model_paths["decoder.int8.onnx"],
            joiner=model_paths["joiner.int8.onnx"],
            tokens=model_paths["tokens.txt"],
            num_threads=max(1, min(4, os.cpu_count() or 1)),
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
        )
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"Could not load Gipformer 1.5 INT8: {exc}") from exc


def _decode_audio_compat(audio: str | os.PathLike[str] | bytes):
    """Decode supported audio containers to mono float32 at 16 kHz using PyAV."""
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise SpeechError("Gipformer audio decoding requires PyAV and NumPy") from exc

    source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    try:
        with av.open(source, mode="r") as container:
            if not container.streams.audio:
                raise SpeechError("Audio file has no audio stream")
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"Could not decode audio: {exc}") from exc
    if not chunks:
        raise SpeechError("Audio file contains no decodable samples")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _format_transcript(text: str) -> str:
    """Convert Gipformer's uppercase token output into readable sentence casing."""
    normalized = " ".join(text.split()).lower()
    return normalized[:1].upper() + normalized[1:] if normalized else ""


def transcribe_audio(audio: str | os.PathLike[str] | bytes, language: str | None = "vi") -> str:
    """Transcribe Vietnamese audio locally with Gipformer 1.5 INT8."""
    code = _language(language)
    if code not in (None, "vi"):
        raise ValueError("Gipformer 1.5 recognizes Vietnamese only; use language='vi'")
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("audio is empty")
    elif isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        if not path.is_file():
            raise SpeechError(f"Audio file does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError("audio file is empty")
    else:
        raise TypeError("audio must be a file path or encoded audio bytes")

    try:
        samples = _decode_audio_compat(audio)
        recognizer = _gipformer_recognizer()
        with _GIPFORMER_LOCK:
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, samples)
            recognizer.decode_streams([stream])
        return _format_transcript(stream.result.text)
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"Could not transcribe Vietnamese audio: {exc}") from exc


@lru_cache(maxsize=4)
def _piper_voice(model_path: str):
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise SpeechError("TTS requires the speech extra: pip install -e '.[speech]'") from exc
    try:
        return PiperVoice.load(model_path)
    except Exception as exc:
        raise SpeechError(f"Could not load Piper voice {model_path}: {exc}") from exc


def synthesize_speech(
    text: str,
    language: str = "vi",
    *,
    voice_dir: str | os.PathLike[str] | None = None,
) -> bytes:
    """Synthesize text to WAV bytes using the local Piper Python API."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    code = _language(language)
    if code is None:
        raise ValueError("TTS requires 'vi' or 'en' language")
    directory = Path(voice_dir) if voice_dir is not None else DEFAULT_VOICE_DIR
    model_path = directory / f"{VOICE_NAMES[code]}.onnx"
    config_path = Path(f"{model_path}.json")
    if not model_path.is_file() or not config_path.is_file():
        raise SpeechError(
            f"Piper voice files missing: {model_path} and {config_path}. "
            f"Download them with: python -m piper.download_voices {VOICE_NAMES[code]} "
            f"--download-dir {directory}"
        )
    try:
        with tempfile.TemporaryFile(mode="w+b") as output:
            with wave.open(output, "wb") as wav_file:
                _piper_voice(str(model_path.resolve())).synthesize_wav(text.strip(), wav_file)
            output.seek(0)
            return output.read()
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"Could not synthesize {code} speech: {exc}") from exc


class SpeechService:
    """Speech adapter with temporary WAV paths for UI audio components."""

    def __init__(self, voice_dir: str | os.PathLike[str] | None = None):
        self.voice_dir = voice_dir
        self._directory = Path(tempfile.mkdtemp(prefix="blip-voice-"))
        atexit.register(self.cleanup)

    def transcribe(self, audio_path: str | os.PathLike[str], language: str = "vi") -> TranscriptResult:
        started = time.perf_counter()
        text = transcribe_audio(audio_path, language)
        if not text:
            raise SpeechError("No speech was recognized in the recording")
        return TranscriptResult(text, _language(language) or "vi", (time.perf_counter() - started) * 1000)

    def synthesize(self, text: str, language: str = "vi") -> Path:
        audio = synthesize_speech(text, language, voice_dir=self.voice_dir)
        self._directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self._directory, suffix=".wav", delete=False) as file:
            file.write(audio)
            return Path(file.name)

    def cleanup(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)
