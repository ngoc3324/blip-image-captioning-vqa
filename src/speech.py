"""Lazy speech-to-text and text-to-speech adapters.

Recorded files are consumed from Gradio's temporary upload area. Generated
audio is kept in a process-scoped temporary directory and removed on reset or
process exit; no audio is copied into the project or dataset.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import atexit
import asyncio
import gc
import io
import os
import shutil
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str
    latency_ms: float
    audio_seconds: float = 0.0
    model_size: str = ""


@lru_cache(maxsize=2)
def _whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError('Chưa cài STT. Chạy: python -m pip install -e ".[speech]"') from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


class TemporaryAudioStore:
    def __init__(self):
        self.directory = Path(tempfile.mkdtemp(prefix="blip-voice-"))
        atexit.register(self.cleanup)

    def new_path(self, suffix: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, path = tempfile.mkstemp(suffix=suffix, dir=self.directory)
        os.close(handle)
        return Path(path)

    def cleanup(self):
        shutil.rmtree(self.directory, ignore_errors=True)


class SpeechService:
    def __init__(self, whisper_size: str = "large-v3"):
        self.whisper_size = whisper_size
        self.audio = TemporaryAudioStore()

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "auto",
        model_size: str | None = None,
    ) -> TranscriptResult:
        if not audio_path:
            raise ValueError("Hãy ghi âm trước khi chuyển thành văn bản.")
        started = time.perf_counter()
        device = "cuda" if _cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        selected_model = model_size or self.whisper_size
        model = _whisper_model(selected_model, device, compute_type)
        audio = normalize_speech_audio(decode_audio_compat(audio_path))
        audio_seconds = len(audio) / 16000
        selected = None if language == "auto" else language
        use_vad = audio_seconds > 45
        prompts = {
            "vi": "Câu hỏi về hình ảnh bằng tiếng Việt, được chép lại đầy đủ và có dấu.",
            "en": "A complete English question about an image.",
        }
        hotwords = {
            "vi": "hình ảnh, mô tả, màu sắc, các đồ vật, người, vị trí, phía trước, phía sau",
            "en": "image, describe, color, object, person, position, foreground, background",
        }
        segments, info = model.transcribe(
            audio,
            language=selected,
            beam_size=8,
            best_of=8,
            patience=1.2,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt=prompts.get(selected),
            hotwords=hotwords.get(selected),
            multilingual=True,
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": 1500, "speech_pad_ms": 500} if use_vad else None,
        )
        completed_segments = list(segments)
        text = " ".join(segment.text.strip() for segment in completed_segments if segment.text.strip()).strip()
        if not text:
            raise RuntimeError("Không nhận diện được lời nói trong đoạn ghi âm.")
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", selected or "auto"),
            latency_ms=(time.perf_counter() - started) * 1000,
            audio_seconds=audio_seconds,
            model_size=selected_model,
        )

    def synthesize(self, text: str, language: str = "vi", backend: str = "edge") -> Path:
        if not text or not text.strip():
            raise ValueError("Không có nội dung để đọc.")
        text = text.strip()
        if backend == "edge":
            return self._edge_tts(text, language)
        if backend == "piper":
            return self._piper_tts(text, language)
        raise ValueError(f"Backend TTS không hợp lệ: {backend}")

    def cleanup(self):
        self.audio.cleanup()
        self.audio = TemporaryAudioStore()

    def _edge_tts(self, text: str, language: str) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError('Chưa cài edge-tts. Chạy: python -m pip install -e ".[speech]"') from exc
        voices = {"vi": "vi-VN-HoaiMyNeural", "en": "en-US-JennyNeural"}
        output = self.audio.new_path(".mp3")

        async def save():
            await edge_tts.Communicate(text, voices.get(language, voices["vi"])).save(str(output))

        asyncio.run(save())
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("Edge TTS không trả về audio")
        return output

    def _piper_tts(self, text: str, language: str) -> Path:
        executable = os.getenv("PIPER_EXECUTABLE")
        model = os.getenv(f"PIPER_{language.upper()}_MODEL")
        if not executable or not model:
            raise RuntimeError("chưa cấu hình PIPER_EXECUTABLE và PIPER_<LANG>_MODEL")
        output = self.audio.new_path(".wav")
        subprocess.run(
            [executable, "--model", model, "--output_file", str(output)],
            input=text,
            text=True,
            encoding="utf-8",
            check=True,
            capture_output=True,
        )
        return output


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def normalize_speech_audio(audio):
    """Remove DC offset and gently raise quiet microphone recordings."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).copy()
    if not samples.size:
        raise RuntimeError("Tệp ghi âm không chứa audio hợp lệ.")
    samples -= float(samples.mean())
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    if peak < 1e-4 or rms < 1e-5:
        raise RuntimeError("Đoạn ghi âm gần như im lặng. Hãy kiểm tra microphone và ghi âm lại.")
    if peak < 0.5:
        samples *= min(0.9 / peak, 4.0)
    return np.clip(samples, -1.0, 1.0)


def decode_audio_compat(audio_path: str | Path, sampling_rate: int = 16000):
    """Decode audio for Whisper, including PyAV versions without metadata_errors.

    faster-whisper 1.2.1 passes ``metadata_errors`` to ``av.open``. PyAV 19
    removed that keyword, so we retry the same decoding pipeline with the
    current API and pass the resulting NumPy array to Whisper.
    """
    try:
        from faster_whisper.audio import decode_audio
        return decode_audio(str(audio_path), sampling_rate=sampling_rate)
    except TypeError as exc:
        if "metadata_errors" not in str(exc):
            raise

    try:
        import av
        import numpy as np
        from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames
    except ImportError as exc:
        raise RuntimeError('Chưa cài bộ giải mã audio. Chạy: python -m pip install -e ".[speech]"') from exc

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None
    with av.open(str(audio_path), mode="r") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
    del resampler
    gc.collect()
    if dtype is None:
        raise RuntimeError("Tệp ghi âm không chứa audio hợp lệ.")
    return np.frombuffer(raw_buffer.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0
