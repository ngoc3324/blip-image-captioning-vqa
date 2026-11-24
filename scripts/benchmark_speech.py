"""Measure STT latency and real-time factor for an audio sample."""

import argparse
import json
from pathlib import Path
import time

from src.speech import SpeechService


def duration_seconds(path: Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('Install speech dependencies with: python -m pip install -e ".[speech]"') from exc
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration / 1_000_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Prefer a reviewed sample close to 10 seconds")
    parser.add_argument("--language", choices=["auto", "vi", "en"], default="auto")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()
    duration = duration_seconds(args.audio)
    service = SpeechService(args.model)
    started = time.perf_counter()
    result = service.transcribe(args.audio, args.language)
    elapsed = time.perf_counter() - started
    report = {
        "audio": str(args.audio),
        "duration_seconds": round(duration, 3),
        "latency_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / duration, 3) if duration else None,
        "detected_language": result.language,
        "transcript": result.text,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    service.cleanup()


if __name__ == "__main__":
    main()
