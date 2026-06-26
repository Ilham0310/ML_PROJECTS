"""Audio loading and validation for Age-Emotion Voice Detection."""

from __future__ import annotations

import os
import wave
from typing import Callable, Optional, Tuple

import numpy as np


class VoiceProcessor:
    """Loads voice notes, validates format, and resamples to 22050 Hz."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}
    TARGET_SAMPLE_RATE = 22050
    MIN_DURATION_SEC = 1.0
    UNSUPPORTED_MESSAGE = "Unsupported format. Please upload WAV, MP3, FLAC, or OGG files."
    UNREADABLE_MESSAGE = "Unable to process audio file"
    SHORT_AUDIO_MESSAGE = "Audio too short. Minimum duration is 1 second."

    def __init__(self, loader: Optional[Callable[[str, int], Tuple[np.ndarray, int]]] = None) -> None:
        self._loader = loader

    def validate_format(self, file_path: str) -> tuple[bool, str | None]:
        """Return whether ``file_path`` has a supported audio extension."""

        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.SUPPORTED_FORMATS:
            return True, None
        return False, self.UNSUPPORTED_MESSAGE

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load audio and return ``(audio_signal, 22050)``.

        Uses librosa when installed. A simple PCM-WAV fallback is provided so
        tests and basic WAV files work in minimal environments.
        """

        valid, error = self.validate_format(file_path)
        if not valid:
            raise ValueError(error)
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            raise ValueError(self.UNREADABLE_MESSAGE)

        try:
            if self._loader is not None:
                audio, sample_rate = self._loader(file_path, self.TARGET_SAMPLE_RATE)
            else:
                audio, sample_rate = self._load_with_librosa_or_wave(file_path)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(self.UNREADABLE_MESSAGE) from exc

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError(self.UNREADABLE_MESSAGE)

        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio = self.resample(audio, sample_rate, self.TARGET_SAMPLE_RATE)
            sample_rate = self.TARGET_SAMPLE_RATE

        duration = audio.size / float(sample_rate)
        if duration < self.MIN_DURATION_SEC:
            raise ValueError(self.SHORT_AUDIO_MESSAGE)
        return audio, sample_rate

    def _load_with_librosa_or_wave(self, file_path: str) -> tuple[np.ndarray, int]:
        try:
            import librosa

            audio, sample_rate = librosa.load(file_path, sr=self.TARGET_SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), int(sample_rate)
        except ImportError:
            if os.path.splitext(file_path)[1].lower() != ".wav":
                raise ValueError(
                    "librosa is required to load MP3, FLAC, and OGG files."
                )
            return self._load_pcm_wave(file_path)

    @staticmethod
    def _load_pcm_wave(file_path: str) -> tuple[np.ndarray, int]:
        with wave.open(file_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            raw = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(VoiceProcessor.UNREADABLE_MESSAGE)

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data.astype(np.float32), int(sample_rate)

    @staticmethod
    def resample(
        audio_signal: np.ndarray,
        original_sample_rate: int,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> np.ndarray:
        """Resample a 1-D signal using linear interpolation."""

        audio = np.asarray(audio_signal, dtype=np.float32).reshape(-1)
        if original_sample_rate <= 0 or target_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if audio.size == 0 or original_sample_rate == target_sample_rate:
            return audio.astype(np.float32)
        duration = audio.size / float(original_sample_rate)
        target_size = max(1, int(round(duration * target_sample_rate)))
        old_positions = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        new_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
        return np.interp(new_positions, old_positions, audio).astype(np.float32)
