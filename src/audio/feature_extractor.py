"""Acoustic feature extraction for voice analysis."""

from __future__ import annotations

import numpy as np


class FeatureExtractor:
    """Extracts a normalized 28-dimensional acoustic feature vector."""

    N_MFCC = 13
    N_CHROMA = 12
    FEATURE_DIM = 28
    MIN_DURATION_SEC = 1.0

    def __init__(self, prefer_librosa: bool = True) -> None:
        self.prefer_librosa = prefer_librosa

    def extract(self, audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract MFCC, spectral, ZCR, and chroma summary features."""

        audio = np.asarray(audio_signal, dtype=np.float32).reshape(-1)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if audio.size / float(sample_rate) < self.MIN_DURATION_SEC:
            raise ValueError("Minimum audio duration of 1 second is required")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio_signal contains non-finite values")

        if self.prefer_librosa:
            try:
                features = self._extract_with_librosa(audio, sample_rate)
            except ImportError:
                features = self._extract_with_numpy(audio, sample_rate)
        else:
            features = self._extract_with_numpy(audio, sample_rate)

        return self.normalize(features)

    def _extract_with_librosa(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.N_MFCC)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=self.N_CHROMA)
        return np.concatenate(
            [
                np.mean(mfcc, axis=1),
                [float(np.mean(centroid))],
                [float(np.mean(bandwidth))],
                [float(np.mean(zcr))],
                np.mean(chroma, axis=1),
            ]
        ).astype(np.float32)

    def _extract_with_numpy(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Small deterministic fallback approximating the required feature groups."""

        frame_count = self.N_MFCC
        segments = np.array_split(audio, frame_count)
        mfcc_like = np.array(
            [float(np.mean(segment)) if segment.size else 0.0 for segment in segments],
            dtype=np.float32,
        )

        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
        energy = float(np.sum(spectrum))
        if energy <= 1e-12:
            centroid = 0.0
            bandwidth = 0.0
        else:
            centroid = float(np.sum(freqs * spectrum) / energy)
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / energy))

        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.int8)))))
        chroma_segments = np.array_split(spectrum, self.N_CHROMA)
        chroma = np.array(
            [float(np.mean(segment)) if segment.size else 0.0 for segment in chroma_segments],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                mfcc_like,
                np.array([centroid, bandwidth, zcr], dtype=np.float32),
                chroma,
            ]
        ).astype(np.float32)

    @staticmethod
    def normalize(features: np.ndarray) -> np.ndarray:
        """Return z-score normalized features with mean 0 and std 1."""

        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.shape != (FeatureExtractor.FEATURE_DIM,):
            raise ValueError(f"Expected feature vector shape (28,), got {vector.shape}")
        mean = float(np.mean(vector))
        std = float(np.std(vector))
        if std < 1e-5:
            normalized = vector - mean
        else:
            normalized = (vector - mean) / std
        return normalized.astype(np.float32)
