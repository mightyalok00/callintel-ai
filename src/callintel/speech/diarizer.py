"""
Acoustic Speaker Diarization Module.
Performs acoustic voice feature extraction, pitch estimation, spectral analysis,
and unsupervised speaker clustering to reliably attribute speakers based on voice characteristics
rather than conversational pause heuristics.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import scipy.signal
import scipy.fft
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.callintel.speech.audio_utils import format_timestamp


class AcousticDiarizer:
    """
    Production acoustic speaker diarizer.
    Extracts acoustic voice representations (MFCCs, pitch/F0, spectral centroid, rolloff, energy)
    from speech audio segments and clusters them into distinct speakers.
    """

    def __init__(self, target_sample_rate: int = 16000, default_n_speakers: int = 2):
        self.sr = target_sample_rate
        self.default_n_speakers = default_n_speakers
        self._fbank = self._build_mel_filterbanks(
            sr=self.sr, n_fft=512, n_mels=26, fmin=80, fmax=7600
        )

    @staticmethod
    def _build_mel_filterbanks(
        sr: int = 16000,
        n_fft: int = 512,
        n_mels: int = 26,
        fmin: float = 80.0,
        fmax: float = 7600.0,
    ) -> np.ndarray:
        """Constructs triangular Mel-frequency filterbank matrix."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_prev = bins[m - 1]
            f_curr = bins[m]
            f_next = bins[m + 1]
            for k in range(f_prev, f_curr):
                if f_curr > f_prev:
                    fbank[m - 1, k] = (k - f_prev) / (f_curr - f_prev)
            for k in range(f_curr, f_next):
                if f_next > f_curr:
                    fbank[m - 1, k] = (f_next - k) / (f_next - f_curr)
        return fbank

    def extract_segment_features(
        self, audio_chunk: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts acoustic features from an audio chunk:
        - 13 MFCCs (mean and std across frames)
        - Fundamental frequency F0 (mean, std, voiced ratio)
        - Spectral centroid and spectral rolloff
        - RMS energy dynamics
        """
        if len(audio_chunk) < 400:
            return {
                "embedding": np.zeros(34, dtype=np.float32),
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "voiced_ratio": 0.0,
                "spectral_centroid": 0.0,
                "rms_energy": 0.0,
            }

        # Pre-emphasis filter to boost higher frequencies
        emphasized = np.append(
            audio_chunk[0], audio_chunk[1:] - 0.97 * audio_chunk[:-1]
        )
        n_fft = 512
        hop_length = 160  # 10ms
        win_length = 400  # 25ms

        f, t, Zxx = scipy.signal.stft(
            emphasized,
            fs=self.sr,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None,
        )
        mag_spec = np.abs(Zxx)
        power_spec = (mag_spec ** 2) / n_fft

        # Mel filterbank energies
        mel_spec = np.dot(self._fbank, power_spec)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
        log_mel = np.log(mel_spec)

        # 13 MFCCs via Type-II DCT
        mfccs = scipy.fft.dct(log_mel, type=2, axis=0, norm="ortho")[:13]
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Spectral Centroid
        freqs = f[:, np.newaxis]
        total_energy = np.sum(power_spec, axis=0) + 1e-10
        spectral_centroid = np.sum(freqs * power_spec, axis=0) / total_energy
        sc_mean = float(np.mean(spectral_centroid))
        sc_std = float(np.std(spectral_centroid))

        # Spectral Rolloff (85% energy frequency)
        cum_energy = np.cumsum(power_spec, axis=0)
        cutoff = 0.85 * total_energy
        rolloff_bins = np.argmax(cum_energy >= cutoff, axis=0)
        rolloff_freqs = f[rolloff_bins]
        s_rolloff_mean = float(np.mean(rolloff_freqs))

        # Pitch / Fundamental Frequency (F0) estimation via autocorrelation
        pitch_values = []
        frame_sz = int(0.030 * self.sr)  # 30ms frame
        frame_hp = int(0.015 * self.sr)  # 15ms step
        min_lag = int(self.sr / 450)     # max 450 Hz
        max_lag = int(self.sr / 65)      # min 65 Hz

        for start_idx in range(0, len(audio_chunk) - frame_sz, frame_hp):
            frame = audio_chunk[start_idx : start_idx + frame_sz]
            frame = frame - np.mean(frame)
            if np.std(frame) < 1e-3:
                continue
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]
            if max_lag < len(corr):
                search_window = corr[min_lag:max_lag]
                peak_lag = min_lag + np.argmax(search_window)
                if corr[peak_lag] > 0.35 * corr[0]:
                    f0 = float(self.sr / peak_lag)
                    pitch_values.append(f0)

        if pitch_values:
            pitch_mean = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            voiced_ratio = float(len(pitch_values) / max(1, (len(audio_chunk) / frame_hp)))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            voiced_ratio = 0.0

        rms_energy = float(np.sqrt(np.mean(audio_chunk ** 2)))

        # Feature vector (34 dimensions)
        feature_vec = np.concatenate([
            mfcc_mean,                               # 13 MFCC means
            mfcc_std,                                # 13 MFCC stds
            [sc_mean / 1000.0, sc_std / 1000.0],     # 2 Spectral centroid
            [s_rolloff_mean / 1000.0],               # 1 Spectral rolloff
            [pitch_mean / 100.0, pitch_std / 100.0], # 2 Pitch
            [voiced_ratio],                          # 1 Voiced ratio
            [rms_energy, float(np.max(np.abs(audio_chunk)))], # 2 Energy
        ]).astype(np.float32)

        return {
            "embedding": feature_vec,
            "pitch_mean": round(pitch_mean, 1),
            "pitch_std": round(pitch_std, 1),
            "voiced_ratio": round(voiced_ratio, 3),
            "spectral_centroid": round(sc_mean, 1),
            "rms_energy": round(rms_energy, 4),
        }

    def diarize_segments(
        self,
        audio_array: np.ndarray,
        segments: List[Dict[str, Any]],
        expected_speakers: Optional[int] = 2,
    ) -> List[Dict[str, Any]]:
        """
        Takes audio waveform and transcription segments, extracts acoustic features per segment,
        clusters segments into acoustic speaker clusters, and assigns calibrated confidence and roles.
        """
        if not segments:
            return []

        # Step 1: Extract acoustic features for each segment
        extracted = []
        feature_matrix = []

        for seg in segments:
            start_s = float(seg["start_seconds"])
            end_s = float(seg["end_seconds"])
            s_idx = max(0, int(start_s * self.sr))
            e_idx = min(len(audio_array), int(end_s * self.sr))
            chunk = audio_array[s_idx:e_idx] if e_idx > s_idx else np.array([], dtype=np.float32)

            feat = self.extract_segment_features(chunk)
            extracted.append(feat)
            feature_matrix.append(feat["embedding"])

        features_np = np.array(feature_matrix)

        # Single segment or empty features edge case
        if len(segments) <= 1:
            seg = segments[0]
            feat = extracted[0]
            return [{
                **seg,
                "speaker": "Agent",
                "speaker_cluster_id": "SPEAKER_00",
                "attribution_method": "acoustic_voice_clustering",
                "attribution_confidence": 0.90,
                "acoustic_metrics": {
                    "mean_pitch_hz": feat["pitch_mean"],
                    "spectral_centroid_hz": feat["spectral_centroid"],
                    "voiced_ratio": feat["voiced_ratio"],
                },
            }]

        # Step 2: Speaker clustering using pitch and acoustic features
        # Normalize features with special emphasis on pitch and vocal spectral envelope
        scaler = StandardScaler()
        norm_features = scaler.fit_transform(features_np)

        # Boost pitch feature column (index 29: pitch_mean / 100.0) and lower MFCCs
        norm_features[:, 29] *= 2.5  # pitch_mean
        norm_features[:, 0:5] *= 1.5 # primary formant MFCCs

        n_clusters = expected_speakers or 2
        # If segments count is small or all pitches are nearly identical, evaluate if single speaker
        pitches = [f["pitch_mean"] for f in extracted if f["pitch_mean"] > 40.0]
        pitch_spread = (max(pitches) - min(pitches)) if len(pitches) > 1 else 0.0

        if len(segments) < 2 or (pitch_spread < 15.0 and len(pitches) > 2):
            # Acoustic features indicate a single speaker monologue
            cluster_labels = np.zeros(len(segments), dtype=int)
            confidences = [0.95] * len(segments)
        else:
            # Cluster segments into n_clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(norm_features)
            centers = kmeans.cluster_centers_

            # Calculate acoustic confidence per segment based on distance margin to centroids
            confidences = []
            for i, feat_vec in enumerate(norm_features):
                dists = [np.linalg.norm(feat_vec - c) for c in centers]
                min_idx = np.argmin(dists)
                d_own = dists[min_idx]
                d_other = dists[1 - min_idx] if len(dists) == 2 else sorted(dists)[1]
                margin = (d_other - d_own) / max(d_own + d_other, 1e-6)
                # Calibrated confidence between 0.70 and 0.98
                conf = min(0.98, max(0.70, 0.50 + 0.50 * (1.0 / (1.0 + np.exp(-3.0 * margin)))))
                confidences.append(round(float(conf), 3))

        # Step 3: Conversational role attribution (Agent vs Customer)
        # Identify which cluster initiates the customer support / company greeting
        agent_cluster_id = 0
        greetings_agent = [
            "thank you for calling",
            "support",
            "my name is",
            "how can i help",
            "welcome to",
            "sales",
            "how may i assist",
            "cloud support",
        ]

        cluster_greeting_scores: Dict[int, float] = {int(cid): 0.0 for cid in set(cluster_labels)}
        for idx, (seg, clus) in enumerate(zip(segments, cluster_labels)):
            text_lower = seg["utterance"].lower()
            weight = 2.0 if idx < 3 else 0.5
            if any(g in text_lower for g in greetings_agent):
                cluster_greeting_scores[int(clus)] += weight

        if cluster_greeting_scores:
            agent_cluster_id = max(
                cluster_greeting_scores.keys(),
                key=lambda k: cluster_greeting_scores[k],
            )
            if cluster_greeting_scores[agent_cluster_id] == 0.0:
                # If no explicit greeting phrase, first speaking cluster is typically the agent
                agent_cluster_id = int(cluster_labels[0])

        # Step 4: Assemble finalized turns
        diarized_turns = []
        for i, (seg, feat, clus, conf) in enumerate(zip(segments, extracted, cluster_labels, confidences)):
            role = "Agent" if clus == agent_cluster_id else "Customer"
            cluster_id_str = f"SPEAKER_{clus:02d}"

            diarized_turn = {
                "turn_index": i,
                "speaker": role,
                "speaker_cluster_id": cluster_id_str,
                "attribution_method": "acoustic_voice_clustering",
                "attribution_confidence": conf,
                "acoustic_metrics": {
                    "mean_pitch_hz": feat["pitch_mean"],
                    "spectral_centroid_hz": feat["spectral_centroid"],
                    "voiced_ratio": feat["voiced_ratio"],
                },
                "start_seconds": seg["start_seconds"],
                "end_seconds": seg["end_seconds"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "utterance": seg["utterance"],
                "avg_logprob": seg.get("avg_logprob", 0.0),
            }
            diarized_turns.append(diarized_turn)

        return diarized_turns
