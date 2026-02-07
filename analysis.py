import numpy as np
import librosa
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Real-time audio feature extraction with smoothing."""
    
    def __init__(self, sample_rate=44100, hop_length=512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = 2048
        
        self.smoothing_window = 10
        self.rms_history = deque(maxlen=self.smoothing_window)
        self.centroid_history = deque(maxlen=self.smoothing_window)
        self.onset_history = deque(maxlen=5)
        
        self.rms = 0.0
        self.spectral_centroid = 0.0
        self.onset_strength = 0.0
        self.band_low = 0.0
        self.band_mid = 0.0
        self.band_high = 0.0
        
        self.onset_threshold = 0.3
        self.previous_onset = 0.0
        
        logger.info(f"AudioAnalyzer initialized: sr={sample_rate}, hop={hop_length}")
    
    def analyze_buffer(self, audio_buffer):
        """Analyze audio buffer and extract features."""
        if audio_buffer is None or len(audio_buffer) == 0:
            return self.get_features()
        
        audio = audio_buffer.flatten()
        
        if len(audio) < self.frame_length:
            audio = np.pad(audio, (0, self.frame_length - len(audio)))
        
        rms_value = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_mean = np.mean(rms_value)
        self.rms_history.append(rms_mean)
        self.rms = np.mean(self.rms_history)
        
        try:
            centroid = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate, 
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )[0]
            centroid_mean = np.mean(centroid) / (self.sample_rate / 2)
            self.centroid_history.append(centroid_mean)
            self.spectral_centroid = np.mean(self.centroid_history)
        except:
            pass
        
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        onset_value = np.mean(onset_env) if len(onset_env) > 0 else 0.0
        
        onset_delta = max(0, onset_value - self.previous_onset)
        self.onset_history.append(onset_delta)
        
        if onset_delta > self.onset_threshold:
            self.onset_strength = min(1.0, onset_delta * 2.0)
        else:
            self.onset_strength *= 0.9
        
        self.previous_onset = onset_value
        
        self._analyze_bands(audio)
        
        return self.get_features()
    
    def _analyze_bands(self, audio):
        """Analyze frequency bands (low, mid, high)."""
        try:
            stft = np.abs(librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length))
            
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            low_mask = freqs < 200
            mid_mask = (freqs >= 200) & (freqs < 2000)
            high_mask = freqs >= 2000
            
            self.band_low = np.mean(stft[low_mask]) if np.any(low_mask) else 0.0
            self.band_mid = np.mean(stft[mid_mask]) if np.any(mid_mask) else 0.0
            self.band_high = np.mean(stft[high_mask]) if np.any(high_mask) else 0.0
            
            max_val = max(self.band_low, self.band_mid, self.band_high, 1e-6)
            self.band_low /= max_val
            self.band_mid /= max_val
            self.band_high /= max_val
        except:
            pass
    
    def get_features(self):
        """Return current feature values."""
        return {
            'rms': float(self.rms),
            'spectral_centroid': float(self.spectral_centroid),
            'onset': float(self.onset_strength),
            'band_low': float(self.band_low),
            'band_mid': float(self.band_mid),
            'band_high': float(self.band_high)
        }
    
    def reset(self):
        """Reset all feature histories."""
        self.rms_history.clear()
        self.centroid_history.clear()
        self.onset_history.clear()
        self.rms = 0.0
        self.spectral_centroid = 0.0
        self.onset_strength = 0.0
        self.band_low = 0.0
        self.band_mid = 0.0
        self.band_high = 0.0
