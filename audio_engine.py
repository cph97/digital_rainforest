import numpy as np
import sounddevice as sd
import logging
from pathlib import Path
import librosa
from scipy import signal
import threading
import queue

logger = logging.getLogger(__name__)

class GranularSynth:
    """Multi-layer granular synthesis engine with glitch effects."""
    
    def __init__(self, sample_rate=44100, layer_id=0):
        self.sample_rate = sample_rate
        self.layer_id = layer_id
        self.samples = []
        self.grain_size = 0.5
        self.grain_density = 8.0
        self.playback_rate = 1.0
        self.position = 0
        self.current_sample_idx = 0
        self.sample_position = 0
        self.pitch_shift = 1.0
        self.volume = 0.6
        
        # Glitch parameters
        self.stutter_active = False
        self.stutter_buffer = None
        self.stutter_position = 0
        self.stutter_length = 0

        self.current_is_bird = False
        self._bird_start_events = 0
        
    def load_sample(self, audio_data, is_bird=False):
        """Load audio sample for granular processing."""
        self.samples.append((audio_data, bool(is_bird)))
    
    def generate(self, num_frames, density_mod=1.0, glitch_prob=0.0):
        """Generate granular audio output with glitch effects."""
        if not self.samples:
            return np.zeros(num_frames, dtype=np.float32)
        
        output = np.zeros(num_frames, dtype=np.float32)
        sample, is_bird = self.samples[self.current_sample_idx]
        self.current_is_bird = bool(is_bird)
        
        for i in range(num_frames):
            # Glitch/stutter effect
            if np.random.rand() < glitch_prob and not self.stutter_active:
                # Start stutter
                self.stutter_active = True
                self.stutter_length = int(np.random.uniform(0.02, 0.15) * self.sample_rate)
                self.stutter_position = 0
                stutter_start = max(0, self.sample_position - self.stutter_length)
                self.stutter_buffer = sample[stutter_start:stutter_start + self.stutter_length].copy()
            
            if self.stutter_active:
                # Play stutter buffer
                if self.stutter_position < len(self.stutter_buffer):
                    output[i] = self.stutter_buffer[self.stutter_position] * self.volume * 1.2
                    self.stutter_position += 1
                else:
                    # Loop stutter or end
                    if np.random.rand() < 0.7:
                        self.stutter_position = 0
                    else:
                        self.stutter_active = False
            else:
                # Normal playback with pitch shift
                if self.sample_position < len(sample):
                    # Simple pitch shift via playback rate
                    read_pos = int(self.sample_position * self.pitch_shift)
                    if read_pos < len(sample):
                        output[i] = sample[read_pos] * self.volume
                    self.sample_position += 1
                else:
                    # Switch sample or loop
                    if np.random.rand() < 0.4 * density_mod:
                        self.current_sample_idx = (self.current_sample_idx + 1) % len(self.samples)
                        sample, is_bird = self.samples[self.current_sample_idx]
                        self.current_is_bird = bool(is_bird)
                        if self.current_is_bird:
                            self._bird_start_events += 1
                        # Vary pitch per sample
                        self.pitch_shift = np.random.uniform(0.8, 1.2)
                    self.sample_position = int(np.random.rand() * len(sample) * 0.3)
        
        return output

    def consume_bird_start_events(self):
        n = int(self._bird_start_events)
        self._bird_start_events = 0
        return n


class FMSynth:
    """Enhanced FM synthesis for electronic drones with multiple operators and randomness."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.mod_phase = 0.0
        self.mod_phase2 = 0.0
        self.carrier_freq = 10.0
        self.mod_freq = 3.0
        self.mod_freq2 = 7.0
        self.mod_index = 2.0
        self.mod_index2 = 1.5
        
        # Randomness parameters
        self.freq_drift = 0.0
        self.mod_drift = 0.0
        self.drift_speed = 0.002
        self.last_random_time = 0
        
    def set_frequency(self, freq):
        """Set carrier frequency."""
        self.carrier_freq = freq
    
    def generate(self, num_frames, mod_amount=1.0):
        """Generate FM synthesis output with dual modulators and randomness (vectorized)."""
        # Add slow random drift to frequency
        self.freq_drift += (np.random.randn() * 0.5 - self.freq_drift) * self.drift_speed
        self.mod_drift += (np.random.randn() * 0.3 - self.mod_drift) * self.drift_speed
        
        # Apply drift to carrier and modulator frequencies
        carrier_freq = self.carrier_freq * (1.0 + self.freq_drift * 0.02)  # ±2% drift
        mod_freq = self.mod_freq * (1.0 + self.mod_drift * 0.1)  # ±10% drift
        
        # Vectorized phase generation
        phase_inc = 2 * np.pi * carrier_freq / self.sample_rate
        mod_inc = 2 * np.pi * mod_freq / self.sample_rate
        mod_inc2 = 2 * np.pi * self.mod_freq2 / self.sample_rate
        
        phases = self.phase + np.arange(num_frames) * phase_inc
        mod_phases = self.mod_phase + np.arange(num_frames) * mod_inc
        mod_phases2 = self.mod_phase2 + np.arange(num_frames) * mod_inc2
        
        # Dual modulator FM with random amplitude variation
        amp_variation = 0.8 + np.random.rand() * 0.4  # 0.8 to 1.2
        modulator1 = np.sin(mod_phases) * self.mod_index * mod_amount
        modulator2 = np.sin(mod_phases2) * self.mod_index2 * mod_amount * 0.5
        output = np.sin(phases + modulator1 + modulator2) * 0.008 * amp_variation  # Reduced from 0.03 to 0.008
        
        # Update phases
        self.phase = (phases[-1] + phase_inc) % (2 * np.pi)
        self.mod_phase = (mod_phases[-1] + mod_inc) % (2 * np.pi)
        self.mod_phase2 = (mod_phases2[-1] + mod_inc2) % (2 * np.pi)
        
        return output.astype(np.float32)


class WavetableSynth:
    """Wavetable synthesis with morphing between waveforms."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.frequency = 10.0
        
        # Create wavetables
        table_size = 2048
        self.wavetables = []
        
        # Sine wave
        self.wavetables.append(np.sin(np.linspace(0, 2 * np.pi, table_size, endpoint=False)))
        
        # Saw wave
        self.wavetables.append(np.linspace(-1, 1, table_size, endpoint=False))
        
        # Square wave
        square = np.ones(table_size)
        square[table_size//2:] = -1
        self.wavetables.append(square)
        
        # Triangle wave
        triangle = np.concatenate([
            np.linspace(-1, 1, table_size//2, endpoint=False),
            np.linspace(1, -1, table_size//2, endpoint=False)
        ])
        self.wavetables.append(triangle)
        
        self.current_table = 0
        self.table_morph = 0.0
    
    def generate(self, num_frames, morph_amount=0.0):
        """Generate wavetable output with morphing."""
        output = np.zeros(num_frames, dtype=np.float32)
        table_size = len(self.wavetables[0])
        
        # Morph between wavetables
        table_idx = int(morph_amount * (len(self.wavetables) - 1))
        table_idx = np.clip(table_idx, 0, len(self.wavetables) - 2)
        morph_frac = (morph_amount * (len(self.wavetables) - 1)) - table_idx
        
        for i in range(num_frames):
            # Linear interpolation in wavetable
            idx = int(self.phase * table_size) % table_size
            
            # Morph between two wavetables
            sample1 = self.wavetables[table_idx][idx]
            sample2 = self.wavetables[table_idx + 1][idx]
            output[i] = (sample1 * (1 - morph_frac) + sample2 * morph_frac) * 0.15
            
            self.phase += self.frequency / self.sample_rate
            if self.phase >= 1.0:
                self.phase -= 1.0
        
        return output


class AdditiveSynth:
    """Additive synthesis with evolving harmonics."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.fundamental = 60.0
        self.num_harmonics = 12
        self.phases = np.zeros(self.num_harmonics)
        self.harmonic_amps = np.array([1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.025, 0.008, 0.033])  
    
    def generate(self, num_frames, harmonic_shift=0.0):
        """Generate additive synthesis with evolving harmonics (vectorized)."""
        output = np.zeros(num_frames, dtype=np.float32)
        
        # Process all harmonics in parallel
        for h in range(self.num_harmonics):
            freq = self.fundamental * (h + 1 + harmonic_shift * h * 0.1)
            phase_inc = 2 * np.pi * freq / self.sample_rate
            
            # Generate phase array for this harmonic
            phases = self.phases[h] + np.arange(num_frames) * phase_inc
            
            # Evolving amplitude (simplified for performance)
            amp = self.harmonic_amps[h] * (1.0 + np.sin(phases * 0.1) * 0.3)
            
            # Add this harmonic to output
            output += np.sin(phases) * amp
            
            # Update phase
            self.phases[h] = (phases[-1] + phase_inc) % (2 * np.pi)
        
        return output * 0.05


class NoiseSynth:
    """Filtered noise synthesis for textures."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.filter_freq = 1000.0
        self.resonance = 0.5
        self.lp_state = 0.0
        self.bp_state = 0.0
    
    def generate(self, num_frames, filter_mod=1.0):
        """Generate filtered noise (optimized with simple lowpass)."""
        # Generate white noise
        noise = np.random.randn(num_frames).astype(np.float32) * 0.3
        
        # Modulate filter frequency
        freq = self.filter_freq * filter_mod
        freq = np.clip(freq, 20, self.sample_rate * 0.45)
        
        # Simple one-pole lowpass filter (much faster than state variable)
        alpha = 2.0 * np.pi * freq / self.sample_rate
        alpha = np.clip(alpha, 0, 1)
        
        # Apply filter
        output = np.zeros(num_frames, dtype=np.float32)
        output[0] = noise[0] * alpha + self.lp_state * (1 - alpha)
        
        for i in range(1, num_frames):
            output[i] = noise[i] * alpha + output[i-1] * (1 - alpha)
        
        self.lp_state = output[-1]
        
        return output * 0.2


class PercussiveSynth:
    """Simple percussive synthesis for rhythmic elements."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.trigger_probability = 0.02
        
    def generate(self, num_frames, probability_mod=1.0):
        """Generate percussive hits."""
        output = np.zeros(num_frames, dtype=np.float32)
        
        for i in range(num_frames):
            if np.random.rand() < (self.trigger_probability * probability_mod):
                decay_length = int(0.05 * self.sample_rate)
                freq = np.random.uniform(800, 3000)
                
                for d in range(min(decay_length, num_frames - i)):
                    t = d / self.sample_rate
                    envelope = np.exp(-t * 30)
                    noise = np.random.randn() * 0.5
                    tone = np.sin(2 * np.pi * freq * t) * 0.5
                    
                    if i + d < num_frames:
                        output[i + d] += (noise + tone) * envelope * 0.02
        
        return output


class BitCrusher:
    """Bit depth reduction for lo-fi digital glitch effects."""
    
    def __init__(self):
        self.bit_depth = 16
        self.wet = 0.0
    
    def process(self, input_signal, bit_depth=8):
        """Reduce bit depth of signal."""
        if bit_depth >= 16:
            return input_signal
        
        # Quantize to lower bit depth
        levels = 2 ** bit_depth
        quantized = np.round(input_signal * levels) / levels
        
        return input_signal * (1 - self.wet) + quantized * self.wet


class SampleRateReducer:
    """Sample rate reduction for retro digital artifacts."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.reduction_factor = 1
        self.last_sample = 0.0
    
    def process(self, input_signal, target_rate=44100):
        """Reduce effective sample rate."""
        if target_rate >= self.sample_rate:
            return input_signal
        
        output = np.zeros_like(input_signal)
        reduction = int(self.sample_rate / target_rate)
        
        for i in range(len(input_signal)):
            if i % reduction == 0:
                self.last_sample = input_signal[i]
            output[i] = self.last_sample
        
        return output


class BufferScrambler:
    """Scramble audio buffer for glitch effects."""
    
    def __init__(self):
        self.scramble_probability = 0.0
    
    def process(self, input_signal, probability=0.1, chunk_size=64):
        """Randomly scramble chunks of audio."""
        if probability <= 0:
            return input_signal
        
        output = input_signal.copy()
        num_chunks = len(input_signal) // chunk_size
        
        for i in range(num_chunks):
            if np.random.rand() < probability:
                start = i * chunk_size
                end = start + chunk_size
                # Reverse, repeat, or skip chunks
                effect = np.random.randint(0, 3)
                if effect == 0:  # Reverse
                    output[start:end] = output[start:end][::-1]
                elif effect == 1:  # Repeat previous
                    if i > 0:
                        prev_start = (i - 1) * chunk_size
                        output[start:end] = output[prev_start:prev_start + chunk_size]
                elif effect == 2:  # Silence
                    output[start:end] = 0
        
        return output


class FeedbackDelay:
    """Feedback delay effect for rhythmic echoes."""
    
    def __init__(self, sample_rate=44100, delay_time=0.375, feedback=0.5):
        self.sample_rate = sample_rate
        self.delay_time = delay_time  # seconds
        self.feedback = feedback  # 0.0 to 0.95
        self.wet = 0.3  # dry/wet mix
        
        # Create delay buffer
        self.buffer_size = int(sample_rate * delay_time)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
    
    def process(self, input_signal):
        """Apply feedback delay to input signal."""
        output = np.zeros_like(input_signal)
        
        for i in range(len(input_signal)):
            # Read from delay buffer
            delayed = self.buffer[self.write_pos]
            
            # Mix input with delayed signal (feedback)
            self.buffer[self.write_pos] = input_signal[i] + delayed * self.feedback
            
            # Output is mix of dry and wet
            output[i] = input_signal[i] * (1 - self.wet) + delayed * self.wet
            
            # Advance write position
            self.write_pos = (self.write_pos + 1) % self.buffer_size
        
        return output
    
    def get_delayed_energy(self):
        """Get current energy from delay buffer for visual feedback."""
        # Sample multiple points from delay buffer for average energy
        sample_points = min(512, self.buffer_size // 4)
        samples = []
        for i in range(sample_points):
            idx = (self.write_pos - i) % self.buffer_size
            samples.append(self.buffer[idx])
        
        # Return RMS energy of delayed signal
        if samples:
            return np.sqrt(np.mean(np.array(samples) ** 2))
        return 0.0
    
    def set_delay_time(self, delay_time):
        """Change delay time (creates new buffer)."""
        self.delay_time = delay_time
        new_size = int(self.sample_rate * delay_time)
        if new_size != self.buffer_size:
            self.buffer_size = new_size
            self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
            self.write_pos = 0


class Reverb:
    """Simple Schroeder reverb."""
    
    def __init__(self, sample_rate=44100, room_size=0.5):
        self.sample_rate = sample_rate
        
        comb_delays = [int(sample_rate * d * room_size) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
        self.comb_buffers = [np.zeros(d) for d in comb_delays]
        self.comb_indices = [0] * len(comb_delays)
        
        allpass_delays = [int(sample_rate * d) for d in [0.005, 0.0017]]
        self.allpass_buffers = [np.zeros(d) for d in allpass_delays]
        self.allpass_indices = [0] * len(allpass_delays)
        
        self.feedback = 0.7
        self.wet = 0.3
    
    def process(self, input_signal):
        """Apply reverb to input signal."""
        output = np.zeros_like(input_signal)
        
        for i, sample in enumerate(input_signal):
            comb_sum = 0.0
            
            for j, buf in enumerate(self.comb_buffers):
                idx = self.comb_indices[j]
                comb_out = buf[idx]
                buf[idx] = sample + comb_out * self.feedback
                comb_sum += comb_out
                
                self.comb_indices[j] = (idx + 1) % len(buf)
            
            allpass_out = comb_sum / len(self.comb_buffers)
            
            for j, buf in enumerate(self.allpass_buffers):
                idx = self.allpass_indices[j]
                buf_out = buf[idx]
                buf[idx] = allpass_out + buf_out * 0.5
                allpass_out = buf_out - allpass_out * 0.5
                
                self.allpass_indices[j] = (idx + 1) % len(buf)
            
            output[i] = sample * (1 - self.wet) + allpass_out * self.wet
        
        return output


class AudioEngine:
    """Real-time generative audio engine with multi-layer synthesis and effects."""
    
    def __init__(self, sample_rate=44100, block_size=512, config=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.config = config or {}
        
        self.running = False
        self.stream = None
        
        # Multiple granular layers for rich texture
        self.granular_layers = [
            GranularSynth(sample_rate, layer_id=0),  # Main layer
            GranularSynth(sample_rate, layer_id=1),  # Background layer
            GranularSynth(sample_rate, layer_id=2),  # Accent layer
        ]
        
        # Set different volumes for each layer
        self.granular_layers[0].volume = 0.7  # Main
        self.granular_layers[1].volume = 0.4  # Background
        self.granular_layers[2].volume = 0.5  # Accent
        
        self.fm_synth = FMSynth(sample_rate)
        self.percussion = PercussiveSynth(sample_rate)
        
        # New computational synthesis engines
        self.wavetable = WavetableSynth(sample_rate)
        self.additive = AdditiveSynth(sample_rate)
        self.noise_synth = NoiseSynth(sample_rate)
        
        # Multiple reverbs for depth
        self.reverb_short = Reverb(sample_rate, room_size=0.3)  # Short reverb
        self.reverb_long = Reverb(sample_rate, room_size=0.8)   # Long reverb
        
        self.reverb_short.wet = 0.2
        self.reverb_long.wet = 0.15
        
        # Feedback delay for rhythmic echoes
        self.feedback_delay = FeedbackDelay(sample_rate, delay_time=0.375, feedback=0.55)
        self.feedback_delay.wet = 0.35  # 35% wet for noticeable echoes
        
        # Glitch effects
        self.bit_crusher = BitCrusher()
        self.sample_rate_reducer = SampleRateReducer(sample_rate)
        self.buffer_scrambler = BufferScrambler()
        
        self.master_gain = self.config.get('audio_gain', 0.5)
        
        self.pointcloud_stats = None
        self.visual_feedback = {
            'camera_height': 0.0,
            'camera_speed': 0.0,
            'glitch_energy': 0.0,
            'point_density': 0.0,
            'delay_energy': 0.0,
            'bird_playing': False,
            'bird_spawn_events': 0,
            'bird_energy': 0.0,
            'stutter_active': False,
            'stutter_intensity': 0.0
        }

        self._bird_spawn_phase = 0.0
        self._bird_prev_energy = 0.0
        self._bird_energy_baseline = 0.0
        
        self.output_buffer = np.zeros(block_size, dtype=np.float32)
        self.analysis_queue = queue.Queue(maxsize=10)
        
        logger.info(f"AudioEngine initialized: sr={sample_rate}, block={block_size}, layers=3")
    
    def load_audio_samples(self, audio_folder):
        """Load audio samples from folder and distribute across layers."""
        audio_path = Path(audio_folder)
        
        if not audio_path.exists():
            logger.warning(f"Audio folder not found: {audio_folder}")
            return
        
        extensions = ['*.wav', '*.ogg', '*.mp3']
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(audio_path.glob(ext))
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_folder}")
            return
        
        audio_files = sorted(audio_files)
        logger.info(f"Loading {len(audio_files)} audio files across {len(self.granular_layers)} layers...")
        
        # Load more samples and distribute across layers
        bird_tags = (
            'staurois',
            'nightingale',
            'nightingales',
            'macaws',
            'lorikeets',
            'kingfishers',
            'hummingbirds',
            'birds',
        )

        for idx, audio_file in enumerate(audio_files):
            try:
                audio_data, sr = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)

                stem = audio_file.stem.lower()
                is_bird = any(tag in stem for tag in bird_tags)
                
                # Distribute samples across layers
                # Layer 0: All samples
                # Layer 1: Every other sample (background)
                # Layer 2: Every third sample (accent)
                self.granular_layers[0].load_sample(audio_data, is_bird=is_bird)
                
                if idx % 2 == 0:
                    self.granular_layers[1].load_sample(audio_data, is_bird=is_bird)
                
                if idx % 3 == 0:
                    self.granular_layers[2].load_sample(audio_data, is_bird=is_bird)
                
                logger.info(f"Loaded: {audio_file.name} (is_bird={is_bird})")
            except Exception as e:
                logger.warning(f"Failed to load {audio_file.name}: {e}")
    
    def set_pointcloud_stats(self, stats):
        """Set pointcloud statistics for sonification."""
        self.pointcloud_stats = stats
        
        if stats and 'height_range' in stats:
            height_min, height_max = stats['height_range']
            height_range = height_max - height_min
            
            base_freq = 55.0 + (height_range / 10.0) * 20.0
            self.fm_synth.set_frequency(base_freq)
            
            logger.info(f"Sonification: base freq = {base_freq:.1f} Hz from height range {height_range:.2f}")
    
    def update_visual_feedback(self, camera_pos, camera_speed, glitch_intensity, point_density):
        """Update visual feedback parameters for audio control."""
        if self.pointcloud_stats and 'height_range' in self.pointcloud_stats:
            height_min, height_max = self.pointcloud_stats['height_range']
            height_norm = (camera_pos[1] - height_min) / (height_max - height_min + 1e-6)
            self.visual_feedback['camera_height'] = np.clip(height_norm, 0, 1)
        
        self.visual_feedback['camera_speed'] = np.clip(camera_speed / 10.0, 0, 1)
        self.visual_feedback['glitch_energy'] = np.clip(glitch_intensity, 0, 1)
        self.visual_feedback['point_density'] = np.clip(point_density, 0, 1)
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback with computational synthesis and glitch effects."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        try:
            # Generate multiple granular layers with different parameters
            grain_density_mod = 0.5 + self.visual_feedback['point_density'] * 1.5
            # VERY HIGH glitch probability for frequent glitch effects
            glitch_prob = self.visual_feedback['glitch_energy'] * 0.15  # 15x increase (was 0.01, then 0.05)
            
            # Layer 0: Main layer with HIGH glitch
            layer0 = self.granular_layers[0].generate(frames, density_mod=grain_density_mod, glitch_prob=glitch_prob)
            
            # Layer 1: Background layer with SOME glitch now
            layer1 = self.granular_layers[1].generate(frames, density_mod=grain_density_mod * 0.6, glitch_prob=glitch_prob * 0.5)
            
            # Layer 2: Accent layer with MAXIMUM glitch for dramatic effect
            layer2 = self.granular_layers[2].generate(frames, density_mod=grain_density_mod * 1.2, glitch_prob=glitch_prob * 4.0)  # 4x (was 3x)

            # Per-block event count: how many times we switched into a bird-tagged sample
            try:
                bird_start_events = int(
                    self.granular_layers[0].consume_bird_start_events()
                    + self.granular_layers[1].consume_bird_start_events()
                    + self.granular_layers[2].consume_bird_start_events()
                )
            except Exception:
                bird_start_events = 0

            try:
                self.visual_feedback['bird_playing'] = bool(
                    getattr(self.granular_layers[0], 'current_is_bird', False)
                    or getattr(self.granular_layers[1], 'current_is_bird', False)
                    or getattr(self.granular_layers[2], 'current_is_bird', False)
                )
            except Exception:
                self.visual_feedback['bird_playing'] = False

            # Bird-only energy (RMS) from layers currently playing bird-tagged samples
            try:
                bird_mix = np.zeros(frames, dtype=np.float32)
                if getattr(self.granular_layers[0], 'current_is_bird', False):
                    bird_mix += layer0
                if getattr(self.granular_layers[1], 'current_is_bird', False):
                    bird_mix += layer1
                if getattr(self.granular_layers[2], 'current_is_bird', False):
                    bird_mix += layer2
                bird_energy = float(np.sqrt(np.mean(bird_mix * bird_mix) + 1e-12))
            except Exception:
                bird_energy = 0.0
            self.visual_feedback['bird_energy'] = float(bird_energy)

            # Continuous spawn rate while bird audio is playing.
            # This makes spawns "perfectly" audio reactive (no dependence on analyzer timing).
            # Rate is energy-driven; also add an onset-like burst when energy exceeds a moving baseline.
            dt = float(frames) / float(self.sample_rate)
            continuous_events = 0
            burst_events = 0
            if bool(self.visual_feedback['bird_playing']):
                # Energy -> events/sec mapping
                # Tune here if needed.
                e = float(np.clip((bird_energy - 0.01) / 0.06, 0.0, 1.0))
                rate = 0.8 + 5.0 * e  # 0.8..5.8 events/sec
                self._bird_spawn_phase += rate * dt
                continuous_events = int(self._bird_spawn_phase)
                self._bird_spawn_phase -= float(continuous_events)

                # Moving-average baseline for onset-like bursts inside bird samples
                # (do NOT confuse with bird_start_events; this is within-sample energy onset approximation)
                alpha = float(self.config.get('bird_energy_baseline_alpha', 0.08))
                self._bird_energy_baseline = (1.0 - alpha) * float(self._bird_energy_baseline) + alpha * float(bird_energy)

                onset_threshold = float(self.config.get('bird_energy_onset_threshold', 0.02))
                if float(bird_energy) - float(self._bird_energy_baseline) > onset_threshold:
                    burst_events = int(self.config.get('bird_onset_burst_events', 2))
            else:
                # Reset phase so we don't "carry" spawns between bird sections
                self._bird_spawn_phase = 0.0
                self._bird_energy_baseline = 0.0

            self._bird_prev_energy = float(bird_energy)

            # Final per-block spawn events (NOT cumulative)
            self.visual_feedback['bird_spawn_events'] = int(bird_start_events + continuous_events + burst_events)
            
            # Mix granular layers
            granular_mix = layer0 + layer1 + layer2
            
            # Apply feedback delay to granular samples for rhythmic echoes
            granular_mix = self.feedback_delay.process(granular_mix)
            
            # Get delayed energy for visual feedback synchronization
            delayed_energy = self.feedback_delay.get_delayed_energy()
            self.visual_feedback['delay_energy'] = np.clip(delayed_energy * 5.0, 0, 1)  # Scale for visuals
            
            # Computational synthesis layers (optimized - only run some conditionally)
            # FM synthesis for drone (always on)
            fm_mod = 0.5 + self.visual_feedback['camera_height'] * 1.5
            fm_out = self.fm_synth.generate(frames, mod_amount=fm_mod)
            
            # Wavetable synthesis - only when camera is moving
            if self.visual_feedback['camera_speed'] > 0.1:
                wavetable_morph = self.visual_feedback['camera_speed']
                wavetable_out = self.wavetable.generate(frames, morph_amount=wavetable_morph) * 0.6
            else:
                wavetable_out = 0
            
            # Additive synthesis - only when point density is significant
            if self.visual_feedback['point_density'] > 0.2:
                additive_shift = self.visual_feedback['point_density'] * 2.0
                additive_out = self.additive.generate(frames, harmonic_shift=additive_shift) * 0.5
            else:
                additive_out = 0
            
            # Filtered noise - only when glitch is active
            if self.visual_feedback['glitch_energy'] > 0.3:
                noise_filter_mod = 0.5 + self.visual_feedback['glitch_energy'] * 3.0
                noise_out = self.noise_synth.generate(frames, filter_mod=noise_filter_mod) * 0.4
            else:
                noise_out = 0
            
            # Percussion
            perc_prob = 0.5 + self.visual_feedback['glitch_energy'] * 1.5
            perc_out = self.percussion.generate(frames, probability_mod=perc_prob)
            
            # Mix all synthesis sources (reduced synth contribution)
            synth_mix = fm_out + wavetable_out + additive_out + noise_out + perc_out
            mixed = granular_mix * 0.7 + synth_mix * 0.3
            
            # Apply glitch effects based on glitch energy
            glitch_amount = self.visual_feedback['glitch_energy']
            
            # Track stutter activity across all layers for bird density mapping
            stutter_count = sum(1 for layer in self.granular_layers if layer.stutter_active)
            stutter_active = stutter_count > 0
            stutter_intensity = stutter_count / len(self.granular_layers)  # 0.0 to 1.0
            
            self.visual_feedback['stutter_active'] = stutter_active
            self.visual_feedback['stutter_intensity'] = stutter_intensity
            
            # Detect if glitch is currently active (for bird spawn control) - MUCH LOWER threshold
            glitch_active = glitch_amount > 0.15 or stutter_active
            self.visual_feedback['glitch_active'] = glitch_active
            
            # Bit crushing - VERY LOW threshold for constant glitch presence
            if glitch_amount > 0.1:  # Was 0.2, now triggers almost always
                bit_depth = int(16 - glitch_amount * 12)  # 16 down to 4 bits
                self.bit_crusher.wet = glitch_amount * 0.9  # Maximum wet amount
                mixed = self.bit_crusher.process(mixed, bit_depth=bit_depth)
            
            # Sample rate reduction - VERY LOW threshold
            if glitch_amount > 0.15:  # Was 0.3
                target_rate = int(44100 - glitch_amount * 38000)  # Down to ~6kHz
                mixed = self.sample_rate_reducer.process(mixed, target_rate=target_rate)
            
            # Buffer scrambling - VERY LOW threshold, MAXIMUM aggression
            if glitch_amount > 0.03:  # Was 0.15, now triggers very early
                scramble_prob = glitch_amount * 0.9  # Was 0.5, now maximum aggression
                mixed = self.buffer_scrambler.process(mixed, probability=scramble_prob, chunk_size=128)
            
            # Apply dual reverbs for depth
            # Short reverb for clarity
            short_reverb_amount = 0.2 + self.visual_feedback['camera_speed'] * 0.15
            self.reverb_short.wet = np.clip(short_reverb_amount, 0.1, 0.4)
            reverb_short = self.reverb_short.process(mixed)
            
            # Long reverb for atmosphere
            long_reverb_amount = 0.15 + self.visual_feedback['point_density'] * 0.2
            self.reverb_long.wet = np.clip(long_reverb_amount, 0.05, 0.3)
            reverb_long = self.reverb_long.process(mixed)
            
            # Blend reverbs
            processed = mixed * 0.5 + reverb_short * 0.3 + reverb_long * 0.2
            
            # Soft clipping for warmth
            processed = np.tanh(processed * self.master_gain)
            
            self.output_buffer = processed.copy()
            
            if not self.analysis_queue.full():
                self.analysis_queue.put(processed.copy())
            
            outdata[:, 0] = processed
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata.fill(0)
    
    def start(self):
        """Start audio stream."""
        if self.running:
            return
        
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            self.running = True
            logger.info("Audio engine started")
        except Exception as e:
            logger.error(f"Failed to start audio engine: {e}")
            self.running = False
    
    def stop(self):
        """Stop audio stream."""
        stream = getattr(self, 'stream', None)
        self.running = False

        if stream is None:
            self.stream = None
            logger.info("Audio engine stopped")
            return

        try:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                try:
                    stream.abort()
                except Exception:
                    pass
        finally:
            self.stream = None
            logger.info("Audio engine stopped")
    
    def get_analysis_buffer(self):
        """Get audio buffer for analysis."""
        try:
            return self.analysis_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_master_gain(self, gain):
        """Set master output gain."""
        self.master_gain = np.clip(gain, 0.0, 1.0)
