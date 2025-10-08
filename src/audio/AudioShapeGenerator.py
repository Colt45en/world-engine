# Audio-Driven Mathematical Shape Generator
# CONSOLIDATED FROM SCATTERED UNTITLED FILES
# Features: Real-time FFT analysis, multiple shape modes, WebGL integration

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from typing import Tuple, List, Optional
import json
import time

class AudioShapeGenerator:
    """Advanced audio-to-visual shape generation system with multiple parametric shapes."""

    def __init__(self, blocksize: int = 1024, samplerate: int = 44100):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.audio_buffer = np.zeros(blocksize)
        self.spectrum_history = []
        self.max_history = 30  # frames

        # EMA smoothing parameters
        self.ema_alpha = 0.2
        self.L_smooth = 0.0
        self.centroid_smooth = 0.0
        self.flux_smooth = 0.0

        # Normalization ranges (rolling min/max)
        self.norm_ranges = {
            'loudness': {'min': 0.0, 'max': 1.0},
            'centroid': {'min': 0.0, 'max': self.samplerate/2},
            'flux': {'min': 0.0, 'max': 1.0}
        }

        # Shape generation parameters
        self.shape_modes = ['heart', 'rose', 'spiral', 'dragon', 'lissajous', 'circle', 'mandala', 'galaxy', 'flower']
        self.current_shape = 0
        self.bpm = 120.0
        self.beat_phase = 0.0

        # Audio stream
        self.stream = None

        # Visualization
        self.fig = None
        self.ax = None
        self.line = None
        self.animation = None

        # Enhanced visualization features
        self.color_mode = 'hsv'  # hsv, spectrum, beat
        self.trail_mode = False
        self.trail_data = []
        self.max_trail_length = 20

    def audio_callback(self, indata, frames, time, status):
        """Real-time audio input callback."""
        if status:
            print(f'Audio status: {status}')
        self.audio_buffer = indata[:, 0]  # mono channel

    def start_audio_stream(self):
        """Initialize and start the audio input stream."""
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            blocksize=self.blocksize,
            samplerate=self.samplerate
        )
        self.stream.start()
        print(f"🎤 Audio stream started: {self.samplerate}Hz, {self.blocksize} samples")

    def stop_audio_stream(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("🛑 Audio stream stopped")

    def extract_features(self, buf: np.ndarray) -> Tuple[float, float, float]:
        """Extract audio features: loudness, spectral centroid, spectral flux."""
        # RMS loudness
        L = np.sqrt(np.mean(buf**2))

        # Spectrum analysis
        spectrum = np.abs(np.fft.rfft(buf))
        freqs = np.fft.rfftfreq(len(buf), 1/self.samplerate)

        # Spectral centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        else:
            centroid = 0

        # Spectral flux (change from previous frame)
        flux = 0.0
        if len(self.spectrum_history) > 0:
            prev_spectrum = self.spectrum_history[-1]
            if len(spectrum) == len(prev_spectrum):
                flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                flux = flux / np.sum(spectrum) if np.sum(spectrum) > 0 else 0

        # Update spectrum history
        self.spectrum_history.append(spectrum)
        if len(self.spectrum_history) > self.max_history:
            self.spectrum_history.pop(0)

        return L, centroid, flux

    def smooth_and_normalize(self, L: float, centroid: float, flux: float) -> Tuple[float, float, float]:
        """Apply EMA smoothing and normalization."""
        # EMA smoothing
        self.L_smooth = self.ema_alpha * L + (1 - self.ema_alpha) * self.L_smooth
        self.centroid_smooth = self.ema_alpha * centroid + (1 - self.ema_alpha) * self.centroid_smooth
        self.flux_smooth = self.ema_alpha * flux + (1 - self.ema_alpha) * self.flux_smooth

        # Update normalization ranges
        self.norm_ranges['loudness']['min'] = min(self.norm_ranges['loudness']['min'], self.L_smooth)
        self.norm_ranges['loudness']['max'] = max(self.norm_ranges['loudness']['max'], self.L_smooth)

        self.norm_ranges['centroid']['min'] = min(self.norm_ranges['centroid']['min'], self.centroid_smooth)
        self.norm_ranges['centroid']['max'] = max(self.norm_ranges['centroid']['max'], self.centroid_smooth)

        self.norm_ranges['flux']['min'] = min(self.norm_ranges['flux']['min'], self.flux_smooth)
        self.norm_ranges['flux']['max'] = max(self.norm_ranges['flux']['max'], self.flux_smooth)

        # Normalize to 0-1 range
        L_norm = self.normalize_value(self.L_smooth, 'loudness')
        centroid_norm = self.normalize_value(self.centroid_smooth, 'centroid')
        flux_norm = self.normalize_value(self.flux_smooth, 'flux')

        return L_norm, centroid_norm, flux_norm

    def normalize_value(self, value: float, feature: str) -> float:
        """Normalize a value to 0-1 range based on rolling min/max."""
        min_val = self.norm_ranges[feature]['min']
        max_val = self.norm_ranges[feature]['max']

        if max_val - min_val < 1e-10:  # Avoid division by zero
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)

    def update_beat_phase(self):
        """Update beat phase for rhythm synchronization."""
        beat_duration = 60.0 / self.bpm  # seconds per beat
        self.beat_phase = (time.time() % beat_duration) / beat_duration

    # Shape generation methods (CONSOLIDATED from scattered untitled files)
    def generate_heart(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate parametric heart shape driven by audio features."""
        t = np.linspace(0, 2*np.pi, 800)
        scale = 0.5 + 2*L_norm

        # Heart equation with audio modulation
        x = 16 * np.sin(t)**3 * scale
        y = (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) * scale

        # Add beat phase rotation
        rotation = 2*np.pi * self.beat_phase
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r

        return x_rot, y_rot

    def generate_rose(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rose curve with petals determined by spectral centroid."""
        n = max(1, int(2 + round(6*centroid_norm)))  # petals from centroid
        a = 1 + 3*L_norm  # amplitude from loudness

        theta = np.linspace(0, 2*np.pi, 800)
        r = a * np.cos(n*theta)

        # Add flux-based perturbation
        perturbation = 0.1 * flux_norm * np.sin(10*theta)
        r += perturbation

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y

    def generate_spiral(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate logarithmic spiral."""
        theta = np.linspace(0, 6*np.pi, 1000)
        r = (0.1 + L_norm) * theta

        # Centroid affects spiral tightness
        theta_mod = theta * (0.5 + centroid_norm)

        x = r * np.cos(theta_mod)
        y = r * np.sin(theta_mod)

        return x, y

    def generate_dragon(self, L_norm: float, centroid_norm: float, flux_norm: float, iterations: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dragon curve approximation."""
        # Simple iterative dragon curve
        points = [0+0j, 1+0j]

        for i in range(min(iterations, int(8 + 4*centroid_norm))):
            new_points = [points[0]]
            for j in range(len(points)-1):
                mid = (points[j] + points[j+1]) / 2
                rot = (points[j+1] - points[j]) * 1j
                mid = points[j] + rot/2
                new_points.extend([mid, points[j+1]])
            points = new_points

        pts = np.array(points) * (0.5 + 3*L_norm)

        # Add flux-based jitter
        jitter = 0.05 * flux_norm * (np.random.random(len(pts)) - 0.5)
        pts += jitter

        return pts.real, pts.imag

    def generate_lissajous(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Lissajous curves with audio-driven frequency ratios."""
        t = np.linspace(0, 2*np.pi, 1000)

        a = 3
        b = int(2 + round(centroid_norm * 8))  # frequency ratio from centroid

        phase_offset = 2*np.pi * self.beat_phase

        x = np.sin(a*t + phase_offset) * (1 + 3*L_norm)
        y = np.cos(b*t) * (1 + 3*L_norm)

        # Add flux-based amplitude modulation
        amp_mod = 1 + 0.2 * flux_norm * np.sin(10*t)
        x *= amp_mod
        y *= amp_mod

        return x, y

    def generate_circle(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate circle with audio-driven radius and perturbations."""
        theta = np.linspace(0, 2*np.pi, 500)
        r = 1 + 3*L_norm

        # Add centroid-based oval deformation
        x = r * np.cos(theta) * (1 + 0.3*centroid_norm)
        y = r * np.sin(theta) * (1 - 0.3*centroid_norm)

        # Add flux-based edge perturbations
        perturbation = 0.1 * flux_norm * np.sin(20*theta)
        r_perturbed = r + perturbation

        x = r_perturbed * np.cos(theta) * (1 + 0.3*centroid_norm)
        y = r_perturbed * np.sin(theta) * (1 - 0.3*centroid_norm)

        return x, y

    def generate_mandala(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mandala-like pattern with multiple concentric shapes."""
        theta = np.linspace(0, 2*np.pi, 1000)
        x_total, y_total = np.array([]), np.array([])

        # Multiple rings with different frequencies
        num_rings = int(3 + 5*centroid_norm)
        for i in range(num_rings):
            r_base = (i + 1) * (0.5 + 2*L_norm)
            frequency = (i + 1) * (1 + int(10*flux_norm))

            # Ring with audio-driven modulation
            r = r_base + 0.3 * r_base * np.sin(frequency * theta + 2*np.pi*self.beat_phase)

            x_ring = r * np.cos(theta)
            y_ring = r * np.sin(theta)

            x_total = np.concatenate([x_total, x_ring, [np.nan]])
            y_total = np.concatenate([y_total, y_ring, [np.nan]])

        return x_total, y_total

    def generate_galaxy(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate galaxy spiral pattern."""
        theta = np.linspace(0, 8*np.pi, 1500)

        # Multiple spiral arms
        num_arms = int(2 + 4*centroid_norm)
        x_total, y_total = np.array([]), np.array([])

        for arm in range(num_arms):
            arm_offset = (2*np.pi * arm) / num_arms
            theta_arm = theta + arm_offset

            # Exponential spiral with audio modulation
            r = (0.1 + 2*L_norm) * np.exp(0.2 * theta_arm)

            # Add flux-based perturbation
            r += 0.2 * flux_norm * np.sin(20*theta_arm + 2*np.pi*self.beat_phase)

            x_arm = r * np.cos(theta_arm)
            y_arm = r * np.sin(theta_arm)

            x_total = np.concatenate([x_total, x_arm, [np.nan]])
            y_total = np.concatenate([y_total, y_arm, [np.nan]])

        return x_total, y_total

    def generate_flower(self, L_norm: float, centroid_norm: float, flux_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate flower pattern with petals."""
        theta = np.linspace(0, 2*np.pi, 1000)

        # Number of petals from centroid
        num_petals = int(4 + 12*centroid_norm)

        # Base radius with loudness scaling
        base_radius = 1 + 3*L_norm

        # Petal shape using multiple frequency components
        r = base_radius * (1 + 0.5*np.cos(num_petals*theta))
        r += 0.3 * base_radius * np.cos(2*num_petals*theta)

        # Add flux-based dynamics
        r += 0.2 * base_radius * flux_norm * np.sin(num_petals*theta + 2*np.pi*self.beat_phase)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y

    def generate_current_shape(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the currently selected shape driven by real-time audio."""
        self.update_beat_phase()

        L, centroid, flux = self.extract_features(self.audio_buffer)
        L_norm, centroid_norm, flux_norm = self.smooth_and_normalize(L, centroid, flux)

        shape_generators = [
            self.generate_heart,
            self.generate_rose,
            self.generate_spiral,
            self.generate_dragon,
            self.generate_lissajous,
            self.generate_circle,
            self.generate_mandala,
            self.generate_galaxy,
            self.generate_flower
        ]

        return shape_generators[self.current_shape](L_norm, centroid_norm, flux_norm)

    def setup_visualization(self):
        """Set up matplotlib visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.line, = self.ax.plot([], [], lw=2)

        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        self.ax.grid(True, alpha=0.3)

        self.title = self.ax.set_title(f"Shape: {self.shape_modes[self.current_shape].title()}",
                                       fontsize=16, color='white')

        # Key press handler
        def on_key(event):
            if event.key in map(str, range(1, 10)):
                shape_idx = int(event.key) - 1
                if shape_idx < len(self.shape_modes):
                    self.current_shape = shape_idx
                    self.title.set_text(f"Shape: {self.shape_modes[self.current_shape].title()}")
            elif event.key == ' ':  # spacebar to cycle shapes
                self.current_shape = (self.current_shape + 1) % len(self.shape_modes)
                self.title.set_text(f"Shape: {self.shape_modes[self.current_shape].title()}")
            elif event.key == 't':  # toggle trail mode
                self.trail_mode = not self.trail_mode
                if not self.trail_mode:
                    self.trail_data.clear()
                print(f"Trail mode: {'ON' if self.trail_mode else 'OFF'}")
            elif event.key == 'c':  # cycle color mode
                color_modes = ['hsv', 'spectrum', 'beat']
                current_idx = color_modes.index(self.color_mode)
                self.color_mode = color_modes[(current_idx + 1) % len(color_modes)]
                print(f"Color mode: {self.color_mode}")
            elif event.key == 'r':  # reset normalization ranges
                self.norm_ranges = {
                    'loudness': {'min': 0.0, 'max': 1.0},
                    'centroid': {'min': 0.0, 'max': self.samplerate/2},
                    'flux': {'min': 0.0, 'max': 1.0}
                }
                print("Normalization ranges reset")
            elif event.key == 's':  # save current frame
                self.save_current_frame()
            elif event.key == 'e':  # export data
                self.export_shape_data()
            elif event.key == 'h':  # show help
                self.show_help()

        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def update_visualization(self, frame):
        """Update visualization frame with enhanced features."""
        try:
            x, y = self.generate_current_shape()

            # Get audio features for coloring and effects
            L, centroid, flux = self.extract_features(self.audio_buffer)
            L_norm, centroid_norm, flux_norm = self.smooth_and_normalize(L, centroid, flux)

            # Trail mode
            if self.trail_mode:
                self.trail_data.append((x.copy(), y.copy()))
                if len(self.trail_data) > self.max_trail_length:
                    self.trail_data.pop(0)

                # Clear previous plots
                self.ax.clear()
                self.ax.set_xlim(-15, 15)
                self.ax.set_ylim(-15, 15)
                self.ax.set_aspect('equal')
                self.ax.set_facecolor('black')
                self.ax.grid(True, alpha=0.3)

                # Draw trail with fading alpha
                for i, (trail_x, trail_y) in enumerate(self.trail_data):
                    alpha = (i + 1) / len(self.trail_data) * 0.8
                    color = self.get_audio_color(L_norm, centroid_norm, flux_norm, alpha)
                    self.ax.plot(trail_x, trail_y, color=color, linewidth=1 + 3*alpha*L_norm, alpha=alpha)

                self.title = self.ax.set_title(
                    f"Shape: {self.shape_modes[self.current_shape].title()} (Trail Mode)",
                    fontsize=16, color='white'
                )
            else:
                # Normal mode
                self.line.set_data(x, y)
                color = self.get_audio_color(L_norm, centroid_norm, flux_norm)
                self.line.set_color(color)
                self.line.set_linewidth(1 + 4 * L_norm)  # thickness from loudness

            return (self.line,) if not self.trail_mode else ()
        except Exception as e:
            print(f"Visualization error: {e}")
            return (self.line,) if not self.trail_mode else ()

    def get_audio_color(self, L_norm: float, centroid_norm: float, flux_norm: float, alpha: float = 1.0):
        """Get color based on audio features and current color mode."""
        if self.color_mode == 'hsv':
            # HSV color mapping
            hue = centroid_norm
            saturation = 0.8 + 0.2 * flux_norm
            value = 0.6 + 0.4 * L_norm
            color = plt.cm.hsv(hue)

        elif self.color_mode == 'spectrum':
            # Spectral color mapping
            wavelength = 380 + (centroid_norm * 400)  # 380-780 nm range
            color = self.wavelength_to_rgb(wavelength)

        elif self.color_mode == 'beat':
            # Beat-synchronized color cycling
            beat_color = (self.beat_phase + flux_norm) % 1.0
            color = plt.cm.rainbow(beat_color)

        # Apply alpha if provided
        if alpha < 1.0:
            color = (*color[:3], alpha)

        return color

    def wavelength_to_rgb(self, wavelength: float) -> Tuple[float, float, float]:
        """Convert wavelength to RGB color (simplified approximation)."""
        if wavelength < 380 or wavelength > 780:
            return (0.5, 0.5, 0.5)  # Gray for out of range

        if wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif wavelength < 510:
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif wavelength < 645:
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0

        return (r, g, b)

    def start_visualization(self):
        """Start the real-time visualization."""
        self.setup_visualization()

        self.animation = FuncAnimation(
            self.fig,
            self.update_visualization,
            interval=50,  # 20 FPS
            blit=True
        )

        plt.show()

    def export_shape_data(self, filename: str = None) -> dict:
        """Export current shape data as JSON."""
        if filename is None:
            filename = f"audio_shape_{self.shape_modes[self.current_shape]}_{int(time.time())}.json"

        x, y = self.generate_current_shape()

        data = {
            'timestamp': time.time(),
            'shape_mode': self.shape_modes[self.current_shape],
            'audio_features': {
                'loudness': float(self.L_smooth),
                'centroid': float(self.centroid_smooth),
                'flux': float(self.flux_smooth)
            },
            'normalization_ranges': self.norm_ranges,
            'points': {
                'x': x.tolist(),
                'y': y.tolist()
            },
            'parameters': {
                'bpm': self.bpm,
                'beat_phase': self.beat_phase,
                'samplerate': self.samplerate,
                'blocksize': self.blocksize
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Shape data exported to {filename}")
        return data

    def save_current_frame(self, filename: str = None) -> str:
        """Save the current visualization as an image."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"audio_shape_{self.shape_modes[self.current_shape]}_{timestamp}.png"

        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight',
                           facecolor='black', edgecolor='none')
            print(f"🖼️ Frame saved as {filename}")
        else:
            print("❌ No visualization to save")

        return filename

    def get_audio_stats(self) -> dict:
        """Get current audio analysis statistics."""
        L, centroid, flux = self.extract_features(self.audio_buffer)
        L_norm, centroid_norm, flux_norm = self.smooth_and_normalize(L, centroid, flux)

        return {
            'raw_features': {
                'loudness': float(L),
                'spectral_centroid': float(centroid),
                'spectral_flux': float(flux)
            },
            'normalized_features': {
                'loudness_norm': float(L_norm),
                'centroid_norm': float(centroid_norm),
                'flux_norm': float(flux_norm)
            },
            'current_shape': self.shape_modes[self.current_shape],
            'beat_phase': float(self.beat_phase),
            'bpm': float(self.bpm),
            'color_mode': self.color_mode,
            'trail_mode': self.trail_mode,
            'normalization_ranges': self.norm_ranges
        }

    def show_help(self):
        """Display help information."""
        help_text = """
        🎵 Audio Shape Generator - Help
        ================================

        KEYBOARD CONTROLS:
        1-9      : Select shape by number
        Space    : Cycle through shapes
        T        : Toggle trail mode
        C        : Cycle color modes
        R        : Reset normalization ranges
        S        : Save current frame as PNG
        E        : Export shape data as JSON
        H        : Show this help

        SHAPES AVAILABLE:
        1. Heart      : Parametric heart curve
        2. Rose       : Mathematical rose pattern
        3. Spiral     : Logarithmic spiral
        4. Dragon     : Dragon curve fractal
        5. Lissajous  : Lissajous figure
        6. Circle     : Audio-modulated circle
        7. Mandala    : Concentric ring pattern
        8. Galaxy     : Multi-arm spiral galaxy
        9. Flower     : Petal-based flower shape

        AUDIO MAPPING:
        - Loudness    : Shape size and line thickness
        - Centroid    : Shape complexity and color hue
        - Flux        : Perturbations and dynamics
        - Beat Phase  : Rotation and rhythm sync
        """
        print(help_text)

    def run(self):
        """Main execution loop."""
        print("🎵 Enhanced Audio Shape Generator v2.0")
        print("=" * 50)
        print("CONTROLS:")
        print("  1-9      : Select shape by number")
        print("  Space    : Cycle through shapes")
        print("  T        : Toggle trail mode")
        print("  C        : Cycle color modes (HSV/Spectrum/Beat)")
        print("  R        : Reset normalization ranges")
        print("  S        : Save current frame as PNG")
        print("  E        : Export shape data as JSON")
        print("  H        : Show detailed help")
        print("  Ctrl+C   : Quit")
        print("=" * 50)
        print("SHAPES:")
        for i, shape in enumerate(self.shape_modes):
            print(f"  {i+1}: {shape.title()}")
        print("=" * 50)

        try:
            self.start_audio_stream()

            # Add real-time stats display
            import threading
            def stats_display():
                while self.stream and self.stream.active:
                    L, centroid, flux = self.extract_features(self.audio_buffer)
                    L_norm, centroid_norm, flux_norm = self.smooth_and_normalize(L, centroid, flux)

                    stats = f"\rL:{L_norm:.2f} C:{centroid_norm:.2f} F:{flux_norm:.2f} | " \
                           f"Shape: {self.shape_modes[self.current_shape]} | " \
                           f"BPM: {self.bpm:.1f} | Beat: {self.beat_phase:.2f}"
                    print(stats, end="", flush=True)
                    time.sleep(0.1)

            stats_thread = threading.Thread(target=stats_display, daemon=True)
            stats_thread.start()

            self.start_visualization()

        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.stop_audio_stream()


# Usage example and CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio-Driven Mathematical Shape Generator")
    parser.add_argument('--blocksize', type=int, default=1024, help="Audio buffer size")
    parser.add_argument('--samplerate', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--bpm', type=float, default=120.0, help="BPM for rhythm sync")
    parser.add_argument('--shape', type=int, default=0, help="Initial shape (0-8)")
    parser.add_argument('--trail', action='store_true', help="Start with trail mode enabled")
    parser.add_argument('--color-mode', choices=['hsv', 'spectrum', 'beat'], default='hsv',
                       help="Initial color mode")

    args = parser.parse_args()

    # Create and run the generator
    generator = AudioShapeGenerator(
        blocksize=args.blocksize,
        samplerate=args.samplerate
    )

    generator.bpm = args.bpm
    generator.current_shape = args.shape % len(generator.shape_modes)
    generator.trail_mode = args.trail
    generator.color_mode = args.color_mode

    generator.run()
