import moderngl_window as mglw
from moderngl_window import geometry
import moderngl
import numpy as np
import logging
import sys
from pathlib import Path
import time

from pointcloud_loader import PointCloudLoader
from renderer import PointCloudRenderer
from audio_engine import AudioEngine
from analysis import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    'point_size': 3.0,
    'max_points': 6000000,
    'temporal_feedback': 0.92,  # Very high feedback for long vertical trails
    'smear_strength': 0.2,   # Databending smear intensity
    'audio_gain': 1.5,  # Increased for stronger audio reactivity
    'grain_density_range': (25.0, 20.0),
    'glitch_intensity': 0.0001,  # Moderate glitch for tears 
    
    'pointcloud_folder': './pointcloud',
    'pointcloud_name': 'rainforest_0',
    'audio_folder': './audio',
    
    'sample_rate': 44100,
    'audio_block_size': 512,
    
    'window_width': 1920,
    'window_height': 1080,
    
    # Normalization and camera
    'normalize_pointcloud': True,
    'center_pointcloud': True,
    'auto_camera_fit': True,
    'fov': 60.0,
    'debug_overlay': True,
    'bypass_postprocess': False,  # Enable post-processing for databending effect
    
    # Rendering mode
    'render_mode': 'cubes',  # 'points' or 'cubes'
    'cube_size': 0.0008,

    'bird_count_min': 1,
    'bird_count_max': 10,
    'bird_spread': 0.2,

    'bird_spawn_layout': 'world',
    'bird_spawn_min_dist': 1.5,
    'bird_spawn_max_dist': 4.0,
    'bird_species_ring_height': 0.0,
    'bird_species_stack_step': 0.05,
    'bird_species_jitter': 0.005,
    'birds_frozen': False,
    'birds_follow_camera': False,

    'bird_nav_center_follow_camera': False,

    'bird_species_manual_space': 'camera',
    'bird_species_manual_offsets': [
        [ 0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0, 0.0],
        [ 0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0],
        [ 0.0,  0.0, 0.0],
        [0.0,  0.0, 0.0],
        [ 0.0,  0.0,  0.0],
    ],

    'camera_auto_orbit_enabled': False,
    'camera_orbit_speed': 0.12,

    'camera_random_rotation_enabled': True,
    'camera_random_rotation_interval': 3.5,
    'camera_random_yaw_range': 25.0,
    'camera_random_pitch_range': 10.0,
    'camera_random_rotation_smoothness': 1.8,

    'camera_initial_at_birds': True,
    'camera_initial_at_birds_height': 0.25,

    'birds_idle_anim_enabled': True,
    'birds_idle_anim_amp': 0.03,
    'birds_idle_anim_speed': 1.2,

    'bird_camera_enabled': False,
    'bird_camera_center_enabled': False,
    'audio_camera_enabled': False,
}


class RainforestVisualizer(mglw.WindowConfig):
    """Main application window for rainforest audiovisual system."""
    
    gl_version = (3, 3)
    title = "Rainforest Organism - GPU Audiovisual System"
    window_size = (CONFIG['window_width'], CONFIG['window_height'])
    aspect_ratio = None
    resizable = True
    vsync = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        logger.info("Initializing Rainforest Visualizer...")
        
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        
        logger.info("Loading point cloud...")
        loader = PointCloudLoader(
            CONFIG['pointcloud_folder'],
            CONFIG['pointcloud_name'],
            max_points=CONFIG['max_points'],
            normalize=CONFIG['normalize_pointcloud'],
            center=CONFIG['center_pointcloud']
        )
        positions, colors = loader.load()
        self.pointcloud_stats = loader.get_statistics()
        self.bounding_radius = loader.bounding_radius
        self.original_centroid = loader.original_centroid
        
        logger.info("Initializing audio engine...")
        self.audio_engine = AudioEngine(
            sample_rate=CONFIG['sample_rate'],
            block_size=CONFIG['audio_block_size'],
            config=CONFIG
        )
        
        self.audio_engine.load_audio_samples(CONFIG['audio_folder'])
        self.audio_engine.set_pointcloud_stats(self.pointcloud_stats)
        
        logger.info("Initializing audio analyzer...")
        self.audio_analyzer = AudioAnalyzer(
            sample_rate=CONFIG['sample_rate'],
            hop_length=CONFIG['audio_block_size']
        )
        
        logger.info("Initializing renderer...")
        self.renderer = PointCloudRenderer(
            self.ctx,
            self.wnd.width,
            self.wnd.height,
            positions,
            colors,
            CONFIG,
            bounding_radius=self.bounding_radius
        )
        
        self.bypass_postprocess = CONFIG.get('bypass_postprocess', True)
        
        self.audio_features = {
            'rms': 0.0,
            'spectral_centroid': 0.0,
            'onset': 0.0,
            'band_low': 0.0,
            'band_mid': 0.0,
            'band_high': 0.0
        }

        self._last_bird_playing = False
        
        self.keys_pressed = set()
        self.mouse_captured = False
        self.last_mouse_pos = None
        
        self.fps_counter = 0
        self.fps_timer = 0.0
        self.current_fps = 0.0
        
        self.show_hud = True
        
        logger.info("Starting audio engine...")
        try:
            self.audio_engine.start()
        except Exception as e:
            logger.error(f"Failed to start audio engine: {e}")
            logger.info("Running in visual-only mode")
        
        logger.info("Initialization complete!")
        logger.info("\n" + "="*60)
        logger.info("CONTROLS:")
        logger.info("  WASD/QE: Move camera (QE = up/down)")
        logger.info("  Mouse: Look around (click window to capture)")
        logger.info("  Scroll: Adjust movement speed")
        logger.info("  Space: Toggle pause/resume")
        logger.info("  G: Toggle glitch effect")
        logger.info("  1/2: Decrease/Increase glitch intensity")
        logger.info("  C: Toggle audio-reactive camera movement")
        logger.info("  T: Toggle bird tracking camera mode")
        logger.info("  N: Track next bird")
        logger.info("  O: Toggle orbit mode around tracked bird")
        logger.info("  R: Reset camera position")
        logger.info("  F: Fit camera to view entire point cloud")
        logger.info("  P: Screenshot")
        logger.info("  H: Toggle HUD")
        logger.info("  ESC: Exit")
        logger.info("="*60 + "\n")
    
    def render(self, time_val, frame_time):
        """Main render loop."""
        self.fps_counter += 1
        self.fps_timer += frame_time
        
        if self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / self.fps_timer
            self.fps_counter = 0
            self.fps_timer = 0.0
        
        self._process_keyboard(frame_time)

        # Always read bird_playing from the audio engine (even if analysis buffer isn't available this frame)
        bird_playing = bool(self.audio_engine.visual_feedback.get('bird_playing', False))
        self.audio_features['bird_playing'] = bird_playing

        audio_buffer = self.audio_engine.get_analysis_buffer()
        if audio_buffer is not None:
            self.audio_features = self.audio_analyzer.analyze_buffer(audio_buffer)

        # Preserve bird_playing even when audio_features is replaced by analyzer output
        self.audio_features['bird_playing'] = bird_playing

        # Add delay energy from audio engine for visual feedback synchronization
        self.audio_features['delay_energy'] = self.audio_engine.visual_feedback.get('delay_energy', 0.0)

        self.audio_features['bird_energy'] = float(self.audio_engine.visual_feedback.get('bird_energy', 0.0))
        
        # Pass glitch state to renderer for bird spawn control
        self.audio_features['glitch_active'] = bool(self.audio_engine.visual_feedback.get('glitch_active', False))
        
        # Pass stutter data for bird density mapping
        self.audio_features['stutter_active'] = bool(self.audio_engine.visual_feedback.get('stutter_active', False))
        self.audio_features['stutter_intensity'] = float(self.audio_engine.visual_feedback.get('stutter_intensity', 0.0))

        self._last_bird_playing = bird_playing
        self.audio_features['bird_spawn_events'] = int(self.audio_engine.visual_feedback.get('bird_spawn_events', 0))
        
        self._update_feedback_loop(frame_time)
        
        self.renderer.update(frame_time, self.audio_features)
        self.renderer.render(self.audio_features)
        
        if self.show_hud:
            self._render_hud()
    
    def _process_keyboard(self, delta_time):
        """Process continuous keyboard input."""
        if 'W' in self.keys_pressed:
            self.renderer.camera.process_keyboard('FORWARD', delta_time)
        if 'S' in self.keys_pressed:
            self.renderer.camera.process_keyboard('BACKWARD', delta_time)
        if 'A' in self.keys_pressed:
            self.renderer.camera.process_keyboard('LEFT', delta_time)
        if 'D' in self.keys_pressed:
            self.renderer.camera.process_keyboard('RIGHT', delta_time)
        if 'Q' in self.keys_pressed:
            self.renderer.camera.process_keyboard('DOWN', delta_time)
        if 'E' in self.keys_pressed:
            self.renderer.camera.process_keyboard('UP', delta_time)
    
    def _update_feedback_loop(self, delta_time):
        """Update audio-visual feedback loop."""
        camera_pos = self.renderer.camera.position
        camera_speed = self.renderer.camera.speed
        glitch_intensity = CONFIG.get('glitch_intensity', 1.0)
        
        point_density = 0.5
        if self.pointcloud_stats:
            bounds = self.pointcloud_stats['bounds']
            if bounds:
                size = bounds['size']
                volume = np.prod(size) if len(size) == 3 else 1.0
                if volume > 0:
                    point_density = min(1.0, self.pointcloud_stats['count'] / (volume * 100))
        
        self.audio_engine.update_visual_feedback(
            camera_pos,
            camera_speed,
            glitch_intensity,
            point_density
        )
    
    def _render_hud(self):
        """Render on-screen HUD with debug info."""
        pass
    
    def key_event(self, key, action, modifiers):
        """Handle keyboard events."""
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.ESCAPE:
                self.close()
                return
            
            key_char = chr(key).upper() if 32 <= key <= 126 else key
            self.keys_pressed.add(key_char)
            logger.debug(f"Key pressed: {key_char}, keys_pressed: {self.keys_pressed}")
            
            if key == self.wnd.keys.SPACE:
                self.renderer.toggle_pause()
            
            elif key == self.wnd.keys.G:
                self.renderer.toggle_glitch()
            
            elif key == self.wnd.keys.NUMBER_1:
                self.renderer.adjust_glitch_intensity(-0.2)
            
            elif key == self.wnd.keys.NUMBER_2:
                self.renderer.adjust_glitch_intensity(0.2)
            
            elif key == self.wnd.keys.R:
                self.renderer.camera.reset(
                    position=self.renderer.initial_camera_pos,
                    yaw=self.renderer.initial_yaw,
                    pitch=self.renderer.initial_pitch
                )
                logger.info("Camera reset")
            
            elif key == self.wnd.keys.F:
                self.renderer.fit_camera()
                logger.info("Camera fitted to view")
            
            elif key == self.wnd.keys.B:
                self.bypass_postprocess = not self.bypass_postprocess
                logger.info(f"Post-processing bypass: {self.bypass_postprocess}")
            
            elif key == self.wnd.keys.P:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                self.renderer.screenshot(filename)
            
            elif key == self.wnd.keys.H:
                self.show_hud = not self.show_hud
            
            elif key == self.wnd.keys.C:
                self.renderer.toggle_audio_camera()
            
            elif key == self.wnd.keys.T:
                self.renderer.toggle_bird_tracking()
            
            elif key == self.wnd.keys.N:
                self.renderer.next_tracked_bird()
            
            elif key == self.wnd.keys.O:
                self.renderer.toggle_camera_orbit()
        
        elif action == self.wnd.keys.ACTION_RELEASE:
            key_char = chr(key).upper() if 32 <= key <= 126 else key
            if key_char in self.keys_pressed:
                self.keys_pressed.remove(key_char)
                logger.debug(f"Key released: {key_char}")
    
    def mouse_position_event(self, x, y, dx, dy):
        """Handle mouse movement."""
        if self.mouse_captured:
            self.renderer.camera.process_mouse_movement(dx, -dy)
    
    def mouse_press_event(self, x, y, button):
        """Handle mouse button press."""
        if button == 1:
            self.mouse_captured = True
            self.wnd.mouse_exclusivity = True
            self.wnd.cursor = False
    
    def mouse_release_event(self, x, y, button):
        """Handle mouse button release."""
        if button == 1:
            self.mouse_captured = False
            self.wnd.mouse_exclusivity = False
            self.wnd.cursor = True
    
    def mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll."""
        self.renderer.camera.process_scroll(y_offset)
    
    def resize(self, width, height):
        """Handle window resize."""
        self.renderer.resize(width, height)
    
    def close(self):
        """Clean up resources."""
        logger.info("Shutting down...")

        try:
            try:
                self.audio_engine.stop()
            except Exception as e:
                logger.error(f"Audio engine stop failed: {e}")

            try:
                self.renderer.release()
            except Exception as e:
                logger.error(f"Renderer release failed: {e}")
        finally:
            super().close()
            logger.info("Shutdown complete")


def main():
    """Entry point."""
    logger.info("="*60)
    logger.info("Rainforest Organism - GPU Audiovisual System")
    logger.info("="*60)
    
    pointcloud_path = Path(CONFIG['pointcloud_folder'])
    if not pointcloud_path.exists():
        logger.error(f"Point cloud folder not found: {pointcloud_path}")
        logger.info("Creating folder structure...")
        pointcloud_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Please place your point cloud file in: {pointcloud_path}")
        logger.warning(f"Expected filename: {CONFIG['pointcloud_name']}.ply/xyz/npy")
        return
    
    audio_path = Path(CONFIG['audio_folder'])
    if not audio_path.exists():
        logger.warning(f"Audio folder not found: {audio_path}")
        logger.info("Creating audio folder...")
        audio_path.mkdir(parents=True, exist_ok=True)
        logger.info("Running without audio samples (generative synthesis only)")
    
    try:
        mglw.run_window_config(RainforestVisualizer)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
