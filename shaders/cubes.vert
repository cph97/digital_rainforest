#version 330

uniform mat4 projection;
uniform mat4 view;
uniform float time;
uniform float audio_rms;
uniform float audio_onset;
uniform float wind_strength;
uniform float cube_size;
uniform float delay_energy;

// Per-vertex attributes (cube geometry)
in vec3 in_vert;
in vec3 in_norm;

// Per-instance attributes (point cloud data)
in vec3 instance_position;
in vec3 instance_color;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_position;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void main() {
    float noise = hash(instance_position);
    float noise2 = hash(instance_position * 1.5);
    float phase = time * 0.3 + noise * 6.28318; // Slower time for gentle movement
    
    // Audio activity level - controls how much displacement is applied
    // When audio is silent/quiet, this approaches 0 and cubes return to original positions
    float audio_activity = clamp(audio_rms * 2.0 + audio_onset * 0.5, 0.0, 1.0);
    
    // Wind animation - only active when there's audio
    vec3 wind_offset = vec3(
        sin(phase) * wind_strength * 0.1,
        cos(phase * 1.3) * wind_strength * 0.08,
        sin(phase * 0.7) * wind_strength * 0.09
    ) * audio_activity;
    
    // Audio-reactive pulse - cubes expand/contract with audio
    float pulse = 1.0 + (audio_onset * 0.15 * noise + audio_rms * 0.1) * audio_activity;
    
    // Databending vertical displacement - only active with audio
    float base_strength = 0.5;
    float databend_strength = base_strength + audio_rms * 0.3;
    float time_variation = sin(time * 0.3 + noise * 6.28318);
    
    // Random vertical offset per cube (databending effect)
    float vertical_smear = (noise2 - 0.5) * databend_strength * time_variation * audio_activity;
    
    // Strong audio-reactive vertical displacement
    // Onsets cause sudden jumps, RMS causes continuous drift
    float audio_smear = (audio_onset * 0.8 * (noise - 0.5) + audio_rms * 0.4 * sin(time + noise * 6.28318)) * audio_activity;
    vertical_smear += audio_smear;
    
    // Add delayed audio effect for visual echo synchronization
    // Creates secondary displacement that follows the feedback delay
    float delay_smear = delay_energy * 0.6 * (noise2 - 0.5) * sin(time * 0.5 + noise * 3.14159);
    vertical_smear += delay_smear;
    
    // Delayed horizontal movement for more visible echo effect
    vec3 delay_offset = vec3(
        cos(time * 0.4 + noise * 6.28318) * delay_energy * 0.15,
        0.0,
        sin(time * 0.4 + noise2 * 6.28318) * delay_energy * 0.15
    );
    
    // Apply all offsets - interpolate between original position and displaced position
    vec3 world_pos = instance_position + wind_offset * pulse + delay_offset;
    world_pos.y += vertical_smear; // Vertical databending displacement
    
    // Blend between original position and displaced position based on audio
    world_pos = mix(instance_position, world_pos, audio_activity);
    
    // Scale and position the cube vertex
    vec3 vertex_pos = in_vert * cube_size + world_pos;
    
    gl_Position = projection * view * vec4(vertex_pos, 1.0);
    
    // Color variation based on displacement
    float displacement_brightness = 1.0 + abs(vertical_smear) * 0.5;
    float brightness = displacement_brightness * (1.0 + audio_rms * 0.4 + audio_onset * 0.6);
    v_color = instance_color * brightness;
    v_normal = in_norm;
    v_position = vertex_pos;
}
