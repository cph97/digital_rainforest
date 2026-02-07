#version 330

uniform mat4 mvp;
uniform float point_size;
uniform float time;
uniform float audio_rms;
uniform float audio_onset;
uniform float wind_strength;

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;
out float v_depth;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void main() {
    vec3 pos = in_position;
    
    float noise = hash(in_position);
    float phase = time * 0.5 + noise * 6.28318;
    
    vec3 wind_offset = vec3(
        sin(phase) * wind_strength * 0.02,
        cos(phase * 1.3) * wind_strength * 0.01,
        sin(phase * 0.7) * wind_strength * 0.015
    );
    
    float pulse = 1.0 + audio_onset * 0.3 * noise;
    pos += wind_offset * pulse;
    
    vec4 view_pos = mvp * vec4(pos, 1.0);
    gl_Position = view_pos;
    
    float size_mod = 1.0 + audio_rms * 0.5;
    gl_PointSize = point_size * size_mod;
    
    float brightness = 1.0 + audio_rms * 0.4 + audio_onset * 0.6;
    v_color = in_color * brightness;
    
    v_depth = view_pos.z;
}
